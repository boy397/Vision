"""FastAPI backend — REST + WebSocket endpoints for the Vision pipeline."""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from backend.src.config import Config, load_config
from backend.src.pipeline import VisionPipeline
from backend.src.utils.logging import setup_logging

logger = logging.getLogger(__name__)

# ── Global state ──
_pipeline: VisionPipeline | None = None
_config: Config | None = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    """Startup / shutdown lifecycle."""
    global _pipeline, _config
    setup_logging()
    logger.info("Starting Vision Assistive System...")
    _config = load_config()
    _pipeline = VisionPipeline(_config)
    logger.info(f"Mode: {_config.current_mode} | LLM: {_config.get_active_llm_provider()} | Device: {_config.detection.device}")
    yield
    logger.info("Shutting down...")


app = FastAPI(
    title="Vision Assistive System",
    description="Assistive vision + voice API for visually impaired users",
    version="1.0.0",
    lifespan=lifespan,
)


def _get_pipeline() -> VisionPipeline:
    assert _pipeline is not None, "Pipeline not initialized"
    return _pipeline


def _get_config() -> Config:
    assert _config is not None, "Config not initialized"
    return _config


# ── CORS ──

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Updated at runtime from config
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Models ──

class ModeSwitchRequest(BaseModel):
    mode: str


class ChatRequest(BaseModel):
    question: str


# ── REST Endpoints ──

@app.get("/health")
async def health():
    """Health check."""
    config = _get_config()
    return {
        "status": "ok",
        "mode": config.current_mode,
        "llm_provider": config.get_active_llm_provider(),
        "tts_provider": config.tts.provider,
        "stt_provider": config.stt.provider,
    }


@app.get("/config")
async def get_config():
    """Return current configuration as JSON."""
    config = _get_config()
    return config.model_dump()


@app.post("/config/mode")
async def switch_mode(req: ModeSwitchRequest):
    """Switch between medical and retail modes via API (also doable via voice)."""
    pipeline = _get_pipeline()
    try:
        pipeline.config.switch_mode(req.mode)
        # Re-init detector for new mode
        from backend.src.detection.detector import Detector
        det_config = pipeline.config.get_active_detection()
        pipeline.detector = Detector(det_config.model_dump())
        pipeline.state_tracker.reset()

        return {"status": "ok", "mode": req.mode, "message": f"Switched to {req.mode} mode"}
    except ValueError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})


@app.post("/scan")
async def scan_image(image: UploadFile = File(...)):
    """Upload an image for one-shot scan + analysis.

    Returns detection results, LLM analysis, and TTS text.
    """
    pipeline = _get_pipeline()

    contents = await image.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if frame is None:
        return JSONResponse(status_code=400, content={"error": "Could not decode image"})

    result = await pipeline.process_frame(frame, force=True)
    return result.to_dict()


@app.post("/scan/audio")
async def scan_with_audio(image: UploadFile = File(...)):
    """Scan image and return streaming audio response."""
    pipeline = _get_pipeline()

    contents = await image.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if frame is None:
        return JSONResponse(status_code=400, content={"error": "Could not decode image"})

    result = await pipeline.process_frame(frame, force=True)

    if result.tts_text:
        async def audio_stream():
            async for chunk in pipeline.speak(result.tts_text):
                yield chunk

        return StreamingResponse(audio_stream(), media_type="audio/mpeg")

    return JSONResponse(content=result.to_dict())


@app.post("/voice")
async def process_voice(audio: UploadFile = File(...)):
    """Upload voice audio for command processing.

    STT → Intent Classification → Action Dispatch.
    """
    pipeline = _get_pipeline()
    contents = await audio.read()
    result = await pipeline.process_voice(contents)
    return result


@app.post("/chat")
async def chat(req: ChatRequest):
    """Follow-up chat with last detection context."""
    pipeline = _get_pipeline()
    result = await pipeline._handle_follow_up(req.question)
    return result


# ── WebSocket (Continuous Mode) ──

@app.websocket("/stream")
async def stream_ws(websocket: WebSocket):
    """WebSocket for continuous mode.

    Client sends:
      - binary: image frame (JPEG bytes)
      - text: JSON commands ({"action": "switch_mode", "mode": "retail"})

    Server sends:
      - text: JSON with detection results
      - binary: TTS audio chunks
    """
    await websocket.accept()
    pipeline = _get_pipeline()
    logger.info("WebSocket client connected")

    try:
        while True:
            data = await websocket.receive()

            if "bytes" in data and data["bytes"]:
                # Image frame
                nparr = np.frombuffer(data["bytes"], np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                if frame is not None:
                    result = await pipeline.process_frame(frame)
                    if result.state_changed and result.analysis:
                        await websocket.send_json(result.to_dict())

                        # Stream TTS audio
                        if result.tts_text:
                            async for chunk in pipeline.speak(result.tts_text):
                                await websocket.send_bytes(chunk)

            elif "text" in data and data["text"]:
                import json
                try:
                    cmd = json.loads(data["text"])
                    action = cmd.get("action", "")

                    if action == "switch_mode":
                        mode = cmd.get("mode", "medical")
                        pipeline.config.switch_mode(mode)
                        from backend.src.detection.detector import Detector
                        det_config = pipeline.config.get_active_detection()
                        pipeline.detector = Detector(det_config.model_dump())
                        pipeline.state_tracker.reset()
                        await websocket.send_json({"action": "mode_switched", "mode": mode})

                    elif action == "voice":
                        # Voice audio sent as base64 in text message
                        import base64
                        audio_b64 = cmd.get("audio", "")
                        audio_bytes = base64.b64decode(audio_b64)
                        result = await pipeline.process_voice(audio_bytes)
                        await websocket.send_json(result)

                except json.JSONDecodeError:
                    await websocket.send_json({"error": "Invalid JSON"})

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")


# ── Entry point ──

if __name__ == "__main__":
    import uvicorn

    config = load_config()
    uvicorn.run(
        "backend.src.main:app",
        host=config.server.host,
        port=config.server.port,
        reload=True,
    )
