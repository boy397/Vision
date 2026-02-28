"""FastAPI backend — REST + WebSocket endpoints for the Vision pipeline.

Includes comprehensive debug logging for bottleneck identification.
"""

from __future__ import annotations

import logging
import time
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
    # Load .env file into environment
    from dotenv import load_dotenv
    load_dotenv()

    setup_logging()
    logger.info("Starting Vision Assistive System...")
    _config = load_config()
    _pipeline = VisionPipeline(_config)
    logger.info(f"Mode: {_config.current_mode} | LLM: {_config.get_active_llm_provider()} | Device: {_config.detection.device}")
    logger.info(f"TTS: {_config.tts.provider} | STT: {_config.stt.provider}")
    logger.info(f"VAD: aggressiveness={_config.voice.vad_aggressiveness}, silence_timeout={_config.voice.silence_timeout_ms}ms")
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


class TTSProviderRequest(BaseModel):
    provider: str


class DetectionToggleRequest(BaseModel):
    enabled: bool


# ── REST Endpoints ──

@app.get("/health")
async def health():
    """Health check."""
    config = _get_config()
    pipeline = _get_pipeline()
    return {
        "status": "ok",
        "mode": config.current_mode,
        "llm_provider": config.get_active_llm_provider(),
        "tts_provider": config.tts.provider,
        "stt_provider": config.stt.provider,
        "detection_enabled": config.app.detection_enabled,
        "detector_stats": pipeline.detector.stats,
    }


@app.get("/config")
async def get_config():
    """Return current configuration as JSON."""
    config = _get_config()
    return config.model_dump()


@app.get("/debug/stats")
async def debug_stats():
    """Return debug statistics for all pipeline components."""
    pipeline = _get_pipeline()
    stats = {
        "detector": pipeline.detector.stats,
        "pipeline": pipeline.stats,
    }
    # TTS stats if available
    if hasattr(pipeline.tts, "stats"):
        stats["tts"] = pipeline.tts.stats
    return stats


@app.post("/config/detection")
async def toggle_detection(req: DetectionToggleRequest):
    """Enable or disable YOLO detection at runtime.

    When disabled: pic → Gemini Vision directly (no YOLO gate).
    When enabled: pic → YOLO → (if object found) → Gemini Vision.
    """
    config = _get_config()
    config.toggle_detection(req.enabled)
    logger.info(f"[/config/detection] Detection {'ENABLED' if req.enabled else 'DISABLED'}")
    return {
        "status": "ok",
        "detection_enabled": req.enabled,
        "message": f"YOLO detection {'enabled' if req.enabled else 'disabled'} — "
                   f"frames now go {'YOLO → LLM' if req.enabled else 'directly to LLM'}",
    }


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
    start = time.perf_counter()

    contents = await image.read()
    logger.debug(f"[/scan] Received image: {len(contents)} bytes")

    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if frame is None:
        logger.warning("[/scan] Failed to decode image")
        return JSONResponse(status_code=400, content={"error": "Could not decode image"})

    logger.debug(f"[/scan] Decoded frame: {frame.shape}")
    result = await pipeline.process_frame(frame, force=True)

    elapsed_ms = (time.perf_counter() - start) * 1000
    logger.info(
        f"[/scan] Completed in {elapsed_ms:.0f}ms: "
        f"{len(result.detections)} detections, "
        f"state_changed={result.state_changed}, "
        f"tts_len={len(result.tts_text)}"
    )

    return result.to_dict()


@app.post("/scan/audio")
async def scan_with_audio(image: UploadFile = File(...)):
    """Scan image and return streaming audio response."""
    pipeline = _get_pipeline()
    start = time.perf_counter()

    contents = await image.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if frame is None:
        return JSONResponse(status_code=400, content={"error": "Could not decode image"})

    result = await pipeline.process_frame(frame, force=True)
    scan_ms = (time.perf_counter() - start) * 1000
    logger.info(f"[/scan/audio] Scan completed in {scan_ms:.0f}ms, TTS text: {len(result.tts_text)} chars")

    if result.tts_text:
        async def audio_stream():
            chunk_count = 0
            total_bytes = 0
            tts_start = time.perf_counter()
            async for chunk in pipeline.speak(result.tts_text):
                chunk_count += 1
                total_bytes += len(chunk)
                yield chunk
            tts_ms = (time.perf_counter() - tts_start) * 1000
            logger.info(
                f"[/scan/audio] TTS streaming done: "
                f"{chunk_count} chunks, {total_bytes} bytes, {tts_ms:.0f}ms"
            )

        return StreamingResponse(audio_stream(), media_type="audio/mpeg")

    return JSONResponse(content=result.to_dict())


@app.post("/voice")
async def process_voice(audio: UploadFile = File(...)):
    """Upload voice audio for command processing.

    STT → Intent Classification → Action Dispatch.
    """
    pipeline = _get_pipeline()
    start = time.perf_counter()
    contents = await audio.read()
    logger.info(f"[/voice] Received audio: {len(contents)} bytes")
    result = await pipeline.process_voice(contents)
    elapsed_ms = (time.perf_counter() - start) * 1000
    logger.info(f"[/voice] Completed in {elapsed_ms:.0f}ms: action={result.get('action')}")
    return result


@app.post("/chat")
async def chat(req: ChatRequest):
    """Follow-up chat with last detection context."""
    pipeline = _get_pipeline()
    start = time.perf_counter()
    result = await pipeline._handle_follow_up(req.question)
    elapsed_ms = (time.perf_counter() - start) * 1000
    logger.info(f"[/chat] Completed in {elapsed_ms:.0f}ms")
    return result


@app.post("/config/tts")
async def update_tts_provider(req: TTSProviderRequest):
    """Switch TTS provider at runtime (elevenlabs | sarvam)."""
    config = _get_config()
    pipeline = _get_pipeline()
    allowed = ["elevenlabs", "sarvam"]
    if req.provider not in allowed:
        return JSONResponse(status_code=400, content={"error": f"Unknown TTS provider: {req.provider}. Allowed: {allowed}"})
    config.tts.provider = req.provider
    # Re-create TTS provider
    from backend.src.providers.factory import create_provider
    pipeline.tts = create_provider("tts", config)
    logger.info(f"[/config/tts] Switched TTS provider to: {req.provider}")
    return {"status": "ok", "tts_provider": req.provider}


class TTSRequest(BaseModel):
    text: str


@app.post("/tts")
async def text_to_speech(req: TTSRequest):
    """Convert text to speech audio. Lightweight — no scan, just TTS."""
    pipeline = _get_pipeline()

    if not req.text:
        return JSONResponse(status_code=400, content={"error": "No text provided"})

    start = time.perf_counter()
    logger.info(f"[/tts] Request: text_len={len(req.text)}")

    async def audio_stream():
        chunk_count = 0
        total_bytes = 0
        async for chunk in pipeline.speak(req.text):
            chunk_count += 1
            total_bytes += len(chunk)
            yield chunk
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.info(
            f"[/tts] Streaming done: "
            f"{chunk_count} chunks, {total_bytes} bytes, {elapsed_ms:.0f}ms"
        )

    return StreamingResponse(audio_stream(), media_type="audio/mpeg")


# ── WebSocket (Voice Streaming with VAD) ──

@app.websocket("/voice/stream")
async def voice_stream_ws(websocket: WebSocket):
    """WebSocket for always-on voice with server-side VAD.

    Client sends:
      - binary: raw PCM audio chunks (16-bit, 16kHz, mono)

    Server sends:
      - text/JSON: intent classification results
      - binary: TTS audio response
    """
    await websocket.accept()
    pipeline = _get_pipeline()
    config = _get_config()
    logger.info("[WS /voice/stream] 🟢 Client connected")

    # Create VAD instance for this session
    from backend.src.voice.vad import VAD
    vad = VAD(
        aggressiveness=config.voice.vad_aggressiveness,
        silence_timeout_ms=config.voice.silence_timeout_ms,
    )

    # Session stats
    session_start = time.time()
    total_audio_bytes = 0
    total_messages = 0
    total_utterances = 0

    try:
        while True:
            data = await websocket.receive()

            if "bytes" in data and data["bytes"]:
                audio_chunk = data["bytes"]
                total_audio_bytes += len(audio_chunk)
                total_messages += 1

                # Log every 100 messages to avoid flooding
                if total_messages % 100 == 0:
                    elapsed = time.time() - session_start
                    logger.debug(
                        f"[WS /voice/stream] Session stats: "
                        f"messages={total_messages}, "
                        f"audio_bytes={total_audio_bytes}, "
                        f"elapsed={elapsed:.1f}s, "
                        f"utterances={total_utterances}, "
                        f"vad_triggered={vad.is_triggered}"
                    )

                # Feed to VAD — returns full utterance when speech ends
                utterance = vad.process_chunk(audio_chunk)

                if utterance is not None:
                    total_utterances += 1
                    # Speech ended — process the utterance
                    logger.info(
                        f"[WS /voice/stream] 🎙️ Utterance #{total_utterances}: "
                        f"{len(utterance)} bytes, processing..."
                    )

                    process_start = time.perf_counter()
                    result = await pipeline.process_voice(utterance)
                    process_ms = (time.perf_counter() - process_start) * 1000

                    logger.info(
                        f"[WS /voice/stream] Voice result: "
                        f"action={result.get('action')}, "
                        f"text='{result.get('text', '')[:50]}', "
                        f"process_time={process_ms:.0f}ms"
                    )

                    await websocket.send_json(result)

                    # If the result has TTS text, stream audio back
                    tts_text = result.get("tts_text") or result.get("message", "")
                    if tts_text and result.get("action") not in ("scan", "start_continuous", "stop_continuous"):
                        try:
                            tts_start = time.perf_counter()
                            tts_chunks = 0
                            tts_bytes = 0
                            async for chunk in pipeline.speak(tts_text):
                                await websocket.send_bytes(chunk)
                                tts_chunks += 1
                                tts_bytes += len(chunk)
                            # Send end-of-audio marker
                            await websocket.send_json({"type": "tts_done"})
                            tts_ms = (time.perf_counter() - tts_start) * 1000
                            logger.info(
                                f"[WS /voice/stream] TTS streamed: "
                                f"{tts_chunks} chunks, {tts_bytes} bytes, "
                                f"{tts_ms:.0f}ms"
                            )
                        except Exception as e:
                            logger.error(f"[WS /voice/stream] TTS streaming error: {e}")

            elif "text" in data and data["text"]:
                import json
                try:
                    cmd = json.loads(data["text"])
                    logger.debug(f"[WS /voice/stream] Text command: {cmd}")
                    if cmd.get("action") == "reset_vad":
                        vad.reset()
                        await websocket.send_json({"type": "vad_reset"})
                        logger.info("[WS /voice/stream] VAD reset by client")
                    elif cmd.get("action") == "ping":
                        await websocket.send_json({"type": "pong", "timestamp": time.time()})
                    elif cmd.get("action") == "get_stats":
                        await websocket.send_json({
                            "type": "stats",
                            "vad": vad.stats,
                            "session": {
                                "elapsed_s": round(time.time() - session_start, 1),
                                "total_audio_bytes": total_audio_bytes,
                                "total_messages": total_messages,
                                "total_utterances": total_utterances,
                            },
                        })
                except json.JSONDecodeError:
                    logger.warning("[WS /voice/stream] Invalid JSON received")

    except WebSocketDisconnect:
        elapsed = time.time() - session_start
        logger.info(
            f"[WS /voice/stream] 🔴 Client disconnected after {elapsed:.1f}s: "
            f"messages={total_messages}, audio={total_audio_bytes} bytes, "
            f"utterances={total_utterances}"
        )


# ── WebSocket (Continuous Scan Mode) ──

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
    logger.info("[WS /stream] 🟢 Client connected")

    # Session stats
    session_start = time.time()
    frame_count = 0
    state_change_count = 0
    skipped_frames = 0

    try:
        while True:
            data = await websocket.receive()

            if "bytes" in data and data["bytes"]:
                frame_start = time.perf_counter()
                frame_count += 1

                # Image frame
                nparr = np.frombuffer(data["bytes"], np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                if frame is not None:
                    result = await pipeline.process_frame(frame)

                    frame_ms = (time.perf_counter() - frame_start) * 1000

                    if result.state_changed and result.analysis:
                        state_change_count += 1
                        logger.info(
                            f"[WS /stream] Frame #{frame_count}: STATE CHANGED "
                            f"({len(result.detections)} dets, {frame_ms:.0f}ms)"
                        )
                        await websocket.send_json(result.to_dict())

                        # Stream TTS audio
                        if result.tts_text:
                            tts_start = time.perf_counter()
                            tts_chunks = 0
                            async for chunk in pipeline.speak(result.tts_text):
                                await websocket.send_bytes(chunk)
                                tts_chunks += 1
                            tts_ms = (time.perf_counter() - tts_start) * 1000
                            logger.debug(
                                f"[WS /stream] TTS: {tts_chunks} chunks in {tts_ms:.0f}ms"
                            )
                    else:
                        skipped_frames += 1

                    # Periodic summary
                    if frame_count % 50 == 0:
                        elapsed = time.time() - session_start
                        fps = frame_count / elapsed if elapsed > 0 else 0
                        logger.info(
                            f"[WS /stream] Summary: {frame_count} frames in {elapsed:.0f}s "
                            f"({fps:.1f} fps), {state_change_count} state changes, "
                            f"{skipped_frames} skipped"
                        )
                else:
                    logger.warning(
                        f"[WS /stream] Frame #{frame_count}: failed to decode "
                        f"({len(data['bytes'])} bytes)"
                    )

            elif "text" in data and data["text"]:
                import json
                try:
                    cmd = json.loads(data["text"])
                    action = cmd.get("action", "")
                    logger.debug(f"[WS /stream] Command: {cmd}")

                    if action == "switch_mode":
                        mode = cmd.get("mode", "medical")
                        pipeline.config.switch_mode(mode)
                        from backend.src.detection.detector import Detector
                        det_config = pipeline.config.get_active_detection()
                        pipeline.detector = Detector(det_config.model_dump())
                        pipeline.state_tracker.reset()
                        await websocket.send_json({"action": "mode_switched", "mode": mode})
                        logger.info(f"[WS /stream] Mode switched to: {mode}")

                    elif action == "voice":
                        # Voice audio sent as base64 in text message
                        import base64
                        audio_b64 = cmd.get("audio", "")
                        audio_bytes = base64.b64decode(audio_b64)
                        logger.info(f"[WS /stream] Voice command: {len(audio_bytes)} bytes")
                        result = await pipeline.process_voice(audio_bytes)
                        await websocket.send_json(result)

                    elif action == "ping":
                        await websocket.send_json({"type": "pong", "timestamp": time.time()})

                except json.JSONDecodeError:
                    await websocket.send_json({"error": "Invalid JSON"})

    except WebSocketDisconnect:
        elapsed = time.time() - session_start
        logger.info(
            f"[WS /stream] 🔴 Client disconnected after {elapsed:.1f}s: "
            f"frames={frame_count}, state_changes={state_change_count}, "
            f"skipped={skipped_frames}"
        )


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
