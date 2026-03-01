"""FastAPI backend — REST + WebSocket endpoints for the Vision pipeline.

Includes comprehensive debug logging for bottleneck identification.
"""

from __future__ import annotations

import asyncio
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


class LLMProviderRequest(BaseModel):
    provider: str  # google | groq | azure | vllm


class LLMModelRequest(BaseModel):
    provider: str
    model: str


# ── REST Endpoints ──

@app.get("/health")
async def health():
    """Health check."""
    config = _get_config()
    pipeline = _get_pipeline()
    active_provider = config.get_active_llm_provider()
    return {
        "status": "ok",
        "mode": config.current_mode,
        "llm_provider": active_provider,
        "llm_model": getattr(getattr(config.llm, active_provider, None), "model", "?"),
        "tts_provider": config.tts.provider,
        "stt_provider": config.stt.provider,
        "detection_enabled": config.app.detection_enabled,
        "available_models": config.get_available_models(active_provider),
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


@app.get("/config/llm/models")
async def get_llm_models(provider: str | None = None):
    """Get available models for a provider (defaults to active provider).

    Returns the list from config.yml — add models there and they show up here.
    """
    config = _get_config()
    provider = provider or config.get_active_llm_provider()
    return {
        "provider": provider,
        "active_model": getattr(getattr(config.llm, provider, None), "model", "?"),
        "models": config.get_available_models(provider),
    }


@app.post("/config/llm")
async def switch_llm_provider(req: LLMProviderRequest):
    """Switch vision LLM provider at runtime (google | groq | vllm).

    google → Gemini Flash (fast, accurate, great for Indian text).
    groq   → Llama Vision / Kimi K2 (near-zero latency on Groq hardware).
    vllm   → Local high-performance LLM deployment.
    """
    config = _get_config()
    pipeline = _get_pipeline()
    try:
        config.toggle_llm(req.provider)
        # Re-create LLM instance with new provider
        from backend.src.providers.factory import create_provider
        pipeline.llm = create_provider("llm", config)
        model = getattr(getattr(config.llm, req.provider, None), "model", "?")
        logger.info(f"[/config/llm] Switched LLM to: {req.provider} ({model})")
        return {
            "status": "ok",
            "llm_provider": req.provider,
            "llm_model": model,
            "available_models": config.get_available_models(req.provider),
            "message": f"Vision LLM switched to {req.provider} ({model})",
        }
    except ValueError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})


@app.post("/config/llm/model")
async def switch_llm_model(req: LLMModelRequest):
    """Switch the active model within an LLM provider.

    Models are defined in config.yml under each provider's `available_models`.
    """
    config = _get_config()
    pipeline = _get_pipeline()
    try:
        config.switch_model(req.provider, req.model)
        # Also ensure this provider is active
        config.toggle_llm(req.provider)
        # Re-create LLM instance with new model
        from backend.src.providers.factory import create_provider
        pipeline.llm = create_provider("llm", config)
        logger.info(f"[/config/llm/model] Switched to: {req.provider} / {req.model}")
        return {
            "status": "ok",
            "llm_provider": req.provider,
            "llm_model": req.model,
            "message": f"Model switched to {req.model}",
        }
    except ValueError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})


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


@app.post("/detect")
async def detect_only(image: UploadFile = File(...)):
    """Lightweight YOLO detection + tracking only (no LLM).

    Returns bounding boxes with persistent track IDs.
    Designed for continuous real-time tracking at high FPS (~30-50ms per frame).
    """
    pipeline = _get_pipeline()
    start = time.perf_counter()

    contents = await image.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if frame is None:
        return JSONResponse(status_code=400, content={"error": "Could not decode image"})

    det_config = pipeline.config.get_active_detection()
    detections = pipeline.detector.detect(
        frame,
        confidence=det_config.confidence,
        filter_classes=None,  # No class filter — detect everything
    )

    elapsed_ms = (time.perf_counter() - start) * 1000
    det_dicts = [d.to_dict() for d in detections]

    logger.debug(
        f"[/detect] {len(det_dicts)} objects in {elapsed_ms:.0f}ms"
    )

    return {
        "detections": det_dicts,
        "elapsed_ms": round(elapsed_ms, 1),
        "frame_shape": list(frame.shape[:2]),  # [height, width]
    }


@app.post("/scan")
async def scan_image(
    image: UploadFile | None = File(None),
    images: list[UploadFile] = File(default=[]),
):
    """Upload one or more images for scan + analysis.

    Supports both single image (legacy) and multi-image (new dual-frame capture).
    Returns detection results, LLM analysis, and TTS text.
    """
    pipeline = _get_pipeline()
    start = time.perf_counter()

    # Collect all uploaded files
    upload_files: list[UploadFile] = []
    if images:
        upload_files = images
    elif image:
        upload_files = [image]
    else:
        return JSONResponse(status_code=400, content={"error": "No image provided"})

    frames = []
    for i, img_file in enumerate(upload_files):
        contents = await img_file.read()
        logger.debug(f"[/scan] Image {i+1}: {len(contents)} bytes")
        nparr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            logger.warning(f"[/scan] Failed to decode image {i+1}")
            continue
        frames.append(frame)

    if not frames:
        return JSONResponse(status_code=400, content={"error": "Could not decode any image"})

    logger.debug(f"[/scan] Decoded {len(frames)} frame(s): shapes={[f.shape for f in frames]}")
    result = await pipeline.process_frame(frames[0], force=True, extra_frames=frames[1:])

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


@app.post("/voice/listen")
async def voice_listen(audio: UploadFile = File(...)):
    """Lightweight always-on voice command listener.

    Accepts short audio clips (~2s) from the continuous mic cycle.
    Runs STT → Intent classification. Returns:
      - has_command: whether a meaningful command was detected
      - action, text, etc. if a command was found

    Designed to be called rapidly (every ~2s) without blocking the client.
    Empty/silent clips return immediately with has_command=false.
    """
    pipeline = _get_pipeline()
    start = time.perf_counter()
    contents = await audio.read()

    # Quick size check — very short clips are likely silence
    if len(contents) < 500:
        return {"has_command": False, "reason": "too_short"}

    # STT
    try:
        text = await pipeline.stt.transcribe(contents)
    except Exception as e:
        logger.warning(f"[/voice/listen] STT error: {e}")
        return {"has_command": False, "reason": "stt_error"}

    elapsed_stt = (time.perf_counter() - start) * 1000

    if not text or not text.strip():
        logger.debug(f"[/voice/listen] No speech detected ({elapsed_stt:.0f}ms)")
        return {"has_command": False, "reason": "no_speech"}

    # Intent classification
    result = pipeline.intent_classifier.classify(text)
    elapsed_ms = (time.perf_counter() - start) * 1000

    # Filter out low-confidence / unknown intents
    from backend.src.voice.intent import Intent
    actionable_intents = {
        Intent.SCAN, Intent.START_CONTINUOUS, Intent.STOP_CONTINUOUS,
        Intent.SWITCH_MODE, Intent.REPEAT,
    }

    if result.intent in actionable_intents and result.confidence >= 0.5:
        logger.info(
            f"[/voice/listen] ✅ COMMAND: '{text}' → {result.intent.value} "
            f"(conf={result.confidence:.2f}, {elapsed_ms:.0f}ms)"
        )
        # Dispatch to get the full result (same as /voice)
        dispatch_result = await pipeline._dispatch_intent(result)
        return {
            "has_command": True,
            "action": dispatch_result.get("action"),
            "text": text,
            "intent": result.intent.value,
            "confidence": result.confidence,
            **dispatch_result,
        }

    # No actionable command — could be ambient speech
    logger.debug(
        f"[/voice/listen] No command: '{text}' → {result.intent.value} "
        f"(conf={result.confidence:.2f}, {elapsed_ms:.0f}ms)"
    )
    return {
        "has_command": False,
        "text": text,
        "intent": result.intent.value,
        "confidence": result.confidence,
        "reason": "no_actionable_command",
    }


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


# ── WebSocket (Sarvam Streaming STT + TTS — Ultra Low Latency) ──

@app.websocket("/voice/sarvam")
async def voice_sarvam_ws(websocket: WebSocket):
    """Ultra-low-latency voice pipeline using Sarvam's real WebSocket APIs.

    Frontend sends:
      - binary: WAV audio chunks (16-bit PCM, 16kHz, mono)
        Each binary message should be a complete or partial WAV clip.
        On mic release/VAD trigger, the frontend sends a text message {"action":"flush"}
      - text JSON: {"action":"flush"} to force STT to process buffered audio
      - text JSON: {"action":"ping"} for keepalive
      - text JSON: {"action":"scan_result","frame_b64":"..."} when user says scan —
        frontend takes a photo and sends it here for vision analysis

    Server sends:
      - text JSON: {"type":"transcript","text":"..."} incremental/final transcript
      - text JSON: {"type":"intent","action":"...","text":"..."} voice command result
      - binary: raw MP3 audio chunks (TTS response) — play immediately as they arrive
      - text JSON: {"type":"tts_done"} after all TTS audio sent
      - text JSON: {"type":"scan_result","detections":[...],"analysis":{...},"tts_text":"..."}
      - text JSON: {"type":"pong"} in reply to ping

    Architecture:
      Phone WAV → Sarvam STT WS (saaras:v3) → Intent classify
        ├─ if scan → take photo (frontend) → vision LLM → Sarvam TTS WS → MP3 chunks
        ├─ if followup/repeat → LLM chat → Sarvam TTS WS → MP3 chunks
        └─ else → Sarvam TTS WS for ack → MP3 chunks
    """
    await websocket.accept()
    pipeline = _get_pipeline()
    cfg = _get_config()
    logger.info("[WS /voice/sarvam] 🟢 Client connected")

    session_start = time.time()
    total_utterances = 0

    # Lazy import streaming providers
    from backend.src.providers.stt.sarvam_stt_streaming import SarvamStreamingSTT
    from backend.src.providers.tts.sarvam_tts_streaming import SarvamStreamingTTS

    stt_config = cfg.stt.sarvam.model_dump() if hasattr(cfg.stt, "sarvam") else {}
    tts_config = cfg.tts.sarvam.model_dump() if hasattr(cfg.tts, "sarvam") else {}

    streaming_tts = SarvamStreamingTTS(tts_config)

    # Audio accumulation buffer — collect chunks between flush signals
    audio_buffer: list[bytes] = []
    # Queue for photo frames sent by the frontend for scan processing
    scan_queue: asyncio.Queue = asyncio.Queue()

    async def stream_tts_response(text: str) -> None:
        """Stream TTS audio chunks back to the client via the WebSocket."""
        if not text:
            return
        tts_start = time.perf_counter()
        tts_chunks = 0
        tts_bytes = 0
        try:
            async for chunk in streaming_tts.synthesize(text):
                await websocket.send_bytes(chunk)
                tts_chunks += 1
                tts_bytes += len(chunk)
        except Exception as e:
            logger.error(f"[WS /voice/sarvam] TTS streaming error: {e}")
        finally:
            await websocket.send_json({"type": "tts_done"})
            tts_ms = (time.perf_counter() - tts_start) * 1000
            logger.info(
                f"[WS /voice/sarvam] TTS: {tts_chunks} chunks, "
                f"{tts_bytes} bytes, {tts_ms:.0f}ms"
            )

    async def process_utterance(audio_bytes: bytes) -> None:
        """Transcribe a complete utterance via Sarvam streaming STT and handle intent."""
        nonlocal total_utterances
        total_utterances += 1
        utt_num = total_utterances

        logger.info(
            f"[WS /voice/sarvam] 🎙️ Utterance #{utt_num}: "
            f"{len(audio_bytes)} bytes → STT (WS)"
        )

        # Use the one-shot streaming STT session (WS, but request-response style)
        from backend.src.providers.stt.sarvam_stt_streaming import SarvamStreamingSTTSession
        stt_session = SarvamStreamingSTTSession(stt_config)

        try:
            transcript = await stt_session.transcribe_wav(audio_bytes, sample_rate=16000)
        except Exception as e:
            logger.error(f"[WS /voice/sarvam] STT error: {e}")
            return

        if not transcript or not transcript.strip():
            logger.debug(f"[WS /voice/sarvam] Utterance #{utt_num}: no speech")
            return

        logger.info(f"[WS /voice/sarvam] Utterance #{utt_num}: \"{transcript}\"")
        await websocket.send_json({"type": "transcript", "text": transcript})

        # Intent classification
        result = pipeline.intent_classifier.classify(transcript)
        from backend.src.voice.intent import Intent
        logger.info(
            f"[WS /voice/sarvam] Intent: {result.intent.value} "
            f"(conf={result.confidence:.2f})"
        )

        if result.intent == Intent.SCAN:
            # Ask frontend to take a photo and send it back
            await websocket.send_json({
                "type": "intent",
                "action": "scan",
                "text": transcript,
                "message": "Scanning now.",
            })
            # Immediately speak the ack while we wait for the photo
            asyncio.create_task(stream_tts_response("Scanning now."))

            # Wait for the photo from the frontend (up to 10s)
            try:
                frame_msg = await asyncio.wait_for(scan_queue.get(), timeout=10.0)
                import base64 as b64
                import cv2
                import numpy as np
                frame_bytes = b64.b64decode(frame_msg)
                nparr = np.frombuffer(frame_bytes, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if frame is not None:
                    scan_start = time.perf_counter()
                    scan_result = await pipeline.process_frame(frame, force=True)
                    scan_ms = (time.perf_counter() - scan_start) * 1000
                    logger.info(
                        f"[WS /voice/sarvam] Scan complete in {scan_ms:.0f}ms, "
                        f"tts_len={len(scan_result.tts_text)}"
                    )
                    await websocket.send_json({
                        "type": "scan_result",
                        **scan_result.to_dict(),
                    })
                    if scan_result.tts_text:
                        await stream_tts_response(scan_result.tts_text)
                else:
                    logger.warning("[WS /voice/sarvam] Could not decode scan frame")
            except asyncio.TimeoutError:
                logger.warning("[WS /voice/sarvam] Scan: no frame received within 10s")

        elif result.intent == Intent.START_CONTINUOUS:
            dispatch = await pipeline._dispatch_intent(result)
            await websocket.send_json({"type": "intent", **dispatch})
            await stream_tts_response(dispatch.get("message", ""))

        elif result.intent == Intent.STOP_CONTINUOUS:
            dispatch = await pipeline._dispatch_intent(result)
            await websocket.send_json({"type": "intent", **dispatch})
            await stream_tts_response(dispatch.get("message", ""))

        elif result.intent == Intent.SWITCH_MODE:
            dispatch = await pipeline._dispatch_intent(result)
            await websocket.send_json({"type": "intent", **dispatch})
            await stream_tts_response(dispatch.get("message", ""))

        elif result.intent == Intent.REPEAT:
            dispatch = await pipeline._dispatch_intent(result)
            await websocket.send_json({"type": "intent", **dispatch})
            if dispatch.get("tts_text"):
                await stream_tts_response(dispatch["tts_text"])

        elif result.intent == Intent.FOLLOW_UP:
            dispatch = await pipeline._dispatch_intent(result)
            await websocket.send_json({"type": "intent", **dispatch})
            if dispatch.get("tts_text"):
                await stream_tts_response(dispatch["tts_text"])

        else:
            # Unrecognized — if we have context, treat as follow-up
            if pipeline._last_analysis:
                dispatch = await pipeline._handle_follow_up(transcript)
                await websocket.send_json({"type": "intent", **dispatch})
                if dispatch.get("tts_text"):
                    await stream_tts_response(dispatch["tts_text"])
            else:
                await websocket.send_json({
                    "type": "intent",
                    "action": "unknown",
                    "text": transcript,
                    "message": "Command not recognized",
                })

    try:
        import json as _json
        while True:
            data = await websocket.receive()

            # ── Client disconnected ───────────────────────────────────────────
            if data.get("type") == "websocket.disconnect":
                break

            # ── Binary = WAV audio chunk from mic ────────────────────────────
            if "bytes" in data and data["bytes"]:
                audio_buffer.append(data["bytes"])

            # ── Text = control commands ───────────────────────────────────────
            elif "text" in data and data["text"]:
                try:
                    cmd = _json.loads(data["text"])
                except _json.JSONDecodeError:
                    continue

                action = cmd.get("action", "")

                if action == "flush":
                    # User stopped speaking — process accumulated audio
                    if audio_buffer:
                        combined = b"".join(audio_buffer)
                        audio_buffer.clear()
                        # Fire-and-forget so we can keep receiving
                        asyncio.create_task(process_utterance(combined))

                elif action == "scan_frame":
                    # Frontend took a photo in response to our scan intent
                    frame_b64 = cmd.get("frame_b64", "")
                    if frame_b64:
                        await scan_queue.put(frame_b64)
                        logger.info("[WS /voice/sarvam] 📸 Scan frame received from frontend")

                elif action == "ping":
                    await websocket.send_json({"type": "pong", "timestamp": time.time()})

                elif action == "reset":
                    audio_buffer.clear()
                    pipeline.state_tracker.reset()
                    await websocket.send_json({"type": "reset_ok"})

    except WebSocketDisconnect:
        pass  # Normal disconnect
    except RuntimeError as e:
        # Starlette raises RuntimeError if receive/send called after disconnect
        if "disconnect" not in str(e).lower():
            logger.error(f"[WS /voice/sarvam] RuntimeError: {e}", exc_info=True)
    except Exception as e:
        logger.error(f"[WS /voice/sarvam] ❌ Unhandled error: {e}", exc_info=True)
        try:
            await websocket.close(code=1011)
        except Exception:
            pass

    elapsed = time.time() - session_start
    logger.info(
        f"[WS /voice/sarvam] 🔴 Session ended after {elapsed:.1f}s, "
        f"utterances={total_utterances}"
    )




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
