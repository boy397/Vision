"""Sarvam STT Streaming — WebSocket-based real-time speech-to-text.

Uses Sarvam's saaras:v3 streaming endpoint for ultra-low latency transcription.
The backend acts as a proxy: receives PCM/WAV audio from the frontend via its own
WebSocket, and relays it to Sarvam's WebSocket STT API in real time.

API docs: https://docs.sarvam.ai/api-reference-docs/speech-to-text/apis/streaming

Key facts:
  - Model: saaras:v3 (highest accuracy, multi-mode)
  - Supported audio for streaming: WAV or raw PCM only (not MP3/AAC/M4A)
  - Mode: transcribe | translate | verbatim | translit | codemix
  - Response events: speech_start, speech_end, transcript
"""

from __future__ import annotations

import asyncio
import base64
import logging
import os
import subprocess
import time
from typing import AsyncIterator, Callable

logger = logging.getLogger(__name__)

_SARVAM_STT_WS_URL = "wss://api.sarvam.ai/speech-to-text-streaming"


def _is_wav(audio_bytes: bytes) -> bool:
    """Return True if audio_bytes starts with a RIFF/WAV header."""
    return len(audio_bytes) >= 4 and audio_bytes[:4] == b"RIFF"


def _convert_to_wav(audio_bytes: bytes, sample_rate: int = 16000) -> bytes:
    """Convert any audio format to 16-bit mono WAV using ffmpeg.

    Writes input to a temp file (M4A/MP4 needs random access for moov atom),
    streams WAV output from stdout. No leftover files — temp is deleted in finally.
    Raises RuntimeError on failure.
    """
    import tempfile

    tmp_path = None
    try:
        # Write audio to temp file so ffmpeg can seek for moov atom
        with tempfile.NamedTemporaryFile(suffix=".m4a", delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        result = subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", tmp_path,           # read from temp file (seekable)
                "-f", "wav",              # output format: WAV
                "-ar", str(sample_rate),  # resample to target rate
                "-ac", "1",               # mono
                "-acodec", "pcm_s16le",   # 16-bit little-endian PCM
                "pipe:1",                 # stream WAV to stdout
            ],
            capture_output=True,
            timeout=30,
        )
        if result.returncode != 0:
            err = result.stderr.decode("utf-8", errors="replace")[-500:]
            raise RuntimeError(f"ffmpeg conversion failed: {err}")
        return result.stdout
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass




class SarvamStreamingSTT:
    """Manages a live Sarvam STT WebSocket connection.

    Usage:
        async with SarvamStreamingSTT(config) as stt:
            await stt.send_audio(pcm_bytes)
            async for event in stt.events():
                if event["type"] == "transcript":
                    text = event["text"]
    """

    def __init__(self, config: dict) -> None:
        self.config = config
        self._ws = None
        self._recv_task: asyncio.Task | None = None
        self._event_queue: asyncio.Queue = asyncio.Queue()
        self._closed = False

    async def __aenter__(self):
        await self._connect()
        return self

    async def __aexit__(self, *args):
        await self.close()

    async def _connect(self) -> None:
        """Open WebSocket connection to Sarvam STT API."""
        try:
            from sarvamai import AsyncSarvamAI
        except ImportError:
            raise RuntimeError(
                "sarvamai package not installed. Run: pip install sarvamai"
            )

        api_key = os.getenv("SARVAM_API_KEY", "")
        if not api_key:
            raise ValueError("SARVAM_API_KEY env var not set")

        model = self.config.get("streaming_model", "saaras:v3")
        mode = self.config.get("streaming_mode", "transcribe")
        language = self.config.get("language", "en-IN")

        logger.info(
            f"[SarvamStreamingSTT] Connecting: model={model}, mode={mode}, lang={language}"
        )

        self._client = AsyncSarvamAI(api_subscription_key=api_key)
        self._context_manager = self._client.speech_to_text_streaming.connect(
            model=model,
            mode=mode,
            language_code=language,
            high_vad_sensitivity=True,
            vad_signals=True,
            flush_signal=True,
        )
        self._ws = await self._context_manager.__aenter__()
        self._closed = False
        logger.info("[SarvamStreamingSTT] ✅ WebSocket connected")

        # Start background receiver
        self._recv_task = asyncio.create_task(self._receive_loop())

    async def _receive_loop(self) -> None:
        """Background task: read messages from Sarvam and put into queue."""
        try:
            async for message in self._ws:
                if self._closed:
                    break
                await self._event_queue.put(message)
                logger.debug(f"[SarvamStreamingSTT] Event: {message}")
        except Exception as e:
            if not self._closed:
                logger.error(f"[SarvamStreamingSTT] Receive error: {e}")
            await self._event_queue.put({"type": "error", "error": str(e)})
        finally:
            await self._event_queue.put({"type": "_done"})

    async def send_audio_wav(self, audio_bytes: bytes, sample_rate: int = 16000) -> None:
        """Send a WAV audio chunk to Sarvam for streaming transcription.

        Args:
            audio_bytes: Raw WAV bytes (must be WAV format for streaming API).
            sample_rate: Sample rate of the audio (default 16000).
        """
        if self._ws is None or self._closed:
            return

        audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
        try:
            await self._ws.transcribe(
                audio=audio_b64,
                encoding="audio/wav",
                sample_rate=sample_rate,
            )
        except Exception as e:
            logger.error(f"[SarvamStreamingSTT] Send error: {e}")

    async def flush(self) -> None:
        """Force processing of buffered audio — use when mic stops."""
        if self._ws is None or self._closed:
            return
        try:
            await self._ws.flush()
            logger.debug("[SarvamStreamingSTT] Flushed")
        except Exception as e:
            logger.error(f"[SarvamStreamingSTT] Flush error: {e}")

    async def events(self) -> AsyncIterator[dict]:
        """Yield events from Sarvam: speech_start, speech_end, transcript."""
        while True:
            event = await self._event_queue.get()
            if event.get("type") == "_done":
                break
            yield event

    async def next_transcript(self, timeout: float = 10.0) -> str | None:
        """Wait for the next transcript event and return the text.

        Returns None on timeout or error.
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            remaining = deadline - time.monotonic()
            try:
                event = await asyncio.wait_for(self._event_queue.get(), timeout=remaining)
            except asyncio.TimeoutError:
                return None

            t = event.get("type", "")
            if t == "_done" or t == "error":
                return None
            if t == "transcript":
                return event.get("text", "")
        return None

    async def close(self) -> None:
        """Close the WebSocket connection."""
        if self._closed:
            return
        self._closed = True
        if self._recv_task:
            self._recv_task.cancel()
        if self._ws is not None:
            try:
                await self._context_manager.__aexit__(None, None, None)
            except Exception:
                pass
        logger.info("[SarvamStreamingSTT] Connection closed")


class SarvamStreamingSTTSession:
    """One-shot streaming STT: opens connection, sends all audio, returns transcript.

    Useful as a drop-in replacement for the REST STT — but uses the WS API
    for much faster response (no upload round-trip, streaming decode).
    """

    def __init__(self, config: dict) -> None:
        self.config = config

    async def transcribe_wav(self, audio_bytes: bytes, sample_rate: int = 16000) -> str:
        """Transcribe a complete audio clip via Sarvam streaming WS.

        Accepts any audio format — automatically converts non-WAV (e.g. M4A from
        Android) to 16-bit mono WAV using ffmpeg before sending to Sarvam.
        Returns the final transcript text.
        """
        try:
            from sarvamai import AsyncSarvamAI
        except ImportError:
            raise RuntimeError("sarvamai package not installed. Run: pip install sarvamai")

        api_key = os.getenv("SARVAM_API_KEY", "")
        if not api_key:
            raise ValueError("SARVAM_API_KEY env var not set")

        model = self.config.get("streaming_model", "saaras:v3")
        mode = self.config.get("streaming_mode", "transcribe")
        language = self.config.get("language", "en-IN")

        start = time.perf_counter()

        # ── Format detection & conversion ─────────────────────────────────────
        # Sarvam streaming STT only accepts WAV or raw PCM.
        # Android records M4A (MPEG4/AAC) — detect by magic bytes and convert.
        if not _is_wav(audio_bytes):
            fmt = audio_bytes[4:12] if len(audio_bytes) >= 12 else b""
            logger.info(
                f"[SarvamStreamingSTT] Non-WAV audio detected "
                f"(magic={audio_bytes[:4].hex()}, ftyp={fmt!r}) — "
                f"converting {len(audio_bytes)} bytes via ffmpeg..."
            )
            loop = asyncio.get_event_loop()
            try:
                audio_bytes = await loop.run_in_executor(
                    None, _convert_to_wav, audio_bytes, sample_rate
                )
                logger.info(
                    f"[SarvamStreamingSTT] Converted to WAV: {len(audio_bytes)} bytes "
                    f"in {(time.perf_counter() - start) * 1000:.0f}ms"
                )
            except Exception as conv_err:
                logger.error(f"[SarvamStreamingSTT] ffmpeg conversion failed: {conv_err}")
                return ""
        # ─────────────────────────────────────────────────────────────────────

        audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")

        client = AsyncSarvamAI(api_subscription_key=api_key)
        transcript = ""

        async with client.speech_to_text_streaming.connect(
            model=model,
            mode=mode,
            language_code=language,
            high_vad_sensitivity=True,
            flush_signal=True,
        ) as ws:
            await ws.transcribe(
                audio=audio_b64,
                encoding="audio/wav",
                sample_rate=sample_rate,
            )
            await ws.flush()

            async for message in ws:
                # SDK returns SpeechToTextStreamingResponse (Pydantic), not a dict
                # message.type: "data" | "error" | "events"
                # message.data: SpeechToTextTranscriptionData | ErrorData | EventsData
                msg_type = getattr(message, "type", None)

                if msg_type == "data":
                    data = getattr(message, "data", None)
                    transcript = getattr(data, "transcript", "") or ""
                    if transcript:
                        break  # Got the final transcript

                elif msg_type == "error":
                    data = getattr(message, "data", None)
                    err_msg = getattr(data, "message", str(data))
                    logger.error(f"[SarvamStreamingSTT] API error: {err_msg}")
                    break

                elif msg_type == "events":
                    data = getattr(message, "data", None)
                    event_type = getattr(data, "event_type", None)
                    logger.debug(f"[SarvamStreamingSTT] VAD event: {event_type}")
                    # Continue — events don't contain transcripts

        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.info(f"[SarvamStreamingSTT] Transcribed in {elapsed_ms:.0f}ms: \"{transcript}\"")
        return transcript

