"""Sarvam TTS provider — Hindi/regional language text-to-speech.

Note on streaming: Sarvam's REST API (text-to-speech) returns complete audio
as base64. For true streaming, Sarvam supports a WebSocket endpoint.
This implementation uses the REST endpoint but chunks the base64 response
for pseudo-streaming. For true low-latency streaming, consider the WebSocket API.

Sarvam streaming TTS WebSocket: wss://api.sarvam.ai/text-to-speech/streaming
Supported but not yet integrated here — REST is more reliable for now.
"""

from __future__ import annotations

import logging
import os
import time
from typing import AsyncIterator

import httpx

from backend.src.providers.base import BaseTTS

logger = logging.getLogger(__name__)

_SARVAM_TTS_URL = "https://api.sarvam.ai/text-to-speech"
# Sarvam WebSocket streaming endpoint (for future integration)
_SARVAM_TTS_WS_URL = "wss://api.sarvam.ai/text-to-speech/streaming"

# Chunk size for pseudo-streaming from REST response (4KB)
_PSEUDO_STREAM_CHUNK_SIZE = 4096


class SarvamTTS(BaseTTS):
    """Sarvam AI text-to-speech for Indian languages.

    Current mode: REST API with chunked output (pseudo-streaming).
    Sarvam DOES support WebSocket streaming — can be added for lower latency.
    """

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self._total_synth_calls = 0
        self._total_synth_time_ms = 0.0
        self._total_bytes_streamed = 0
        logger.info(
            f"[SarvamTTS] Initialized: "
            f"model={config.get('model', 'bulbul:v2')}, "
            f"language={config.get('language', 'hi-IN')}, "
            f"speaker={config.get('speaker', 'meera')}, "
            f"mode=REST (pseudo-streaming, chunk_size={_PSEUDO_STREAM_CHUNK_SIZE})"
        )

    async def synthesize(self, text: str) -> AsyncIterator[bytes]:
        """Convert text to speech via Sarvam API.

        Uses REST endpoint — returns full audio then chunks it for streaming.
        """
        self._total_synth_calls += 1
        call_id = self._total_synth_calls
        text_preview = text[:80] + "..." if len(text) > 80 else text

        logger.info(
            f"[SarvamTTS] Synth #{call_id} START: "
            f"text_len={len(text)}, preview=\"{text_preview}\""
        )

        start_time = time.perf_counter()

        try:
            api_key = os.getenv("SARVAM_API_KEY", "")
            if not api_key:
                raise ValueError("SARVAM_API_KEY env var not set")

            payload = {
                "inputs": [text],
                "target_language_code": self.config.get("language", "hi-IN"),
                "speaker": self.config.get("speaker", "meera"),
                "model": self.config.get("model", "bulbul:v2"),
            }

            logger.debug(
                f"[SarvamTTS] Synth #{call_id} request: "
                f"model={payload['model']}, lang={payload['target_language_code']}, "
                f"speaker={payload['speaker']}"
            )

            request_start = time.perf_counter()

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    _SARVAM_TTS_URL,
                    json=payload,
                    headers={
                        "api-subscription-key": api_key,
                        "Content-Type": "application/json",
                    },
                    timeout=30.0,
                )

                request_ms = (time.perf_counter() - request_start) * 1000
                logger.info(
                    f"[SarvamTTS] Synth #{call_id} API response: "
                    f"status={response.status_code}, time={request_ms:.0f}ms"
                )

                response.raise_for_status()

                data = response.json()
                audios = data.get("audios", [])

                if not audios:
                    logger.warning(
                        f"[SarvamTTS] Synth #{call_id}: No audio returned from API"
                    )
                    return

                # Decode and stream in chunks (pseudo-streaming)
                import base64
                total_bytes = 0
                chunk_count = 0
                first_chunk_time = None

                for audio_idx, audio_b64 in enumerate(audios):
                    audio_bytes = base64.b64decode(audio_b64)
                    audio_len = len(audio_bytes)

                    logger.debug(
                        f"[SarvamTTS] Synth #{call_id} audio[{audio_idx}]: "
                        f"{audio_len} bytes (b64_len={len(audio_b64)})"
                    )

                    # Chunk the audio for streaming-like delivery
                    offset = 0
                    while offset < audio_len:
                        chunk = audio_bytes[offset:offset + _PSEUDO_STREAM_CHUNK_SIZE]
                        offset += len(chunk)
                        chunk_count += 1
                        total_bytes += len(chunk)

                        if first_chunk_time is None:
                            first_chunk_time = time.perf_counter()
                            ttfb = (first_chunk_time - start_time) * 1000
                            logger.info(
                                f"[SarvamTTS] Synth #{call_id} TTFB: {ttfb:.0f}ms"
                            )

                        yield chunk

            elapsed_ms = (time.perf_counter() - start_time) * 1000
            self._total_synth_time_ms += elapsed_ms
            self._total_bytes_streamed += total_bytes

            logger.info(
                f"[SarvamTTS] Synth #{call_id} DONE: "
                f"{chunk_count} chunks, {total_bytes} bytes, "
                f"{elapsed_ms:.0f}ms total"
            )

        except Exception as e:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            logger.error(
                f"[SarvamTTS] Synth #{call_id} ERROR after {elapsed_ms:.0f}ms: {e}"
            )
            raise

    @property
    def stats(self) -> dict:
        """Return TTS statistics for debugging."""
        return {
            "total_calls": self._total_synth_calls,
            "total_time_ms": round(self._total_synth_time_ms, 1),
            "total_bytes": self._total_bytes_streamed,
            "avg_time_ms": round(
                self._total_synth_time_ms / self._total_synth_calls, 1
            )
            if self._total_synth_calls
            else 0,
            "mode": "REST (pseudo-streaming)",
        }
