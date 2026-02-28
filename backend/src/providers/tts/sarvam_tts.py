"""Sarvam TTS provider — Hindi/regional language text-to-speech.

API docs: https://docs.sarvam.ai/api-reference-docs/text-to-speech/convert

Key fields (verified against docs 2026-02-28):
  - "text"   : str  (NOT "inputs": [...])
  - "target_language_code": BCP-47 string (e.g. "en-IN", "hi-IN")
  - "speaker": must match the chosen model version
      bulbul:v3 → Shubh (default), Aditya, Ritu, Priya, Neha, Rahul, ...
      bulbul:v2 → Anushka (default), Manisha, Vidya, Arya, Abhilash, Karun, Hitesh
  - "model"  : "bulbul:v3" (latest) | "bulbul:v2" (legacy)
  - "pace"   : 0.5–2.0 (v3) | 0.3–3.0 (v2)

Note: Sarvam REST API returns complete base64 audio (not true streaming).
For streaming, Sarvam offers a WebSocket endpoint (see docs).
"""

from __future__ import annotations

import base64
import logging
import os
import time
from typing import AsyncIterator

import httpx

from backend.src.providers.base import BaseTTS

logger = logging.getLogger(__name__)

_SARVAM_TTS_URL = "https://api.sarvam.ai/text-to-speech"

# Chunk size for pseudo-streaming from REST response (4KB)
_PSEUDO_STREAM_CHUNK_SIZE = 4096


class SarvamTTS(BaseTTS):
    """Sarvam AI text-to-speech for Indian languages.

    Uses REST API. Returns full audio then chunks it for streaming-like delivery.
    """

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self._total_synth_calls = 0
        self._total_synth_time_ms = 0.0
        self._total_bytes_streamed = 0

        model = config.get("model", "bulbul:v3")
        speaker = config.get("speaker", "shubh")
        language = config.get("language", "en-IN")

        logger.info(
            f"[SarvamTTS] Initialized: model={model}, "
            f"language={language}, speaker={speaker}"
        )

    async def synthesize(self, text: str) -> AsyncIterator[bytes]:
        """Convert text to speech via Sarvam REST API.

        API field: "text" (string) — NOT "inputs" (array).
        Returns base64 audio, chunked for pseudo-streaming.
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

            model = self.config.get("model", "bulbul:v3")
            speaker = self.config.get("speaker", "shubh")
            language = self.config.get("language", "en-IN")

            # Correct payload per Sarvam API docs (2026-02-28):
            # Field is "text" (str), NOT "inputs" (list)
            payload: dict = {
                "text": text,
                "target_language_code": language,
                "speaker": speaker,
                "model": model,
            }

            # pace is optional, add only if configured
            if "pace" in self.config:
                payload["pace"] = self.config["pace"]

            logger.debug(
                f"[SarvamTTS] Synth #{call_id} payload: "
                f"model={model}, lang={language}, speaker={speaker}, "
                f"text_len={len(text)}"
            )

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

                request_ms = (time.perf_counter() - start_time) * 1000

                # Log full error body before raise_for_status for debugging
                if response.status_code >= 400:
                    logger.error(
                        f"[SarvamTTS] Synth #{call_id} HTTP {response.status_code} "
                        f"after {request_ms:.0f}ms: {response.text[:500]}"
                    )

                response.raise_for_status()

                logger.info(
                    f"[SarvamTTS] Synth #{call_id} API response: "
                    f"status={response.status_code}, time={request_ms:.0f}ms"
                )

                data = response.json()
                audios = data.get("audios", [])

                if not audios:
                    logger.warning(
                        f"[SarvamTTS] Synth #{call_id}: "
                        f"No audio returned. Response: {data}"
                    )
                    return

                # Decode and stream in chunks (pseudo-streaming)
                total_bytes = 0
                chunk_count = 0
                first_chunk_time = None

                for audio_idx, audio_b64 in enumerate(audios):
                    audio_bytes = base64.b64decode(audio_b64)
                    audio_len = len(audio_bytes)

                    logger.debug(
                        f"[SarvamTTS] Synth #{call_id} audio[{audio_idx}]: "
                        f"{audio_len} bytes"
                    )

                    # Yield in chunks for streaming-like delivery
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
                f"{chunk_count} chunks, {total_bytes} bytes, {elapsed_ms:.0f}ms total"
            )

        except httpx.HTTPStatusError:
            # Already logged above with response body
            raise
        except Exception as e:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            logger.error(
                f"[SarvamTTS] Synth #{call_id} ERROR after {elapsed_ms:.0f}ms: {e}"
            )
            raise

    @property
    def stats(self) -> dict:
        return {
            "total_calls": self._total_synth_calls,
            "total_time_ms": round(self._total_synth_time_ms, 1),
            "total_bytes": self._total_bytes_streamed,
            "avg_time_ms": round(
                self._total_synth_time_ms / self._total_synth_calls, 1
            ) if self._total_synth_calls else 0,
            "mode": "REST (pseudo-streaming)",
        }
