"""Sarvam STT provider — Indian language speech-to-text.

API docs: https://docs.sarvam.ai/api-reference-docs/speech-to-text/transcribe

Key facts (verified 2026-02-28):
  - URL: POST https://api.sarvam.ai/speech-to-text  (NOT /speech-to-text-translate)
  - Accepts: multipart form data
    - file: audio file (WAV, MP3, AAC, M4A, OGG, OPUS, FLAC, WebM, AMR)
    - model: "saarika:v2.5" (default) | "saaras:v3"
    - language_code: BCP-47 (optional for saarika:v2.5). "en-IN", "hi-IN", "unknown"
  - Response: { "transcript": "...", "language_code": "...", ... }
  - Audio works best at 16kHz. Multi-channel is auto-merged.
"""

from __future__ import annotations

import logging
import os
import time

import httpx

from backend.src.providers.base import BaseSTT

logger = logging.getLogger(__name__)

# CORRECT endpoint (NOT /speech-to-text-translate which is a different API)
_SARVAM_STT_URL = "https://api.sarvam.ai/speech-to-text"


class SarvamSTT(BaseSTT):
    """Sarvam AI speech-to-text for Indian + English languages."""

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self._total_calls = 0
        self._total_time_ms = 0.0

        model = config.get("model", "saarika:v2.5")
        language = config.get("language", "en-IN")
        logger.info(f"[SarvamSTT] Initialized: model={model}, language={language}")

    async def transcribe(self, audio: bytes) -> str:
        """Transcribe audio bytes via Sarvam REST API.

        Args:
            audio: Raw audio bytes. Sarvam auto-detects codec for most formats.
                   Supported: WAV, MP3, AAC, M4A, OGG, OPUS, FLAC, WebM, AMR.
        """
        self._total_calls += 1
        call_id = self._total_calls
        start = time.perf_counter()

        logger.info(
            f"[SarvamSTT] Transcribe #{call_id}: "
            f"{len(audio)} bytes audio"
        )

        try:
            api_key = os.getenv("SARVAM_API_KEY", "")
            if not api_key:
                raise ValueError("SARVAM_API_KEY env var not set")

            model = self.config.get("model", "saarika:v2.5")
            language = self.config.get("language", "en-IN")

            # Detect format from magic bytes for proper MIME type
            mime_type, ext = self._detect_format(audio)
            logger.debug(
                f"[SarvamSTT] #{call_id}: detected format={ext}, "
                f"mime={mime_type}, model={model}, lang={language}"
            )

            form_data = {
                "model": model,
                "language_code": language,
            }

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    _SARVAM_STT_URL,
                    files={"file": (f"audio{ext}", audio, mime_type)},
                    data=form_data,
                    headers={"api-subscription-key": api_key},
                    timeout=30.0,
                )

                elapsed_ms = (time.perf_counter() - start) * 1000

                # Log error body before raising
                if response.status_code >= 400:
                    logger.error(
                        f"[SarvamSTT] #{call_id} HTTP {response.status_code} "
                        f"after {elapsed_ms:.0f}ms: {response.text[:500]}"
                    )

                response.raise_for_status()

                data = response.json()
                transcript = data.get("transcript", "")

                self._total_time_ms += elapsed_ms
                logger.info(
                    f"[SarvamSTT] #{call_id}: "
                    f"\"{transcript}\" ({elapsed_ms:.0f}ms)"
                )

                return transcript

        except httpx.HTTPStatusError:
            raise
        except Exception as e:
            elapsed_ms = (time.perf_counter() - start) * 1000
            logger.error(
                f"[SarvamSTT] #{call_id} ERROR after {elapsed_ms:.0f}ms: {e}"
            )
            return ""

    @staticmethod
    def _detect_format(audio: bytes) -> tuple[str, str]:
        """Detect audio format from magic bytes for correct MIME type.

        Returns (mime_type, extension).
        """
        if len(audio) < 12:
            return "audio/wav", ".wav"

        # WAV: RIFF....WAVE
        if audio[:4] == b"RIFF" and audio[8:12] == b"WAVE":
            return "audio/wav", ".wav"
        # MP3: ID3 tag or sync bytes
        if audio[:3] == b"ID3" or (audio[0] == 0xFF and (audio[1] & 0xE0) == 0xE0):
            return "audio/mpeg", ".mp3"
        # OGG
        if audio[:4] == b"OggS":
            return "audio/ogg", ".ogg"
        # FLAC
        if audio[:4] == b"fLaC":
            return "audio/flac", ".flac"
        # M4A / AAC / MP4 (ftyp box at offset 4)
        if audio[4:8] == b"ftyp":
            return "audio/mp4", ".m4a"
        # WebM (EBML header)
        if audio[:4] == b"\x1a\x45\xdf\xa3":
            return "audio/webm", ".webm"
        # AMR
        if audio[:6] == b"#!AMR\n":
            return "audio/amr", ".amr"

        # Default to wav
        return "audio/wav", ".wav"

    @property
    def stats(self) -> dict:
        return {
            "total_calls": self._total_calls,
            "total_time_ms": round(self._total_time_ms, 1),
            "avg_time_ms": round(
                self._total_time_ms / self._total_calls, 1
            ) if self._total_calls else 0,
        }
