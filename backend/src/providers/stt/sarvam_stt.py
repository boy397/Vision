"""Sarvam STT provider — Indian language speech-to-text (optional)."""

from __future__ import annotations

import logging
import os

import httpx

from backend.src.providers.base import BaseSTT

logger = logging.getLogger(__name__)

_SARVAM_STT_URL = "https://api.sarvam.ai/speech-to-text-translate"


class SarvamSTT(BaseSTT):
    """Sarvam AI speech-to-text for Indian languages."""

    def __init__(self, config: dict) -> None:
        super().__init__(config)

    async def transcribe(self, audio: bytes) -> str:
        """Transcribe audio bytes via Sarvam API."""
        try:
            api_key = os.getenv("SARVAM_API_KEY", "")
            if not api_key:
                raise ValueError("SARVAM_API_KEY env var not set")

            # Sarvam expects multipart form data with audio file
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    _SARVAM_STT_URL,
                    files={"file": ("audio.wav", audio, "audio/wav")},
                    data={
                        "language_code": self.config.get("language", "hi-IN"),
                        "model": self.config.get("model", "saarika:v2"),
                    },
                    headers={"api-subscription-key": api_key},
                    timeout=30.0,
                )
                response.raise_for_status()

                data = response.json()
                return data.get("transcript", "")

        except Exception as e:
            logger.error(f"Sarvam STT error: {e}")
            return ""
