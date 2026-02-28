"""Sarvam TTS provider — Hindi/regional language text-to-speech (optional)."""

from __future__ import annotations

import logging
import os
from typing import AsyncIterator

import httpx

from backend.src.providers.base import BaseTTS

logger = logging.getLogger(__name__)

_SARVAM_TTS_URL = "https://api.sarvam.ai/text-to-speech"


class SarvamTTS(BaseTTS):
    """Sarvam AI text-to-speech for Indian languages."""

    def __init__(self, config: dict) -> None:
        super().__init__(config)

    async def synthesize(self, text: str) -> AsyncIterator[bytes]:
        """Convert text to speech via Sarvam API."""
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
                response.raise_for_status()

                data = response.json()
                # Sarvam returns base64-encoded audio
                import base64
                for audio_b64 in data.get("audios", []):
                    yield base64.b64decode(audio_b64)

        except Exception as e:
            logger.error(f"Sarvam TTS error: {e}")
            raise
