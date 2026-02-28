"""ElevenLabs TTS provider — streaming audio synthesis."""

from __future__ import annotations

import logging
import os
from typing import AsyncIterator

from backend.src.providers.base import BaseTTS

logger = logging.getLogger(__name__)


class ElevenLabsTTS(BaseTTS):
    """ElevenLabs streaming text-to-speech."""

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self._client = None

    def _get_client(self):
        """Lazy-init ElevenLabs client."""
        if self._client is None:
            from elevenlabs import ElevenLabs
            api_key = os.getenv("ELEVENLABS_API_KEY", "")
            if not api_key:
                raise ValueError("ELEVENLABS_API_KEY env var not set")
            self._client = ElevenLabs(api_key=api_key)
        return self._client

    async def synthesize(self, text: str) -> AsyncIterator[bytes]:
        """Convert text to speech, yielding audio chunks."""
        try:
            client = self._get_client()
            voice_id = self.config.get("voice_id", "JBFqnCBsd6RMkjVDRZzb")
            model = self.config.get("model", "eleven_turbo_v2_5")

            audio_stream = client.text_to_speech.convert(
                voice_id=voice_id,
                text=text,
                model_id=model,
                voice_settings={
                    "stability": self.config.get("stability", 0.5),
                    "similarity_boost": self.config.get("similarity_boost", 0.75),
                },
            )

            # ElevenLabs returns an iterator of audio bytes
            for chunk in audio_stream:
                if chunk:
                    yield chunk

        except Exception as e:
            logger.error(f"ElevenLabs TTS error: {e}")
            raise
