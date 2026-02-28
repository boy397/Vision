"""Google Cloud Speech-to-Text provider."""

from __future__ import annotations

import logging
import os

from backend.src.providers.base import BaseSTT

logger = logging.getLogger(__name__)


class GoogleSTT(BaseSTT):
    """Google Cloud Speech-to-Text implementation."""

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self._client = None

    def _get_client(self):
        """Lazy-init Google Speech client."""
        if self._client is None:
            from google.cloud import speech
            self._client = speech.SpeechClient()
        return self._client

    async def transcribe(self, audio: bytes) -> str:
        """Transcribe audio bytes to text."""
        try:
            from google.cloud import speech

            client = self._get_client()
            language = self.config.get("language", "en-IN")

            audio_obj = speech.RecognitionAudio(content=audio)
            config = speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=16000,
                language_code=language,
                model=self.config.get("model", "latest_long"),
                enable_automatic_punctuation=True,
            )

            response = client.recognize(config=config, audio=audio_obj)

            transcript = ""
            for result in response.results:
                transcript += result.alternatives[0].transcript

            return transcript.strip()

        except Exception as e:
            logger.error(f"Google STT error: {e}")
            return ""
