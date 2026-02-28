"""FunASR Speech-to-Text provider for fast local processing."""

from __future__ import annotations

import logging

from backend.src.providers.base import BaseSTT

logger = logging.getLogger(__name__)


class FunASRSTT(BaseSTT):
    """FunASR Local Speech-to-Text implementation (Streaming supported)."""

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self._model = None

    def _get_model(self):
        """Lazy-init FunASR AutoModel."""
        if self._model is None:
            from funasr import AutoModel
            # Load model - fallback to a fast English model if not specified
            model_name = self.config.get("model", "paraformer-en")
            logger.info(f"Loading FunASR model: {model_name}...")
            self._model = AutoModel(model=model_name, disable_update=True)
            logger.info("FunASR model loaded successfully.")
        return self._model

    async def transcribe(self, audio: bytes) -> str:
        """Transcribe audio bytes to text locally."""
        try:
            model = self._get_model()
            
            # FunASR generally expects a waveform array or a file path.
            # Convert bytes to numpy array (16kHz, 16-bit, Mono)
            import numpy as np
            audio_array = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0

            # Run inference
            res = model.generate(input=audio_array)
            
            if res and isinstance(res, list) and len(res) > 0:
                result_item = res[0]
                if isinstance(result_item, dict):
                    transcript = result_item.get("text", "")
                    return transcript.strip()
                elif isinstance(result_item, str):
                    return result_item.strip()
            
            return ""

        except Exception as e:
            logger.error(f"FunASR STT error: {e}")
            return ""
