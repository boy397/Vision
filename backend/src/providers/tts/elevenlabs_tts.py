"""ElevenLabs TTS provider — true streaming audio synthesis.

Uses convert_as_stream() for chunked transfer encoding (real-time streaming).
Falls back to convert() if streaming endpoint is unavailable.
"""

from __future__ import annotations

import logging
import os
import time
from typing import AsyncIterator

from backend.src.providers.base import BaseTTS

logger = logging.getLogger(__name__)


class ElevenLabsTTS(BaseTTS):
    """ElevenLabs streaming text-to-speech.

    Uses the streaming API (convert_as_stream) for minimum latency.
    Audio is yielded chunk-by-chunk as it's generated.
    """

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self._client = None
        self._total_synth_calls = 0
        self._total_synth_time_ms = 0.0
        self._total_bytes_streamed = 0
        logger.info(
            f"[ElevenLabsTTS] Initialized: "
            f"model={config.get('model', 'eleven_turbo_v2_5')}, "
            f"voice_id={config.get('voice_id', 'JBFqnCBsd6RMkjVDRZzb')}, "
            f"stream={config.get('stream', True)}"
        )

    def _get_client(self):
        """Lazy-init ElevenLabs client."""
        if self._client is None:
            from elevenlabs import ElevenLabs
            api_key = os.getenv("ELEVENLABS_API_KEY", "")
            if not api_key:
                raise ValueError("ELEVENLABS_API_KEY env var not set")
            self._client = ElevenLabs(api_key=api_key)
            logger.info("[ElevenLabsTTS] Client initialized")
        return self._client

    async def synthesize(self, text: str) -> AsyncIterator[bytes]:
        """Convert text to speech, yielding audio chunks via streaming API.

        Uses convert_as_stream() which streams audio as it's generated,
        giving much lower time-to-first-byte than convert().
        """
        self._total_synth_calls += 1
        call_id = self._total_synth_calls
        text_preview = text[:80] + "..." if len(text) > 80 else text

        logger.info(
            f"[ElevenLabsTTS] Synth #{call_id} START: "
            f"text_len={len(text)}, preview=\"{text_preview}\""
        )

        start_time = time.perf_counter()
        first_chunk_time = None
        chunk_count = 0
        total_bytes = 0

        try:
            client = self._get_client()
            voice_id = self.config.get("voice_id", "JBFqnCBsd6RMkjVDRZzb")
            model = self.config.get("model", "eleven_turbo_v2_5")

            voice_settings = {
                "stability": self.config.get("stability", 0.5),
                "similarity_boost": self.config.get("similarity_boost", 0.75),
            }

            # Use streaming endpoint for lower TTFB
            use_stream = self.config.get("stream", True)

            if use_stream:
                try:
                    # convert_as_stream() — sends audio chunks as they're generated
                    logger.debug(
                        f"[ElevenLabsTTS] Using convert_as_stream() "
                        f"(model={model}, voice={voice_id})"
                    )
                    audio_stream = client.text_to_speech.convert_as_stream(
                        voice_id=voice_id,
                        text=text,
                        model_id=model,
                        voice_settings=voice_settings,
                    )
                except AttributeError:
                    # Older SDK version — fallback to convert()
                    logger.warning(
                        "[ElevenLabsTTS] convert_as_stream() not available, "
                        "falling back to convert() (non-streaming)"
                    )
                    audio_stream = client.text_to_speech.convert(
                        voice_id=voice_id,
                        text=text,
                        model_id=model,
                        voice_settings=voice_settings,
                    )
            else:
                # Non-streaming mode (for debugging or when latency doesn't matter)
                logger.debug("[ElevenLabsTTS] Using convert() (non-streaming)")
                audio_stream = client.text_to_speech.convert(
                    voice_id=voice_id,
                    text=text,
                    model_id=model,
                    voice_settings=voice_settings,
                )

            # Yield chunks as they arrive
            for chunk in audio_stream:
                if chunk:
                    chunk_count += 1
                    chunk_len = len(chunk)
                    total_bytes += chunk_len

                    if first_chunk_time is None:
                        first_chunk_time = time.perf_counter()
                        ttfb = (first_chunk_time - start_time) * 1000
                        logger.info(
                            f"[ElevenLabsTTS] Synth #{call_id} TTFB: {ttfb:.0f}ms "
                            f"(first chunk: {chunk_len} bytes)"
                        )

                    yield chunk

            # Final stats
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            self._total_synth_time_ms += elapsed_ms
            self._total_bytes_streamed += total_bytes

            logger.info(
                f"[ElevenLabsTTS] Synth #{call_id} DONE: "
                f"{chunk_count} chunks, {total_bytes} bytes, "
                f"{elapsed_ms:.0f}ms total, "
                f"avg_chunk_size={total_bytes // max(chunk_count, 1)} bytes"
            )

        except Exception as e:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            logger.error(
                f"[ElevenLabsTTS] Synth #{call_id} ERROR after {elapsed_ms:.0f}ms: {e}"
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
        }
