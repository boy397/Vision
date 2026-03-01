"""Sarvam TTS Streaming — WebSocket-based real-time text-to-speech.

Uses Sarvam's bulbul:v3 streaming endpoint for ultra-low latency audio synthesis.
The backend acts as a relay: the frontend sends text (or derives it from LLM output),
this class opens a WS to Sarvam and yields MP3 chunks as they arrive.

API docs: https://docs.sarvam.ai/api-reference-docs/text-to-speech/api/streaming

Key facts:
  - Model: bulbul:v3
  - Input message types: config, text, flush, ping
  - Output message types: audio (base64 MP3 chunks), completion event
  - output_audio_codec: mp3 (default), wav, aac, opus, flac, pcm, mulaw, alaw
"""

from __future__ import annotations

import asyncio
import base64
import logging
import os
import time
from typing import AsyncIterator

logger = logging.getLogger(__name__)


class SarvamStreamingTTS:
    """Sarvam WebSocket TTS — yields MP3 audio chunks as they arrive.

    This is truly streaming: the first audio chunk can be played immediately
    while synthesis of the remaining text is still in progress.

    Usage:
        async for chunk in SarvamStreamingTTS(config).synthesize(text):
            # chunk is raw MP3 bytes, ready to send to client
            yield chunk
    """

    def __init__(self, config: dict) -> None:
        self.config = config
        self._total_calls = 0
        self._total_time_ms = 0.0
        self._total_bytes = 0

        model = config.get("model", "bulbul:v3")
        speaker = config.get("speaker", "shubh")
        language = config.get("language", "en-IN")
        logger.info(
            f"[SarvamStreamingTTS] Initialized: model={model}, "
            f"lang={language}, speaker={speaker}"
        )

    async def synthesize(self, text: str) -> AsyncIterator[bytes]:
        """Convert text to speech via Sarvam WebSocket streaming API.

        Yields raw MP3 bytes progressively — TTFB much lower than REST.
        """
        try:
            from sarvamai import AsyncSarvamAI, AudioOutput, EventResponse
        except ImportError:
            raise RuntimeError(
                "sarvamai package not installed. Run: pip install sarvamai"
            )

        api_key = os.getenv("SARVAM_API_KEY", "")
        if not api_key:
            raise ValueError("SARVAM_API_KEY env var not set")

        self._total_calls += 1
        call_id = self._total_calls
        text_preview = text[:80] + "..." if len(text) > 80 else text

        model = self.config.get("model", "bulbul:v3")
        speaker = self.config.get("speaker", "shubh")
        language = self.config.get("language", "en-IN")
        pace = self.config.get("pace", None)

        logger.info(
            f"[SarvamStreamingTTS] Synth #{call_id} START: "
            f"text_len={len(text)}, preview=\"{text_preview}\""
        )

        start_time = time.perf_counter()
        first_chunk_time = None
        chunk_count = 0
        total_bytes = 0

        client = AsyncSarvamAI(api_subscription_key=api_key)

        try:
            # send_completion_event=True gives us a clean "final" signal to break the loop
            async with client.text_to_speech_streaming.connect(
                model=model,
                send_completion_event=True,
            ) as ws:
                # Step 1: Send config (must be first message)
                configure_kwargs = {
                    "target_language_code": language,
                    "speaker": speaker,
                    "output_audio_codec": "mp3",
                    "min_buffer_size": 40,
                    "max_chunk_length": 200,
                }
                if pace is not None:
                    configure_kwargs["pace"] = pace
                await ws.configure(**configure_kwargs)

                # Step 2: Send text
                await ws.convert(text)

                # Step 3: Flush to force processing
                await ws.flush()

                logger.debug(f"[SarvamStreamingTTS] #{call_id}: config+text+flush sent")

                # Step 4: Stream audio chunks as they arrive
                async for message in ws:
                    if isinstance(message, AudioOutput):
                        chunk_count += 1
                        audio_chunk = base64.b64decode(message.data.audio)
                        total_bytes += len(audio_chunk)

                        if first_chunk_time is None:
                            first_chunk_time = time.perf_counter()
                            ttfb = (first_chunk_time - start_time) * 1000
                            logger.info(
                                f"[SarvamStreamingTTS] #{call_id} TTFB: {ttfb:.0f}ms"
                            )

                        yield audio_chunk

                    elif isinstance(message, EventResponse):
                        event_type = getattr(message.data, "event_type", "")
                        logger.debug(
                            f"[SarvamStreamingTTS] #{call_id}: event={event_type}"
                        )
                        if event_type == "final":
                            break

        except Exception as e:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            logger.error(
                f"[SarvamStreamingTTS] #{call_id} ERROR after {elapsed_ms:.0f}ms: {e}"
            )
            raise

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        self._total_time_ms += elapsed_ms
        self._total_bytes += total_bytes

        logger.info(
            f"[SarvamStreamingTTS] #{call_id} DONE: "
            f"{chunk_count} chunks, {total_bytes} bytes, {elapsed_ms:.0f}ms total"
        )

    @property
    def stats(self) -> dict:
        return {
            "total_calls": self._total_calls,
            "total_time_ms": round(self._total_time_ms, 1),
            "total_bytes": self._total_bytes,
            "avg_time_ms": round(
                self._total_time_ms / self._total_calls, 1
            ) if self._total_calls else 0,
            "mode": "WebSocket (true streaming)",
        }
