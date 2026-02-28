"""Voice Activity Detection using webrtcvad."""

from __future__ import annotations

import logging
import struct
import collections

logger = logging.getLogger(__name__)


class VAD:
    """Voice Activity Detector using webrtcvad.

    Buffers audio frames and triggers when speech ends
    (silence after speech detected).
    """

    def __init__(
        self,
        aggressiveness: int = 2,
        sample_rate: int = 16000,
        frame_duration_ms: int = 30,
        silence_timeout_ms: int = 800,
    ) -> None:
        self._sample_rate = sample_rate
        self._frame_duration_ms = frame_duration_ms
        self._silence_timeout_ms = silence_timeout_ms
        self._frame_size = int(sample_rate * frame_duration_ms / 1000)
        self._silence_frames = int(silence_timeout_ms / frame_duration_ms)

        self._vad = None
        self._aggressiveness = aggressiveness
        self._ring_buffer: collections.deque = collections.deque(maxlen=self._silence_frames)
        self._triggered = False
        self._voiced_frames: list[bytes] = []

    def _get_vad(self):
        """Lazy-init webrtcvad instance."""
        if self._vad is None:
            import webrtcvad
            self._vad = webrtcvad.Vad(self._aggressiveness)
        return self._vad

    def process_frame(self, frame: bytes) -> bytes | None:
        """Process a single audio frame.

        Args:
            frame: Raw PCM audio bytes (16-bit, mono).

        Returns:
            Complete utterance audio bytes when speech ends, None otherwise.
        """
        vad = self._get_vad()
        is_speech = vad.is_speech(frame, self._sample_rate)

        if not self._triggered:
            self._ring_buffer.append((frame, is_speech))
            num_voiced = sum(1 for _, speech in self._ring_buffer if speech)

            # Trigger when enough voiced frames detected
            if num_voiced > 0.8 * self._ring_buffer.maxlen:
                self._triggered = True
                self._voiced_frames = [f for f, _ in self._ring_buffer]
                self._ring_buffer.clear()
        else:
            self._voiced_frames.append(frame)
            self._ring_buffer.append((frame, is_speech))
            num_unvoiced = sum(1 for _, speech in self._ring_buffer if not speech)

            # End of speech when enough silence detected
            if num_unvoiced > 0.8 * self._ring_buffer.maxlen:
                self._triggered = False
                audio = b"".join(self._voiced_frames)
                self._voiced_frames = []
                self._ring_buffer.clear()
                return audio

        return None

    def reset(self) -> None:
        """Reset VAD state."""
        self._triggered = False
        self._voiced_frames = []
        self._ring_buffer.clear()
