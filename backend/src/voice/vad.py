"""Voice Activity Detection using webrtcvad.

Includes proper frame splitting, debug logging, and continuous operation tracking.
"""

from __future__ import annotations

import logging
import collections
import time

logger = logging.getLogger(__name__)


class VAD:
    """Voice Activity Detector using webrtcvad.

    Buffers audio frames and triggers when speech ends
    (silence after speech detected).

    IMPORTANT: webrtcvad requires frames of exact size:
      - 10ms, 20ms, or 30ms at 8000/16000/32000/48000 Hz
      - For 16kHz mono 16-bit @ 30ms = 480 samples = 960 bytes
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
        # Number of samples per frame (e.g. 480 for 30ms at 16kHz)
        self._frame_samples = int(sample_rate * frame_duration_ms / 1000)
        # Byte size per frame (16-bit = 2 bytes per sample)
        self._frame_bytes = self._frame_samples * 2
        self._silence_frames = int(silence_timeout_ms / frame_duration_ms)

        self._vad = None
        self._aggressiveness = aggressiveness
        self._ring_buffer: collections.deque = collections.deque(maxlen=self._silence_frames)
        self._triggered = False
        self._voiced_frames: list[bytes] = []
        # Buffer for incomplete frames from the network
        self._pending_bytes = b""

        # Debug stats
        self._total_frames_processed = 0
        self._speech_frames = 0
        self._silence_frames_count = 0
        self._last_log_time = time.time()
        self._utterance_count = 0
        self._speech_start_time: float | None = None

        logger.info(
            f"[VAD] Initialized: aggressiveness={aggressiveness}, "
            f"sample_rate={sample_rate}Hz, frame_duration={frame_duration_ms}ms, "
            f"frame_bytes={self._frame_bytes}, "
            f"silence_timeout={silence_timeout_ms}ms ({self._silence_frames} frames)"
        )

    def _get_vad(self):
        """Lazy-init webrtcvad instance."""
        if self._vad is None:
            import webrtcvad
            self._vad = webrtcvad.Vad(self._aggressiveness)
            logger.info(f"[VAD] webrtcvad initialized (aggressiveness={self._aggressiveness})")
        return self._vad

    def process_chunk(self, audio_data: bytes) -> bytes | None:
        """Process an arbitrary-sized audio chunk.

        Splits into correct frame sizes for webrtcvad and processes each.
        Returns complete utterance when speech ends, None otherwise.

        Args:
            audio_data: Raw PCM audio bytes (16-bit, mono, 16kHz).
                       Can be any size — will be buffered and split.

        Returns:
            Complete utterance audio bytes when speech ends, None otherwise.
        """
        # Append to pending buffer
        self._pending_bytes += audio_data

        # Log periodically (every 5 seconds)
        now = time.time()
        if now - self._last_log_time > 5.0:
            logger.debug(
                f"[VAD] Stats: total_frames={self._total_frames_processed}, "
                f"speech_frames={self._speech_frames}, "
                f"silence_frames={self._silence_frames_count}, "
                f"triggered={self._triggered}, "
                f"voiced_buffer={len(self._voiced_frames)} frames, "
                f"pending_bytes={len(self._pending_bytes)}, "
                f"utterances_so_far={self._utterance_count}"
            )
            self._last_log_time = now

        # Process all complete frames from the buffer
        result = None
        while len(self._pending_bytes) >= self._frame_bytes:
            frame = self._pending_bytes[:self._frame_bytes]
            self._pending_bytes = self._pending_bytes[self._frame_bytes:]
            result = self._process_single_frame(frame)
            if result is not None:
                # Return the utterance immediately; remaining bytes stay buffered
                return result

        return None

    def process_frame(self, frame: bytes) -> bytes | None:
        """Process audio — delegates to process_chunk for proper frame splitting.

        Kept for backward compatibility. Works with any size input now.
        """
        return self.process_chunk(frame)

    def _process_single_frame(self, frame: bytes) -> bytes | None:
        """Process a single correctly-sized audio frame.

        Args:
            frame: Exactly self._frame_bytes of raw PCM audio.

        Returns:
            Complete utterance audio bytes when speech ends, None otherwise.
        """
        vad = self._get_vad()
        self._total_frames_processed += 1

        try:
            is_speech = vad.is_speech(frame, self._sample_rate)
        except Exception as e:
            logger.error(
                f"[VAD] webrtcvad.is_speech() error: {e}, "
                f"frame_len={len(frame)}, expected={self._frame_bytes}"
            )
            return None

        if is_speech:
            self._speech_frames += 1
        else:
            self._silence_frames_count += 1

        if not self._triggered:
            self._ring_buffer.append((frame, is_speech))
            num_voiced = sum(1 for _, speech in self._ring_buffer if speech)
            threshold = 0.8 * self._ring_buffer.maxlen

            # Trigger when enough voiced frames detected
            if num_voiced > threshold:
                self._triggered = True
                self._speech_start_time = time.time()
                self._voiced_frames = [f for f, _ in self._ring_buffer]
                self._ring_buffer.clear()
                logger.info(
                    f"[VAD] 🎙️ Speech STARTED "
                    f"(voiced={num_voiced}/{self._ring_buffer.maxlen}, "
                    f"threshold={threshold:.0f})"
                )
        else:
            self._voiced_frames.append(frame)
            self._ring_buffer.append((frame, is_speech))
            num_unvoiced = sum(1 for _, speech in self._ring_buffer if not speech)
            threshold = 0.8 * self._ring_buffer.maxlen

            # End of speech when enough silence detected
            if num_unvoiced > threshold:
                self._triggered = False
                audio = b"".join(self._voiced_frames)
                duration_ms = len(audio) / (self._sample_rate * 2) * 1000
                speech_duration = (
                    f"{time.time() - self._speech_start_time:.2f}s"
                    if self._speech_start_time
                    else "unknown"
                )
                self._utterance_count += 1
                self._voiced_frames = []
                self._ring_buffer.clear()
                self._speech_start_time = None

                logger.info(
                    f"[VAD] 🔇 Speech ENDED — utterance #{self._utterance_count}: "
                    f"{len(audio)} bytes ({duration_ms:.0f}ms audio), "
                    f"wall_time={speech_duration}"
                )
                return audio

        return None

    def reset(self) -> None:
        """Reset VAD state."""
        logger.info(
            f"[VAD] Reset (was triggered={self._triggered}, "
            f"buffered_frames={len(self._voiced_frames)}, "
            f"pending_bytes={len(self._pending_bytes)})"
        )
        self._triggered = False
        self._voiced_frames = []
        self._ring_buffer.clear()
        self._pending_bytes = b""
        self._speech_start_time = None

    @property
    def is_triggered(self) -> bool:
        """Whether VAD is currently in speech-detected state."""
        return self._triggered

    @property
    def stats(self) -> dict:
        """Return current VAD statistics for debugging."""
        return {
            "total_frames": self._total_frames_processed,
            "speech_frames": self._speech_frames,
            "silence_frames": self._silence_frames_count,
            "utterance_count": self._utterance_count,
            "is_triggered": self._triggered,
            "buffered_voiced_frames": len(self._voiced_frames),
            "pending_bytes": len(self._pending_bytes),
        }
