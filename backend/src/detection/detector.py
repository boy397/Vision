"""YOLO26n object detector with BoT-SORT tracking and mode-specific class filtering.

Uses model.track() instead of model.predict() to assign persistent IDs
to detected objects, avoiding redundant re-analysis of the same object.
"""

from __future__ import annotations

import logging
<<<<<<< Updated upstream
import time
=======
>>>>>>> Stashed changes

import numpy as np

logger = logging.getLogger(__name__)


class Detection:
    """Single detection result with optional tracking ID."""

    __slots__ = ("class_name", "confidence", "bbox", "class_id", "track_id")

    def __init__(
        self,
        class_name: str,
        confidence: float,
        bbox: list[float],
        class_id: int = -1,
        track_id: int = -1,
    ):
        self.class_name = class_name
        self.confidence = confidence
        self.bbox = bbox  # [x1, y1, x2, y2]
        self.class_id = class_id
        self.track_id = track_id  # Persistent across frames (-1 if no tracking)

    def to_dict(self) -> dict:
        d = {
            "class_name": self.class_name,
            "confidence": round(self.confidence, 3),
            "bbox": [round(c, 1) for c in self.bbox],
        }
        if self.track_id >= 0:
            d["track_id"] = self.track_id
        return d


class Detector:
    """YOLO-based object detector with BoT-SORT tracking.

    Uses ultralytics YOLO model.track() for persistent object tracking.
    Supports mode-specific class filtering and configurable confidence thresholds.
    """

    def __init__(self, config: dict) -> None:
        self._model = None
        self._model_name = config.get("model", "yolo26n")
        self._device = config.get("device", "cuda")
        self._input_size = config.get("input_size", 640)
        self._default_confidence = config.get("confidence", 0.45)

        # Tracking state
        self._use_tracking = True
        self._tracked_ids: set[int] = set()  # IDs we've already fully analyzed
        self._frame_count = 0
        self._total_detect_time_ms = 0.0

        logger.info(
            f"[Detector] Initialized: model={self._model_name}, "
            f"device={self._device}, input_size={self._input_size}, "
            f"confidence={self._default_confidence}, tracking={self._use_tracking}"
        )

    def _load_model(self):
        """Lazy-load YOLO model."""
        if self._model is None:
            from ultralytics import YOLO
            model_path = f"{self._model_name}.pt"
            logger.info(f"[Detector] Loading YOLO model: {model_path} on {self._device}")
            load_start = time.perf_counter()
            self._model = YOLO(model_path)
            load_ms = (time.perf_counter() - load_start) * 1000
            logger.info(f"[Detector] Model loaded in {load_ms:.0f}ms")
        return self._model

    def detect(
        self,
        frame: np.ndarray,
        confidence: float | None = None,
        filter_classes: list[str] | None = None,
    ) -> list[Detection]:
        """Run detection + tracking on a frame.

        Uses model.track() with BoT-SORT for persistent IDs.
        Falls back to model.predict() if tracking errors out.

        Args:
            frame: BGR image as numpy array.
            confidence: Override confidence threshold.
            filter_classes: Only return detections matching these class names.

        Returns:
            List of Detection objects with track_id set.
        """
        model = self._load_model()
        conf = confidence or self._default_confidence
        self._frame_count += 1

        start = time.perf_counter()

        # Try tracking first, fall back to plain predict
        results = None
        tracking_active = False

        if self._use_tracking:
            try:
                results = model.track(
                    frame,
                    conf=conf,
                    imgsz=self._input_size,
                    device=self._device,
                    verbose=False,
                    persist=True,  # Keep tracker state across calls
                    tracker="botsort.yaml",  # BoT-SORT tracker
                )
                tracking_active = True
            except Exception as e:
                logger.warning(
                    f"[Detector] Tracking failed, falling back to predict: {e}"
                )
                self._use_tracking = False

        if results is None:
            results = model.predict(
                frame,
                conf=conf,
                imgsz=self._input_size,
                device=self._device,
                verbose=False,
            )

        elapsed_ms = (time.perf_counter() - start) * 1000
        self._total_detect_time_ms += elapsed_ms

        detections: list[Detection] = []
        new_track_ids: list[int] = []
        reused_track_ids: list[int] = []

        for result in results:
            if result.boxes is None:
                continue
            for idx, box in enumerate(result.boxes):
                cls_id = int(box.cls[0])
                cls_name = model.names.get(cls_id, f"class_{cls_id}")
                det_conf = float(box.conf[0])
                bbox = box.xyxy[0].tolist()

                # Extract track ID if available
                track_id = -1
                if tracking_active and box.id is not None:
                    track_id = int(box.id[0])

                det = Detection(
                    class_name=cls_name,
                    confidence=det_conf,
                    bbox=bbox,
                    class_id=cls_id,
                    track_id=track_id,
                )

                # Apply class filter if specified
                if filter_classes is None or cls_name in filter_classes:
                    detections.append(det)

                    # Track new vs seen IDs
                    if track_id >= 0:
                        if track_id in self._tracked_ids:
                            reused_track_ids.append(track_id)
                        else:
                            new_track_ids.append(track_id)

        # Debug logging
        avg_ms = self._total_detect_time_ms / self._frame_count if self._frame_count else 0
        logger.debug(
            f"[Detector] Frame #{self._frame_count}: "
            f"{len(detections)} detections in {elapsed_ms:.1f}ms "
            f"(avg={avg_ms:.1f}ms), "
            f"tracking={'ON' if tracking_active else 'OFF'}, "
            f"new_tracks={new_track_ids}, reused_tracks={reused_track_ids}"
        )

        if detections:
            for d in detections:
                logger.debug(
                    f"  → {d.class_name} (conf={d.confidence:.2f}, "
                    f"track_id={d.track_id}, "
                    f"bbox=[{d.bbox[0]:.0f},{d.bbox[1]:.0f},{d.bbox[2]:.0f},{d.bbox[3]:.0f}])"
                )

        return detections

    def is_new_object(self, detection: Detection) -> bool:
        """Check if a tracked object is new (not previously analyzed).

        Args:
            detection: Detection with a track_id.

        Returns:
            True if this track_id hasn't been seen before.
        """
        if detection.track_id < 0:
            return True  # No tracking → always treat as new
        if detection.track_id in self._tracked_ids:
            return False
        return True

    def mark_analyzed(self, detection: Detection) -> None:
        """Mark a tracked object as analyzed (won't trigger re-analysis).

        Args:
            detection: Detection to mark as done.
        """
        if detection.track_id >= 0:
            self._tracked_ids.add(detection.track_id)
            logger.debug(
                f"[Detector] Marked track_id={detection.track_id} "
                f"({detection.class_name}) as analyzed. "
                f"Total tracked: {len(self._tracked_ids)}"
            )

    def clear_tracked(self) -> None:
        """Clear tracked object IDs (e.g., on mode switch or reset)."""
        count = len(self._tracked_ids)
        self._tracked_ids.clear()
        logger.info(f"[Detector] Cleared {count} tracked object IDs")

    @property
    def stats(self) -> dict:
        """Return detector statistics for debugging."""
        return {
            "frame_count": self._frame_count,
            "avg_detect_ms": round(
                self._total_detect_time_ms / self._frame_count, 1
            )
            if self._frame_count
            else 0,
            "tracked_objects": len(self._tracked_ids),
            "tracking_active": self._use_tracking,
        }
