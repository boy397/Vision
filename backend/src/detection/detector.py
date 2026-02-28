"""YOLO26n object detector with mode-specific class filtering."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class Detection:
    """Single detection result."""

    __slots__ = ("class_name", "confidence", "bbox", "class_id")

    def __init__(self, class_name: str, confidence: float, bbox: list[float], class_id: int = -1):
        self.class_name = class_name
        self.confidence = confidence
        self.bbox = bbox  # [x1, y1, x2, y2]
        self.class_id = class_id

    def to_dict(self) -> dict:
        return {
            "class_name": self.class_name,
            "confidence": round(self.confidence, 3),
            "bbox": [round(c, 1) for c in self.bbox],
        }


class Detector:
    """YOLO-based object detector.

    Wraps ultralytics YOLO model. Supports mode-specific class filtering
    and configurable confidence thresholds.
    """

    def __init__(self, config: dict) -> None:
        self._model = None
        self._model_name = config.get("model", "yolo26n")
        self._device = config.get("device", "cuda")
        self._input_size = config.get("input_size", 640)
        self._default_confidence = config.get("confidence", 0.45)

    def _load_model(self):
        """Lazy-load YOLO model."""
        if self._model is None:
            from ultralytics import YOLO
            model_path = f"{self._model_name}.pt"
            logger.info(f"Loading YOLO model: {model_path} on {self._device}")
            self._model = YOLO(model_path)
        return self._model

    def detect(
        self,
        frame: np.ndarray,
        confidence: float | None = None,
        filter_classes: list[str] | None = None,
    ) -> list[Detection]:
        """Run detection on a frame.

        Args:
            frame: BGR image as numpy array.
            confidence: Override confidence threshold.
            filter_classes: Only return detections matching these class names.

        Returns:
            List of Detection objects.
        """
        model = self._load_model()
        conf = confidence or self._default_confidence

        results = model.predict(
            frame,
            conf=conf,
            imgsz=self._input_size,
            device=self._device,
            verbose=False,
        )

        detections: list[Detection] = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                cls_id = int(box.cls[0])
                cls_name = model.names.get(cls_id, f"class_{cls_id}")
                det_conf = float(box.conf[0])
                bbox = box.xyxy[0].tolist()

                det = Detection(
                    class_name=cls_name,
                    confidence=det_conf,
                    bbox=bbox,
                    class_id=cls_id,
                )

                # Apply class filter if specified
                if filter_classes is None or cls_name in filter_classes:
                    detections.append(det)

        return detections
