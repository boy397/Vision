"""State change tracker — prevents redundant Tier 2 API calls.

Compares detections frame-to-frame using IoU. If detected objects
and positions haven't changed, returns state_changed=False.

Now includes debug logging for every decision point.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def _iou(box_a: list[float], box_b: list[float]) -> float:
    """Compute Intersection over Union between two bounding boxes."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter

    return inter / union if union > 0 else 0.0


class StateTracker:
    """Tracks detection state across frames to gate Tier 2 calls.

    If the same objects are in roughly the same positions as the
    previous frame, state_changed=False → skip expensive LLM call.
    """

    def __init__(self, iou_threshold: float = 0.3) -> None:
        self._iou_threshold = iou_threshold
        self._previous_detections: list[dict] = []
        self._previous_classes: set[str] = set()
        self._check_count = 0
        self._change_count = 0
        self._no_change_count = 0

        logger.info(f"[StateTracker] Initialized: iou_threshold={iou_threshold}")

    def check(self, detections: list[dict]) -> bool:
        """Check if detections represent a state change from previous frame.

        Args:
            detections: List of detection dicts with 'class_name' and 'bbox'.

        Returns:
            True if state changed (should trigger Tier 2), False if same.
        """
        self._check_count += 1
        current_classes = {d["class_name"] for d in detections}

        # No detections → state changed only if we previously had detections
        if not detections:
            changed = bool(self._previous_detections)
            self._previous_detections = []
            self._previous_classes = set()
            if changed:
                self._change_count += 1
            else:
                self._no_change_count += 1
            logger.debug(
                f"[StateTracker] Check #{self._check_count}: "
                f"no detections, was={'non-empty' if changed else 'empty'} → "
                f"changed={changed}"
            )
            return changed

        # Different classes → state changed
        if current_classes != self._previous_classes:
            logger.debug(
                f"[StateTracker] Check #{self._check_count}: "
                f"classes changed: {self._previous_classes} → {current_classes} → changed=True"
            )
            self._update(detections, current_classes)
            self._change_count += 1
            return True

        # Different count → state changed
        if len(detections) != len(self._previous_detections):
            logger.debug(
                f"[StateTracker] Check #{self._check_count}: "
                f"count changed: {len(self._previous_detections)} → {len(detections)} → changed=True"
            )
            self._update(detections, current_classes)
            self._change_count += 1
            return True

        # Check if positions changed significantly (IoU-based)
        matched = 0
        for curr in detections:
            for prev in self._previous_detections:
                if curr["class_name"] == prev["class_name"]:
                    iou = _iou(curr["bbox"], prev["bbox"])
                    if iou > self._iou_threshold:
                        matched += 1
                        break

        # If all objects matched → no state change
        if matched == len(detections):
            self._no_change_count += 1
            logger.debug(
                f"[StateTracker] Check #{self._check_count}: "
                f"all {matched}/{len(detections)} matched (IoU>{self._iou_threshold}) → "
                f"changed=False [total: {self._change_count} changes, "
                f"{self._no_change_count} no-changes]"
            )
            return False

        logger.debug(
            f"[StateTracker] Check #{self._check_count}: "
            f"position change: {matched}/{len(detections)} matched → changed=True"
        )
        self._update(detections, current_classes)
        self._change_count += 1
        return True

    def _update(self, detections: list[dict], classes: set[str]) -> None:
        """Update stored state."""
        self._previous_detections = detections
        self._previous_classes = classes

    def reset(self) -> None:
        """Reset state tracker (e.g. on mode switch)."""
        logger.info(
            f"[StateTracker] Reset (had {len(self._previous_detections)} detections, "
            f"{self._check_count} checks, {self._change_count} changes)"
        )
        self._previous_detections = []
        self._previous_classes = set()

    @property
    def stats(self) -> dict:
        """Return state tracker statistics."""
        return {
            "total_checks": self._check_count,
            "state_changes": self._change_count,
            "no_changes": self._no_change_count,
            "iou_threshold": self._iou_threshold,
            "current_objects": len(self._previous_detections),
        }
