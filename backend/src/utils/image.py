"""Image utilities — ROI cropping, encoding, resizing."""

from __future__ import annotations

import base64
import io
import logging

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


def crop_roi(frame: np.ndarray, bbox: list[float], padding: int = 10) -> np.ndarray:
    """Crop region of interest from frame using bounding box.

    Args:
        frame: Full BGR image.
        bbox: [x1, y1, x2, y2] bounding box.
        padding: Extra pixels around the box.

    Returns:
        Cropped image as numpy array.
    """
    h, w = frame.shape[:2]
    x1 = max(0, int(bbox[0]) - padding)
    y1 = max(0, int(bbox[1]) - padding)
    x2 = min(w, int(bbox[2]) + padding)
    y2 = min(h, int(bbox[3]) + padding)
    return frame[y1:y2, x1:x2]


def encode_jpeg(frame: np.ndarray, quality: int = 85) -> bytes:
    """Encode numpy frame to JPEG bytes."""
    params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    _, buffer = cv2.imencode(".jpg", frame, params)
    return buffer.tobytes()


def frame_to_base64(frame: np.ndarray, quality: int = 85) -> str:
    """Encode frame to base64 JPEG string."""
    return base64.b64encode(encode_jpeg(frame, quality)).decode()


def resize_frame(frame: np.ndarray, max_dim: int = 1280) -> np.ndarray:
    """Resize frame so largest dimension ≤ max_dim, preserving aspect ratio."""
    h, w = frame.shape[:2]
    if max(h, w) <= max_dim:
        return frame
    scale = max_dim / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)


def bytes_to_frame(image_bytes: bytes) -> np.ndarray:
    """Convert image bytes to numpy array (BGR)."""
    nparr = np.frombuffer(image_bytes, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
