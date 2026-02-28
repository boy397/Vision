"""Structured logging with per-stage latency tracking."""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from typing import Generator


def setup_logging(level: str = "INFO") -> None:
    """Configure structured logging."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s │ %(levelname)-7s │ %(name)-25s │ %(message)s",
        datefmt="%H:%M:%S",
    )


@contextmanager
def latency_tracker(stage: str, logger_instance: logging.Logger | None = None) -> Generator[dict, None, None]:
    """Context manager to measure and log latency for a pipeline stage.

    Usage:
        with latency_tracker("yolo_inference", logger) as metrics:
            result = model.predict(frame)
        # metrics["elapsed_ms"] now has the duration
    """
    log = logger_instance or logging.getLogger("latency")
    metrics: dict = {"stage": stage, "elapsed_ms": 0.0}
    start = time.perf_counter()
    try:
        yield metrics
    finally:
        elapsed = (time.perf_counter() - start) * 1000
        metrics["elapsed_ms"] = round(elapsed, 2)
        log.info(f"⏱ {stage}: {elapsed:.1f}ms")
