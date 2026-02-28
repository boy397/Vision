"""Structured logging with per-stage latency tracking.

Set LOG_LEVEL env var to DEBUG for full pipeline visibility.
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import contextmanager
from typing import Generator


def setup_logging(level: str | None = None) -> None:
    """Configure structured logging.

    Reads LOG_LEVEL from environment if not specified.
    Default: DEBUG (to see all pipeline debug output).
    """
    if level is None:
        level = os.getenv("LOG_LEVEL", "DEBUG")

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.DEBUG),
        format="%(asctime)s │ %(levelname)-7s │ %(name)-30s │ %(message)s",
        datefmt="%H:%M:%S",
    )

    # Suppress noisy third-party loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("ultralytics").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)


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
        # Use INFO for stages >100ms (potential bottlenecks), DEBUG otherwise
        if elapsed > 100:
            log.info(f"⏱ {stage}: {elapsed:.1f}ms ⚠️ SLOW")
        else:
            log.info(f"⏱ {stage}: {elapsed:.1f}ms")
