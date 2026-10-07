"""Central logging configuration.

One place formats every log line so backend output stays greppable:

    2026-10-08 12:00:00 | INFO | app.api.routes.health | health check requested

Call ``configure_logging()`` once, at app startup (see main.py).
"""

from __future__ import annotations

import logging
import sys

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def configure_logging(level: str = "INFO") -> None:
    """Configure the root logger exactly once (idempotent)."""
    root = logging.getLogger()
    if root.handlers:  # already configured (e.g. under uvicorn reload workers)
        root.setLevel(level)
        return

    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(logging.Formatter(fmt=LOG_FORMAT, datefmt=DATE_FORMAT))

    root.addHandler(handler)
    root.setLevel(level)

    # Quiet noisy third-party loggers unless something goes wrong.
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced logger, typically ``get_logger(__name__)``."""
    return logging.getLogger(name)
