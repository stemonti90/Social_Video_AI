"""Logging setup shared across the package."""
from __future__ import annotations

import logging
import sys
from pathlib import Path

_FMT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
_DATEFMT = "%H:%M:%S"


def get_logger(name: str = "avp") -> logging.Logger:
    return logging.getLogger(name)


def setup_logging(level: int = logging.INFO, logfile: Path | None = None) -> logging.Logger:
    """Configure the 'avp' logger. Idempotent — clears existing handlers."""
    root = logging.getLogger("avp")
    root.setLevel(level)
    root.handlers.clear()
    root.propagate = False

    stream = logging.StreamHandler(sys.stderr)
    stream.setFormatter(logging.Formatter(_FMT, _DATEFMT))
    root.addHandler(stream)

    if logfile is not None:
        logfile.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(logfile)
        fh.setFormatter(logging.Formatter(_FMT, _DATEFMT))
        root.addHandler(fh)

    return root
