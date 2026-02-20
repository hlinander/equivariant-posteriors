"""
Logging utility using Python's logging module.

Default level is WARNING so background threads don't spam stdout.
Set EP_LOG_LEVEL=DEBUG or EP_LOG_LEVEL=INFO to see more.
"""
import logging
import os

_logger = logging.getLogger("ep")

if not _logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(asctime)s [%(name)s] %(message)s", datefmt="%H:%M:%S"))
    _logger.addHandler(_handler)

_level = os.environ.get("EP_LOG_LEVEL", "WARNING").upper()
_logger.setLevel(getattr(logging, _level, logging.WARNING))


def log(tag: str, message: str):
    """Log a message at INFO level."""
    _logger.info("[%s] %s", tag, message)


def log_next_in(tag: str, message: str, next_seconds: float):
    """Log a message with countdown to next event at INFO level."""
    _logger.info("[%s] %s (next in %.0fs)", tag, message, next_seconds)
