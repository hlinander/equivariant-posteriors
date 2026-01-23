"""
Simple timestamped logging utility.
"""
from datetime import datetime


def log(tag: str, message: str):
    """Print a timestamped log message."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] [{tag}] {message}")


def log_next_in(tag: str, message: str, next_seconds: float):
    """Print a timestamped log message with countdown to next event."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] [{tag}] {message} (next in {next_seconds:.0f}s)")
