"""
Simple verbosity-aware logger with optional timestamps and timed blocks.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from typing import Optional, TextIO


LEVEL_MAP = {"error": 0, "info": 1, "debug": 2}


@dataclass
class VerbosityLogger:
    """
    Emit log lines with configurable verbosity and optional timestamps.

    >>> logger = VerbosityLogger(level="info", timestamps=False)
    >>> logger.info("hello world")
    [INFO] hello world
    >>> logger.debug("hidden")
    >>> logger.error("boom")
    [ERROR] boom
    """

    level: str = "info"
    timestamps: bool = True
    log_file: Optional[str] = None
    enabled: bool = True

    def __post_init__(self) -> None:
        """
        Normalize level names and clamp invalid values.
        """

        self.level = self.level.lower()
        if self.level not in LEVEL_MAP:
            self.level = "info"
        self._level_value = LEVEL_MAP[self.level]
        self._file_handle: Optional[TextIO] = None
        if self.log_file:
            self._file_handle = open(self.log_file, "a", encoding="utf-8")

    def __del__(self) -> None:
        if self._file_handle:
            try:
                self._file_handle.close()
            except OSError:
                pass

    def _emit(self, text: str) -> None:
        print(text)
        if self._file_handle:
            self._file_handle.write(text + "\n")
            self._file_handle.flush()

    def log(self, level: str, message: str) -> None:
        """
        Emit a log line if the level is enabled.

        >>> logger = VerbosityLogger(level="debug", timestamps=False)
        >>> logger.log("debug", "detailed")
        [DEBUG] detailed
        """

        if not self.enabled:
            return
        level = level.lower()
        if level not in LEVEL_MAP:
            return
        if LEVEL_MAP[level] > self._level_value:
            return
        prefix = f"[{level.upper()}]"
        if self.timestamps:
            ts = time.strftime("%Y-%m-%d %H:%M:%S")
            self._emit(f"{ts} {prefix} {message}")
        else:
            self._emit(f"{prefix} {message}")

    def info(self, message: str) -> None:
        """Convenience wrapper for info-level logs."""

        self.log("info", message)

    def debug(self, message: str) -> None:
        """Convenience wrapper for debug-level logs."""

        self.log("debug", message)

    def error(self, message: str) -> None:
        """Convenience wrapper for error-level logs."""

        self.log("error", message)


class TimedBlock:
    """
    Context manager that logs start/end timestamps and duration.

    >>> logger = VerbosityLogger(level="info", timestamps=False)
    >>> with TimedBlock(logger, "demo block"):  # doctest: +ELLIPSIS
    ...     _ = sum(range(10))
    [INFO] demo block starting
    [INFO] demo block finished in ...
    """

    def __init__(self, logger: VerbosityLogger, label: str) -> None:
        self.logger = logger
        self.label = label
        self.start_time: Optional[float] = None

    def __enter__(self):
        self.start_time = time.perf_counter()
        self.logger.info(f"{self.label} starting")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time is None:
            return False
        duration = time.perf_counter() - self.start_time
        self.logger.info(f"{self.label} finished in {duration:.2f}s")
        return False
