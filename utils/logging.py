"""Logging utilities for the Qwen3-TTS API with structured logging support.

This module provides logging utilities including structured JSON logging
for better observability and debugging in production environments.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime
from logging.handlers import TimedRotatingFileHandler
from typing import Any, Dict, Optional


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Create or retrieve a logger instance.

    Args:
        name: Optional logger name. Uses root logger if None.

    Returns:
        Configured logger instance.

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Processing started", extra={"job_id": "123"})
    """
    logger = logging.getLogger(name)

    stdout_handler: Optional[logging.Handler] = None
    for existing in logger.handlers:
        if (
            isinstance(existing, logging.StreamHandler)
            and getattr(existing, "stream", None) is sys.stdout
        ):
            stdout_handler = existing
            break

    if stdout_handler is None:
        stdout_handler = logging.StreamHandler(sys.stdout)
        logger.addHandler(stdout_handler)

    # Use structured logging from environment if configured
    from config.settings import get_settings

    try:
        settings = get_settings()
        use_structured = settings.structured_logging
        log_level = getattr(logging, settings.log_level.upper(), logging.INFO)
        log_to_file = bool(getattr(settings, "log_to_file", False))
        log_file_path = str(getattr(settings, "log_file_path", "") or "")
        log_retention_days = int(getattr(settings, "log_retention_days", 10))
    except Exception:
        use_structured = False
        log_level = logging.INFO
        log_to_file = False
        log_file_path = ""
        log_retention_days = 10

    if use_structured:
        formatter = StructuredFormatter()
    else:
        formatter = logging.Formatter(
            fmt="%(asctime)s %(levelname)s %(name)s %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    stdout_handler.setFormatter(formatter)

    if log_to_file and log_file_path:
        existing_paths = {
            getattr(h, "baseFilename", None)
            for h in logger.handlers
            if isinstance(h, (logging.FileHandler, TimedRotatingFileHandler))
        }
        if log_file_path not in existing_paths:
            try:
                os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
                file_handler = TimedRotatingFileHandler(
                    log_file_path,
                    when="midnight",
                    interval=1,
                    backupCount=max(1, log_retention_days),
                    utc=True,
                    encoding="utf-8",
                    delay=True,
                )
                file_handler.setFormatter(formatter)
                logger.addHandler(file_handler)
            except Exception:
                # Never break the service because file logging couldn't initialize.
                # stdout logging is still available via Docker logs.
                logger.warning("Failed to initialize file logging", exc_info=True)

    logger.setLevel(log_level)
    logger.propagate = False

    return logger


class StructuredFormatter(logging.Formatter):
    """JSON formatter for structured logging.

    Outputs logs as JSON objects with consistent schema for parsing
    by log aggregation tools.
    """

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON.

        Args:
            record: Log record to format.

        Returns:
            JSON string representation of the log record.
        """
        log_data: Dict[str, Any] = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        # Add extra fields from logger.info(..., extra={...})
        if hasattr(record, "__dict__"):
            for key, value in record.__dict__.items():
                if key not in [
                    "name",
                    "msg",
                    "args",
                    "created",
                    "filename",
                    "funcName",
                    "levelname",
                    "levelno",
                    "lineno",
                    "module",
                    "msecs",
                    "message",
                    "pathname",
                    "process",
                    "processName",
                    "relativeCreated",
                    "thread",
                    "threadName",
                    "exc_info",
                    "exc_text",
                    "stack_info",
                ]:
                    log_data[key] = value

        return json.dumps(log_data, default=str)


def status_log(component: str, event: str, **kwargs: Any) -> None:
    """Emit a structured status log for operational monitoring.

    This is a convenience function for emitting standardized status events
    that can be easily parsed by monitoring systems.

    Args:
        component: Component name (e.g., 'gpu_pool', 'api', 'worker').
        event: Event name (e.g., 'job_started', 'model_loaded').
        **kwargs: Additional context fields.

    Example:
        >>> status_log("gpu_pool", "job_assigned", job_id="123", gpu_id=0)
    """
    logger = get_logger("status")
    logger.info(f"{component}.{event}", extra={"component": component, "event": event, **kwargs})


def structured_error_log(
    job_id: str, correlation_id: str, error: Exception, extra: Optional[Dict[str, Any]] = None
) -> None:
    """Log an error with structured context for debugging.

    Args:
        job_id: Job identifier.
        correlation_id: Request correlation ID.
        error: Exception that occurred.
        extra: Additional context fields.

    Example:
        >>> try:
        ...     risky_operation()
        ... except Exception as e:
        ...     structured_error_log("job123", "corr456", e, {"phase": "tts"})
    """
    logger = get_logger("errors")
    context = {
        "job_id": job_id,
        "correlation_id": correlation_id,
        "error_type": type(error).__name__,
        "error_message": str(error),
    }
    if extra:
        context.update(extra)

    logger.error(f"Error in job {job_id}", extra=context, exc_info=error)


__all__ = [
    "get_logger",
    "status_log",
    "structured_error_log",
    "StructuredFormatter",
]
