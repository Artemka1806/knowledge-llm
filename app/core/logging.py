import json
import logging
import os
import sys
from datetime import datetime, timezone

from .config import AppSettings


class JsonFormatter(logging.Formatter):
    """Simple JSON formatter that includes all record extras.

    Output fields:
    - time, level, logger, message
    - any extra keys passed via `extra={...}`
    """

    # keys provided by LogRecord we usually don't want to duplicate
    _exclude = {
        "name",
        "msg",
        "args",
        "levelname",
        "levelno",
        "pathname",
        "filename",
        "module",
        "exc_info",
        "exc_text",
        "stack_info",
        "lineno",
        "funcName",
        "created",
        "msecs",
        "relativeCreated",
        "thread",
        "threadName",
        "processName",
        "process",
    }

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "time": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for k, v in record.__dict__.items():
            if k in self._exclude:
                continue
            # Avoid overwriting basic keys
            if k in payload:
                k = f"_{k}"
            try:
                json.dumps(v)
                payload[k] = v
            except TypeError:
                payload[k] = str(v)
        return json.dumps(payload, ensure_ascii=False)


def setup_logging(settings: AppSettings | None = None) -> None:
    """Configure root logger with JSON output to stdout.

    LOG_LEVEL env var controls level (default INFO).
    Idempotent: will not add duplicate handlers if already configured.
    """
    level_name = os.getenv("LOG_LEVEL") or "INFO"
    level = getattr(logging, level_name.upper(), logging.INFO)

    root = logging.getLogger()
    root.setLevel(level)

    # Idempotent handler setup
    already = any(isinstance(h, logging.StreamHandler) for h in root.handlers)
    if not already:
        handler = logging.StreamHandler(stream=sys.stdout)
        handler.setLevel(level)
        handler.setFormatter(JsonFormatter())
        root.addHandler(handler)

    # Reduce noise from uvicorn/access if needed; keep defaults otherwise
    for noisy in ("uvicorn.error", "uvicorn.access"):
        logging.getLogger(noisy).setLevel(logging.INFO)
