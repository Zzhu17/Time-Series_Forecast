from __future__ import annotations

import json
import logging
import sys
import time
from typing import Any, Dict, Optional


def setup_json_logger(level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger("ts-forecast")
    if logger.handlers:
        return logger
    logger.setLevel(level)
    handler = logging.StreamHandler(sys.stdout)

    class JsonFormatter(logging.Formatter):
        def format(self, record: logging.LogRecord) -> str:
            payload: Dict[str, Any] = {
                "level": record.levelname,
                "msg": record.getMessage(),
                "time": int(time.time() * 1000),
                "logger": record.name,
            }
            if record.__dict__.get("extra_fields"):
                try:
                    payload.update(record.__dict__["extra_fields"])
                except Exception:
                    pass
            return json.dumps(payload, ensure_ascii=False)

    handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def log_json(logger: logging.Logger, msg: str, **extra_fields: Any) -> None:
    logger.info(msg, extra={"extra_fields": extra_fields})
