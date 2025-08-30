from __future__ import annotations

import json
import logging
import os
from datetime import UTC, datetime
from typing import Any


class JsonLogFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:  # noqa: D401 (concise)
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, UTC).isoformat(),
            "level": record.levelname,
            "name": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def configure_logging(level: str = "INFO", fmt: str = "plain") -> None:
    level_value = getattr(logging, level.upper(), logging.INFO)
    root = logging.getLogger()
    root.setLevel(level_value)

    # Remove only handlers previously added by this configurator to avoid
    # interfering with external handlers (e.g., pytest's caplog).
    for existing in list(root.handlers):
        if getattr(existing, "_pa_handler", False):
            root.removeHandler(existing)

    handler = logging.StreamHandler()
    if fmt.lower() == "json":
        handler.setFormatter(JsonLogFormatter())
    else:
        formatter = logging.Formatter(
            fmt="%(asctime)s %(levelname)s [%(name)s] %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
        handler.setFormatter(formatter)
    # Mark this handler so we can safely replace it later without touching others
    setattr(handler, "_pa_handler", True)
    root.addHandler(handler)

    # Reduce verbosity of noisy libraries
    for noisy in ("httpx", "urllib3", "openai", "langchain", "langgraph"):
        logging.getLogger(noisy).setLevel(max(level_value, logging.WARNING))

    # Honor PYTHONWARNINGS to show/hide warnings if needed
    if os.environ.get("PYTHONWARNINGS"):
        logging.captureWarnings(True)
