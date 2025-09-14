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


def configure_logging(
    level: str = "INFO",
    fmt: str = "plain",
    verbose: bool = False,
    agent_progress: bool = False,
) -> None:
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

    # Parse env toggles in case Settings/CLI aren't available yet
    def _truthy(value: str | None) -> bool:
        if not value:
            return False
        return value.strip().lower() in {"1", "true", "yes", "on"}

    verbose_enabled = verbose or _truthy(os.environ.get("VERBOSE"))
    agent_progress_enabled = agent_progress or _truthy(os.environ.get("AGENT_PROGRESS"))

    # Always allow our package logs to be verbose when requested
    if verbose_enabled:
        logging.getLogger("portfolio_advisor").setLevel(logging.DEBUG)

    # Reduce verbosity of noisy libraries by default, with an exception:
    # when agent_progress is enabled, allow LangGraph/LangChain INFO logs through.
    for noisy in ("httpx", "urllib3", "openai", "langchain", "langgraph"):
        if noisy in {"langgraph", "langchain"} and agent_progress_enabled:
            logging.getLogger(noisy).setLevel(logging.INFO)
        else:
            logging.getLogger(noisy).setLevel(max(level_value, logging.WARNING))

    # Honor PYTHONWARNINGS to show/hide warnings if needed
    if os.environ.get("PYTHONWARNINGS"):
        logging.captureWarnings(True)
