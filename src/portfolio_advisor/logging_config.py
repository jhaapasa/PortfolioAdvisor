from __future__ import annotations

import json
import logging
import os
from datetime import UTC, datetime
import atexit
import threading
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
    log_libraries: bool = False,
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
    # Optionally suppress library logs at the handler level unless explicitly enabled
    if not log_libraries and not (agent_progress):
        noisy_prefixes = (
            "httpx",
            "urllib3",
            "openai",
            "langchain",
            "langgraph",
            "matplotlib",
        )

        class _LibraryFilter(logging.Filter):
            def filter(self, record: logging.LogRecord) -> bool:  # noqa: D401 (concise)
                name = record.name or ""
                # Always allow our own package
                if name.startswith("portfolio_advisor"):
                    return True
                # Always allow warnings/errors from any library
                if int(getattr(record, "levelno", logging.INFO)) >= int(logging.WARNING):
                    return True
                # Drop records from noisy libraries
                return not any(name.startswith(p) for p in noisy_prefixes)

        handler.addFilter(_LibraryFilter())

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

    # Reduce verbosity of noisy libraries by default, unless explicitly enabled
    for noisy in ("httpx", "urllib3", "openai", "langchain", "langgraph", "matplotlib"):
        if log_libraries:
            logging.getLogger(noisy).setLevel(level_value)
        elif noisy in {"langgraph", "langchain"} and agent_progress_enabled:
            logging.getLogger(noisy).setLevel(logging.INFO)
        else:
            logging.getLogger(noisy).setLevel(max(level_value, logging.WARNING))

    # Honor PYTHONWARNINGS to show/hide warnings if needed
    if os.environ.get("PYTHONWARNINGS"):
        logging.captureWarnings(True)

    # Register a best-effort graceful shutdown to run before interpreter teardown
    # to avoid late-finalization issues in Python 3.13 threading cleanup.
    def _graceful_shutdown() -> None:  # pragma: no cover - called at process exit
        try:
            # Ensure worker threads/executors are joined before objects get GC'd
            shutdown = getattr(threading, "_shutdown", None)
            if callable(shutdown):
                shutdown()
        except Exception:
            pass
        try:
            logging.shutdown()
        except Exception:
            pass

    # Ensure we only register once even if configure_logging is called multiple times
    if not getattr(configure_logging, "_pa_shutdown_registered", False):
        try:
            atexit.register(_graceful_shutdown)
            setattr(configure_logging, "_pa_shutdown_registered", True)
        except Exception:
            pass
