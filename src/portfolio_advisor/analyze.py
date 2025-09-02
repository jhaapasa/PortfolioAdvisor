from __future__ import annotations

import logging
from datetime import UTC, datetime

import os
from pathlib import Path

from .config import Settings
from .errors import ConfigurationError
from .graph import build_graph
from .io_utils import write_output_text
from .logging_config import configure_logging

logger = logging.getLogger(__name__)


def analyze_portfolio(
    input_dir: str,
    output_dir: str,
    **overrides: dict,
) -> str:
    """Analyze portfolio inputs and write a minimal report.

    Returns the path to the generated output file.
    """
    # Configure logging early using overrides if provided
    log_level = str(overrides.pop("log_level", overrides.pop("LOG_LEVEL", "INFO")))
    log_format = str(overrides.pop("log_format", overrides.pop("LOG_FORMAT", "plain")))
    verbose = bool(overrides.pop("verbose", overrides.pop("VERBOSE", False)))
    agent_progress = bool(overrides.pop("agent_progress", overrides.pop("AGENT_PROGRESS", False)))
    configure_logging(
        level=log_level,
        fmt=log_format,
        verbose=verbose,
        agent_progress=agent_progress,
    )

    # Build settings (env + overrides)
    try:
        settings = Settings(input_dir=input_dir, output_dir=output_dir, **overrides)
    except Exception as exc:
        raise ConfigurationError(str(exc)) from exc
    settings.ensure_directories()

    # Initialize global LangChain cache (SQLite) with optional read-bypass.
    try:
        from langchain_core.globals import set_llm_cache
        from langchain_community.cache import SQLiteCache

        cache_dir = Path("./cache")
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / "langchain_cache.sqlite3"

        base_cache = SQLiteCache(database_path=str(cache_path))

        if settings.skip_llm_cache:
            # Adapter that bypasses lookup but writes updates to the underlying cache.
            class _ReadBypassCache:
                def __init__(self, inner):
                    self._inner = inner

                def lookup(self, prompt: str, llm_string: str):  # type: ignore[override]
                    return None

                def update(self, prompt: str, llm_string: str, result):  # type: ignore[override]
                    return self._inner.update(prompt, llm_string, result)

            set_llm_cache(_ReadBypassCache(base_cache))
        else:
            set_llm_cache(base_cache)
    except Exception:  # pragma: no cover - cache setup best effort
        logger.warning("LLM cache setup failed; continuing without cache.")

    state = {
        "settings": settings,
        "requested_at": datetime.now(UTC).isoformat(),
    }

    app = build_graph()
    if agent_progress:
        # Stream state updates and log keys for simple progress visibility
        result = None
        try:
            for chunk in app.stream(state, stream_mode="values"):
                try:
                    logger.info("[agent] update: %s", ", ".join(sorted(chunk.keys())))
                except Exception:  # pragma: no cover - defensive
                    logger.info("[agent] update received")
                result = chunk
        except Exception:  # pragma: no cover - fallback if stream unsupported
            result = app.invoke(state)
    else:
        result = app.invoke(state)

    # Compose simple markdown output
    raw_docs = result.get("raw_docs", []) or []
    input_names = [str(doc.get("name", "")) for doc in raw_docs]
    lines = [
        "# Portfolio Analysis Report",
        f"Generated: {datetime.now(UTC).isoformat()}",
        "",
        "## Inputs",
        *(f"- {n}" for n in input_names),
        "",
        "## Plan",
        *(f"- {s}" for s in (result.get("plan", {}) or {}).get("steps", [])),
        "",
        "## Analysis",
        result.get("analysis", "No analysis produced."),
        "",
    ]

    output_path = write_output_text(settings.output_dir, "analysis.md", "\n".join(lines))
    return output_path
