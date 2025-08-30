from __future__ import annotations

import logging
from datetime import UTC, datetime

from .config import Settings
from .errors import ConfigurationError
from .graph import build_graph
from .io_utils import list_input_files, read_files_preview, write_output_text
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
    configure_logging(level=log_level, fmt=log_format)

    # Build settings (env + overrides)
    try:
        settings = Settings(input_dir=input_dir, output_dir=output_dir, **overrides)
    except Exception as exc:
        raise ConfigurationError(str(exc)) from exc
    settings.ensure_directories()

    files = list_input_files(settings.input_dir)
    previews = read_files_preview(files)

    state = {
        "settings": settings,
        "files": files,
        "previews": previews,
        "requested_at": datetime.now(UTC).isoformat(),
    }

    app = build_graph()
    result = app.invoke(state)

    # Compose simple markdown output
    lines = [
        "# Portfolio Analysis Report",
        f"Generated: {datetime.now(UTC).isoformat()}",
        "",
        "## Inputs",
        *(f"- {p.name}" for p in files),
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
