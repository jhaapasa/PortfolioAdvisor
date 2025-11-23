"""Application entrypoint for running the portfolio analysis graph.

Builds `Settings` from environment + CLI overrides, configures logging, initializes
the LangChain cache, and executes the LangGraph app to produce outputs.
"""

from __future__ import annotations

import logging
import os
from datetime import UTC, datetime
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
    # Build settings (env + overrides)
    try:
        settings = Settings(input_dir=input_dir, output_dir=output_dir, **overrides)
    except Exception as exc:
        raise ConfigurationError(str(exc)) from exc
    settings.ensure_directories()

    # Configure logging using Settings (Settings/CLI override env defaults)
    configure_logging(
        level=settings.log_level,
        fmt=settings.log_format,
        verbose=bool(settings.verbose),
        agent_progress=bool(settings.agent_progress),
        log_libraries=bool(getattr(settings, "log_libraries", False)),
    )

    # Initialize global LangChain cache (SQLite) with optional read-bypass.
    try:
        from langchain_community.cache import SQLiteCache
        from langchain_core.globals import set_llm_cache

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
    if settings.agent_progress:
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
    resolved = result.get("resolved_holdings", []) or []
    unresolved = result.get("unresolved_entities", []) or []

    lines = [
        "# Portfolio Analysis Report",
        f"Generated: {datetime.now(UTC).isoformat()}",
        "",
        "## Inputs",
        *(f"- {n}" for n in input_names),
        "",
        "## Resolver Summary",
        f"Resolved holdings: {len(resolved)}",
        f"Unresolved entities: {len(unresolved)}",
        "",
        "## Plan",
        *(f"- {s}" for s in (result.get("plan", {}) or {}).get("steps", [])),
        "",
    ]

    # Add market comparison section if available
    market_context = result.get("market_context")
    if market_context and market_context.portfolio_metrics:
        pm = market_context.portfolio_metrics
        lines.extend(
            [
                "## Market Comparison",
                "",
                "### Portfolio Risk Metrics",
            ]
        )

        # Average betas
        if pm.average_beta_vs_benchmarks:
            for benchmark, beta in pm.average_beta_vs_benchmarks.items():
                lines.append(f"- Average Beta vs {benchmark}: {beta:.2f}")

        # Sharpe ratios
        lines.extend(
            [
                f"- Portfolio Sharpe Ratio (1yr): {pm.portfolio_sharpe:.2f}",
                f"- Average Stock Sharpe Ratio: {pm.average_stock_sharpe:.2f}",
                "",
                "### Benchmark Performance",
            ]
        )

        # Stocks outperforming
        if pm.stocks_outperforming:
            for benchmark, count in pm.stocks_outperforming.items():
                pct = (count / pm.total_stocks * 100) if pm.total_stocks > 0 else 0
                lines.append(
                    f"- Stocks outperforming {benchmark}: {count} of {pm.total_stocks} ({pct:.0f}%)"
                )

        # Top contributors
        if pm.top_contributors:
            lines.extend(
                [
                    "",
                    "### Top Contributors to Outperformance (vs SPY)",
                ]
            )
            for i, contrib in enumerate(pm.top_contributors[:5], 1):
                ticker = contrib["ticker"]
                excess_sharpe = contrib.get("excess_sharpe", 0)
                excess_return = contrib.get("excess_return")
                if excess_return is not None:
                    lines.append(
                        f"{i}. **{ticker}**: Excess Sharpe {excess_sharpe:+.2f}, "
                        f"Excess Return {excess_return:+.1%}"
                    )
                else:
                    lines.append(f"{i}. **{ticker}**: Excess Sharpe {excess_sharpe:+.2f}")

        lines.append("")

    lines.extend(
        [
            "## Analysis",
            result.get("analysis", "No analysis produced."),
            "",
        ]
    )

    output_path = write_output_text(settings.output_dir, "analysis.md", "\n".join(lines))
    # Write resolved positions JSON for debugging/visibility
    try:
        import json as _json

        resolved_json_path = os.path.join(settings.output_dir, "resolved_positions.json")
        with open(resolved_json_path, "w", encoding="utf-8") as fh:
            _json.dump(
                {
                    "resolved_holdings": resolved,
                    "unresolved_entities": unresolved,
                },
                fh,
                ensure_ascii=False,
                indent=2,
            )
        logger.info("Wrote output: %s", resolved_json_path)
    except Exception as exc:  # pragma: no cover - best-effort write
        logger.warning("Failed to write resolved_positions.json: %s", exc)
    return output_path
