from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .analyze import analyze_portfolio
from .config import Settings
from .graphs.stocks import update_ticker
from .logging_config import configure_logging


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="PortfolioAdvisor CLI")
    p.add_argument("--input-dir", required=True, help="Path to input directory")
    p.add_argument("--output-dir", required=True, help="Path to output directory")
    p.add_argument("--portfolio-dir", help="Optional portfolio state directory (defaults to <output_dir>/portfolio)")

    # Mode selection
    p.add_argument(
        "--mode",
        choices=["portfolio", "stock"],
        default="portfolio",
        help="Select 'portfolio' analysis or single 'stock' update",
    )
    p.add_argument("--ticker", help="Single stock ticker for --mode stock")

    # Env overrides
    p.add_argument("--openai-api-key")
    p.add_argument("--openai-base-url")
    p.add_argument("--openai-model")
    p.add_argument("--request-timeout-s", type=int)
    p.add_argument("--max-tokens", type=int)
    p.add_argument("--temperature", type=float)

    # Parser settings (override env)
    p.add_argument("--parser-max-retries", type=int)
    p.add_argument("--parser-max-doc-chars", type=int)

    # Logging
    p.add_argument("--log-level", default=None, help="Logging level (e.g., INFO, DEBUG)")
    p.add_argument("--log-format", default=None, choices=["plain", "json"], help="Log format")
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Enable DEBUG logs for the portfolio_advisor package",
    )
    p.add_argument(
        "--agent-progress",
        action="store_true",
        help="Show LangGraph agent progress messages",
    )
    p.add_argument(
        "--skip-llm-cache",
        action="store_true",
        help="Force LLM calls to bypass cache lookup but write results to cache",
    )

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    overrides = {k: v for k, v in vars(args).items() if v is not None}
    input_dir = overrides.pop("input_dir")
    output_dir = overrides.pop("output_dir")
    mode = overrides.pop("mode", "portfolio")
    ticker = overrides.pop("ticker", None)
    if mode == "portfolio":
        try:
            output_path = analyze_portfolio(input_dir=input_dir, output_dir=output_dir, **overrides)
        except Exception as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1
        print(output_path)
        return 0
    # mode == stock
    if not ticker:
        print("Error: --ticker is required when --mode stock", file=sys.stderr)
        return 1
    try:
        # Build settings and configure logging similar to portfolio mode
        settings = Settings(input_dir=input_dir, output_dir=output_dir, **overrides)
        settings.ensure_directories()
        configure_logging(
            level=settings.log_level,
            fmt=settings.log_format,
            verbose=bool(settings.verbose),
            agent_progress=bool(settings.agent_progress),
        )
        update_ticker(settings, ticker)
    except Exception as exc:  # pragma: no cover - network/provider specific
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    # Print the ticker directory path as output
    ticker_dir = Path(output_dir) / "stocks" / "tickers" / ticker
    print(str(ticker_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
