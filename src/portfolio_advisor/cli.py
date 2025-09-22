from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .analyze import analyze_portfolio
from .config import Settings
from .graphs.stocks import update_instrument
from .logging_config import configure_logging
from .models.canonical import InstrumentKey


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="PortfolioAdvisor CLI")
    p.add_argument("--input-dir", required=True, help="Path to input directory")
    p.add_argument("--output-dir", required=True, help="Path to output directory")
    p.add_argument(
        "--portfolio-dir",
        help=("Optional portfolio state directory " "(defaults to <output_dir>/portfolio)"),
    )

    # Mode selection
    p.add_argument(
        "--mode",
        choices=["portfolio", "stock"],
        default="portfolio",
        help="Select 'portfolio' analysis or single 'stock' update",
    )
    p.add_argument("--ticker", help="Single stock ticker for --mode stock")
    p.add_argument("--instrument-id", help="Canonical instrument_id for --mode stock")
    p.add_argument(
        "--wavelet",
        action="store_true",
        help="Compute MODWT(SWT) J=5 Sym4 wavelet analysis on log-returns",
    )

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
        "--log-libraries",
        action="store_true",
        help=(
            "Allow library logs (httpx, urllib3, openai, langchain, langgraph) "
            "at configured level"
        ),
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
    instrument_id = overrides.pop("instrument_id", None)
    if mode == "portfolio":
        try:
            # Ensure wavelet flag is forwarded to Settings via overrides
            output_path = analyze_portfolio(input_dir=input_dir, output_dir=output_dir, **overrides)
        except Exception as exc:
            print(f"Error: {exc}", file=sys.stderr)
            try:
                import logging as _logging

                _logging.shutdown()
            except Exception:
                pass
            return 1
        print(output_path)
        try:
            import logging as _logging

            _logging.shutdown()
        except Exception:
            pass
        return 0
    # mode == stock
    if not (ticker or instrument_id):
        print("Error: --ticker or --instrument-id is required when --mode stock", file=sys.stderr)
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
            log_libraries=bool(getattr(settings, "log_libraries", False)),
        )
        # Prefer existing instrument_id/slug if a ticker dir already exists for this symbol
        iid = instrument_id
        if not iid and ticker:
            try:
                from .stocks.db import StockPaths

                paths = StockPaths(root=(Path(settings.output_dir) / "stocks"))
                tickers_root = paths.root / "tickers"
                if tickers_root.exists():
                    for candidate in tickers_root.iterdir():
                        if not candidate.is_dir():
                            continue
                        meta_path = candidate / "meta.json"
                        if not meta_path.exists():
                            continue
                        import json as _json

                        with meta_path.open("r", encoding="utf-8") as fh:
                            meta = _json.load(fh) or {}
                        if str(meta.get("primary_ticker") or "").upper() == str(ticker).upper():
                            cand_iid = meta.get("instrument_id")
                            if cand_iid:
                                iid = cand_iid
                                break
            except Exception:
                iid = None
        # Fallback to composite if still unknown
        iid = iid or (f"cid:stocks:us:composite:{ticker}" if ticker else None)
        symbol = None
        if iid and not ticker:
            try:
                symbol = InstrumentKey.parse(iid).symbol
            except Exception:
                symbol = None
        instrument = {
            "instrument_id": iid,
            "primary_ticker": ticker or symbol,
        }
        requested = None
        if bool(overrides.get("wavelet")):
            # Include standard analysis artifacts plus wavelet output
            requested = [
                "primary.ohlc_daily",
                "analysis.returns",
                "analysis.volatility",
                "analysis.sma_20_50_100_200",
                "analysis.wavelet_modwt_j5_sym4",
            ]
        update_instrument(settings, instrument, requested_artifacts=requested)
    except Exception as exc:  # pragma: no cover - network/provider specific
        print(f"Error: {exc}", file=sys.stderr)
        try:
            import logging as _logging

            _logging.shutdown()
        except Exception:
            pass
        return 1
    # Print the ticker directory path as output
    from .utils.slug import instrument_id_to_slug

    slug = instrument_id_to_slug(instrument.get("instrument_id"))
    ticker_dir = Path(output_dir) / "stocks" / "tickers" / slug
    print(str(ticker_dir))
    try:
        import logging as _logging

        _logging.shutdown()
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
