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
        choices=["portfolio", "stock", "extract-text"],
        default="portfolio",
        help=(
            "Select 'portfolio' analysis, single 'stock' update, "
            "or 'extract-text' for article text extraction"
        ),
    )
    p.add_argument("--ticker", help="Single stock ticker for --mode stock")
    p.add_argument("--instrument-id", help="Canonical instrument_id for --mode stock")
    p.add_argument(
        "--wavelet",
        action="store_true",
        help=(
            "Compute MODWT(SWT) Sym4 wavelet analysis. "
            "Use --wavelet-level to set decomposition level (default 5)."
        ),
    )
    p.add_argument(
        "-J",
        "--wavelet-level",
        type=int,
        help="Wavelet decomposition level J (1-8). Implies --wavelet when provided.",
    )
    p.add_argument(
        "--fetch-news/--no-fetch-news",
        dest="fetch_news",
        default=None,
        action=argparse.BooleanOptionalAction,
        help="Fetch news articles when updating stock data (default: True)",
    )
    p.add_argument(
        "--extract-text/--no-extract-text",
        dest="extract_text",
        default=None,
        action=argparse.BooleanOptionalAction,
        help="Extract text from HTML articles using Ollama (experimental, default: False)",
    )
    p.add_argument(
        "--force-extraction",
        action="store_true",
        help="Force re-extraction of text even if already extracted",
    )
    p.add_argument(
        "--all-stocks",
        action="store_true",
        help="Process all stocks (for extract-text mode)",
    )
    p.add_argument(
        "--enable-boundary-extension",
        action="store_true",
        help="Enable boundary stabilization for trend filters",
    )
    p.add_argument(
        "--boundary-strategy",
        choices=["linear", "gaussian_process"],
        default="linear",
        help="Forecasting strategy for boundary extension (default: linear)",
    )
    p.add_argument(
        "--boundary-steps",
        type=int,
        default=10,
        help="Number of steps to forecast for boundary extension (default: 10)",
    )
    p.add_argument(
        "--boundary-lookback",
        type=int,
        default=30,
        help="Lookback period for boundary extension model (default: 30)",
    )
    p.add_argument(
        "--boundary-noise-injection",
        action="store_true",
        help="Inject noise into the boundary extension to mimic historical volatility",
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
    p.add_argument(
        "--include-news",
        dest="include_news_report",
        action="store_true",
        help="Generate LLM-based 7d per-stock news+technical report",
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
    # If user provided a wavelet level, ensure wavelet analysis is enabled
    if "wavelet_level" in overrides:
        overrides["wavelet"] = True
    # Map enable_boundary_extension to boundary_extension
    if "enable_boundary_extension" in overrides:
        overrides["boundary_extension"] = overrides.pop("enable_boundary_extension")
    if "boundary_noise_injection" in overrides:
        # No remapping needed as Settings alias matches CLI arg name (mostly)
        # Settings alias is BOUNDARY_NOISE_INJECTION -> boundary_noise_injection
        # CLI arg is boundary_noise_injection. It's fine.
        pass
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

    if mode == "extract-text":
        # Handle text extraction mode
        all_stocks = overrides.pop("all_stocks", False)
        force_extraction = overrides.pop("force_extraction", False)
        if not all_stocks and not (ticker or instrument_id):
            print(
                "Error: --ticker, --instrument-id, or --all-stocks is required "
                "when --mode extract-text",
                file=sys.stderr,
            )
            return 1

        try:
            # Build settings and configure logging
            settings = Settings(input_dir=input_dir, output_dir=output_dir, **overrides)
            settings.ensure_directories()
            configure_logging(
                level=settings.log_level,
                fmt=settings.log_format,
                verbose=bool(settings.verbose),
                agent_progress=bool(settings.agent_progress),
                log_libraries=bool(getattr(settings, "log_libraries", False)),
            )

            from .services.ollama_service import OllamaService
            from .stocks.article_extraction import ArticleTextExtractionService
            from .stocks.db import StockPaths
            from .utils.slug import instrument_id_to_slug

            paths = StockPaths(root=(Path(settings.output_dir) / "stocks"))

            with OllamaService(
                base_url=settings.ollama_base_url, timeout_s=settings.ollama_timeout_s
            ) as ollama:
                extractor = ArticleTextExtractionService(
                    paths=paths, ollama_service=ollama, model=settings.extraction_model
                )

                if all_stocks:
                    # Extract for all stocks
                    stats = extractor.extract_portfolio_articles(
                        portfolio_path=Path(settings.output_dir),
                        force=force_extraction,
                        max_workers=settings.extraction_max_workers,
                    )
                    print(f"Processed {stats['processed_tickers']} tickers:")
                    print(f"  Articles: {stats['total_articles']}")
                    print(f"  Extracted: {stats['extracted']}")
                    print(f"  Skipped: {stats['skipped']}")
                    print(f"  Errors: {stats['errors']}")
                else:
                    # Extract for single ticker
                    if ticker and not instrument_id:
                        # Try to find existing instrument_id for this ticker
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
                                if (
                                    str(meta.get("primary_ticker") or "").upper()
                                    == str(ticker).upper()
                                ):
                                    instrument_id = meta.get("instrument_id")
                                    break

                    if not instrument_id:
                        instrument_id = f"stocks/us/{ticker}"

                    slug = instrument_id_to_slug(instrument_id)
                    stats = extractor.extract_all_articles(
                        ticker_slug=slug,
                        force=force_extraction,
                        batch_size=settings.extraction_batch_size,
                    )
                    print(f"Processed {ticker or instrument_id}:")
                    print(f"  Articles: {stats['total_articles']}")
                    print(f"  Extracted: {stats['extracted']}")
                    print(f"  Skipped: {stats['skipped']}")
                    print(f"  Errors: {stats['errors']}")

        except Exception as exc:
            print(f"Error: {exc}", file=sys.stderr)
            try:
                import logging as _logging

                _logging.shutdown()
            except Exception:
                pass
            return 1

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
