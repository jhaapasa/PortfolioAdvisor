from __future__ import annotations

import datetime as dt
import logging
from pathlib import Path
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from ..agents.news_summary import summarize_news_node
from ..agents.stock_report_collator import collate_report_node
from ..config import Settings
from ..models.canonical import InstrumentKey
from ..services.ollama_service import OllamaService
from ..services.polygon_client import PolygonClient
from ..stocks.analysis import (
    compute_sma_series,
    compute_trailing_returns,
    compute_volatility_annualized,
)
from ..stocks.article_extraction import ArticleTextExtractionService
from ..stocks.db import (
    StockPaths,
    append_ohlc_rows,
    compute_last_complete_trading_day,
    ensure_ticker_scaffold,
    read_meta,
    read_primary_ohlc,
    utcnow_iso,
    write_meta,
    write_primary_ohlc,
)
from ..stocks.news import StockNewsService
from ..stocks.plotting import (
    plot_wavelet_variance_spectrum,
    render_candlestick_ohlcv_1y,
    render_candlestick_ohlcv_2y_wavelet_trends,
)
from ..stocks.wavelet import (
    compute_histograms,
    compute_modwt_logprice,
    compute_modwt_logreturns,
    compute_variance_spectrum,
    reconstruct_logprice_series,
    to_coefficients_json,
    to_reconstructed_prices_json,
    to_volatility_histogram_json,
)
from ..utils.slug import instrument_id_to_slug


class StockState(TypedDict, total=False):
    settings: Any
    instrument: dict
    requested_artifacts: list[str]
    updates_needed: list[str]
    _slug: str
    _paths: Any
    news_summary: dict  # {markdown: str, json: dict | None}
    artifacts: dict  # {artifact_name: {paths, metadata}}


_logger = logging.getLogger(__name__)


def _resolve_ticker_node(state: StockState) -> dict:
    # Ensure primary_ticker is set; derive from instrument_id when missing
    instrument = state["instrument"]
    ticker = instrument.get("primary_ticker")
    if not ticker:
        iid = str(instrument.get("instrument_id") or "")
        try:
            symbol = InstrumentKey.parse(iid).symbol
        except Exception:
            symbol = None
        if symbol:
            updated = dict(instrument)
            updated["primary_ticker"] = symbol
            return {"instrument": updated}
    return {}


def _check_db_state_node(state: StockState) -> dict:
    settings = state["settings"]
    s: Settings = settings
    paths = StockPaths(root=(Path(s.output_dir) / "stocks"))
    instrument = state["instrument"]
    slug = instrument_id_to_slug(str(instrument.get("instrument_id")))

    ensure_ticker_scaffold(paths, slug)
    read_meta(paths, slug)
    requested = state.get("requested_artifacts") or [
        "primary.ohlc_daily",
        "analysis.returns",
        "analysis.volatility",
        "analysis.sma_20_50_100_200",
    ]

    updates: list[str] = []
    # Decide if primary needs update by comparing last coverage end with today
    ohlc = read_primary_ohlc(paths, slug)
    coverage_end = (ohlc.get("coverage") or {}).get("end_date")
    has_rows = bool(ohlc.get("data") or [])
    last_trading_day = compute_last_complete_trading_day()

    if "primary.ohlc_daily" in requested:
        needs_primary = not has_rows or coverage_end is None or coverage_end < last_trading_day
        if needs_primary:
            updates.append("primary.ohlc_daily")

    # Check if news needs updating (independent of OHLC)
    fetch_news = getattr(settings, "fetch_news", True)
    if fetch_news and has_rows:  # Only fetch news if we have stock data
        # Check if news index exists and when it was last updated
        news_index_path = paths.news_index_json(slug)
        needs_news_update = True
        if news_index_path.exists():
            try:
                import json

                with news_index_path.open("r") as f:
                    news_index = json.load(f)
                last_news_update = news_index.get("last_updated", "")
                if last_news_update:
                    # Only update if last update was more than 1 hour ago
                    from datetime import UTC, datetime, timedelta

                    last_update_time = datetime.fromisoformat(
                        last_news_update.replace("Z", "+00:00")
                    )
                    if datetime.now(UTC) - last_update_time < timedelta(hours=1):
                        needs_news_update = False
            except Exception:
                pass  # If anything fails, update news

        if needs_news_update:
            updates.append("primary.news")

    # Check if text extraction is explicitly requested
    extract_text = getattr(settings, "extract_text", False)
    if extract_text and has_rows and news_index_path.exists():
        # Only run extraction if explicitly enabled via --extract-text flag
        try:
            import json

            with news_index_path.open("r") as f:
                news_index = json.load(f)
            articles = news_index.get("articles", {})
            needs_extraction = False

            for article_info in articles.values():
                if article_info.get("has_full_content") and not article_info.get("text_extracted"):
                    needs_extraction = True
                    break

            if needs_extraction:
                updates.append("primary.text_extraction")
        except Exception:
            pass  # If anything fails, skip extraction

    # Analysis artifacts depend on OHLC.
    # Even if primary is unchanged, allow recompute when requested.
    for art in ("analysis.returns", "analysis.volatility", "analysis.sma_20_50_100_200"):
        if art in requested:
            updates.append(art)
    # Allow wavelet analysis when requested via settings flag, wavelet level, or explicit request
    want_wavelet = (
        bool(getattr(settings, "wavelet", False))
        or bool(getattr(settings, "wavelet_level", 0))
        or ("analysis.wavelet_modwt" in requested)
    )
    if want_wavelet:
        updates.append("analysis.wavelet_modwt")

    _logger.info(
        "stocks.check_db_state: slug=%s requested=%s coverage_end=%s updates=%s fetch_news=%s",
        slug,
        ",".join(requested),
        coverage_end,
        ",".join(updates),
        fetch_news,
    )
    return {"updates_needed": updates, "_slug": slug, "_paths": paths}


def _fetch_primary_node(state: StockState) -> dict:
    settings: Settings = state["settings"]
    instrument = state["instrument"]
    iid = str(instrument.get("instrument_id"))
    ticker_val = instrument.get("primary_ticker")
    if not ticker_val:
        try:
            ticker_val = InstrumentKey.parse(iid).symbol
        except Exception:
            ticker_val = ""
    ticker = str(ticker_val)
    slug = state.get("_slug") or instrument_id_to_slug(iid)
    paths = state.get("_paths") or StockPaths(root=(Path(settings.output_dir) / "stocks"))

    # Ensure HTTP resources are released promptly by using context manager
    with PolygonClient(
        api_key=settings.polygon_api_key,
        trace=settings.verbose,
        timeout_s=settings.polygon_timeout_s,
    ) as client:
        # Determine date range: fetch full history if none, else from last coverage end + 1 day
        current = read_primary_ohlc(paths, slug)
        start_date = current.get("coverage", {}).get("end_date")
        if start_date:
            # next day after end_date
            start = str(dt.datetime.strptime(start_date, "%Y-%m-%d").date() + dt.timedelta(days=1))
        else:
            # default to a reasonable backfill horizon (5y) to avoid huge first-run fetches
            start = (dt.date.today() - dt.timedelta(days=5 * 365)).isoformat()
        # Use last complete trading day as end to avoid requesting intraday/unsupported ranges
        end = compute_last_complete_trading_day()

        _logger.info(
            "stocks.fetch_primary: slug=%s ticker=%s range=%s..%s",
            slug,
            ticker,
            start,
            end,
        )
        # Check if we need to fetch OHLC data
        needs = state.get("updates_needed") or []
        if "primary.ohlc_daily" in needs:
            # If start is after end, nothing to fetch
            if start > end:
                new_rows = []
            else:
                new_rows = list(client.list_aggs_daily(ticker=ticker, from_date=start, to_date=end))
            if new_rows:
                merged = append_ohlc_rows(current, new_rows)
            else:
                merged = current
            merged["instrument_id"] = iid
            merged["primary_ticker"] = ticker
            write_primary_ohlc(paths, slug, merged)
            _logger.info(
                "stocks.fetch_primary: slug=%s wrote ohlc rows=%d coverage_end=%s",
                slug,
                len(merged.get("data", [])),
                (merged.get("coverage") or {}).get("end_date"),
            )
        else:
            # No OHLC update needed, just use current data
            merged = current

        # Fetch news if enabled
        fetch_news = getattr(settings, "fetch_news", True)
        _logger.info(
            "stocks.fetch_primary: news fetching check - fetch_news=%s ticker=%s",
            fetch_news,
            ticker,
        )
        if fetch_news and ticker:
            try:
                with StockNewsService(paths, client) as news_service:
                    news_stats = news_service.update_ticker_news(
                        ticker_slug=slug, ticker_symbol=ticker, days_back=7
                    )
                    _logger.info(
                        "stocks.fetch_primary: slug=%s news update stats=%s",
                        slug,
                        news_stats,
                    )
            except Exception:
                _logger.warning(
                    "stocks.fetch_primary: news fetching failed for %s", ticker, exc_info=True
                )

        return {}


def _extract_article_text_node(state: StockState) -> dict:
    """Extract text from downloaded articles using Ollama."""
    settings: Settings = state["settings"]
    instrument = state["instrument"]
    slug = state.get("_slug") or instrument_id_to_slug(str(instrument.get("instrument_id")))
    paths = state.get("_paths") or StockPaths(root=(Path(settings.output_dir) / "stocks"))
    ticker = str(instrument.get("primary_ticker", ""))

    # Check if extraction is needed
    needs = state.get("updates_needed") or []
    if "primary.text_extraction" not in needs:
        return {}

    # Check if ollama is configured
    if not hasattr(settings, "ollama_base_url"):
        _logger.warning("Ollama not configured, skipping text extraction")
        return {}

    try:
        # Initialize services
        with OllamaService(
            base_url=settings.ollama_base_url, timeout_s=settings.ollama_timeout_s
        ) as ollama:
            # Check if ollama is available
            if not ollama.is_available():
                _logger.warning("Ollama service not available, skipping text extraction")
                return {}

            # Check if model exists
            if not ollama.model_exists(settings.extraction_model):
                _logger.warning(
                    "Model %s not found. Please run: ollama pull %s",
                    settings.extraction_model,
                    settings.extraction_model,
                )
                return {}

            # Initialize extraction service
            extractor = ArticleTextExtractionService(
                paths=paths, ollama_service=ollama, model=settings.extraction_model
            )

            # Extract text from articles
            force = getattr(settings, "force_extraction", False)
            stats = extractor.extract_all_articles(
                ticker_slug=slug, force=force, batch_size=settings.extraction_batch_size
            )

            _logger.info(
                "stocks.extract_text: slug=%s ticker=%s extracted=%d skipped=%d errors=%d",
                slug,
                ticker,
                stats["extracted"],
                stats["skipped"],
                stats["errors"],
            )

    except Exception:
        _logger.warning("stocks.extract_text: extraction failed for %s", ticker, exc_info=True)

    return {}


def _compute_returns_node(state: StockState) -> dict:
    settings: Settings = state["settings"]
    instrument = state["instrument"]
    slug = state.get("_slug") or instrument_id_to_slug(str(instrument.get("instrument_id")))
    paths = state.get("_paths") or StockPaths(root=(Path(settings.output_dir) / "stocks"))
    ohlc = read_primary_ohlc(paths, slug)
    # Skip recompute if returns.json exists with matching as_of
    as_of = (ohlc.get("coverage") or {}).get("end_date")
    existing = paths.analysis_returns_json(slug)
    if existing.exists():
        try:
            import json as _json

            with existing.open("r", encoding="utf-8") as fh:
                prev = _json.load(fh)
            if prev.get("as_of") == as_of:
                return {}
        except Exception:
            pass
    returns = compute_trailing_returns(ohlc)
    from ..stocks.db import _write_json, utcnow_iso  # reuse private helper within package

    out = {**returns, "generated_at": utcnow_iso()}
    _write_json(paths.analysis_returns_json(slug), out)
    return {}


def _compute_volatility_node(state: StockState) -> dict:
    settings: Settings = state["settings"]
    instrument = state["instrument"]
    slug = state.get("_slug") or instrument_id_to_slug(str(instrument.get("instrument_id")))
    paths = state.get("_paths") or StockPaths(root=(Path(settings.output_dir) / "stocks"))
    ohlc = read_primary_ohlc(paths, slug)
    # Skip recompute if volatility.json exists with matching as_of and window
    as_of = (ohlc.get("coverage") or {}).get("end_date")
    existing = paths.analysis_volatility_json(slug)
    if existing.exists():
        try:
            import json as _json

            with existing.open("r", encoding="utf-8") as fh:
                prev = _json.load(fh)
            if prev.get("as_of") == as_of and int(prev.get("window") or 21) == 21:
                return {}
        except Exception:
            pass
    vol = compute_volatility_annualized(ohlc, window=21)
    from ..stocks.db import _write_json, utcnow_iso

    out = {**vol, "generated_at": utcnow_iso()}
    _write_json(paths.analysis_volatility_json(slug), out)
    return {}


def _compute_sma_node(state: StockState) -> dict:
    settings: Settings = state["settings"]
    instrument = state["instrument"]
    slug = state.get("_slug") or instrument_id_to_slug(str(instrument.get("instrument_id")))
    paths = state.get("_paths") or StockPaths(root=(Path(settings.output_dir) / "stocks"))
    ohlc = read_primary_ohlc(paths, slug)
    # Skip recompute if sma json exists and coverage end matches
    existing = paths.analysis_sma_json(slug)
    if existing.exists():
        try:
            import json as _json

            with existing.open("r", encoding="utf-8") as fh:
                prev = _json.load(fh)
            prev_cov = (prev.get("coverage") or {}).get("end_date")
            if prev_cov == (ohlc.get("coverage") or {}).get("end_date"):
                return {}
        except Exception:
            pass
    sma = compute_sma_series(ohlc)
    from ..stocks.db import _write_json, utcnow_iso

    out = {**sma, "generated_at": utcnow_iso()}
    _write_json(paths.analysis_sma_json(slug), out)
    return {}


def _render_report_node(state: StockState) -> dict:
    settings: Settings = state["settings"]
    instrument = state["instrument"]
    slug = state.get("_slug") or instrument_id_to_slug(str(instrument.get("instrument_id")))
    paths = state.get("_paths") or StockPaths(root=(Path(settings.output_dir) / "stocks"))
    ohlc = read_primary_ohlc(paths, slug)

    # Render 1Y candlestick into report folder
    ticker_dir = paths.ticker_dir(slug)
    try:
        written = render_candlestick_ohlcv_1y(ticker_dir, ohlc)
        if written is not None:
            _logger.info("stocks.render_report: wrote %s", written)
    except Exception:
        _logger.warning("stocks.render_report: rendering failed", exc_info=True)

    # Render wavelet variance spectrum bar chart when histogram exists
    try:
        hist_path = paths.analysis_wavelet_hist_json(slug)
        if hist_path.exists():
            import json as _json

            with hist_path.open("r", encoding="utf-8") as fh:
                hist_doc = _json.load(fh)
            spectrum = hist_doc.get("variance_spectrum") or {}
            per_level = spectrum.get("per_level") or {}
            # Normalize to percentages
            from ..stocks.wavelet import normalize_variance_spectrum

            normalized = normalize_variance_spectrum(per_level)
            title = "Wavelet Variance Spectrum (Normalized)"
            as_of = (ohlc.get("coverage") or {}).get("end_date")
            subtitle = f"{ohlc.get('primary_ticker') or ''} — as of {as_of}" if as_of else None
            out = plot_wavelet_variance_spectrum(
                ticker_dir, normalized, title=title, subtitle=subtitle
            )
            if out is not None:
                _logger.info("stocks.render_report: wrote %s", out)
    except Exception:
        _logger.warning("stocks.render_report: wavelet spectrum rendering failed", exc_info=True)

    # Render 2Y candlestick with wavelet trend overlays (if reconstructions exist)
    try:
        import json as _json

        recon_doc = None
        recon_path = paths.analysis_wavelet_reconstructed_prices_json(slug)
        if recon_path.exists():
            with recon_path.open("r", encoding="utf-8") as fh:
                recon_doc = _json.load(fh)
        out2 = render_candlestick_ohlcv_2y_wavelet_trends(ticker_dir, ohlc, recon_doc)
        if out2 is not None:
            _logger.info("stocks.render_report: wrote %s", out2)
    except Exception:
        _logger.warning("stocks.render_report: 2y wavelet overlay rendering failed", exc_info=True)
    return {}


def _compute_wavelet_node(state: StockState) -> dict:
    settings: Settings = state["settings"]
    instrument = state["instrument"]
    slug = state.get("_slug") or instrument_id_to_slug(str(instrument.get("instrument_id")))
    paths = state.get("_paths") or StockPaths(root=(Path(settings.output_dir) / "stocks"))
    requested = set(state.get("requested_artifacts") or [])
    if "analysis.wavelet_modwt" not in requested and not bool(getattr(settings, "wavelet", False)):
        return {}
    ohlc = read_primary_ohlc(paths, slug)
    rows = ohlc.get("data", []) or []
    dates = [r.get("date") for r in rows]
    closes = [float(r.get("close", 0.0)) for r in rows]

    # Compute SWT-based MODWT on full history, slice internally to last 2Y
    try:
        lvl = int(getattr(settings, "wavelet_level", 5))
        result = compute_modwt_logreturns(dates=dates, closes=closes, level=lvl, wavelet="sym4")
    except Exception:
        # Best-effort; do not fail the whole pipeline
        _logger.warning("stocks.wavelet: transform failed", exc_info=True)
        return {}

    # Baseline returns aligned to result.dates for energy check
    import numpy as _np

    def _compute_lr(_closes: list[float]) -> _np.ndarray:
        arr = _np.asarray(_closes, dtype=float)
        lr = _np.diff(_np.log(arr))
        return lr

    all_lr = _compute_lr(closes)
    # Align by dates: result dates correspond to closes[1:] dates; select the last target_len
    target_len = len(result.dates)
    baseline_returns = all_lr[-target_len:]
    spectrum = compute_variance_spectrum(result, baseline_returns)
    histos = compute_histograms(result.details, bins=50)

    # Enrich metadata
    analysis_start = result.dates[0] if result.dates else None
    analysis_end = result.dates[-1] if result.dates else None
    meta = {
        **result.meta,
        "ticker": ohlc.get("primary_ticker"),
        "instrument_id": ohlc.get("instrument_id"),
        "analysis_start": analysis_start,
        "analysis_end": analysis_end,
    }

    # Write outputs
    from ..stocks.db import _write_json, utcnow_iso  # reuse internal writer

    coeffs_doc = to_coefficients_json(ticker=slug, result=result)
    coeffs_doc["metadata"].update({"analysis_start": analysis_start, "analysis_end": analysis_end})
    coeffs_doc["generated_at"] = utcnow_iso()
    _write_json(paths.analysis_wavelet_coeffs_json(slug), coeffs_doc)

    hist_doc = to_volatility_histogram_json(
        ticker=slug, spectrum=spectrum, histos=histos, meta=meta
    )
    hist_doc["generated_at"] = utcnow_iso()
    _write_json(paths.analysis_wavelet_hist_json(slug), hist_doc)

    # Also compute SWT on log price and reconstruct price series
    try:
        lvl = int(getattr(settings, "wavelet_level", 5))
        price_result = compute_modwt_logprice(dates=dates, closes=closes, level=lvl, wavelet="sym4")
        price_coeffs_doc = to_coefficients_json(ticker=slug, result=price_result)
        price_coeffs_doc["metadata"].update(
            {"analysis_start": analysis_start, "analysis_end": analysis_end}
        )
        price_coeffs_doc["generated_at"] = utcnow_iso()
        _write_json(paths.analysis_wavelet_coeffs_logprice_json(slug), price_coeffs_doc)

        recon_dates, recon_map, recon_meta = reconstruct_logprice_series(
            dates=dates,
            closes=closes,
            level=int(getattr(settings, "wavelet_level", 5)),
            wavelet="sym4",
            max_level=6,  # Use consistent max level for MRA consistency
        )
        recon_doc = to_reconstructed_prices_json(
            ticker=slug, dates=recon_dates, recon=recon_map, meta=recon_meta
        )
        recon_doc["generated_at"] = utcnow_iso()
        _write_json(paths.analysis_wavelet_reconstructed_prices_json(slug), recon_doc)
    except Exception:
        _logger.warning("stocks.wavelet: log-price transform/reconstruction failed", exc_info=True)

    # Update artifacts in meta
    try:
        meta_doc = read_meta(paths, slug)
        meta_doc.setdefault("artifacts", {}).setdefault("analysis.wavelet_modwt", {})[
            "last_updated"
        ] = utcnow_iso()
        write_meta(paths, slug, meta_doc)
    except Exception:
        _logger.warning("stocks.wavelet: failed to update meta", exc_info=True)

    return {}


def _commit_metadata_node(state: StockState) -> dict:
    settings: Settings = state["settings"]
    instrument = state["instrument"]
    iid = str(instrument.get("instrument_id"))
    slug = instrument_id_to_slug(iid)
    paths = StockPaths(root=(Path(settings.output_dir) / "stocks"))
    meta = read_meta(paths, slug)
    # Set last complete trading day to primary coverage end date when available
    ohlc = read_primary_ohlc(paths, slug)
    end_date = (ohlc.get("coverage") or {}).get("end_date")
    if end_date:
        meta["last_complete_trading_day"] = end_date
    meta["instrument_id"] = iid
    # Prefer ticker recorded in OHLC; fallback to instrument and then parse from instrument_id
    ticker_val = ohlc.get("primary_ticker") or instrument.get("primary_ticker")
    if not ticker_val:
        try:
            ticker_val = InstrumentKey.parse(iid).symbol
        except Exception:
            ticker_val = None
    if ticker_val is not None:
        meta["primary_ticker"] = str(ticker_val)
    meta["slug"] = slug
    meta.setdefault("artifacts", {})
    for art in (
        "primary.ohlc_daily",
        "analysis.returns",
        "analysis.volatility",
        "analysis.sma_20_50_100_200",
    ):
        meta["artifacts"].setdefault(art, {})["last_updated"] = utcnow_iso()
    write_meta(paths, slug, meta)
    return {}


def build_stocks_graph() -> Any:
    graph = StateGraph(StockState)
    graph.add_node("resolve_ticker", _resolve_ticker_node)
    graph.add_node("check_db_state", _check_db_state_node)
    graph.add_node("fetch_primary", _fetch_primary_node)
    graph.add_node("extract_article_text", _extract_article_text_node)
    graph.add_node("compute_returns", _compute_returns_node)
    graph.add_node("compute_volatility", _compute_volatility_node)
    graph.add_node("compute_sma", _compute_sma_node)
    graph.add_node("compute_wavelet", _compute_wavelet_node)
    graph.add_node("commit_metadata", _commit_metadata_node)
    graph.add_node("render_report", _render_report_node)
    # New LLM nodes for news summarization and final 7d report
    graph.add_node("summarize_news", summarize_news_node)
    graph.add_node("collate_report", collate_report_node)

    graph.set_entry_point("resolve_ticker")
    graph.add_edge("resolve_ticker", "check_db_state")

    # Conditionally run primary fetch using conditional edges
    def _route_after_check(state: StockState):
        needs = state.get("updates_needed") or []
        # Go to fetch_primary if either OHLC or news needs updating
        if "primary.ohlc_daily" in needs or "primary.news" in needs:
            return "fetch_primary"
        # Go to text extraction if needed
        elif "primary.text_extraction" in needs:
            return "extract_article_text"
        else:
            return "compute_returns"

    graph.add_conditional_edges("check_db_state", _route_after_check)

    # Route after fetch_primary
    def _route_after_fetch(state: StockState):
        needs = state.get("updates_needed") or []
        # Check if text extraction is needed after fetching news
        if "primary.text_extraction" in needs:
            return "extract_article_text"
        else:
            return "compute_returns"

    graph.add_conditional_edges("fetch_primary", _route_after_fetch)
    graph.add_edge("extract_article_text", "compute_returns")
    # Always compute analysis when requested
    graph.add_edge("compute_returns", "compute_volatility")
    graph.add_edge("compute_volatility", "compute_sma")
    # Wavelet is optional; run after SMA (node will check requested_artifacts)
    graph.add_edge("compute_sma", "compute_wavelet")
    graph.add_edge("compute_wavelet", "render_report")

    # Insert news summarization and report collation before commit metadata
    def _route_after_render(state: StockState):
        settings = state.get("settings")
        include = bool(getattr(settings, "include_news_report", False))
        _logger.info(
            "stocks.route_after_render: include_news_report=%s -> %s",
            include,
            "summarize_news" if include else "commit_metadata",
        )
        return "summarize_news" if include else "commit_metadata"

    graph.add_conditional_edges("render_report", _route_after_render)
    graph.add_edge("summarize_news", "collate_report")
    graph.add_edge("collate_report", "commit_metadata")
    graph.add_edge("commit_metadata", END)
    return graph.compile()


def update_instrument(
    settings: Settings, instrument: dict, requested_artifacts: list[str] | None = None
) -> None:
    compiled = build_stocks_graph()
    state: StockState = {"settings": settings, "instrument": instrument}
    if requested_artifacts:
        state["requested_artifacts"] = requested_artifacts
    try:
        compiled.invoke(state)
    except Exception:
        _logger.warning("stocks.update_instrument: invoke failed", exc_info=True)


def update_all_for_instruments(
    settings: Settings, instruments: list[dict], requested_artifacts: list[str] | None = None
) -> None:
    # Cap concurrency to a small number to reduce IO/API contention
    import concurrent.futures as _futures

    max_workers = 4
    # Explicitly request safe cancellation of pending futures on context exit
    with _futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(update_instrument, settings, inst, requested_artifacts)
            for inst in instruments
        ]
        try:
            for f in futures:
                try:
                    f.result()
                except Exception:
                    # Best-effort; errors are isolated per instrument, but log for visibility
                    _logger.warning("stocks.update_all: instrument update failed", exc_info=True)
        finally:
            # Python 3.9+ ThreadPoolExecutor supports shutdown(cancel_futures=True)
            try:
                executor.shutdown(wait=True, cancel_futures=True)  # type: ignore[arg-type]
            except TypeError:
                # Older versions without cancel_futures
                executor.shutdown(wait=True)
