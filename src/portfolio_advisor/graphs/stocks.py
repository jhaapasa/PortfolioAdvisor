from __future__ import annotations

import datetime as dt
import logging
from pathlib import Path
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from ..config import Settings
from ..models.canonical import InstrumentKey
from ..services.polygon_client import PolygonClient
from ..stocks.analysis import (
    compute_sma_series,
    compute_trailing_returns,
    compute_volatility_annualized,
)
from ..stocks.wavelet import (
    compute_histograms,
    compute_modwt_logreturns,
    compute_variance_spectrum,
    to_coefficients_json,
    to_volatility_histogram_json,
)
from ..stocks.db import (
    StockPaths,
    append_ohlc_rows,
    ensure_ticker_scaffold,
    read_meta,
    read_primary_ohlc,
    utcnow_iso,
    write_meta,
    write_primary_ohlc,
)
from ..stocks.plotting import render_candlestick_ohlcv_1y
from ..utils.slug import instrument_id_to_slug


class StockState(TypedDict, total=False):
    settings: Any
    instrument: dict
    requested_artifacts: list[str]
    updates_needed: list[str]
    _slug: str
    _paths: Any


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
    today_utc = dt.datetime.now(dt.UTC).date().isoformat()
    # On fresh start (no rows), or if coverage is behind today, fetch primary
    if "primary.ohlc_daily" in requested and (not has_rows or coverage_end != today_utc):
        updates.append("primary.ohlc_daily")
    # Analysis artifacts depend on OHLC.
    # Even if primary is unchanged, allow recompute when requested.
    for art in ("analysis.returns", "analysis.volatility", "analysis.sma_20_50_100_200"):
        if art in requested:
            updates.append(art)
    # Allow wavelet analysis when requested via settings flag or explicit request
    want_wavelet = bool(getattr(settings, "wavelet", False)) or (
        "analysis.wavelet_modwt_j5_sym4" in requested
    )
    if want_wavelet:
        updates.append("analysis.wavelet_modwt_j5_sym4")

    _logger.info(
        "stocks.check_db_state: slug=%s requested=%s coverage_end=%s updates=%s",
        slug,
        ",".join(requested),
        coverage_end,
        ",".join(updates),
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

    client = PolygonClient(
        api_key=settings.polygon_api_key,
        trace=settings.verbose,
        timeout_s=settings.polygon_timeout_s,
    )

    # Determine date range: fetch full history if none, else from last coverage end + 1 day
    current = read_primary_ohlc(paths, slug)
    start_date = current.get("coverage", {}).get("end_date")
    if start_date:
        # next day after end_date
        start = str(dt.datetime.strptime(start_date, "%Y-%m-%d").date() + dt.timedelta(days=1))
    else:
        # default to a reasonable backfill horizon (5y) to avoid huge first-run fetches
        start = (dt.date.today() - dt.timedelta(days=5 * 365)).isoformat()
    today = dt.datetime.now(dt.UTC).date().isoformat()

    _logger.info(
        "stocks.fetch_primary: slug=%s ticker=%s range=%s..%s",
        slug,
        ticker,
        start,
        today,
    )
    new_rows = list(client.list_aggs_daily(ticker=ticker, from_date=start, to_date=today))
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
    return {}


def _compute_wavelet_node(state: StockState) -> dict:
    settings: Settings = state["settings"]
    instrument = state["instrument"]
    slug = state.get("_slug") or instrument_id_to_slug(str(instrument.get("instrument_id")))
    paths = state.get("_paths") or StockPaths(root=(Path(settings.output_dir) / "stocks"))
    requested = set(state.get("requested_artifacts") or [])
    if "analysis.wavelet_modwt_j5_sym4" not in requested and not bool(
        getattr(settings, "wavelet", False)
    ):
        return {}
    ohlc = read_primary_ohlc(paths, slug)
    rows = ohlc.get("data", []) or []
    dates = [r.get("date") for r in rows]
    closes = [float(r.get("close", 0.0)) for r in rows]

    # Compute SWT-based MODWT on full history, slice internally to last 2Y
    try:
        result = compute_modwt_logreturns(dates=dates, closes=closes, level=5, wavelet="sym4")
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

    # Update artifacts in meta
    try:
        meta_doc = read_meta(paths, slug)
        meta_doc.setdefault("artifacts", {}).setdefault("analysis.wavelet_modwt_j5_sym4", {})[
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
    graph.add_node("compute_returns", _compute_returns_node)
    graph.add_node("compute_volatility", _compute_volatility_node)
    graph.add_node("compute_sma", _compute_sma_node)
    graph.add_node("compute_wavelet", _compute_wavelet_node)
    graph.add_node("commit_metadata", _commit_metadata_node)
    graph.add_node("render_report", _render_report_node)

    graph.set_entry_point("resolve_ticker")
    graph.add_edge("resolve_ticker", "check_db_state")

    # Conditionally run primary fetch using conditional edges
    def _route_after_check(state: StockState):
        needs = state.get("updates_needed") or []
        return "fetch_primary" if "primary.ohlc_daily" in needs else "compute_returns"

    graph.add_conditional_edges("check_db_state", _route_after_check)
    graph.add_edge("fetch_primary", "compute_returns")
    # Always compute analysis when requested
    graph.add_edge("compute_returns", "compute_volatility")
    graph.add_edge("compute_volatility", "compute_sma")
    # Wavelet is optional; run after SMA (node will check requested_artifacts)
    graph.add_edge("compute_sma", "compute_wavelet")
    graph.add_edge("compute_wavelet", "render_report")
    graph.add_edge("render_report", "commit_metadata")
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
    with _futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(update_instrument, settings, inst, requested_artifacts)
            for inst in instruments
        ]
        for f in futures:
            try:
                f.result()
            except Exception:
                # Best-effort; errors are isolated per instrument, but log for visibility
                _logger.warning("stocks.update_all: instrument update failed", exc_info=True)
