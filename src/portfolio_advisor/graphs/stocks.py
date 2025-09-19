from __future__ import annotations

import datetime as dt
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
from ..utils.slug import instrument_id_to_slug


class StockState(TypedDict, total=False):
    settings: Any
    instrument: dict
    requested_artifacts: list[str]
    updates_needed: list[str]
    _slug: str
    _paths: Any


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
    today_utc = dt.datetime.now(dt.UTC).date().isoformat()
    if "primary.ohlc_daily" in requested and coverage_end != today_utc:
        updates.append("primary.ohlc_daily")
    # Analysis artifacts depend on OHLC.
    # Even if primary is unchanged, allow recompute when requested.
    for art in ("analysis.returns", "analysis.volatility", "analysis.sma_20_50_100_200"):
        if art in requested:
            updates.append(art)

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

    new_rows = list(client.list_aggs_daily(ticker=ticker, from_date=start, to_date=today))
    if new_rows:
        merged = append_ohlc_rows(current, new_rows)
    else:
        merged = current
    merged["instrument_id"] = iid
    merged["primary_ticker"] = ticker
    write_primary_ohlc(paths, slug, merged)
    return {}


def _compute_returns_node(state: StockState) -> dict:
    settings: Settings = state["settings"]
    instrument = state["instrument"]
    slug = state.get("_slug") or instrument_id_to_slug(str(instrument.get("instrument_id")))
    paths = state.get("_paths") or StockPaths(root=(Path(settings.output_dir) / "stocks"))
    ohlc = read_primary_ohlc(paths, slug)
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
    sma = compute_sma_series(ohlc)
    from ..stocks.db import _write_json, utcnow_iso

    out = {**sma, "generated_at": utcnow_iso()}
    _write_json(paths.analysis_sma_json(slug), out)
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
    graph.add_node("commit_metadata", _commit_metadata_node)

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
    graph.add_edge("compute_sma", "commit_metadata")
    graph.add_edge("commit_metadata", END)
    return graph.compile()


def update_instrument(
    settings: Settings, instrument: dict, requested_artifacts: list[str] | None = None
) -> None:
    compiled = build_stocks_graph()
    state: StockState = {"settings": settings, "instrument": instrument}
    if requested_artifacts:
        state["requested_artifacts"] = requested_artifacts
    compiled.invoke(state)


def update_all_for_instruments(
    settings: Settings, instruments: list[dict], requested_artifacts: list[str] | None = None
) -> None:
    for inst in instruments:
        update_instrument(settings, inst, requested_artifacts=requested_artifacts)
