from __future__ import annotations

import contextlib
import datetime as dt
import json
import os
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any


# Minimal single-process file lock using directory creation. This avoids new deps.
@contextlib.contextmanager
def file_lock(lock_path: Path, timeout_s: int = 10) -> Iterator[None]:  # pragma: no cover - simple
    import time

    deadline = time.time() + timeout_s
    while True:
        try:
            os.mkdir(lock_path)
            break
        except FileExistsError:
            if time.time() > deadline:
                raise TimeoutError(f"Timed out acquiring lock: {lock_path}")
            time.sleep(0.05)
    try:
        yield
    finally:
        with contextlib.suppress(Exception):
            os.rmdir(lock_path)


@dataclass
class StockPaths:
    root: Path

    def ticker_dir(self, ticker: str) -> Path:
        # Parameter represents the folder slug for the instrument (instrument_id slug)
        return self.root / "tickers" / ticker

    def meta_json(self, ticker: str) -> Path:
        return self.ticker_dir(ticker) / "meta.json"

    def primary_ohlc_json(self, ticker: str) -> Path:
        return self.ticker_dir(ticker) / "primary" / "ohlc_daily.json"

    def analysis_returns_json(self, ticker: str) -> Path:
        return self.ticker_dir(ticker) / "analysis" / "returns.json"

    def analysis_volatility_json(self, ticker: str) -> Path:
        return self.ticker_dir(ticker) / "analysis" / "volatility.json"

    def analysis_sma_json(self, ticker: str) -> Path:
        return self.ticker_dir(ticker) / "analysis" / "sma_20_50_100_200.json"

    def lock_dir(self, ticker: str) -> Path:
        return self.ticker_dir(ticker) / ".lock"


def ensure_ticker_scaffold(paths: StockPaths, ticker: str) -> None:
    base = paths.ticker_dir(ticker)
    (base / "primary").mkdir(parents=True, exist_ok=True)
    (base / "analysis").mkdir(parents=True, exist_ok=True)


def _read_json(path: Path) -> dict[str, Any] | list[dict[str, Any]]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def utcnow_iso() -> str:
    return dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def read_meta(paths: StockPaths, slug: str) -> dict[str, Any]:
    meta_path = paths.meta_json(slug)
    data = _read_json(meta_path) or {}
    if not data:
        data = {
            "slug": slug,
            "instrument_id": None,
            "primary_ticker": None,
            "last_complete_trading_day": None,
            "artifacts": {},
        }
    return data


def write_meta(paths: StockPaths, slug: str, meta: dict[str, Any]) -> None:
    _write_json(paths.meta_json(slug), meta)


def read_primary_ohlc(paths: StockPaths, slug: str) -> dict[str, Any]:
    data = _read_json(paths.primary_ohlc_json(slug)) or {}
    if not data:
        data = {
            "instrument_id": None,
            "primary_ticker": None,
            "source": "polygon.io",
            "price_currency": "USD",
            "fields": ["date", "open", "high", "low", "close", "volume", "vwap"],
            "data": [],
            "coverage": {"start_date": None, "end_date": None},
            "generated_at": utcnow_iso(),
        }
    return data


def write_primary_ohlc(paths: StockPaths, slug: str, ohlc: dict[str, Any]) -> None:
    ohlc["generated_at"] = utcnow_iso()
    _write_json(paths.primary_ohlc_json(slug), ohlc)


def append_ohlc_rows(existing: dict[str, Any], new_rows: list[dict[str, Any]]) -> dict[str, Any]:
    # Merge by date, prefer newer rows for overlapping dates
    by_date: dict[str, dict[str, Any]] = {
        r["date"]: r for r in existing.get("data", []) if r.get("date")
    }
    for r in new_rows:
        d = r.get("date")
        if not d:
            continue
        by_date[d] = {
            "date": d,
            "open": float(r.get("open", 0.0)),
            "high": float(r.get("high", 0.0)),
            "low": float(r.get("low", 0.0)),
            "close": float(r.get("close", 0.0)),
            "volume": int(r.get("volume", 0)),
            "vwap": float(r["vwap"]) if r.get("vwap") is not None else None,
        }
    merged = list(by_date.values())
    merged.sort(key=lambda x: x["date"])
    coverage = {
        "start_date": merged[0]["date"] if merged else None,
        "end_date": merged[-1]["date"] if merged else None,
    }
    return {
        **existing,
        "data": merged,
        "coverage": coverage,
    }


def compute_last_complete_trading_day(today_utc: dt.date | None = None) -> str:
    """Return the most recent weekday prior to the given (or current) UTC date.

    Always steps back at least one day from today, then rolls back through
    weekends so the result is Monday–Friday.
    """
    d = today_utc or dt.datetime.utcnow().date()
    candidate = d - dt.timedelta(days=1)
    while candidate.weekday() >= 5:  # 5=Saturday, 6=Sunday
        candidate -= dt.timedelta(days=1)
    return candidate.isoformat()
