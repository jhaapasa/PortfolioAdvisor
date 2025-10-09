from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..utils.fs import utcnow_iso, write_json_atomic


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

    def analysis_wavelet_coeffs_json(self, ticker: str) -> Path:
        return self.ticker_dir(ticker) / "analysis" / "wavelet_coefficients_logreturns_sym4.json"

    def analysis_wavelet_hist_json(self, ticker: str) -> Path:
        return self.ticker_dir(ticker) / "analysis" / "volatility_histogram.json"

    def analysis_wavelet_coeffs_logprice_json(self, ticker: str) -> Path:
        return self.ticker_dir(ticker) / "analysis" / "wavelet_coefficients_logprice_sym4.json"

    def analysis_wavelet_reconstructed_prices_json(self, ticker: str) -> Path:
        return self.ticker_dir(ticker) / "analysis" / "wavelet_reconstructed_prices.json"

    def report_dir(self, ticker: str) -> Path:
        return self.ticker_dir(ticker) / "report"

    def report_candle_1y_png(self, ticker: str) -> Path:
        return self.report_dir(ticker) / "candle_ohlcv_1y.png"

    def report_candle_2y_wavelet_trends_png(self, ticker: str) -> Path:
        return self.report_dir(ticker) / "candle_ohlcv_2y_wavelet_trends.png"

    def lock_dir(self, ticker: str) -> Path:
        return self.ticker_dir(ticker) / ".lock"

    def news_dir(self, ticker: str) -> Path:
        return self.ticker_dir(ticker) / "primary" / "news"

    def news_index_json(self, ticker: str) -> Path:
        return self.news_dir(ticker) / "index.json"

    def news_article_json(self, ticker: str, article_id: str) -> Path:
        return self.news_dir(ticker) / f"{article_id}.json"

    def news_articles_dir(self, ticker: str) -> Path:
        return self.news_dir(ticker) / "articles"

    def news_article_content(self, ticker: str, article_id: str, extension: str = "html") -> Path:
        return self.news_articles_dir(ticker) / f"{article_id}.{extension}"


def ensure_ticker_scaffold(paths: StockPaths, ticker: str) -> None:
    base = paths.ticker_dir(ticker)
    (base / "primary").mkdir(parents=True, exist_ok=True)
    (base / "analysis").mkdir(parents=True, exist_ok=True)
    (base / "report").mkdir(parents=True, exist_ok=True)


def _read_json(path: Path) -> dict[str, Any] | list[dict[str, Any]]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _write_json(path: Path, data: Any) -> None:
    write_json_atomic(path, data)


# use shared utcnow_iso from utils.fs


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
    d = today_utc or dt.datetime.now(dt.UTC).date()
    candidate = d - dt.timedelta(days=1)
    while candidate.weekday() >= 5:  # 5=Saturday, 6=Sunday
        candidate -= dt.timedelta(days=1)
    return candidate.isoformat()
