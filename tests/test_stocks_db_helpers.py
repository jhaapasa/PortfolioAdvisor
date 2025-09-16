from __future__ import annotations

import datetime as dt

from portfolio_advisor.stocks.db import (
    StockPaths,
    append_ohlc_rows,
    compute_last_complete_trading_day,
    ensure_ticker_scaffold,
    read_meta,
    read_primary_ohlc,
    write_meta,
    write_primary_ohlc,
)


def test_compute_last_complete_trading_day_weekday():
    # Tuesday -> Monday
    tuesday = dt.date(2025, 9, 16)  # Tuesday
    assert compute_last_complete_trading_day(tuesday) == "2025-09-15"


def test_compute_last_complete_trading_day_monday():
    # Monday -> Friday
    monday = dt.date(2025, 9, 15)  # Monday
    assert compute_last_complete_trading_day(monday) == "2025-09-12"


def test_compute_last_complete_trading_day_weekend():
    # Saturday -> Friday, Sunday -> Friday
    saturday = dt.date(2025, 9, 13)  # Saturday
    sunday = dt.date(2025, 9, 14)  # Sunday
    assert compute_last_complete_trading_day(saturday) == "2025-09-12"
    assert compute_last_complete_trading_day(sunday) == "2025-09-12"


def test_defaults_and_write_read_roundtrip(tmp_path):
    paths = StockPaths(root=tmp_path)
    ensure_ticker_scaffold(paths, "ZZZ")

    # Defaults when files absent
    meta = read_meta(paths, "ZZZ")
    assert meta["ticker"] == "ZZZ"
    ohlc = read_primary_ohlc(paths, "ZZZ")
    assert ohlc["ticker"] == "ZZZ" and ohlc["data"] == []

    # Write round-trip
    meta["artifacts"]["primary.ohlc_daily"] = {"last_updated": "now"}
    write_meta(paths, "ZZZ", meta)
    assert read_meta(paths, "ZZZ")["artifacts"]["primary.ohlc_daily"]["last_updated"] == "now"

    # Append rows and verify coverage
    merged = append_ohlc_rows(
        ohlc,
        [
            {"date": "2025-01-01", "open": 1, "high": 1, "low": 1, "close": 1, "volume": 1},
            {"date": "2025-01-02", "open": 2, "high": 2, "low": 2, "close": 2, "volume": 2},
        ],
    )
    write_primary_ohlc(paths, "ZZZ", merged)
    saved = read_primary_ohlc(paths, "ZZZ")
    assert saved["coverage"]["start_date"] == "2025-01-01"
    assert saved["coverage"]["end_date"] == "2025-01-02"
