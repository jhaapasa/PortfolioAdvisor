from __future__ import annotations

import json
from pathlib import Path

from portfolio_advisor.stocks.db import (
    StockPaths,
    append_ohlc_rows,
    ensure_ticker_scaffold,
    read_meta,
    read_primary_ohlc,
    write_meta,
    write_primary_ohlc,
)


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

