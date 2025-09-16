from __future__ import annotations

from portfolio_advisor.stocks.db import append_ohlc_rows


def test_append_ohlc_rows_merges_and_updates_coverage():
    existing = {
        "ticker": "AAPL",
        "data": [
            {"date": "2024-12-30", "open": 10, "high": 11, "low": 9, "close": 10, "volume": 1},
            {"date": "2024-12-31", "open": 11, "high": 12, "low": 10, "close": 11, "volume": 2},
        ],
        "coverage": {"start_date": "2024-12-30", "end_date": "2024-12-31"},
    }
    new_rows = [
        {"date": "2024-12-31", "open": 12, "high": 13, "low": 11, "close": 12, "volume": 3},
        {"date": "2025-01-02", "open": 13, "high": 14, "low": 12, "close": 13, "volume": 4},
    ]

    merged = append_ohlc_rows(existing, new_rows)

    # Overlapping date should be replaced with new values
    d_map = {r["date"]: r for r in merged["data"]}
    assert d_map["2024-12-31"]["close"] == 12
    # New date should be appended and coverage updated
    assert merged["coverage"]["end_date"] == "2025-01-02"
    assert merged["coverage"]["start_date"] == "2024-12-30"
