from __future__ import annotations

from portfolio_advisor.services.polygon_client import PolygonClient


def test_list_aggs_daily_normalizes_output(polygon_stub):
    pc = PolygonClient(api_key="x")
    calls = polygon_stub(
        [
            {"t": 1735689600000, "o": 10, "h": 11, "l": 9, "c": 10.5, "v": 100, "vw": 10.2},
            {
                "timestamp": 1735948800000,
                "open": 10.6,
                "high": 11.2,
                "low": 10.2,
                "close": 11.0,
                "volume": 110,
                "vwap": None,
            },
        ]
    )
    rows = list(pc.list_aggs_daily("AAPL", from_date="2025-01-01", to_date="2025-01-05"))

    assert len(rows) == 2
    r0, r1 = rows
    # First row has short keys
    assert r0["t"] == 1735689600000
    assert r0["o"] == 10 and r0["v"] == 100 and r0["vw"] == 10.2
    # Second row has long keys
    assert r1["timestamp"] == 1735948800000
    assert r1["close"] == 11.0 and r1["vwap"] is None
    # Check that the function was called correctly
    called = calls[0]
    assert called.ticker == "AAPL"
    assert called.limit == 50000
