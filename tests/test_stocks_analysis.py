from __future__ import annotations

from portfolio_advisor.stocks.analysis import (
    compute_sma_series,
    compute_trailing_returns,
    compute_volatility_annualized,
)


def make_ohlc(closes: list[float], start: str = "2024-12-01") -> dict:
    from datetime import date, timedelta

    y, m, d = map(int, start.split("-"))
    rows = []
    cur = date(y, m, d)
    for i, c in enumerate(closes):
        rows.append(
            {
                "date": str(cur),
                "open": c,
                "high": c,
                "low": c,
                "close": c,
                "volume": 100 + i,
            }
        )
        cur = cur + timedelta(days=1)
    return {"ticker": "TEST", "data": rows, "coverage": {"end_date": rows[-1]["date"]}}


def test_trailing_returns_windows():
    closes = [100, 101, 102, 103, 104, 105]
    ohlc = make_ohlc(closes)
    r = compute_trailing_returns(ohlc)
    assert r["windows"]["d5"] == (105 / 100) - 1
    assert r["windows"]["d21"] is None


def test_volatility_and_sma_basic():
    closes = [100 + i for i in range(30)]
    ohlc = make_ohlc(closes)
    v = compute_volatility_annualized(ohlc, window=21)
    assert v["window"] == 21
    s = compute_sma_series(ohlc, windows=[5, 10])
    last = s["data"][-1]
    assert "sma5" in last and "sma10" in last
