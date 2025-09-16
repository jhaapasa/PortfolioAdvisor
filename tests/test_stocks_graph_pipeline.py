from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from portfolio_advisor.config import Settings
from portfolio_advisor.graphs.stocks import update_ticker
from portfolio_advisor.services.polygon_client import PolygonClient


def _ts_ms(yyyy_mm_dd: str) -> int:
    dt = datetime.strptime(yyyy_mm_dd, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def test_update_ticker_pipeline_writes_all_artifacts(tmp_path, monkeypatch):
    # Arrange: stub PolygonClient.list_aggs_daily to return two days of data
    # Provide normalized rows as returned by PolygonClient.list_aggs_daily
    bars = [
        {
            "date": "2025-01-02",
            "open": 10.0,
            "high": 11.0,
            "low": 9.0,
            "close": 10.5,
            "volume": 100,
            "vwap": 10.3,
        },
        {
            "date": "2025-01-03",
            "open": 10.6,
            "high": 11.2,
            "low": 10.2,
            "close": 11.0,
            "volume": 110,
            "vwap": 10.8,
        },
    ]

    def fake_list_aggs_daily(self, ticker, from_date, to_date, adjusted=True, limit=50000):  # noqa: ARG001
        return iter(bars)

    monkeypatch.setattr(PolygonClient, "list_aggs_daily", fake_list_aggs_daily)

    out_dir = tmp_path / "out"
    inp_dir = tmp_path / "in"
    out_dir.mkdir(parents=True, exist_ok=True)
    inp_dir.mkdir(parents=True, exist_ok=True)

    settings = Settings(input_dir=str(inp_dir), output_dir=str(out_dir), verbose=True)

    # Act
    update_ticker(settings, ticker="TEST")

    # Assert: all expected artifacts exist and have reasonable content
    base = Path(out_dir) / "stocks" / "tickers" / "TEST"
    primary = base / "primary" / "ohlc_daily.json"
    returns = base / "analysis" / "returns.json"
    vol = base / "analysis" / "volatility.json"
    sma = base / "analysis" / "sma_20_50_100_200.json"
    meta = base / "meta.json"

    for p in (primary, returns, vol, sma, meta):
        assert p.exists(), f"missing artifact: {p}"

    with primary.open("r", encoding="utf-8") as fh:
        p = json.load(fh)
    assert p["ticker"] == "TEST"
    assert len(p["data"]) == 2
    assert p["coverage"]["end_date"] == "2025-01-03"

    with returns.open("r", encoding="utf-8") as fh:
        r = json.load(fh)
    assert r["depends_on"] == ["primary.ohlc_daily"]
    assert r["ticker"] == "TEST"

    with vol.open("r", encoding="utf-8") as fh:
        v = json.load(fh)
    assert v["window"] == 21
    assert v["depends_on"] == ["primary.ohlc_daily"]

    with sma.open("r", encoding="utf-8") as fh:
        s = json.load(fh)
    assert s["windows"] == [20, 50, 100, 200]
    assert s["coverage"]["end_date"] == "2025-01-03"

    with meta.open("r", encoding="utf-8") as fh:
        m = json.load(fh)
    assert m["ticker"] == "TEST"
    assert m["artifacts"]["primary.ohlc_daily"]["last_updated"]

