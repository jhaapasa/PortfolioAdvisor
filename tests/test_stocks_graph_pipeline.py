from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from portfolio_advisor.config import Settings
from portfolio_advisor.graphs.stocks import update_instrument
from portfolio_advisor.services.polygon_client import PolygonClient


def _ts_ms(yyyy_mm_dd: str) -> int:
    dt = datetime.strptime(yyyy_mm_dd, "%Y-%m-%d").replace(tzinfo=UTC)
    return int(dt.timestamp() * 1000)


def test_update_ticker_pipeline_writes_all_artifacts(tmp_path, monkeypatch):
    # Arrange: stub PolygonClient.list_aggs_daily to return a modest set of data
    # Provide normalized rows as returned by PolygonClient.list_aggs_daily
    # Generate ~260 trading-like days for report rendering
    # Use a date range that's definitely in the past to ensure primary fetch is triggered
    bars = []
    from datetime import date, timedelta

    d = date(2023, 1, 1)
    price = 100.0
    added = 0
    while added < 260:
        if d.weekday() < 5:
            open_p = price
            high_p = open_p * 1.01
            low_p = open_p * 0.99
            close_p = open_p * (1.0 + (0.001 if (added % 7) else -0.0005))
            price = close_p
            bars.append(
                {
                    "date": d.isoformat(),
                    "open": open_p,
                    "high": high_p,
                    "low": low_p,
                    "close": close_p,
                    "volume": 1000 + added,
                    "vwap": (high_p + low_p + close_p) / 3,
                }
            )
            added += 1
        d += timedelta(days=1)

    def fake_list_aggs_daily(
        self, ticker, from_date, to_date, adjusted=True, limit=50000
    ):  # noqa: ARG001
        return iter(bars)

    monkeypatch.setattr(PolygonClient, "list_aggs_daily", fake_list_aggs_daily)

    out_dir = tmp_path / "out"
    inp_dir = tmp_path / "in"
    out_dir.mkdir(parents=True, exist_ok=True)
    inp_dir.mkdir(parents=True, exist_ok=True)

    settings = Settings(input_dir=str(inp_dir), output_dir=str(out_dir), verbose=True)

    # Act
    update_instrument(
        settings,
        instrument={
            "instrument_id": "cid:stocks:us:composite:TEST",
            "primary_ticker": "TEST",
        },
    )

    # Assert: all expected artifacts exist and have reasonable content
    base = Path(out_dir) / "stocks" / "tickers" / "cid-stocks-us-composite-test"
    primary = base / "primary" / "ohlc_daily.json"
    returns = base / "analysis" / "returns.json"
    vol = base / "analysis" / "volatility.json"
    sma = base / "analysis" / "sma_20_50_100_200.json"
    meta = base / "meta.json"

    for p in (primary, returns, vol, sma, meta):
        assert p.exists(), f"missing artifact: {p}"

    with primary.open("r", encoding="utf-8") as fh:
        p = json.load(fh)
    assert p["primary_ticker"] == "TEST"
    assert len(p["data"]) >= 180
    assert p["coverage"]["end_date"]

    with returns.open("r", encoding="utf-8") as fh:
        r = json.load(fh)
    assert r["depends_on"] == ["primary.ohlc_daily"]
    assert r["primary_ticker"] == "TEST"

    with vol.open("r", encoding="utf-8") as fh:
        v = json.load(fh)
    assert v["window"] == 21
    assert v["depends_on"] == ["primary.ohlc_daily"]

    with sma.open("r", encoding="utf-8") as fh:
        s = json.load(fh)
    assert s["windows"] == [20, 50, 100, 200]
    assert s["coverage"]["end_date"]

    # Report image should exist when sufficient data present
    report_img = base / "report" / "candle_ohlcv_1y.png"
    assert report_img.exists()
    assert report_img.stat().st_size > 0

    with meta.open("r", encoding="utf-8") as fh:
        m = json.load(fh)
    assert m["primary_ticker"] == "TEST"
    assert m["artifacts"]["primary.ohlc_daily"]["last_updated"]
