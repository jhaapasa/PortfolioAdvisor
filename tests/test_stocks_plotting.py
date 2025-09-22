from __future__ import annotations

import datetime as dt
from pathlib import Path

from portfolio_advisor.stocks.plotting import (
    plot_wavelet_variance_spectrum,
    render_candlestick_ohlcv_1y,
)
from portfolio_advisor.stocks.wavelet import normalize_variance_spectrum


def _make_ohlc(days: int = 260) -> dict:
    base = dt.date(2024, 1, 1)
    rows = []
    price = 100.0
    for i in range(days):
        d = base + dt.timedelta(days=i)
        if d.weekday() >= 5:
            continue
        open_p = price
        high_p = open_p * 1.01
        low_p = open_p * 0.99
        close_p = open_p * (1.0 + (0.002 if (i % 10) == 0 else -0.001))
        price = close_p
        rows.append(
            {
                "date": d.isoformat(),
                "open": open_p,
                "high": high_p,
                "low": low_p,
                "close": close_p,
                "volume": 100000 + i,
            }
        )
    coverage_end = rows[-1]["date"] if rows else None
    return {
        "instrument_id": "stocks:US:NASDAQ:AAPL",
        "primary_ticker": "AAPL",
        "data": rows,
        "coverage": {"start_date": rows[0]["date"] if rows else None, "end_date": coverage_end},
    }


def test_render_candlestick_ohlcv_1y(tmp_path: Path):
    ohlc = _make_ohlc(300)
    out = render_candlestick_ohlcv_1y(tmp_path, ohlc)
    assert out is not None
    assert out.exists()
    assert out.stat().st_size > 0


def test_render_skips_on_short_data(tmp_path: Path):
    ohlc = _make_ohlc(50)
    out = render_candlestick_ohlcv_1y(tmp_path, ohlc)
    assert out is None


def test_plot_wavelet_variance_spectrum(tmp_path: Path):
    # Construct a simple per-level variance and normalize
    per_level = {"D1": 1.0, "D2": 2.0, "D3": 3.0, "D4": 4.0, "D5": 5.0, "S5": 6.0}
    normalized = normalize_variance_spectrum(per_level)
    # Percentages should sum to ~100
    assert abs(sum(normalized.values()) - 100.0) < 1e-6
    # Render
    out = plot_wavelet_variance_spectrum(tmp_path, normalized, title="Wavelet Variance Spectrum")
    assert out.exists()
    assert out.stat().st_size > 0
