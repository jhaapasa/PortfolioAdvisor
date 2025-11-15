"""Integration tests for cone of influence visualization in wavelet plots."""

from pathlib import Path

import numpy as np
import pytest

from portfolio_advisor.stocks.plotting import (
    _create_coi_plot_segments,
    render_candlestick_ohlcv_2y_wavelet_trends,
)


@pytest.fixture
def sample_series():
    """Create a sample pandas Series for testing."""
    try:
        import pandas as pd
    except ImportError:
        pytest.skip("pandas not available")

    dates = pd.date_range("2024-01-01", periods=100, freq="D")
    values = np.sin(np.linspace(0, 4 * np.pi, 100)) * 100 + 1000
    return pd.Series(values, index=dates)


def test_create_coi_plot_segments_basic(sample_series):
    """Test basic COI plot segment creation."""
    # COI: first 10 and last 10 samples
    plots = _create_coi_plot_segments(
        series=sample_series, coi_start=10, coi_end=90, color="#d62728", width=2.0, alpha=0.8
    )

    # Should create 3 segments: left COI, reliable, right COI
    assert len(plots) == 3


def test_create_coi_plot_segments_no_left_coi(sample_series):
    """Test COI when start is at beginning."""
    plots = _create_coi_plot_segments(
        series=sample_series, coi_start=0, coi_end=90, color="#d62728", width=2.0, alpha=0.8
    )

    # Should create 2 segments: reliable and right COI
    assert len(plots) == 2


def test_create_coi_plot_segments_no_right_coi(sample_series):
    """Test COI when end is at series end."""
    plots = _create_coi_plot_segments(
        series=sample_series, coi_start=10, coi_end=100, color="#d62728", width=2.0, alpha=0.8
    )

    # Should create 2 segments: left COI and reliable
    assert len(plots) == 2


def test_create_coi_plot_segments_no_coi(sample_series):
    """Test when entire series is reliable (no COI)."""
    plots = _create_coi_plot_segments(
        series=sample_series, coi_start=0, coi_end=100, color="#d62728", width=2.0, alpha=0.8
    )

    # Should create 1 segment: all reliable
    assert len(plots) == 1


def test_create_coi_plot_segments_full_coi(sample_series):
    """Test when COI covers entire series."""
    plots = _create_coi_plot_segments(
        series=sample_series, coi_start=50, coi_end=50, color="#d62728", width=2.0, alpha=0.8
    )

    # Should create 2 segments: left and right COI (no reliable region)
    assert len(plots) == 2


def test_create_coi_plot_segments_boundary_clipping(sample_series):
    """Test that out-of-bounds COI values are clipped."""
    # Negative start should be clipped to 0
    plots = _create_coi_plot_segments(
        series=sample_series, coi_start=-10, coi_end=90, color="#d62728", width=2.0, alpha=0.8
    )
    assert len(plots) >= 1  # Should handle gracefully

    # End beyond series length should be clipped
    plots = _create_coi_plot_segments(
        series=sample_series, coi_start=10, coi_end=200, color="#d62728", width=2.0, alpha=0.8
    )
    assert len(plots) >= 1  # Should handle gracefully


def test_create_coi_plot_segments_with_nans(sample_series):
    """Test COI plotting with NaN values in series."""
    # Introduce some NaNs
    series_with_nans = sample_series.copy()
    series_with_nans.iloc[20:30] = np.nan

    plots = _create_coi_plot_segments(
        series=series_with_nans,
        coi_start=10,
        coi_end=90,
        color="#d62728",
        width=2.0,
        alpha=0.8,
    )

    # Should still create segments (mplfinance handles NaNs)
    assert len(plots) >= 1


def test_render_with_coi_metadata(tmp_path: Path):
    """Test rendering wavelet trends with COI metadata."""
    # Create mock OHLC data
    ohlc = {
        "primary_ticker": "TEST",
        "data": [
            {
                "date": f"2024-{m:02d}-01",
                "open": 100 + i,
                "high": 105 + i,
                "low": 95 + i,
                "close": 102 + i,
                "volume": 1000000,
            }
            for i, m in enumerate(range(1, 13), 1)
        ]
        * 50,  # ~600 days
        "coverage": {"end_date": "2025-12-31"},
    }

    # Create reconstruction doc with COI boundaries
    recon_doc = {
        "metadata": {
            "level": 5,
            "wavelet": "sym4",
            "coi_boundaries": {
                "S5": [112, 392],  # COI on both ends
                "S4": [56, 448],
                "S3": [28, 476],
            },
        },
        "reconstructions": {
            "S5": [{"date": row["date"], "value": row["close"]} for row in ohlc["data"][-504:]],
            "S4": [
                {"date": row["date"], "value": row["close"] * 1.01} for row in ohlc["data"][-504:]
            ],
            "S3": [
                {"date": row["date"], "value": row["close"] * 0.99} for row in ohlc["data"][-504:]
            ],
        },
    }

    # Render the plot
    out_path = render_candlestick_ohlcv_2y_wavelet_trends(tmp_path, ohlc, recon_doc)

    # Verify output was created
    assert out_path is not None
    assert out_path.exists()
    assert out_path.suffix == ".png"


def test_render_without_coi_metadata(tmp_path: Path):
    """Test rendering wavelet trends without COI metadata (backward compatibility)."""
    # Create mock OHLC data
    ohlc = {
        "primary_ticker": "TEST",
        "data": [
            {
                "date": f"2024-{m:02d}-01",
                "open": 100 + i,
                "high": 105 + i,
                "low": 95 + i,
                "close": 102 + i,
                "volume": 1000000,
            }
            for i, m in enumerate(range(1, 13), 1)
        ]
        * 50,
        "coverage": {"end_date": "2025-12-31"},
    }

    # Reconstruction doc WITHOUT COI boundaries
    recon_doc = {
        "metadata": {
            "level": 5,
            "wavelet": "sym4",
        },
        "reconstructions": {
            "S5": [{"date": row["date"], "value": row["close"]} for row in ohlc["data"][-504:]],
        },
    }

    # Should still render successfully (fallback to simple lines)
    out_path = render_candlestick_ohlcv_2y_wavelet_trends(tmp_path, ohlc, recon_doc)

    assert out_path is not None
    assert out_path.exists()


def test_render_with_partial_coi_metadata(tmp_path: Path):
    """Test rendering when only some levels have COI metadata."""
    ohlc = {
        "primary_ticker": "TEST",
        "data": [
            {
                "date": f"2024-{m:02d}-01",
                "open": 100 + i,
                "high": 105 + i,
                "low": 95 + i,
                "close": 102 + i,
                "volume": 1000000,
            }
            for i, m in enumerate(range(1, 13), 1)
        ]
        * 50,
        "coverage": {"end_date": "2025-12-31"},
    }

    # COI only for S5, not for S4
    recon_doc = {
        "metadata": {
            "level": 5,
            "wavelet": "sym4",
            "coi_boundaries": {
                "S5": [112, 392],  # Only S5 has COI
            },
        },
        "reconstructions": {
            "S5": [{"date": row["date"], "value": row["close"]} for row in ohlc["data"][-504:]],
            "S4": [
                {"date": row["date"], "value": row["close"] * 1.01} for row in ohlc["data"][-504:]
            ],
        },
    }

    # Should render with COI for S5, simple line for S4
    out_path = render_candlestick_ohlcv_2y_wavelet_trends(tmp_path, ohlc, recon_doc)

    assert out_path is not None
    assert out_path.exists()
