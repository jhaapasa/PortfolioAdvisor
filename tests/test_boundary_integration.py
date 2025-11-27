"""Integration test for boundary stabilization with stock analysis."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from portfolio_advisor.graphs.stocks import _compute_boundary_extension_node
from portfolio_advisor.stocks.db import (
    StockPaths,
    ensure_ticker_scaffold,
    read_primary_ohlc,
    write_primary_ohlc,
)


@pytest.fixture
def temp_stock_db(tmp_path: Path) -> tuple[Path, StockPaths]:
    """Create temporary stock database for testing."""
    stock_db = tmp_path / "stocks"
    stock_db.mkdir()
    paths = StockPaths(root=stock_db)
    return stock_db, paths


@pytest.fixture
def sample_ohlc_data() -> dict:
    """Create sample OHLC data for a ticker."""
    import pandas as pd

    dates = pd.bdate_range(end="2024-01-31", periods=252)

    rows = []
    for i, date in enumerate(dates):
        rows.append(
            {
                "date": date.strftime("%Y-%m-%d"),
                "open": 100.0 + i * 0.1,
                "high": 101.0 + i * 0.1,
                "low": 99.0 + i * 0.1,
                "close": 100.0 + i * 0.1,
                "volume": 1000000,
                "vwap": 100.0 + i * 0.1,
            }
        )

    return {
        "instrument_id": "cid:stock:us:test",
        "primary_ticker": "TEST",
        "source": "polygon.io",
        "price_currency": "USD",
        "fields": ["date", "open", "high", "low", "close", "volume", "vwap"],
        "data": rows,
        "coverage": {
            "start_date": rows[0]["date"],
            "end_date": rows[-1]["date"],
        },
        "generated_at": "2024-01-31T00:00:00Z",
    }


def test_boundary_extension_node_integration(temp_stock_db, sample_ohlc_data):
    """Test that boundary extension node works in stock pipeline."""
    _stock_db, paths = temp_stock_db

    # Setup: write OHLC data
    slug = "cid-stock-us-test"
    ensure_ticker_scaffold(paths, slug)
    write_primary_ohlc(paths, slug, sample_ohlc_data)

    # Create a mock settings object
    class MockSettings:
        output_dir = str(_stock_db.parent)
        boundary_extension = True
        boundary_strategy = "linear"
        boundary_sanitization = False
        boundary_lookback = 30
        boundary_steps = 10

    # Create state
    state = {
        "settings": MockSettings(),
        "instrument": {
            "instrument_id": "cid:stock:us:test",
            "primary_ticker": "TEST",
        },
        "requested_artifacts": [],
        "_slug": slug,
        "_paths": paths,
    }

    # Execute node
    result = _compute_boundary_extension_node(state)

    # Verify result
    assert result == {}  # Node returns empty dict on success

    # Check that boundary extension file was created
    boundary_path = paths.analysis_boundary_extension_json(slug)
    assert boundary_path.exists()

    # Load and validate the extension
    with boundary_path.open("r") as f:
        extension_data = json.load(f)

    assert extension_data["instrument_id"] == "cid:stock:us:test"
    assert extension_data["primary_ticker"] == "TEST"
    assert extension_data["strategy"] == "linear"
    assert extension_data["parameters"]["steps"] == 10
    assert extension_data["parameters"]["lookback"] == 30
    assert len(extension_data["extension"]) == 10

    # Verify extension dates are in the future
    last_real_date = sample_ohlc_data["coverage"]["end_date"]
    first_ext_date = extension_data["extension"][0]["date"]
    assert first_ext_date > last_real_date


def test_boundary_extension_node_disabled(temp_stock_db, sample_ohlc_data):
    """Test that boundary extension node skips when disabled."""
    _stock_db, paths = temp_stock_db

    # Setup: write OHLC data
    slug = "cid-stock-us-test2"
    ensure_ticker_scaffold(paths, slug)
    write_primary_ohlc(paths, slug, sample_ohlc_data)

    # Create a mock settings object with boundary extension disabled
    class MockSettings:
        output_dir = str(_stock_db.parent)
        boundary_extension = False

    # Create state
    state = {
        "settings": MockSettings(),
        "instrument": {
            "instrument_id": "cid:stock:us:test2",
            "primary_ticker": "TEST2",
        },
        "requested_artifacts": [],
        "_slug": slug,
        "_paths": paths,
    }

    # Execute node
    result = _compute_boundary_extension_node(state)

    # Verify result
    assert result == {}

    # Check that boundary extension file was NOT created
    boundary_path = paths.analysis_boundary_extension_json(slug)
    assert not boundary_path.exists()


def test_boundary_extension_with_gaussian_process(temp_stock_db, sample_ohlc_data):
    """Test boundary extension with Gaussian Process strategy."""
    _stock_db, paths = temp_stock_db

    # Setup: write OHLC data
    slug = "cid-stock-us-test3"
    ensure_ticker_scaffold(paths, slug)
    write_primary_ohlc(paths, slug, sample_ohlc_data)

    # Create a mock settings object with GP strategy
    class MockSettings:
        output_dir = str(_stock_db.parent)
        boundary_extension = True
        boundary_strategy = "gaussian_process"
        boundary_sanitization = False
        boundary_lookback = 50
        boundary_steps = 5

    # Create state
    state = {
        "settings": MockSettings(),
        "instrument": {
            "instrument_id": "cid:stock:us:test3",
            "primary_ticker": "TEST3",
        },
        "requested_artifacts": [],
        "_slug": slug,
        "_paths": paths,
    }

    # Execute node
    result = _compute_boundary_extension_node(state)

    # Verify result
    assert result == {}

    # Check that boundary extension file was created
    boundary_path = paths.analysis_boundary_extension_json(slug)
    assert boundary_path.exists()

    # Load and validate the extension
    with boundary_path.open("r") as f:
        extension_data = json.load(f)

    assert extension_data["strategy"] == "gaussian_process"
    assert extension_data["parameters"]["steps"] == 5
    assert extension_data["parameters"]["lookback"] == 50
    assert len(extension_data["extension"]) == 5


def test_boundary_extension_insufficient_data(temp_stock_db):
    """Test that boundary extension node skips when data is insufficient."""
    _stock_db, paths = temp_stock_db

    # Create minimal OHLC data (< 30 rows)
    import pandas as pd

    dates = pd.bdate_range(end="2024-01-31", periods=10)

    rows = []
    for i, date in enumerate(dates):
        rows.append(
            {
                "date": date.strftime("%Y-%m-%d"),
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.0,
                "volume": 1000000,
                "vwap": 100.0,
            }
        )

    minimal_ohlc = {
        "instrument_id": "cid:stock:us:test4",
        "primary_ticker": "TEST4",
        "data": rows,
        "coverage": {
            "start_date": rows[0]["date"],
            "end_date": rows[-1]["date"],
        },
    }

    # Setup
    slug = "cid-stock-us-test4"
    ensure_ticker_scaffold(paths, slug)
    write_primary_ohlc(paths, slug, minimal_ohlc)

    # Create a mock settings object
    class MockSettings:
        output_dir = str(_stock_db.parent)
        boundary_extension = True
        boundary_strategy = "linear"
        boundary_sanitization = False
        boundary_lookback = 30
        boundary_steps = 10

    # Create state
    state = {
        "settings": MockSettings(),
        "instrument": {
            "instrument_id": "cid:stock:us:test4",
            "primary_ticker": "TEST4",
        },
        "requested_artifacts": [],
        "_slug": slug,
        "_paths": paths,
    }

    # Execute node - should skip due to insufficient data
    result = _compute_boundary_extension_node(state)

    # Verify result
    assert result == {}

    # Check that boundary extension file was NOT created (insufficient data)
    boundary_path = paths.analysis_boundary_extension_json(slug)
    assert not boundary_path.exists()


def test_boundary_extension_loaded_by_plotting(temp_stock_db, sample_ohlc_data):
    """Test that plotting can load and use boundary extension data."""
    from portfolio_advisor.stocks.plotting import render_candlestick_ohlcv_1y

    _stock_db, paths = temp_stock_db

    # Setup: write OHLC data
    slug = "cid-stock-us-test5"
    ensure_ticker_scaffold(paths, slug)
    write_primary_ohlc(paths, slug, sample_ohlc_data)

    # Create boundary extension
    class MockSettings:
        output_dir = str(_stock_db.parent)
        boundary_extension = True
        boundary_strategy = "linear"
        boundary_sanitization = False
        boundary_lookback = 30
        boundary_steps = 10

    state = {
        "settings": MockSettings(),
        "instrument": {
            "instrument_id": "cid:stock:us:test5",
            "primary_ticker": "TEST5",
        },
        "requested_artifacts": [],
        "_slug": slug,
        "_paths": paths,
    }

    _compute_boundary_extension_node(state)

    # Load OHLC and extension
    ohlc = read_primary_ohlc(paths, slug)
    boundary_path = paths.analysis_boundary_extension_json(slug)

    with boundary_path.open("r") as f:
        extension_metadata = json.load(f)

    # Render chart with extension (just verify it doesn't crash)
    ticker_dir = paths.ticker_dir(slug)
    result_path = render_candlestick_ohlcv_1y(ticker_dir, ohlc, extension_metadata)

    # Should produce a chart
    assert result_path is not None
    assert result_path.exists()
    assert result_path.name == "candle_ohlcv_1y.png"

