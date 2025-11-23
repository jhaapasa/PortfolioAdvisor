"""Tests for MarketMetricsService."""

import json

import numpy as np
import pytest

from portfolio_advisor.models.market import ReferenceTickerMetrics
from portfolio_advisor.stocks.analysis import MarketMetricsService


@pytest.fixture
def mock_ohlc_data():
    """Generate mock OHLC data for testing."""
    data = []
    base_price = 100.0
    # Generate 600 days of data with some volatility
    for i in range(600):
        # Add some random walk with drift
        daily_return = 0.0002 + np.random.normal(0, 0.02)  # ~0.02% daily drift, 2% daily vol
        base_price *= 1 + daily_return
        data.append(
            {
                "date": f"2024-{1 + i // 250:02d}-{1 + (i % 250) // 21:02d}",
                "open": base_price * 0.99,
                "high": base_price * 1.01,
                "low": base_price * 0.98,
                "close": base_price,
            }
        )
    return data


@pytest.fixture
def mock_stock_db(tmp_path, mock_ohlc_data):
    """Create mock stock database structure."""
    # Create SPY data
    spy_dir = tmp_path / "tickers" / "cid-etf-us-spy" / "primary"
    spy_dir.mkdir(parents=True)
    with open(spy_dir / "ohlc_daily.json", "w") as f:
        json.dump({"data": mock_ohlc_data}, f)

    # Create AAPL data with slightly different returns
    aapl_data = []
    base_price = 150.0
    for i in range(600):
        daily_return = 0.0003 + np.random.normal(0, 0.025)  # Higher vol than SPY
        base_price *= 1 + daily_return
        aapl_data.append(
            {
                "date": f"2024-{1 + i // 250:02d}-{1 + (i % 250) // 21:02d}",
                "open": base_price * 0.99,
                "high": base_price * 1.01,
                "low": base_price * 0.98,
                "close": base_price,
            }
        )

    aapl_dir = tmp_path / "tickers" / "cid-stocks-us-composite-aapl" / "primary"
    aapl_dir.mkdir(parents=True)
    with open(aapl_dir / "ohlc_daily.json", "w") as f:
        json.dump({"data": aapl_data}, f)

    return tmp_path


@pytest.fixture
def service(mock_stock_db):
    """Create MarketMetricsService instance."""
    risk_free_rate_daily = 0.045 / 252  # 4.5% annual
    return MarketMetricsService(mock_stock_db, risk_free_rate_daily)


class TestMarketMetricsService:
    """Test MarketMetricsService functionality."""

    def test_compute_metrics_for_symbol(self, service):
        """Test computing metrics for a symbol."""
        metrics = service.compute_metrics_for_symbol("SPY", "cid-etf-us-spy", [63, 126, 252, 504])

        assert isinstance(metrics, ReferenceTickerMetrics)
        assert metrics.symbol == "SPY"
        assert metrics.as_of is not None

        # Check returns are computed for all horizons
        assert 63 in metrics.returns
        assert 126 in metrics.returns
        assert 252 in metrics.returns
        assert 504 in metrics.returns

        # Check Sharpe ratios are computed
        assert 63 in metrics.sharpe_ratios
        assert 126 in metrics.sharpe_ratios
        assert 252 in metrics.sharpe_ratios
        assert 504 in metrics.sharpe_ratios

        # Check volatility is computed
        assert metrics.volatility_annualized is not None
        assert metrics.volatility_annualized > 0

    def test_caching(self, service):
        """Test that results are cached."""
        # First call
        metrics1 = service.compute_metrics_for_symbol("SPY", "cid-etf-us-spy", [252])

        # Second call should return cached result
        metrics2 = service.compute_metrics_for_symbol("SPY", "cid-etf-us-spy", [252])

        assert metrics1 is metrics2  # Same object reference

    def test_compute_beta(self, service):
        """Test beta computation between two symbols."""
        beta, r_squared = service.compute_beta(
            "AAPL", "cid-stocks-us-composite-aapl", "SPY", "cid-etf-us-spy", window_days=252
        )

        assert beta is not None
        assert r_squared is not None
        assert 0 <= r_squared <= 1  # R-squared should be between 0 and 1

    def test_insufficient_data(self, service):
        """Test handling of insufficient data."""
        # Create a ticker with only 100 days of data
        short_dir = service.stock_db_root / "tickers" / "short-ticker" / "primary"
        short_dir.mkdir(parents=True)

        short_data = []
        base_price = 50.0
        for i in range(100):
            base_price *= 1 + np.random.normal(0, 0.02)
            short_data.append({"date": f"2024-01-{i+1:02d}", "close": base_price})

        with open(short_dir / "ohlc_daily.json", "w") as f:
            json.dump({"data": short_data}, f)

        # Try to compute metrics with horizons longer than available data
        metrics = service.compute_metrics_for_symbol("SHORT", "short-ticker", [63, 126, 252])

        # Should have data for 63 days but not 252
        assert metrics.returns[63] is not None
        assert metrics.returns[252] is None
        assert metrics.sharpe_ratios[252] is None

    def test_clear_cache(self, service):
        """Test cache clearing."""
        # Populate cache
        service.compute_metrics_for_symbol("SPY", "cid-etf-us-spy", [252])

        # Clear cache
        service.clear_cache()

        # Should recompute (we can't easily test this isn't the same object,
        # but we can verify it works)
        metrics2 = service.compute_metrics_for_symbol("SPY", "cid-etf-us-spy", [252])

        assert metrics2.symbol == "SPY"

    def test_missing_ticker(self, service):
        """Test handling of missing ticker data."""
        with pytest.raises(ValueError, match="No OHLC data found"):
            service.compute_metrics_for_symbol("MISSING", "missing-ticker", [252])

    def test_sharpe_ratio_calculation(self, service):
        """Test Sharpe ratio calculation logic."""
        # Create deterministic returns for testing
        returns = np.array([0.01, -0.005, 0.008, -0.002, 0.012, 0.003, -0.001, 0.007])
        risk_free_rate = 0.0001

        sharpe = service._compute_sharpe_ratio(returns, risk_free_rate, annualize=False)

        # Manual calculation
        excess_returns = returns - risk_free_rate
        expected_sharpe = np.mean(excess_returns) / np.std(excess_returns, ddof=1)

        assert sharpe is not None
        assert abs(sharpe - expected_sharpe) < 0.0001

    def test_volatility_calculation(self, service):
        """Test volatility calculation."""
        # Create returns with known volatility
        returns = np.array([0.01, -0.01, 0.01, -0.01] * 10)  # 40 returns

        vol = service._compute_volatility(returns, window=20)

        # Last 20 returns
        recent_returns = returns[-20:]
        expected_vol = np.std(recent_returns, ddof=1) * np.sqrt(252)

        assert vol is not None
        assert abs(vol - expected_vol) < 0.0001
