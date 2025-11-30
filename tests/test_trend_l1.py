"""Tests for L1 trend filtering (sparse trend extraction)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from portfolio_advisor.trend.l1_filter import (
    L1TrendFilter,
    TrendResult,
    extract_l1_trend,
    to_trend_json,
)


class TestL1TrendFilter:
    """Tests for L1TrendFilter class."""

    def test_fit_transform_basic(self):
        """Test basic trend extraction."""
        # Create simple uptrend data
        dates = pd.date_range("2024-01-01", periods=100, freq="D")
        values = np.linspace(100, 120, 100) + np.random.randn(100) * 0.5
        series = pd.Series(values, index=dates)

        filter_obj = L1TrendFilter(lambda_param=10.0)
        result = filter_obj.fit_transform(series)

        assert isinstance(result, TrendResult)
        assert len(result.trend) == len(series)
        assert len(result.velocity) == len(series)
        assert len(result.knots) == len(series)
        assert result.lambda_used == 10.0
        # Trend should follow the general direction
        assert result.trend.iloc[-1] > result.trend.iloc[0]

    def test_fit_transform_piecewise_linear(self):
        """Test that filter detects structural breaks in piecewise linear data."""
        # Create data with clear regime change
        dates = pd.date_range("2024-01-01", periods=100, freq="D")

        # First half: uptrend, second half: downtrend
        x1 = np.linspace(100, 150, 50)
        x2 = np.linspace(150, 130, 50)
        values = np.concatenate([x1, x2])

        series = pd.Series(values, index=dates)

        filter_obj = L1TrendFilter(lambda_param=1.0)
        result = filter_obj.fit_transform(series)

        # Should detect at least one knot near the regime change
        assert result.knot_count() >= 1
        # Velocity should change sign
        velocity_change = result.velocity.iloc[60] - result.velocity.iloc[40]
        assert velocity_change < 0  # Changed from upward to downward

    def test_fit_transform_flat_trend(self):
        """Test filter with flat/stationary data."""
        dates = pd.date_range("2024-01-01", periods=50, freq="D")
        values = np.ones(50) * 100.0 + np.random.randn(50) * 0.1
        series = pd.Series(values, index=dates)

        filter_obj = L1TrendFilter(lambda_param=100.0)
        result = filter_obj.fit_transform(series)

        # With high lambda and flat data, should have few/no knots
        # Velocity should be near zero
        assert np.abs(result.velocity.iloc[-1]) < 1.0

    def test_requires_minimum_length(self):
        """Test that filter requires at least 3 data points."""
        dates = pd.date_range("2024-01-01", periods=2, freq="D")
        series = pd.Series([100.0, 101.0], index=dates)

        filter_obj = L1TrendFilter()
        with pytest.raises(ValueError, match="at least 3 points"):
            filter_obj.fit_transform(series)

    def test_invalid_lambda_raises(self):
        """Test that invalid lambda parameter raises error."""
        with pytest.raises(ValueError, match="must be positive"):
            L1TrendFilter(lambda_param=0.0)

        with pytest.raises(ValueError, match="must be positive"):
            L1TrendFilter(lambda_param=-1.0)

    def test_invalid_solver_raises(self):
        """Test that invalid solver raises error."""
        with pytest.raises(ValueError, match="Unknown solver"):
            L1TrendFilter(solver="UNKNOWN")

    def test_handles_nan_values(self):
        """Test that filter handles NaN values by forward filling."""
        dates = pd.date_range("2024-01-01", periods=50, freq="D")
        values = np.linspace(100, 110, 50).astype(float)
        values[10] = np.nan
        values[25] = np.nan
        series = pd.Series(values, index=dates)

        filter_obj = L1TrendFilter(lambda_param=10.0)
        result = filter_obj.fit_transform(series)

        # Should complete without error
        assert len(result.trend) == 50
        assert not np.any(np.isnan(result.trend.values))

    def test_high_lambda_smooths_more(self):
        """Test that higher lambda produces smoother trend (fewer knots)."""
        dates = pd.date_range("2024-01-01", periods=100, freq="D")
        # Noisy data with multiple local peaks
        values = 100 + np.sin(np.linspace(0, 4 * np.pi, 100)) * 10 + np.random.randn(100) * 2
        series = pd.Series(values, index=dates)

        # Low lambda: more sensitive to changes
        result_low = L1TrendFilter(lambda_param=0.5).fit_transform(series)
        # High lambda: smoother
        result_high = L1TrendFilter(lambda_param=100.0).fit_transform(series)

        # Higher lambda should produce fewer knots
        assert result_high.knot_count() <= result_low.knot_count()

    def test_preserves_index(self):
        """Test that filter preserves the original series index."""
        dates = pd.date_range("2024-01-01", periods=50, freq="D")
        series = pd.Series(np.linspace(100, 110, 50), index=dates)

        filter_obj = L1TrendFilter()
        result = filter_obj.fit_transform(series)

        pd.testing.assert_index_equal(result.trend.index, dates)
        pd.testing.assert_index_equal(result.velocity.index, dates)
        pd.testing.assert_index_equal(result.knots.index, dates)

    def test_solver_fallback(self):
        """Test that solver fallback mechanism works."""
        # This test verifies the fallback mechanism exists
        # In practice, some solvers may fail for certain problems, triggering fallback
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        series = pd.Series(np.linspace(100, 110, 30), index=dates)

        # Test with OSQP (the primary/default solver)
        filter_obj = L1TrendFilter(lambda_param=1.0, solver="OSQP")
        result = filter_obj.fit_transform(series)

        # The result should use one of the supported solvers
        assert result.solver_stats["solver"] in ["OSQP", "ECOS", "SCS"]
        assert result.solver_stats["status"] in ["optimal", "optimal_inaccurate"]

        # Test that specifying a non-default solver is accepted
        # (may fallback to OSQP if solver unavailable/fails)
        for solver in ["ECOS", "SCS"]:
            filter_obj = L1TrendFilter(lambda_param=1.0, solver=solver)
            result = filter_obj.fit_transform(series)
            # Should produce valid result regardless of which solver ultimately ran
            assert result.solver_stats["status"] in ["optimal", "optimal_inaccurate"]
            assert len(result.trend) == 30


class TestL1TrendFilterAutoTune:
    """Tests for automatic lambda tuning via BIC."""

    def test_auto_tune_basic(self):
        """Test automatic lambda tuning."""
        dates = pd.date_range("2024-01-01", periods=100, freq="D")
        # Clear piecewise linear signal
        x1 = np.linspace(100, 130, 50)
        x2 = np.linspace(130, 120, 50)
        values = np.concatenate([x1, x2]) + np.random.randn(100) * 0.5
        series = pd.Series(values, index=dates)

        filter_obj = L1TrendFilter(lambda_param=1.0, auto_tune=True)
        result = filter_obj.fit_transform(series)

        # Auto-tuned lambda should be reasonable
        assert result.lambda_used > 0.0
        assert result.bic != 0.0

    def test_bic_computed_without_auto_tune(self):
        """Test that BIC is computed even without auto-tuning."""
        dates = pd.date_range("2024-01-01", periods=50, freq="D")
        series = pd.Series(np.linspace(100, 110, 50), index=dates)

        filter_obj = L1TrendFilter(lambda_param=10.0, auto_tune=False)
        result = filter_obj.fit_transform(series)

        # BIC should be computed
        assert result.bic != 0.0


class TestTrendResult:
    """Tests for TrendResult dataclass."""

    def test_knot_dates(self):
        """Test knot_dates method returns correct dates."""
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        trend = pd.Series(np.arange(10), index=dates)
        velocity = pd.Series(np.ones(10), index=dates)
        knots = pd.Series(
            [False, False, True, False, False, True, False, False, False, False], index=dates
        )

        result = TrendResult(
            trend=trend,
            velocity=velocity,
            knots=knots,
            lambda_used=1.0,
            bic=100.0,
            solver_stats={},
        )

        knot_dates = result.knot_dates()
        assert len(knot_dates) == 2
        assert "2024-01-03" in knot_dates[0]
        assert "2024-01-06" in knot_dates[1]

    def test_knot_count(self):
        """Test knot_count method."""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        trend = pd.Series(np.arange(5), index=dates)
        velocity = pd.Series(np.ones(5), index=dates)
        knots = pd.Series([False, True, True, False, True], index=dates)

        result = TrendResult(
            trend=trend,
            velocity=velocity,
            knots=knots,
            lambda_used=1.0,
            bic=100.0,
            solver_stats={},
        )

        assert result.knot_count() == 3


class TestExtractL1Trend:
    """Tests for extract_l1_trend convenience function."""

    def test_extract_with_auto_tune(self):
        """Test convenience function with auto-tuning."""
        dates = pd.date_range("2024-01-01", periods=50, freq="D")
        series = pd.Series(np.linspace(100, 120, 50), index=dates)

        result = extract_l1_trend(series)

        assert isinstance(result, TrendResult)
        assert result.lambda_used > 0.0

    def test_extract_with_explicit_lambda(self):
        """Test convenience function with explicit lambda."""
        dates = pd.date_range("2024-01-01", periods=50, freq="D")
        series = pd.Series(np.linspace(100, 120, 50), index=dates)

        result = extract_l1_trend(series, lambda_param=25.0, auto_tune=False)

        assert result.lambda_used == 25.0


class TestToTrendJson:
    """Tests for to_trend_json serialization function."""

    def test_serialization_basic(self):
        """Test basic JSON serialization."""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        result = TrendResult(
            trend=pd.Series([100.0, 101.0, 102.0, 103.0, 104.0], index=dates),
            velocity=pd.Series([0.0, 1.0, 1.0, 1.0, 1.0], index=dates),
            knots=pd.Series([False, False, True, False, False], index=dates),
            lambda_used=10.0,
            bic=50.0,
            solver_stats={"solver": "OSQP", "status": "optimal"},
        )

        json_doc = to_trend_json("AAPL", result)

        assert json_doc["ticker"] == "AAPL"
        assert json_doc["lambda"] == 10.0
        assert json_doc["bic"] == 50.0
        assert json_doc["knots_count"] == 1
        assert json_doc["current_velocity"] == 1.0
        assert len(json_doc["trend"]) == 5
        assert len(json_doc["knots"]) == 1
        assert json_doc["solver"] == "OSQP"

    def test_serialization_with_original_length(self):
        """Test serialization includes original_length metadata."""
        dates = pd.date_range("2024-01-01", periods=3, freq="D")
        result = TrendResult(
            trend=pd.Series([100.0, 101.0, 102.0], index=dates),
            velocity=pd.Series([0.0, 1.0, 1.0], index=dates),
            knots=pd.Series([False, False, False], index=dates),
            lambda_used=5.0,
            bic=25.0,
            solver_stats={},
        )

        json_doc = to_trend_json("TEST", result, original_length=100)

        assert json_doc["metadata"]["original_length"] == 100
        assert json_doc["metadata"]["trend_length"] == 3


class TestSecondDiffMatrix:
    """Tests for second difference matrix construction."""

    def test_matrix_shape(self):
        """Test that second diff matrix has correct shape."""
        filter_obj = L1TrendFilter()

        D = filter_obj._build_second_diff_matrix(10)
        assert D.shape == (8, 10)  # (n-2, n)

        D = filter_obj._build_second_diff_matrix(5)
        assert D.shape == (3, 5)

    def test_matrix_values(self):
        """Test that second diff matrix produces correct differences."""
        filter_obj = L1TrendFilter()

        D = filter_obj._build_second_diff_matrix(5)
        x = np.array([1, 2, 4, 7, 11])  # Second diffs: 1, 0, 0

        result = D @ x
        # D[0] = x[0] - 2*x[1] + x[2] = 1 - 4 + 4 = 1
        # D[1] = x[1] - 2*x[2] + x[3] = 2 - 8 + 7 = 1
        # D[2] = x[2] - 2*x[3] + x[4] = 4 - 14 + 11 = 1
        np.testing.assert_array_almost_equal(result, [1, 1, 1])

    def test_matrix_too_small_raises(self):
        """Test that building matrix for n<3 raises error."""
        filter_obj = L1TrendFilter()

        with pytest.raises(ValueError, match="n=2 < 3"):
            filter_obj._build_second_diff_matrix(2)


class TestIntegration:
    """Integration tests for L1 trend filtering."""

    def test_realistic_stock_data(self):
        """Test with realistic stock price simulation."""
        # Simulate 1 year of daily data with trend + noise
        np.random.seed(42)
        dates = pd.bdate_range(end="2024-01-31", periods=252)

        # Random walk with drift
        returns = np.random.randn(252) * 0.02 + 0.0005  # ~12% annual return
        prices = 100 * np.exp(np.cumsum(returns))

        series = pd.Series(prices, index=dates)

        filter_obj = L1TrendFilter(lambda_param=50.0)
        result = filter_obj.fit_transform(series)

        # Basic sanity checks
        assert len(result.trend) == 252
        assert result.trend.iloc[-1] > 0
        # Trend should be smoother than original
        original_std = series.diff().std()
        trend_std = result.trend.diff().std()
        assert trend_std < original_std

    def test_with_boundary_extension(self):
        """Test L1 filter works with boundary-extended data."""
        # Create base data
        dates = pd.bdate_range(end="2024-01-31", periods=100)
        prices = np.linspace(100, 120, 100) + np.random.randn(100) * 1.0

        # Simulate boundary extension by adding future points
        extended_dates = pd.bdate_range(start=dates[-1] + pd.Timedelta(days=1), periods=10)
        extended_prices = np.linspace(120, 125, 10)

        full_dates = dates.append(extended_dates)
        full_prices = np.concatenate([prices, extended_prices])
        series = pd.Series(full_prices, index=full_dates)

        filter_obj = L1TrendFilter(lambda_param=20.0)
        result = filter_obj.fit_transform(series)

        # Should work on full extended series
        assert len(result.trend) == 110

        # Truncate to original length (simulate post-processing)
        original_trend = result.trend.iloc[:100]
        assert len(original_trend) == 100

    def test_velocity_sign_correctness(self):
        """Test that velocity has correct sign for trends."""
        dates = pd.date_range("2024-01-01", periods=50, freq="D")

        # Clear uptrend
        uptrend = pd.Series(np.linspace(100, 150, 50), index=dates)
        result_up = L1TrendFilter(lambda_param=100.0).fit_transform(uptrend)

        # Velocity should be predominantly positive
        assert (result_up.velocity > 0).sum() > 40

        # Clear downtrend
        downtrend = pd.Series(np.linspace(150, 100, 50), index=dates)
        result_down = L1TrendFilter(lambda_param=100.0).fit_transform(downtrend)

        # Velocity should be predominantly negative
        assert (result_down.velocity < 0).sum() > 40

    def test_knots_at_regime_changes(self):
        """Test that knots are detected at clear regime changes."""
        dates = pd.date_range("2024-01-01", periods=150, freq="D")

        # Three-regime data: up, flat, down
        regime1 = np.linspace(100, 130, 50)  # Up
        regime2 = np.ones(50) * 130  # Flat
        regime3 = np.linspace(130, 110, 50)  # Down

        prices = np.concatenate([regime1, regime2, regime3])
        series = pd.Series(prices, index=dates)

        filter_obj = L1TrendFilter(lambda_param=1.0)
        result = filter_obj.fit_transform(series)

        # Should detect regime changes
        knot_indices = np.where(result.knots.values)[0]

        # At least 2 knots (at regime boundaries)
        assert len(knot_indices) >= 2

        # Knots should be near the regime boundaries (indices 50 and 100)
        has_knot_near_50 = any(abs(k - 50) < 10 for k in knot_indices)
        has_knot_near_100 = any(abs(k - 100) < 10 for k in knot_indices)
        assert has_knot_near_50 or has_knot_near_100
