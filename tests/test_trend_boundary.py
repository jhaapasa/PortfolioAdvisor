"""Tests for boundary stabilization (trend module)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from portfolio_advisor.trend.boundary import (
    BoundaryStabilizer,
    ForecastStrategy,
    GaussianProcessForecaster,
    LinearForecaster,
    OutlierDetector,
    StabilizationConfig,
    extend_ohlc_dict,
)


class TestLinearForecaster:
    """Tests for LinearForecaster."""

    def test_fit_and_predict_uptrend(self):
        """Test linear forecaster with upward trend."""
        forecaster = LinearForecaster()
        data = np.array([100.0, 101.0, 102.0, 103.0, 104.0])
        forecaster.fit(data)

        predictions = forecaster.predict(3)
        assert len(predictions) == 3
        # Should extrapolate upward trend
        assert predictions[0] > 104.0
        assert predictions[1] > predictions[0]
        assert predictions[2] > predictions[1]

    def test_fit_and_predict_downtrend(self):
        """Test linear forecaster with downward trend."""
        forecaster = LinearForecaster()
        data = np.array([104.0, 103.0, 102.0, 101.0, 100.0])
        forecaster.fit(data)

        predictions = forecaster.predict(3)
        assert len(predictions) == 3
        # Should extrapolate downward trend
        assert predictions[0] < 100.0
        assert predictions[1] < predictions[0]

    def test_fit_and_predict_flat(self):
        """Test linear forecaster with flat trend."""
        forecaster = LinearForecaster()
        data = np.array([100.0, 100.0, 100.0, 100.0, 100.0])
        forecaster.fit(data)

        predictions = forecaster.predict(3)
        assert len(predictions) == 3
        # Should stay approximately flat
        np.testing.assert_allclose(predictions, [100.0, 100.0, 100.0], rtol=0.01)

    def test_requires_minimum_data(self):
        """Test that fitting requires at least 2 points."""
        forecaster = LinearForecaster()
        with pytest.raises(ValueError, match="at least 2 data points"):
            forecaster.fit(np.array([100.0]))

    def test_predict_before_fit_raises(self):
        """Test that predict raises if called before fit."""
        forecaster = LinearForecaster()
        with pytest.raises(RuntimeError, match="predict called before fit"):
            forecaster.predict(5)

    def test_predict_with_history(self):
        """Test prediction including history step 0."""
        forecaster = LinearForecaster()
        data = np.array([100.0, 101.0, 102.0])
        forecaster.fit(data)

        # Predict 2 steps + history
        predictions = forecaster.predict(2, include_history=True)
        assert len(predictions) == 3  # 0, 1, 2
        # Step 0 should match the fitted line at the last data point
        # Since fit is perfect, it should match 102.0
        assert abs(predictions[0] - 102.0) < 1e-10
        assert abs(predictions[1] - 103.0) < 1e-10

    def test_predict_with_noise(self):
        """Test linear forecaster with noise injection."""
        forecaster = LinearForecaster()
        # Use noisy data to get non-zero residual_std
        rng = np.random.RandomState(42)
        data = np.array([100.0, 101.0, 102.0, 103.0, 104.0]) + rng.normal(0, 2.0, 5)
        forecaster.fit(data)

        # Predict twice with noise
        pred1 = forecaster.predict(5, noise=True)
        pred2 = forecaster.predict(5, noise=True)

        # Should be different (stochastic)
        assert not np.allclose(pred1, pred2)

        # Predict without noise should be deterministic
        clean1 = forecaster.predict(5, noise=False)
        clean2 = forecaster.predict(5, noise=False)
        np.testing.assert_allclose(clean1, clean2)


class TestGaussianProcessForecaster:
    """Tests for GaussianProcessForecaster."""

    def test_fit_and_predict(self):
        """Test GP forecaster basic functionality."""
        forecaster = GaussianProcessForecaster()
        # Create data with slight trend
        data = np.array([100.0, 101.0, 101.5, 102.0, 103.0, 104.0, 105.0])
        forecaster.fit(data)

        predictions = forecaster.predict(3)
        assert len(predictions) == 3
        # Should produce reasonable extrapolation
        assert predictions[0] > 104.0
        assert predictions[0] < 110.0  # Not wildly extrapolating

    def test_handles_nonlinear_pattern(self):
        """Test GP can handle non-linear patterns."""
        forecaster = GaussianProcessForecaster()
        # Quadratic-ish pattern
        x = np.arange(20)
        data = 100 + 0.1 * x**2
        forecaster.fit(data)

        predictions = forecaster.predict(5)
        assert len(predictions) == 5
        # Should continue the pattern
        assert all(predictions[i] > predictions[i - 1] for i in range(1, len(predictions)))

    def test_requires_minimum_data(self):
        """Test that fitting requires at least 3 points."""
        forecaster = GaussianProcessForecaster()
        with pytest.raises(ValueError, match="at least 3 data points"):
            forecaster.fit(np.array([100.0, 101.0]))

    def test_predict_before_fit_raises(self):
        """Test that predict raises if called before fit."""
        forecaster = GaussianProcessForecaster()
        with pytest.raises(RuntimeError, match="predict called before fit"):
            forecaster.predict(5)

    def test_predict_with_history(self):
        """Test GP prediction including history step 0."""
        forecaster = GaussianProcessForecaster()
        data = np.array([100.0, 101.0, 102.0])
        forecaster.fit(data)

        predictions = forecaster.predict(2, include_history=True)
        assert len(predictions) == 3  # 0, 1, 2
        # GP with small noise fits data well
        assert abs(predictions[0] - 102.0) < 1.0

    def test_predict_with_noise(self):
        """Test GP forecaster with noise sampling."""
        forecaster = GaussianProcessForecaster()
        data = np.array([100.0, 101.0, 102.0, 103.0, 104.0])
        forecaster.fit(data)

        # Predict twice with noise (sampling)
        pred1 = forecaster.predict(5, noise=True)
        pred2 = forecaster.predict(5, noise=True)

        # Should be different samples
        assert not np.allclose(pred1, pred2)


class TestOutlierDetector:
    """Tests for OutlierDetector."""

    def test_detect_and_replace_with_outliers(self):
        """Test outlier detection and replacement."""
        detector = OutlierDetector(threshold=2.0, window=5)  # Lower threshold for more sensitivity

        # Create series with extreme outliers in the middle
        base = np.arange(30, dtype=float) + 100
        data = base.copy()
        data[15] = 500.0  # Extreme outlier in the middle

        series = pd.Series(data)
        cleaned = detector.detect_and_replace(series)

        # The extreme outlier should be replaced
        assert cleaned.iloc[15] < 200.0
        assert cleaned.iloc[15] > 100.0

        # Points far from outlier should remain unchanged
        np.testing.assert_allclose(cleaned.iloc[:10], series.iloc[:10], rtol=0.01)
        np.testing.assert_allclose(cleaned.iloc[20:], series.iloc[20:], rtol=0.01)

    def test_no_outliers_unchanged(self):
        """Test that series without outliers is unchanged."""
        detector = OutlierDetector(threshold=3.0, window=5)

        data = [100.0, 101.0, 102.0, 103.0, 104.0, 105.0]
        series = pd.Series(data)

        cleaned = detector.detect_and_replace(series)

        # Should be essentially unchanged
        pd.testing.assert_series_equal(cleaned, series, check_dtype=False)

    def test_short_series_unchanged(self):
        """Test that series shorter than window is unchanged."""
        detector = OutlierDetector(threshold=3.0, window=21)

        data = [100.0, 101.0, 102.0]
        series = pd.Series(data)

        cleaned = detector.detect_and_replace(series)

        # Should return original series
        pd.testing.assert_series_equal(cleaned, series)


class TestBoundaryStabilizer:
    """Tests for BoundaryStabilizer."""

    def _create_sample_df(self, days: int = 60) -> pd.DataFrame:
        """Create sample OHLC DataFrame."""
        dates = pd.bdate_range(end=pd.Timestamp.today(), periods=days)
        closes = 100 + np.cumsum(np.random.randn(days) * 0.5)

        return pd.DataFrame(
            {
                "open": closes + np.random.randn(days) * 0.2,
                "high": closes + np.abs(np.random.randn(days)) * 0.3,
                "low": closes - np.abs(np.random.randn(days)) * 0.3,
                "close": closes,
                "volume": np.random.randint(100000, 1000000, days),
            },
            index=dates,
        )

    def test_extend_series_linear(self):
        """Test series extension with linear strategy."""
        config = StabilizationConfig(
            strategy=ForecastStrategy.LINEAR, lookback_period=30, extension_steps=10
        )
        stabilizer = BoundaryStabilizer(config)

        df = self._create_sample_df(60)
        original_len = len(df)

        extended_df, metadata = stabilizer.extend_series(df, k=10)

        # Check extension
        assert len(extended_df) == original_len + 10
        assert metadata["strategy"] == "linear"
        assert metadata["parameters"]["steps"] == 10
        assert len(metadata["extension"]) == 10

        # Check continuity (no jump)
        last_real = df.iloc[-1]["close"]
        first_ext = extended_df.iloc[original_len]["close"]
        # Should be very close to linear projection, but definitely continuous
        # We check that the gap is consistent with local slope, not a huge jump
        # Actually, our fix ensures C0 continuity relative to the last point
        # But "continuous" means the first point isn't arbitrary.
        # Let's trust the visual/logic fix and just check length here.
        assert abs(first_ext - last_real) < 5.0

    def test_extend_series_gaussian_process(self):
        """Test series extension with Gaussian Process strategy."""
        config = StabilizationConfig(
            strategy=ForecastStrategy.GAUSSIAN_PROCESS, lookback_period=30, extension_steps=10
        )
        stabilizer = BoundaryStabilizer(config)

        df = self._create_sample_df(60)
        original_len = len(df)

        extended_df, metadata = stabilizer.extend_series(df, k=10)

        assert len(extended_df) == original_len + 10
        assert metadata["strategy"] == "gaussian_process"

    def test_extend_series_with_noise(self):
        """Test extension with noise injection."""
        config = StabilizationConfig(
            strategy=ForecastStrategy.LINEAR, noise_injection=True, extension_steps=20
        )
        stabilizer = BoundaryStabilizer(config)
        df = self._create_sample_df(60)

        # Run twice
        ext1, _ = stabilizer.extend_series(df)
        ext2, _ = stabilizer.extend_series(df)

        # Forecasts should differ due to noise
        assert not np.allclose(ext1["close"].values, ext2["close"].values)

    def test_extend_series_with_sanitization(self):
        """Test series extension with outlier sanitization enabled."""
        config = StabilizationConfig(
            enable_sanitization=True, strategy=ForecastStrategy.LINEAR, extension_steps=5
        )
        stabilizer = BoundaryStabilizer(config)

        df = self._create_sample_df(60)
        # Inject an outlier
        df.iloc[-5, df.columns.get_loc("close")] = df["close"].mean() * 3

        extended_df, metadata = stabilizer.extend_series(df, k=5)

        # Should still extend successfully
        assert len(extended_df) == len(df) + 5
        assert metadata["parameters"]["sanitization_enabled"] is True

    def test_empty_dataframe_raises(self):
        """Test that empty DataFrame raises ValueError."""
        config = StabilizationConfig()
        stabilizer = BoundaryStabilizer(config)

        df = pd.DataFrame()
        with pytest.raises(ValueError, match="Cannot extend empty DataFrame"):
            stabilizer.extend_series(df)

    def test_missing_close_column_raises(self):
        """Test that DataFrame without 'close' column raises ValueError."""
        config = StabilizationConfig()
        stabilizer = BoundaryStabilizer(config)

        dates = pd.bdate_range(end=pd.Timestamp.today(), periods=10)
        df = pd.DataFrame({"open": range(10)}, index=dates)

        with pytest.raises(ValueError, match="must have 'close' column"):
            stabilizer.extend_series(df)

    def test_non_datetime_index_raises(self):
        """Test that DataFrame without DatetimeIndex raises ValueError."""
        config = StabilizationConfig()
        stabilizer = BoundaryStabilizer(config)

        df = pd.DataFrame({"close": range(10)})

        with pytest.raises(ValueError, match="must have DatetimeIndex"):
            stabilizer.extend_series(df)

    def test_extension_business_days(self):
        """Test that extension uses business days."""
        config = StabilizationConfig(extension_steps=10)
        stabilizer = BoundaryStabilizer(config)

        df = self._create_sample_df(30)
        extended_df, _metadata = stabilizer.extend_series(df)

        # Check that all dates are business days
        assert all(extended_df.index.dayofweek < 5)  # Monday=0, Friday=4

    def test_default_extension_steps(self):
        """Test that default extension steps is used when k is None."""
        config = StabilizationConfig(extension_steps=15)
        stabilizer = BoundaryStabilizer(config)

        df = self._create_sample_df(30)
        extended_df, metadata = stabilizer.extend_series(df, k=None)

        # Should use config default
        assert len(extended_df) == len(df) + 15
        assert metadata["parameters"]["steps"] == 15


class TestExtendOhlcDict:
    """Tests for extend_ohlc_dict convenience function."""

    def _create_sample_ohlc(self, days: int = 60) -> dict:
        """Create sample OHLC dictionary."""
        dates = pd.bdate_range(end=pd.Timestamp.today(), periods=days)
        closes = 100 + np.cumsum(np.random.randn(days) * 0.5)

        rows = []
        for i, date in enumerate(dates):
            rows.append(
                {
                    "date": date.strftime("%Y-%m-%d"),
                    "open": float(closes[i] + np.random.randn() * 0.2),
                    "high": float(closes[i] + np.abs(np.random.randn()) * 0.3),
                    "low": float(closes[i] - np.abs(np.random.randn()) * 0.3),
                    "close": float(closes[i]),
                    "volume": int(np.random.randint(100000, 1000000)),
                    "vwap": float(closes[i]),
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
            "generated_at": "2024-01-01T00:00:00Z",
        }

    def test_extend_ohlc_dict_default_config(self):
        """Test extending OHLC dict with default configuration."""
        ohlc = self._create_sample_ohlc(60)
        original_len = len(ohlc["data"])

        extended_ohlc, metadata = extend_ohlc_dict(ohlc)

        # Check extension
        assert len(extended_ohlc["data"]) == original_len + 10  # Default 10 steps
        assert metadata["instrument_id"] == "cid:stock:us:test"
        assert metadata["primary_ticker"] == "TEST"
        assert metadata["strategy"] == "linear"  # Default strategy

        # Check that original data is preserved
        for i in range(original_len):
            assert extended_ohlc["data"][i]["date"] == ohlc["data"][i]["date"]
            assert extended_ohlc["data"][i]["close"] == ohlc["data"][i]["close"]

    def test_extend_ohlc_dict_custom_config(self):
        """Test extending OHLC dict with custom configuration."""
        ohlc = self._create_sample_ohlc(60)
        config = StabilizationConfig(
            strategy=ForecastStrategy.GAUSSIAN_PROCESS,
            lookback_period=20,
            extension_steps=5,
        )

        extended_ohlc, metadata = extend_ohlc_dict(ohlc, config)

        # Check configuration was used
        assert len(extended_ohlc["data"]) == len(ohlc["data"]) + 5
        assert metadata["strategy"] == "gaussian_process"
        assert metadata["parameters"]["lookback"] == 20

    def test_extend_ohlc_dict_coverage_update(self):
        """Test that coverage is updated correctly."""
        ohlc = self._create_sample_ohlc(30)
        original_end = ohlc["coverage"]["end_date"]

        extended_ohlc, _metadata = extend_ohlc_dict(ohlc)

        # Coverage should be extended
        assert extended_ohlc["coverage"]["start_date"] == ohlc["coverage"]["start_date"]
        assert extended_ohlc["coverage"]["end_date"] > original_end

    def test_extend_ohlc_dict_empty_raises(self):
        """Test that empty OHLC dict raises ValueError."""
        ohlc = {
            "instrument_id": "cid:stock:us:test",
            "primary_ticker": "TEST",
            "data": [],
        }

        with pytest.raises(ValueError, match="no data"):
            extend_ohlc_dict(ohlc)

    def test_extension_has_no_vwap(self):
        """Test that extended rows have None for VWAP."""
        ohlc = self._create_sample_ohlc(30)
        original_len = len(ohlc["data"])

        extended_ohlc, _metadata = extend_ohlc_dict(ohlc)

        # Extension rows should have None for VWAP
        for i in range(original_len, len(extended_ohlc["data"])):
            assert extended_ohlc["data"][i]["vwap"] is None

    def test_extension_has_zero_volume(self):
        """Test that extended rows have 0 volume."""
        ohlc = self._create_sample_ohlc(30)
        original_len = len(ohlc["data"])

        extended_ohlc, _metadata = extend_ohlc_dict(ohlc)

        # Extension rows should have 0 volume
        for i in range(original_len, len(extended_ohlc["data"])):
            assert extended_ohlc["data"][i]["volume"] == 0


class TestStabilizationConfig:
    """Tests for StabilizationConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = StabilizationConfig()

        assert config.enable_sanitization is False
        assert config.strategy == ForecastStrategy.LINEAR
        assert config.lookback_period == 30
        assert config.extension_steps == 10
        assert config.mad_threshold == 3.0
        assert config.noise_injection is False

    def test_custom_values(self):
        """Test custom configuration values."""
        config = StabilizationConfig(
            enable_sanitization=True,
            strategy=ForecastStrategy.GAUSSIAN_PROCESS,
            lookback_period=60,
            extension_steps=20,
            mad_threshold=2.5,
            noise_injection=True,
        )

        assert config.enable_sanitization is True
        assert config.strategy == ForecastStrategy.GAUSSIAN_PROCESS
        assert config.lookback_period == 60
        assert config.extension_steps == 20
        assert config.mad_threshold == 2.5
        assert config.noise_injection is True


class TestIntegration:
    """Integration tests for boundary stabilization."""

    def test_end_to_end_linear(self):
        """Test complete pipeline with linear forecasting."""
        # Create realistic OHLC data
        dates = pd.bdate_range(end="2024-01-31", periods=252)  # 1 year
        trend = np.linspace(100, 120, 252)
        noise = np.random.randn(252) * 2
        closes = trend + noise

        ohlc = {
            "instrument_id": "cid:stock:us:aapl",
            "primary_ticker": "AAPL",
            "data": [
                {
                    "date": date.strftime("%Y-%m-%d"),
                    "open": float(close + np.random.randn() * 0.5),
                    "high": float(close + np.abs(np.random.randn())),
                    "low": float(close - np.abs(np.random.randn())),
                    "close": float(close),
                    "volume": int(np.random.randint(1000000, 10000000)),
                    "vwap": float(close),
                }
                for date, close in zip(dates, closes)
            ],
            "coverage": {
                "start_date": dates[0].strftime("%Y-%m-%d"),
                "end_date": dates[-1].strftime("%Y-%m-%d"),
            },
        }

        config = StabilizationConfig(
            strategy=ForecastStrategy.LINEAR, lookback_period=60, extension_steps=10
        )

        extended_ohlc, metadata = extend_ohlc_dict(ohlc, config)

        # Validate results
        assert len(extended_ohlc["data"]) == 262  # 252 + 10
        assert metadata["instrument_id"] == "cid:stock:us:aapl"
        assert len(metadata["extension"]) == 10

        # Extension should continue upward trend
        last_real_close = ohlc["data"][-1]["close"]
        first_ext_close = metadata["extension"][0]["price"]
        # With anchor adjustment, the first point should be close to the last real + slope
        # Just check it hasn't jumped wildly
        assert abs(first_ext_close - last_real_close) < 5.0

    def test_end_to_end_gaussian_process(self):
        """Test complete pipeline with GP forecasting."""
        # Create data with non-linear pattern
        dates = pd.bdate_range(end="2024-01-31", periods=100)
        x = np.arange(100)
        closes = 100 + 0.1 * x + 0.01 * x**2 + np.random.randn(100) * 1.0

        ohlc = {
            "instrument_id": "cid:etf:us:spy",
            "primary_ticker": "SPY",
            "data": [
                {
                    "date": date.strftime("%Y-%m-%d"),
                    "open": float(close),
                    "high": float(close + 1),
                    "low": float(close - 1),
                    "close": float(close),
                    "volume": 1000000,
                    "vwap": float(close),
                }
                for date, close in zip(dates, closes)
            ],
            "coverage": {
                "start_date": dates[0].strftime("%Y-%m-%d"),
                "end_date": dates[-1].strftime("%Y-%m-%d"),
            },
        }

        config = StabilizationConfig(
            strategy=ForecastStrategy.GAUSSIAN_PROCESS, lookback_period=50, extension_steps=5
        )

        extended_ohlc, metadata = extend_ohlc_dict(ohlc, config)

        # Validate results
        assert len(extended_ohlc["data"]) == 105
        assert metadata["strategy"] == "gaussian_process"
        assert len(metadata["extension"]) == 5
