"""Boundary stabilization for trend filtering.

This module implements synthetic time series extension to mitigate edge effects
in signal processing filters (like the ℓ1 trend filter). By forecasting a short
horizon into the future, it provides the filter with "future" data at t_now,
stabilizing trend extraction at the most critical point.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol

import numpy as np
import pandas as pd

_logger = logging.getLogger(__name__)


class ForecastStrategy(Enum):
    """Available forecasting strategies for boundary extension."""

    LINEAR = "linear"
    GAUSSIAN_PROCESS = "gaussian_process"


@dataclass
class StabilizationConfig:
    """Configuration for boundary stabilization.

    Attributes:
        enable_sanitization: Whether to apply outlier detection/cleaning
        strategy: Forecasting method to use
        lookback_period: Number of recent periods to use for fitting
        extension_steps: Number of steps to forecast into the future
        mad_threshold: Threshold for outlier detection (in MAD units)
    """

    enable_sanitization: bool = False
    strategy: ForecastStrategy = ForecastStrategy.LINEAR
    lookback_period: int = 30
    extension_steps: int = 10
    mad_threshold: float = 3.0


class ForecastModel(Protocol):
    """Protocol for forecasting models used in boundary extension."""

    def fit(self, data: np.ndarray) -> None:
        """Fit the model to historical data.

        Args:
            data: 1D array of historical values (e.g., close prices)
        """
        ...

    def predict(self, steps: int) -> np.ndarray:
        """Generate forecast for the specified number of steps.

        Args:
            steps: Number of future periods to predict

        Returns:
            1D array of forecasted values
        """
        ...


class OutlierDetector:
    """Detect and replace outliers using rolling Median Absolute Deviation (MAD).

    This is an optional sanitization step to remove microstructure noise
    that could distort trend filters.
    """

    def __init__(self, threshold: float = 3.0, window: int = 21):
        """Initialize outlier detector.

        Args:
            threshold: Number of MAD units to consider as outlier
            window: Rolling window size for MAD calculation
        """
        self.threshold = threshold
        self.window = window

    def detect_and_replace(self, series: pd.Series) -> pd.Series:
        """Detect outliers and replace with rolling median.

        Args:
            series: Time series to clean

        Returns:
            Cleaned series with outliers replaced by rolling median
        """
        if len(series) < self.window:
            _logger.debug(
                "OutlierDetector.skip: series too short (%d < %d)", len(series), self.window
            )
            return series

        # Compute rolling median and MAD
        rolling_median = series.rolling(window=self.window, center=True).median()
        deviations = np.abs(series - rolling_median)
        rolling_mad = deviations.rolling(window=self.window, center=True).median()

        # Identify outliers
        outlier_mask = deviations > (self.threshold * rolling_mad)

        # Replace outliers with rolling median
        cleaned = series.copy()
        cleaned[outlier_mask] = rolling_median[outlier_mask]

        # Fill any NaN values at edges with original values
        cleaned = cleaned.fillna(series)

        outlier_count = outlier_mask.sum()
        if outlier_count > 0:
            _logger.debug(
                "OutlierDetector: replaced %d outliers (%.1f%%)",
                outlier_count,
                100.0 * outlier_count / len(series),
            )

        return cleaned


class LinearForecaster:
    """Simple linear extrapolation forecaster.

    Fits a linear regression to recent data and extrapolates into the future.
    Fast, robust, and captures immediate local momentum without overfitting.
    """

    def __init__(self):
        """Initialize linear forecaster."""
        self._coeffs: np.ndarray | None = None
        self._last_x: float = 0.0

    def fit(self, data: np.ndarray) -> None:
        """Fit linear model to data.

        Args:
            data: 1D array of historical values
        """
        if len(data) < 2:
            raise ValueError(f"LinearForecaster requires at least 2 data points, got {len(data)}")

        # Create time index
        x = np.arange(len(data))
        y = data

        # Fit linear polynomial (degree 1)
        self._coeffs = np.polyfit(x, y, deg=1)
        self._last_x = len(data) - 1

        _logger.debug(
            "LinearForecaster.fit: slope=%.4f, intercept=%.4f", self._coeffs[0], self._coeffs[1]
        )

    def predict(self, steps: int) -> np.ndarray:
        """Generate linear forecast.

        Args:
            steps: Number of future periods to predict

        Returns:
            Array of forecasted values
        """
        if self._coeffs is None:
            raise RuntimeError("LinearForecaster.predict called before fit")

        # Extrapolate from last observed point
        future_x = np.arange(self._last_x + 1, self._last_x + 1 + steps)
        forecast = np.polyval(self._coeffs, future_x)

        return forecast


class GaussianProcessForecaster:
    """Gaussian Process Regression forecaster.

    Uses a non-parametric Bayesian approach that can:
    - Adapt to non-stationary local changes
    - Model heteroskedasticity explicitly
    - Produce smooth continuations ideal for boundary conditions
    """

    def __init__(self, length_scale: float = 10.0, noise_level: float = 0.1):
        """Initialize GP forecaster.

        Args:
            length_scale: Characteristic length scale for RBF kernel
            noise_level: Expected noise level in the data
        """
        self.length_scale = length_scale
        self.noise_level = noise_level
        self._gp = None
        self._last_x: float = 0.0

    def fit(self, data: np.ndarray) -> None:
        """Fit GP model to data.

        Args:
            data: 1D array of historical values
        """
        if len(data) < 3:
            raise ValueError(
                f"GaussianProcessForecaster requires at least 3 data points, got {len(data)}"
            )

        try:
            from sklearn.gaussian_process import GaussianProcessRegressor
            from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
        except ImportError as exc:
            raise RuntimeError("scikit-learn is required for GaussianProcessForecaster") from exc

        # Prepare training data
        X = np.arange(len(data)).reshape(-1, 1)
        y = data

        # Define kernel: ConstantKernel * RBF + WhiteKernel
        # RBF captures smooth trends, WhiteKernel models noise
        kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(
            length_scale=self.length_scale, length_scale_bounds=(1.0, 100.0)
        ) + WhiteKernel(noise_level=self.noise_level, noise_level_bounds=(1e-5, 1.0))

        # Fit GP
        self._gp = GaussianProcessRegressor(
            kernel=kernel, n_restarts_optimizer=5, alpha=1e-6, normalize_y=True
        )
        self._gp.fit(X, y)
        self._last_x = len(data) - 1

        _logger.debug("GaussianProcessForecaster.fit: kernel=%s", self._gp.kernel_)

    def predict(self, steps: int) -> np.ndarray:
        """Generate GP forecast.

        Args:
            steps: Number of future periods to predict

        Returns:
            Array of forecasted values (mean predictions)
        """
        if self._gp is None:
            raise RuntimeError("GaussianProcessForecaster.predict called before fit")

        # Predict future points
        future_x = np.arange(self._last_x + 1, self._last_x + 1 + steps).reshape(-1, 1)
        forecast, _std = self._gp.predict(future_x, return_std=True)

        return forecast


class BoundaryStabilizer:
    """Main service for boundary stabilization and extension.

    Orchestrates sanitization and forecasting to extend time series
    synthetically, providing stable boundary conditions for trend filters.
    """

    def __init__(self, config: StabilizationConfig):
        """Initialize boundary stabilizer.

        Args:
            config: Configuration for stabilization behavior
        """
        self.config = config
        self.outlier_detector = (
            OutlierDetector(threshold=config.mad_threshold) if config.enable_sanitization else None
        )

    def _get_forecaster(self) -> ForecastModel:
        """Create forecaster instance based on configuration.

        Returns:
            Initialized forecaster instance
        """
        if self.config.strategy == ForecastStrategy.LINEAR:
            return LinearForecaster()
        elif self.config.strategy == ForecastStrategy.GAUSSIAN_PROCESS:
            return GaussianProcessForecaster()
        else:
            raise ValueError(f"Unknown forecast strategy: {self.config.strategy}")

    def extend_series(
        self, df: pd.DataFrame, k: int | None = None
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Extend OHLC DataFrame with forecasted future values.

        Args:
            df: DataFrame with OHLC data (requires 'close' column and DatetimeIndex)
            k: Number of steps to forecast (defaults to config.extension_steps)

        Returns:
            Tuple of:
            - Extended DataFrame with original + forecasted rows
            - Extension metadata dict (for serialization)
        """
        if k is None:
            k = self.config.extension_steps

        if len(df) == 0:
            raise ValueError("Cannot extend empty DataFrame")

        if "close" not in df.columns:
            raise ValueError("DataFrame must have 'close' column")

        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame must have DatetimeIndex")

        # Extract close prices
        close_series = df["close"].copy()

        # Optional: sanitize data
        if self.config.enable_sanitization and self.outlier_detector is not None:
            close_series = self.outlier_detector.detect_and_replace(close_series)
            _logger.debug("BoundaryStabilizer: sanitization applied")

        # Select lookback window
        lookback = min(self.config.lookback_period, len(close_series))
        training_data = close_series.tail(lookback).values

        # Fit forecaster
        forecaster = self._get_forecaster()
        forecaster.fit(training_data)

        # Generate forecast
        forecast_values = forecaster.predict(k)

        _logger.info(
            "BoundaryStabilizer: extended series by %d steps using %s strategy",
            k,
            self.config.strategy.value,
        )

        # Create future dates (business days)
        last_date = df.index[-1]
        future_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=k)

        # Create extension DataFrame
        extension_df = pd.DataFrame(
            {
                "close": forecast_values,
                "open": forecast_values,  # Simple approximation
                "high": forecast_values,  # Simple approximation
                "low": forecast_values,  # Simple approximation
                "volume": 0,  # No volume for forecasted data
            },
            index=future_dates,
        )

        # Combine original and extension
        extended_df = pd.concat([df, extension_df])

        # Create metadata for serialization
        metadata = {
            "strategy": self.config.strategy.value,
            "parameters": {
                "lookback": lookback,
                "steps": k,
                "sanitization_enabled": self.config.enable_sanitization,
            },
            "last_real_date": last_date.strftime("%Y-%m-%d"),
            "extension": [
                {"date": date.strftime("%Y-%m-%d"), "price": float(price)}
                for date, price in zip(future_dates, forecast_values)
            ],
        }

        return extended_df, metadata


def extend_ohlc_dict(
    ohlc: dict[str, Any], config: StabilizationConfig | None = None
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Extend OHLC dictionary with boundary stabilization.

    This is a convenience function that works directly with the OHLC
    dictionary format used in the stock database.

    Args:
        ohlc: OHLC dictionary with 'data' list and metadata
        config: Stabilization configuration (uses defaults if None)

    Returns:
        Tuple of:
        - Extended OHLC dictionary (original + forecasted rows)
        - Extension metadata for serialization
    """
    if config is None:
        config = StabilizationConfig()

    # Convert to DataFrame
    rows = ohlc.get("data", [])
    if not rows:
        raise ValueError("OHLC dictionary has no data")

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()

    # Ensure numeric types
    for col in ("open", "high", "low", "close"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "volume" in df.columns:
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0).astype(int)

    # Extend
    stabilizer = BoundaryStabilizer(config)
    extended_df, metadata = stabilizer.extend_series(df)

    # Convert back to OHLC dict format
    extended_rows = []
    for date_idx, row in extended_df.iterrows():
        extended_rows.append(
            {
                "date": date_idx.strftime("%Y-%m-%d"),
                "open": float(row.get("open", 0.0)),
                "high": float(row.get("high", 0.0)),
                "low": float(row.get("low", 0.0)),
                "close": float(row.get("close", 0.0)),
                "volume": int(row.get("volume", 0)),
                "vwap": None,  # No VWAP for forecasted data
            }
        )

    extended_ohlc = {
        **ohlc,
        "data": extended_rows,
        "coverage": {
            "start_date": extended_rows[0]["date"] if extended_rows else None,
            "end_date": extended_rows[-1]["date"] if extended_rows else None,
        },
    }

    # Add instrument metadata to extension metadata
    full_metadata = {
        "instrument_id": ohlc.get("instrument_id"),
        "primary_ticker": ohlc.get("primary_ticker"),
        **metadata,
    }

    return extended_ohlc, full_metadata
