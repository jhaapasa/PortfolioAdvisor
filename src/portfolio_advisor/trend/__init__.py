"""Trend analysis modules for adaptive trend and risk analysis."""

from .boundary import (
    BoundaryStabilizer,
    ForecastModel,
    ForecastStrategy,
    GaussianProcessForecaster,
    LinearForecaster,
    OutlierDetector,
    StabilizationConfig,
)
from .l1_filter import L1TrendFilter, TrendResult, extract_l1_trend, to_trend_json

__all__ = [
    # Boundary stabilization (Module 1)
    "BoundaryStabilizer",
    "ForecastModel",
    "ForecastStrategy",
    "GaussianProcessForecaster",
    "LinearForecaster",
    "OutlierDetector",
    "StabilizationConfig",
    # L1 Trend filtering (Module 2)
    "L1TrendFilter",
    "TrendResult",
    "extract_l1_trend",
    "to_trend_json",
]
