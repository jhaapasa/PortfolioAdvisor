"""L1 Trend Filtering for sparse trend extraction.

This module implements ℓ1 Trend Filtering to isolate the underlying trend
of a financial time series as a piecewise linear function. The method explicitly
identifies "knots" (points where trend velocity changes) rather than smoothing
them out, making it superior for detecting structural market breaks.

Mathematical formulation:
    minimize (1/2)||y - x||_2^2 + λ||Dx||_1

where D is the second difference matrix.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import cvxpy as cp
import numpy as np
import pandas as pd

_logger = logging.getLogger(__name__)

# Relative tolerance for identifying non-zero second differences (knots)
# This is multiplied by the data scale (median absolute value) to get the actual threshold.
# For a $100 stock, this gives a tolerance of ~0.01 (1 cent change in second derivative).
# This filters out numerical noise from the solver while preserving meaningful structural breaks.
_KNOT_RELATIVE_TOLERANCE = 1e-4


@dataclass
class TrendResult:
    """Result of L1 trend filtering.

    Attributes:
        trend: The extracted piecewise linear trend signal
        velocity: First difference of the trend (slope/rate of change)
        knots: Boolean mask where trend velocity changes (structural breaks)
        lambda_used: The regularization parameter used
        bic: Bayesian Information Criterion value for the fit
        solver_stats: Additional solver statistics
    """

    trend: pd.Series
    velocity: pd.Series
    knots: pd.Series
    lambda_used: float
    bic: float
    solver_stats: dict[str, Any]

    def knot_dates(self) -> list[str]:
        """Return list of dates where knots occur."""
        return [str(idx) for idx, is_knot in self.knots.items() if is_knot]

    def knot_count(self) -> int:
        """Return the number of knots."""
        return int(self.knots.sum())


class L1TrendFilter:
    """L1 Trend Filtering using convex optimization.

    Implements the ℓ1 trend filter which finds a piecewise linear trend
    by penalizing the ℓ1 norm of the second difference of the trend.

    Parameters:
        lambda_param: Regularization strength (higher = smoother/fewer knots)
        auto_tune: If True, automatically tune lambda using BIC
        solver: CVXPY solver to use (default: OSQP)
    """

    # Lambda search range for auto-tuning
    _LAMBDA_MIN = 0.1
    _LAMBDA_MAX = 1000.0
    _LAMBDA_GRID_SIZE = 20

    # Supported solvers in order of preference
    _SOLVERS = ["OSQP", "ECOS", "SCS"]

    def __init__(
        self,
        lambda_param: float = 1.0,
        auto_tune: bool = False,
        solver: str = "OSQP",
    ):
        """Initialize L1 trend filter.

        Args:
            lambda_param: Regularization parameter (default 1.0)
            auto_tune: Whether to auto-tune lambda using BIC grid search
            solver: CVXPY solver name (OSQP, ECOS, or SCS)
        """
        if lambda_param <= 0:
            raise ValueError(f"lambda_param must be positive, got {lambda_param}")

        self.lambda_param = lambda_param
        self.auto_tune = auto_tune
        self.solver = solver.upper()

        if self.solver not in self._SOLVERS:
            raise ValueError(f"Unknown solver '{solver}'. Supported: {self._SOLVERS}")

    def fit_transform(self, series: pd.Series) -> TrendResult:
        """Extract trend from time series using L1 filtering.

        Args:
            series: Input time series (e.g., close prices)
                    Must have a monotonic index (DatetimeIndex recommended)

        Returns:
            TrendResult with trend, velocity, knots, and metadata
        """
        if len(series) < 3:
            raise ValueError(f"Series must have at least 3 points, got {len(series)}")

        # Extract values and preserve index
        y = series.values.astype(float)
        index = series.index

        # Handle NaN values by forward filling
        if np.any(~np.isfinite(y)):
            _logger.warning("L1TrendFilter: input contains NaN/inf values, forward filling")
            y = pd.Series(y).ffill().bfill().values

        # Determine lambda (auto-tune or use provided)
        if self.auto_tune:
            lambda_opt, bic_opt = self._optimize_lambda(y)
            _logger.info(
                "L1TrendFilter: auto-tuned lambda=%.2f (BIC=%.2f)",
                lambda_opt,
                bic_opt,
            )
        else:
            lambda_opt = self.lambda_param
            bic_opt = 0.0  # Will be computed after solving

        # Solve L1 trend filtering problem
        trend, solver_stats = self._solve_l1(y, lambda_opt)

        # Compute BIC if not already done
        if not self.auto_tune:
            bic_opt = self._compute_bic(y, trend)

        # Compute velocity (first difference)
        velocity = np.diff(trend, prepend=np.nan)
        velocity[0] = velocity[1] if len(velocity) > 1 else 0.0

        # Identify knots (non-zero second differences)
        # Use a scale-relative tolerance: for $100 stock, tolerance ~1e-4; for $1 stock, ~1e-6
        data_scale = np.median(np.abs(y))
        knot_tolerance = _KNOT_RELATIVE_TOLERANCE * data_scale
        second_diff = np.diff(trend, n=2)
        knots_arr = np.zeros(len(trend), dtype=bool)
        # Knots are at indices 1..n-2 (where second difference is defined)
        knots_arr[1:-1] = np.abs(second_diff) > knot_tolerance

        # Build result
        trend_series = pd.Series(trend, index=index, name="trend")
        velocity_series = pd.Series(velocity, index=index, name="velocity")
        knots_series = pd.Series(knots_arr, index=index, name="knots")

        return TrendResult(
            trend=trend_series,
            velocity=velocity_series,
            knots=knots_series,
            lambda_used=lambda_opt,
            bic=bic_opt,
            solver_stats=solver_stats,
        )

    def _solve_l1(self, y: np.ndarray, lam: float) -> tuple[np.ndarray, dict[str, Any]]:
        """Solve the L1 trend filtering optimization problem.

        Minimizes: (1/2)||y - x||_2^2 + λ||Dx||_1
        where D is the second difference matrix.

        Args:
            y: Input signal (1D array)
            lam: Regularization parameter

        Returns:
            Tuple of (trend array, solver stats dict)
        """
        n = len(y)

        # Define optimization variable
        x = cp.Variable(n)

        # Build second difference matrix D
        # D[i] = x[i] - 2*x[i+1] + x[i+2] for i = 0..n-3
        D = self._build_second_diff_matrix(n)

        # Objective: (1/2)||y - x||_2^2 + λ||Dx||_1
        objective = cp.Minimize(0.5 * cp.sum_squares(y - x) + lam * cp.norm1(D @ x))

        # Solve
        problem = cp.Problem(objective)
        solver_stats = self._solve_with_fallback(problem)

        if problem.status not in ["optimal", "optimal_inaccurate"]:
            raise RuntimeError(f"L1TrendFilter optimization failed: {problem.status}")

        return x.value, solver_stats

    def _build_second_diff_matrix(self, n: int) -> np.ndarray:
        """Build the second difference matrix D.

        The matrix D has shape (n-2, n) and computes:
        (Dx)[i] = x[i] - 2*x[i+1] + x[i+2]

        Args:
            n: Length of the signal

        Returns:
            Second difference matrix of shape (n-2, n)
        """
        if n < 3:
            raise ValueError(f"Cannot build second difference matrix for n={n} < 3")

        # Create banded matrix with [1, -2, 1] pattern
        D = np.zeros((n - 2, n))
        for i in range(n - 2):
            D[i, i] = 1
            D[i, i + 1] = -2
            D[i, i + 2] = 1

        return D

    def _solve_with_fallback(self, problem: cp.Problem) -> dict[str, Any]:
        """Attempt to solve using primary solver with fallbacks.

        Args:
            problem: CVXPY problem to solve

        Returns:
            Dictionary with solver statistics
        """
        # Try primary solver first
        solvers_to_try = [self.solver] + [s for s in self._SOLVERS if s != self.solver]

        last_error = None
        for solver_name in solvers_to_try:
            try:
                solver = getattr(cp, solver_name, None)
                if solver is None:
                    continue

                # Solver-specific settings
                kwargs: dict[str, Any] = {}
                if solver_name == "OSQP":
                    kwargs = {"max_iter": 10000, "eps_abs": 1e-6, "eps_rel": 1e-6}
                elif solver_name == "ECOS":
                    kwargs = {"max_iters": 200}
                elif solver_name == "SCS":
                    kwargs = {"max_iters": 5000}

                problem.solve(solver=solver, **kwargs)

                if problem.status in ["optimal", "optimal_inaccurate"]:
                    _logger.debug(
                        "L1TrendFilter: solved with %s (status=%s, obj=%.4f)",
                        solver_name,
                        problem.status,
                        problem.value or 0.0,
                    )
                    return {
                        "solver": solver_name,
                        "status": problem.status,
                        "objective": float(problem.value) if problem.value else 0.0,
                        "iterations": getattr(problem.solver_stats, "num_iters", None),
                    }

            except Exception as e:
                last_error = e
                _logger.debug("L1TrendFilter: solver %s failed: %s", solver_name, e)
                continue

        # All solvers failed
        raise RuntimeError(f"All solvers failed. Last error: {last_error}") from last_error

    def _compute_bic(self, y: np.ndarray, trend: np.ndarray) -> float:
        """Compute Bayesian Information Criterion for the fit.

        BIC = n * ln(MSE) + ln(n) * df
        where df = number of knots + 2 (accounting for linear trend parameters)

        Args:
            y: Original signal
            trend: Fitted trend

        Returns:
            BIC value (lower is better)
        """
        n = len(y)

        # Compute MSE
        residuals = y - trend
        mse = np.mean(residuals**2)

        # Count degrees of freedom (knots + 2)
        # Use scale-relative tolerance for consistent knot detection
        data_scale = np.median(np.abs(y))
        knot_tolerance = _KNOT_RELATIVE_TOLERANCE * data_scale
        second_diff = np.diff(trend, n=2)
        n_knots = np.sum(np.abs(second_diff) > knot_tolerance)
        df = n_knots + 2

        # BIC formula
        # Add small epsilon to MSE to avoid log(0) for perfect fits
        bic = n * np.log(mse + 1e-10) + np.log(n) * df

        return float(bic)

    def _optimize_lambda(self, y: np.ndarray) -> tuple[float, float]:
        """Find optimal lambda using BIC grid search.

        Args:
            y: Input signal

        Returns:
            Tuple of (optimal_lambda, optimal_bic)
        """
        # Create log-spaced grid of lambda values
        lambdas = np.logspace(
            np.log10(self._LAMBDA_MIN),
            np.log10(self._LAMBDA_MAX),
            self._LAMBDA_GRID_SIZE,
        )

        best_lambda = lambdas[0]
        best_bic = float("inf")

        for lam in lambdas:
            try:
                trend, _ = self._solve_l1(y, lam)
                bic = self._compute_bic(y, trend)

                if bic < best_bic:
                    best_bic = bic
                    best_lambda = lam

                # Count knots using scale-relative tolerance
                data_scale = np.median(np.abs(y))
                knot_tol = _KNOT_RELATIVE_TOLERANCE * data_scale
                _logger.debug(
                    "L1TrendFilter: lambda=%.2f -> BIC=%.2f, knots=%d",
                    lam,
                    bic,
                    np.sum(np.abs(np.diff(trend, n=2)) > knot_tol),
                )

            except Exception as e:
                _logger.debug("L1TrendFilter: lambda=%.2f failed: %s", lam, e)
                continue

        return float(best_lambda), float(best_bic)


def extract_l1_trend(
    series: pd.Series,
    lambda_param: float | None = None,
    auto_tune: bool = True,
) -> TrendResult:
    """Convenience function to extract L1 trend from a time series.

    Args:
        series: Input time series (e.g., close prices with DatetimeIndex)
        lambda_param: Optional regularization parameter (if None, auto-tune)
        auto_tune: Whether to auto-tune lambda (default True if lambda_param is None)

    Returns:
        TrendResult with extracted trend, velocity, and knots
    """
    if lambda_param is None:
        auto_tune = True
        lambda_param = 1.0  # Starting point for auto-tune

    filter_obj = L1TrendFilter(
        lambda_param=lambda_param,
        auto_tune=auto_tune,
    )

    return filter_obj.fit_transform(series)


def to_trend_json(
    ticker: str,
    result: TrendResult,
    original_length: int | None = None,
    original_prices: pd.Series | None = None,
) -> dict[str, Any]:
    """Serialize TrendResult to JSON-compatible dictionary.

    Args:
        ticker: Stock ticker symbol
        result: TrendResult to serialize
        original_length: Original series length before any extension (for metadata)
        original_prices: Original price series for MSE calculation (optional)

    Returns:
        Dictionary suitable for JSON serialization
    """
    # Compute MSE if original prices provided
    mse = None
    if original_prices is not None and len(result.trend) > 0:
        # Align indices for comparison
        common_idx = result.trend.index.intersection(original_prices.index)
        if len(common_idx) > 0:
            aligned_trend = result.trend.loc[common_idx]
            aligned_prices = original_prices.loc[common_idx]
            mse = float(np.mean((aligned_prices.values - aligned_trend.values) ** 2))

    return {
        "ticker": ticker,
        "lambda": result.lambda_used,
        "bic": result.bic,
        "knots_count": result.knot_count(),
        "current_velocity": float(result.velocity.iloc[-1]) if len(result.velocity) > 0 else 0.0,
        "mse": mse,
        "solver": result.solver_stats.get("solver", "unknown"),
        "trend": [{"date": str(idx), "value": float(val)} for idx, val in result.trend.items()],
        "knots": result.knot_dates(),
        "metadata": {
            "original_length": original_length,
            "trend_length": len(result.trend),
            **result.solver_stats,
        },
    }
