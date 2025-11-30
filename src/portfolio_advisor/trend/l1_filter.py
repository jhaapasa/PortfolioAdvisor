"""L1 Trend Filtering for sparse trend extraction.

This module implements ℓ1 Trend Filtering to isolate the underlying trend
of a financial time series as a piecewise linear function. The method explicitly
identifies "knots" (points where trend velocity changes) rather than smoothing
them out, making it superior for detecting structural market breaks.

Mathematical formulation:
    minimize (1/2)||y - x||_2^2 + λ||Dx||_1

where D is the second difference matrix.

Lambda Selection Strategies:
    - YAMADA: HP-equivalent via Yamada method (matches HP filter RSS)
    - BIC: Data-driven Bayesian Information Criterion
    - MANUAL: User-specified lambda value
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
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


# =============================================================================
# Enums and Constants
# =============================================================================


class TrendTimescale(Enum):
    """Predefined timescales for trend extraction.

    These correspond to typical trading cycle lengths and map to HP filter
    lambda values calibrated for daily trading data (252 days/year).
    """

    WEEKLY = "weekly"  # ~5 trading day cycles
    MONTHLY = "monthly"  # ~21 trading day cycles
    QUARTERLY = "quarterly"  # ~63 trading day cycles


class LambdaStrategy(Enum):
    """Strategy for selecting the L1 regularization parameter.

    - YAMADA: Derives lambda from HP filter equivalence (recommended for timescale targeting)
    - BIC: Data-driven selection via Bayesian Information Criterion
    - MANUAL: User-specified lambda value
    """

    YAMADA = "yamada"
    BIC = "bic"
    MANUAL = "manual"


# HP lambda presets for daily trading data (252 days/year)
# Derived from Ravn-Uhlig scaling: λ ∝ T^4
# Reference: Ravn, M. O., & Uhlig, H. (2002). "On Adjusting the HP Filter
# for the Frequency of Observations."
HP_LAMBDA_PRESETS: dict[TrendTimescale, float] = {
    TrendTimescale.WEEKLY: 270,  # ~5 trading days
    TrendTimescale.MONTHLY: 14_400,  # ~21 trading days (Ravn-Uhlig monthly)
    TrendTimescale.QUARTERLY: 1_600_000,  # ~63 trading days
}


# =============================================================================
# HP Filter Implementation
# =============================================================================


def compute_hp_trend(y: np.ndarray, lambda_hp: float) -> np.ndarray:
    """Compute Hodrick-Prescott filter trend.

    Solves: min_x (1/2)||y - x||_2^2 + λ||D²x||_2^2

    Uses closed-form solution: x = (I + λ D'D)^{-1} y

    Args:
        y: Input time series (1D array).
        lambda_hp: HP smoothing parameter.

    Returns:
        HP trend estimate as numpy array.
    """
    n = len(y)
    if n < 3:
        raise ValueError(f"HP filter requires at least 3 points, got {n}")

    # Build second difference matrix D (n-2 x n)
    D = np.zeros((n - 2, n))
    for i in range(n - 2):
        D[i, i] = 1
        D[i, i + 1] = -2
        D[i, i + 2] = 1

    # Closed-form solution: x = (I + λ D'D)^{-1} y
    identity = np.eye(n)
    A = identity + lambda_hp * (D.T @ D)

    # Solve linear system (more stable than matrix inversion)
    return np.linalg.solve(A, y)


# =============================================================================
# Yamada Equivalence
# =============================================================================


class YamadaEquivalence:
    """Implements the Yamada Equivalence method for deriving L1 lambda from HP benchmarks.

    The Yamada method calibrates λ_L1 such that the Sum of Squared Residuals (SSR)
    of the L1 filter matches that of an HP filter with known frequency-based λ_HP.
    This creates a bridge between established economic benchmarks and sparse estimation.

    Reference: Yamada, H. (2018). "A New Method for Specifying the Tuning
    Parameter of ℓ₁ Trend Filtering."

    Attributes:
        tolerance: Convergence tolerance for bisection search.
        max_iterations: Maximum bisection iterations.
        lambda_min: Lower bound for lambda search.
        lambda_max: Upper bound for lambda search.
    """

    def __init__(
        self,
        tolerance: float = 1e-3,
        max_iterations: int = 50,
        lambda_min: float = 0.01,
        lambda_max: float = 10_000,
    ):
        """Initialize Yamada equivalence calculator.

        Args:
            tolerance: Convergence tolerance for lambda (default 1e-3).
            max_iterations: Maximum bisection iterations (default 50).
            lambda_min: Lower bound for lambda search (default 0.01).
            lambda_max: Upper bound for lambda search (default 10000).
        """
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        self.lambda_min = lambda_min
        self.lambda_max = lambda_max

    def find_equivalent_l1_lambda(
        self,
        y: np.ndarray,
        timescale: TrendTimescale,
        l1_solver: Callable[[np.ndarray, float], np.ndarray],
    ) -> tuple[float, float, float]:
        """Find L1 lambda that produces equivalent RSS to HP filter.

        Args:
            y: Input time series (1D numpy array).
            timescale: Target timescale (WEEKLY, MONTHLY, QUARTERLY).
            l1_solver: Callable that solves L1 problem given (y, lambda).

        Returns:
            Tuple of (l1_lambda, target_rss, hp_lambda).
        """
        lambda_hp = HP_LAMBDA_PRESETS[timescale]
        hp_trend = compute_hp_trend(y, lambda_hp)
        target_rss = float(np.sum((y - hp_trend) ** 2))

        l1_lambda = self._bisection_search(y, target_rss, l1_solver)

        _logger.debug(
            "Yamada equivalence: timescale=%s, λ_HP=%.0f, target_RSS=%.2f → λ_L1=%.4f",
            timescale.value,
            lambda_hp,
            target_rss,
            l1_lambda,
        )

        return l1_lambda, target_rss, lambda_hp

    def _bisection_search(
        self,
        y: np.ndarray,
        target_rss: float,
        l1_solver: Callable[[np.ndarray, float], np.ndarray],
    ) -> float:
        """Bisection search to find L1 lambda matching target RSS.

        RSS is monotonically increasing with lambda, guaranteeing convergence.

        Args:
            y: Input time series.
            target_rss: Target residual sum of squares from HP filter.
            l1_solver: Callable that solves L1 problem given (y, lambda).

        Returns:
            L1 lambda value that produces RSS close to target_rss.
        """
        lambda_min = self.lambda_min
        lambda_max = self.lambda_max

        # Track last valid result in case of solver failures
        last_valid_lambda = (lambda_min + lambda_max) / 2

        for iteration in range(self.max_iterations):
            if (lambda_max - lambda_min) < self.tolerance:
                break

            lambda_test = (lambda_min + lambda_max) / 2

            try:
                trend = l1_solver(y, lambda_test)
                rss = float(np.sum((y - trend) ** 2))
                last_valid_lambda = lambda_test

                _logger.debug(
                    "Yamada bisection iter=%d: λ=%.4f, RSS=%.2f (target=%.2f)",
                    iteration,
                    lambda_test,
                    rss,
                    target_rss,
                )

                if rss > target_rss:
                    # L1 is too smooth (high lambda produces more residual), reduce lambda
                    lambda_max = lambda_test
                else:
                    # L1 is too rough (low lambda fits too closely), increase lambda
                    lambda_min = lambda_test

            except Exception as e:
                _logger.warning(
                    "Yamada bisection: solver failed at λ=%.4f: %s",
                    lambda_test,
                    e,
                )
                # Try to continue with adjusted bounds
                lambda_max = lambda_test

        return (lambda_min + lambda_max) / 2 if lambda_min < lambda_max else last_valid_lambda


@dataclass
class TrendResult:
    """Result of L1 trend filtering.

    Attributes:
        trend: The extracted piecewise linear trend signal
        velocity: First difference of the trend (slope/rate of change)
        knots: Boolean mask where trend velocity changes (structural breaks)
        lambda_used: The L1 regularization parameter used
        bic: Bayesian Information Criterion value for the fit
        solver_stats: Additional solver statistics
        strategy: Lambda selection strategy used ('yamada', 'bic', 'manual')
        timescale: Target timescale for Yamada method (None for other strategies)
        hp_lambda_equivalent: Corresponding HP lambda (for Yamada method)
        rss: Residual sum of squares
    """

    trend: pd.Series
    velocity: pd.Series
    knots: pd.Series
    lambda_used: float
    bic: float
    solver_stats: dict[str, Any]
    # New fields for Yamada equivalence
    strategy: str = field(default="manual")
    timescale: str | None = field(default=None)
    hp_lambda_equivalent: float | None = field(default=None)
    rss: float | None = field(default=None)

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

    Supports multiple lambda selection strategies:
    - YAMADA: HP-equivalent via Yamada method (recommended for timescale targeting)
    - BIC: Data-driven Bayesian Information Criterion
    - MANUAL: User-specified lambda value

    Parameters:
        lambda_param: Regularization strength for MANUAL strategy
        strategy: Lambda selection strategy (yamada, bic, manual)
        timescale: Target timescale for YAMADA strategy (weekly, monthly, quarterly)
        auto_tune: Deprecated, use strategy='bic' instead
        solver: CVXPY solver to use (default: OSQP)
    """

    # Lambda search range for auto-tuning (BIC strategy)
    _LAMBDA_MIN = 0.1
    _LAMBDA_MAX = 1000.0
    _LAMBDA_GRID_SIZE = 20

    # Supported solvers in order of preference
    _SOLVERS = ["OSQP", "ECOS", "SCS"]

    def __init__(
        self,
        lambda_param: float = 50.0,
        strategy: LambdaStrategy | str = LambdaStrategy.YAMADA,
        timescale: TrendTimescale | str = TrendTimescale.MONTHLY,
        auto_tune: bool = False,
        solver: str = "OSQP",
    ):
        """Initialize L1 trend filter.

        Args:
            lambda_param: Regularization parameter for MANUAL strategy (default 50.0)
            strategy: Lambda selection strategy (default: yamada)
            timescale: Target timescale for Yamada method (default: monthly)
            auto_tune: Deprecated; use strategy='bic' instead
            solver: CVXPY solver name (OSQP, ECOS, or SCS)
        """
        if lambda_param <= 0:
            raise ValueError(f"lambda_param must be positive, got {lambda_param}")

        # Handle string inputs for enums
        if isinstance(strategy, str):
            strategy = LambdaStrategy(strategy.lower())
        if isinstance(timescale, str):
            timescale = TrendTimescale(timescale.lower())

        # Backwards compatibility: auto_tune=True implies BIC strategy
        if auto_tune:
            strategy = LambdaStrategy.BIC

        self.lambda_param = lambda_param
        self.strategy = strategy
        self.timescale = timescale
        self.solver = solver.upper()
        self._yamada = YamadaEquivalence()

        if self.solver not in self._SOLVERS:
            raise ValueError(f"Unknown solver '{solver}'. Supported: {self._SOLVERS}")

    def fit_transform(self, series: pd.Series) -> TrendResult:
        """Extract trend from time series using L1 filtering.

        Uses the configured lambda selection strategy:
        - YAMADA: Derives lambda from HP filter equivalence at target timescale
        - BIC: Searches for optimal lambda via grid search
        - MANUAL: Uses the provided lambda_param

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

        # Determine lambda based on strategy
        hp_lambda_equiv: float | None = None
        target_rss: float | None = None
        timescale_str: str | None = None

        if self.strategy == LambdaStrategy.YAMADA:
            lambda_opt, target_rss, hp_lambda_equiv = self._yamada.find_equivalent_l1_lambda(
                y, self.timescale, self._solve_l1_array
            )
            timescale_str = self.timescale.value
            _logger.info(
                "L1TrendFilter: Yamada equivalence timescale=%s, λ_HP=%.0f → λ_L1=%.2f",
                timescale_str,
                hp_lambda_equiv,
                lambda_opt,
            )
            bic_opt = 0.0  # Will be computed after solving

        elif self.strategy == LambdaStrategy.BIC:
            lambda_opt, bic_opt = self._optimize_lambda(y)
            _logger.info(
                "L1TrendFilter: BIC auto-tuned lambda=%.2f (BIC=%.2f)",
                lambda_opt,
                bic_opt,
            )

        else:  # MANUAL
            lambda_opt = self.lambda_param
            bic_opt = 0.0  # Will be computed after solving

        # Solve L1 trend filtering problem
        trend, solver_stats = self._solve_l1(y, lambda_opt)

        # Compute BIC if not already done (for YAMADA and MANUAL strategies)
        if self.strategy != LambdaStrategy.BIC:
            bic_opt = self._compute_bic(y, trend)

        # Compute RSS if not already done
        if target_rss is None:
            target_rss = float(np.sum((y - trend) ** 2))

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
            strategy=self.strategy.value,
            timescale=timescale_str,
            hp_lambda_equivalent=hp_lambda_equiv,
            rss=target_rss,
        )

    def fit_transform_timescale(
        self,
        series: pd.Series,
        timescale: TrendTimescale | str,
    ) -> TrendResult:
        """Convenience method to extract trend at a specific timescale.

        Temporarily overrides the configured timescale and uses Yamada strategy.

        Args:
            series: Input time series
            timescale: Target timescale (weekly, monthly, quarterly)

        Returns:
            TrendResult with trend at the specified timescale
        """
        if isinstance(timescale, str):
            timescale = TrendTimescale(timescale.lower())

        # Temporarily override settings
        original_timescale = self.timescale
        original_strategy = self.strategy

        self.timescale = timescale
        self.strategy = LambdaStrategy.YAMADA

        try:
            return self.fit_transform(series)
        finally:
            self.timescale = original_timescale
            self.strategy = original_strategy

    def _solve_l1_array(self, y: np.ndarray, lam: float) -> np.ndarray:
        """Solve L1 problem and return just the trend array.

        Wrapper for bisection search in Yamada equivalence.

        Args:
            y: Input signal (1D array)
            lam: Regularization parameter

        Returns:
            Trend array (without solver stats)
        """
        trend, _ = self._solve_l1(y, lam)
        return trend

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
    strategy: LambdaStrategy | str = LambdaStrategy.YAMADA,
    timescale: TrendTimescale | str = TrendTimescale.MONTHLY,
    auto_tune: bool = False,
) -> TrendResult:
    """Convenience function to extract L1 trend from a time series.

    Args:
        series: Input time series (e.g., close prices with DatetimeIndex)
        lambda_param: Regularization parameter for MANUAL strategy
        strategy: Lambda selection strategy (yamada, bic, manual)
        timescale: Target timescale for Yamada method (weekly, monthly, quarterly)
        auto_tune: Deprecated; use strategy='bic' instead

    Returns:
        TrendResult with extracted trend, velocity, and knots
    """
    # Handle backwards compatibility
    if auto_tune:
        strategy = LambdaStrategy.BIC

    if lambda_param is None:
        lambda_param = 50.0  # Default for MANUAL strategy

    filter_obj = L1TrendFilter(
        lambda_param=lambda_param,
        strategy=strategy,
        timescale=timescale,
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
        "lambda_l1": result.lambda_used,
        "strategy": result.strategy,
        "timescale": result.timescale,
        "hp_lambda_equivalent": result.hp_lambda_equivalent,
        "rss": result.rss,
        "bic": result.bic,
        "knots_count": result.knot_count(),
        "current_velocity": float(result.velocity.iloc[-1]) if len(result.velocity) > 0 else 0.0,
        "mse": mse,
        "trend": [{"date": str(idx), "value": float(val)} for idx, val in result.trend.items()],
        "knots": result.knot_dates(),
        "solver_stats": result.solver_stats,
        "metadata": {
            "original_length": original_length,
            "trend_length": len(result.trend),
        },
    }
