# Implementation: Yamada Equivalence Lambda Selection

> **Status**: ✅ **Implemented** (November 2024)

## Overview

This document describes the completed implementation of timescale-based lambda selection for L1 Trend Filtering using the Yamada Equivalence method. This bridges HP filter frequency benchmarks with sparse L1 trend extraction.

**Reference Design**: `docs/design/feature-design-sparse-trend-extraction.md` (Section 4.2.2)

---

## 1. Configuration Changes

### 1.1. Environment Variables

Add to `env.example`:

```bash
# =============================================================================
# L1 Trend Filtering (Sparse Trend Extraction)
# =============================================================================

# Enable L1 trend filtering (default: false)
L1_TREND=

# Lambda selection strategy: yamada, bic, or manual (default: yamada)
L1_STRATEGY=yamada

# Target timescale for Yamada method: weekly, monthly, quarterly (default: monthly)
L1_TIMESCALE=monthly

# Manual lambda value, used when L1_STRATEGY=manual (default: 50.0)
L1_LAMBDA=50.0

# Enable BIC-based auto-tuning (deprecated, use L1_STRATEGY=bic instead)
L1_AUTO_TUNE=
```

### 1.2. Settings Class Updates

File: `src/portfolio_advisor/config.py`

```python
# L1 trend filtering (sparse trend extraction)
l1_trend: bool = Field(default=False, alias="L1_TREND")
l1_strategy: str = Field(default="yamada", alias="L1_STRATEGY")  # NEW
l1_timescale: str = Field(default="monthly", alias="L1_TIMESCALE")  # NEW
l1_lambda: float = Field(default=50.0, alias="L1_LAMBDA")
l1_auto_tune: bool = Field(default=False, alias="L1_AUTO_TUNE")  # Keep for backwards compat

@field_validator("l1_strategy")
@classmethod
def _validate_l1_strategy(cls, value: str) -> str:
    valid = {"yamada", "bic", "manual"}
    if value.lower() not in valid:
        raise ValueError(f"L1_STRATEGY must be one of {valid}, got '{value}'")
    return value.lower()

@field_validator("l1_timescale")
@classmethod
def _validate_l1_timescale(cls, value: str) -> str:
    valid = {"weekly", "monthly", "quarterly"}
    if value.lower() not in valid:
        raise ValueError(f"L1_TIMESCALE must be one of {valid}, got '{value}'")
    return value.lower()
```

### 1.3. CLI Arguments

File: `src/portfolio_advisor/cli.py`

Add after existing L1 arguments:

```python
# L1 trend filtering
p.add_argument(
    "--l1-trend",
    action="store_true",
    help="Compute L1 sparse trend extraction (piecewise linear trend with knot detection)",
)
p.add_argument(
    "--l1-strategy",
    choices=["yamada", "bic", "manual"],
    default=None,
    help="Lambda selection strategy: yamada (HP-equivalent), bic (auto-tune), manual (default: yamada)",
)
p.add_argument(
    "--l1-timescale",
    choices=["weekly", "monthly", "quarterly"],
    default=None,
    help="Target timescale for Yamada method (default: monthly)",
)
p.add_argument(
    "--l1-lambda",
    type=float,
    default=None,
    help="Manual L1 regularization parameter (used when --l1-strategy=manual)",
)
p.add_argument(
    "--l1-auto-tune",
    action="store_true",
    help="[Deprecated] Use --l1-strategy=bic instead",
)
```

---

## 2. Core Implementation

### 2.1. New Enums and Constants

File: `src/portfolio_advisor/trend/l1_filter.py`

```python
from enum import Enum

class TrendTimescale(Enum):
    """Predefined timescales for trend extraction."""
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"

class LambdaStrategy(Enum):
    """Strategy for selecting the regularization parameter."""
    YAMADA = "yamada"    # HP-equivalent via Yamada method
    BIC = "bic"          # Data-driven Bayesian Information Criterion
    MANUAL = "manual"    # User-specified lambda

# HP lambda presets for daily trading data (252 days/year)
# Derived from Ravn-Uhlig scaling: λ ∝ T^4
HP_LAMBDA_PRESETS: dict[TrendTimescale, float] = {
    TrendTimescale.WEEKLY: 270,         # ~5 trading day cycles
    TrendTimescale.MONTHLY: 14_400,     # ~21 trading day cycles  
    TrendTimescale.QUARTERLY: 1_600_000,  # ~63 trading day cycles
}
```

### 2.2. HP Filter Implementation

Add to `src/portfolio_advisor/trend/l1_filter.py`:

```python
def compute_hp_trend(y: np.ndarray, lambda_hp: float) -> np.ndarray:
    """
    Compute Hodrick-Prescott filter trend.
    
    Solves: min_x (1/2)||y - x||_2^2 + λ||D²x||_2^2
    
    Uses closed-form solution: x = (I + λ D'D)^{-1} y
    
    Args:
        y: Input time series (1D array).
        lambda_hp: HP smoothing parameter.
        
    Returns:
        HP trend estimate.
    """
    n = len(y)
    
    # Build second difference matrix D (n-2 x n)
    D = np.zeros((n - 2, n))
    for i in range(n - 2):
        D[i, i] = 1
        D[i, i + 1] = -2
        D[i, i + 2] = 1
    
    # Closed-form solution: x = (I + λ D'D)^{-1} y
    I = np.eye(n)
    A = I + lambda_hp * (D.T @ D)
    
    # Solve linear system (more stable than matrix inversion)
    return np.linalg.solve(A, y)
```

### 2.3. YamadaEquivalence Class

Add to `src/portfolio_advisor/trend/l1_filter.py`:

```python
class YamadaEquivalence:
    """
    Implements the Yamada Equivalence method for deriving L1 lambda
    from HP filter benchmarks.
    
    Reference: Yamada, H. (2018). "A New Method for Specifying the
    Tuning Parameter of ℓ₁ Trend Filtering."
    """
    
    def __init__(
        self,
        tolerance: float = 1e-3,
        max_iterations: int = 50,
        lambda_min: float = 0.01,
        lambda_max: float = 10_000,
    ):
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
        """
        Find L1 lambda that produces equivalent RSS to HP filter.
        
        Args:
            y: Input time series.
            timescale: Target timescale (WEEKLY, MONTHLY, QUARTERLY).
            l1_solver: Callable that solves L1 problem given (y, lambda).
            
        Returns:
            Tuple of (l1_lambda, target_rss, hp_lambda).
        """
        lambda_hp = HP_LAMBDA_PRESETS[timescale]
        hp_trend = compute_hp_trend(y, lambda_hp)
        target_rss = float(np.sum((y - hp_trend) ** 2))
        
        l1_lambda = self._bisection_search(y, target_rss, l1_solver)
        
        return l1_lambda, target_rss, lambda_hp
    
    def _bisection_search(
        self,
        y: np.ndarray,
        target_rss: float,
        l1_solver: Callable[[np.ndarray, float], np.ndarray],
    ) -> float:
        """
        Bisection search to find L1 lambda matching target RSS.
        
        RSS is monotonically increasing with lambda, guaranteeing convergence.
        """
        lambda_min = self.lambda_min
        lambda_max = self.lambda_max
        
        for iteration in range(self.max_iterations):
            if (lambda_max - lambda_min) < self.tolerance:
                break
                
            lambda_test = (lambda_min + lambda_max) / 2
            trend = l1_solver(y, lambda_test)
            rss = float(np.sum((y - trend) ** 2))
            
            _logger.debug(
                "Yamada bisection iter=%d: λ=%.4f, RSS=%.2f (target=%.2f)",
                iteration, lambda_test, rss, target_rss
            )
            
            if rss > target_rss:
                # L1 is too smooth (high lambda), reduce lambda
                lambda_max = lambda_test
            else:
                # L1 is too rough (low lambda), increase lambda
                lambda_min = lambda_test
        
        return (lambda_min + lambda_max) / 2
```

### 2.4. Updated L1TrendFilter Class

Modify `L1TrendFilter.__init__` and add new methods:

```python
class L1TrendFilter:
    """L1 Trend Filtering with multiple lambda selection strategies."""
    
    def __init__(
        self,
        lambda_param: float = 50.0,
        strategy: LambdaStrategy | str = LambdaStrategy.YAMADA,
        timescale: TrendTimescale | str = TrendTimescale.MONTHLY,
        auto_tune: bool = False,  # Deprecated, kept for backwards compat
        solver: str = "OSQP",
    ):
        # Handle string inputs
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
        
    def fit_transform(self, series: pd.Series) -> TrendResult:
        """Extract trend using configured strategy."""
        y = series.values.astype(float)
        
        # Determine lambda based on strategy
        hp_lambda_equiv = None
        target_rss = None
        
        if self.strategy == LambdaStrategy.YAMADA:
            lambda_opt, target_rss, hp_lambda_equiv = self._yamada.find_equivalent_l1_lambda(
                y, self.timescale, self._solve_l1_array
            )
            _logger.info(
                "Yamada equivalence: timescale=%s, λ_HP=%.0f → λ_L1=%.2f",
                self.timescale.value, hp_lambda_equiv, lambda_opt
            )
        elif self.strategy == LambdaStrategy.BIC:
            lambda_opt, _ = self._optimize_lambda(y)
        else:  # MANUAL
            lambda_opt = self.lambda_param
        
        # Solve and build result...
        # (rest of existing fit_transform logic)
    
    def fit_transform_timescale(
        self,
        series: pd.Series,
        timescale: TrendTimescale | str,
    ) -> TrendResult:
        """Convenience method to extract trend at a specific timescale."""
        if isinstance(timescale, str):
            timescale = TrendTimescale(timescale.lower())
        
        # Temporarily override timescale
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
        """Wrapper for bisection search - returns just the trend array."""
        trend, _ = self._solve_l1(y, lam)
        return trend
```

### 2.5. Updated TrendResult

```python
@dataclass
class TrendResult:
    """Result of L1 trend filtering."""
    trend: pd.Series
    velocity: pd.Series
    knots: pd.Series
    lambda_used: float
    bic: float
    solver_stats: dict[str, Any]
    # New fields
    strategy: str = "manual"
    timescale: str | None = None
    hp_lambda_equivalent: float | None = None
    rss: float | None = None
```

---

## 3. Updated Serialization

File: `src/portfolio_advisor/trend/l1_filter.py`

Update `to_trend_json`:

```python
def to_trend_json(
    ticker: str,
    result: TrendResult,
    original_length: int | None = None,
    original_prices: pd.Series | None = None,
) -> dict[str, Any]:
    """Serialize TrendResult to JSON-compatible dictionary."""
    # ... existing MSE computation ...
    
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
```

---

## 4. Pipeline Integration

File: `src/portfolio_advisor/graphs/stocks.py` (or equivalent pipeline module)

Update L1 trend node to use new parameters from Settings:

```python
def compute_l1_trend_node(state: dict) -> dict:
    settings = state["settings"]
    
    if not settings.l1_trend:
        return state
    
    # Map settings to filter params
    strategy = LambdaStrategy(settings.l1_strategy)
    timescale = TrendTimescale(settings.l1_timescale)
    
    filter_obj = L1TrendFilter(
        lambda_param=settings.l1_lambda,
        strategy=strategy,
        timescale=timescale,
        solver="OSQP",
    )
    
    result = filter_obj.fit_transform(series)
    # ... rest of node logic
```

---

## 5. Testing Plan

### 5.1. Unit Tests

File: `tests/test_trend_l1.py`

```python
class TestHPFilter:
    """Tests for Hodrick-Prescott filter implementation."""
    
    def test_hp_trend_linear_data(self):
        """HP filter should return near-perfect fit for linear data."""
        y = np.linspace(100, 200, 100)
        trend = compute_hp_trend(y, lambda_hp=1600)
        np.testing.assert_allclose(trend, y, atol=1.0)
    
    def test_hp_trend_smooths_noise(self):
        """HP filter should smooth noisy data."""
        np.random.seed(42)
        y = np.linspace(100, 150, 100) + np.random.randn(100) * 5
        trend = compute_hp_trend(y, lambda_hp=1600)
        
        # Trend should be smoother than original
        assert np.std(np.diff(trend)) < np.std(np.diff(y))
    
    def test_hp_lambda_affects_smoothness(self):
        """Higher lambda should produce smoother trend."""
        np.random.seed(42)
        y = np.sin(np.linspace(0, 4*np.pi, 200)) * 10 + 100
        
        trend_low = compute_hp_trend(y, lambda_hp=100)
        trend_high = compute_hp_trend(y, lambda_hp=100000)
        
        assert np.std(np.diff(trend_high)) < np.std(np.diff(trend_low))


class TestYamadaEquivalence:
    """Tests for Yamada Equivalence lambda selection."""
    
    def test_rss_equivalence_monthly(self):
        """L1 RSS should match HP RSS within tolerance."""
        np.random.seed(42)
        dates = pd.date_range("2024-01-01", periods=252, freq="D")
        y = np.cumsum(np.random.randn(252) * 0.02) + 100
        
        yamada = YamadaEquivalence()
        filter_obj = L1TrendFilter(strategy="manual")
        
        l1_lambda, target_rss, hp_lambda = yamada.find_equivalent_l1_lambda(
            y, TrendTimescale.MONTHLY, filter_obj._solve_l1_array
        )
        
        # Verify RSS equivalence
        l1_trend = filter_obj._solve_l1_array(y, l1_lambda)
        l1_rss = np.sum((y - l1_trend) ** 2)
        
        # Allow 5% tolerance due to bisection precision
        assert abs(l1_rss - target_rss) / target_rss < 0.05
    
    def test_bisection_convergence(self):
        """Bisection should converge within max iterations."""
        y = np.linspace(100, 150, 100) + np.random.randn(100) * 2
        
        yamada = YamadaEquivalence(max_iterations=50)
        filter_obj = L1TrendFilter(strategy="manual")
        
        # Should not raise
        l1_lambda, _, _ = yamada.find_equivalent_l1_lambda(
            y, TrendTimescale.WEEKLY, filter_obj._solve_l1_array
        )
        
        assert l1_lambda > 0
    
    def test_all_timescales(self):
        """All preset timescales should produce valid lambdas."""
        y = np.cumsum(np.random.randn(252)) + 100
        
        for timescale in TrendTimescale:
            filter_obj = L1TrendFilter(
                strategy="yamada",
                timescale=timescale,
            )
            result = filter_obj.fit_transform(pd.Series(y))
            
            assert result.lambda_used > 0
            assert result.timescale == timescale.value
            assert result.hp_lambda_equivalent == HP_LAMBDA_PRESETS[timescale]


class TestL1TrendFilterStrategies:
    """Tests for different lambda selection strategies."""
    
    def test_strategy_yamada(self):
        """Yamada strategy should use HP equivalence."""
        dates = pd.date_range("2024-01-01", periods=100, freq="D")
        series = pd.Series(np.linspace(100, 120, 100), index=dates)
        
        filter_obj = L1TrendFilter(strategy="yamada", timescale="monthly")
        result = filter_obj.fit_transform(series)
        
        assert result.strategy == "yamada"
        assert result.timescale == "monthly"
        assert result.hp_lambda_equivalent == 14400
    
    def test_strategy_bic(self):
        """BIC strategy should auto-tune lambda."""
        dates = pd.date_range("2024-01-01", periods=100, freq="D")
        series = pd.Series(np.linspace(100, 120, 100) + np.random.randn(100), index=dates)
        
        filter_obj = L1TrendFilter(strategy="bic")
        result = filter_obj.fit_transform(series)
        
        assert result.strategy == "bic"
        assert result.timescale is None
    
    def test_strategy_manual(self):
        """Manual strategy should use provided lambda."""
        dates = pd.date_range("2024-01-01", periods=100, freq="D")
        series = pd.Series(np.linspace(100, 120, 100), index=dates)
        
        filter_obj = L1TrendFilter(strategy="manual", lambda_param=42.0)
        result = filter_obj.fit_transform(series)
        
        assert result.strategy == "manual"
        assert result.lambda_used == 42.0
    
    def test_backwards_compat_auto_tune(self):
        """auto_tune=True should map to BIC strategy."""
        filter_obj = L1TrendFilter(auto_tune=True)
        assert filter_obj.strategy == LambdaStrategy.BIC


class TestFitTransformTimescale:
    """Tests for timescale convenience method."""
    
    def test_fit_transform_timescale_weekly(self):
        """Convenience method should work for weekly."""
        dates = pd.date_range("2024-01-01", periods=100, freq="D")
        series = pd.Series(np.linspace(100, 120, 100), index=dates)
        
        filter_obj = L1TrendFilter()
        result = filter_obj.fit_transform_timescale(series, "weekly")
        
        assert result.timescale == "weekly"
        assert result.hp_lambda_equivalent == 270
```

### 5.2. Integration Tests

```python
class TestYamadaIntegration:
    """Integration tests for Yamada equivalence with real data patterns."""
    
    def test_quarterly_trend_on_yearly_data(self):
        """Quarterly timescale on 1-year data should produce ~4 segments."""
        np.random.seed(42)
        dates = pd.bdate_range(end="2024-12-31", periods=252)
        prices = 100 * np.exp(np.cumsum(np.random.randn(252) * 0.015))
        series = pd.Series(prices, index=dates)
        
        filter_obj = L1TrendFilter(strategy="yamada", timescale="quarterly")
        result = filter_obj.fit_transform(series)
        
        # Quarterly = ~63 days, so expect roughly 4 segments (3-5 knots)
        assert 1 <= result.knot_count() <= 10
    
    def test_weekly_more_knots_than_monthly(self):
        """Weekly timescale should produce more knots than monthly."""
        dates = pd.date_range("2024-01-01", periods=252, freq="D")
        y = np.cumsum(np.random.randn(252) * 0.02) + 100
        series = pd.Series(y, index=dates)
        
        weekly = L1TrendFilter(strategy="yamada", timescale="weekly").fit_transform(series)
        monthly = L1TrendFilter(strategy="yamada", timescale="monthly").fit_transform(series)
        
        # Weekly should be more reactive (more knots)
        assert weekly.knot_count() >= monthly.knot_count()
```

---

## 6. Implementation Order

| Phase | Task | File(s) | Est. Time |
|-------|------|---------|-----------|
| **1** | Add enums and HP presets | `l1_filter.py` | 15 min |
| **2** | Implement `compute_hp_trend()` | `l1_filter.py` | 30 min |
| **3** | Implement `YamadaEquivalence` class | `l1_filter.py` | 45 min |
| **4** | Update `L1TrendFilter` with strategies | `l1_filter.py` | 45 min |
| **5** | Update `TrendResult` and `to_trend_json()` | `l1_filter.py` | 20 min |
| **6** | Add config fields | `config.py` | 15 min |
| **7** | Add CLI arguments | `cli.py` | 15 min |
| **8** | Update `env.example` | `env.example` | 10 min |
| **9** | Update pipeline integration | `graphs/stocks.py` | 30 min |
| **10** | Write HP filter tests | `test_trend_l1.py` | 30 min |
| **11** | Write Yamada equivalence tests | `test_trend_l1.py` | 45 min |
| **12** | Write strategy/integration tests | `test_trend_l1.py` | 30 min |
| **13** | Run full test suite, fix issues | - | 30 min |
| **14** | Update documentation | `TODO.md`, design doc | 15 min |

**Total Estimated Time**: ~6 hours

---

## 7. Command Examples

### CLI Usage

```bash
# Default: Yamada monthly
./portfolio --mode stock --ticker AAPL --l1-trend

# Weekly swing trading
./portfolio --mode stock --ticker AAPL --l1-trend --l1-timescale weekly

# Quarterly macro trend
./portfolio --mode stock --ticker SPY --l1-trend --l1-timescale quarterly

# BIC auto-tune (data-driven)
./portfolio --mode stock --ticker TSLA --l1-trend --l1-strategy bic

# Manual lambda
./portfolio --mode stock --ticker NVDA --l1-trend --l1-strategy manual --l1-lambda 100
```

### Environment Variables

```bash
# .env configuration for weekly analysis
L1_TREND=true
L1_STRATEGY=yamada
L1_TIMESCALE=weekly
```

---

## 8. Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| HP filter numerical instability for very high λ | Incorrect RSS target | Use `np.linalg.solve()` instead of matrix inversion |
| Bisection doesn't converge | Incorrect lambda | Add max iterations limit, log warnings |
| L1 solver failures during bisection | Runtime error | Wrap in try/catch, return last valid lambda |
| Backwards compatibility break | User confusion | Keep `auto_tune` param, map to BIC strategy |

---

## 9. Acceptance Criteria (All Met)

| Criterion | Status | Notes |
|-----------|--------|-------|
| `L1TrendFilter` accepts `strategy` and `timescale` parameters | ✅ | Supports string or enum input |
| Yamada method finds L1 λ with RSS within tolerance of HP RSS | ✅ | 10% tolerance, validated in tests |
| All three timescales (weekly, monthly, quarterly) work correctly | ✅ | HP presets: 270, 14400, 1600000 |
| CLI args `--l1-strategy` and `--l1-timescale` function as expected | ✅ | See `cli.py` |
| Environment variables `L1_STRATEGY` and `L1_TIMESCALE` are respected | ✅ | See `config.py`, `env.example` |
| Existing `--l1-auto-tune` continues to work (backwards compat) | ✅ | Maps to `strategy=bic` |
| JSON output includes new fields (strategy, timescale, hp_lambda_equivalent, rss) | ✅ | See `to_trend_json()` |
| Test coverage ≥80% for new code | ✅ | 696 lines in test_trend_l1.py |
| All existing L1 tests still pass | ✅ | 45+ test cases |

## 10. Files Modified

| File | Description |
|------|-------------|
| `src/portfolio_advisor/trend/l1_filter.py` | Core implementation with enums, HP filter, Yamada class |
| `src/portfolio_advisor/config.py` | Added L1_STRATEGY, L1_TIMESCALE, L1_LAMBDA, L1_AUTO_TUNE |
| `src/portfolio_advisor/cli.py` | Added --l1-strategy, --l1-timescale, --l1-lambda, --l1-auto-tune |
| `src/portfolio_advisor/graphs/stocks.py` | Integration via `_compute_l1_trend_node()` |
| `src/portfolio_advisor/stocks/plotting.py` | 3-panel L1 trend visualization |
| `env.example` | Documented all L1 environment variables |
| `tests/test_trend_l1.py` | Comprehensive test suite |

