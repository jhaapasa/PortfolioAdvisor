# Implementation: L1 Trend Filtering (Sparse Trend Extraction)

> **Status**: ✅ **Implemented** (November 2024)
>
> **Design**: `docs/design/feature-design-sparse-trend-extraction.md`

## Overview

L1 Trend Filtering extracts piecewise linear trends from financial time series by solving a convex optimization problem that penalizes changes in slope (second derivatives). Unlike traditional moving averages or Hodrick-Prescott filters that produce smooth curves, L1 filtering identifies structural breaks ("knots") where the trend velocity changes—making it ideal for detecting market regime changes.

## Key Features

### Lambda Selection Strategies

| Strategy | Description | Use Case |
|----------|-------------|----------|
| **Yamada** | Derives λ from HP filter equivalence | Timescale targeting (weekly/monthly/quarterly) |
| **BIC** | Data-driven Bayesian Information Criterion | Exploratory analysis, unknown timescale |
| **Manual** | User-specified λ value | Research, reproducibility |

### Timescale Presets (Yamada Strategy)

| Timescale | HP Lambda | Trading Days | Interpretation |
|-----------|-----------|--------------|----------------|
| Weekly | 270 | ~5 | Intra-month volatility; swing trading |
| Monthly | 14,400 | ~21 | Standard position trading benchmark |
| Quarterly | 1,600,000 | ~63 | Macro trend; earnings cycles |

### Outputs

- **Trend**: Piecewise linear price trend
- **Velocity**: Slope of each linear segment ($/day)
- **Knots**: Timestamps where trend velocity changes
- **RSS**: Residual sum of squares
- **BIC**: Bayesian Information Criterion value

## Usage

### Command Line

```bash
# Default: Yamada method with monthly timescale
portfolio-advisor --mode stock --ticker AAPL --l1-trend

# Weekly swing trading analysis
portfolio-advisor --mode stock --ticker AAPL --l1-trend --l1-timescale weekly

# Quarterly macro trend
portfolio-advisor --mode stock --ticker SPY --l1-trend --l1-timescale quarterly

# BIC auto-tuning (data-driven)
portfolio-advisor --mode stock --ticker TSLA --l1-trend --l1-strategy bic

# Manual lambda specification
portfolio-advisor --mode stock --ticker NVDA --l1-trend --l1-strategy manual --l1-lambda 100
```

### Environment Variables

```bash
# Enable L1 trend filtering
L1_TREND=true

# Lambda selection strategy: yamada, bic, or manual
L1_STRATEGY=yamada

# Target timescale for Yamada method: weekly, monthly, quarterly
L1_TIMESCALE=monthly

# Manual lambda value (used when L1_STRATEGY=manual)
L1_LAMBDA=50.0

# [Deprecated] Use L1_STRATEGY=bic instead
L1_AUTO_TUNE=false
```

### Python API

```python
from portfolio_advisor.trend.l1_filter import L1TrendFilter, extract_l1_trend

# Using the filter class
filter_obj = L1TrendFilter(
    strategy="yamada",
    timescale="monthly",
)
result = filter_obj.fit_transform(price_series)

print(f"Lambda: {result.lambda_used:.2f}")
print(f"Knots: {result.knot_count()}")
print(f"Current velocity: {result.velocity.iloc[-1]:.2f} $/day")

# Convenience function
result = extract_l1_trend(price_series, strategy="yamada", timescale="weekly")

# Extract at multiple timescales
weekly = filter_obj.fit_transform_timescale(price_series, "weekly")
monthly = filter_obj.fit_transform_timescale(price_series, "monthly")
quarterly = filter_obj.fit_transform_timescale(price_series, "quarterly")
```

## Implementation Details

### Source Files

| File | Description |
|------|-------------|
| `src/portfolio_advisor/trend/l1_filter.py` | Core L1 filter implementation |
| `src/portfolio_advisor/graphs/stocks.py` | Pipeline integration (`_compute_l1_trend_node`) |
| `src/portfolio_advisor/stocks/plotting.py` | 3-panel visualization (`render_l1_trend_chart`) |
| `src/portfolio_advisor/config.py` | Settings (L1_TREND, L1_STRATEGY, etc.) |
| `src/portfolio_advisor/cli.py` | CLI arguments |

### Classes and Functions

#### `TrendTimescale` Enum
Predefined timescales: `WEEKLY`, `MONTHLY`, `QUARTERLY`

#### `LambdaStrategy` Enum
Selection strategies: `YAMADA`, `BIC`, `MANUAL`

#### `L1TrendFilter`
Main filter class with:
- `fit_transform(series)` — Extract trend using configured strategy
- `fit_transform_timescale(series, timescale)` — Extract at specific timescale

#### `YamadaEquivalence`
Implements HP-equivalent lambda derivation via bisection search.

#### `TrendResult`
Dataclass containing:
- `trend`, `velocity`, `knots` — Core output series
- `lambda_used`, `strategy`, `timescale` — Configuration used
- `hp_lambda_equivalent`, `rss`, `bic` — Quality metrics
- `solver_stats` — Solver diagnostics

### Visualization

The L1 trend chart (`candle_ohlcv_2y_l1_trends.png`) contains three panels:

1. **Price & Structural Trend**
   - Gray candlesticks with high-low range
   - Blue piecewise linear trend line
   - Orange diamond markers at knot positions

2. **Trend Velocity (Regime State)**
   - Step plot of velocity ($/day)
   - Green fill for uptrend, red for downtrend
   - Zero reference line

3. **Residuals (Noise)**
   - Bar chart of (price - trend)
   - Standard deviation bands

### Pipeline Integration

The L1 trend computation integrates with the stock analysis pipeline:

```
fetch_primary → compute_returns → compute_volatility → compute_sma
    → compute_boundary_extension → compute_wavelet → compute_l1_trend
    → render_report → commit_metadata
```

The `_compute_l1_trend_node`:
1. Loads OHLC data from the database
2. Optionally uses boundary extension for end-point stabilization
3. Runs L1 filter with configured strategy
4. Truncates back to original length if extended
5. Saves JSON artifact to `analysis/trend_l1.json`

### Boundary Extension Support

When `--enable-boundary-extension` is used, L1 filtering leverages the extended price series to stabilize the trend at the current time point:

1. Load boundary extension metadata
2. Extend the close price series with forecasted values
3. Run L1 filter on extended series
4. Truncate result back to original length

This mitigates end-point bias in the optimization.

## Testing

Tests are in `tests/test_trend_l1.py` with ~700 lines covering:

- Basic trend extraction
- Piecewise linear detection
- Lambda effects on smoothness
- HP filter correctness
- Yamada RSS equivalence
- All timescale presets
- Strategy selection
- Backwards compatibility
- Edge cases (NaN values, short series)

Run tests:
```bash
source .venv/bin/activate
pytest tests/test_trend_l1.py -v
```

## Output Artifacts

### JSON Schema (`analysis/trend_l1.json`)

```json
{
  "ticker": "AAPL",
  "lambda_l1": 42.5,
  "strategy": "yamada",
  "timescale": "monthly",
  "hp_lambda_equivalent": 14400,
  "rss": 1523.45,
  "bic": 850.2,
  "knots_count": 24,
  "current_velocity": 0.75,
  "mse": 3.05,
  "trend": [
    {"date": "2024-01-02", "value": 185.23},
    ...
  ],
  "knots": ["2024-01-15", "2024-03-22", ...],
  "solver_stats": {
    "solver": "OSQP",
    "status": "optimal",
    "iterations": 127
  },
  "metadata": {
    "original_length": 504,
    "trend_length": 504
  },
  "generated_at": "2024-11-30T12:00:00Z",
  "depends_on": ["primary.ohlc_daily"]
}
```

### Chart (`report/candle_ohlcv_2y_l1_trends.png`)

Three-panel PNG visualization (14×10 inches, 150 DPI).

## Dependencies

- `cvxpy>=1.5.0` — Convex optimization framework
- `ecos>=2.0.12` — Fallback solver
- `numpy`, `pandas` — Numerical computation
- `matplotlib` — Visualization

## Performance

- OSQP solver: O(n) per iteration with banded matrix structure
- Yamada bisection: ~30 iterations to converge
- Typical solve time: 50-200ms for 500-point series
- BIC grid search: 20 lambda values tested

## References

- Kim, S. J. et al. (2009). "ℓ₁ Trend Filtering." *SIAM Review*, 51(2), 339-360.
- Yamada, H. (2018). "A New Method for Specifying the Tuning Parameter of ℓ₁ Trend Filtering."
- Ravn, M. O., & Uhlig, H. (2002). "On Adjusting the HP Filter for the Frequency of Observations."

---

*Last updated: November 2024*

