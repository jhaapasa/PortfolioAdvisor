# TODO

Tracked improvements and future work for PortfolioAdvisor, organized by category.

## Capabilities

### Market Comparison & Benchmark Analysis
- [ ] Implement `MarketMetricsService` for centralized metrics computation (returns, Sharpe, volatility, beta)
- [ ] Add `EnsureReferenceFreshNode` and `ComputeReferenceMetricsNode` to portfolio graph
- [ ] Generate standalone market overview report with benchmark performance tables
- [ ] Enhance per-stock reports with beta coefficients and Sharpe comparisons
- [ ] Enhance portfolio reports with risk-adjusted performance vs. benchmarks
- [ ] Add top contributors analysis (stocks driving over/underperformance)

See: `docs/design/feature-design-market-overview-report.md`

### Trend & Risk Analysis Subsystem
- [x] Implement ℓ₁ trend filtering for sparse piecewise-linear trend extraction *(November 2024)*
- [x] Add knot detection for structural trend breaks *(November 2024)*
- [x] Implement Yamada Equivalence for timescale-based lambda selection *(November 2024)*
- [ ] Add Bayesian Online Change Point Detection (BOCPD) for regime shifts
- [ ] Build adaptive risk envelope construction using quantile regression
- [ ] Implement multiscale spectral verification via MODWT decomposition

See: `docs/design/feature-requirements-trend-and-risk-module.md`, `docs/implementation/l1-trend-filtering-implementation.md`

### Advanced Risk Metrics
- [ ] Sortino ratio (downside deviation)
- [ ] Maximum drawdown calculation and visualization
- [ ] CVaR (Conditional Value at Risk)
- [ ] Jensen's alpha (CAPM-based excess return)
- [ ] Conditional beta (separate up-market vs. down-market betas)
- [ ] Tracking error and information ratio

### Market Data Enhancements
- [ ] Fetch live 3-month Treasury yield from FRED API for risk-free rate
- [ ] Add sector ETF benchmarks (XLK, XLF, XLE, XLV, etc.)
- [ ] Track volatility indices (VIX, VXN, RVX)
- [ ] Support intraday data granularity (beyond daily)
- [ ] Add international benchmark expansion (region-specific indices)

### Factor Models
- [ ] Fama-French three-factor model
- [ ] Fama-French five-factor model
- [ ] Carhart four-factor model (momentum)
- [ ] Custom factor model support

### Visualization
- [ ] Rolling beta time series charts
- [ ] Rolling Sharpe ratio visualization
- [ ] Rolling correlation with indices
- [ ] Scatter plots of stock returns vs. market returns with beta line
- [ ] Regime change probability overlays

### Article Extraction
- [ ] Evaluate alternative extraction models for better quality
- [ ] Add extraction quality scoring/validation
- [ ] Support additional extraction backends beyond Ollama
- [ ] Enable extraction by default once quality improves

### Portfolio Features
- [ ] Custom user-defined benchmarks
- [ ] Portfolio rebalancing suggestions
- [ ] Tax-lot tracking integration
- [ ] Multi-account aggregation improvements

## Testability

### Missing Documentation
- [ ] Create `stock-news-7day-report-implementation.md`
- [ ] Create `portfolio-ingestion-implementation.md`
- [ ] Add status badges to design docs (Implemented/Partial/Proposed)

### Test Coverage Gaps
- [ ] Add tests for `stocks/coi_distortion.py` (currently 0% coverage)
- [ ] Increase coverage on `stocks/plotting.py` (currently 80%)
- [ ] Add integration tests for market comparison pipeline
- [ ] Create regression test suite with known-good outputs

### Test Infrastructure
- [ ] Create reusable OHLC test fixtures with various patterns (trending, volatile, flat)
- [ ] Add property-based tests for numerical computations (hypothesis)
- [ ] Create mock Polygon.io responses for offline testing
- [ ] Document test patterns and conventions

### Validation
- [ ] Add numerical accuracy tests against reference implementations
- [ ] Benchmark Sharpe/beta calculations against external tools
- [ ] Validate wavelet outputs against PyWavelets directly

## Optimization

### Computational Performance
- [ ] Implement O(n) primal-dual solver for ℓ₁ trend filtering (tridiagonal optimization) — *Currently using CVXPY+OSQP which is O(n) with sparse matrix support*
- [ ] Add incremental wavelet computation (avoid full recompute on new data)
- [ ] Parallelize stock updates across portfolio (configurable worker count)
- [ ] Cache aligned return series in `MarketMetricsService` to avoid recomputation

### Data Efficiency
- [ ] Implement incremental OHLC updates (append-only vs. full refetch)
- [ ] Add delta-based news index updates
- [ ] Compress historical OHLC data for large histories
- [ ] Lazy-load analysis artifacts only when needed

### LLM Optimization
- [ ] Optimize prompts for token efficiency
- [ ] Add prompt caching for repeated structures
- [ ] Batch multiple small LLM calls where possible
- [ ] Profile and reduce LLM latency in critical paths

### I/O Performance
- [ ] Batch file writes for related artifacts
- [ ] Add async I/O for parallel file operations
- [ ] Profile JSON serialization bottlenecks
- [ ] Consider memory-mapped files for large datasets

### Numerical Stability
- [ ] Add weighting schemes for synthetic/real data transitions in boundary extension
- [ ] Implement spectral leakage mitigation for wavelet boundaries
- [ ] Handle edge cases in beta computation (low variance, short histories)

---

*Last updated: November 30, 2024*

