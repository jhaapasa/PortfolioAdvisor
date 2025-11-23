# Feature Design: Market Comparison and Benchmark Analysis

Status: Proposed

Last Updated: 2025-11-17

Related:
- `docs/design/stock-analysis-plan.md`
- `docs/design/feature-design-stock-news-7day-report.md`

## Summary

Produce market context data that enriches both per-stock and portfolio-level reports by comparing performance against key market indices and benchmarks, while also generating a dedicated market overview report. This includes:
- Fetching and maintaining reference ticker data for major market indices (SPY, QQQ, IWM, etc.)
- Computing beta coefficients for portfolio stocks relative to market indices
- Computing and comparing Sharpe ratios for portfolio stocks and benchmarks
- Generating comparative performance metrics across multiple time horizons
- Creating a standalone market overview report showing benchmark performance and trends
- Integrating market comparison metrics into existing stock and portfolio reports

**Key architectural decision**: All metrics (returns, Sharpe ratios, volatility, beta) are computed by a single `MarketMetricsService` that caches results to eliminate redundant computation. Nodes become thin orchestrators that call the service and populate state, ensuring consistency and minimizing failure surface area.

The goal is to help users understand whether their portfolio and individual stocks are outperforming/underperforming the broader market on a risk-adjusted basis, how correlated they are with market movements, and which stocks contribute most to portfolio over or underperformance. This data is computed as part of the normal portfolio update flow and surfaces in three places: (1) a dedicated market overview report, (2) enhanced per-stock reports, and (3) enhanced portfolio reports.

## Requirements

### Goals
- Maintain a curated set of reference tickers representing key market segments:
  - Broad market: SPY (S&P 500), VTI (Total Stock Market)
  - Large cap growth: QQQ (Nasdaq 100)
  - Small cap: IWM (Russell 2000)
  - International: EFA (Developed Markets), EEM (Emerging Markets)
  - Bonds: AGG (Aggregate Bond), TLT (Long-term Treasury)
  - Risk-free rate: use a configurable proxy (e.g., 3-month Treasury yield)
- For each portfolio stock, compute:
  - Beta relative to each major index (SPY, QQQ, IWM as defaults)
  - Sharpe ratio for multiple time horizons (3mo, 6mo, 1yr, 2yr)
- For each reference ticker, compute:
  - Returns across time horizons
  - Sharpe ratios for the same horizons
  - Historical volatility
- Generate a standalone market overview report with:
  - Market performance summary table (index returns by horizon)
  - Market risk metrics table (Sharpe ratios, volatility for each index)
  - Market trend assessment and context
- Enrich existing per-stock reports with:
  - Beta coefficients and interpretation (correlation with market indices)
  - Sharpe ratio comparison vs. key benchmarks
  - Risk-adjusted performance assessment
- Enrich portfolio-level reports with:
  - Portfolio-wide risk-adjusted performance vs. key benchmarks (SPY, QQQ, IWM)
  - Identification of stocks contributing most to over/underperformance
  - Portfolio average beta and Sharpe metrics
  - Comparison of portfolio Sharpe vs. benchmark Sharpe ratios
- Reuse existing stock database infrastructure (primary OHLC data, analysis modules)
- Support lazy updates: refresh reference ticker data on the same schedule as portfolio stocks
- Compute market reference data as part of normal portfolio update flow, before per-stock reports

### Future Ideas
- **Volatility indices**: Track VIX (S&P 500), VXN (Nasdaq 100), RVX (Russell 2000) to assess market fear/uncertainty alongside price performance
- **Trend and rate of change analysis**: Compute momentum indicators and trend strength for both individual stocks and reference indices to identify regime changes and trend quality
- **Real-time intraday data**: Move beyond daily granularity for more responsive metrics
- **Sector-specific benchmarks**: Add sector ETFs (XLK for tech, XLF for financials, XLE for energy, etc.) for more targeted comparisons
- **Multi-factor models**: Implement Fama-French three-factor or five-factor models, Carhart four-factor (momentum), or custom factor models
- **Jensen's alpha**: Compute risk-adjusted excess return relative to CAPM to identify true outperformance
- **Conditional beta**: Separate beta calculations for up markets vs. down markets to assess asymmetric risk
- **Downside risk metrics**: Sortino ratio (downside deviation), maximum drawdown, CVaR (Conditional Value at Risk)
- **Tracking error and information ratio**: For comparing to active benchmarks and assessing consistency of outperformance
- **Custom user-defined benchmarks**: Allow users to specify their own reference portfolios or blended indices
- **International benchmark expansion**: Region-specific indices for global portfolio diversification analysis
- **Live risk-free rate**: Fetch real-time 3-month Treasury yield from FRED API or similar source instead of using configured static rate
- **Rolling statistics visualization**: Time series charts of rolling beta, rolling Sharpe, rolling correlation with indices

## User Experience (Report Shape)

Market comparison data appears in three places: a dedicated market overview report, enhanced per-stock reports, and enhanced portfolio reports.

### Standalone Market Overview Report

A dedicated report showing benchmark performance and market context:

```markdown
# Market Overview Report

Generated: {timestamp}
As of: {last_complete_trading_day}

## Market Performance Summary

| Index | Symbol | 3mo | 6mo | 1yr | 2yr |
|-------|--------|-----|-----|-----|-----|
| S&P 500 | SPY | +5.2% | +12.3% | +18.7% | +24.5% |
| Nasdaq 100 | QQQ | +7.1% | +15.8% | +22.4% | +28.9% |
| Russell 2000 | IWM | +3.8% | +9.2% | +14.1% | +19.3% |
| Total Stock Market | VTI | +5.0% | +11.8% | +18.2% | +23.8% |
| Developed Markets | EFA | +3.2% | +8.5% | +13.2% | +17.9% |
| Emerging Markets | EEM | +2.1% | +6.8% | +11.5% | +15.2% |
| Aggregate Bond | AGG | +1.2% | +2.8% | +4.5% | +6.2% |
| 20+ Year Treasury | TLT | +0.8% | +1.9% | +3.2% | +4.8% |

## Market Risk Metrics

| Index | Symbol | Volatility (21d) | Sharpe (1yr) | Sharpe (2yr) |
|-------|--------|------------------|--------------|--------------|
| S&P 500 | SPY | 18.2% | 1.05 | 0.98 |
| Nasdaq 100 | QQQ | 19.8% | 1.18 | 1.15 |
| Russell 2000 | IWM | 24.3% | 0.89 | 0.87 |
| Total Stock Market | VTI | 18.5% | 1.02 | 0.96 |
| Developed Markets | EFA | 16.2% | 0.92 | 0.88 |
| Emerging Markets | EEM | 21.5% | 0.78 | 0.75 |
| Aggregate Bond | AGG | 5.8% | 0.65 | 0.72 |
| 20+ Year Treasury | TLT | 12.4% | 0.42 | 0.48 |

## Market Assessment

### Performance Summary (1yr)
*[Structured/Templated Content]*
- **Equities**: Strong performance led by Nasdaq 100 (+22.4%), outpacing S&P 500 (+18.7%) and Russell 2000 (+14.1%)
- **International**: Developed markets (+13.2%) trailing U.S. indices; emerging markets (+11.5%) lagging further
- **Fixed Income**: Modest gains with aggregate bonds (+4.5%) and treasuries (+3.2%)

### Risk-Adjusted Returns (1yr Sharpe)
*[Structured/Templated Content]*
- **Best**: Nasdaq 100 (1.18), S&P 500 (1.05), Total Market (1.02)
- **Moderate**: Developed Markets (0.92), Russell 2000 (0.89)
- **Lower**: Emerging Markets (0.78), Aggregate Bond (0.65), Long Treasury (0.42)

### Volatility Environment
*[Structured/Templated Content]*
- **Equity volatility elevated**: Russell 2000 (24.3%) highest, followed by Emerging Markets (21.5%)
- **Large cap stability**: S&P 500 (18.2%) and Nasdaq 100 (19.8%) relatively lower
- **Fixed income volatility subdued**: Aggregate Bond (5.8%), Long Treasury (12.4%)

### Market Themes and Context
*[LLM-Generated Analysis]*

The market is exhibiting a clear risk-on posture with pronounced leadership from growth-oriented equities. The significant outperformance of the Nasdaq 100 (+22.4%) relative to the broader S&P 500 (+18.7%) suggests continued investor preference for technology and growth stocks, despite the sector carrying slightly higher volatility (19.8% vs 18.2%).

**Notable Rotation Patterns:**
Small-cap equities (Russell 2000) are underperforming on both absolute returns (+14.1%) and risk-adjusted basis (Sharpe 0.89 vs SPY 1.05), indicating a large-cap bias in the current market environment. This divergence often signals caution among investors who favor the perceived safety and liquidity of mega-cap names.

**Geographic Divergence:**
U.S. equity dominance is striking, with domestic indices outpacing international developed markets by 5+ percentage points. The weak performance of emerging markets (Sharpe 0.78) reflects ongoing concerns about global growth, geopolitical tensions, or dollar strength.

**Fixed Income Messaging:**
The poor risk-adjusted returns in treasuries (Sharpe 0.42) and aggregate bonds (0.65) underscore the challenging environment for fixed income. Long-duration treasuries showing minimal gains (+3.2%) with elevated volatility (12.4%) suggests rising rate expectations or inflation concerns persist.

**Implications:**
The market favors risk assets (equities over bonds), growth over value (QQQ > SPY), and size/quality over breadth (SPY > IWM). This concentration of returns in large-cap growth names may present portfolio concentration risks but also reflects strong fundamental performance in technology and growth sectors.

Notes:
- All returns are total returns including dividends
- Sharpe ratios assume {risk_free_rate}% annual risk-free rate
- Volatility is annualized 21-day historical volatility
- Data as of {last_complete_trading_day}
```

### Per-Stock Report Enhancement

Add a new section to existing per-stock reports:

```markdown
# {TICKER} — Analysis Report

[... existing report sections ...]

## Market Comparison

**Beta Coefficients (1yr)**
- vs. SPY: 1.24 (24% more volatile than S&P 500)
- vs. QQQ: 0.98 (similar volatility to Nasdaq 100)
- vs. IWM: 1.45 (45% more volatile than small caps)

**Risk-Adjusted Performance (1yr)**
| Metric | {TICKER} | SPY | QQQ | IWM |
|--------|----------|-----|-----|-----|
| Return | +24.3% | +18.7% | +22.4% | +14.1% |
| Sharpe Ratio | 1.32 | 1.05 | 1.18 | 0.89 |
| Volatility | 21.5% | 18.2% | 19.8% | 24.3% |

**Assessment**: Outperforming market on risk-adjusted basis. Higher returns than all three benchmarks with moderate correlation to Nasdaq 100 (beta 0.98). Sharpe ratio of 1.32 exceeds S&P 500 (1.05) by 26%.

**Multi-Horizon Sharpe Ratios**
| Horizon | {TICKER} | SPY | QQQ | IWM |
|---------|----------|-----|-----|-----|
| 3mo | 1.45 | 1.12 | 1.28 | 0.95 |
| 6mo | 1.38 | 1.08 | 1.22 | 0.91 |
| 1yr | 1.32 | 1.05 | 1.18 | 0.89 |
| 2yr | 1.28 | 0.98 | 1.15 | 0.87 |

Notes:
- Beta computed using 252-day trailing returns
- Sharpe ratios assume {risk_free_rate}% annual risk-free rate
- All metrics based on daily data
```

### Portfolio-Level Report Enhancement

Add a new section to portfolio-level reports:

```markdown
# Portfolio Analysis Report

[... existing portfolio sections ...]

## Risk-Adjusted Performance vs. Market

**Portfolio Risk Metrics**
- Average Beta (vs. SPY): 1.12
- Average Sharpe Ratio (1yr): 1.15
- Portfolio Sharpe (1yr): 1.18

**Benchmark Comparison (1yr)**
| Metric | Portfolio | SPY | QQQ | IWM |
|--------|-----------|-----|-----|-----|
| Total Return | +21.4% | +18.7% | +22.4% | +14.1% |
| Sharpe Ratio | 1.18 | 1.05 | 1.18 | 0.89 |
| Volatility | 19.8% | 18.2% | 19.8% | 24.3% |

**Assessment**: Portfolio matches Nasdaq 100 risk-adjusted performance (both Sharpe 1.18) while outperforming S&P 500 by 12% on a Sharpe basis. Return of 21.4% achieved with similar volatility to broad market.

**Top Contributors to Outperformance (vs. SPY)**

Stocks contributing positively:
1. {TICKER_1}: +3.2% excess return, Sharpe 1.32 (vs. SPY 1.05)
2. {TICKER_2}: +2.8% excess return, Sharpe 1.28
3. {TICKER_3}: +2.1% excess return, Sharpe 1.22

Stocks underperforming:
1. {TICKER_10}: -1.5% vs. SPY, Sharpe 0.78
2. {TICKER_11}: -0.9% vs. SPY, Sharpe 0.82

**Risk-Adjusted Performance Summary**
- Stocks outperforming SPY (Sharpe basis): 8 of 12 (67%)
- Stocks outperforming QQQ (Sharpe basis): 5 of 12 (42%)
- Stocks with beta > 1.2 (high volatility): 4 of 12 (33%)

Notes:
- Portfolio-level Sharpe computed from aggregate portfolio returns and volatility
- Contribution analysis based on position-weighted excess Sharpe ratios
- All metrics based on trailing 252 trading days
```

## Inputs & Data Constraints

### Reference Tickers
Maintain the following reference tickers in the stock database (configured in `config.py` with role descriptions):
- **SPY**: S&P 500 ETF (primary broad market benchmark)
- **QQQ**: Nasdaq 100 ETF (tech/growth benchmark)
- **IWM**: Russell 2000 ETF (small cap benchmark)
- **VTI**: Total Stock Market ETF (alternative broad market)
- **EFA**: MSCI EAFE ETF (international developed markets)
- **EEM**: MSCI Emerging Markets ETF
- **AGG**: Aggregate Bond ETF
- **TLT**: 20+ Year Treasury ETF

Each reference ticker includes:
- **Symbol**: ticker symbol for data fetching
- **Name**: full name (e.g., "S&P 500")
- **Role**: description of what the ticker represents, used for LLM prompt context and market assessment generation

These tickers are stored and updated using the same mechanisms as portfolio stocks (via `graphs/stocks.py`).

### Risk-Free Rate
- Use a configurable risk-free rate (default: 4.5% annually, or 0.018% daily assuming 252 trading days)
- Future enhancement: fetch live 3-month Treasury yield from a data source
- Store in configuration (`config.py` or settings)

### Portfolio Stock Data
- Reuse existing OHLC data from `output/stocks/tickers/{slug}/primary/ohlc_daily.json`
- Reuse existing returns and volatility analyses where available
- Compute additional metrics (beta, Sharpe) as needed

### Time Horizons
- 3 months (63 trading days)
- 6 months (126 trading days)
- 1 year (252 trading days)
- 2 years (504 trading days)

## Architecture & Flow

### Data Layer (Reuse + Extend)

1. **Reference Ticker Maintenance**
   - Reference tickers configured in `MarketComparisonSettings` with symbol, name, and role
   - On market overview report generation, ensure all reference tickers are up-to-date
   - Use existing `graphs/stocks.py` pipeline to fetch/update OHLC data
   - Store under `output/stocks/tickers/{slug}/` like any other ticker

2. **Market Metrics Service** (in `src/portfolio_advisor/stocks/analysis.py`)
   - **Single centralized service**: `MarketMetricsService` computes all market comparison metrics
   - **Primary API**: `compute_metrics_for_symbol(symbol, slug, horizons)` → returns, Sharpe ratios, volatility
     - Computes all metrics in one pass for any ticker (reference or portfolio stock)
     - Internal caching eliminates redundant computation within a run
   - **Beta computation**: `compute_beta(stock, benchmark)` → beta, R-squared
     - Uses simple linear regression: `beta = cov(stock, market) / var(market)`
     - Requires aligned daily return series for both stock and benchmark
   - **Internal helpers** (private methods):
     - `_compute_sharpe_ratio(returns, risk_free_rate, annualize)` → Sharpe ratio
     - `_compute_daily_returns(ohlc_data)` → log returns
     - `_compute_volatility(returns, window)` → annualized volatility
     - `_align_returns(returns1, returns2)` → aligned return series
   - **Design**: See "Interfaces (Proposed) > Market Metrics Service" section for full implementation details
   - **Usage**: Nodes call service methods rather than implementing calculation logic

3. **Analysis Artifact Storage**
   - Per stock: `output/stocks/tickers/{slug}/analysis/market_comparison.json`
     - Single consolidated artifact with all comparison data (returns, Sharpe, volatility, betas)
     - Computed by `MarketMetricsService` and written by `ComputeStockMarketComparisonsNode`
   - Per report run: `output/portfolio/market_overview_{date}.md` (Markdown report)
   - Per report run: `output/portfolio/market_overview_{date}.json` (consolidated reference ticker metrics)
   - **No separate artifacts** for individual metrics (returns, Sharpe, volatility); all consolidated

### Integration into Portfolio Update Flow

Market comparison data is computed as part of the normal portfolio analysis workflow, before individual stock reports are generated. State is managed via the `MarketContext` object in the main graph state.

**Modified Portfolio Update Flow:**

**Step 0: Initialize State** (NEW - graph start)
- Create `state["market_context"] = MarketContext()`
- All fields initialized to empty/None

**Step 1: Ensure Reference Tickers Are Fresh** (NEW STEP - inserted early)
- Node: `EnsureReferenceFreshNode`
- Input: `MarketComparisonSettings.reference_symbols` from config
- Action: Use `graphs/stocks.py::update_ticker()` for each reference ticker
- Output: Ensures primary OHLC data is up-to-date for all benchmarks
- State: No writes to `market_context` yet

**Step 2: Compute Reference Ticker Metrics** (NEW STEP)
- Node: `ComputeReferenceMetricsNode`
- Input: `MarketComparisonSettings.reference_tickers` from config
- Action: For each reference ticker:
  - Call `MarketMetricsService.compute_metrics_for_symbol(symbol, slug, horizons)`
  - Service loads OHLC data from `output/stocks/tickers/{slug}/primary/ohlc_daily.json`
  - Service computes returns, Sharpe ratios (all horizons), and volatility in one pass
  - Service caches result internally for reuse
- **State Write**: Populates `state["market_context"].reference_metrics`
  - Key: ticker symbol (e.g., "SPY")
  - Value: `ReferenceTickerMetrics` object returned from service
- **Readiness**: After this step, `market_context.is_ready_for_stock_comparisons()` returns True

**Step 3: Generate Market Overview Report** (NEW STEP - early, before portfolio updates)
- Node: `GenerateMarketOverviewReportNode`
- **State Read**: `state["market_context"].reference_metrics`
- Action: 
  - Generate templated performance and risk metrics tables
  - Invoke LLM with reference ticker roles to generate "Market Themes and Context" narrative
  - Write `output/portfolio/market_overview_{date}.md` and `.json`
- **State Write**: `state["market_context"].market_overview_generated = "/path/to/report.md"`
- Output: Standalone market overview report (independent of portfolio stocks)

**Step 4: Ensure Portfolio Tickers Are Fresh** (existing step)
- Input: list of portfolio tickers (from ingestion)
- Use existing update mechanisms
- No `market_context` interaction

**Step 5: Compute Per-Stock Market Comparisons** (NEW STEP - before per-stock reports)
- Node: `ComputeStockMarketComparisonsNode`
- **State Read**: `state["market_context"].reference_metrics`
- Input: list of portfolio stock tickers
- Action: For each portfolio stock:
  - Call `MarketMetricsService.compute_metrics_for_symbol(ticker, slug, horizons)` to get stock metrics (returns, Sharpe, volatility)
  - Service reuses cached reference metrics if available
  - For each benchmark in `MarketComparisonSettings.default_benchmarks`:
    - Call `MarketMetricsService.compute_beta(stock_returns, benchmark_returns)` using 252-day aligned returns
    - Returns beta and R-squared for goodness-of-fit
  - Construct `StockMarketComparison` from service outputs (no redundant computation)
  - Write `output/stocks/tickers/{slug}/analysis/market_comparison.json` (single artifact with all comparison data)
- **State Write**: Populates `state["market_context"].stock_comparisons`
  - Key: stock ticker (e.g., "AAPL")
  - Value: `StockMarketComparison` object
- **Readiness**: After this step, `market_context.is_ready_for_portfolio_metrics()` returns True

**Step 6: Compute Portfolio Market Metrics** (NEW STEP - before portfolio report)
- Node: `ComputePortfolioMarketMetricsNode`
- **State Read**: 
  - `state["market_context"].stock_comparisons`
  - `state["market_context"].reference_metrics`
- Input: portfolio position data (for weighting)
- Action:
  - Compute position-weighted average beta for each benchmark
  - Compute portfolio-level Sharpe ratio (from aggregate portfolio returns and volatility)
  - Compute average stock Sharpe (mean across all stocks)
  - Count stocks outperforming each benchmark (on Sharpe basis)
  - Identify top 5 contributors (position-weighted excess Sharpe vs. SPY)
- **State Write**: `state["market_context"].portfolio_metrics = PortfolioMarketMetrics(...)`

**Step 7: Generate Per-Stock Reports** (existing step, enhanced)
- **State Read**: `state["market_context"].stock_comparisons[ticker]`
- Action:
  - Read `analysis/market_comparison.json` from disk (also in state for convenience)
  - Add "Market Comparison" section to existing stock report
  - Include beta table, risk-adjusted performance table, multi-horizon Sharpe table
  - Use `state["market_context"].reference_metrics` for benchmark data in tables

**Step 8: Generate Portfolio-Level Report** (existing step, enhanced)
- **State Read**: 
  - `state["market_context"].portfolio_metrics`
  - `state["market_context"].reference_metrics` (for benchmark data)
- Action:
  - Add "Risk-Adjusted Performance vs. Market" section to portfolio report
  - Include benchmark comparison table using `portfolio_metrics` and `reference_metrics`
  - Include top contributors list from `portfolio_metrics.top_contributors`
  - Include risk-adjusted summary stats

**Step 9: Cleanup** (automatic)
- LangGraph state is scoped to the run; no explicit cleanup needed
- `MarketContext` is garbage collected when graph execution completes

**No Separate CLI Command**
- Market comparison is automatically computed during normal `pa analyze` or portfolio update commands
- Configuration options (benchmark list, horizons, risk-free rate) are set in `config.py`, not via CLI flags

**State Ownership Summary:**

| Field | Writer Node | Reader Nodes |
|-------|-------------|--------------|
| `reference_metrics` | ComputeReferenceMetricsNode | GenerateMarketOverviewReportNode, ComputeStockMarketComparisonsNode, ComputePortfolioMarketMetricsNode, stock/portfolio report nodes |
| `stock_comparisons` | ComputeStockMarketComparisonsNode | ComputePortfolioMarketMetricsNode, stock report nodes |
| `portfolio_metrics` | ComputePortfolioMarketMetricsNode | Portfolio report node |
| `market_overview_generated` | GenerateMarketOverviewReportNode | (optional: for testing/verification) |

### Graph Integration

Integrate market comparison into the main portfolio analysis graph. Add new nodes to the existing flow:

**New Nodes (early in pipeline):**
- `EnsureReferenceFreshNode`: updates all reference tickers (SPY, QQQ, IWM, etc.) using existing stock update mechanisms
- `ComputeReferenceMetricsNode`: computes returns, Sharpe, volatility for all benchmarks across all horizons; stores in shared state

**New Nodes (before report generation):**
- `ComputeStockMarketComparisonsNode`: for each portfolio stock, computes beta vs. benchmarks and Sharpe across horizons; writes `analysis/market_comparison.json` per stock
- `ComputePortfolioMarketMetricsNode`: aggregates per-stock comparisons; computes portfolio-level beta, Sharpe, and contribution analysis; stores in shared state
- `GenerateMarketOverviewReportNode`: generates standalone market overview report from reference ticker metrics; uses templating for structured tables and summaries; invokes LLM with reference ticker roles from config to generate "Market Themes and Context" narrative analysis; writes `output/portfolio/market_overview_{date}.md` and `.json`

**Enhanced Existing Nodes:**
- Stock report generation nodes: read `analysis/market_comparison.json` and add "Market Comparison" section
- Portfolio report generation nodes: read portfolio-level market metrics from state and add "Risk-Adjusted Performance vs. Market" section

**Integration Point:**
- In `graph.py` or the main portfolio analysis orchestrator
- Reference ticker updates happen early (after ingestion, before stock updates)
- Per-stock market comparisons happen after all stock data is fresh, before report generation

## Interfaces (Proposed)

### Market Metrics Service

A single service that computes all metrics for any ticker (reference or portfolio stock), eliminating redundant computation and centralizing calculation logic.

```python
from pathlib import Path
from typing import Optional
import numpy as np

class MarketMetricsService:
    """
    Centralized service for computing market metrics (returns, Sharpe, volatility, beta).
    
    Design principles:
    - Single entry point for computing metrics for any ticker
    - Internal caching to avoid redundant computation within a run
    - Stateless API (except cache); safe for concurrent use
    - All calculations use same methodology for consistency
    """
    
    def __init__(self, stock_db_root: Path, risk_free_rate_daily: float):
        self.stock_db_root = stock_db_root
        self.risk_free_rate_daily = risk_free_rate_daily
        self._cache: dict[str, ReferenceTickerMetrics] = {}
    
    def compute_metrics_for_symbol(
        self, 
        symbol: str, 
        slug: str, 
        horizons: list[int]
    ) -> ReferenceTickerMetrics:
        """
        Compute returns, Sharpe ratios, and volatility for a ticker.
        
        Args:
            symbol: ticker symbol (e.g., "SPY", "AAPL")
            slug: canonical slug for file paths
            horizons: list of horizons in days (e.g., [63, 126, 252, 504])
        
        Returns:
            ReferenceTickerMetrics with all computed metrics
        
        Caches result internally to avoid redundant computation.
        """
        cache_key = f"{symbol}:{','.join(map(str, horizons))}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        # Load OHLC data
        ohlc_path = self.stock_db_root / "tickers" / slug / "primary" / "ohlc_daily.json"
        ohlc_data = self._load_ohlc(ohlc_path)
        
        # Compute daily log returns
        daily_returns = self._compute_daily_returns(ohlc_data)
        
        # Compute returns for each horizon
        returns = {}
        for horizon in horizons:
            returns[horizon] = self._compute_horizon_return(ohlc_data, horizon)
        
        # Compute Sharpe ratios for each horizon
        sharpe_ratios = {}
        for horizon in horizons:
            sharpe_ratios[horizon] = self._compute_sharpe_ratio(
                daily_returns[-horizon:], 
                self.risk_free_rate_daily,
                annualize=True
            )
        
        # Compute 21-day annualized volatility
        volatility = self._compute_volatility(daily_returns, window=21)
        
        result = ReferenceTickerMetrics(
            symbol=symbol,
            returns=returns,
            sharpe_ratios=sharpe_ratios,
            volatility_annualized=volatility,
            as_of=ohlc_data[-1]["date"]
        )
        
        self._cache[cache_key] = result
        return result
    
    def compute_beta(
        self,
        stock_symbol: str,
        stock_slug: str,
        benchmark_symbol: str,
        benchmark_slug: str,
        window_days: int = 252
    ) -> tuple[float, float]:
        """
        Compute beta and R-squared for stock vs. benchmark.
        
        Args:
            stock_symbol, stock_slug: stock identifier
            benchmark_symbol, benchmark_slug: benchmark identifier
            window_days: lookback window (default 252 = 1 year)
        
        Returns:
            (beta, r_squared)
        
        Beta = cov(stock, benchmark) / var(benchmark)
        R-squared = coefficient of determination from linear regression
        """
        # Get metrics (will use cache if available)
        stock_metrics = self.compute_metrics_for_symbol(stock_symbol, stock_slug, [window_days])
        benchmark_metrics = self.compute_metrics_for_symbol(benchmark_symbol, benchmark_slug, [window_days])
        
        # Load daily returns for alignment
        stock_returns = self._load_daily_returns(stock_slug, window_days)
        benchmark_returns = self._load_daily_returns(benchmark_slug, window_days)
        
        # Align returns (ensure same dates)
        aligned_stock, aligned_benchmark = self._align_returns(stock_returns, benchmark_returns)
        
        # Compute beta via covariance
        covariance = np.cov(aligned_stock, aligned_benchmark)[0, 1]
        benchmark_variance = np.var(aligned_benchmark, ddof=1)
        beta = covariance / benchmark_variance
        
        # Compute R-squared via linear regression
        correlation = np.corrcoef(aligned_stock, aligned_benchmark)[0, 1]
        r_squared = correlation ** 2
        
        return beta, r_squared
    
    def clear_cache(self):
        """Clear internal cache. Call between portfolio runs."""
        self._cache.clear()
    
    # Private helper methods
    def _load_ohlc(self, path: Path) -> list[dict]:
        """Load OHLC data from JSON."""
        # Implementation loads and returns ohlc_daily.json data
        ...
    
    def _compute_daily_returns(self, ohlc_data: list[dict]) -> np.ndarray:
        """Compute daily log returns from OHLC data."""
        closes = np.array([bar["close"] for bar in ohlc_data])
        return np.diff(np.log(closes))
    
    def _compute_horizon_return(self, ohlc_data: list[dict], horizon: int) -> float:
        """Compute total return over horizon (simple return, not annualized)."""
        if len(ohlc_data) < horizon + 1:
            return None
        return (ohlc_data[-1]["close"] / ohlc_data[-horizon - 1]["close"]) - 1.0
    
    def _compute_sharpe_ratio(
        self, 
        returns: np.ndarray, 
        risk_free_rate_daily: float, 
        annualize: bool = True
    ) -> float:
        """
        Compute Sharpe ratio.
        
        Sharpe = (mean(returns) - risk_free_rate) / std(returns)
        If annualize=True, scale by sqrt(252) for daily data.
        """
        excess_returns = returns - risk_free_rate_daily
        mean_excess = np.mean(excess_returns)
        std_excess = np.std(excess_returns, ddof=1)
        sharpe = mean_excess / std_excess if std_excess > 0 else 0.0
        return sharpe * np.sqrt(252) if annualize else sharpe
    
    def _compute_volatility(self, returns: np.ndarray, window: int = 21) -> float:
        """Compute annualized volatility over trailing window."""
        if len(returns) < window:
            return None
        recent_returns = returns[-window:]
        return np.std(recent_returns, ddof=1) * np.sqrt(252)
    
    def _load_daily_returns(self, slug: str, window_days: int) -> np.ndarray:
        """Load and compute daily returns for alignment."""
        # Implementation loads OHLC and computes returns
        ...
    
    def _align_returns(
        self, 
        returns1: np.ndarray, 
        returns2: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Align two return series by date (assumes both have date metadata)."""
        # Implementation aligns based on dates, handles missing data
        # For now, assume both series cover same trading days and just truncate to min length
        min_len = min(len(returns1), len(returns2))
        return returns1[-min_len:], returns2[-min_len:]
```

**Key Benefits:**

1. **Single Source of Truth**: All metrics computed using same methodology
2. **No Redundant Computation**: Internal cache ensures each ticker's metrics computed once per run
3. **Simplified Nodes**: Nodes become thin wrappers that call service and populate state
4. **Testability**: Service can be unit tested independently with mocked OHLC data
5. **Maintainability**: Calculation logic centralized; changes propagate automatically

**Usage Pattern in Nodes:**

```python
# In ComputeReferenceMetricsNode:
service = MarketMetricsService(stock_db_root, risk_free_rate_daily)
for ticker in reference_tickers:
    metrics = service.compute_metrics_for_symbol(ticker.symbol, ticker.slug, horizons)
    state["market_context"].reference_metrics[ticker.symbol] = metrics

# In ComputeStockMarketComparisonsNode:
service = MarketMetricsService(stock_db_root, risk_free_rate_daily)
for stock_ticker in portfolio_tickers:
    stock_metrics = service.compute_metrics_for_symbol(stock_ticker, slug, horizons)
    betas = {}
    for benchmark_symbol in default_benchmarks:
        beta, r_squared = service.compute_beta(stock_ticker, slug, benchmark_symbol, benchmark_slug)
        betas[benchmark_symbol] = beta
    
    comparison = StockMarketComparison(
        ticker=stock_ticker,
        slug=slug,
        betas=betas,
        sharpe_ratios=stock_metrics.sharpe_ratios,
        returns=stock_metrics.returns,
        as_of=stock_metrics.as_of
    )
    state["market_context"].stock_comparisons[stock_ticker] = comparison
```

### State Object: MarketContext

Define a typed state object to carry market comparison data through the LangGraph pipeline. This object is part of the main graph state and follows LangGraph's state management patterns.

```python
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class ReferenceTickerMetrics:
    """Metrics for a single reference ticker."""
    symbol: str
    returns: dict[int, float]  # horizon_days -> return (e.g., {63: 0.052, 126: 0.123, ...})
    sharpe_ratios: dict[int, float]  # horizon_days -> Sharpe ratio
    volatility_annualized: float  # 21-day annualized volatility
    as_of: str  # ISO date

@dataclass
class StockMarketComparison:
    """Market comparison metrics for a single portfolio stock."""
    ticker: str
    slug: str
    betas: dict[str, float]  # benchmark_symbol -> beta (e.g., {"SPY": 1.24, "QQQ": 0.98})
    sharpe_ratios: dict[int, float]  # horizon_days -> Sharpe ratio
    returns: dict[int, float]  # horizon_days -> return
    as_of: str  # ISO date

@dataclass
class PortfolioMarketMetrics:
    """Portfolio-level market comparison metrics."""
    average_beta_vs_benchmarks: dict[str, float]  # benchmark_symbol -> average beta
    portfolio_sharpe: float  # Portfolio-level Sharpe ratio (1yr default)
    average_stock_sharpe: float  # Mean of individual stock Sharpe ratios
    stocks_outperforming: dict[str, int]  # benchmark_symbol -> count of stocks outperforming
    total_stocks: int
    top_contributors: list[dict]  # [{ticker, excess_sharpe, excess_return}, ...]
    as_of: str  # ISO date

@dataclass
class MarketContext:
    """
    State object for market comparison data, carried through the LangGraph pipeline.
    
    Lifecycle:
    - Initialized at graph start (all fields None/empty)
    - Populated progressively by market comparison nodes
    - Read by report generation nodes
    - Cleared/reset between portfolio analysis runs
    
    Ownership:
    - reference_metrics: Written by ComputeReferenceMetricsNode, read by GenerateMarketOverviewReportNode and stock/portfolio comparison nodes
    - stock_comparisons: Written by ComputeStockMarketComparisonsNode, read by stock report nodes and ComputePortfolioMarketMetricsNode
    - portfolio_metrics: Written by ComputePortfolioMarketMetricsNode, read by portfolio report node
    - market_overview_generated: Written by GenerateMarketOverviewReportNode (path to generated report)
    """
    
    # Reference ticker metrics (all benchmarks)
    reference_metrics: dict[str, ReferenceTickerMetrics] = field(default_factory=dict)
    # Key: ticker symbol (e.g., "SPY"), Value: ReferenceTickerMetrics
    
    # Per-stock market comparisons (portfolio stocks only)
    stock_comparisons: dict[str, StockMarketComparison] = field(default_factory=dict)
    # Key: stock ticker (e.g., "AAPL"), Value: StockMarketComparison
    
    # Portfolio-level aggregated metrics
    portfolio_metrics: Optional[PortfolioMarketMetrics] = None
    
    # Path to generated market overview report (for tracking/testing)
    market_overview_generated: Optional[str] = None
    
    def is_ready_for_stock_comparisons(self) -> bool:
        """Check if reference metrics are available for stock comparisons."""
        return len(self.reference_metrics) > 0
    
    def is_ready_for_portfolio_metrics(self) -> bool:
        """Check if stock comparisons are available for portfolio aggregation."""
        return len(self.stock_comparisons) > 0
    
    def get_reference_metric(self, symbol: str) -> Optional[ReferenceTickerMetrics]:
        """Safely retrieve reference metric by symbol."""
        return self.reference_metrics.get(symbol)
    
    def get_stock_comparison(self, ticker: str) -> Optional[StockMarketComparison]:
        """Safely retrieve stock comparison by ticker."""
        return self.stock_comparisons.get(ticker)
```

**Error Handling and Validation:**

- **Missing Reference Data**: If `ComputeReferenceMetricsNode` fails to compute metrics for a reference ticker (e.g., insufficient data), that ticker is skipped and a warning is logged. Downstream nodes check `market_context.get_reference_metric(symbol)` and handle `None` gracefully.
- **Missing Stock Comparisons**: If beta or Sharpe computation fails for a stock (e.g., <252 days of data), `ComputeStockMarketComparisonsNode` writes a partial `StockMarketComparison` with available fields and marks unavailable metrics as `None`. Stock reports note "Insufficient data" for missing metrics.
- **Empty State**: If `reference_metrics` is empty after Step 2, the market overview report node logs an error and skips report generation. Per-stock and portfolio comparisons also skip gracefully.
- **State Consistency**: Each node validates expected state fields before proceeding. Example: `ComputeStockMarketComparisonsNode` checks `market_context.is_ready_for_stock_comparisons()` at the start.

**State Mutation Rules:**

- Nodes MUST NOT modify existing entries in `reference_metrics` or `stock_comparisons` after writing them
- Nodes MAY add new entries to dictionaries (e.g., adding a new stock comparison)
- The `portfolio_metrics` field is written exactly once by `ComputePortfolioMarketMetricsNode`
- LangGraph's state update mechanism ensures thread-safety if nodes run concurrently (though current design is sequential)

**Integration with Main Graph State:**

The `MarketContext` object is added to the main portfolio analysis graph state (e.g., in `src/portfolio_advisor/graph.py`):

```python
from typing_extensions import TypedDict

class PortfolioAnalysisState(TypedDict):
    # ... existing state fields ...
    market_context: MarketContext
```

**Lifecycle and Guarantees:**

1. **Initialization**: `MarketContext()` is created at the start of each portfolio analysis run
2. **Progressive Population**:
   - Step 2 (ComputeReferenceMetricsNode): populates `reference_metrics`
   - Step 4 (ComputeStockMarketComparisonsNode): populates `stock_comparisons`
   - Step 5 (ComputePortfolioMarketMetricsNode): populates `portfolio_metrics`
   - Step 6 (GenerateMarketOverviewReportNode): populates `market_overview_generated`
3. **Read Access**: Downstream nodes read from `state["market_context"]` without modification
4. **Cleanup**: State is cleared at the end of each portfolio analysis run (automatic via LangGraph state scoping)
5. **Persistence**: Not persisted across runs; computed fresh each time (reference ticker data comes from file-based stock database)

### Configuration

Add to `src/portfolio_advisor/config.py`:

```python
from dataclasses import dataclass

@dataclass
class ReferenceTicker:
    """Reference ticker configuration with role description for LLM context."""
    symbol: str
    name: str
    role: str  # Description for LLM prompts and market assessment

class MarketComparisonSettings:
    """Configuration for market comparison and benchmark analysis."""
    
    # Reference tickers with roles for LLM prompt inclusion
    reference_tickers: list[ReferenceTicker] = [
        ReferenceTicker(
            symbol="SPY",
            name="S&P 500",
            role="Primary broad U.S. large-cap equity benchmark"
        ),
        ReferenceTicker(
            symbol="QQQ",
            name="Nasdaq 100",
            role="U.S. large-cap growth and technology benchmark"
        ),
        ReferenceTicker(
            symbol="IWM",
            name="Russell 2000",
            role="U.S. small-cap equity benchmark"
        ),
        ReferenceTicker(
            symbol="VTI",
            name="Total Stock Market",
            role="Broad U.S. total market equity benchmark"
        ),
        ReferenceTicker(
            symbol="EFA",
            name="MSCI EAFE",
            role="International developed markets equity benchmark"
        ),
        ReferenceTicker(
            symbol="EEM",
            name="MSCI Emerging Markets",
            role="Emerging markets equity benchmark"
        ),
        ReferenceTicker(
            symbol="AGG",
            name="Aggregate Bond",
            role="U.S. investment-grade bond benchmark"
        ),
        ReferenceTicker(
            symbol="TLT",
            name="20+ Year Treasury",
            role="U.S. long-duration treasury bond benchmark"
        ),
    ]
    
    # Default benchmarks for beta calculations (subset of reference_tickers)
    default_benchmarks: list[str] = ["SPY", "QQQ", "IWM"]
    
    # Time horizons for performance and Sharpe ratio calculations
    time_horizons_days: list[int] = [63, 126, 252, 504]  # 3mo, 6mo, 1yr, 2yr
    
    # Risk-free rate (configurable; future: fetch from FRED API)
    risk_free_rate_annual: float = 0.045  # 4.5%
    risk_free_rate_daily: float = 0.045 / 252  # ~0.000178
    
    @property
    def reference_symbols(self) -> list[str]:
        """Extract just the ticker symbols for data fetching."""
        return [ticker.symbol for ticker in self.reference_tickers]
    
    @property
    def benchmark_roles(self) -> dict[str, str]:
        """Map of ticker symbol to role description for LLM prompts."""
        return {ticker.symbol: ticker.role for ticker in self.reference_tickers}
```

**Usage in LLM Prompts:**

The `role` field provides context for LLM-based market assessment. When generating the market overview report's "Market Themes and Context" section, the LLM is prompted with:

```
You are analyzing market performance across the following benchmarks:
- SPY (S&P 500): Primary broad U.S. large-cap equity benchmark
- QQQ (Nasdaq 100): U.S. large-cap growth and technology benchmark
- IWM (Russell 2000): U.S. small-cap equity benchmark
- VTI (Total Stock Market): Broad U.S. total market equity benchmark
- EFA (MSCI EAFE): International developed markets equity benchmark
- EEM (MSCI Emerging Markets): Emerging markets equity benchmark
- AGG (Aggregate Bond): U.S. investment-grade bond benchmark
- TLT (20+ Year Treasury): U.S. long-duration treasury bond benchmark

Performance data (1yr):
{structured data for returns, Sharpe ratios, volatility}

Generate a narrative analysis highlighting:
1. Notable rotation patterns or leadership themes (e.g., growth vs value, large vs small cap)
2. Geographic divergence (domestic vs international performance)
3. Asset class trends (equities vs fixed income, risk-on vs risk-off)
4. Volatility regime and what it suggests about market sentiment
5. Implications for portfolio positioning

Be concise, insightful, and data-driven. Focus on actionable themes rather than simply restating the numbers.
```

This ensures the LLM understands what each ticker represents and can provide meaningful, contextual interpretation beyond what the templated summaries provide.

### Analysis Artifact: `analysis/market_comparison.json`

Per stock:

```json
{
  "ticker": "AAPL",
  "slug": "cid-stocks-us-composite-aapl",
  "as_of": "2025-11-17",
  "betas": {
    "SPY": {
      "value": 1.24,
      "window_days": 252,
      "r_squared": 0.78
    },
    "QQQ": {
      "value": 0.98,
      "window_days": 252,
      "r_squared": 0.82
    },
    "IWM": {
      "value": 1.45,
      "window_days": 252,
      "r_squared": 0.65
    }
  },
  "sharpe_ratios": {
    "63": 1.45,
    "126": 1.38,
    "252": 1.32,
    "504": 1.28
  },
  "returns": {
    "63": 0.087,
    "126": 0.142,
    "252": 0.243,
    "504": 0.389
  },
  "volatility_annualized": 0.24,
  "depends_on": ["primary.ohlc_daily", "analysis.returns"],
  "generated_at": "2025-11-17T10:30:00Z"
}
```

### Optional Testing Artifact: `market_metrics_{date}.json`

For testing and debugging purposes, an optional portfolio-level summary can be written:

```json
{
  "as_of": "2025-11-17",
  "reference_tickers": {
    "SPY": {
      "returns": {"63": 0.052, "126": 0.123, "252": 0.187, "504": 0.245},
      "sharpe_ratios": {"63": 1.12, "126": 1.08, "252": 1.05, "504": 0.98},
      "volatility": 0.182
    }
    // ... other reference tickers
  },
  "portfolio_stocks": ["AAPL", "MSFT", ...],
  "portfolio_summary": {
    "average_beta_vs_SPY": 1.12,
    "portfolio_sharpe_252d": 1.18,
    "average_stock_sharpe_252d": 1.15,
    "stocks_outperforming_SPY_252d": 8,
    "total_stocks": 12,
    "top_contributors": [
      {"ticker": "AAPL", "excess_sharpe": 0.27, "excess_return": 0.032},
      {"ticker": "MSFT", "excess_sharpe": 0.23, "excess_return": 0.028}
    ]
  },
  "generated_at": "2025-11-17T10:30:00Z"
}
```

Note: This artifact is optional and primarily for validation. The primary outputs are the enhanced stock and portfolio reports.

## Storage & Outputs

### Market Overview Artifacts
- `output/portfolio/market_overview_{date}.md`: Standalone market overview report with benchmark performance and risk metrics
- `output/portfolio/market_overview_{date}.json`: Structured reference ticker metrics (computed once by `MarketMetricsService`, used by all reports)

### Per-Stock Artifacts
- `output/stocks/tickers/{slug}/analysis/market_comparison.json`: **Single consolidated artifact** containing:
  - Stock metrics (returns, Sharpe ratios, volatility) computed by `MarketMetricsService`
  - Beta coefficients vs. each benchmark (SPY, QQQ, IWM)
  - R-squared for each beta calculation
  - All data needed for "Market Comparison" section in stock reports
- Per-stock reports enhanced with "Market Comparison" section (reads from above artifact)

### Portfolio-Level Artifacts
- Portfolio reports enhanced with "Risk-Adjusted Performance vs. Market" section (aggregates from `MarketContext.portfolio_metrics`)
- No separate JSON artifact needed; all data in `MarketContext` state during run

**Note on Artifact Consolidation:**

Previously, the design had separate artifacts for different metrics (returns.json, sharpe.json, etc.). Now:
- `MarketMetricsService` computes all metrics in one pass
- Single `market_comparison.json` per stock contains all comparison data
- `market_overview_{date}.json` contains all reference ticker metrics
- No duplication; downstream nodes read from these canonical sources or from `MarketContext` state

## Testing Strategy

### Unit Tests: MarketMetricsService
- `test_compute_metrics_for_symbol()`: test returns, Sharpe, volatility computation with synthetic OHLC data
- `test_compute_beta()`: test beta and R-squared calculation with aligned return series
- `test_service_caching()`: verify metrics are computed once and cached correctly
- `test_sharpe_ratio_helper()`: test Sharpe calculation with known returns and risk-free rate
- `test_align_returns()`: test return series alignment with different date ranges
- `test_insufficient_data_handling()`: test graceful handling of stocks with <252 days of data

### Unit Tests: State Objects
- `test_market_context_readiness()`: test `is_ready_for_*()` methods
- `test_market_context_getters()`: test safe retrieval methods with missing data
- `test_reference_ticker_config()`: ensure reference tickers load correctly from config with roles

### Integration Tests
- `test_market_metrics_service_integration()`: test service with real stock database fixtures
  - Verify metrics computed correctly for multiple tickers
  - Verify caching works (second call returns cached result)
  - Verify beta computation uses cached reference metrics
- `test_market_overview_report_generation()`: test standalone market overview report creation
  - Assert `market_overview_{date}.md` is created using metrics from `MarketMetricsService`
  - Assert report contains market performance summary table with all reference tickers
  - Assert report contains market risk metrics table
  - Assert report contains market assessment section
  - Assert `market_overview_{date}.json` contains consolidated reference metrics
- `test_market_comparison_pipeline()`: end-to-end with mocked OHLC data for portfolio + references
  - Verify `MarketMetricsService` is instantiated once and reused across nodes
  - Assert beta values are sensible (within expected range)
  - Assert Sharpe ratios are computed for all horizons
  - Assert per-stock `market_comparison.json` files contain consolidated data (metrics + betas)
  - Assert stock reports contain "Market Comparison" section
  - Assert portfolio report contains "Risk-Adjusted Performance vs. Market" section
  - Verify no redundant computation (check service cache hits)
- `test_portfolio_analysis_with_market_comparison()`: run full portfolio analysis workflow with fixtures, verify all three report types are generated, metrics service is used efficiently, and all data is consistent across reports

### Regression Tests
- Use a known portfolio snapshot with historical data
- Compute metrics and compare against manually verified values

## Risks & Limitations

### Data Availability
- Beta and Sharpe require sufficient historical data (at least 252 days for 1-year metrics)
- Newly added stocks may not have enough history → report should gracefully omit or note insufficient data

### Beta Interpretation
- Beta is a single-factor model; does not capture all sources of risk or return
- R-squared should be reported alongside beta to indicate goodness-of-fit

### Sharpe Ratio Assumptions
- Assumes normally distributed returns (may underestimate tail risk)
- Sensitive to choice of risk-free rate
- Not directly comparable across assets with different liquidity or tax treatment

### Market Regime Changes
- Beta is backward-looking and may not reflect current market conditions
- Consider adding a note in the report about the lookback period and regime stability

### Reference Ticker Selection
- Chosen benchmarks (SPY, QQQ, IWM) are U.S.-centric
- May not be appropriate for international portfolios → future enhancement

## Implementation Plan (Incremental)

### Phase 1: Data Infrastructure and Core Calculations
1. Define `MarketContext` state object and constituent dataclasses (`ReferenceTickerMetrics`, `StockMarketComparison`, `PortfolioMarketMetrics`) in a new module (e.g., `models/market.py`)
2. Add `MarketComparisonSettings` to `config.py` (reference tickers with roles, benchmarks, horizons, risk-free rate)
3. Integrate `MarketContext` into main graph state (`PortfolioAnalysisState` in `graph.py`)
4. **Implement `MarketMetricsService`** in `stocks/analysis.py`:
   - **Public API**:
     - `compute_metrics_for_symbol(symbol, slug, horizons) -> ReferenceTickerMetrics`: Single entry point that computes returns, Sharpe ratios, and volatility for any ticker
     - `compute_beta(stock_symbol, stock_slug, benchmark_symbol, benchmark_slug) -> (beta, r_squared)`: Computes beta coefficient and goodness-of-fit
   - **Internal implementation** (private methods):
     - `_compute_sharpe_ratio()`: Sharpe calculation from returns
     - `_compute_daily_returns()`: Log returns from OHLC
     - `_compute_volatility()`: Annualized volatility
     - `_align_returns()`: Align two return series for beta computation
     - `_load_ohlc()`: Load OHLC data from stock database
   - Caches results internally to avoid redundant computation within a run
   - See "Interfaces (Proposed) > Market Metrics Service" for full implementation
5. Add unit tests for `MarketMetricsService` public API and verify caching behavior
6. Service is stateless except for internal cache (cleared between portfolio runs)

### Phase 2: Reference Ticker Management
8. Add `EnsureReferenceFreshNode` to main portfolio graph
9. Add `ComputeReferenceMetricsNode`:
   - Instantiate `MarketMetricsService` with stock DB root and risk-free rate
   - For each reference ticker, call `service.compute_metrics_for_symbol()`
   - Populate `state["market_context"].reference_metrics` from service output
10. Ensure reference tickers use existing stock database infrastructure
11. Add integration test for reference ticker updates and metric computation using service

### Phase 3: Per-Stock Market Comparison
12. Implement `ComputeStockMarketComparisonsNode`:
    - Uses same `MarketMetricsService` instance (benefits from cache)
    - For each stock: `service.compute_metrics_for_symbol()` for stock metrics
    - For each benchmark: `service.compute_beta()` for beta and R-squared
    - Constructs `StockMarketComparison` from service outputs (no redundant computation)
13. Node populates `state["market_context"].stock_comparisons` and writes single consolidated `analysis/market_comparison.json` per stock
14. Add unit and integration tests verifying service is reused, cache works, and state is populated correctly

### Phase 4: Standalone Market Overview Report
15. Implement `GenerateMarketOverviewReportNode` to create standalone market overview report
16. Template for market performance summary table and risk metrics table
17. Add market assessment section with hybrid approach:
    - Use structured templating for performance summaries, risk-adjusted return rankings, and volatility statistics
    - Use LLM with reference ticker roles from config to generate narrative analysis highlighting notable themes, trends, and market context (e.g., rotation patterns, regime changes, relative outperformance drivers)
18. Node reads `state["market_context"].reference_metrics` and writes `market_overview_{date}.md` and `.json`
19. Add integration test verifying market overview report is created with expected sections (structured content) and LLM-generated insights

### Phase 5: Report Enhancement - Per-Stock Reports
20. Enhance stock report generation to read `state["market_context"].stock_comparisons[ticker]` and `analysis/market_comparison.json`
21. Add "Market Comparison" section template to stock reports (Markdown generation)
22. Include beta table, risk-adjusted performance table, multi-horizon Sharpe table
23. Add integration test verifying stock report contains market comparison section with data from `MarketContext`

### Phase 6: Report Enhancement - Portfolio-Level Reports
24. Implement `ComputePortfolioMarketMetricsNode` to aggregate per-stock comparisons
25. Node reads `state["market_context"].stock_comparisons` and writes `state["market_context"].portfolio_metrics`
26. Compute portfolio-level metrics: position-weighted average beta, portfolio Sharpe, contribution analysis
27. Enhance portfolio report generation to read `state["market_context"].portfolio_metrics` and add "Risk-Adjusted Performance vs. Market" section
28. Include benchmark comparison table, top contributors, risk-adjusted summary
29. Add integration test verifying portfolio report contains market comparison section with correct aggregated data

### Phase 7: Graph Integration and Orchestration
30. Wire new nodes into main portfolio analysis graph (`graph.py`) with `MarketContext` state management
31. Ensure correct ordering: state init → reference updates → reference metrics → market overview report → portfolio updates → stock comparisons → portfolio metrics → stock/portfolio reports
32. Add state validation at each node (check `is_ready_for_*()` methods)
33. Test end-to-end portfolio analysis workflow with all three report types and verify state flow
34. Update documentation (README, architecture docs)

### Phase 8: Polish and Extensions
35. Add R-squared to beta output for goodness-of-fit assessment
36. Enhance error handling for missing/partial state data (implement graceful degradation)
37. Add visualization: scatter plot of stock returns vs. market returns with beta line (optional)
38. Handle edge cases: insufficient data, missing benchmarks, new stocks gracefully with proper state updates

## Acceptance Criteria

- Given a portfolio with at least 1 year of historical data:
  - Reference tickers (SPY, QQQ, IWM, etc.) are automatically fetched and stored in the stock database during portfolio analysis
  - Beta coefficients are computed for each portfolio stock vs. each default benchmark (SPY, QQQ, IWM)
  - Sharpe ratios are computed for each portfolio stock and each benchmark across all time horizons (3mo, 6mo, 1yr, 2yr)
  - A standalone market overview report is generated (`output/portfolio/market_overview_{date}.md`) containing:
    - Market performance summary table with returns across all horizons for all reference tickers
    - Market risk metrics table with Sharpe ratios and volatility for all reference tickers
    - Market assessment section with performance analysis, risk-adjusted return ranking, volatility environment, and trend context
  - Per-stock `analysis/market_comparison.json` files are created with beta, Sharpe, and returns data
  - Per-stock reports are enhanced with a "Market Comparison" section containing:
    - Beta coefficients vs. SPY, QQQ, IWM with interpretation
    - Risk-adjusted performance table comparing stock to benchmarks
    - Multi-horizon Sharpe ratio comparison table
  - Portfolio-level report is enhanced with a "Risk-Adjusted Performance vs. Market" section containing:
    - Portfolio risk metrics (average beta, portfolio Sharpe)
    - Benchmark comparison table
    - Top contributors to over/underperformance analysis
    - Risk-adjusted performance summary statistics
- Tests pass deterministically with mocked OHLC data
- Market comparison is automatically integrated into normal `pa analyze` workflow (no separate command needed)
- All three report types (market overview, per-stock, portfolio) are generated in a single analysis run
- Reports gracefully handle insufficient data (e.g., new stocks with <252 days of history) by omitting or noting unavailable metrics

## Future Enhancements (Out of Scope for Initial Implementation)

- Fetch live risk-free rate from data source (e.g., FRED API for 3-month Treasury yield)
- Multi-factor models (Fama-French, Carhart)
- Jensen's alpha (risk-adjusted excess return relative to CAPM)
- Conditional beta (beta during up markets vs. down markets)
- Downside deviation and Sortino ratio
- Tracking error and information ratio (for comparing to active benchmarks)
- Sector-specific benchmarks (e.g., XLK for tech, XLF for financials)
- Custom user-defined benchmarks
- Visualization: returns scatter plots, rolling beta charts, Sharpe ratio time series

## Related Work & References

- Existing modules:
  - `src/portfolio_advisor/stocks/analysis.py`: returns, volatility, SMA
  - `src/portfolio_advisor/graphs/stocks.py`: stock data pipeline
  - `src/portfolio_advisor/services/polygon_client.py`: OHLC data fetching
- Design documents:
  - `docs/design/stock-analysis-plan.md`: foundational plan for stock database and analysis
- External references:
  - Beta: https://en.wikipedia.org/wiki/Beta_(finance)
  - Sharpe ratio: https://en.wikipedia.org/wiki/Sharpe_ratio
  - CAPM: https://en.wikipedia.org/wiki/Capital_asset_pricing_model

