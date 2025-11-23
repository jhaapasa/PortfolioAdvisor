"""Market comparison state objects and dataclasses."""

from dataclasses import dataclass, field


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
    - reference_metrics: Written by ComputeReferenceMetricsNode, read by
      GenerateMarketOverviewReportNode and stock/portfolio comparison nodes
    - stock_comparisons: Written by ComputeStockMarketComparisonsNode, read by stock
      report nodes and ComputePortfolioMarketMetricsNode
    - portfolio_metrics: Written by ComputePortfolioMarketMetricsNode, read by portfolio
      report node
    - market_overview_generated: Written by GenerateMarketOverviewReportNode
      (path to generated report)
    """

    # Reference ticker metrics (all benchmarks)
    reference_metrics: dict[str, ReferenceTickerMetrics] = field(default_factory=dict)
    # Key: ticker symbol (e.g., "SPY"), Value: ReferenceTickerMetrics

    # Per-stock market comparisons (portfolio stocks only)
    stock_comparisons: dict[str, StockMarketComparison] = field(default_factory=dict)
    # Key: stock ticker (e.g., "AAPL"), Value: StockMarketComparison

    # Portfolio-level aggregated metrics
    portfolio_metrics: PortfolioMarketMetrics | None = None

    # Path to generated market overview report (for tracking/testing)
    market_overview_generated: str | None = None

    def is_ready_for_stock_comparisons(self) -> bool:
        """Check if reference metrics are available for stock comparisons."""
        return len(self.reference_metrics) > 0

    def is_ready_for_portfolio_metrics(self) -> bool:
        """Check if stock comparisons are available for portfolio aggregation."""
        return len(self.stock_comparisons) > 0

    def get_reference_metric(self, symbol: str) -> ReferenceTickerMetrics | None:
        """Safely retrieve reference metric by symbol."""
        return self.reference_metrics.get(symbol)

    def get_stock_comparison(self, ticker: str) -> StockMarketComparison | None:
        """Safely retrieve stock comparison by ticker."""
        return self.stock_comparisons.get(ticker)
