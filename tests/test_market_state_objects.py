"""Tests for market comparison state objects."""

from portfolio_advisor.config import MarketComparisonSettings, ReferenceTicker
from portfolio_advisor.models.market import (
    MarketContext,
    PortfolioMarketMetrics,
    ReferenceTickerMetrics,
    StockMarketComparison,
)


class TestMarketStateObjects:
    """Test market comparison state objects."""

    def test_reference_ticker_metrics(self):
        """Test ReferenceTickerMetrics dataclass."""
        metrics = ReferenceTickerMetrics(
            symbol="SPY",
            returns={63: 0.05, 126: 0.12, 252: 0.18},
            sharpe_ratios={63: 1.2, 126: 1.1, 252: 1.05},
            volatility_annualized=0.18,
            as_of="2024-11-17",
        )

        assert metrics.symbol == "SPY"
        assert metrics.returns[252] == 0.18
        assert metrics.sharpe_ratios[63] == 1.2
        assert metrics.volatility_annualized == 0.18
        assert metrics.as_of == "2024-11-17"

    def test_stock_market_comparison(self):
        """Test StockMarketComparison dataclass."""
        comparison = StockMarketComparison(
            ticker="AAPL",
            slug="cid-stocks-us-composite-aapl",
            betas={"SPY": 1.24, "QQQ": 0.98},
            sharpe_ratios={252: 1.32},
            returns={252: 0.24},
            as_of="2024-11-17",
        )

        assert comparison.ticker == "AAPL"
        assert comparison.slug == "cid-stocks-us-composite-aapl"
        assert comparison.betas["SPY"] == 1.24
        assert comparison.sharpe_ratios[252] == 1.32
        assert comparison.returns[252] == 0.24

    def test_portfolio_market_metrics(self):
        """Test PortfolioMarketMetrics dataclass."""
        metrics = PortfolioMarketMetrics(
            average_beta_vs_benchmarks={"SPY": 1.12, "QQQ": 0.95},
            portfolio_sharpe=1.18,
            average_stock_sharpe=1.15,
            stocks_outperforming={"SPY": 8, "QQQ": 5},
            total_stocks=12,
            top_contributors=[{"ticker": "AAPL", "excess_sharpe": 0.27, "excess_return": 0.032}],
            as_of="2024-11-17",
        )

        assert metrics.average_beta_vs_benchmarks["SPY"] == 1.12
        assert metrics.portfolio_sharpe == 1.18
        assert metrics.stocks_outperforming["SPY"] == 8
        assert metrics.total_stocks == 12
        assert len(metrics.top_contributors) == 1

    def test_market_context(self):
        """Test MarketContext state management."""
        context = MarketContext()

        # Initially empty
        assert len(context.reference_metrics) == 0
        assert len(context.stock_comparisons) == 0
        assert context.portfolio_metrics is None
        assert context.market_overview_generated is None
        assert not context.is_ready_for_stock_comparisons()
        assert not context.is_ready_for_portfolio_metrics()

        # Add reference metrics
        spy_metrics = ReferenceTickerMetrics(
            symbol="SPY",
            returns={252: 0.18},
            sharpe_ratios={252: 1.05},
            volatility_annualized=0.18,
            as_of="2024-11-17",
        )
        context.reference_metrics["SPY"] = spy_metrics

        assert context.is_ready_for_stock_comparisons()
        assert context.get_reference_metric("SPY") == spy_metrics
        assert context.get_reference_metric("QQQ") is None

        # Add stock comparison
        aapl_comparison = StockMarketComparison(
            ticker="AAPL",
            slug="cid-stocks-us-composite-aapl",
            betas={"SPY": 1.24},
            sharpe_ratios={252: 1.32},
            returns={252: 0.24},
            as_of="2024-11-17",
        )
        context.stock_comparisons["AAPL"] = aapl_comparison

        assert context.is_ready_for_portfolio_metrics()
        assert context.get_stock_comparison("AAPL") == aapl_comparison
        assert context.get_stock_comparison("MSFT") is None


class TestMarketComparisonSettings:
    """Test market comparison configuration."""

    def test_default_settings(self):
        """Test default MarketComparisonSettings."""
        settings = MarketComparisonSettings()

        # Check reference tickers
        assert len(settings.reference_tickers) == 8
        assert any(t.symbol == "SPY" for t in settings.reference_tickers)
        assert any(t.symbol == "QQQ" for t in settings.reference_tickers)

        # Check default benchmarks
        assert "SPY" in settings.default_benchmarks
        assert "QQQ" in settings.default_benchmarks
        assert "IWM" in settings.default_benchmarks

        # Check time horizons
        assert 63 in settings.time_horizons_days  # 3 months
        assert 252 in settings.time_horizons_days  # 1 year

        # Check risk-free rate
        assert settings.risk_free_rate_annual == 0.045
        assert abs(settings.risk_free_rate_daily - 0.045 / 252) < 0.0001

    def test_reference_symbols_property(self):
        """Test reference_symbols property."""
        settings = MarketComparisonSettings()
        symbols = settings.reference_symbols

        assert "SPY" in symbols
        assert "QQQ" in symbols
        assert "IWM" in symbols
        assert len(symbols) == len(settings.reference_tickers)

    def test_benchmark_roles_property(self):
        """Test benchmark_roles property."""
        settings = MarketComparisonSettings()
        roles = settings.benchmark_roles

        assert roles["SPY"] == "Primary broad U.S. large-cap equity benchmark"
        assert roles["QQQ"] == "U.S. large-cap growth and technology benchmark"
        assert len(roles) == len(settings.reference_tickers)

    def test_reference_ticker_dataclass(self):
        """Test ReferenceTicker dataclass."""
        ticker = ReferenceTicker(
            symbol="SPY", name="S&P 500", role="Primary broad U.S. large-cap equity benchmark"
        )

        assert ticker.symbol == "SPY"
        assert ticker.name == "S&P 500"
        assert ticker.role == "Primary broad U.S. large-cap equity benchmark"
