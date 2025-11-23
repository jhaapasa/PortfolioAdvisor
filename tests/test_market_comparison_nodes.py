"""Tests for market comparison nodes."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from portfolio_advisor.agents.market_comparison import (
    _generate_market_themes_narrative,
    compute_portfolio_market_metrics_node,
    compute_reference_metrics_node,
    compute_stock_market_comparisons_node,
    ensure_reference_fresh_node,
    generate_market_overview_report_node,
)
from portfolio_advisor.models.market import (
    MarketContext,
    ReferenceTickerMetrics,
    StockMarketComparison,
)


@pytest.fixture
def mock_settings(tmp_path):
    """Mock settings for testing."""
    settings = MagicMock()
    settings.output_dir = str(tmp_path / "output")
    settings.portfolio_dir = str(tmp_path / "portfolio")
    Path(settings.portfolio_dir).mkdir(parents=True, exist_ok=True)
    return settings


@pytest.fixture
def sample_state(mock_settings):
    """Sample state for testing."""
    return {
        "settings": mock_settings,
        "market_context": MarketContext(),
        "instruments": [
            {"instrument_id": "cid:stocks:us:composite:AAPL", "primary_ticker": "AAPL"},
            {"instrument_id": "cid:stocks:us:composite:MSFT", "primary_ticker": "MSFT"},
        ],
        "resolved_holdings": [
            {"primary_ticker": "AAPL", "total_value": 60000},
            {"primary_ticker": "MSFT", "total_value": 40000},
        ],
    }


class TestEnsureReferenceFreshNode:
    """Test ensure_reference_fresh_node."""

    @patch("portfolio_advisor.agents.market_comparison.update_instrument")
    def test_updates_reference_tickers(self, mock_update_instrument, sample_state):
        """Test that reference tickers are updated."""
        # Run node
        result = ensure_reference_fresh_node(sample_state)

        # Check that update_instrument was called for each reference ticker
        assert mock_update_instrument.call_count >= 3  # At least SPY, QQQ, IWM

        # Verify some calls
        instruments = [call[0][1] for call in mock_update_instrument.call_args_list]
        tickers = [inst.get("primary_ticker") for inst in instruments]
        assert "SPY" in tickers
        assert "QQQ" in tickers

        # Market context should be preserved
        assert result["market_context"] is sample_state["market_context"]

    @patch(
        "portfolio_advisor.agents.market_comparison.update_instrument",
        side_effect=Exception("Network error"),
    )
    def test_handles_update_failures(self, mock_update_instrument, sample_state):
        """Test graceful handling of update failures."""
        # Should not raise exception
        result = ensure_reference_fresh_node(sample_state)

        assert result["market_context"] is sample_state["market_context"]


class TestComputeReferenceMetricsNode:
    """Test compute_reference_metrics_node."""

    @patch("portfolio_advisor.agents.market_comparison.MarketMetricsService")
    def test_computes_metrics(self, mock_service_class, sample_state):
        """Test that metrics are computed for reference tickers."""
        # Mock service
        mock_service = MagicMock()
        mock_service_class.return_value = mock_service

        # Mock metrics
        spy_metrics = ReferenceTickerMetrics(
            symbol="SPY",
            returns={63: 0.05, 252: 0.18},
            sharpe_ratios={63: 1.1, 252: 1.05},
            volatility_annualized=0.18,
            as_of="2024-11-17",
        )
        mock_service.compute_metrics_for_symbol.return_value = spy_metrics

        # Run node
        result = compute_reference_metrics_node(sample_state)

        # Check service was called
        assert mock_service.compute_metrics_for_symbol.called

        # Check metrics stored in context
        market_context = result["market_context"]
        assert "SPY" in market_context.reference_metrics
        assert market_context.reference_metrics["SPY"] == spy_metrics

        # Check cache cleared
        mock_service.clear_cache.assert_called_once()


class TestComputeStockMarketComparisonsNode:
    """Test compute_stock_market_comparisons_node."""

    @patch("portfolio_advisor.agents.market_comparison.MarketMetricsService")
    def test_computes_comparisons(self, mock_service_class, sample_state, tmp_path):
        """Test that comparisons are computed for portfolio stocks."""
        # Add reference metrics to context first
        sample_state["market_context"].reference_metrics["SPY"] = ReferenceTickerMetrics(
            symbol="SPY",
            returns={252: 0.18},
            sharpe_ratios={252: 1.05},
            volatility_annualized=0.18,
            as_of="2024-11-17",
        )

        # Mock service
        mock_service = MagicMock()
        mock_service_class.return_value = mock_service

        # Mock stock metrics
        aapl_metrics = ReferenceTickerMetrics(
            symbol="AAPL",
            returns={252: 0.24},
            sharpe_ratios={252: 1.32},
            volatility_annualized=0.21,
            as_of="2024-11-17",
        )
        mock_service.compute_metrics_for_symbol.return_value = aapl_metrics
        mock_service.compute_beta.return_value = (1.24, 0.85)  # beta, r_squared

        # Create output directory structure
        stock_db_root = tmp_path / "output" / "stocks"
        aapl_dir = stock_db_root / "tickers" / "cid-stocks-us-composite-aapl" / "analysis"
        aapl_dir.mkdir(parents=True, exist_ok=True)

        # Run node
        result = compute_stock_market_comparisons_node(sample_state)

        # Check comparisons stored in context
        market_context = result["market_context"]
        assert "AAPL" in market_context.stock_comparisons
        comparison = market_context.stock_comparisons["AAPL"]
        assert comparison.ticker == "AAPL"
        assert comparison.betas["SPY"] == 1.24
        assert comparison.sharpe_ratios[252] == 1.32

        # Check JSON file written
        json_path = aapl_dir / "market_comparison.json"
        assert json_path.exists()

        with open(json_path) as f:
            data = json.load(f)
            assert data["ticker"] == "AAPL"
            assert data["betas"]["SPY"]["value"] == 1.24
            assert data["betas"]["SPY"]["r_squared"] == 0.85

    def test_skips_if_not_ready(self, sample_state):
        """Test that node skips if reference metrics not ready."""
        # No reference metrics
        result = compute_stock_market_comparisons_node(sample_state)

        # Should return unchanged context
        assert len(result["market_context"].stock_comparisons) == 0


class TestComputePortfolioMarketMetricsNode:
    """Test compute_portfolio_market_metrics_node."""

    def test_computes_portfolio_metrics(self, sample_state):
        """Test portfolio-level metrics computation."""
        # Set up reference metrics
        sample_state["market_context"].reference_metrics["SPY"] = ReferenceTickerMetrics(
            symbol="SPY",
            returns={252: 0.18},
            sharpe_ratios={252: 1.05},
            volatility_annualized=0.18,
            as_of="2024-11-17",
        )

        # Set up stock comparisons
        sample_state["market_context"].stock_comparisons["AAPL"] = StockMarketComparison(
            ticker="AAPL",
            slug="cid-stocks-us-composite-aapl",
            betas={"SPY": 1.24},
            sharpe_ratios={252: 1.32},
            returns={252: 0.24},
            as_of="2024-11-17",
        )
        sample_state["market_context"].stock_comparisons["MSFT"] = StockMarketComparison(
            ticker="MSFT",
            slug="cid-stocks-us-composite-msft",
            betas={"SPY": 0.98},
            sharpe_ratios={252: 1.15},
            returns={252: 0.20},
            as_of="2024-11-17",
        )

        # Run node
        result = compute_portfolio_market_metrics_node(sample_state)

        # Check portfolio metrics
        portfolio_metrics = result["market_context"].portfolio_metrics
        assert portfolio_metrics is not None

        # Check average beta (weighted: 60% * 1.24 + 40% * 0.98 = 1.136)
        assert abs(portfolio_metrics.average_beta_vs_benchmarks["SPY"] - 1.136) < 0.001

        # Check stocks outperforming
        assert portfolio_metrics.stocks_outperforming["SPY"] == 2  # Both beat SPY

        # Check top contributors
        assert len(portfolio_metrics.top_contributors) == 2
        assert portfolio_metrics.top_contributors[0]["ticker"] == "AAPL"  # Higher excess Sharpe

    def test_skips_if_not_ready(self, sample_state):
        """Test that node skips if stock comparisons not ready."""
        # No stock comparisons
        result = compute_portfolio_market_metrics_node(sample_state)

        # Should return unchanged context
        assert result["market_context"].portfolio_metrics is None


class TestGenerateMarketOverviewReportNode:
    """Test generate_market_overview_report_node."""

    def test_generates_report(self, sample_state, tmp_path):
        """Test market overview report generation."""
        # Set up reference metrics
        sample_state["market_context"].reference_metrics["SPY"] = ReferenceTickerMetrics(
            symbol="SPY",
            returns={63: 0.052, 126: 0.123, 252: 0.187, 504: 0.245},
            sharpe_ratios={63: 1.12, 126: 1.08, 252: 1.05, 504: 0.98},
            volatility_annualized=0.182,
            as_of="2024-11-17",
        )
        sample_state["market_context"].reference_metrics["QQQ"] = ReferenceTickerMetrics(
            symbol="QQQ",
            returns={63: 0.071, 126: 0.158, 252: 0.224, 504: 0.289},
            sharpe_ratios={63: 1.28, 126: 1.22, 252: 1.18, 504: 1.15},
            volatility_annualized=0.198,
            as_of="2024-11-17",
        )

        # Run node
        result = generate_market_overview_report_node(sample_state)

        # Check report generated
        portfolio_dir = Path(sample_state["settings"].portfolio_dir)
        report_files = list(portfolio_dir.glob("market_overview_*.md"))
        assert len(report_files) == 1

        # Check report content
        with open(report_files[0]) as f:
            content = f.read()
            assert "Market Overview Report" in content
            assert "Market Performance Summary" in content
            assert "SPY" in content
            assert "+18.7%" in content  # SPY 1yr return
            assert "Market Risk Metrics" in content

        # Check JSON also created
        json_files = list(portfolio_dir.glob("market_overview_*.json"))
        assert len(json_files) == 1

        # Check state updated
        assert result["market_context"].market_overview_generated is not None

    def test_skips_if_no_metrics(self, sample_state):
        """Test that report generation skips if no reference metrics."""
        # No reference metrics
        generate_market_overview_report_node(sample_state)

        # Should not generate report
        portfolio_dir = Path(sample_state["settings"].portfolio_dir)
        report_files = list(portfolio_dir.glob("market_overview_*.md"))
        assert len(report_files) == 0

    @patch("portfolio_advisor.agents.market_comparison.get_llm")
    def test_includes_llm_market_themes(self, mock_get_llm, sample_state):
        """Test that market overview report includes LLM-generated market themes."""
        # Set up mock LLM
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = (
            "The market is exhibiting strong growth-oriented leadership. "
            "Nasdaq 100 outperformance reflects investor preference for technology. "
            "International markets are showing divergence with emerging markets lagging."
        )
        mock_llm.invoke.return_value = mock_response
        mock_get_llm.return_value = mock_llm

        # Set up reference metrics
        sample_state["market_context"].reference_metrics["SPY"] = ReferenceTickerMetrics(
            symbol="SPY",
            returns={63: 0.052, 126: 0.123, 252: 0.187, 504: 0.245},
            sharpe_ratios={63: 1.12, 126: 1.08, 252: 1.05, 504: 0.98},
            volatility_annualized=0.182,
            as_of="2024-11-17",
        )
        sample_state["market_context"].reference_metrics["QQQ"] = ReferenceTickerMetrics(
            symbol="QQQ",
            returns={63: 0.071, 126: 0.158, 252: 0.224, 504: 0.289},
            sharpe_ratios={63: 1.28, 126: 1.22, 252: 1.18, 504: 1.15},
            volatility_annualized=0.198,
            as_of="2024-11-17",
        )

        # Run node
        generate_market_overview_report_node(sample_state)

        # Verify LLM was called
        assert mock_llm.invoke.called
        call_args = mock_llm.invoke.call_args[0][0]
        assert "market performance data" in call_args.lower()
        assert "benchmark" in call_args.lower()

        # Check report includes LLM-generated content
        portfolio_dir = Path(sample_state["settings"].portfolio_dir)
        report_files = list(portfolio_dir.glob("market_overview_*.md"))
        assert len(report_files) == 1

        with open(report_files[0]) as f:
            content = f.read()
            assert "Market Themes and Context" in content
            assert "growth-oriented leadership" in content
            assert "Nasdaq 100 outperformance" in content
            assert "emerging markets lagging" in content

    @patch("portfolio_advisor.agents.market_comparison.get_llm")
    def test_handles_llm_failure_gracefully(self, mock_get_llm, sample_state):
        """Test that report generation handles LLM failures gracefully."""
        # Set up mock LLM that raises an exception
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = Exception("LLM service unavailable")
        mock_get_llm.return_value = mock_llm

        # Set up reference metrics
        sample_state["market_context"].reference_metrics["SPY"] = ReferenceTickerMetrics(
            symbol="SPY",
            returns={63: 0.052, 126: 0.123, 252: 0.187, 504: 0.245},
            sharpe_ratios={63: 1.12, 126: 1.08, 252: 1.05, 504: 0.98},
            volatility_annualized=0.182,
            as_of="2024-11-17",
        )

        # Run node - should not raise exception
        generate_market_overview_report_node(sample_state)

        # Check report was still generated with fallback message
        portfolio_dir = Path(sample_state["settings"].portfolio_dir)
        report_files = list(portfolio_dir.glob("market_overview_*.md"))
        assert len(report_files) == 1

        with open(report_files[0]) as f:
            content = f.read()
            assert "Market Themes and Context" in content
            assert "unavailable due to LLM error" in content

    def test_market_themes_prompt_includes_benchmark_roles(self, sample_state):
        """Test that market themes prompt includes benchmark role descriptions."""
        from portfolio_advisor.config import MarketComparisonSettings

        # Set up reference metrics
        sample_state["market_context"].reference_metrics["SPY"] = ReferenceTickerMetrics(
            symbol="SPY",
            returns={252: 0.187},
            sharpe_ratios={252: 1.05},
            volatility_annualized=0.182,
            as_of="2024-11-17",
        )
        sample_state["market_context"].reference_metrics["QQQ"] = ReferenceTickerMetrics(
            symbol="QQQ",
            returns={252: 0.224},
            sharpe_ratios={252: 1.18},
            volatility_annualized=0.198,
            as_of="2024-11-17",
        )

        market_settings = MarketComparisonSettings()

        # Generate narrative (will fail but we can check the prompt)
        with patch("portfolio_advisor.agents.market_comparison.get_llm") as mock_get_llm:
            mock_llm = MagicMock()
            mock_response = MagicMock()
            mock_response.content = "Test narrative"
            mock_llm.invoke.return_value = mock_response
            mock_get_llm.return_value = mock_llm

            _generate_market_themes_narrative(
                sample_state["market_context"], market_settings, sample_state["settings"]
            )

            # Check that prompt includes role descriptions
            call_args = mock_llm.invoke.call_args[0][0]
            assert "Primary broad U.S. large-cap equity benchmark" in call_args  # SPY role
            assert "U.S. large-cap growth and technology benchmark" in call_args  # QQQ role
            assert "Performance Data" in call_args
            assert "Risk Metrics" in call_args
