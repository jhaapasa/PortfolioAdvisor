"""Tests for analyst node enhancement with market comparison data."""

from unittest.mock import MagicMock, patch

import pytest

from portfolio_advisor.agents.analyst import analyst_node
from portfolio_advisor.models.market import MarketContext, PortfolioMarketMetrics


@pytest.fixture
def mock_settings():
    """Mock settings for testing."""
    settings = MagicMock()
    return settings


@pytest.fixture
def sample_state_with_market(mock_settings):
    """Sample state with market comparison data."""
    # Create market context with portfolio metrics
    market_context = MarketContext()
    market_context.portfolio_metrics = PortfolioMarketMetrics(
        average_beta_vs_benchmarks={"SPY": 1.12, "QQQ": 0.95},
        portfolio_sharpe=1.18,
        average_stock_sharpe=1.15,
        stocks_outperforming={"SPY": 8, "QQQ": 5},
        total_stocks=12,
        top_contributors=[
            {"ticker": "AAPL", "excess_sharpe": 0.27},
            {"ticker": "MSFT", "excess_sharpe": 0.23},
            {"ticker": "NVDA", "excess_sharpe": 0.19},
        ],
        as_of="2024-11-17",
    )

    return {
        "settings": mock_settings,
        "raw_docs": [{"name": "portfolio.csv", "mime_type": "text/csv", "source_bytes": 1024}],
        "plan": {"steps": ["Parse holdings", "Resolve symbols", "Analyze"]},
        "resolved_holdings": [{"ticker": "AAPL", "quantity": 100}] * 12,
        "unresolved_entities": [],
        "basket_reports": [
            {
                "label": "Technology",
                "slug": "technology",
                "averages": {"d1": 0.012, "d5": 0.025},
                "summary_text": "Tech basket performing well",
            }
        ],
        "market_context": market_context,
    }


@pytest.fixture
def sample_state_without_market(mock_settings):
    """Sample state without market comparison data."""
    return {
        "settings": mock_settings,
        "raw_docs": [{"name": "portfolio.csv", "mime_type": "text/csv", "source_bytes": 1024}],
        "plan": {"steps": ["Parse holdings", "Resolve symbols"]},
        "resolved_holdings": [{"ticker": "AAPL", "quantity": 100}],
        "unresolved_entities": [],
        "basket_reports": [],
    }


class TestAnalystMarketEnhancement:
    """Test analyst node with market comparison enhancements."""

    @patch("portfolio_advisor.agents.analyst.get_llm")
    def test_includes_market_comparison(self, mock_get_llm, sample_state_with_market):
        """Test that market comparison data is included in analyst prompt."""
        # Mock LLM
        mock_llm = MagicMock()
        mock_llm.invoke.return_value.content = (
            "# Portfolio Analysis\n\n"
            "## Holdings Summary\n"
            "12 positions resolved successfully.\n\n"
            "## Market Comparison\n"
            "The portfolio shows an average beta of 1.12 vs SPY, indicating "
            "slightly higher volatility than the market. With a Sharpe ratio of 1.18, "
            "the portfolio is delivering strong risk-adjusted returns.\n\n"
            "67% of holdings are outperforming the S&P 500 benchmark.\n"
        )
        mock_get_llm.return_value = mock_llm

        # Run node
        result = analyst_node(sample_state_with_market)

        # Verify market comparison was included in prompt
        call_args = mock_llm.invoke.call_args[0][0]

        # Check that market metrics are in the prompt
        assert "Market Comparison:" in call_args
        assert "Portfolio Risk Metrics:" in call_args
        assert "Average Beta: SPY: 1.12, QQQ: 0.95" in call_args
        assert "Portfolio Sharpe (1yr): 1.18" in call_args
        assert "Stocks outperforming SPY: 8/12 (67%)" in call_args
        assert "Top Contributors (vs SPY):" in call_args
        assert "AAPL: Excess Sharpe +0.27" in call_args

        # Check the output
        assert "market comparison" in result["analysis"].lower()
        assert "beta of 1.12" in result["analysis"]
        assert "Sharpe ratio of 1.18" in result["analysis"]

    @patch("portfolio_advisor.agents.analyst.get_llm")
    def test_handles_missing_market_data(self, mock_get_llm, sample_state_without_market):
        """Test analyst node works without market comparison data."""
        # Mock LLM
        mock_llm = MagicMock()
        mock_llm.invoke.return_value.content = (
            "# Portfolio Analysis\n\n" "1 position resolved successfully."
        )
        mock_get_llm.return_value = mock_llm

        # Run node
        result = analyst_node(sample_state_without_market)

        # Verify prompt includes placeholder for missing market data
        call_args = mock_llm.invoke.call_args[0][0]
        assert "Market Comparison:" in call_args
        assert "[no market comparison data available]" in call_args

        # Should still produce analysis
        assert "analysis" in result
        assert len(result["analysis"]) > 0

    @patch("portfolio_advisor.agents.analyst.get_llm")
    def test_formats_top_contributors(self, mock_get_llm, sample_state_with_market):
        """Test that top contributors are properly formatted."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value.content = "Analysis"
        mock_get_llm.return_value = mock_llm

        # Run node
        analyst_node(sample_state_with_market)

        # Check top contributors formatting
        call_args = mock_llm.invoke.call_args[0][0]
        assert "1. AAPL: Excess Sharpe +0.27" in call_args
        assert "2. MSFT: Excess Sharpe +0.23" in call_args
        assert "3. NVDA: Excess Sharpe +0.19" in call_args
