"""Tests for stock report enhancement with market comparison data."""

import json
from unittest.mock import MagicMock, patch

import pytest

from portfolio_advisor.agents.stock_report_collator import _build_user_prompt, collate_report_node


@pytest.fixture
def mock_paths(tmp_path):
    """Mock StockPaths object."""
    paths = MagicMock()

    # Create directory structure
    ticker_dir = tmp_path / "tickers" / "test-slug"
    analysis_dir = ticker_dir / "analysis"
    report_dir = ticker_dir / "report"
    analysis_dir.mkdir(parents=True)
    report_dir.mkdir(parents=True)

    # Setup paths
    paths.ticker_dir.return_value = ticker_dir
    paths.analysis_returns_json.return_value = analysis_dir / "returns.json"
    paths.analysis_wavelet_hist_json.return_value = analysis_dir / "wavelet_hist.json"
    paths.report_dir.return_value = report_dir

    return paths


@pytest.fixture
def sample_market_comparison():
    """Sample market comparison data."""
    return {
        "ticker": "AAPL",
        "slug": "cid-stocks-us-composite-aapl",
        "betas": {
            "SPY": {"value": 1.24, "r_squared": 0.85},
            "QQQ": {"value": 0.98, "r_squared": 0.90},
        },
        "sharpe_ratios": {252: 1.32, 504: 1.28},
        "returns": {252: 0.243},
        "volatility_annualized": 0.215,
    }


@pytest.fixture
def sample_state(mock_paths):
    """Sample state for testing."""
    settings = MagicMock()
    settings.include_news_report = True

    return {
        "settings": settings,
        "instrument": {"primary_ticker": "AAPL"},
        "_slug": "test-slug",
        "_paths": mock_paths,
        "news_summary": {
            "markdown": "## Key Developments\n- Apple announces new product...",
            "json": {"sentiment": "positive", "article_count": 5},
        },
    }


class TestBuildUserPrompt:
    """Test _build_user_prompt function."""

    def test_includes_market_comparison(self, sample_market_comparison):
        """Test that market comparison data is included in prompt."""
        prompt = _build_user_prompt(
            ticker="AAPL",
            slug="test-slug",
            news_md="Test news",
            news_json=None,
            returns_doc={"as_of": "2024-11-17", "windows": {"d5": 0.02}},
            vol_hist_doc=None,
            market_comparison=sample_market_comparison,
        )

        assert "Market Comparison" in prompt
        assert "Beta: {SPY: 1.24, QQQ: 0.98}" in prompt
        assert "Sharpe (1yr): 1.32" in prompt
        assert "Return (1yr): 24.3%" in prompt
        assert "Volatility: 21.5%" in prompt

    def test_handles_missing_market_comparison(self):
        """Test prompt generation without market comparison."""
        prompt = _build_user_prompt(
            ticker="AAPL",
            slug="test-slug",
            news_md="Test news",
            news_json=None,
            returns_doc=None,
            vol_hist_doc=None,
            market_comparison=None,
        )

        assert "Market Comparison" not in prompt
        assert "Beta" not in prompt


class TestCollateReportNode:
    """Test collate_report_node functionality."""

    @patch("portfolio_advisor.agents.stock_report_collator.get_llm")
    def test_includes_market_comparison_in_report(
        self, mock_get_llm, sample_state, mock_paths, sample_market_comparison
    ):
        """Test that market comparison is included when generating report."""
        # Setup market comparison file
        mc_path = mock_paths.ticker_dir("test-slug") / "analysis" / "market_comparison.json"
        with open(mc_path, "w") as f:
            json.dump(sample_market_comparison, f)

        # Mock LLM response
        mock_llm = MagicMock()
        mock_llm.invoke.return_value.content = (
            "# AAPL — 7‑Day Update\n\n"
            "## TL;DR\n"
            "Apple showing strong performance with 24.3% annual return, "
            "outperforming SPY. Beta of 1.24 indicates higher volatility than market.\n\n"
            "## Notable News & Events\n"
            "- Apple announces new product...\n"
        )
        mock_get_llm.return_value = mock_llm

        # Run node
        collate_report_node(sample_state)

        # Verify market comparison was loaded and included in prompt
        call_args = mock_llm.invoke.call_args[0][0]

        # Debug print to see actual prompt
        print("\n\nDEBUG - Actual prompt content:")
        print(call_args)
        print("\n\n")

        assert "Market Comparison" in call_args
        assert "Beta" in call_args
        # Check for the actual format used in the prompt
        assert "SPY: 1.24" in call_args or "Beta: {SPY: 1.24" in call_args

        # Verify report was written
        report_path = mock_paths.report_dir("test-slug") / "7d" / "report.md"
        assert report_path.exists()

        with open(report_path) as f:
            content = f.read()
            assert "Beta of 1.24" in content

    def test_handles_missing_market_comparison(self, sample_state, mock_paths):
        """Test report generation without market comparison data."""
        # Don't create market_comparison.json

        with patch("portfolio_advisor.agents.stock_report_collator.get_llm") as mock_get_llm:
            mock_llm = MagicMock()
            mock_llm.invoke.return_value.content = "# AAPL — 7‑Day Update\n\nTest content"
            mock_get_llm.return_value = mock_llm

            # Run node
            collate_report_node(sample_state)

            # Verify prompt doesn't include market comparison
            call_args = mock_llm.invoke.call_args[0][0]
            assert "Market Comparison" not in call_args
