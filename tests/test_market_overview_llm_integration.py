"""Integration test for LLM-generated market overview themes."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from portfolio_advisor.agents.market_comparison import generate_market_overview_report_node
from portfolio_advisor.models.market import MarketContext, ReferenceTickerMetrics


@pytest.fixture
def sample_state_with_full_data(tmp_path):
    """Sample state with comprehensive reference ticker data."""
    settings = MagicMock()
    settings.portfolio_dir = str(tmp_path / "portfolio")
    Path(settings.portfolio_dir).mkdir(parents=True, exist_ok=True)

    market_context = MarketContext()

    # Add comprehensive reference ticker data
    market_context.reference_metrics["SPY"] = ReferenceTickerMetrics(
        symbol="SPY",
        returns={63: 0.026, 126: 0.138, 252: 0.116, 504: 0.462},
        sharpe_ratios={63: 0.85, 126: 0.92, 252: 0.33, 504: 0.89},
        volatility_annualized=0.144,
        as_of="2024-11-21",
    )
    market_context.reference_metrics["QQQ"] = ReferenceTickerMetrics(
        symbol="QQQ",
        returns={63: 0.035, 126: 0.159, 252: 0.172, 504: 0.529},
        sharpe_ratios={63: 1.02, 126: 1.15, 252: 0.48, 504: 0.81},
        volatility_annualized=0.206,
        as_of="2024-11-21",
    )
    market_context.reference_metrics["IWM"] = ReferenceTickerMetrics(
        symbol="IWM",
        returns={63: 0.014, 126: 0.163, 252: 0.022, 504: 0.321},
        sharpe_ratios={63: 0.42, 126: 0.68, 252: -0.10, 504: 0.43},
        volatility_annualized=0.213,
        as_of="2024-11-21",
    )
    market_context.reference_metrics["EFA"] = ReferenceTickerMetrics(
        symbol="EFA",
        returns={63: 0.007, 126: 0.052, 252: 0.196, 504: 0.289},
        sharpe_ratios={63: 0.28, 126: 0.45, 252: 0.80, 504: 0.55},
        volatility_annualized=0.130,
        as_of="2024-11-21",
    )
    market_context.reference_metrics["EEM"] = ReferenceTickerMetrics(
        symbol="EEM",
        returns={63: 0.055, 126: 0.149, 252: 0.223, 504: 0.349},
        sharpe_ratios={63: 0.95, 126: 1.08, 252: 0.88, 504: 0.62},
        volatility_annualized=0.142,
        as_of="2024-11-21",
    )

    return {
        "settings": settings,
        "market_context": market_context,
    }


class TestMarketOverviewLLMIntegration:
    """Integration tests for LLM-generated market themes."""

    @patch("portfolio_advisor.agents.market_comparison.get_llm")
    def test_full_report_with_llm_themes(self, mock_get_llm, sample_state_with_full_data):
        """Test complete market overview report with LLM-generated themes."""
        # Mock LLM to return realistic market analysis
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = """The current market environment exhibits clear growth-oriented leadership, with technology and growth stocks significantly outpacing value and small-cap segments.

**Notable Rotation Patterns:**
The Nasdaq 100's substantial outperformance (+17.2% vs SPY's +11.6%) signals continued investor preference for large-cap growth and technology stocks. Meanwhile, the Russell 2000's near-flat performance (+2.2%) indicates a pronounced large-cap bias, with investors favoring the perceived safety and liquidity of mega-cap names over the broader market breadth.

**Geographic Divergence:**
International markets are showing surprising strength, with emerging markets leading developed markets. The MSCI Emerging Markets' strong showing (+22.3%) and MSCI EAFE's solid performance (+19.6%) both surpassing the S&P 500 suggests a potential broadening of global growth prospects and possibly a weakening dollar environment.

**Asset Class Trends:**
The equity market exhibits clear risk-on sentiment, with all major equity indices posting positive returns while fixed income struggles. Long-duration treasuries showing negative returns (-1.3%) despite rising volatility (8.3%) points to persistent inflation concerns and potential for further rate increases.

**Implications:**
The market favors risk assets over safe havens, growth over value, and international over domestic. However, the concentration in large-cap growth names may present both opportunity and risk, as the narrowing leadership could reverse quickly if economic conditions shift."""
        mock_llm.invoke.return_value = mock_response
        mock_get_llm.return_value = mock_llm

        # Generate report
        result = generate_market_overview_report_node(sample_state_with_full_data)

        # Verify LLM was invoked
        assert mock_llm.invoke.called
        call_args = mock_llm.invoke.call_args[0][0]

        # Verify prompt includes benchmark context
        assert "Primary broad U.S. large-cap equity benchmark" in call_args
        assert "Performance Data" in call_args
        assert "Risk Metrics" in call_args

        # Verify report was created
        portfolio_dir = Path(sample_state_with_full_data["settings"].portfolio_dir)
        report_files = list(portfolio_dir.glob("market_overview_*.md"))
        assert len(report_files) == 1

        # Read and verify report content
        with open(report_files[0]) as f:
            content = f.read()

        # Verify structure
        assert "# Market Overview Report" in content
        assert "## Market Performance Summary" in content
        assert "## Market Risk Metrics" in content
        assert "## Market Assessment" in content

        # Verify templated content
        assert "### Performance Summary (1yr)" in content
        assert "### Risk-Adjusted Returns (1yr Sharpe)" in content
        assert "### Volatility Environment" in content

        # Verify LLM-generated content is present
        assert "### Market Themes and Context" in content
        assert "growth-oriented leadership" in content
        assert "Notable Rotation Patterns" in content
        assert "Geographic Divergence" in content
        assert "Asset Class Trends" in content
        assert "Implications" in content

        # Verify it's not placeholder text
        assert "To be implemented" not in content
        assert "LLM-Generated Analysis" not in content or "future phase" not in content

        # Verify state was updated
        assert result["market_context"].market_overview_generated is not None

    @patch("portfolio_advisor.agents.market_comparison.get_llm")
    def test_handles_llm_failure_gracefully_in_full_flow(
        self, mock_get_llm, sample_state_with_full_data
    ):
        """Test that full report generation handles LLM failures gracefully."""
        # Mock LLM to fail
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = Exception("LLM service unavailable")
        mock_get_llm.return_value = mock_llm

        # Generate report - should not raise exception
        result = generate_market_overview_report_node(sample_state_with_full_data)

        # Verify report was still created
        portfolio_dir = Path(sample_state_with_full_data["settings"].portfolio_dir)
        report_files = list(portfolio_dir.glob("market_overview_*.md"))
        assert len(report_files) == 1

        # Read report and verify fallback message
        with open(report_files[0]) as f:
            content = f.read()

        # Should have structure
        assert "## Market Assessment" in content
        assert "### Market Themes and Context" in content

        # Should have fallback message
        assert "unavailable due to LLM error" in content

        # Should still have templated content
        assert "### Performance Summary (1yr)" in content
        assert "### Risk-Adjusted Returns (1yr Sharpe)" in content

        # State should still be updated
        assert result["market_context"].market_overview_generated is not None

    def test_prompt_includes_all_benchmark_roles(self, sample_state_with_full_data):
        """Test that LLM prompt includes role descriptions for all benchmarks."""
        from portfolio_advisor.config import MarketComparisonSettings

        market_settings = MarketComparisonSettings()

        with patch("portfolio_advisor.agents.market_comparison.get_llm") as mock_get_llm:
            mock_llm = MagicMock()
            mock_response = MagicMock()
            mock_response.content = "Test narrative"
            mock_llm.invoke.return_value = mock_response
            mock_get_llm.return_value = mock_llm

            # Generate report
            generate_market_overview_report_node(sample_state_with_full_data)

            # Verify prompt includes all configured benchmark roles
            call_args = mock_llm.invoke.call_args[0][0]

            # Check for key benchmark roles from config
            assert "Primary broad U.S. large-cap equity benchmark" in call_args  # SPY
            assert "U.S. large-cap growth and technology benchmark" in call_args  # QQQ
            assert "U.S. small-cap equity benchmark" in call_args  # IWM
            assert "International developed markets equity benchmark" in call_args  # EFA
            assert "Emerging markets equity benchmark" in call_args  # EEM

