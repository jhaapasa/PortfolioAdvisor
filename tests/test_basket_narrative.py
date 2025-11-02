from pathlib import Path
from unittest.mock import MagicMock

from portfolio_advisor.agents.basket_narrative import (
    _build_narrative_prompt,
    _read_ticker_report_summary,
    collect_ticker_news_summaries_node,
    generate_basket_narrative_node,
)
from portfolio_advisor.config import Settings


def test_read_ticker_report_summary(tmp_path: Path):
    """Test reading ticker report summary extracts TL;DR and events."""
    stock_dir = tmp_path / "stocks"
    report_dir = stock_dir / "tickers" / "cid-stocks-us-xnas-aapl" / "report" / "7d"
    report_dir.mkdir(parents=True)

    report_content = """TL;DR
- AAPL rose about 2.87% over the past 5 trading days
- Post-earnings tone turned notably positive

Notable News & Events
- Oct 31 (Benzinga): Post-earnings coverage turned bullish
- Oct 30 (Investing.com): Pre-earnings focus on AI strategy
- Oct 30 (The Motley Fool): Early iPhone 17 sales reportedly 14% higher

Sentiment Overview (7d)
- Skew: Notably positive
"""
    report_path = report_dir / "report.md"
    report_path.write_text(report_content, encoding="utf-8")

    result = _read_ticker_report_summary("cid-stocks-us-xnas-aapl", stock_dir)

    assert result is not None
    assert result["slug"] == "cid-stocks-us-xnas-aapl"
    assert "TL;DR" in result["summary_text"]
    assert "AAPL rose" in result["summary_text"]
    assert "Notable Events" in result["summary_text"]
    assert "Benzinga" in result["summary_text"]


def test_read_ticker_report_summary_missing_file(tmp_path: Path):
    """Test reading non-existent report returns None."""
    stock_dir = tmp_path / "stocks"
    result = _read_ticker_report_summary("cid-stocks-us-xnas-aapl", stock_dir)
    assert result is None


def test_collect_ticker_news_summaries_node(tmp_path: Path):
    """Test collecting news summaries from multiple ticker reports."""
    stock_dir = tmp_path / "stocks"

    # Create two ticker reports
    for ticker_slug in ["cid-stocks-us-xnas-aapl", "cid-stocks-us-xnas-tsla"]:
        report_dir = stock_dir / "tickers" / ticker_slug / "report" / "7d"
        report_dir.mkdir(parents=True)
        report_path = report_dir / "report.md"
        report_path.write_text(
            f"TL;DR\n- {ticker_slug} summary\n\nNotable News & Events\n- Event 1",
            encoding="utf-8",
        )

    settings = Settings(input_dir=str(tmp_path), output_dir=str(tmp_path))
    state = {
        "settings": settings,
        "basket": {"id": "test", "label": "Test", "slug": "test"},
        "metrics": {
            "instruments": [
                {
                    "instrument_id": "cid:stocks:us:XNAS:AAPL",
                    "primary_ticker": "AAPL",
                    "d1": 0.01,
                    "d5": 0.02,
                },
                {
                    "instrument_id": "cid:stocks:us:XNAS:TSLA",
                    "primary_ticker": "TSLA",
                    "d1": 0.03,
                    "d5": 0.04,
                },
            ]
        },
    }

    result = collect_ticker_news_summaries_node(state)

    assert "_ticker_news_summaries" in result
    summaries = result["_ticker_news_summaries"]
    assert len(summaries) == 2
    assert summaries[0]["ticker"] == "AAPL"
    assert summaries[1]["ticker"] == "TSLA"
    assert "TL;DR" in summaries[0]["summary_text"]


def test_build_narrative_prompt():
    """Test narrative prompt construction includes key sections."""
    basket_label = "AI Picks"
    basket_slug = "ai-picks"
    metrics = {
        "basket": {"id": "ai_picks", "label": "AI Picks", "slug": "ai-picks"},
        "averages": {"d1": 0.011, "d5": -0.013},
        "instruments": [
            {
                "instrument_id": "cid:stocks:us:XNAS:AAPL",
                "primary_ticker": "AAPL",
                "d1": 0.01,
                "d5": 0.02,
            },
            {
                "instrument_id": "cid:stocks:us:XNAS:TSLA",
                "primary_ticker": "TSLA",
                "d1": 0.03,
                "d5": -0.05,
            },
        ],
        "top_movers": {
            "d5_up": ["cid:stocks:us:XNAS:AAPL"],
            "d5_down": ["cid:stocks:us:XNAS:TSLA"],
        },
    }
    ticker_summaries = [
        {
            "ticker": "AAPL",
            "slug": "cid-stocks-us-xnas-aapl",
            "summary_text": "Strong earnings drove gains",
        }
    ]

    prompt = _build_narrative_prompt(basket_label, basket_slug, metrics, ticker_summaries)

    assert "AI Picks" in prompt
    assert "Performance Summary" in prompt
    assert "+1.10%" in prompt  # d1 average
    assert "Top Movers" in prompt
    assert "AAPL" in prompt
    assert "TSLA" in prompt
    assert "Strong earnings" in prompt
    assert "Synthesize basket-level themes" in prompt


def test_generate_basket_narrative_node_with_llm(monkeypatch, tmp_path: Path):
    """Test narrative generation with mocked LLM."""
    mock_llm = MagicMock()
    mock_response = MagicMock()
    mock_response.content = "## Basket Themes\n- Tech earnings drove mixed results"
    mock_llm.invoke.return_value = mock_response

    def mock_get_llm(settings):
        return mock_llm

    monkeypatch.setattr("portfolio_advisor.agents.basket_narrative.get_llm", mock_get_llm)

    settings = Settings(input_dir=str(tmp_path), output_dir=str(tmp_path))
    state = {
        "settings": settings,
        "basket": {"id": "test", "label": "Test Basket", "slug": "test-basket"},
        "metrics": {
            "averages": {"d1": 0.01, "d5": 0.02},
            "instruments": [
                {
                    "instrument_id": "cid:stocks:us:XNAS:AAPL",
                    "primary_ticker": "AAPL",
                    "d1": 0.01,
                    "d5": 0.02,
                }
            ],
            "top_movers": {"d5_up": ["cid:stocks:us:XNAS:AAPL"], "d5_down": []},
        },
        "_ticker_news_summaries": [
            {"ticker": "AAPL", "slug": "aapl", "summary_text": "Earnings beat"}
        ],
    }

    result = generate_basket_narrative_node(state)

    assert "narrative_md" in result
    assert "Basket Themes" in result["narrative_md"]
    assert "Tech earnings" in result["narrative_md"]
    mock_llm.invoke.assert_called_once()


def test_generate_basket_narrative_node_no_news(tmp_path: Path):
    """Test narrative generation falls back gracefully when no news available."""
    settings = Settings(input_dir=str(tmp_path), output_dir=str(tmp_path))
    state = {
        "settings": settings,
        "basket": {"id": "test", "label": "Test Basket", "slug": "test-basket"},
        "metrics": {
            "averages": {"d1": 0.01, "d5": 0.02},
            "instruments": [],
            "top_movers": {},
        },
        "_ticker_news_summaries": [],  # No news
    }

    result = generate_basket_narrative_node(state)

    assert "narrative_md" in result
    assert "Test Basket" in result["narrative_md"]
    assert "1-day return" in result["narrative_md"]
    assert "not available" in result["narrative_md"]


def test_generate_basket_narrative_node_no_metrics(tmp_path: Path):
    """Test narrative generation skips when no metrics available."""
    settings = Settings(input_dir=str(tmp_path), output_dir=str(tmp_path))
    state = {
        "settings": settings,
        "basket": {"id": "test", "label": "Test Basket", "slug": "test-basket"},
        "metrics": {},  # No metrics
        "_ticker_news_summaries": [],
    }

    result = generate_basket_narrative_node(state)

    # Should return state unchanged
    assert "narrative_md" not in result or result.get("narrative_md") is None
