import json
from pathlib import Path

import pytest

from portfolio_advisor.agents.news_summary import summarize_news_node
from portfolio_advisor.agents.stock_report_collator import collate_report_node
from portfolio_advisor.config import Settings
from portfolio_advisor.stocks.db import StockPaths, ensure_ticker_scaffold


@pytest.fixture()
def settings(tmp_path: Path) -> Settings:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    s = Settings(input_dir=str(input_dir), output_dir=str(output_dir))
    s.ensure_directories()
    return s


def _mk_state(settings: Settings, slug: str, ticker: str) -> dict:
    return {
        "settings": settings,
        "instrument": {"instrument_id": slug, "primary_ticker": ticker},
        "_slug": slug,
        "_paths": StockPaths(root=Path(settings.output_dir) / "stocks"),
    }


def test_summarize_news_node_minimal(settings: Settings, tmp_path: Path):
    slug = "cid-stocks-us-xnas-aapl"
    ticker = "AAPL"
    state = _mk_state(settings, slug, ticker)
    # Provide a minimal news_items list
    state["news_items"] = [
        {
            "id": "art1",
            "title": "Apple announces event",
            "summary": "Apple to host product event next week",
            "sentiment": {"label": "positive", "score": 0.6},
            "publisher": "Example",
            "published_utc": "2025-10-01T12:00:00Z",
        }
    ]

    out = summarize_news_node(state)
    assert "news_summary" in out
    ns = out["news_summary"]
    assert isinstance(ns.get("markdown"), str)
    # Accept either JSON-parsed or markdown-only fallback
    if ns.get("json"):
        assert ns["json"].get("ticker") in (ticker, None)


def test_collate_report_node_writes_outputs(settings: Settings, tmp_path: Path):
    slug = "cid-stocks-us-xnas-aapl"
    ticker = "AAPL"
    paths = StockPaths(root=Path(settings.output_dir) / "stocks")
    ensure_ticker_scaffold(paths, slug)

    # Seed minimal analysis artifacts
    (paths.analysis_returns_json(slug)).write_text(
        json.dumps({"as_of": "2025-10-07", "windows": {"d1": 0.01, "d5": 0.03}}),
        encoding="utf-8",
    )

    state = _mk_state(settings, slug, ticker)
    # Enable report generation
    settings.include_news_report = True

    # Provide a minimal news_summary
    state["news_summary"] = {
        "markdown": "## Notable News & Events\n- Apple event announced",
        "json": {
            "ticker": ticker,
            "slug": slug,
            "window_days": 7,
            "sentiment_overview": {
                "overall_label": "positive",
                "counts": {"positive": 1, "neutral": 0, "negative": 0},
            },
            "notable_events": [{"date": "2025-10-01", "title": "Event", "sentiment": "positive"}],
            "highlights_markdown": "## Notable News & Events\n- Apple event announced",
        },
    }

    out = collate_report_node(state)
    artifacts = out.get("artifacts", {}).get("news_report_7d")
    assert artifacts is not None
    report_path = Path(artifacts["report_path"])  # type: ignore[index]
    metrics_path = Path(artifacts["metrics_path"])  # type: ignore[index]

    assert report_path.exists()
    assert metrics_path.exists()

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert metrics.get("slug") == slug
    assert metrics.get("has_news") is True
