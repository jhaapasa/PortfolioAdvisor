from __future__ import annotations

import json
from pathlib import Path

from portfolio_advisor.config import Settings
from portfolio_advisor.graphs.baskets import build_basket_graph


def test_baskets_metrics_and_outputs(tmp_path: Path):
    out_dir = tmp_path / "out"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Prepare minimal returns.json under instrument_id slugs (fallback to ticker for this test)
    base = out_dir / "stocks" / "tickers"
    for t, d5 in ("AAPL", 0.02), ("NVDA", -0.01):
        tdir = base / t / "analysis"
        tdir.mkdir(parents=True, exist_ok=True)
        content = {"primary_ticker": t, "as_of": "2025-09-15", "windows": {"d5": d5}}
        (tdir / "returns.json").write_text(json.dumps(content), encoding="utf-8")

    settings = Settings(input_dir=str(tmp_path), output_dir=str(out_dir))
    compiled = build_basket_graph()
    state = {
        "settings": settings,
        "basket": {
            "id": "growth_tech",
            "label": "Growth Tech",
            "slug": "growth-tech",
            "tickers": ["AAPL", "NVDA"],
        },
    }
    out = compiled.invoke(state)
    rep = out.get("basket_report")
    assert rep and rep["slug"] == "growth-tech"
    assert Path(rep["metrics_path"]).exists()
    assert Path(rep["report_path"]).exists()


def test_baskets_with_news_reports(tmp_path: Path):
    """Test basket graph incorporates individual ticker news reports."""
    out_dir = tmp_path / "out"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Prepare returns.json and 7d reports for tickers
    base = out_dir / "stocks" / "tickers"
    for ticker, d1, d5, news_summary in [
        ("AAPL", 0.01, 0.03, "Strong iPhone sales drove gains"),
        ("TSLA", -0.02, -0.05, "Production concerns weighed on shares"),
    ]:
        tdir = base / ticker
        analysis_dir = tdir / "analysis"
        analysis_dir.mkdir(parents=True, exist_ok=True)
        returns_content = {
            "primary_ticker": ticker,
            "as_of": "2025-10-31",
            "windows": {"d1": d1, "d5": d5},
        }
        (analysis_dir / "returns.json").write_text(json.dumps(returns_content), encoding="utf-8")

        # Create 7d report
        report_dir = tdir / "report" / "7d"
        report_dir.mkdir(parents=True, exist_ok=True)
        report_content = f"""TL;DR
- {news_summary}

Notable News & Events
- Event 1 for {ticker}
- Event 2 for {ticker}

Sentiment Overview (7d)
- Overall: Positive
"""
        (report_dir / "report.md").write_text(report_content, encoding="utf-8")

    settings = Settings(input_dir=str(tmp_path), output_dir=str(out_dir))
    compiled = build_basket_graph()
    state = {
        "settings": settings,
        "basket": {
            "id": "test_basket",
            "label": "Test Basket",
            "slug": "test-basket",
            "tickers": ["AAPL", "TSLA"],
        },
    }
    out = compiled.invoke(state)

    # Verify report was generated
    rep = out.get("basket_report")
    assert rep and rep["slug"] == "test-basket"
    report_path = Path(rep["report_path"])
    assert report_path.exists()

    # Verify report contains narrative elements
    report_text = report_path.read_text(encoding="utf-8")
    assert "Test Basket" in report_text
    assert "Performance Summary" in report_text
    assert "Top Movers" in report_text

    # Verify metrics file
    metrics_path = Path(rep["metrics_path"])
    assert metrics_path.exists()
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert metrics["basket"]["id"] == "test_basket"
    assert "averages" in metrics
    assert "top_movers" in metrics


def test_baskets_with_ticker_only_definitions(tmp_path: Path):
    """Test that ticker-only baskets get proper composite instrument IDs."""
    out_dir = tmp_path / "out"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Create multiple tickers with different returns
    # When defined with tickers only, they should get composite IDs
    base = out_dir / "stocks" / "tickers"
    tickers_data = [
        ("AAPL", 0.05, 0.10),  # Top gainer
        ("TSLA", 0.02, 0.04),  # Mid performer
        ("NVDA", -0.01, -0.03),  # Top loser
        ("AMD", 0.01, 0.02),  # Another performer
    ]

    # Create data under composite ID slugs since that's what basket will use
    for ticker, d1, d5 in tickers_data:
        # Composite ID: cid:stocks:us:composite:TICKER -> cid-stocks-us-composite-{ticker}
        slug = f"cid-stocks-us-composite-{ticker.lower()}"
        tdir = base / slug / "analysis"
        tdir.mkdir(parents=True, exist_ok=True)
        returns_content = {
            "primary_ticker": ticker,
            "as_of": "2025-11-01",
            "windows": {"d1": d1, "d5": d5},
        }
        (tdir / "returns.json").write_text(json.dumps(returns_content), encoding="utf-8")

    settings = Settings(input_dir=str(tmp_path), output_dir=str(out_dir))
    compiled = build_basket_graph()
    state = {
        "settings": settings,
        "basket": {
            "id": "ticker_only_basket",
            "label": "Ticker Only Basket",
            "slug": "ticker-only-basket",
            "tickers": [t for t, _, _ in tickers_data],  # Ticker-only definition
        },
    }
    out = compiled.invoke(state)

    # Verify metrics were computed correctly
    metrics_path = Path(out["basket_report"]["metrics_path"])
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

    # Verify all instruments have proper composite IDs (not None)
    instruments = metrics["instruments"]
    assert len(instruments) == 4
    for inst in instruments:
        iid = inst["instrument_id"]
        assert iid is not None
        assert iid.startswith("cid:stocks:us:composite:")
        # Verify instrument_id matches the ticker
        ticker = inst["primary_ticker"]
        assert iid == f"cid:stocks:us:composite:{ticker}"

    # Verify top movers include correct instrument IDs
    top_movers = metrics["top_movers"]
    assert len(top_movers["d5_up"]) == 3  # Should have 3 upward movers
    assert len(top_movers["d5_down"]) == 3  # Should have 3 downward movers
    # Top movers should contain instrument IDs, not None
    for mover_id in top_movers["d5_up"] + top_movers["d5_down"]:
        assert mover_id is not None
        assert mover_id.startswith("cid:stocks:us:composite:")

    # Read the report and verify all tickers appear
    report_path = Path(out["basket_report"]["report_path"])
    report_text = report_path.read_text(encoding="utf-8")
    assert "AAPL" in report_text  # Top gainer should appear
    assert "NVDA" in report_text  # Top loser should appear
    assert "Top Movers" in report_text
