from __future__ import annotations

import json
from pathlib import Path

from portfolio_advisor.config import Settings
from portfolio_advisor.graphs.baskets import build_basket_graph


def test_baskets_metrics_and_outputs(tmp_path: Path):
    out_dir = tmp_path / "out"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Prepare minimal returns.json for two tickers
    base = out_dir / "stocks" / "tickers"
    for t, d5 in ("AAPL", 0.02), ("NVDA", -0.01):
        tdir = base / t / "analysis"
        tdir.mkdir(parents=True, exist_ok=True)
        content = {"ticker": t, "as_of": "2025-09-15", "windows": {"d5": d5}}
        (tdir / "returns.json").write_text(json.dumps(content), encoding="utf-8")

    settings = Settings(input_dir=str(tmp_path), output_dir=str(out_dir))
    compiled = build_basket_graph()
    state = {
        "settings": settings,
        "basket": {"id": "growth_tech", "label": "Growth Tech", "slug": "growth-tech", "tickers": ["AAPL", "NVDA"]},
    }
    out = compiled.invoke(state)
    rep = out.get("basket_report")
    assert rep and rep["slug"] == "growth-tech"
    assert Path(rep["metrics_path"]).exists()
    assert Path(rep["report_path"]).exists()


