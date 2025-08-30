from __future__ import annotations

import os
from pathlib import Path

from portfolio_advisor import analyze_portfolio


def test_analyze_portfolio_smoke(tmp_path: Path, monkeypatch):
    input_dir = tmp_path / "in"
    output_dir = tmp_path / "out"
    input_dir.mkdir()
    (input_dir / "positions.csv").write_text("symbol,shares\nAAPL,10\n", encoding="utf-8")

    # Ensure no network call path is used
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    output_path = analyze_portfolio(str(input_dir), str(output_dir), log_level="DEBUG")
    assert os.path.exists(output_path)
    content = Path(output_path).read_text(encoding="utf-8")
    assert "# Portfolio Analysis Report" in content
    assert "positions.csv" in content
