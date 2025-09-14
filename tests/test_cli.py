from __future__ import annotations

from pathlib import Path

from portfolio_advisor.cli import main


def test_cli_success(tmp_path: Path, capsys, monkeypatch):
    input_dir = tmp_path / "in"
    output_dir = tmp_path / "out"
    input_dir.mkdir()
    (input_dir / "positions.csv").write_text("symbol,shares\nAAPL,10\n", encoding="utf-8")

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    rc = main(
        [
            "--input-dir",
            str(input_dir),
            "--output-dir",
            str(output_dir),
            "--log-level",
            "DEBUG",
        ]
    )
    assert rc == 0
    out = capsys.readouterr()
    output_path = Path(out.out.strip())
    assert output_path.exists()
    content = output_path.read_text(encoding="utf-8")
    assert "Portfolio Analysis Report" in content


def test_cli_missing_input_dir(tmp_path: Path, capsys):
    input_dir = tmp_path / "missing"
    output_dir = tmp_path / "out"
    rc = main(["--input-dir", str(input_dir), "--output-dir", str(output_dir)])
    assert rc == 1
    err = capsys.readouterr().err
    assert "Error:" in err
