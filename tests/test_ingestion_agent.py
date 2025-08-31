from __future__ import annotations

from pathlib import Path

from portfolio_advisor.agents.ingestion import ingestion_node


def test_ingestion_basic_types(tmp_path: Path):
    in_dir = tmp_path / "in"
    in_dir.mkdir()

    (in_dir / "a.txt").write_text("hello world", encoding="utf-8")
    (in_dir / "b.md").write_text("# Title\nBody", encoding="utf-8")
    (in_dir / "c.csv").write_text("symbol,shares\nAAPL,10", encoding="utf-8")
    (in_dir / "d.html").write_text("<html><body><p>hi</p></body></html>", encoding="utf-8")

    class S:
        input_dir = str(in_dir)

    state = {"settings": S()}
    out = ingestion_node(state)
    docs = out.get("raw_docs", [])

    names = [d["name"] for d in docs]
    assert names == ["a.txt", "b.md", "c.csv", "d.html"]
    assert any(d["mime_type"].startswith("text/") for d in docs)
    assert all("as_text" in d for d in docs)


def test_ingestion_truncation(tmp_path: Path):
    in_dir = tmp_path / "in"
    in_dir.mkdir()
    big = in_dir / "big.txt"
    big.write_text("x" * (3 * 1024 * 1024), encoding="utf-8")  # 3 MiB

    class S:
        input_dir = str(in_dir)

    out = ingestion_node({"settings": S()})
    docs = out["raw_docs"]
    meta = docs[0]["metadata"]
    assert meta.get("truncated") is True


