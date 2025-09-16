from __future__ import annotations

from pathlib import Path

from portfolio_advisor.analyze import analyze_portfolio


def test_cache_db_created_default(tmp_path, monkeypatch):
    in_dir = tmp_path / "in"
    out_dir = tmp_path / "out"
    in_dir.mkdir()
    (in_dir / "positions.csv").write_text("symbol,shares\nAAPL,10\n", encoding="utf-8")

    # Ensure stub LLM path
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("SKIP_LLM_CACHE", raising=False)

    analyze_portfolio(str(in_dir), str(out_dir))

    cache_path = Path("./cache/langchain_cache.sqlite3")
    assert cache_path.exists()


def test_skip_cache_bypasses_lookup_but_writes(tmp_path, monkeypatch):
    in_dir = tmp_path / "in"
    out_dir = tmp_path / "out"
    in_dir.mkdir()
    (in_dir / "positions.csv").write_text("symbol,shares\nAAPL,10\n", encoding="utf-8")

    # Ensure stub LLM path and enable skip mode
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("SKIP_LLM_CACHE", "1")

    analyze_portfolio(str(in_dir), str(out_dir))

    # Verify read-bypass adapter behavior and writable inner cache
    from langchain_core.globals import get_llm_cache

    cache = get_llm_cache()
    # Lookup should bypass and return None
    assert cache.lookup("prompt", "llm_string") is None
    # Update should persist in underlying cache
    cache.update("prompt", "llm_string", {"result": "ok"})
    # Access inner cache if present to validate write occurred
    inner = getattr(cache, "_inner", cache)
    assert inner.lookup("prompt", "llm_string") is not None
