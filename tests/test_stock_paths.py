from __future__ import annotations

from pathlib import Path

from portfolio_advisor.stocks.db import StockPaths


def test_stock_paths_generates_expected_structure(tmp_path: Path):
    paths = StockPaths(root=tmp_path / "stocks")
    slug = "cid-stocks-us-test"

    assert paths.ticker_dir(slug) == tmp_path / "stocks" / "tickers" / slug
    assert paths.meta_json(slug).suffix == ".json"
    assert paths.primary_ohlc_json(slug).as_posix().endswith("primary/ohlc_daily.json")
    assert paths.news_article_content(slug, "article", "txt").suffix == ".txt"


def test_ensure_ticker_scaffold_creates_directories(tmp_path: Path):
    from portfolio_advisor.stocks.db import ensure_ticker_scaffold

    paths = StockPaths(root=tmp_path / "stocks")
    slug = "cid-stocks-us-test"

    ensure_ticker_scaffold(paths, slug)

    base = paths.ticker_dir(slug)
    assert (base / "primary").exists()
    assert (base / "analysis").exists()
    assert (base / "report").exists()
