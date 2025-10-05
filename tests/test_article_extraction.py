from __future__ import annotations

import json
from pathlib import Path

import pytest

from portfolio_advisor.stocks.article_extraction import (
    ArticleTextExtractionService,
    clean_html,
    extract_article_section,
)
from portfolio_advisor.stocks.db import StockPaths


class StubOllama:
    def __init__(self, *, text: str = "extracted text", available: bool = True):
        self.text = text
        self.available = available
        self.models = {"milkey/reader-lm-v2:Q8_0"}
        self.generate_calls: list[dict] = []

    def generate(self, **kwargs):  # noqa: D401 - simple stub
        self.generate_calls.append(kwargs)
        return self.text

    def is_available(self) -> bool:
        return self.available

    def model_exists(self, model: str) -> bool:
        return model in self.models

    def list_models(self) -> list[str]:
        return list(self.models)


@pytest.fixture
def stock_paths(tmp_path: Path) -> StockPaths:
    root = tmp_path / "stocks"
    root.mkdir(parents=True, exist_ok=True)
    return StockPaths(root=root)


def _prepare_article(paths: StockPaths, ticker_slug: str, article_id: str, html: str) -> None:
    news_dir = paths.news_dir(ticker_slug)
    news_dir.mkdir(parents=True, exist_ok=True)

    articles_dir = paths.news_articles_dir(ticker_slug)
    articles_dir.mkdir(parents=True, exist_ok=True)

    html_path = articles_dir / f"{article_id}.html"
    html_path.write_text(html, encoding="utf-8")

    article_json = paths.news_article_json(ticker_slug, article_id)
    article_json.write_text(
        json.dumps(
            {
                "id": article_id,
                "local_content": {"content_path": f"articles/{article_id}.html"},
            }
        ),
        encoding="utf-8",
    )


def test_clean_html_strips_noise():
    html = """
    <html>
        <head>
            <script>bad()</script>
            <style>body{}</style>
            <meta charset="utf-8" />
            <link rel="stylesheet" href="style.css" />
        </head>
        <body>
            <!-- comment -->
            <div>Content<img src="data:image/png;base64,AAA" /></div>
            <svg><g>vector</g></svg>
        </body>
    </html>
    """

    cleaned = clean_html(html)

    assert "script" not in cleaned
    assert "style" not in cleaned
    assert "meta" not in cleaned
    assert "link" not in cleaned
    assert "[SVG placeholder]" in cleaned
    assert "data:image" not in cleaned


def test_extract_article_section_prefers_specific_containers():
    html = """
    <div class="article-body" id="main-body-container">
        <p>Main content</p>
    </div>
    <div class="main-tags">Tags</div>
    """

    section = extract_article_section(html)
    assert "Main content" in section

    fallback_html = """
    <div itemprop="articleBody">
        <p>Fallback content</p>
    </div>
    <div class="related">Related</div>
    """
    fallback = extract_article_section(fallback_html)
    assert "Fallback content" in fallback


def test_extract_article_text_writes_output(stock_paths: StockPaths):
    ticker_slug = "cid-test"
    article_id = "article1"

    _prepare_article(
        stock_paths,
        ticker_slug,
        article_id,
        "<html><body><h1>Headline</h1><p>Paragraph</p></body></html>",
    )

    ollama = StubOllama(text="Headline\nParagraph")
    service = ArticleTextExtractionService(paths=stock_paths, ollama_service=ollama)

    stats = service.extract_article_text(ticker_slug, article_id)

    assert stats["success"] is True
    assert stats["extracted_chars"] == len("Headline\nParagraph")
    assert ollama.generate_calls

    text_path = stock_paths.news_article_content(ticker_slug, article_id, "txt")
    assert text_path.exists()
    assert text_path.read_text(encoding="utf-8") == "Headline\nParagraph"

    article_json = stock_paths.news_article_json(ticker_slug, article_id)
    with article_json.open("r", encoding="utf-8") as fh:
        article_data = json.load(fh)

    assert article_data["text_extracted"] is True
    assert article_data["text_content"]["content_path"].endswith(".txt")


def test_extract_article_text_skips_when_already_extracted(stock_paths: StockPaths):
    ticker_slug = "cid-test"
    article_id = "already_done"
    _prepare_article(stock_paths, ticker_slug, article_id, "<html></html>")

    article_json = stock_paths.news_article_json(ticker_slug, article_id)
    with article_json.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    data["text_extracted"] = True
    article_json.write_text(json.dumps(data), encoding="utf-8")

    ollama = StubOllama()
    service = ArticleTextExtractionService(paths=stock_paths, ollama_service=ollama)

    stats = service.extract_article_text(ticker_slug, article_id)

    assert stats["skipped"] is True
    assert stats["success"] is True
    assert ollama.generate_calls == []


def test_extract_all_articles_collects_stats(stock_paths: StockPaths):
    ticker_slug = "cid-test"
    index_path = stock_paths.news_index_json(ticker_slug)
    index_path.parent.mkdir(parents=True, exist_ok=True)

    _prepare_article(
        stock_paths,
        ticker_slug,
        "new_article",
        "<html><body><h1>Headline</h1></body></html>",
    )
    _prepare_article(
        stock_paths,
        ticker_slug,
        "skipped_article",
        "<html></html>",
    )

    skipped_json = stock_paths.news_article_json(ticker_slug, "skipped_article")
    with skipped_json.open("r", encoding="utf-8") as fh:
        skipped_data = json.load(fh)
    skipped_data["text_extracted"] = True
    skipped_json.write_text(json.dumps(skipped_data), encoding="utf-8")

    index_path.write_text(
        json.dumps(
            {
                "articles": {
                    "new_article": {},
                    "skipped_article": {},
                }
            }
        ),
        encoding="utf-8",
    )

    service = ArticleTextExtractionService(paths=stock_paths, ollama_service=StubOllama())

    stats = service.extract_all_articles(ticker_slug, batch_size=1)

    assert stats["total_articles"] == 2
    assert stats["extracted"] == 1
    assert stats["skipped"] == 1
    assert stats["errors"] == 0

    with index_path.open("r", encoding="utf-8") as fh:
        updated_index = json.load(fh)
    assert updated_index["last_extraction_run"]


def test_extract_portfolio_articles_handles_availability(stock_paths: StockPaths):
    ticker_slug = "cid-test"
    portfolio_root = stock_paths.root.parent
    ticker_dir = portfolio_root / "stocks" / "tickers" / ticker_slug / "primary" / "news"
    ticker_dir.mkdir(parents=True, exist_ok=True)

    index_path = ticker_dir / "index.json"
    index_path.write_text(json.dumps({"articles": {}}), encoding="utf-8")

    service = ArticleTextExtractionService(paths=stock_paths, ollama_service=StubOllama())

    stats_unavailable = ArticleTextExtractionService(
        paths=stock_paths, ollama_service=StubOllama(available=False)
    ).extract_portfolio_articles(portfolio_root, max_workers=1)
    assert stats_unavailable["errors"] == stats_unavailable["total_tickers"]

    stats = service.extract_portfolio_articles(portfolio_root, max_workers=1)
    assert stats["total_tickers"] == 1
    assert stats["processed_tickers"] == 1
