from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import httpx
import pytest

from portfolio_advisor.services.polygon_client import PolygonClient
from portfolio_advisor.stocks.db import StockPaths
from portfolio_advisor.stocks.news import StockNewsService


class TestPolygonClientNews:
    """Test PolygonClient news-related functionality."""

    def test_list_ticker_news_basic(self):
        """Test basic news listing functionality."""
        mock_client = MagicMock()
        mock_article = MagicMock()
        mock_article.model_dump.return_value = {
            "id": "abc123",
            "publisher": {"name": "Test Publisher"},
            "title": "Test Article",
            "author": "Test Author",
            "published_utc": "2024-09-25T10:00:00Z",
            "article_url": "https://example.com/article",
            "tickers": ["AAPL"],
        }
        mock_client.list_ticker_news.return_value = [mock_article]

        with patch.object(PolygonClient, "_ensure_client", return_value=mock_client):
            client = PolygonClient()
            articles = list(
                client.list_ticker_news(
                    ticker="AAPL",
                    published_utc_gte="2024-09-20",
                    published_utc_lte="2024-09-26",
                )
            )

        assert len(articles) == 1
        assert articles[0]["id"] == "abc123"
        assert articles[0]["title"] == "Test Article"
        mock_client.list_ticker_news.assert_called_once_with(
            ticker="AAPL",
            published_utc_gte="2024-09-20",
            published_utc_lte="2024-09-26",
            limit=1000,
            order="desc",
        )

    def test_list_ticker_news_with_dict_response(self):
        """Test news listing when API returns dicts instead of objects."""
        mock_client = MagicMock()
        mock_client.list_ticker_news.return_value = [
            {
                "id": "xyz789",
                "title": "Dict Article",
                "published_utc": "2024-09-24T14:30:00Z",
            }
        ]

        with patch.object(PolygonClient, "_ensure_client", return_value=mock_client):
            client = PolygonClient()
            articles = list(client.list_ticker_news(ticker="MSFT"))

        assert len(articles) == 1
        assert articles[0]["id"] == "xyz789"
        assert articles[0]["title"] == "Dict Article"


class TestStockNewsService:
    """Test StockNewsService functionality."""

    @pytest.fixture
    def mock_paths(self, tmp_path):
        """Create a mock StockPaths instance."""
        return StockPaths(root=tmp_path / "stocks")

    @pytest.fixture
    def mock_polygon_client(self):
        """Create a mock PolygonClient."""
        return MagicMock(spec=PolygonClient)

    @pytest.fixture
    def news_service(self, mock_paths, mock_polygon_client):
        """Create a StockNewsService instance with mocks."""
        service = StockNewsService(mock_paths, mock_polygon_client)
        # Mock the HTTP client
        service.http_client = MagicMock(spec=httpx.Client)
        return service

    def test_update_ticker_news_no_existing_index(self, news_service, mock_paths, mock_polygon_client):
        """Test updating news when no index exists."""
        ticker_slug = "cid-stocks-us-xnas-aapl"
        ticker_symbol = "AAPL"

        # Mock news articles from Polygon
        mock_polygon_client.list_ticker_news.return_value = [
            {
                "id": "article1",
                "title": "Apple News 1",
                "published_utc": "2024-09-25T10:00:00Z",
                "article_url": "https://example.com/article1",
                "publisher": {"name": "Publisher 1"},
            },
            {
                "id": "article2",
                "title": "Apple News 2",
                "published_utc": "2024-09-24T15:00:00Z",
                "article_url": "https://example.com/article2",
                "publisher": {"name": "Publisher 2"},
            },
        ]

        # Mock successful article downloads
        mock_response = Mock()
        mock_response.content = b"<html>Article content</html>"
        mock_response.headers = {"content-type": "text/html"}
        mock_response.raise_for_status = Mock()
        news_service.http_client.get.return_value = mock_response

        # Execute update
        stats = news_service.update_ticker_news(ticker_slug, ticker_symbol, days_back=7)

        # Verify stats
        assert stats["articles_fetched"] == 2
        assert stats["articles_new"] == 2
        assert stats["articles_downloaded"] == 2
        assert stats["errors"] == []

        # Verify files were created
        news_dir = mock_paths.news_dir(ticker_slug)
        assert news_dir.exists()

        index_path = mock_paths.news_index_json(ticker_slug)
        assert index_path.exists()

        # Verify index content
        with index_path.open("r") as f:
            index = json.load(f)
        assert index["article_count"] == 2
        assert "article1_2024-09-25" in index["articles"]
        assert "article2_2024-09-24" in index["articles"]

        # Verify article metadata files
        article1_path = mock_paths.news_article_json(ticker_slug, "article1_2024-09-25")
        assert article1_path.exists()

        with article1_path.open("r") as f:
            article1_data = json.load(f)
        assert article1_data["title"] == "Apple News 1"
        assert "local_content" in article1_data
        assert article1_data["local_content"]["content_path"] == "articles/article1_2024-09-25.html"

    def test_update_ticker_news_with_existing_articles(
        self, news_service, mock_paths, mock_polygon_client
    ):
        """Test updating news when some articles already exist."""
        ticker_slug = "cid-stocks-us-xnas-aapl"
        ticker_symbol = "AAPL"

        # Create existing index
        news_dir = mock_paths.news_dir(ticker_slug)
        news_dir.mkdir(parents=True, exist_ok=True)
        index_path = mock_paths.news_index_json(ticker_slug)
        existing_index = {
            "last_updated": "2024-09-24T00:00:00Z",
            "article_count": 1,
            "articles": {
                "article1_2024-09-25": {
                    "id": "article1_2024-09-25",
                    "published_utc": "2024-09-25T10:00:00Z",
                    "title": "Existing Article",
                    "has_full_content": True,
                }
            },
        }
        with index_path.open("w") as f:
            json.dump(existing_index, f)

        # Mock news articles - one existing, one new
        mock_polygon_client.list_ticker_news.return_value = [
            {
                "id": "article1",
                "title": "Apple News 1",
                "published_utc": "2024-09-25T10:00:00Z",
                "article_url": "https://example.com/article1",
            },
            {
                "id": "article3",
                "title": "New Article",
                "published_utc": "2024-09-23T12:00:00Z",
                "article_url": "https://example.com/article3",
            },
        ]

        # Mock successful download for new article
        mock_response = Mock()
        mock_response.content = b"<html>New article content</html>"
        mock_response.headers = {"content-type": "text/html"}
        mock_response.raise_for_status = Mock()
        news_service.http_client.get.return_value = mock_response

        # Execute update
        stats = news_service.update_ticker_news(ticker_slug, ticker_symbol, days_back=7)

        # Verify stats
        assert stats["articles_fetched"] == 2
        assert stats["articles_new"] == 1  # Only the new article
        assert stats["articles_downloaded"] == 1
        assert stats["errors"] == []

        # Verify index was updated
        with index_path.open("r") as f:
            index = json.load(f)
        assert index["article_count"] == 2
        assert "article1_2024-09-25" in index["articles"]  # Existing
        assert "article3_2024-09-23" in index["articles"]  # New

    def test_download_article_failure(self, news_service, mock_paths):
        """Test handling of article download failures."""
        ticker_slug = "cid-stocks-us-xnas-aapl"
        article_id = "failed_article"

        # Mock failed download
        news_service.http_client.get.side_effect = httpx.HTTPError("Connection error")

        # Execute download
        result = news_service._download_article(ticker_slug, article_id, "https://example.com/fail")

        # Verify failure
        assert result is None
        news_service.http_client.get.assert_called_once()

    def test_download_article_different_content_types(self, news_service, mock_paths):
        """Test downloading articles with different content types."""
        ticker_slug = "cid-stocks-us-xnas-aapl"

        # Ensure articles directory exists
        articles_dir = mock_paths.news_articles_dir(ticker_slug)
        articles_dir.mkdir(parents=True, exist_ok=True)

        test_cases = [
            ("text/html; charset=utf-8", "html"),
            ("application/json", "json"),
            ("text/plain", "txt"),
            ("application/pdf", "txt"),  # Unknown defaults to txt
        ]

        for content_type, expected_ext in test_cases:
            mock_response = Mock()
            mock_response.content = b"Test content"
            mock_response.headers = {"content-type": content_type}
            mock_response.raise_for_status = Mock()
            news_service.http_client.get.return_value = mock_response

            article_id = f"article_{expected_ext}"
            result = news_service._download_article(
                ticker_slug, article_id, "https://example.com/test"
            )

            assert result is not None
            assert result.suffix == f".{expected_ext}"
            assert result.exists()
            assert result.read_bytes() == b"Test content"

    def test_polygon_api_error_handling(self, news_service, mock_paths, mock_polygon_client):
        """Test handling of Polygon API errors."""
        ticker_slug = "cid-stocks-us-xnas-aapl"
        ticker_symbol = "AAPL"

        # Mock API error
        mock_polygon_client.list_ticker_news.side_effect = Exception("API Error")

        # Execute update
        stats = news_service.update_ticker_news(ticker_slug, ticker_symbol)

        # Verify error handling
        assert stats["articles_fetched"] == 0
        assert stats["articles_new"] == 0
        assert stats["articles_downloaded"] == 0
        assert len(stats["errors"]) == 1
        assert "API Error" in stats["errors"][0]

    def test_context_manager(self, mock_paths, mock_polygon_client):
        """Test StockNewsService context manager functionality."""
        with patch("httpx.Client") as mock_http_client_class:
            mock_http_instance = MagicMock()
            mock_http_client_class.return_value = mock_http_instance

            with StockNewsService(mock_paths, mock_polygon_client) as service:
                assert service.http_client == mock_http_instance

            # Verify cleanup was called
            mock_http_instance.close.assert_called_once()
