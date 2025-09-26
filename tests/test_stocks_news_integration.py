from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from portfolio_advisor.config import Settings
from portfolio_advisor.graphs.stocks import _fetch_primary_node, StockState
from portfolio_advisor.stocks.db import StockPaths


class TestStocksNewsIntegration:
    """Test news fetching integration in the stocks graph."""

    @pytest.fixture
    def mock_settings(self, tmp_path):
        """Create mock settings for testing."""
        return Settings(
            input_dir=str(tmp_path / "input"),
            output_dir=str(tmp_path / "output"),
            polygon_api_key="test_key",
            fetch_news=True,  # Enable news fetching
        )

    @pytest.fixture
    def stock_state(self, mock_settings):
        """Create a stock state for testing."""
        return StockState(
            settings=mock_settings,
            instrument={
                "instrument_id": "cid-stocks-us-xnas-aapl",
                "primary_ticker": "AAPL",
            },
            _slug="cid-stocks-us-xnas-aapl",
            _paths=StockPaths(root=Path(mock_settings.output_dir) / "stocks"),
        )

    @patch("portfolio_advisor.graphs.stocks.PolygonClient")
    @patch("portfolio_advisor.graphs.stocks.StockNewsService")
    def test_fetch_primary_with_news_enabled(
        self, mock_news_service_class, mock_polygon_client_class, stock_state, mock_settings
    ):
        """Test that news is fetched when enabled in settings."""
        # Setup mocks
        mock_polygon_instance = MagicMock()
        mock_polygon_client_class.return_value.__enter__.return_value = mock_polygon_instance
        mock_polygon_instance.list_aggs_daily.return_value = [
            {"date": "2024-09-25", "open": 150.0, "close": 152.0, "volume": 1000000}
        ]

        mock_news_instance = MagicMock()
        mock_news_service_class.return_value.__enter__.return_value = mock_news_instance
        mock_news_instance.update_ticker_news.return_value = {
            "articles_fetched": 5,
            "articles_new": 3,
            "articles_downloaded": 2,
            "errors": [],
        }

        # Execute
        _fetch_primary_node(stock_state)

        # Verify news service was called
        mock_news_service_class.assert_called_once()
        mock_news_instance.update_ticker_news.assert_called_once_with(
            ticker_slug="cid-stocks-us-xnas-aapl", ticker_symbol="AAPL", days_back=7
        )

    @patch("portfolio_advisor.graphs.stocks.PolygonClient")
    @patch("portfolio_advisor.graphs.stocks.StockNewsService")
    def test_fetch_primary_with_news_disabled(
        self, mock_news_service_class, mock_polygon_client_class, stock_state, mock_settings
    ):
        """Test that news is not fetched when disabled in settings."""
        # Disable news fetching
        mock_settings.fetch_news = False
        stock_state["settings"] = mock_settings

        # Setup polygon mock
        mock_polygon_instance = MagicMock()
        mock_polygon_client_class.return_value.__enter__.return_value = mock_polygon_instance
        mock_polygon_instance.list_aggs_daily.return_value = []

        # Execute
        _fetch_primary_node(stock_state)

        # Verify news service was NOT called
        mock_news_service_class.assert_not_called()

    @patch("portfolio_advisor.graphs.stocks.PolygonClient")
    @patch("portfolio_advisor.graphs.stocks.StockNewsService")
    def test_fetch_primary_news_error_handling(
        self, mock_news_service_class, mock_polygon_client_class, stock_state
    ):
        """Test that news fetching errors don't break the pipeline."""
        # Setup polygon mock
        mock_polygon_instance = MagicMock()
        mock_polygon_client_class.return_value.__enter__.return_value = mock_polygon_instance
        mock_polygon_instance.list_aggs_daily.return_value = []

        # Setup news service to raise error
        mock_news_instance = MagicMock()
        mock_news_service_class.return_value.__enter__.return_value = mock_news_instance
        mock_news_instance.update_ticker_news.side_effect = Exception("News API Error")

        # Execute - should not raise
        result = _fetch_primary_node(stock_state)

        # Verify it completed without error
        assert result == {}
        mock_news_instance.update_ticker_news.assert_called_once()

    @patch("portfolio_advisor.graphs.stocks.PolygonClient")
    @patch("portfolio_advisor.graphs.stocks.StockNewsService")
    def test_fetch_primary_no_ticker(
        self, mock_news_service_class, mock_polygon_client_class, stock_state
    ):
        """Test that news is not fetched when ticker is missing."""
        # Remove ticker
        stock_state["instrument"]["primary_ticker"] = ""

        # Setup polygon mock
        mock_polygon_instance = MagicMock()
        mock_polygon_client_class.return_value.__enter__.return_value = mock_polygon_instance
        mock_polygon_instance.list_aggs_daily.return_value = []

        # Execute
        _fetch_primary_node(stock_state)

        # Verify news service was NOT called (no ticker)
        mock_news_service_class.assert_not_called()

    def test_news_paths_integration(self, tmp_path):
        """Test that news paths are correctly integrated with StockPaths."""
        paths = StockPaths(root=tmp_path / "stocks")
        ticker = "cid-stocks-us-xnas-aapl"

        # Test all news-related paths
        news_dir = paths.news_dir(ticker)
        assert news_dir == tmp_path / "stocks" / "tickers" / ticker / "primary" / "news"

        index_path = paths.news_index_json(ticker)
        assert index_path == news_dir / "index.json"

        article_path = paths.news_article_json(ticker, "article123_2024-09-25")
        assert article_path == news_dir / "article123_2024-09-25.json"

        articles_dir = paths.news_articles_dir(ticker)
        assert articles_dir == news_dir / "articles"

        content_path = paths.news_article_content(ticker, "article123_2024-09-25", "html")
        assert content_path == articles_dir / "article123_2024-09-25.html"
