from __future__ import annotations

import datetime as dt
import json
import logging
from pathlib import Path
from typing import Any

import httpx

from ..services.polygon_client import PolygonClient
from ..utils.fs import utcnow_iso, write_json_atomic
from .db import StockPaths

logger = logging.getLogger(__name__)


class StockNewsService:
    """Service for fetching and storing stock news articles."""

    def __init__(self, paths: StockPaths, polygon_client: PolygonClient):
        self.paths = paths
        self.polygon_client = polygon_client
        self.http_client = httpx.Client(timeout=30.0)

    def close(self):
        """Clean up resources."""
        self.http_client.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def update_ticker_news(
        self, ticker_slug: str, ticker_symbol: str, days_back: int = 7
    ) -> dict[str, Any]:
        """Fetch and store news for a ticker from the last N days.

        Returns summary of operation:
            {
                "articles_fetched": 10,
                "articles_new": 3,
                "articles_downloaded": 2,
                "errors": []
            }
        """
        # Calculate date range
        end_date = dt.datetime.now(dt.UTC)
        start_date = end_date - dt.timedelta(days=days_back)

        # Ensure directories exist
        news_dir = self.paths.news_dir(ticker_slug)
        articles_dir = self.paths.news_articles_dir(ticker_slug)
        news_dir.mkdir(parents=True, exist_ok=True)
        articles_dir.mkdir(parents=True, exist_ok=True)

        # Load existing index
        index_path = self.paths.news_index_json(ticker_slug)
        if index_path.exists():
            with index_path.open("r") as f:
                index = json.load(f)
        else:
            index = {"last_updated": None, "article_count": 0, "articles": {}}

        # Track operation stats
        stats = {
            "articles_fetched": 0,
            "articles_new": 0,
            "articles_downloaded": 0,
            "errors": [],
        }

        try:
            # Fetch news from Polygon
            for article in self.polygon_client.list_ticker_news(
                ticker=ticker_symbol,
                published_utc_gte=start_date.strftime("%Y-%m-%d"),
                published_utc_lte=end_date.strftime("%Y-%m-%d"),
                limit=1000,
                order="desc",
            ):
                stats["articles_fetched"] += 1

                # Create unique article ID
                published_date = article.get("published_utc", "unknown")[:10]
                article_id = f"{article['id']}_{published_date}"

                # Skip if already exists
                if article_id in index["articles"]:
                    continue

                stats["articles_new"] += 1

                # Save article metadata
                article_path = self.paths.news_article_json(ticker_slug, article_id)
                article_data = article.copy()

                # Try to download full article content
                if article_url := article.get("article_url"):
                    content_path = self._download_article(ticker_slug, article_id, article_url)
                    if content_path:
                        stats["articles_downloaded"] += 1
                        article_data["local_content"] = {
                            "downloaded_at": utcnow_iso(),
                            "content_path": f"articles/{content_path.name}",
                            "content_size_bytes": content_path.stat().st_size,
                            "content_type": "text/html",
                        }

                # Write article metadata
                write_json_atomic(article_path, article_data)

                # Update index
                index["articles"][article_id] = {
                    "id": article_id,
                    "published_utc": article.get("published_utc"),
                    "title": article.get("title"),
                    "has_full_content": "local_content" in article_data,
                }

        except Exception as e:
            logger.error(f"Error fetching news for {ticker_symbol}: {e}")
            stats["errors"].append(str(e))

        # Update index
        index["last_updated"] = utcnow_iso()
        index["article_count"] = len(index["articles"])
        write_json_atomic(index_path, index)

        return stats

    def _download_article(self, ticker_slug: str, article_id: str, url: str) -> Path | None:
        """Download article content from URL.

        Returns path to saved file or None if download failed.
        """
        try:
            # Download with reasonable timeout
            response = self.http_client.get(
                url,
                headers={"User-Agent": "Mozilla/5.0 (compatible; PortfolioAdvisor/1.0)"},
                follow_redirects=True,
            )
            response.raise_for_status()

            # Determine extension from content type
            content_type = response.headers.get("content-type", "").lower()
            if "html" in content_type:
                extension = "html"
            elif "json" in content_type:
                extension = "json"
            else:
                extension = "txt"

            # Save content
            content_path = self.paths.news_article_content(ticker_slug, article_id, extension)
            content_path.write_bytes(response.content)

            logger.debug(f"Downloaded article {article_id} for {ticker_slug}")
            return content_path

        except Exception as e:
            logger.warning(f"Failed to download article {article_id} from {url}: {e}")
            return None
