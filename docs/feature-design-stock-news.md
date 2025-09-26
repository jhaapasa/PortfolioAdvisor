# Stock News Feature Design

## Overview

This document outlines the design for integrating stock news functionality into the PortfolioAdvisor system using the Polygon.io `/v2/reference/news` API. The feature will fetch recent news articles for each stock, download full article content when possible, and store everything in a structured format that integrates with the existing stock data organization.

## Requirements

1. Pull trailing 7 days of news items whenever stock data is being updated
2. Store news data in the per-security folder under `primary/news`
3. Retain older news if they exist in the folder, and just add any new, not previously existing items when updating
4. Attempt to pull the primary articles from the web using the `article_url` for each news result
5. If article fetching succeeds, note the filename and relative path in the news JSON record
6. If primary article fetching fails, simply don't add a file or record for it

## Architecture

### Directory Structure

```
output/stocks/tickers/{ticker_slug}/
├── primary/
│   ├── ohlc_daily.json
│   └── news/
│       ├── index.json                    # Index of all news articles
│       ├── {article_id}.json            # Individual article metadata
│       └── articles/
│           └── {article_id}.html        # Downloaded article content
├── analysis/
├── report/
└── meta.json
```

### Data Models

#### News Index (`primary/news/index.json`)
```json
{
    "last_updated": "2025-09-26T12:00:00Z",
    "article_count": 42,
    "articles": {
        "nJCjJU8Kqw_2024-09-25": {
            "id": "nJCjJU8Kqw_2024-09-25",
            "published_utc": "2024-09-25T14:30:00Z",
            "title": "Apple Announces New Product Line",
            "has_full_content": true
        }
    }
}
```

#### Individual Article Metadata (`primary/news/{article_id}.json`)
```json
{
    "id": "nJCjJU8Kqw_2024-09-25",
    "publisher": {
        "name": "The Motley Fool",
        "homepage_url": "https://www.fool.com/",
        "logo_url": "https://s3.polygon.io/public/assets/news/logos/themotleyfool.svg",
        "favicon_url": "https://s3.polygon.io/public/assets/news/favicons/themotleyfool.ico"
    },
    "title": "Apple Announces New Product Line",
    "author": "John Doe",
    "published_utc": "2024-09-25T14:30:00Z",
    "article_url": "https://www.fool.com/investing/2024/09/25/apple-new-products/",
    "tickers": ["AAPL"],
    "amp_url": "https://www.fool.com/amp/investing/2024/09/25/apple-new-products/",
    "image_url": "https://g.foolcdn.com/editorial/images/123456/apple-products.jpg",
    "description": "Apple Inc. unveiled its latest product lineup...",
    "keywords": ["Technology", "Consumer Electronics", "AAPL"],
    "local_content": {
        "downloaded_at": "2024-09-26T08:00:00Z",
        "content_path": "articles/nJCjJU8Kqw_2024-09-25.html",
        "content_size_bytes": 45678,
        "content_type": "text/html"
    }
}
```

### Filename Scheme for Articles

Article IDs from Polygon can sometimes be non-unique across time, so we'll use a composite identifier:
- Format: `{polygon_article_id}_{published_date}.{extension}`
- Example: `nJCjJU8Kqw_2024-09-25.html`
- This ensures uniqueness even if Polygon reuses IDs over time

## Implementation Plan

### 1. Extend PolygonClient

Add news-related methods to `src/portfolio_advisor/services/polygon_client.py`:

```python
def list_ticker_news(
    self,
    ticker: str,
    published_utc_gte: str | None = None,
    published_utc_lte: str | None = None,
    limit: int = 1000,
    order: str = "desc"
) -> Iterable[dict[str, Any]]:
    """Yield news articles for a ticker from Polygon's news endpoint.
    
    Parameters:
        ticker: Stock ticker symbol
        published_utc_gte: Filter for articles published on or after (ISO format)
        published_utc_lte: Filter for articles published on or before (ISO format)
        limit: Max results per page (default 1000)
        order: Sort order - 'asc' or 'desc' by published_utc
    """
    client = self._ensure_client()
    
    # The polygon client has a list_ticker_news method
    for article in client.list_ticker_news(
        ticker=ticker,
        published_utc_gte=published_utc_gte,
        published_utc_lte=published_utc_lte,
        limit=limit,
        order=order
    ):
        if hasattr(article, "model_dump"):
            yield article.model_dump()
        elif hasattr(article, "__dict__"):
            yield dict(article.__dict__)
        else:
            yield dict(article)
```

### 2. Update StockPaths

Add news-related paths to `src/portfolio_advisor/stocks/db.py`:

```python
def news_dir(self, ticker: str) -> Path:
    return self.ticker_dir(ticker) / "primary" / "news"

def news_index_json(self, ticker: str) -> Path:
    return self.news_dir(ticker) / "index.json"

def news_article_json(self, ticker: str, article_id: str) -> Path:
    return self.news_dir(ticker) / f"{article_id}.json"

def news_articles_dir(self, ticker: str) -> Path:
    return self.news_dir(ticker) / "articles"

def news_article_content(self, ticker: str, article_id: str, extension: str = "html") -> Path:
    return self.news_articles_dir(ticker) / f"{article_id}.{extension}"
```

### 3. Create News Service

Create `src/portfolio_advisor/stocks/news.py`:

```python
from __future__ import annotations

import datetime as dt
import json
import logging
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx

from ..services.polygon_client import PolygonClient
from ..utils.fs import utcnow_iso, write_json_atomic
from .db import StockPaths

logger = logging.getLogger(__name__)


class StockNewsService:
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
    
    def update_ticker_news(self, ticker_slug: str, ticker_symbol: str, days_back: int = 7) -> dict[str, Any]:
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
            index = {
                "last_updated": None,
                "article_count": 0,
                "articles": {}
            }
        
        # Track operation stats
        stats = {
            "articles_fetched": 0,
            "articles_new": 0,
            "articles_downloaded": 0,
            "errors": []
        }
        
        try:
            # Fetch news from Polygon
            for article in self.polygon_client.list_ticker_news(
                ticker=ticker_symbol,
                published_utc_gte=start_date.strftime("%Y-%m-%d"),
                published_utc_lte=end_date.strftime("%Y-%m-%d"),
                limit=1000,
                order="desc"
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
                    content_path = self._download_article(
                        ticker_slug, article_id, article_url
                    )
                    if content_path:
                        stats["articles_downloaded"] += 1
                        article_data["local_content"] = {
                            "downloaded_at": utcnow_iso(),
                            "content_path": f"articles/{content_path.name}",
                            "content_size_bytes": content_path.stat().st_size,
                            "content_type": "text/html"
                        }
                
                # Write article metadata
                write_json_atomic(article_path, article_data)
                
                # Update index
                index["articles"][article_id] = {
                    "id": article_id,
                    "published_utc": article.get("published_utc"),
                    "title": article.get("title"),
                    "has_full_content": "local_content" in article_data
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
                headers={
                    "User-Agent": "Mozilla/5.0 (compatible; PortfolioAdvisor/1.0)"
                },
                follow_redirects=True
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
```

### 4. Integration Points

#### Update Stock Graph Pipeline

Modify `src/portfolio_advisor/graphs/stocks.py` to include news fetching:

```python
# In the stock update workflow, after fetching OHLC data:
if fetch_news:
    news_service = StockNewsService(paths, polygon_client)
    news_stats = news_service.update_ticker_news(
        ticker_slug=meta["slug"],
        ticker_symbol=meta["primary_ticker"],
        days_back=7
    )
    logger.info(f"News update for {ticker}: {news_stats}")
```

#### Add CLI Flag

Add a flag to control news fetching in `src/portfolio_advisor/cli.py`:

```python
@click.option(
    "--fetch-news/--no-fetch-news",
    default=True,
    help="Fetch news articles when updating stock data (default: True)"
)
```

## Testing Strategy

### Unit Tests

1. Test PolygonClient news method with mocked responses
2. Test StockNewsService with mock HTTP client
3. Test article ID generation and deduplication logic
4. Test error handling for failed downloads

### Integration Tests

1. Test full news fetching pipeline with test fixtures
2. Test incremental updates (no duplicates)
3. Test handling of various article content types
4. Test concurrent news fetching for multiple tickers

## Security and Performance Considerations

### Security
- Sanitize downloaded HTML content before storage
- Validate URLs before downloading
- Use appropriate User-Agent headers
- Respect robots.txt and rate limits

### Performance
- Implement concurrent article downloads (with limits)
- Cache HTTP client connections
- Use atomic writes to prevent corruption
- Consider implementing a download queue for large updates

### Rate Limiting
- Polygon API has rate limits based on subscription tier
- Implement exponential backoff for retries
- Track API usage and warn when approaching limits

## Future Enhancements

1. **Article Processing**
   - Extract plain text from HTML
   - Perform sentiment analysis
   - Extract mentioned tickers and entities

2. **Search and Retrieval**
   - Build full-text search index
   - Create news timeline visualizations
   - Aggregate news by themes/topics

3. **Real-time Updates**
   - Use Polygon WebSocket for real-time news
   - Implement push notifications for significant news

4. **Content Enhancement**
   - Screenshot article pages for archival
   - Extract and store article images
   - Generate article summaries using LLMs

## Migration and Rollback

### Migration
- No existing data to migrate
- Feature is additive and won't affect existing functionality

### Rollback
- Simply disable the `--fetch-news` flag
- News data can be safely deleted without affecting other features
