# Stock News Implementation Notes

## Overview

Stock news fetching provides access to recent news articles for each ticker via the Polygon.io news API. Articles are downloaded, stored locally, and indexed for future analysis and search capabilities.

**Design Document**: `docs/design/feature-design-stock-news.md`

**Implementation Date**: September 2024

**Status**: Production-ready, integrated into stock update pipeline

## Core Implementation Approach

### Directory Structure

**Location**: `output/stocks/tickers/{slug}/primary/news/`

```
primary/news/
  index.json                    # Article registry
  {article_id}.json            # Article metadata
  articles/
    {article_id}.html          # Downloaded article content
    {article_id}.txt           # Extracted text (if extraction enabled)
```

### Article ID Generation

**Problem**: Polygon article IDs may not be unique across time
**Solution**: Composite ID with publish date
```python
article_id = f"{polygon_id}_{published_date}"
# Example: "nJCjJU8Kqw_2024-09-25"
```

**Why**:
- Guarantees uniqueness even if Polygon reuses IDs
- Sortable by date (helps with organization)
- Human-readable

## Component Implementations

### StockNewsService (`stocks/news.py`)

**Purpose**: Fetch and store news articles for tickers

**Constructor**:
```python
def __init__(self, paths: StockPaths, polygon_client: PolygonClient):
    self.paths = paths
    self.polygon_client = polygon_client
    self.http_client = httpx.Client(timeout=30.0)
```

**Key Design**: Separate HTTP client for article downloads
- Polygon client for metadata
- httpx client for actual article content
- Independent timeouts and retry logic

### Update Process

**Main Method**: `update_ticker_news(ticker_slug, ticker_symbol, days_back=7)`

**Flow**:
1. Calculate date range (now - N days to now)
2. Load existing news index (if exists)
3. Fetch news from Polygon API
4. For each article:
   - Generate composite article ID
   - Skip if already in index
   - Download article content (if URL available)
   - Save metadata JSON
   - Update index
5. Write updated index atomically

**Incremental Updates**:
- Checks existing index before fetching
- Skips articles already present
- Only downloads new articles
- Preserves old articles (additive only)

### Article Download

**Method**: `_download_article(ticker_slug, article_id, url)`

**Implementation**:
```python
response = self.http_client.get(
    url,
    headers={"User-Agent": "Mozilla/5.0 (compatible; PortfolioAdvisor/1.0)"},
    follow_redirects=True
)
```

**User-Agent Strategy**:
- Identifies as PortfolioAdvisor
- Compatible with Mozilla format
- Helps avoid bot blocking

**Content-Type Detection**:
```python
content_type = response.headers.get("content-type", "").lower()
if "html" in content_type:
    extension = "html"
elif "json" in content_type:
    extension = "json"
else:
    extension = "txt"
```

**Error Handling**:
- Network failures: Log warning, continue with next article
- HTTP errors: Log, skip article
- Timeouts: Handled by httpx timeout setting
- **Philosophy**: Best effort download, don't fail entire update

### News Index Structure

**Format** (`index.json`):
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

**Index Purpose**:
- Quick lookup of available articles
- Track which articles have full content
- Avoid re-downloading existing articles
- Metadata for search/filtering

### Article Metadata Structure

**Format** (`{article_id}.json`):
```json
{
  "id": "nJCjJU8Kqw_2024-09-25",
  "publisher": {
    "name": "The Motley Fool",
    "homepage_url": "https://www.fool.com/",
    "logo_url": "https://s3.polygon.io/...",
    "favicon_url": "https://s3.polygon.io/..."
  },
  "title": "Apple Announces New Product Line",
  "author": "John Doe",
  "published_utc": "2024-09-25T14:30:00Z",
  "article_url": "https://www.fool.com/...",
  "tickers": ["AAPL"],
  "keywords": ["Technology", "Consumer Electronics"],
  "local_content": {
    "downloaded_at": "2024-09-26T08:00:00Z",
    "content_path": "articles/nJCjJU8Kqw_2024-09-25.html",
    "content_size_bytes": 45678,
    "content_type": "text/html"
  }
}
```

**Key Fields**:
- **publisher**: Source attribution
- **tickers**: Multi-ticker articles supported
- **keywords**: For future categorization
- **local_content**: Present only if download succeeded

## Integration Points

### Stock Update Graph

**Integration** (`graphs/stocks.py`):
```python
if "primary.news" in requested:
    # Fetch 7 days of news
    # Update news index
    # Download new articles
```

**Conditional Execution**:
- Only runs if explicitly requested
- Part of standard stock update pipeline
- Can be skipped for faster updates

### Polygon Client Extension

**Addition** (`services/polygon_client.py`):
```python
def list_ticker_news(
    self,
    ticker: str,
    published_utc_gte: str | None = None,
    published_utc_lte: str | None = None,
    limit: int = 1000,
    order: str = "desc"
) -> Iterable[dict[str, Any]]
```

**Implementation Notes**:
- Uses Polygon's built-in pagination
- Yields results for memory efficiency
- Converts Polygon objects to dicts
- Handles both old and new Polygon SDK versions

## Paths Not Taken

### Database for News Index
- **Not Taken**: SQLite or Postgres for article index
- **Why**: JSON index sufficient for portfolio scale
- **Trade-off**: Slower full-text search, but simple

### Full-Text Extraction During Download
- **Not Taken**: Parse HTML to extract article text immediately
- **Why**: Separate concern, handled by article extraction feature
- **Decision**: Keep download simple, defer extraction

### Real-Time News
- **Not Taken**: WebSocket for live news feed
- **Why**: Batch processing sufficient for portfolio analysis
- **Context**: Not building real-time trading system

### News Sentiment Analysis
- **Not Taken**: Sentiment scoring during fetch
- **Why**: Out of scope for basic news feature
- **Note**: Polygon provides sentiment, we store it

### Article Screenshots
- **Not Taken**: Capture rendered article as image
- **Why**: Complex, storage-heavy, marginal value
- **Trade-off**: Lose visual layout, keep text

### Content Deduplication
- **Not Taken**: Detect duplicate articles from different sources
- **Why**: Polygon already filters somewhat
- **Complexity**: Would need fuzzy matching

### Article Expiration
- **Not Taken**: Auto-delete old articles
- **Why**: Storage cheap, historical context valuable
- **Decision**: Keep all articles indefinitely

## Key Learnings

1. **Composite IDs essential**: Polygon IDs alone not unique enough
2. **Best-effort downloads work**: Don't fail on single article error
3. **User-Agent matters**: Some sites block default Python UA
4. **Separate HTTP client needed**: Polygon client not for general downloads
5. **Index updates must be atomic**: Corruption from partial writes

## Performance Characteristics

**Initial 7-Day Fetch** (per ticker):
- API call: ~500ms
- Typically 5-20 articles
- Downloads: 2-5 seconds (parallel potential)
- Total: ~3-6 seconds

**Daily Incremental Update**:
- API call: ~500ms
- Typically 0-3 new articles
- Downloads: 0-2 seconds
- Total: ~0.5-2.5 seconds

**Storage**:
- Metadata JSON: ~1-3 KB per article
- HTML content: 10-100 KB per article
- Typical: ~50-200 KB per article total
- 7 days of news: ~500 KB - 2 MB per ticker

## Error Handling

### Network Errors
- **Strategy**: Log, continue with next article
- **Rationale**: Partial success better than total failure
- **Recovery**: Retry on next update

### API Rate Limits
- **Detection**: HTTP 429 status
- **Handling**: Currently none (relies on Polygon SDK)
- **Future**: Exponential backoff

### Missing Articles
- **Scenario**: Article URL returns 404
- **Handling**: Skip, mark as unavailable
- **Index**: `has_full_content: false`

### Malformed Data
- **Scenario**: Unexpected Polygon response format
- **Handling**: Try-except with logging
- **Fallback**: Skip article, continue

## Testing Approach

**Unit Tests**:
- Article ID generation
- Index update logic
- Metadata structure
- Date range calculation

**Integration Tests**:
- Full news fetch with mocked Polygon
- Incremental update (no duplicates)
- Download with mocked HTTP responses
- Error recovery scenarios

**Mocking Strategy**:
- Polygon API: Deterministic article list
- HTTP downloads: Pre-recorded responses
- Filesystem: Temp directories

## Configuration

**Fixed Parameters**:
- Days back: 7 (standard window)
- Fetch limit: 1000 articles (Polygon max)
- Download timeout: 30 seconds
- Order: Descending (newest first)

**User-Agent**:
- Fixed: `Mozilla/5.0 (compatible; PortfolioAdvisor/1.0)`
- Not configurable (standardization)

## TODO: Future Improvements

### High Priority
- [ ] Add news summary generation (LLM-based)
- [ ] Support custom date ranges for news fetch
- [ ] Implement retry logic for failed downloads
- [ ] Add rate limit handling with backoff

### Medium Priority
- [ ] Full-text search across article content
- [ ] News timeline visualization
- [ ] Duplicate article detection
- [ ] Publisher allowlist/blocklist
- [ ] Batch download for efficiency

### Low Priority
- [ ] Article tagging and categorization
- [ ] Export news to RSS/email
- [ ] Historical news gap filling
- [ ] Article quality scoring
- [ ] Multi-language support

### Research Items
- [ ] News impact analysis (price correlation)
- [ ] Event detection from news flow
- [ ] Sentiment trend tracking
- [ ] Cross-ticker news correlation

## Dependencies

**Core**:
- `polygon-api-client`: News API access
- `httpx`: Article downloading

**Utilities**:
- `utils.fs`: Atomic writes
- `stocks.db.StockPaths`: Path management

## Known Issues / Limitations

1. **No download retry**: Single attempt per article
2. **No rate limit handling**: Could hit Polygon limits
3. **No content validation**: Saves whatever downloads
4. **Simple error handling**: Continues on any error
5. **No parallel downloads**: Sequential processing
6. **Fixed 7-day window**: Not configurable per ticker
7. **No news alerts**: Passive storage only

## Related Documentation

- Design: `docs/design/feature-design-stock-news.md`
- News Summary: `docs/design/feature-design-stock-news-7day-report.md` (proposed)
- Article Extraction: `docs/implementation/article-extraction-implementation.md`
- Architecture: `docs/Architecture.md`

