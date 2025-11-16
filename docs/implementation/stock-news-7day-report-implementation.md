# Stock News 7-Day Report Implementation Notes

## Overview

The 7-day news report feature generates concise, per-stock updates blending news sentiment analysis with technical performance metrics. Implemented as two cooperating LLM agents that process Polygon news summaries and combine them with technical analysis data.

**Design Document**: `docs/design/feature-design-stock-news-7day-report.md`

**Implementation Date**: October-November 2024

**Status**: Production-ready, integrated into stock update pipeline

## Core Implementation Approach

### Two-Agent Architecture

**Design Decision**: Separate news analysis from report collation
- **NewsSummaryAgent**: Analyze news, extract notable events
- **ReportCollatorAgent**: Combine news + technical metrics into final report

**Why Split**:
- Separation of concerns (news vs metrics)
- Reusable news summaries for other purposes
- Easier to test each component independently
- Can skip technical metrics if unavailable

### Data Flow

```
Polygon News (7 days)
    ↓
NewsSummaryAgent (LLM)
    ↓
news_summary: { markdown, json }
    ↓
ReportCollatorAgent (LLM) + Technical Metrics
    ↓
Final Report (Markdown + JSON)
    ↓
output/stocks/tickers/{slug}/report/7d/
```

## Component Implementations

### NewsSummaryAgent (`agents/news_summary.py`)

**Purpose**: Distill 7 days of Polygon news into notable events and sentiment analysis

**Key Constraint**: Use only Polygon-provided JSON fields
- No HTML parsing
- No article body extraction
- Only: title, summary, sentiment, publisher, published_utc

**Why**: Reliability over depth
- Polygon summaries are pre-vetted
- HTML extraction unreliable (see article-extraction-implementation.md)
- Consistent data quality

#### Input Processing

**News Item Structure** (from Polygon):
```json
{
  "id": "article_id",
  "title": "Apple Announces...",
  "published_utc": "2024-09-25T14:30:00Z",
  "publisher": {"name": "The Motley Fool"},
  "tickers": ["AAPL"],
  "article_url": "https://...",
  "summary": "Brief summary...",  // Optional
  "sentiment": "positive",         // Optional
  "sentiment_score": 0.75          // Optional
}
```

**Compact JSON Formatting**:
```python
def _compact_json_list(items, limit=100):
    # One-line JSON per article
    # Limit to 100 articles for token efficiency
```

**Why Compact**: Keep prompts manageable, focus on recent items

#### LLM Prompt Design

**System Prompt**:
```
You analyze 7 days of Polygon-provided stock news summaries and sentiments.
Use ONLY the provided JSON fields (headline, summary, sentiment, published time, publisher).
Do NOT infer from HTML. Surface notable events and sentiment extremes or shifts.
Skip routine items. Output both Markdown and structured JSON.
Be precise and conservative; do not hallucinate facts.
```

**User Prompt**:
```
Ticker: {ticker} | Slug: {slug} | WindowDays: 7

News JSON (compact list):
{news_compact}

Return JSON with keys: 'highlights_markdown', 'sentiment_overview',
'notable_events', and include 'ticker', 'slug', 'window_days'.
```

**Design Rationale**:
- Clear constraints (no HTML, no hallucination)
- Structured output for downstream use
- Conservative approach (skip vs invent)

#### Output Structure

**Successful Response** (JSON):
```json
{
  "ticker": "AAPL",
  "slug": "cid-stocks-us-xnas-aapl",
  "window_days": 7,
  "sentiment_overview": {
    "overall_label": "positive|neutral|negative",
    "avg_score": 0.0,
    "trend": "rising|flat|falling",
    "counts": {"positive": 5, "neutral": 10, "negative": 2},
    "strongest_day": "2025-10-05"
  },
  "notable_events": [
    {
      "date": "2025-10-04",
      "title": "Rating upgrade by XYZ",
      "why_notable": "Impacts investor sentiment...",
      "sentiment": "positive"
    }
  ],
  "highlights_markdown": "## Notable News & Events\n- ...",
  "notes": ["Based solely on Polygon summaries/sentiment"]
}
```

**Fallback Response** (on error):
```json
{
  "sentiment_overview": {"overall_label": "neutral", ...},
  "notable_events": [],
  "highlights_markdown": "## Notable News & Events\n- [none]",
  "notes": ["Fallback due to LLM error"]
}
```

**Error Handling Strategy**: Always return valid structure
- LLM failures don't crash pipeline
- Graceful degradation
- User sees "no notable events" vs error

#### Caching Strategy

**Implementation**:
```python
# Try to load from disk if not in state
cache_path = paths.news_dir(slug) / "recent_7d.json"
if cache_path.exists():
    news_items = load_from_cache()
```

**Why**: Avoid re-fetching news within same session
- News updates are batch operations
- 7-day window relatively stable
- Performance optimization

### ReportCollatorAgent (`agents/stock_report_collator.py`)

**Purpose**: Combine news summary with technical metrics into cohesive report

**Inputs**:
1. `news_summary`: {markdown, json} from NewsSummaryAgent
2. `returns.json`: Trailing returns (d1, d5, d21, d252)
3. `volatility_histogram.json`: Wavelet variance spectrum

**Why These Metrics**:
- d5 (5-day): Matches 7-day news window approximately
- d1 (1-day): Recent performance context
- Volatility: Risk context for news interpretation

#### Report Structure

**Target Markdown** (from design):
```markdown
# {TICKER} — 7‑Day Update

TL;DR: {one-line synthesis blending sentiment + performance}

## Notable News & Events
- {event 1: date — what happened — why it matters}
- {event 2}

## Sentiment Overview (7d)
- Overall: {positive|neutral|negative}, trend: {rising|flat|falling}
- Counts: {pos}/{neu}/{neg}; strongest day: {date} ({brief})

## Performance Context (7d)
- Return: {pct}
- Realized volatility: {brief}
- Histogram: {brief description}

Notes: Based solely on Polygon summaries/sentiment; no article body parsing.
```

#### Prompt Design

**System Prompt**:
```
You produce a concise per-stock trailing 7-day update.
Inputs: (a) news summary markdown and optional JSON metrics;
        (b) technical metrics including recent returns and volatility histogram.
Output a cohesive Markdown report with sections: TL;DR, Notable News & Events,
Sentiment Overview (7d), Performance Context (7d).
If a metric is missing, omit it without inventing values.
Keep it concise and high-signal.
```

**User Prompt**:
```
Ticker: {ticker} | Slug: {slug}

News Summary (Markdown):
{news_md}

News Summary (JSON):
{news_json}

Technical Metrics:
Returns: d1={x}%, d5={y}%, d21={z}%
Volatility Histogram: {summary_if_available}

Generate the 7-day update report following the specified structure.
```

**Design Choices**:
- Provide both markdown and JSON to LLM (gives flexibility)
- Include technical context in prompt
- Allow omission of missing data
- Enforce structure via instructions

#### Output Files

**Location**: `output/stocks/tickers/{slug}/report/7d/`

**Files**:
1. `report.md`: Final markdown report (primary output)
2. `metrics.json`: Structured metrics for testing/tooling

**Metrics JSON Structure**:
```json
{
  "ticker": "AAPL",
  "slug": "cid-stocks-us-xnas-aapl",
  "window_days": 7,
  "as_of": "2025-11-15",
  "sentiment": {
    "overall": "positive",
    "trend": "rising",
    "counts": {"positive": 5, "neutral": 10, "negative": 2}
  },
  "performance": {
    "d1": 0.012,
    "d5": 0.034,
    "d21": 0.089
  },
  "notable_events_count": 3,
  "report_path": "report/7d/report.md",
  "generated_at": "2025-11-15T10:30:00Z"
}
```

**Why Both Formats**:
- Markdown: Human consumption
- JSON: Testing, automation, downstream tools

## Integration Points

### Stock Update Pipeline

**Integration** (`graphs/stocks.py`):
```python
# Conditional execution based on settings
if settings.include_news_report:
    # Sequence: fetch_news → summarize_news → collate_report
```

**Activation**:
```python
settings = Settings(include_news_report=True)
```

**Default**: Disabled (opt-in feature)
- Adds LLM costs
- Requires news data
- Not always needed

### CLI

**Example Usage**:
```bash
# Update stock with news report
python -m portfolio_advisor.cli \
    --mode stock \
    --ticker AAPL \
    --include-news-report \
    --output-dir ./output
```

**Flag**: `--include-news-report`
- Boolean flag
- Enables both agents
- Requires news fetching enabled

## Paths Not Taken

### Multi-Week Horizons
- **Not Taken**: 30-day, 90-day reports
- **Why**: 7 days balances recency vs signal
- **Decision**: Can add timeframes later if needed

### Article Body Analysis
- **Not Taken**: Parse and analyze full HTML articles
- **Why**: Unreliable extraction (see article-extraction)
- **Trade-off**: Depth vs reliability
- **Choice**: Polygon summaries sufficient

### Real-Time Sentiment Tracking
- **Not Taken**: Intraday sentiment updates
- **Why**: End-of-day batch processing model
- **Context**: Not building real-time trading system

### Multiple LLM Providers
- **Not Taken**: Support for Claude, local models, etc.
- **Why**: Existing LLM factory abstracts this
- **Note**: Works with any provider via `llm.py`

### Quantitative Sentiment Scoring
- **Not Taken**: Custom sentiment model or scoring
- **Why**: Polygon provides sentiment already
- **Trade-off**: Control vs simplicity

### Historical Report Archive
- **Not Taken**: Store report history, compare week-to-week
- **Why**: Out of scope for MVP
- **Future**: Could add timestamped archives

### Interactive Reports
- **Not Taken**: HTML with charts, interactive elements
- **Why**: Markdown sufficient for current use
- **Future**: Could generate HTML from markdown

## Key Learnings

1. **Separation of concerns works well**: Two agents cleaner than one
2. **Polygon summaries are good enough**: Don't need full article text
3. **Structured output valuable**: JSON + Markdown serves multiple use cases
4. **Graceful degradation essential**: LLM failures can't break pipeline
5. **7-day window is sweet spot**: Recent enough, enough data points

## Performance Characteristics

**NewsSummaryAgent** (per ticker):
- News loading: <100ms (from cache/disk)
- LLM call: 2-5 seconds
- JSON parsing: <10ms
- **Total**: ~2-5 seconds

**ReportCollatorAgent** (per ticker):
- Load metrics: <50ms
- Build prompt: <10ms
- LLM call: 3-7 seconds  
- Write files: <50ms
- **Total**: ~3-7 seconds

**End-to-End** (per ticker):
- News summary: ~2-5 seconds
- Report collation: ~3-7 seconds
- **Total**: ~5-12 seconds

**Portfolio Scale** (50 tickers):
- Sequential: ~4-10 minutes
- Bottleneck: LLM calls
- **Note**: Could parallelize LLM calls in future

## Error Handling

### Missing News Data
- **Scenario**: No news articles in 7-day window
- **Handling**: Skip report generation
- **Log**: Warning, not error

### LLM Failures
- **Scenario**: API timeout, rate limit, parsing error
- **Handling**: Fallback to default structure
- **User Impact**: Sees "no notable events" placeholder

### Missing Technical Metrics
- **Scenario**: Returns or volatility not yet computed
- **Handling**: Omit from report
- **Instruction**: LLM told to skip missing data

### Invalid JSON from LLM
- **Scenario**: LLM returns markdown instead of JSON
- **Handling**: Treat as markdown-only response
- **Fallback**: JSON set to None, markdown preserved

## Testing Approach

**Unit Tests**:
- Prompt construction with various inputs
- JSON parsing (valid and invalid)
- Fallback mechanisms
- Metric loading logic

**Integration Tests**:
- Full pipeline with mocked LLM
- Various news scenarios (empty, sparse, rich)
- Missing metrics handling
- File output validation

**Mocking Strategy**:
- LLM responses: Pre-recorded JSON and markdown
- News data: Synthetic Polygon-style fixtures
- Metrics: Minimal valid JSON documents

## Configuration

**Settings Fields**:
```python
include_news_report: bool = False  # Opt-in
```

**Hard-coded Parameters**:
- Window: 7 days
- News limit: 100 articles max
- Report location: `report/7d/`

**Why Hard-coded**:
- Reasonable defaults
- Simplifies implementation
- Can parameterize later if needed

## TODO: Future Improvements

### High Priority
- [ ] Add d1 return to standard analysis (currently ad-hoc in prompts)
- [ ] Parallel LLM calls for portfolio-wide reports
- [ ] Cache LLM responses for unchanged news windows
- [ ] Add report generation status tracking

### Medium Priority
- [ ] Multi-timeframe reports (30d, 90d)
- [ ] Historical report archive with comparisons
- [ ] Sentiment trend charts
- [ ] Export to HTML/PDF formats

### Low Priority
- [ ] Custom sentiment scoring (vs Polygon's)
- [ ] News source filtering (publisher allowlist)
- [ ] Keyword/theme extraction
- [ ] Cross-ticker news correlation

### Research
- [ ] Evaluate alternative LLMs for news summary quality
- [ ] A/B test prompt variations
- [ ] User study: Which report sections most valuable?
- [ ] Automated quality scoring for reports

## Dependencies

**Core**:
- `llm.py`: LLM factory for any provider
- `stocks/news.py`: News fetching (prerequisite)
- `stocks/analysis.py`: Technical metrics

**Utilities**:
- `stocks/db.StockPaths`: Path management
- `utils/fs.py`: File operations

## Known Issues / Limitations

1. **7-day window fixed**: Not configurable per ticker
2. **No report versioning**: Overwrites previous report
3. **No change tracking**: Can't compare to prior week
4. **LLM dependency**: Requires API access
5. **Sequential processing**: No parallelization yet
6. **No quality validation**: Accepts any LLM output
7. **English only**: No multi-language support

## Related Documentation

- Design: `docs/design/feature-design-stock-news-7day-report.md`
- Stock News: `docs/implementation/stock-news-implementation.md`
- Article Extraction: `docs/implementation/article-extraction-implementation.md` (related but separate)
- Architecture: `docs/Architecture.md` (agents section)

