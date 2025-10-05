# Article Text Extraction Feature Design

## Overview

This document outlines the design for extracting article text from downloaded HTML news articles using the local milkey/reader-lm-v2:Q8_0 model via ollama. The feature will process HTML files and store the extracted text alongside the original HTML, enabling easier analysis and search capabilities.

## Requirements

1. Use ollama with milkey/reader-lm-v2:Q8_0 model for text extraction
2. Store extracted text in a file next to the HTML file
3. Operate as a separate processing step after article download
4. Support independent triggering for testing and patching older libraries
5. Non-destructive operation - only augment, never delete
6. Support force overwriting of previously extracted text

## Architecture

### Directory Structure

```
output/stocks/tickers/{ticker_slug}/
├── primary/
│   └── news/
│       ├── index.json                    # Enhanced with extraction tracking
│       ├── {article_id}.json            # Article metadata
│       └── articles/
│           ├── {article_id}.html        # Downloaded article content
│           └── {article_id}.txt         # Extracted text content
```

### Enhanced News Index

The news index will be enhanced to track extraction status:

```json
{
    "last_updated": "2025-09-26T12:00:00Z",
    "article_count": 42,
    "articles": {
        "nJCjJU8Kqw_2024-09-25": {
            "id": "nJCjJU8Kqw_2024-09-25",
            "published_utc": "2024-09-25T14:30:00Z",
            "title": "Apple Announces New Product Line",
            "has_full_content": true,
            "text_extracted": true,
            "text_extracted_at": "2025-09-26T14:00:00Z",
            "text_extraction_model": "milkey/reader-lm-v2:Q8_0"
        }
    }
}
```

### Components

#### 1. OllamaService (`src/portfolio_advisor/services/ollama_service.py`)

A generic service for interacting with the ollama API:

```python
class OllamaService:
    def __init__(self, base_url: str = "http://localhost:11434", timeout_s: int = 60):
        """Initialize ollama service with configurable endpoint."""
        
    def generate(self, model: str, prompt: str, **kwargs) -> str:
        """Send a generation request to ollama."""
        
    def is_available(self) -> bool:
        """Check if ollama service is available."""
```

#### 2. ArticleTextExtractionService (`src/portfolio_advisor/stocks/article_extraction.py`)

Service for extracting text from HTML articles:

```python
class ArticleTextExtractionService:
    def __init__(self, paths: StockPaths, ollama_service: OllamaService):
        """Initialize with stock paths and ollama service."""
        
    def extract_article_text(
        self, 
        ticker_slug: str, 
        article_id: str,
        force: bool = False
    ) -> dict[str, Any]:
        """Extract text from a single article."""
        
    def extract_all_articles(
        self,
        ticker_slug: str,
        force: bool = False,
        batch_size: int = 10
    ) -> dict[str, Any]:
        """Extract text from all articles for a ticker."""
        
    def extract_portfolio_articles(
        self,
        portfolio_path: Path,
        force: bool = False,
        max_workers: int = 4
    ) -> dict[str, Any]:
        """Extract text from all articles in portfolio."""
```

#### 3. Prompt Engineering

The extraction prompt will be designed to:
- Extract main article content
- Remove advertisements and navigation
- Preserve article structure (paragraphs, headings)
- Handle various news site formats

Example prompt template:
```
Extract the main article text from the following HTML. 
Include only the article title, subtitle, author, date, and body paragraphs.
Remove all advertisements, navigation, comments, and related articles.
Format the output as plain text with proper paragraph breaks.

HTML:
{html_content}

EXTRACTED TEXT:
```

### Integration Points

#### 1. Stock Pipeline Integration

Add an optional text extraction step after news fetching:

```python
# In _check_db_state_node
if "primary.text_extraction" in requested:
    needs_extraction = _check_unextracted_articles(paths, slug)
    if needs_extraction:
        updates.append("primary.text_extraction")

# New node
def _extract_article_text_node(state: StockState) -> dict:
    """Extract text from downloaded articles."""
    # Implementation
```

#### 2. CLI Commands

##### Add to existing stock update command:
```bash
# Extract text during stock update
python -m portfolio_advisor.cli \
    --mode stock \
    --ticker AAPL \
    --extract-text \
    --input-dir ./input \
    --output-dir ./output
```

##### New standalone extraction command:
```bash
# Extract text for specific ticker
python -m portfolio_advisor.cli \
    --mode extract-text \
    --ticker AAPL \
    --force \
    --input-dir ./input \
    --output-dir ./output

# Extract text for all tickers
python -m portfolio_advisor.cli \
    --mode extract-text \
    --all \
    --input-dir ./input \
    --output-dir ./output
```

### Configuration

New settings in `Settings` class:
```python
# Ollama configuration
ollama_base_url: str = "http://localhost:11434"
ollama_timeout_s: int = 60
extraction_model: str = "milkey/reader-lm-v2:Q8_0"
extraction_batch_size: int = 10
extraction_max_workers: int = 4
```

### Error Handling

1. **Ollama unavailable**: Log warning, skip extraction, continue processing
2. **Model not found**: Log error with instructions to pull model
3. **Extraction failure**: Log per-article errors, update stats, continue with next
4. **HTML parsing issues**: Fallback to basic text extraction

### Performance Considerations

1. **Batching**: Process articles in configurable batches to avoid overwhelming ollama
2. **Concurrency**: Use thread pool for parallel ticker processing
3. **Caching**: Skip already-extracted articles unless force=True
4. **Progress tracking**: Log progress for large extraction runs

### Testing Strategy

1. **Unit tests**: Mock ollama responses, test extraction logic
2. **Integration tests**: Test with real ollama (if available in CI)
3. **Test fixtures**: Sample HTML articles from various news sources
4. **Edge cases**: Empty articles, malformed HTML, very large articles

## Implementation Plan

1. **Phase 1**: Core extraction services
   - Implement OllamaService
   - Implement ArticleTextExtractionService
   - Add configuration options

2. **Phase 2**: Pipeline integration
   - Add extraction node to stocks graph
   - Update check_db_state logic
   - Enhance news index structure

3. **Phase 3**: CLI and tooling
   - Add CLI flags for stock mode
   - Implement standalone extraction mode
   - Add progress reporting

4. **Phase 4**: Testing and refinement
   - Write comprehensive tests
   - Test with various news sources
   - Optimize prompt engineering

## Usage Examples

### Extract during regular stock update
```bash
python -m portfolio_advisor.cli \
    --mode stock \
    --ticker AAPL \
    --extract-text \
    --input-dir ./input \
    --output-dir ./output
```

### Batch extraction for existing news library
```bash
# Extract for all stocks with progress
python -m portfolio_advisor.cli \
    --mode extract-text \
    --all \
    --verbose \
    --input-dir ./input \
    --output-dir ./output

# Force re-extraction for specific ticker
python -m portfolio_advisor.cli \
    --mode extract-text \
    --ticker MSFT \
    --force \
    --input-dir ./input \
    --output-dir ./output
```

### Programmatic usage
```python
from portfolio_advisor.config import Settings
from portfolio_advisor.services.ollama_service import OllamaService
from portfolio_advisor.stocks.article_extraction import ArticleTextExtractionService
from portfolio_advisor.stocks.db import StockPaths

settings = Settings(input_dir="./input", output_dir="./output")
paths = StockPaths(root=Path(settings.output_dir) / "stocks")
ollama = OllamaService(settings.ollama_base_url)
extractor = ArticleTextExtractionService(paths, ollama)

# Extract for single ticker
stats = extractor.extract_all_articles("cid-stocks-us-xnas-aapl", force=False)
print(f"Extracted {stats['extracted']} articles")
```

## Future Enhancements

1. **Model selection**: Support for different models based on article type
2. **Language detection**: Automatic language-specific model selection
3. **Structured extraction**: Extract specific fields (quotes, financial data)
4. **Quality scoring**: Rate extraction quality and flag for review
5. **Incremental updates**: Extract only new articles since last run
6. **API endpoint**: REST API for on-demand extraction
