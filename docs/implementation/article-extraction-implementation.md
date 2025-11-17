# Article Text Extraction Implementation Notes

## Overview

Article text extraction uses a local LLM (via Ollama) to extract clean article text from downloaded HTML content. The feature is **experimental and disabled by default** due to quality limitations with the current model.

**Design Document**: `docs/design/feature-design-article-text-extraction.md`

**Implementation Date**: September 2024

**Status**: Implemented but disabled by default, requires explicit activation

## Core Implementation Approach

### Ollama Integration

**Service**: `OllamaService` (`services/ollama_service.py`)

**Purpose**: Generic wrapper for Ollama API
- Not article-specific (reusable for other LLM tasks)
- Handles HTTP communication with Ollama server
- Availability checking
- Error handling

**Configuration**:
```python
ollama_base_url: str = "http://localhost:11434"  # Default Ollama URL
ollama_timeout_s: int = 60
extraction_model: str = "milkey/reader-lm-v2:Q8_0"
```

### Model Choice: reader-lm-v2

**Model**: `milkey/reader-lm-v2:Q8_0`
- Specialized for text extraction from web content
- Quantized (Q8_0) for smaller size
- Runs locally via Ollama (no API costs)

**Limitations** (why disabled by default):
1. **Quality Issues**: Sometimes generates generic content instead of verbatim extraction
2. **Content Mixing**: May mix content from different page sections
3. **Repetition**: Large articles can result in repetitive output
4. **Slow**: Large HTML files (>30KB) can take 60+ seconds

### Directory Structure

**Enhancement to news structure**:
```
primary/news/
  index.json                    # Enhanced with extraction tracking
  {article_id}.json            # Enhanced with extraction metadata
  articles/
    {article_id}.html          # Original HTML
    {article_id}.txt           # Extracted text (NEW)
```

## Component Implementations

### OllamaService (`services/ollama_service.py`)

**Interface**:
```python
class OllamaService:
    def __init__(self, base_url: str, timeout_s: int):
        ...
    
    def is_available(self) -> bool:
        # Check if Ollama is running
        
    def generate(self, model: str, prompt: str, **kwargs) -> str:
        # Generate text with specified model
```

**Key Design Decisions**:
- Stateless: No session management
- Synchronous: Simpler than async for batch processing
- Minimal: Only essential methods

**Availability Check**:
```python
def is_available(self) -> bool:
    try:
        response = self.client.get(f"{self.base_url}/api/tags")
        return response.status_code == 200
    except:
        return False
```

**Why**: Fail gracefully if Ollama not running

### ArticleTextExtractionService (`stocks/article_extraction.py`)

**Purpose**: Extract text from HTML articles using LLM

**Main Methods**:
1. `extract_article_text(ticker_slug, article_id)`: Single article
2. `extract_all_articles(ticker_slug)`: All articles for ticker
3. `extract_portfolio_articles(portfolio_path)`: Batch across portfolio

**Preprocessing** (`extract_article_section()`):
```python
def extract_article_section(html: str) -> str:
    # Remove script and style tags
    # Extract body content if present
    # Basic whitespace normalization
    # Limit to reasonable size
```

**Why Preprocess**:
- Remove noise (scripts, styles)
- Reduce token count
- Focus on article content
- Prevent timeout on huge files

**Size Limit**: 100,000 characters after cleaning
- Protects against memory issues
- Keeps processing time reasonable
- Most articles well under this limit

### Extraction Process

**Flow**:
```python
def extract_article_text(ticker_slug, article_id, force=False):
    # 1. Check if already extracted (unless force)
    # 2. Read HTML file
    # 3. Preprocess HTML
    # 4. Build extraction prompt
    # 5. Call Ollama
    # 6. Save extracted text
    # 7. Update article metadata
    # 8. Update news index
```

**Prompt Template**:
```
Extract the main article text from the following HTML.
Include only the article title, subtitle, author, date, and body paragraphs.
Remove all advertisements, navigation, comments, and related articles.
Format the output as plain text with proper paragraph breaks.

HTML:
{html_content}

EXTRACTED TEXT:
```

**Prompt Design Rationale**:
- Clear instructions on what to include/exclude
- Explicit format specification
- Simple structure (easier for model to follow)

### Index Enhancement

**New Fields in `index.json`**:
```json
{
  "articles": {
    "article_id": {
      "text_extracted": true,
      "text_extracted_at": "2025-09-26T14:00:00Z",
      "text_extraction_model": "milkey/reader-lm-v2:Q8_0"
    }
  }
}
```

**Purpose**:
- Track extraction status
- Record when extracted (for re-extraction)
- Version tracking (model used)

**New Fields in `{article_id}.json`**:
```json
{
  "local_content": {
    "text_path": "articles/{article_id}.txt",
    "text_size_bytes": 12345,
    "text_extraction_model": "milkey/reader-lm-v2:Q8_0"
  }
}
```

## Integration Points

### CLI Activation

**Disabled by Default**: Must explicitly enable
```bash
# Extract during stock update
python -m portfolio_advisor.cli \
    --mode stock \
    --ticker AAPL \
    --extract-text \
    --output-dir ./output

# Standalone extraction mode
python -m portfolio_advisor.cli \
    --mode extract-text \
    --ticker AAPL \
    --force \
    --output-dir ./output
```

**Why Disabled**:
- Model quality not production-ready
- Slow processing time
- Requires Ollama running locally
- Optional enhancement, not core feature

### Stock Graph Integration

**Conditional Node**:
```python
if "primary.text_extraction" in requested:
    # Run extraction service
    # Update news index
```

**Not in Default Pipeline**: Must be explicitly requested

## Paths Not Taken

### Direct HTML Parsing
- **Not Taken**: BeautifulSoup/lxml for text extraction
- **Why**: Too fragile, site-specific
- **LLM Advantage**: Adaptive to different site structures

### Cloud LLM Services
- **Not Taken**: OpenAI/Claude for extraction
- **Why**: Cost (many articles), privacy (article content)
- **Local Advantage**: Free, private, offline capable

### Different Models
- **Not Taken**: GPT-4, Claude, or other LLMs
- **Why**: Cost, latency, API dependency
- **Not Taken**: Smaller/faster models
- **Why**: Quality already marginal with reader-lm-v2

### Real-Time Extraction
- **Not Taken**: Extract immediately after download
- **Why**: Too slow for interactive use
- **Decision**: Batch background processing

### Structured Extraction
- **Not Taken**: Extract title, author, date as separate fields
- **Why**: Added complexity, current model struggles
- **Decision**: Plain text output simpler

### Quality Filtering
- **Not Taken**: Automatic quality scoring and rejection
- **Why**: No reliable metric for "good" extraction
- **Future**: Could compare length, coherence, etc.

## Why This Feature Is Disabled

### Primary Reasons

1. **Model Quality Issues**:
   - Generic content generation instead of verbatim extraction
   - Mixing content from ads, sidebars, related articles
   - Hallucination of phrases not in original

2. **Performance Issues**:
   - 30-60 seconds per large article
   - Not practical for real-time use
   - Portfolio-wide extraction takes hours

3. **Inconsistent Results**:
   - Same article can produce different extractions
   - Quality varies widely by site structure
   - No way to verify correctness automatically

4. **Dependency Requirement**:
   - Requires Ollama running locally
   - Model must be pulled (~4GB download)
   - Setup barrier for users

### When It Might Be Useful

- Research: Manual review of extracted text
- Experimentation: Testing different prompts/models
- Selective extraction: High-value articles only
- Fallback: When article HTML can't be parsed

## Key Learnings

1. **LLM text extraction is hard**: Not as simple as it seems
2. **Model matters enormously**: Reader-lm-v2 not good enough
3. **HTML preprocessing helps**: But doesn't solve core issues
4. **Slow is a dealbreaker**: Can't wait minutes per article
5. **Better to disable than ship poor quality**: User trust matters

## Performance Characteristics

**Single Article**:
- Preprocessing: ~50ms
- LLM call: 10-60 seconds (varies by article size)
- Save: ~10ms
- **Total**: 10-60 seconds

**Ticker (20 articles)**:
- Sequential: 3-20 minutes
- Could parallelize but risks overloading Ollama

**Portfolio (50 tickers, 1000 articles)**:
- Sequential: 3-20 hours
- Impractical for regular use

## Error Handling

### Ollama Unavailable
- **Check**: `ollama_service.is_available()`
- **Action**: Log warning, skip extraction
- **User**: Informed to start Ollama

### Model Not Found
- **Error**: Ollama returns model not found
- **Action**: Log error with pull instructions
- **Command**: `ollama pull milkey/reader-lm-v2:Q8_0`

### Extraction Timeout
- **Cause**: Very large HTML files
- **Handling**: httpx timeout (60s default)
- **Fallback**: Skip article, log error

### Quality Issues
- **Detection**: Not implemented
- **Handling**: Save whatever returned
- **User**: Manual review needed

## Testing Approach

**Unit Tests**:
- HTML preprocessing logic
- Prompt construction
- Metadata update logic

**Integration Tests**:
- Full extraction with mocked Ollama
- Index update correctness
- Force re-extraction

**Manual Testing**:
- Real articles from various sources
- Quality evaluation (subjective)
- Performance measurement

## Configuration

**Settings**:
```python
ollama_base_url: str = "http://localhost:11434"
ollama_timeout_s: int = 60
extraction_model: str = "milkey/reader-lm-v2:Q8_0"
extraction_batch_size: int = 10  # For batch processing
```

**Not Configurable** (hard-coded):
- Prompt template
- HTML size limit (100K chars)
- Extraction file extension (.txt)

## TODO: Future Improvements

### Critical (To Make Feature Viable)
- [ ] **Test alternative models**: Try other extraction-focused LLMs
- [ ] **Hybrid approach**: Traditional parsing + LLM for ambiguous cases
- [ ] **Quality scoring**: Auto-detect poor extractions
- [ ] **Caching/retries**: Handle temporary Ollama failures

### High Priority
- [ ] Async/parallel extraction for better throughput
- [ ] Progress reporting for long extractions
- [ ] Extraction queue management
- [ ] Skip already-extracted articles by default

### Medium Priority
- [ ] Structured extraction (title, author, date fields)
- [ ] Multiple model support (fallback chain)
- [ ] Prompt engineering experiments
- [ ] Quality metrics (coherence, completeness)

### Research
- [ ] Fine-tune model on news article dataset
- [ ] Compare LLM extraction vs traditional parsing
- [ ] Optimal prompt templates per publisher
- [ ] Active learning for quality improvement

## Alternative Approaches to Consider

### Readability Algorithms
- **Option**: Use readability.js-style extraction
- **Pro**: Fast, deterministic
- **Con**: Site-specific heuristics, brittle

### API-Based Services
- **Option**: Use diffbot, mercury, or similar
- **Pro**: High quality, maintained
- **Con**: Cost, API dependency, rate limits

### Publisher APIs
- **Option**: Use official news APIs where available
- **Pro**: Highest quality, structured
- **Con**: Limited coverage, authentication

### Hybrid: Parse + LLM Cleanup
- **Option**: Parse with BS4, clean with LLM
- **Pro**: Faster, better quality
- **Con**: More complex pipeline

## Recommendation

**Short Term**: Keep disabled, use for research only
**Medium Term**: Explore alternative models or hybrid approach
**Long Term**: Consider API-based solution or custom fine-tuned model

The feature serves as infrastructure for future improvements, but current quality doesn't justify enabling by default.

## Dependencies

**Core**:
- `ollama` (external service, must be running)
- `httpx`: HTTP client for Ollama API

**Utilities**:
- `stocks.db.StockPaths`: Path management
- `utils.fs`: File operations

## Known Issues / Limitations

1. **Poor extraction quality**: Primary blocker
2. **Slow processing**: 10-60s per article
3. **No quality validation**: Can't detect bad extractions
4. **Single model only**: No alternatives
5. **No parallelization**: Sequential processing
6. **Ollama dependency**: Setup barrier
7. **No progress tracking**: Black box during extraction
8. **Memory usage**: Large prompts consume RAM

## Related Documentation

- Design: `docs/design/feature-design-article-text-extraction.md`
- Stock News: `docs/implementation/stock-news-implementation.md`
- Third-party: `docs/thirdparty/readerlm-v2-readme.md`

