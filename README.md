# PortfolioAdvisor

A LangGraph-based portfolio analysis tool using AI agents to parse, resolve, and analyze investment holdings from multiple data sources.

## Overview

PortfolioAdvisor orchestrates a pipeline of specialized agents to:
- **Ingest** portfolio data from CSV, text, or HTML files
- **Parse** holdings using LLM-powered extraction
- **Resolve** ticker symbols via the Polygon.io API
- **Analyze** portfolio composition and generate reports
- **Track** stock performance with historical data and technical indicators

See `docs/Architecture.md` for the complete architecture overview.

## Quickstart

1. **Prerequisites:** Python 3.13.7 (e.g., via pyenv)

2. **Clone and bootstrap:**
   ```bash
   git clone https://github.com/jhaapasa/PortfolioAdvisor.git
   cd PortfolioAdvisor
   ./scripts/bootstrap
   source .venv/bin/activate
   ```

3. **Configure environment:**
   - Copy `env.example` to `.env`
   - Add your API keys:
     - `OPENAI_API_KEY` (required for LLM features)
     - `POLYGON_API_KEY` (optional, for symbol resolution and stock data)

4. **Run the analyzer:**
   ```bash
   portfolio-advisor --input-dir ./input --output-dir ./output
   ```

   The tool generates `analysis.md` and supporting files in the output directory.

## CLI Usage

The CLI supports three modes of operation:

### Portfolio Mode (default)

Analyze portfolio holdings from input files:

```bash
portfolio-advisor --input-dir ./input --output-dir ./output
```

**Key flags:**
- `--portfolio-dir` — Override portfolio state directory (defaults to `<output_dir>/portfolio`)
- `--include-news` — Generate LLM-based 7-day per-stock news + technical report

### Stock Mode

Update data for a single stock ticker:

```bash
portfolio-advisor --mode stock --ticker AAPL --input-dir ./input --output-dir ./output
```

**Key flags:**
- `--ticker` — Stock ticker symbol (e.g., `AAPL`)
- `--instrument-id` — Canonical instrument ID (alternative to ticker)
- `--wavelet` — Compute MODWT wavelet analysis
- `-J, --wavelet-level` — Wavelet decomposition level 1–8 (implies `--wavelet`)
- `--fetch-news/--no-fetch-news` — Fetch news articles (default: enabled)
- `--enable-boundary-extension` — Enable boundary stabilization for trend filters
- `--boundary-strategy` — Forecasting strategy: `linear` or `gaussian_process`
- `--boundary-steps` — Forecast steps for boundary extension (default: 10)
- `--boundary-lookback` — Lookback period for boundary model (default: 30)
- `--boundary-noise-injection` — Inject noise into boundary extension

### Extract-Text Mode

Extract text from cached news article HTML using local LLM (Ollama):

```bash
# Single ticker
portfolio-advisor --mode extract-text --ticker AAPL --input-dir ./input --output-dir ./output

# All tickers in portfolio
portfolio-advisor --mode extract-text --all-stocks --input-dir ./input --output-dir ./output
```

**Key flags:**
- `--ticker` or `--instrument-id` — Target ticker
- `--all-stocks` — Process all tickers in portfolio
- `--force-extraction` — Re-extract even if already processed
- `--extract-text/--no-extract-text` — Enable/disable text extraction

### Common Flags

**LLM Configuration:**
- `--openai-api-key` — Override OpenAI API key
- `--openai-base-url` — Custom API endpoint
- `--openai-model` — Model name (default: `gpt-4o-mini`)
- `--temperature` — Sampling temperature (default: `0.2`)
- `--max-tokens` — Max output tokens
- `--request-timeout-s` — Request timeout in seconds

**Parser Settings:**
- `--parser-max-retries` — Max retry attempts on validation errors
- `--parser-max-doc-chars` — Max characters per document

**Logging:**
- `--log-level` — Logging verbosity: `DEBUG`, `INFO`, `WARNING`, `ERROR`
- `--log-format` — Format style: `plain` or `json`
- `--verbose` — Enable DEBUG logs for portfolio_advisor package
- `--agent-progress` — Show LangGraph agent progress messages
- `--log-libraries` — Allow library logs (httpx, langchain, etc.)
- `--skip-llm-cache` — Bypass LLM cache reads (writes still occur)

## Configuration

Settings are loaded from environment variables (`.env` file in development) and can be overridden via CLI flags. Copy `env.example` to `.env` and fill in your values.

### CLI Flag to Environment Variable Mapping

Most CLI flags map directly to environment variables (e.g., `--log-level` → `LOG_LEVEL`). Notable exceptions:

| CLI Flag | Environment Variable |
|----------|---------------------|
| `--input-dir` | (CLI only, required) |
| `--output-dir` | (CLI only, required) |
| `--enable-boundary-extension` | `BOUNDARY_EXTENSION` |
| `--include-news` | `INCLUDE_NEWS_REPORT` |
| `--portfolio-dir` | `PORTFOLIO_DIR` |

### Required Settings
- `--input-dir` — Directory containing portfolio data files (CLI only)
- `--output-dir` — Directory for analysis output, created if needed (CLI only)

### Paths
- `PORTFOLIO_DIR` — Override portfolio state directory (defaults to `<output_dir>/portfolio`)

### API Keys
- `OPENAI_API_KEY` — OpenAI API key (required for LLM features; stub mode if absent)
- `POLYGON_API_KEY` — Polygon.io API key (optional; enables symbol resolution and stock data)

### LLM Configuration
- `OPENAI_MODEL` — Model name (default: `gpt-4o-mini`)
- `OPENAI_BASE_URL` — Custom API endpoint (optional)
- `TEMPERATURE` — Sampling temperature (default: `0.2`)
- `MAX_TOKENS` — Max output tokens (optional)
- `REQUEST_TIMEOUT_S` — Request timeout in seconds (default: `60`)

### Parsing Settings
- `PARSER_MAX_RETRIES` — Max retry attempts on validation errors (default: `2`)
- `PARSER_MAX_DOC_CHARS` — Max characters per document (default: `20000`)

### Symbol Resolution
- `POLYGON_BASE_URL` — Polygon.io API endpoint override (optional)
- `POLYGON_TIMEOUT_S` — Request timeout (default: `10`)
- `RESOLVER_DEFAULT_LOCALE` — Fallback market locale (default: `us`)
- `RESOLVER_PREFERRED_MICS` — Comma-separated exchange codes (default: `XNAS,XNYS,ARCX`)
- `RESOLVER_CONFIDENCE_THRESHOLD` — Minimum match score (default: `0.6`)

### Wavelet Analysis
- `WAVELET` — Enable wavelet analysis (default: `false`)
- `WAVELET_LEVEL` — Decomposition level 1–8 (default: `5`)

### News Integration
- `FETCH_NEWS` — Fetch news articles for stocks (default: `true`)
- `INCLUDE_NEWS_REPORT` — Generate 7-day news + technical reports (default: `false`)

### Boundary Stabilization
- `BOUNDARY_EXTENSION` — Enable boundary stabilization (default: `false`)
- `BOUNDARY_STRATEGY` — Forecasting strategy: `linear` or `gaussian_process` (default: `linear`)
- `BOUNDARY_LOOKBACK` — Lookback period for boundary model (default: `30`)
- `BOUNDARY_STEPS` — Forecast steps for boundary extension (default: `10`)
- `BOUNDARY_SANITIZATION` — Enable boundary sanitization (default: `false`)
- `BOUNDARY_NOISE_INJECTION` — Inject noise into boundary extension (default: `false`)

### Ollama / Text Extraction
- `OLLAMA_BASE_URL` — Ollama server URL (default: `http://localhost:11434`)
- `OLLAMA_TIMEOUT_S` — Request timeout (default: `120`)
- `EXTRACTION_MODEL` — Model for text extraction (default: `milkey/reader-lm-v2:Q8_0`)
- `EXTRACTION_BATCH_SIZE` — Articles per batch (default: `10`)
- `EXTRACTION_MAX_WORKERS` — Parallel extraction workers (default: `4`)

### Logging
- `LOG_LEVEL` — Logging verbosity: `DEBUG`, `INFO`, `WARNING`, `ERROR` (default: `INFO`)
- `LOG_FORMAT` — Format style: `plain` or `json` (default: `plain`)
- `VERBOSE` — Enable verbose agent logging (default: `false`)
- `AGENT_PROGRESS` — Show LangGraph execution details (default: `false`)
- `LOG_LIBRARIES` — Allow library logs at configured level (default: `false`)
- `SKIP_LLM_CACHE` — Bypass LLM cache reads (writes still occur)

### Example with CLI Overrides

```bash
portfolio-advisor \
  --input-dir ./input \
  --output-dir ./output \
  --openai-model gpt-4o-mini \
  --temperature 0.2 \
  --log-format json \
  --parser-max-retries 3
```

### Stub Mode Behavior
When `OPENAI_API_KEY` is not provided, the system uses a stub LLM that returns placeholder responses. This enables testing and demos without API costs.

## Features

### Portfolio Analysis Pipeline
- **Multi-format ingestion:** Reads CSV, TXT, HTML, Markdown, and email files
- **LLM-powered parsing:** Extracts holdings with automatic retry on validation errors
- **Symbol resolution:** Resolves tickers via Polygon.io with confidence scoring
- **Canonical tracking:** Maintains normalized holdings with instrument keys
- **Portfolio reports:** Generates summary analysis with holdings breakdown

### Stock Data & Analytics
- **Historical OHLC data:** Daily price history via Polygon.io
- **Technical indicators:** Returns (5/21/252-day), volatility (21-day annualized), SMAs (20/50/100/200-day)
- **Wavelet analysis:** Multi-timescale signal decomposition for trend detection
- **Boundary stabilization:** Price extension for trend filter edge effects
- **News integration:** Fetches and summarizes recent news articles per ticker
- **Visual reports:** Generates candlestick charts with volume for Markdown embedding

### Basket Analysis
- **Basket persistence:** Tracks custom stock groupings with metadata
- **Performance metrics:** Aggregates returns, volatility, and correlation across baskets
- **Comparative analysis:** Identifies top performers and risk contributors

## Development

### Setup
After cloning the repository, run the bootstrap script to set up the development environment:
```bash
./scripts/bootstrap
source .venv/bin/activate
```

### Development Workflow
- **Format code:** `./scripts/format` — Applies Black formatting
- **Lint code:** `./scripts/lint` — Runs Ruff linter
- **Run tests:** `./scripts/test` — Executes pytest with coverage report
- **Pre-commit:** Run all three commands before committing

### Testing
Tests are located in `tests/` and mirror the `src/` structure. Run with coverage:
```bash
source .venv/bin/activate
pytest -q --cov=src/portfolio_advisor --cov-report=term-missing
```

Target: maintain >80% code coverage.

### Architecture
See `docs/Architecture.md` for detailed design documentation, including:
- Agent orchestration flow (ingestion → parsing → resolution → analysis)
- State management and LangGraph integration
- Extension points for new providers and agents

## Project Structure

```
PortfolioAdvisor/
├── src/portfolio_advisor/     # Application source code
│   ├── agents/                # LangGraph agent implementations
│   ├── graphs/                # Multi-agent graph orchestration
│   ├── models/                # Data models (parsed, canonical, market)
│   ├── portfolio/             # Portfolio state persistence
│   ├── services/              # External API clients
│   ├── stocks/                # Stock data and analysis modules
│   ├── tools/                 # Reusable utilities
│   ├── trend/                 # Trend analysis (boundary stabilization)
│   └── utils/                 # Helper functions
├── tests/                     # Unit and integration tests
├── docs/                      # Technical documentation
│   ├── design/                # Feature design documents
│   ├── implementation/        # Implementation notes
│   └── research/              # Research notes (wavelets, COI)
├── scripts/                   # Development tooling
├── input/                     # Sample portfolio data
└── output/                    # Generated reports and data
```

## Logging and Error Handling

- Logging defaults to `INFO` level with plain text format
- Switch to JSON logs via `LOG_FORMAT=json` or `--log-format json`
- Sensitive values (API keys, credentials) are never logged
- Clear exceptions with context for configuration and I/O errors
- LangGraph and LangChain library noise is suppressed by default

## Documentation

Detailed documentation is available in the `docs/` directory:
- **Architecture.md** — System design and agent orchestration
- **design/** — Feature specifications and design documents
- **implementation/** — Implementation notes and status tracking
- **research/** — Wavelet analysis and COI research notes
- **thirdparty/** — External API documentation (Polygon.io)

## License

MIT License — See `LICENSE` file for details.
