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

## Configuration

Settings are loaded from environment variables (`.env` file in development) and can be overridden via CLI flags.

### Required Settings
- `--input-dir` — Directory containing portfolio data files
- `--output-dir` — Directory for analysis output (created if needed)

### API Keys
- `OPENAI_API_KEY` — OpenAI API key (required for LLM features; stub mode if absent)
- `POLYGON_API_KEY` — Polygon.io API key (optional; enables symbol resolution)

### LLM Configuration
- `OPENAI_MODEL` — Model name (default: `gpt-4o-mini`)
- `OPENAI_BASE_URL` — Custom API endpoint (optional)
- `TEMPERATURE` — Sampling temperature (default: `0.2`)
- `MAX_TOKENS` — Max output tokens (optional)
- `REQUEST_TIMEOUT_S` — Request timeout in seconds (default: `60`)

### Parsing Settings
- `PARSER_MAX_RETRIES` — Max retry attempts on validation errors (default: `2`)
- `PARSER_MAX_DOC_CHARS` — Max characters per document (default: `20000`)

### Logging
- `LOG_LEVEL` — Logging verbosity: `DEBUG`, `INFO`, `WARNING`, etc. (default: `INFO`)
- `LOG_FORMAT` — Format style: `plain` or `json` (default: `plain`)
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
│   ├── models/                # Data models (parsed & canonical)
│   ├── services/              # External API clients
│   ├── stocks/                # Stock data and analysis modules
│   └── tools/                 # Reusable utilities
├── tests/                     # Unit and integration tests
├── docs/                      # Technical documentation
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
- **stock-analysis-plan.md** — Stock data collection and analysis strategy
- **polygon-io-api-guide.md** — Comprehensive Polygon.io API reference
- **Feature designs** — Detailed specs for baskets, news, and portfolio ingestion
- **Wavelet analysis** — Research notes on multi-timescale signal decomposition

## License

MIT License — See `LICENSE` file for details.

