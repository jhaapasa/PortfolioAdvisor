## Stock Analysis Plan

This document defines a tight, minimal, and extensible plan for daily stock data collection and layered analyses, implemented as a separate LangGraph graph and a simple file-based "database" for inspectability.

### Design Guidelines

1. Separate graph: Implement stock data/analysis in its own LangGraph graph that can be called by the main agent or run independently.
2. File-based database: Persist primary and derived data as files inside a designated directory for readability/inspectability; no external DB required.
3. Additive database: Add new stocks as they appear; do not delete stocks.
4. Lazy updates: Update on demand when new stocks appear, a new trading day is available, or a requested analysis is missing/stale.
5. Primary data via Polygon.io: Fetch daily OHLC time series and store as JSON.
6. Modular, layered analyses: Each analysis writes a separate file; easy to extend with new analysis types and formats.
7. Canonical tickers: Use canonical internal ticker names for directory and file names to ensure discoverability.

### Goals and Scope (Initial)

- Daily OHLC price data for all available history up to last complete trading day
- Derived analyses:
  - Historical volatility (annualized), computed from daily log returns
  - Trailing returns for week, month, year (5, 21, 252 trading days)
  - Simple Moving Averages (SMA) for 20, 50, 100, 200-day windows

### Directory Layout (Stock Database)

- Base directory: `output/stocks/`
- Structure:

```
output/stocks/
  index.json                     # Optional: high-level metadata and ticker registry
  tickers/
    AAPL/                        # Canonical ticker directory
      meta.json                  # Per-ticker metadata and last-updated markers
      primary/
        ohlc_daily.json          # Primary daily OHLC time series
      analysis/
        returns.json             # Trailing returns (5/21/252)
        volatility.json          # Historical volatility (e.g., 21-day annualized)
        sma_20_50_100_200.json   # SMAs as time series (20/50/100/200)
    MSFT/
      ...
```

Notes:
- Canonical tickers come from existing symbol resolution (`tools/symbol_resolver.py`) and canonical models (`models/canonical.py`).
- The database is append-only in terms of tickers. Files are updated in place, with clear metadata for last updated dates.

### File Formats

- `meta.json` (per ticker):
  - `ticker`: canonical ticker string
  - `last_complete_trading_day`: ISO date for most recent complete trading day known
  - `artifacts`: map of artifact name → `{ last_updated: ISO datetime, depends_on: [...], version }`

- `primary/ohlc_daily.json`:
  - `ticker`: canonical ticker
  - `source`: `polygon.io`
  - `price_currency`: e.g., `USD`
  - `fields`: `date, open, high, low, close, volume, vwap?`
  - `data`: array of daily bars sorted ascending by date
  - `coverage`: `{ start_date, end_date }`
  - `generated_at`: ISO datetime

- `analysis/returns.json` (aggregate):
  - `ticker`
  - `as_of`: last complete trading day used
  - `windows`: `{ d5, d21, d252 }` returns as decimals
  - `method`: `simple_total_return`
  - `depends_on`: `["primary.ohlc_daily"]`
  - `generated_at`

- `analysis/volatility.json` (aggregate):
  - `ticker`
  - `as_of`
  - `window`: `21` (initial default)
  - `annualization_factor`: `sqrt(252)`
  - `volatility`: decimal (e.g., 0.22 = 22%)
  - `method`: `std(log_returns) * annualization_factor`
  - `depends_on`: `["primary.ohlc_daily"]`
  - `generated_at`

- `analysis/sma_20_50_100_200.json` (time series):
  - `ticker`
  - `windows`: `[20, 50, 100, 200]`
  - `data`: array of `{ date, sma20?, sma50?, sma100?, sma200? }`
  - `coverage`: `{ start_date, end_date }`
  - `depends_on`: `["primary.ohlc_daily"]`
  - `generated_at`

All dates use `YYYY-MM-DD`. All times use ISO 8601 with timezone `Z` where applicable.

### Lazy Update Policy

- Triggers:
  - New ticker observed during portfolio analysis → create directory, fetch full OHLC history, compute baseline analyses
  - New trading day detected (via market calendar or Polygon) → append missing OHLC bars and update analyses incrementally
  - Analysis requested but missing/stale (e.g., parameters changed) → recompute the specific analysis only
- Staleness rules:
  - `primary/ohlc_daily.json` is stale if `coverage.end_date < last_complete_trading_day`
  - An analysis is stale if its `generated_at` precedes the latest `primary` coverage end, or its configuration/version changed
- Incremental updates:
  - Prefer appending only new bars, not refetching entire history
  - Recompute analyses only for new ranges where possible (e.g., incremental SMA/returns), falling back to full recompute if needed

### LangGraph: Stock Data and Analysis Graph

Module proposal: `src/portfolio_advisor/graphs/stocks.py` (separate from `graph.py`). Nodes:

- `ResolveCanonicalTickerNode`: maps input tickers to canonical form using existing tools
- `CheckDatabaseStateNode`: reads ticker `meta.json` and determines required updates (primary and/or analyses)
- `FetchDailyOHLCNode`: uses `services/polygon_client.py` to fetch missing OHLC bars; writes `primary/ohlc_daily.json`
- `ComputeReturnsNode`: computes trailing 5/21/252-day returns; writes `analysis/returns.json`
- `ComputeVolatilityNode`: computes 21-day annualized volatility; writes `analysis/volatility.json`
- `ComputeSmaNode`: computes 20/50/100/200 SMAs; writes `analysis/sma_20_50_100_200.json`
- `CommitMetadataNode`: updates `meta.json` with `artifacts.*.last_updated` and `last_complete_trading_day`

### Report images

- Per-ticker `report/` directory contains rendered charts for Markdown embedding.
- Currently generated:
  - `report/candle_ohlcv_1y.png`: 1-year candlestick with volume (PNG, 1200×600, dpi=150).
- Example embed in Markdown:
  `![OHLCV 1Y](output/stocks/tickers/<slug>/report/candle_ohlcv_1y.png)`

Entrypoints:
- `update_ticker(ticker, requested_artifacts=all)`: ensures primary and requested analyses are fresh
- `update_all_for_portfolio(tickers, requested_artifacts)`: batch orchestration from the main agent

Behavior:
- Idempotent: running the graph multiple times yields the same files unless new data/parameters appear
- Fault-tolerant: retries with backoff on network errors; partial results do not corrupt existing files
- Observable: each node logs concise actions via existing logging config

### Analysis Details (Initial Set)

- Trailing returns (5/21/252): `close[t] / close[t-n] - 1` (skip if insufficient history)
- Historical volatility (21d): `std(log(close_t/close_{t-1})) * sqrt(252)` (requires ≥ 21 returns)
- SMAs: rolling mean of close over 20/50/100/200 days; emit only when window coverage is complete

### Best-Practice Tweaks

- Use a single, minimal `meta.json` per ticker for update orchestration and provenance
- Name artifacts with stable keys: `primary.ohlc_daily`, `analysis.returns`, `analysis.volatility`, `analysis.sma_20_50_100_200`
- File locking: use a lightweight file lock (single-process safe) to avoid concurrent writes; if concurrency is later added, extend to per-artifact locks
- Version fields on artifacts to enable non-breaking schema evolution
- Keep polygon responses normalized (store only needed fields) to reduce file size; retain provenance fields in headers

### Integration With Main Agent

- The main agent calls `update_all_for_portfolio` with the set of canonical tickers it sees
- For unknown tickers, the stock graph creates folders and seeds initial data/analyses lazily
- For daily runs, the main agent can emit a `ensure_fresh=True` flag to trigger day-level refresh checks

### Testing Strategy

- Unit tests for:
  - Canonicalization mapping and directory naming
  - Polygon fetch pagination/windowing and normalization
  - Incremental append logic for OHLC
  - Each analysis module on small, deterministic fixtures
- Integration tests:
  - End-to-end update of a new ticker (cold start)
  - Daily incremental refresh producing minimal diffs
  - Idempotency (no changes when up-to-date)

### Milestones

1. Scaffolding: directory layout helpers, `meta.json` management, and file locking
2. Primary data: Polygon OHLC full-history fetch + incremental append
3. Analyses: returns, volatility(21d), SMAs(20/50/100/200)
4. LangGraph: wire nodes, implement `update_ticker` and portfolio batch entrypoints
5. Tests: unit and integration for the above; basic docs update

### Minimal Configuration

- Database root: `output/stocks/` (fixed for now)
- Volatility window: `21` (fixed for now)
- Trading-day counts: week=5, month=21, year=252 (fixed for now)

These can become parameters later if/when needed.


