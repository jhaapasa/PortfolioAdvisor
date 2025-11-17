# Stock Analysis Implementation Notes

## Overview

The stock analysis feature provides a file-based database for storing and analyzing daily stock data. Implemented as a separate LangGraph graph (`graphs/stocks.py`) that can be called by the main portfolio agent or run independently.

**Design Document**: `docs/design/stock-analysis-plan.md`

**Implementation Date**: September 2024

**Status**: Production-ready, actively used

## Core Implementation Approach

### File-Based Database

**Choice**: JSON files instead of a traditional database
- **Why**: Inspectability, simplicity, version control friendly
- **How**: Each ticker gets a directory under `output/stocks/tickers/{slug}/`
- **Trade-offs**: Less efficient for large-scale queries, but sufficient for portfolio-sized datasets

```
output/stocks/tickers/{instrument_id_slug}/
  meta.json                    # Metadata and update tracking
  primary/
    ohlc_daily.json            # Primary OHLC data from Polygon
    news/                      # News articles (separate feature)
  analysis/
    returns.json               # Trailing returns (1/5/21/252 days)
    volatility.json            # Annualized volatility
    sma_20_50_100_200.json     # Simple moving averages
    wavelet_*.json             # Wavelet analysis results
  report/
    *.png                      # Generated chart images
```

### Lazy Update Strategy

**Implementation**: Updates triggered only when needed
- New ticker appears in portfolio → full historical fetch
- New trading day detected → incremental append
- Analysis missing/stale → recompute specific analysis only

**Staleness Detection**:
```python
# Primary data is stale if:
coverage.end_date < last_complete_trading_day

# Analysis is stale if:
analysis.generated_at < primary.generated_at
```

### Atomic Writes

**Critical Detail**: All writes use `.tmp` swap pattern
```python
# Pattern from utils.fs.write_json_atomic
temp_path = path.with_suffix('.tmp')
temp_path.write_text(json.dumps(data))
os.replace(temp_path, path)  # Atomic on POSIX
```

**Why**: Prevents partial writes from corrupting data during crashes or interrupts.

## Component Implementations

### StockPaths (`stocks/db.py`)

**Purpose**: Path management abstraction
- Maps logical artifact names to filesystem paths
- Uses instrument_id slugs for directory names (not raw tickers)
- Ensures consistent path construction across modules

**Key Design Decision**: Slug-based directories
- **What**: `cid-stocks-us-xnas-aapl` instead of `AAPL`
- **Why**: Uniqueness across markets/exchanges, URL-safe
- **How**: `utils.slug.slugify()` from canonical instrument_id

### Primary Data Fetching (`graphs/stocks.py`)

**Polygon Integration**:
```python
def _fetch_daily_ohlc_node(state):
    # Uses services/polygon_client.py
    # Fetches all available history on first run
    # Appends only new bars on subsequent runs
```

**Incremental Logic**:
1. Check existing `coverage.end_date`
2. Fetch from `end_date + 1` to `last_complete_trading_day`
3. Append new bars to existing data array
4. Update coverage metadata

**Trade-off**: Fetching full history vs incremental
- Chose incremental for efficiency
- Fallback to full refetch if gaps detected

### Analysis Modules (`stocks/analysis.py`)

**Trailing Returns**:
- Windows: 1, 5, 21, 252 trading days
- Method: Simple total return `(close_t / close_t-n) - 1`
- Handles insufficient history gracefully (returns null)

**Volatility**:
- Window: 21 days (fixed for now)
- Method: `std(log_returns) * sqrt(252)`
- Annualization factor: 252 trading days

**SMAs**:
- Windows: 20, 50, 100, 200 days
- Stored as time series (not just latest values)
- Only computes when full window available

### Wavelet Analysis (`stocks/wavelet.py`)

**Major Addition**: MODWT-based multi-scale decomposition
- Transform: Maximal Overlap Discrete Wavelet Transform
- Wavelet: sym4 (near-symmetric, good for derivatives)
- Levels: Up to 6 (captures ~2-128 day cycles)

**Implementations**:
1. `compute_modwt_logreturns()`: Decompose log returns for volatility
2. `compute_modwt_logprice()`: Decompose log prices for trend
3. `reconstruct_logprice_series()`: Rebuild smooth trends per level

**COI Handling**:
- Calculate cone of influence boundaries per level
- Mark unreliable regions near time series edges
- Used for visualization (dotted vs solid lines)

**Variance Spectrum**:
- Decomposes total variance across time scales
- Shows which frequencies contribute most to risk
- Stored in `volatility_histogram.json`

## LangGraph Pipeline

**Graph Structure** (`graphs/stocks.py`):
```
check_db_state → [conditional: what needs updating]
    ├→ fetch_ohlc → compute_returns → compute_volatility → compute_sma
    ├→ compute_wavelet_logreturns → compute_wavelet_variance
    ├→ compute_wavelet_logprice → reconstruct_trends
    └→ generate_plots → commit_metadata
```

**State Management**:
- Minimal state: just `instrument_id`, `slug`, `settings`
- Each node reads/writes files directly
- No state accumulation (prevents memory issues)

**Entry Points**:
1. `update_all_for_instruments(settings, instruments)`: Batch mode from main graph
2. Individual ticker updates via direct node invocation

## Plotting (`stocks/plotting.py`)

**Chart Types**:
1. `candle_ohlcv_1y.png`: 1-year candlestick with volume
2. `candle_ohlcv_2y_wavelet_trends.png`: 2-year with wavelet trends overlay

**Wavelet Trend Visualization**:
- Overlays smooth trends from multiple levels (S1-S6)
- Uses COI boundaries for dotted/solid line distinction
- Color coding: Different colors per time scale
- Alpha gradients within COI for progressive uncertainty (subtle)

**Technical Choices**:
- Library: matplotlib with mplfinance for candlesticks
- Format: PNG at 150 DPI, 1200×600 size
- Style: Dark background, high contrast

## Integration with Main Graph

**Call Site** (`graph.py:_update_stocks_node`):
```python
def _update_stocks_node(state: GraphState):
    instruments = state.get("instruments", [])
    update_all_for_instruments(settings, instruments)
```

**Flow**:
1. Main graph resolves portfolio holdings
2. Commits holdings to portfolio persistence
3. Extracts instrument list
4. Passes to stock update node
5. Stock graph updates all tickers in parallel
6. Returns to main graph for basket analysis

**Error Handling**: Try-except at integration point
- Stock failures don't crash main graph
- Allows partial portfolio analysis

## Paths Not Taken

### Database Choice
- **Not Taken**: PostgreSQL or SQLite
- **Why**: Added complexity, harder to inspect, version control unfriendly
- **Trade-off**: Slower queries, but acceptable for portfolio scale

### Update Frequency
- **Not Taken**: Real-time or intraday updates
- **Why**: Daily data sufficient for portfolio analysis
- **Decision**: Focus on end-of-day completeness

### Parallel Processing
- **Not Taken**: Multi-processing for ticker updates
- **Why**: Polygon rate limits, file locking complexity
- **Current**: Sequential with potential for async in future

### Volume Analysis
- **Not Taken**: Deep volume analysis beyond basic OHLCV
- **Why**: Price-focused for initial release
- **Note**: Design allows future expansion

### Alternative Wavelet Approaches
- **Not Taken**: Continuous Wavelet Transform (CWT)
- **Why**: MODWT better for time-localized analysis
- **Not Taken**: DWT instead of MODWT
- **Why**: MODWT is shift-invariant (critical for finance)

### COI Visualization
- **Not Taken**: Confidence bands (see separate doc)
- **Not Taken**: Color gradients for distortion
- **Why**: Visual clutter, dotted lines sufficient

## Key Learnings

1. **Atomic writes are critical**: Lost data during development before implementing
2. **Slug-based naming prevents conflicts**: Had ticker collision issues early on
3. **Lazy updates work well**: Avoids unnecessary API calls
4. **Wavelet preprocessing matters**: Wrong input (raw prices vs log returns) gives meaningless results
5. **COI is essential**: Users need to know where data is reliable

## Performance Characteristics

**Initial Ticker Load**:
- Full history fetch: ~2-5 seconds (depends on history length)
- All analyses: ~1-2 seconds
- Chart generation: ~500ms per chart
- **Total cold start**: ~5-10 seconds per ticker

**Incremental Daily Update**:
- Fetch 1 day: <500ms
- Update analyses: <500ms
- Regenerate charts: ~500ms
- **Total daily update**: ~1-2 seconds per ticker

**Portfolio Scale** (50 tickers):
- Initial load: ~5-10 minutes (one-time)
- Daily updates: ~1-2 minutes
- Disk usage: ~2-5 MB per ticker

## Testing Approach

**Unit Tests**:
- Individual analysis functions with deterministic fixtures
- Path construction and slug generation
- COI calculation correctness
- Atomic write mechanics

**Integration Tests**:
- End-to-end ticker update (cold start)
- Incremental update produces minimal diffs
- Idempotency (no changes when up-to-date)
- Error recovery (partial data, missing files)

**Mocking Strategy**:
- Polygon API responses mocked for deterministic tests
- File system operations use temp directories
- Date/time frozen for reproducibility

## Configuration

**Fixed Parameters** (not yet configurable):
- Volatility window: 21 days
- SMA windows: [20, 50, 100, 200]
- Wavelet: sym4
- Wavelet levels: 6
- Trading days/year: 252

**Rationale**: Start simple, add configurability when needed

## TODO: Future Improvements

### High Priority
- [ ] Add d1 (1-day) return to standard analysis (currently ad-hoc)
- [ ] Configurable wavelet parameters (wavelet family, levels)
- [ ] Multi-timeframe support (weekly, monthly aggregates)
- [ ] Batch polygon fetching for rate limit efficiency

### Medium Priority
- [ ] Parallel ticker updates with proper locking
- [ ] Incremental chart updates (only regenerate if data changed)
- [ ] Custom analysis plugins (user-defined indicators)
- [ ] Compression for old data (gzip JSON for archives)
- [ ] Database migration tool for schema changes

### Low Priority / Exploratory
- [ ] Volume-weighted analysis (beyond VWAP)
- [ ] Correlation analysis across portfolio tickers
- [ ] Regime detection using wavelets
- [ ] Interactive plots (plotly/bokeh instead of PNG)
- [ ] WebSocket for real-time updates
- [ ] Export to CSV/Excel for external tools

### Research Items
- [ ] Alternative wavelet families comparison (db4, coif4)
- [ ] Adaptive COI thresholds based on analysis type
- [ ] Machine learning features from wavelet coefficients
- [ ] Multi-resolution correlation analysis

## Dependencies

**Core**:
- `polygon-api-client`: Market data fetching
- `pywt` (PyWavelets): Wavelet transforms
- `numpy`: Numerical operations
- `matplotlib`: Chart generation
- `mplfinance`: Candlestick charts

**Utilities**:
- `langgraph`: Workflow orchestration
- Custom: `utils.fs`, `utils.slug`

## Known Issues / Limitations

1. **No timezone handling**: Assumes UTC, market hours not considered
2. **Missing data handling**: Gaps in OHLC not explicitly detected
3. **File locking**: Single-process only, no distributed locking
4. **Memory usage**: Large histories loaded entirely into memory
5. **Error propagation**: Silent failures in batch mode

## Related Documentation

- Design: `docs/design/stock-analysis-plan.md`
- COI Implementation: `docs/implementation/wavelet-coi-work-summary.md`
- Architecture: `docs/Architecture.md`
- Wavelet Theory: `docs/research-wavelet-analysis.md`

