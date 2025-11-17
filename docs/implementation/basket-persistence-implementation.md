# Basket Persistence and Analysis Implementation Notes

## Overview

Portfolio persistence and basket analysis were implemented to provide a structured, file-based storage for portfolio state and automated basket-level reporting. The system persists canonical holdings, tracks changes over time, and generates LLM-powered summaries for each investment basket.

**Design Document**: `docs/design/feature-design-basket-persistence-and-analysis.md`

**Implementation Date**: October 2024

**Status**: Production-ready, integrated into main graph

## Core Implementation Approach

### File-Based Portfolio Storage

**Directory Structure**:
```
<portfolio_dir>/
  portfolio.json              # Metadata summary
  holdings.json               # Current canonical holdings
  baskets/
    index.json                # Basket registry
    <basket-slug>/
      basket.json             # Basket metadata
      positions.json          # Holdings in this basket
  history/
    portfolio_changes.jsonl   # Append-only change log
```

**Why File-Based**:
- Inspectable: Can view portfolio state with any text editor
- Version-controllable: Changes trackable in git
- Simple: No database setup required
- Atomic: Easy to implement safe concurrent access

### Atomic Writes with Lock Files

**Implementation** (`portfolio/persistence.py`):
```python
def write_current_holdings(portfolio_dir: str, holdings: list[dict]) -> str:
    paths = PortfolioPaths(root=Path(portfolio_dir))
    with dir_lock(paths.lock_dir()):
        # Sort for stable diffs
        sorted_holdings = sorted(holdings, key=lambda h: h.get("instrument_id", ""))
        write_json_atomic(paths.holdings_json(), sorted_holdings)
```

**Lock Strategy**:
- Directory-based lock: `portfolio/.lock/`
- Prevents concurrent writes to same portfolio
- Released automatically via context manager
- Simple but effective for single-machine use

**Atomic Write Pattern**:
1. Write to `.tmp` file
2. Use `os.replace()` for atomic swap
3. Handles crashes gracefully (no partial writes)

## Component Implementations

### PortfolioPaths (`portfolio/persistence.py`)

**Purpose**: Path abstraction for portfolio files
- Similar pattern to `StockPaths`
- Centralizes all path construction
- Makes refactoring file structure easier

**Key Methods**:
- `holdings_json()`: Current portfolio state
- `basket_dir(slug)`: Per-basket directory
- `changes_log()`: JSONL change history

### Holdings Persistence

**Data Model** (based on `CanonicalHolding`):
```json
{
  "instrument_id": "cid:stocks:us:XNAS:AAPL",
  "primary_ticker": "AAPL",
  "asset_class": "stocks",
  "locale": "us",
  "mic": "XNAS",
  "symbol": "AAPL",
  "company_name": "Apple Inc.",
  "currency": "USD",
  "as_of": "2025-09-15",
  "quantity": 120.0,
  "weight": 0.085,
  "account": "IRA",
  "basket": "Growth Tech",
  "source_doc_id": "Portfolio_Positions_Aug-30-2025 IRA.csv"
}
```

**Stable Sorting**:
- Always sort by `instrument_id` before writing
- Makes diffs human-readable
- Git-friendly (minimal line changes)

### Basket Derivation

**Implementation** (`_derive_baskets()`):
```python
def _derive_baskets(holdings: list[dict]) -> list[dict]:
    # Group holdings by basket label
    # Generate slug from label
    # Count positions per basket
    # Return sorted list
```

**Key Decisions**:
- Baskets derived from holdings, not stored separately
- Slugs generated consistently: `slugify(label)`
- Empty/none baskets excluded from basket registry
- Deterministic sorting for stable output

**Slug Generation**:
- `"Growth Tech"` → `"growth-tech"`
- `"Value & Dividends"` → `"value-dividends"`
- Lower-case, alphanumeric + hyphens only

### Change History (JSONL)

**Format**: One JSON object per line
```json
{"ts":"2025-09-16T12:34:56Z","as_of":"2025-09-15","op":"add","instrument_id":"cid:stocks:us:XNAS:NVDA","primary_ticker":"NVDA","next":{"weight":0.04,"quantity":20.0}}
{"ts":"2025-09-16T12:34:56Z","as_of":"2025-09-15","op":"update","instrument_id":"cid:stocks:us:XNAS:AAPL","primary_ticker":"AAPL","prev":{"weight":0.08},"next":{"weight":0.085}}
{"ts":"2025-09-16T12:34:56Z","as_of":"2025-09-15","op":"remove","instrument_id":"cid:stocks:us:XNAS:PYPL","primary_ticker":"PYPL"}
```

**Why JSONL**:
- Append-only: Never modify history
- Line-oriented: Easy to process with standard tools
- Human-readable: Can grep/tail for specific changes
- No database required

**Diff Logic** (`append_history_diffs()`):
```python
# Compare old vs new holdings by instrument_id
# Detect: additions, removals, updates
# Track changes in: quantity, weight, basket, account
# Write compact change records
```

**Tracked Fields**:
- `quantity`: Share count changes
- `weight`: Portfolio allocation changes
- `basket`: Basket reassignments
- `account`: Account transfers

**Not Tracked** (intentionally):
- Price changes (not held in holdings)
- Company name updates (metadata)
- Symbol changes (would be new instrument_id)

## Basket Analysis Graph

**Implementation** (`graphs/baskets.py`):

### Graph Structure
```
collect_inputs → compute_metrics → format_report → write_outputs
```

**State**: `BasketState` TypedDict
- Minimal state passing
- Each node reads from files or prior state
- No large data accumulation

### Node: collect_inputs

**Purpose**: Gather ticker-level data for basket
```python
def _collect_inputs_node(state: BasketState):
    basket = state["basket"]
    tickers = basket["tickers"]
    # Read returns.json for each ticker
    # Collect into rows for analysis
```

**Data Sources**:
- `output/stocks/tickers/{slug}/analysis/returns.json`
- Reads d1 and d5 trailing returns
- Handles missing files gracefully

### Node: compute_metrics

**Calculations**:
1. Equal-weight basket returns (d1, d5)
2. Top 3 movers up/down by d1
3. Top 3 movers up/down by d5

**Implementation**:
```python
# Average returns across tickers
d1_avg = mean([t["d1"] for t in tickers if t["d1"] is not None])

# Sort by return magnitude
sorted_by_d1 = sorted(tickers, key=lambda t: t["d1"], reverse=True)
top_d1_up = sorted_by_d1[:3]
top_d1_down = sorted_by_d1[-3:]
```

**Output Structure**:
```json
{
  "basket": {"id": "growth_tech", "label": "Growth Tech"},
  "as_of": "2025-09-15",
  "tickers": [
    {"ticker": "AAPL", "d1": 0.012, "d5": 0.034},
    {"ticker": "NVDA", "d1": -0.008, "d5": 0.041}
  ],
  "averages": {"d1": 0.002, "d5": 0.019},
  "top_movers": {
    "d1_up": ["AAPL", "MSFT", "AMZN"],
    "d1_down": ["NVDA", "GOOGL", "TSLA"]
  }
}
```

### Node: format_report

**LLM Integration**:
- Uses existing LLM factory from `llm.py`
- Prompt includes basket name, averages, top movers
- Generates 3-6 bullet points
- Focus: Notable developments and risks
- No recommendations (just observations)

**Prompt Strategy**:
```
Basket: Growth Tech
Equal-weight performance:
- Day: +0.2%
- Week: +1.9%

Top movers (day): AAPL +1.2%, MSFT +0.9%, AMZN +0.7%
Bottom movers (day): NVDA -0.8%, GOOGL -0.5%, TSLA -0.4%

Provide 3-6 concise bullets highlighting notable developments.
```

**Output**: Markdown section for basket highlights

### Node: write_outputs

**Files Created**:
1. `output/baskets/{slug}/metrics.json`: Quantitative metrics
2. `output/baskets/{slug}/report.md`: LLM-generated narrative

**Return Value**:
```python
{
  "basket_report": {
    "id": basket_id,
    "label": basket_label,
    "slug": basket_slug,
    "metrics_path": str(metrics_path),
    "report_path": str(report_path),
    "summary_text": report_markdown
  }
}
```

## Integration with Main Graph

**Call Site** (`graph.py:_commit_portfolio_node`):
```python
def _commit_portfolio_node(state: GraphState):
    # Write holdings, portfolio header, baskets
    holdings = state["resolved_holdings"]
    write_current_holdings(portfolio_dir, holdings)
    write_portfolio_header(portfolio_dir, holdings)
    write_baskets_views(portfolio_dir, holdings)
    
    # Prepare for downstream graphs
    baskets = _derive_baskets(holdings)
    instruments = [{"id": h["instrument_id"], ...} for h in holdings]
    
    return {"instruments": instruments, "baskets": baskets}
```

**Flow**:
1. Resolve holdings → canonical format
2. **Commit portfolio** (persistence layer)
3. Extract baskets and instruments
4. Fan-out to stock updates
5. Fan-out to basket analysis
6. Collect basket reports
7. Pass to analyst for final summary

### Analyst Integration

**Enhancement** (`agents/analyst.py`):
- Receives `basket_reports` from state
- Includes basket summaries in final analysis
- Section: "Basket Highlights"
- Format: One paragraph per basket

**Prompt Addition**:
```
## Basket Highlights

Growth Tech (12 positions):
{basket_summary_text}

Value & Dividends (10 positions):
{basket_summary_text}
```

## Paths Not Taken

### Database Storage
- **Not Taken**: PostgreSQL/SQLite for holdings
- **Why**: File-based aligns with stock analysis approach
- **Trade-off**: Harder to query, but more transparent

### Rich Basket Rules
- **Not Taken**: Explicit basket classification rules engine
- **Why**: Manual basket assignment sufficient initially
- **Future**: Could add auto-classification based on rules

### Weighted Basket Metrics
- **Not Taken**: Portfolio-weight or market-cap weighted averages
- **Why**: Equal-weight simpler, less data required
- **Decision**: Can add weighted mode later if needed

### Portfolio Valuation
- **Not Taken**: Marking to market with current prices
- **Why**: Out of scope for initial release
- **Note**: Holdings have weights but no dollar values yet

### Multi-Currency Support
- **Not Taken**: Currency conversion and normalization
- **Why**: Most portfolios single currency initially
- **Future**: Needed for international portfolios

### Advanced Change Tracking
- **Not Taken**: Full audit trail with reasons for changes
- **Why**: Simple diff sufficient for now
- **Not Taken**: Performance attribution
- **Why**: Requires historical prices and complex calculations

### Real-Time Updates
- **Not Taken**: WebSocket or polling for portfolio changes
- **Why**: Daily batch processing sufficient
- **Context**: Portfolio changes are infrequent

## Key Learnings

1. **JSONL is excellent for append-only logs**: Easy to process, human-readable
2. **Slugs prevent file system issues**: Basket names can have special chars
3. **Derived baskets work well**: Don't need separate basket definitions
4. **LLM summaries add value**: Better than just showing numbers
5. **Sorting holdings is critical**: Makes diffs meaningful

## Performance Characteristics

**Portfolio Commit**:
- 50 holdings: ~100ms
- Includes: Holdings write, basket derivation, history diff
- Dominated by: File I/O and JSON serialization

**Basket Analysis**:
- Per basket: ~2-5 seconds
- Breakdown: 0.5s collect, 0.5s compute, 2-4s LLM, 0.5s write
- Dominated by: LLM call for narrative generation

**Typical Portfolio** (3 baskets):
- Persistence: ~100ms
- Basket analysis: ~6-15 seconds total
- **Total overhead**: ~6-15 seconds on top of stock updates

## Testing Approach

**Unit Tests**:
- Basket derivation from holdings
- Diff calculation (add/update/remove)
- Slug generation consistency
- Path construction

**Integration Tests**:
- Full portfolio commit cycle
- Basket analysis end-to-end
- Change history accumulation over multiple runs
- Empty portfolio handling
- Baskets without holdings

**Mocking**:
- LLM responses mocked for deterministic tests
- File I/O uses temp directories
- Stock data fixtures for basket inputs

## Configuration

**Portfolio Directory**:
- Default: `<output_dir>/portfolio`
- Configurable via `--portfolio-dir` CLI flag
- Settings: `portfolio_dir` field

**Basket Analysis**:
- Return windows: d1 (1 day), d5 (5 days) - fixed
- Equal-weight only (no weighted mode yet)
- Top movers count: 3 - fixed

## TODO: Future Improvements

### High Priority
- [ ] Add historical portfolio snapshots (not just diffs)
- [ ] Support weighted basket metrics (by position size)
- [ ] Handle missing ticker data gracefully in basket analysis
- [ ] Add basket composition change detection

### Medium Priority
- [ ] Portfolio valuation with current prices
- [ ] Multi-currency support with FX conversion
- [ ] Basket drift tracking (deviation from target allocations)
- [ ] Performance attribution by basket
- [ ] Tax lot tracking for accounts

### Low Priority
- [ ] Basket rebalancing recommendations
- [ ] Historical basket performance charts
- [ ] Export basket reports to PDF
- [ ] Email/notification integration for changes
- [ ] Web dashboard for portfolio visualization

### Research Items
- [ ] Auto-basket classification based on holdings
- [ ] Machine learning for basket optimization
- [ ] Risk decomposition by basket
- [ ] Factor exposure by basket

## Dependencies

**Core**:
- `langgraph`: Workflow orchestration
- Custom LLM client: Basket narrative generation

**Utilities**:
- `utils.fs`: Atomic writes, directory locking
- `utils.slug`: Slug generation
- `models.canonical`: CanonicalHolding schema

## Known Issues / Limitations

1. **Single-process only**: Directory lock not distributed
2. **No historical snapshots**: Only current state + diffs
3. **No basket definitions**: Baskets exist only through holdings
4. **Equal-weight only**: No support for weighted metrics yet
5. **No portfolio valuation**: Weights but no dollar amounts
6. **No multi-currency**: Assumes single currency
7. **LLM dependency**: Basket reports require LLM access

## Related Documentation

- Design: `docs/design/feature-design-basket-persistence-and-analysis.md`
- Architecture: `docs/Architecture.md`
- Stock Analysis: `docs/implementation/stock-analysis-implementation.md`

