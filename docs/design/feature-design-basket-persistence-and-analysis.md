# Design and plan for Basket Persistence and Basic Basket Analysis

This document defines a simple, opinionated on-disk portfolio persistence model and a first basket analysis workflow. It aligns with existing components (`graphs/stocks.py`, `models/canonical.py`) and keeps analysis results transient in `output/`.

## Goals and scope

- Persist the latest canonical portfolio to a dedicated, file-based `portfolio` folder.
- Log diffs between runs to a human-readable history file.
- After persistence, fan out to:
  - existing per-ticker technical analysis sub-graph, and
  - a new per-basket analysis sub-graph that summarizes recent developments.
- Feed basket reports into the final analyst agent for a skimmable 5-minute summary (day and week focus).

Out of scope (for now): valuations, multi-currency conversion, rich basket rules, long-term analysis history (beyond a simple JSONL change log), drift tracking, performance attribution.

## Design principles

- Keep it simple and opinionated. Defaults over options; add knobs later if needed.
- JSON files only; human-readable with stable keys and indentation.
- Overwrite the current state atomically each run; record changes to an append-only history file.
- Treat analysis outputs as transient artifacts under `output/` (not part of persistent portfolio state).
- Breaking changes acceptable while in early development.

## Configuration

- Add `portfolio_dir` to `Settings` and CLI `--portfolio-dir`.
  - Default if unset: `<output_dir>/portfolio` (keeps the single-source-of-truth colocated with outputs, but logically separate from `output/stocks`).
  - `Settings.ensure_directories()` creates `portfolio_dir` if missing.

## On-disk layout (file-based storage)

```
<portfolio_dir>/
  portfolio.json              # portfolio metadata summary (latest only)
  holdings.json               # array of canonical holdings (latest only)
  baskets/
    index.json                # array of baskets: { id, label, slug, size, last_updated }
    <basket-slug>/
      basket.json             # { id, label, strategy|sector|theme, notes }
      positions.json          # array subset of holdings for this basket (denormalized view)
  history/
    portfolio_changes.jsonl   # append-only, one JSON object per line

<output_dir>/
  stocks/                     # already implemented (per-ticker OHLC + analysis)
  baskets/
    <basket-slug>/
      metrics.json            # computed per-basket metrics (transient)
      report.md               # LLM-written human summary (transient)
```

Notes:
- `holdings.json` is the canonical latest snapshot (overwrite each run).
- `portfolio.json` is a compact header for quick inspection and runtime convenience.
- `baskets/*/positions.json` is derived from `holdings.json` (overwrite each run). No extra state is carried here beyond convenience.
- Only `history/portfolio_changes.jsonl` grows over time.

## Data models (storage shape)

Leverage existing `CanonicalHolding` fields from `src/portfolio_advisor/models/canonical.py`. A holding row in `holdings.json` uses a stable subset:

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

`portfolio.json` (concise header, overwrite each run):

```json
{
  "id": "default",
  "as_of": "2025-09-15",
  "valuation_ccy": "USD",
  "num_holdings": 42,
  "num_baskets": 3,
  "baskets": [
    { "id": "growth_tech", "label": "Growth Tech", "slug": "growth-tech", "size": 12 },
    { "id": "value_dividends", "label": "Value & Dividends", "slug": "value-dividends", "size": 10 }
  ]
}
```

`baskets/index.json` (overwrite each run): array of `{ id, label, slug, size, last_updated }`. Each `baskets/<slug>/basket.json` contains `{ id, label, strategy|sector|theme, notes }` if available.

History file: `history/portfolio_changes.jsonl` — one compact object per line:

```json
{"ts":"2025-09-16T12:34:56Z","as_of":"2025-09-15","op":"add","instrument_id":"cid:stocks:us:XNAS:NVDA","primary_ticker":"NVDA","next":{"weight":0.04,"quantity":20.0,"basket":"Growth Tech","account":"Taxable"}}
{"ts":"2025-09-16T12:34:56Z","as_of":"2025-09-15","op":"update","instrument_id":"cid:stocks:us:XNAS:AAPL","primary_ticker":"AAPL","prev":{"weight":0.08},"next":{"weight":0.085}}
{"ts":"2025-09-16T12:34:56Z","as_of":"2025-09-15","op":"remove","instrument_id":"cid:stocks:us:XNAS:PYPL","primary_ticker":"PYPL"}
```

Diff policy (first pass):
- Key by `instrument_id`.
- Consider changes in `quantity`, `weight`, `basket`, `account` as meaningful updates.
- `as_of` is carried into each entry; if absent, inherit from current run header.
- Write atomically using a `*.tmp` swap to avoid partial writes (same approach as `stocks/db.py`).

## Data flow changes (LangGraph)

Add a persistence stage and two fan-outs after the resolver join. The high-level flow becomes:

1) Ingestion → per-doc Parsing (fan-out) → join → per-holding Resolver (fan-out) → join.
2) Persist portfolio (new):
   - Write `holdings.json`, `portfolio.json`.
   - Derive baskets, write `baskets/index.json` and `baskets/<slug>/positions.json`.
   - Compute diff vs previous `holdings.json` and append `history/portfolio_changes.jsonl`.
3) Fan-out sub-graphs:
   - Stocks: call existing `graphs/stocks.update_all_for_portfolio(settings, tickers)` with all tickers in the portfolio.
   - Baskets: for each basket, run the new `baskets` sub-graph (see below).
4) Final analyst agent: consume basket reports and emit a portfolio-level summary.

New/updated state keys exchanged across nodes:
- `resolved_holdings`: list[dict] (from resolver) — canonical holdings driving persistence.
- `portfolio_persisted`: bool (marker for later stages).
- `basket_reports`: list[dict] of `{ id, label, slug, metrics_path, report_path, summary_text }`.

Implementation touchpoints:
- `src/portfolio_advisor/graph.py`: add a `commit_portfolio_node` before invoking stock and basket fan-outs; then pass tickers/baskets as inputs.
- `src/portfolio_advisor/graphs/stocks.py`: already provides `update_all_for_portfolio` — reuse unchanged.
- `src/portfolio_advisor/agents/analyst.py`: prompt updated to include basket summaries (see below).
- New helper module `src/portfolio_advisor/portfolio/persistence.py` for writing/reading portfolio files and computing diffs using atomic writes (mirrors `stocks/db.py` technique).

## New baskets sub-graph

Module: `src/portfolio_advisor/graphs/baskets.py`

Inputs per invocation:
- `settings`: `Settings`
- `basket`: `{ id, label, slug, tickers: list[str] }`

Nodes (sequential for simplicity):
1. `collect_inputs`:
   - Source of truth: per-ticker analysis artifacts already written by the stocks graph under `<output_dir>/stocks/tickers/<T>/analysis/`.
   - Files read (best-effort): `returns.json` (required), `volatility.json` (optional), `sma_20_50_100_200.json` (optional, for context only).
2. `compute_basket_metrics` (first capabilities):
   - For each ticker, extract trailing returns windows from `returns.json`.
   - Compute basket-level equal-weight averages for day and week.
     - Day = 1 trading day (`d1`), Week = 5 trading days (`d5`).
     - If `d1` is not yet produced by stocks analysis, compute via last two closes ad hoc; later we can add `d1` to `compute_trailing_returns` for consistency.
   - Identify top 3 movers up/down by `d1` and by `d5`.
   - Emit `metrics.json` with shape:

```json
{
  "basket": { "id": "growth_tech", "label": "Growth Tech", "slug": "growth-tech" },
  "as_of": "2025-09-15",
  "tickers": [
    { "ticker": "AAPL", "d1": 0.012, "d5": 0.034 },
    { "ticker": "NVDA", "d1": -0.008, "d5": 0.041 }
  ],
  "averages": { "d1": 0.002, "d5": 0.019 },
  "top_movers": {
    "d1_up": ["AAPL", "MSFT", "AMZN"],
    "d1_down": ["NVDA", "GOOGL", "TSLA"],
    "d5_up": ["NVDA", "AAPL", "AMZN"],
    "d5_down": ["PYPL", "INTC", "TSLA"]
  },
  "depends_on": ["stocks.analysis.returns"],
  "generated_at": "2025-09-16T12:34:56Z"
}
```

3. `summarize_with_llm` (LLM summary; simple initial prompt):
   - Prompt includes: basket name, equal-weight day/week averages, and per-ticker day/week returns (capped to 30 tickers).
   - Output: short markdown with 3–6 bullets highlighting notable developments and risks. No recommendations.
4. `write_outputs`:
   - Write `metrics.json` and `report.md` under `<output_dir>/baskets/<slug>/`.
   - Return pointers (`metrics_path`, `report_path`) and `summary_text` in state for the final analyst.

## Analyst agent changes

Extend `ANALYST_PROMPT_TEMPLATE` usage to include basket summaries:
- Provide a new section "Basket Highlights" listing each basket with the one-paragraph LLM summary generated by the baskets sub-graph.
- Keep the overall tone concise and factual, focusing on the past day and week.

Sketch of additional prompt variables:
- `basket_summaries`: array of `{ label, slug, averages: { d1, d5 }, highlights_md }`.

## Update semantics

- Overwrite current files using atomic write (`*.tmp` then `os.replace`).
- Keep `holdings.json` stable and sorted (by `instrument_id`) to make diffs human-friendly.
- Only `history/portfolio_changes.jsonl` is append-only; all other files are rewritten from the latest truth each run.
- Slugging rule for basket folder names: lower-case, non-alphanumeric → `-`, collapse repeats, trim `-`.

## Minimal implementation plan (incremental)

1) Persistence helpers
   - Create `portfolio/persistence.py` with:
     - `write_current_holdings(path, holdings)`
     - `write_portfolio_header(path, holdings)`
     - `write_baskets_views(path, holdings)`
     - `append_history_diffs(path, old_holdings, new_holdings, as_of)`
     - All functions use atomic writes; mirror patterns from `stocks/db.py`.

2) Graph wiring
   - Add `commit_portfolio_node` to `graph.py` after resolver join; write files and compute history diffs; produce `tickers` and `baskets` for downstream.
   - Call `update_all_for_portfolio(settings, tickers)` (existing API) to refresh per-ticker analysis.
   - For each basket, invoke `graphs/baskets.py` compiled app; collect `basket_reports` in state.

3) Analyst
   - Update `analyst_node` to include a "Basket Highlights" section based on `basket_reports`.

4) CLI/Settings
   - Add `portfolio_dir` to `Settings` and CLI `--portfolio-dir` (default `<output_dir>/portfolio`).

5) Tests (first pass)
   - Persistence: writes are atomic; files exist with expected shapes; diff emits expected JSONL entries.
   - Basket metrics: given synthetic per-ticker `returns.json`, compute averages and top movers deterministically.
   - Analyst: prompt includes basket summaries when present.

## Future improvements (not in first pass)

- Use portfolio weights (if present) for weighted basket averages (default remains equal-weight for simplicity).
- Add `d1` window to `compute_trailing_returns` to avoid ad hoc daily calc.
- Optional `report.json` alongside `report.md` for machine-readable basket highlights.
- Portfolio valuation snapshot and multi-currency normalization.
- Basket rule engine (explicit rules per basket with auto-classification) once we have consistent instrument metadata coverage.


