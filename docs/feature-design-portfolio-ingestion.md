# Design and plan for Portfolio Ingestion

This is the first step in creating a portfolio analysis agent team. We will ingest semi-structured and unstructired data, and distill an exact, schema-compliant stock portfolio from it for further analysis. We will implement analysis later.

## High-level approach

- **Goal**: Turn arbitrary input docs into a canonical portfolio with validated tickers, and account and basket
  assignments.

## Core agents (LangGraph nodes)

- **Ingestion Agent**: Collect raw docs (Markdown, Plain Text, HTML, Emails, CSV). Handles extraction to plain text
  and basic structure (sections, tables, footnotes).
- **LLM Parsing Agent**: Convert extracted text into a structured portfolio holdings draft using a strict JSON
  schema and few-shot examples.
- **Ticker Resolver Agent**: Map names/ISIN/CUSIP/SEDOL to primary tickers; handle duplicates, share
  classes, delistings, and ADRs; normalize to an internal instrument key. The result should be a ticket-level validated portfolio holdings data structure.
- **Validator/Merger Agent**: Detect conflicts (sum of weights ≠ 100%, currency mismatches, unknown
  tickers); loop for auto-fixes; merge multi-doc inputs. This acts as a last line of defense critic for the portfolio data, ensuring consistency beyond individual tickers.

## Orchestration with LangGraph

- **State shape (`PortfolioState`)**: `raw_docs`, `extracted_text`, `parsed_holdings`,
  `unresolved_entities`, `canonical_portfolio`, `analytics`, `insights`, `errors`, `artifacts`,
  `run_metadata`.
- **Flow**:
  1. Fan-out over docs → extract → parse accounts, baskets and tickers (map) → resolve tickers (map) →
     validate/merge (may loop)
- **Map/Reduce patterns**: Per-document parsing and per-ticker resolution in parallel; aggregate at
  merge/analytics stages.
- **Guardrails**: Structured outputs with JSON schema; retries on invalid schema

## Data models (Pydantic-style; illustrative only)

- **ParsedHolding**: `{ name, identifiers, quantity|weight, currency, as_of, source_doc, confidence, account, basket }`
- **CanonicalHolding**: `{ instrument_id, primary_ticker, exchange, quantity, weight, currency, account_id, basket_id, cost_basis, as_of }`
- **Basket**: `{ id, label, strategy|sector|theme, rules }`
- **Portfolio**: `{ id, version, holdings[], totals, valuation_ccy, benchmark_id, as_of }`

## Key tools/services (behind interfaces)

- **Symbol resolution**: `symbol_resolver.lookup(name|isin|cusip)` with provider abstractions;
  deterministic fallbacks and caching. Use https://polygon.io/docs/rest/stocks/tickers/all-tickers APIs as primary tool

## Repository structure (aligned with project rules)

- `src/models/` – portfolio, holding, basket, analytics, insight schemas
- `src/agents/` – `ingestion.py`, `parser.py`, `resolver.py`, `validator.py`
- `src/graph/portfolio_team.py` – LangGraph graph assembly, state, edges
- `src/tools/` – `symbol_resolver.py`
- `src/services/` – provider adapters (abstracted) - use polygon.io API (polygon-api-client for python)
- `tests/` – mirrors `src/`; fixtures for sample docs and golden outputs

## Prompts and guardrails

- **Extraction prompt**: Few-shot with a strict JSON schema; require totals and per-position
  confidence; instruct to output “unknown” rather than hallucinate.
- **Classification prompt**: Inputs include instrument metadata; return basket id(s) with confidence
  and rationale; enable deterministic rules to take precedence.
- **Conflict handling**: If totals don’t match constraints, propose a fix set; add a reconciliation
  reason.

## Testing and evaluation

- **Unit tests**: Deterministic fixtures (K-forms, fact sheets) → golden parsed JSON; symbol resolver
  edge cases; basket rules.
- **Property tests**: Sums and constraints invariants; idempotency under enrichment.
- **Regression tests**: Golden analytics outputs; snapshot diffs for insights text.
- **Metrics**: Parsing precision/recall, resolution accuracy, basket agreement rate, analytics
  stability.

## Observability and ops

- **Structured logs**: Include `run_id`, doc source, `as_of`, counts; LLM call durations and token
  use.
- **Tracing**: Per-agent spans; record inputs/outputs (redacted).
- **Caching**: Results for LLM structured outputs, lookups, prices/FX with TTL.
- **Rate limiting/retries**: For all external calls; exponential backoff.
- **Config**: Environment-based provider keys; no secrets in code.

## Incremental roadmap (safe, small steps)

1. **Milestone 1**: Ingestion → LLM parser with JSON schema + tests on 3 sample docs.
2. **Milestone 2**: Ticker resolver + validation loop; HITL acceptance path.
