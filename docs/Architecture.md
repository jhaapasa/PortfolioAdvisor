# Architecture Overview

This document explains the overall design of PortfolioAdvisor, the data flow through its agents,
configuration surface, and guidance for extending the system.

## Goals
- Simple, reliable end-to-end portfolio ingestion and analysis
- Minimal, opinionated configuration with sensible defaults
- Clear boundaries between ingestion, parsing, resolution, and analysis

## High-level Flow

1. CLI gathers paths and overrides and calls the application entrypoint.
2. The entrypoint builds settings, configures logging, and initializes the LangChain cache.
3. A LangGraph state machine orchestrates agents:
   - ingestion → planner → parse (fan-out) → resolve (fan-out) → analyst
4. Results are compiled into `analysis.md` and a debugging JSON file with resolved/unresolved items.

Sequence sketch:

```
cli → analyze_portfolio → build_graph →
  ingestion ──> planner ──> dispatch_parse ──(fan-out: doc)→ parse_one ──> join_after_parse ──>
  dispatch_resolve ──(fan-out: holding)→ resolve_one ──> join_after_resolve ──> analyst → END
```

## Graph State Contract

The LangGraph state is a dictionary with well-known keys:
- `settings`: application settings instance
- `raw_docs`: list[dict] from ingestion (path, name, mime_type, as_text, metadata)
- `plan`: planner output (`steps`, `rationale`)
- `parsed_holdings`: additive channel of dicts produced by parse fan-out
- `resolved_holdings`: additive channel of canonical holding dicts
- `unresolved_entities`: additive channel of unresolved records with `reason`
- `errors`: additive channel of error strings
- `analysis`: final narrative string (optional)

## Agents

- Ingestion (`agents/ingestion.py`)
  - Discovers files in `input_dir`, extracts plain text for common types (txt/md/html/csv/eml)
  - Applies a 2 MiB hard cap per file to protect memory
  - Ignores OS artifacts like `.DS_Store`

- Parser (`agents/parser.py`)
  - Prompts an LLM to emit strict JSON conforming to `ParsedHoldingsResult`
  - Retries on schema validation issues (`PARSER_MAX_RETRIES`, default 2)
  - Truncates input to `PARSER_MAX_DOC_CHARS` (default 20000)

- Resolver (`agents/resolver.py` + `tools/symbol_resolver.py`)
  - Optionally queries Polygon via `services/polygon_client.py`
  - Heuristic ranking: active + preferred MIC + currency match
  - Offline mode when no API key; returns unresolved records

- Analyst (`agents/analyst.py`)
  - Produces a simple narrative summary using the LLM

## Models

- `models/parsed.py`
  - `ParsedHolding`, `ParsedHoldingsResult` (schema-driven LLM output)

- `models/canonical.py`
  - `InstrumentKey` (string form `cid:asset_class:locale:mic:symbol`)
  - `CanonicalHolding` (normalized, enriched record)

## Configuration

Settings are defined in `src/portfolio_advisor/config.py` and loaded from environment (with `.env`
in dev) and may be overridden via CLI flags.

- Required: `--input-dir`, `--output-dir`
- LLM: `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `OPENAI_MODEL`, `REQUEST_TIMEOUT_S`, `MAX_TOKENS`, `TEMPERATURE`
- Logging: `LOG_LEVEL`, `LOG_FORMAT`, `VERBOSE`, `AGENT_PROGRESS`
- Parser: `PARSER_MAX_RETRIES`, `PARSER_MAX_DOC_CHARS`
- Polygon / Resolver: `POLYGON_API_KEY`, `POLYGON_BASE_URL`, `POLYGON_TIMEOUT_S`,
  `RESOLVER_DEFAULT_LOCALE`, `RESOLVER_PREFERRED_MICS`, `RESOLVER_CONFIDENCE_THRESHOLD`
- Cache: `SKIP_LLM_CACHE` (when set, bypasses read but writes results to cache)

Notes:
- Logging is configured from `Settings` and may be influenced by env variables for convenience.
- With no `OPENAI_API_KEY`, a stub LLM path returns placeholder text for tests/demos.

## Logging and Errors

- Logs default to plain/INFO; JSON format is supported.
- Library noise is reduced by default; enabling `AGENT_PROGRESS` raises verbosity for
  LangGraph/LangChain to INFO.
- Errors raise meaningful exceptions; file and configuration errors include context.

## Extensibility

- Adding a new provider: implement a thin client in `services/` and adapt `SymbolResolver` heuristics
- Adding a new agent: create a node function, add to the graph, and update state keys
- Adjusting prompts or retry logic: update `agents/parser.py` prompt constants and tuning

## Repository layout

- `src/portfolio_advisor/`: application modules
- `tests/`: unit and integration tests
- `docs/`: documentation (this file)
- `scripts/`: developer tooling (bootstrap, lint, format, test)
