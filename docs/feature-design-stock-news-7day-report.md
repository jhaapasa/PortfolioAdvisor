# Feature Design: Per-Stock 7‑Day News Sentiment and Notable Events Report

Status: Proposed

Last Updated: 2025-10-09

Related:
- `docs/feature-design-stock-news.md`
- `docs/feature-design-article-text-extraction.md`

## Summary
Produce a concise per-stock trailing 7‑day update that blends:
- A news-driven sentiment summary and notable events (from Polygon summaries and sentiment only)
- Technical context: trailing 7‑day returns and volatility histogram highlights

The goal is to quickly alert the user to anything notable while glossing over unexceptional news.
This feature adds two LLM agent nodes: one to generate a focused news update, another to collate the
news with existing technical analysis into a final per-stock Markdown report.

## Requirements
- Use Polygon news JSON fields only (e.g., headline, summary, sentiment, tickers, published time);
  do not rely on downloaded or parsed HTML content yet.
- Highlight notable events or dramatic sentiment/sentiment changes; avoid summarizing routine items.
- Focus strictly on the trailing 7 calendar days.
- Implement two separate LLM nodes:
  1) NewsSummaryAgent: distill Polygon news into a short, high-signal update per stock.
  2) ReportCollatorAgent: merge the news update with returns + volatility histogram highlights.
- Do not broaden technical analysis beyond what already exists; reuse current capabilities.
- Output a concise Markdown report suitable for the per-stock analysis view, plus a small structured
  JSON metrics artifact to support testing and downstream usage.

Non-goals (initial cut):
- No HTML/article-body extraction, parsing, or external web crawling.
- No multi-week/month horizons or long-range trend analysis.
- No new quantitative signals beyond what already exists for returns/volatility.

## User Experience (report shape)
Target Markdown structure for each stock (example skeleton):

```
# {TICKER} — 7‑Day Update

TL;DR: {one-line synthesis blending sentiment + performance}

## Notable News & Events
- {event 1: date — what happened — why it matters}
- {event 2}
- {only include items above a notability threshold}

## Sentiment Overview (7d)
- Overall: {positive|neutral|negative}, trend: {rising|flat|falling}
- Counts: {pos}/{neu}/{neg}; strongest day: {date} ({brief})

## Performance Context (7d)
- Return: {pct}
- Realized volatility: {brief}
- Histogram: {brief description; link to artifact if available}

Notes: Based solely on Polygon summaries/sentiment; no article body parsing.
```

## Inputs & Data Constraints
- Polygon News (from `src/portfolio_advisor/stocks/news.py`):
  - Use only JSON fields available from Polygon (e.g., headline, summary, sentiment label/score,
    publisher, published timestamp). Treat summaries as authoritative; do not inspect HTML.
  - Time window filter: now − 7 days to now (inclusive of market/non-market days).
- Technical Metrics (reuse existing modules):
  - Trailing 7‑day return for the stock.
  - Volatility histogram or a textual summary built from existing artifacts in `stocks.analysis` /
    `stocks.plotting`.

## Architecture & Flow
We add two LLM nodes and a thin orchestration layer in the stocks graph.

1) FetchRecentNews (existing):
   - Use existing utilities in `src/portfolio_advisor/stocks/news.py` to fetch recent items for a
     ticker within the 7‑day window. The output is a list of dicts with headline, summary,
     sentiment, published time, and any tickers metadata.

2) NewsSummaryAgent (new LLM node):
   - Input: ticker, 7‑day news list (headline, summary, sentiment, published time).
   - Behavior: identify notable events and dramatic sentiment or sentiment changes. Downweight or
     omit routine/low-signal items. Output structured JSON plus a compact Markdown snippet.
   - Output JSON (schema below) supports testing and later reuse.

3) ReportCollatorAgent (new LLM node):
   - Input: ticker, NewsSummaryAgent JSON + Markdown, and existing technical metrics for 7 days
     (return, volatility histogram/summary).
   - Behavior: produce the final per-stock Markdown report following the UX structure; do not
     invent metrics.

4) Orchestration (graph + CLI):
   - Integrate the two nodes in `src/portfolio_advisor/graphs/stocks.py`.
   - Expose via CLI (e.g., `pa stocks report --ticker AAPL --window 7 --include-news`). Defaults
     can enable the news section once stable.
   - Derive `slug` via `instrument_id_to_slug(...)` and construct paths with `StockPaths(root=Path(settings.output_dir)/"stocks")`.
     Persist all per-stock artifacts under `stocks/tickers/{slug}/...` using `StockPaths` helpers.

## Notability Heuristics (LLM-guided, data-aware)
- Prefer items with explicit corporate actions, guidance changes, regulatory actions, legal matters,
  executive changes, large customer/partner announcements, rating/price-target changes, earnings,
  or material operational events.
- Consider sentiment magnitude and change over the 7‑day window; flag extremes or abrupt flips.
- Collapse duplicates/near-duplicates from multiple outlets into one bullet with a stronger signal.
- Skip headlines with neutral sentiment and generic updates unless they tie to a notable theme.

## Interfaces (proposed)
Structured output from NewsSummaryAgent (JSON):

```json
{
  "ticker": "AAPL",
  "slug": "cid-stocks-us-xnas-aapl",
  "window_days": 7,
  "sentiment_overview": {
    "overall_label": "positive|neutral|negative",
    "avg_score": 0.0,
    "trend": "rising|flat|falling",
    "counts": {"positive": 0, "neutral": 0, "negative": 0},
    "strongest_day": "2025-10-05"
  },
  "notable_events": [
    {
      "date": "2025-10-04",
      "title": "Rating upgrade by XYZ",
      "why_notable": "Impacts investor sentiment and near-term price action.",
      "sentiment": "positive"
    }
  ],
  "highlights_markdown": "## Notable News & Events\n- ...",
  "notes": ["Based solely on Polygon summaries/sentiment"]
}
```

ReportCollatorAgent final output:
- Primary: Markdown string following the UX skeleton
- Secondary: metrics artifact (JSON) capturing: 7‑day return, realized volatility summary,
  sentiment counts, overall label, number of notable events, and file paths to any artifacts
  (e.g., histogram image) if produced by existing modules.

## Prompt Design (sketch)
NewsSummaryAgent system prompt:

```
You analyze 7 days of Polygon-provided stock news summaries and sentiments. Use ONLY the provided
JSON fields (headline, summary, sentiment, published time, publisher). Do NOT infer from HTML.
Your job is to surface notable events and sentiment extremes or shifts. Skip routine items.
Output both (1) a concise Markdown section and (2) a structured JSON payload per the schema.
Be precise and conservative; do not hallucinate facts.
```

ReportCollatorAgent system prompt:

```
You produce a concise per-stock 7‑day update. Inputs: a news summary (Markdown + JSON) and
technical metrics (7‑day return and volatility histogram/summary). Output a cohesive Markdown
report following the specified sections. If a metric is missing, omit it without inventing values.
Do not change numeric precision beyond simple rounding for readability.
```

## Storage & Outputs
- Per-stock outputs under `output/stocks/tickers/{slug}/report/7d/`:
  - `report.md`: final Markdown report
  - `metrics.json`: structured metrics for tests and downstream tools
- Use `StockPaths.report_dir(slug) / "7d"` as the base directory for these artifacts.
- Reuse existing LLM cache mechanisms to avoid repeated generations for unchanged inputs.

## Testing Strategy
- Unit tests for NewsSummaryAgent prompt/assembly logic using deterministic fixtures of Polygon
  news JSON. Assert schema correctness, notability filtering, and conservative behavior.
- Unit tests for ReportCollatorAgent: given fixed news JSON and technical metrics, assert Markdown
  includes required sections and omits missing data gracefully.
- Integration test: end-to-end pipeline for a dummy ticker with stubbed news and mocked LLM
  responses; verify output files are written and parsable.
- CLI test: run `stocks report` with `--include-news` on fixtures; verify artifacts.

## Risks & Limitations
- Polygon sentiment may be coarse or inconsistent; we avoid over-interpretation.
- Sparse news within 7 days → report may be minimal by design (this is acceptable).
- No article-body parsing yet; nuanced context may be missed until HTML extraction stabilizes.

## Implementation Plan (Incremental)
1) Orchestration plumbing in `graphs/stocks.py`: add two nodes and data passing between them.
2) Implement NewsSummaryAgent using existing `llm.py`/service integration; add prompt templates.
3) Implement ReportCollatorAgent; define minimal metrics JSON alongside Markdown output.
4) Wire CLI entry in `src/portfolio_advisor/cli.py` (e.g., `stocks report --include-news --window 7`).
5) Write unit tests for both agents and integration/CLI tests using fixtures and mocks.
6) Default enablement decision: start behind `--include-news`; consider default-on after bake-in.

## Acceptance Criteria
- Given 7 days of Polygon news (summaries + sentiment) for a ticker, the system produces:
  - A Markdown report with the sections outlined under UX
  - A metrics JSON with sentiment counts, overall label, number of notable events, and technical
    metrics included when available
- Routine/low-signal items are omitted; notable events are surfaced with a clear rationale.
- Tests pass deterministically with mocked LLMs.


