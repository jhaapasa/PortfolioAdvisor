"""NewsSummaryAgent: distill 7-day Polygon news summaries into high-signal output.

Constraints:
- Use only Polygon-provided JSON fields (headline/title, summary if present, sentiment if present,
  publisher, published_utc, tickers). Do not parse HTML.
- Focus on notable events and sentiment extremes or changes.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

from ..llm import get_llm

logger = logging.getLogger(__name__)


NEWS_SUMMARY_SYSTEM = (
    "You analyze 7 days of Polygon-provided stock news summaries and sentiments. "
    "Use ONLY the provided JSON fields (title/headline, summary, sentiment label/score, "
    "published time, publisher). Do NOT infer from HTML. "
    "Surface notable events and sentiment extremes or shifts. "
    "Skip routine or low-signal items. "
    "Output both (1) concise Markdown and (2) a structured JSON payload per schema. "
    "Be precise and conservative; do not hallucinate facts."
)


NEWS_SUMMARY_USER_TEMPLATE = (
    "Ticker: {ticker} | Slug: {slug} | WindowDays: 7\n\n"
    "News JSON (compact list):\n{news_compact}\n\n"
    "Return a JSON object with keys: 'highlights_markdown', 'sentiment_overview', "
    "'notable_events', and include 'ticker', 'slug', 'window_days'."
)


@dataclass
class NewsSummaryInput:
    ticker: str
    slug: str
    news: list[dict[str, Any]]


def _compact_json_list(items: list[dict[str, Any]], limit: int = 100) -> str:
    """Return a compact JSON lines string for the first N items."""
    lines: list[str] = []
    for it in items[:limit]:
        try:
            lines.append(json.dumps(it, separators=(",", ":")))
        except Exception:
            lines.append(str(it))
    return "\n".join(lines) if lines else "[]"


def summarize_news_node(state: dict) -> dict:
    """LLM node to summarize 7-day news for a stock into notable events and sentiment overview.

    Expected state:
      - settings: Settings
      - instrument: { instrument_id, primary_ticker }
      - _slug: slug
      - news_items (optional): list of Polygon article dicts for last 7 days
    Returns state with 'news_summary': { markdown, json } when news is available.
    """
    settings = state["settings"]
    ticker = (state.get("instrument") or {}).get("primary_ticker") or ""
    slug = state.get("_slug") or ""
    news_items: list[dict[str, Any]] = state.get("news_items") or []
    logger.info(
        "news_summary: ticker=%s slug=%s news_items_count=%d", ticker, slug, len(news_items)
    )

    # If not provided by the graph, try to read recent 7d items from disk
    if (not news_items) and slug and state.get("_paths") is not None:
        try:
            paths = state.get("_paths")
            cache_path = paths.news_dir(slug) / "recent_7d.json"
            logger.info("news_summary: attempting to load from %s", cache_path)
            if cache_path.exists():
                with cache_path.open("r", encoding="utf-8") as fh:
                    doc = json.load(fh) or {}
                news_items = list(doc.get("items") or [])
                logger.info("news_summary: loaded %d items from disk", len(news_items))
            else:
                logger.warning("news_summary: cache file does not exist: %s", cache_path)
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("news_summary: failed to load from disk: %s", exc, exc_info=True)
            news_items = []

    if not ticker or not slug or not news_items:
        # Nothing to do
        logger.warning(
            "news_summary: skipping (ticker=%s slug=%s news_items=%d)",
            bool(ticker),
            bool(slug),
            len(news_items),
        )
        return state

    # Prepare prompt
    compact = _compact_json_list(news_items)
    prompt = NEWS_SUMMARY_USER_TEMPLATE.format(ticker=ticker, slug=slug, news_compact=compact)

    try:
        llm = get_llm(settings)
        resp = llm.invoke(NEWS_SUMMARY_SYSTEM + "\n\n" + prompt)
        content = getattr(resp, "content", str(resp))
    except Exception as exc:  # pragma: no cover - provider/network specific
        logger.warning("news_summary: LLM failed, using fallback: %s", exc)
        content = json.dumps(
            {
                "ticker": ticker,
                "slug": slug,
                "window_days": 7,
                "sentiment_overview": {
                    "overall_label": "neutral",
                    "avg_score": 0.0,
                    "trend": "flat",
                    "counts": {"positive": 0, "neutral": 0, "negative": 0},
                },
                "notable_events": [],
                "highlights_markdown": "## Notable News & Events\n- [none]",
                "notes": [
                    "Fallback due to LLM error; based solely on Polygon summaries/sentiment."
                ],
            }
        )

    # Try to parse LLM output as JSON; if not JSON, wrap as markdown-only
    summary_json: dict[str, Any] | None
    summary_md: str | None
    try:
        doc = json.loads(content)
        summary_json = doc
        summary_md = doc.get("highlights_markdown") or ""
        logger.info("news_summary: parsed JSON output, markdown_length=%d", len(summary_md))
    except Exception:
        summary_json = None
        summary_md = content
        logger.info("news_summary: using raw markdown output, length=%d", len(summary_md))

    news_summary_data = {
        "markdown": summary_md,
        "json": summary_json,
    }
    logger.info(
        "news_summary: completed, returning news_summary to state (md_len=%d json=%s)",
        len(summary_md),
        summary_json is not None,
    )
    return {"news_summary": news_summary_data}
