from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from ..llm import get_llm

logger = logging.getLogger(__name__)

BASKET_NARRATIVE_SYSTEM = """You are a portfolio analyst synthesizing basket-level \
insights from individual stock reports.

Your task:
1. Identify common themes across the basket (earnings, sector trends, macro factors)
2. Correlate top movers with their specific news catalysts
3. Highlight notable divergences or concentrations of risk
4. Keep the narrative concise (3-6 bullets) and focused on the trailing week

Output format:
- Use markdown with ## headings
- Be factual and specific; cite tickers when relevant
- Focus on actionable insights, not just data repetition
- Avoid recommendations or predictions

Do not:
- Repeat information already visible in the performance table
- Include generic market commentary without basket-specific relevance
- Make buy/sell recommendations
"""


def _build_narrative_prompt(
    basket_label: str,
    basket_slug: str,
    metrics: dict[str, Any],
    ticker_summaries: list[dict[str, Any]],
) -> str:
    """Build the user prompt for basket narrative generation."""
    parts = [f"Basket: {basket_label}"]

    # Performance overview
    avgs = metrics.get("averages", {})
    d1_avg = avgs.get("d1")
    d5_avg = avgs.get("d5")
    if d1_avg is not None or d5_avg is not None:
        parts.append("\n## Performance Summary")
        if d1_avg is not None:
            parts.append(f"- Average 1-day: {d1_avg:+.2%}")
        if d5_avg is not None:
            parts.append(f"- Average 5-day: {d5_avg:+.2%}")

    # Top movers
    movers = metrics.get("top_movers", {})
    instruments = {i["instrument_id"]: i for i in metrics.get("instruments", [])}

    def _format_movers(key: str, label: str) -> str:
        ids = movers.get(key, [])
        lines = [f"\n{label}:"]
        for iid in ids[:3]:
            inst = instruments.get(iid, {})
            ticker = inst.get("primary_ticker", iid)
            ret_key = "d1" if "d1" in key else "d5"
            ret_val = inst.get(ret_key)
            if ret_val is not None:
                lines.append(f"  - {ticker}: {ret_val:+.2%}")
        return "\n".join(lines) if len(lines) > 1 else ""

    parts.append("\n## Top Movers (5D)")
    parts.append(_format_movers("d5_up", "Up"))
    parts.append(_format_movers("d5_down", "Down"))

    # Ticker summaries (news context)
    if ticker_summaries:
        parts.append("\n## Individual Ticker News Context (Last 7 Days)")
        for ts in ticker_summaries:
            ticker = ts.get("ticker", "")
            summary = ts.get("summary_text", "").strip()
            if summary and ticker:
                # Truncate very long summaries
                if len(summary) > 500:
                    summary = summary[:497] + "..."
                parts.append(f"\n### {ticker}")
                parts.append(summary)

    parts.append(
        "\n\nSynthesize basket-level themes and correlate top movers with their catalysts. "
        "Output 3-6 markdown bullets under appropriate headings (e.g., '## Basket Themes', "
        "'## Notable Events', '## Risk Considerations')."
    )

    return "\n".join(parts)


def generate_basket_narrative_node(state: dict) -> dict:
    """LLM node to generate narrative basket report from ticker news and performance data.

    Expected state:
      - settings: Settings
      - basket: { id, label, slug }
      - metrics: basket metrics dict (with averages, top_movers, instruments)
      - _ticker_news_summaries (optional): list of { ticker, slug, summary_text, report_path }

    Returns state with 'narrative_md': markdown narrative report.
    """
    settings = state["settings"]
    basket = state["basket"]
    metrics = state.get("metrics", {})
    ticker_summaries = state.get("_ticker_news_summaries", [])

    basket_label = basket.get("label", "")
    basket_slug = basket.get("slug", "")

    logger.info(
        "basket_narrative: basket=%s ticker_summaries=%d", basket_slug, len(ticker_summaries)
    )

    if not metrics:
        logger.warning("basket_narrative: no metrics available, skipping")
        return state

    # If no news summaries, generate simpler output
    if not ticker_summaries:
        logger.info("basket_narrative: no ticker news summaries, generating basic report")
        avgs = metrics.get("averages", {})
        d1 = avgs.get("d1")
        d5 = avgs.get("d5")
        basic_md = f"## {basket_label}\n\n"
        if d1 is not None:
            basic_md += f"- Average 1-day return: {d1:+.2%}\n"
        if d5 is not None:
            basic_md += f"- Average 5-day return: {d5:+.2%}\n"
        basic_md += "\n*Individual ticker news summaries not available.*"
        return {**state, "narrative_md": basic_md}

    # Build prompt and invoke LLM
    prompt = _build_narrative_prompt(basket_label, basket_slug, metrics, ticker_summaries)

    try:
        llm = get_llm(settings)
        resp = llm.invoke(BASKET_NARRATIVE_SYSTEM + "\n\n" + prompt)
        content = getattr(resp, "content", str(resp))
        logger.info("basket_narrative: LLM generated %d chars", len(content))
    except Exception as exc:  # pragma: no cover - provider/network specific
        logger.warning("basket_narrative: LLM failed, using fallback: %s", exc)
        content = f"## {basket_label}\n\n*LLM narrative generation failed.*\n\n{prompt}"

    return {**state, "narrative_md": content}


def _read_ticker_report_summary(slug: str, stock_output_dir: Path) -> dict[str, Any] | None:
    """Read 7-day report for a ticker if available."""
    report_path = stock_output_dir / "tickers" / slug / "report" / "7d" / "report.md"
    if not report_path.exists():
        return None

    try:
        summary_text = report_path.read_text(encoding="utf-8")
        # Extract just the TL;DR and notable events for brevity
        lines = summary_text.split("\n")
        tldr_section = []
        events_section = []
        in_tldr = False
        in_events = False

        for line in lines:
            if line.strip().startswith("TL;DR"):
                in_tldr = True
                in_events = False
                continue
            elif "Notable News" in line or "Notable Events" in line:
                in_tldr = False
                in_events = True
                continue
            elif line.startswith("##") or line.startswith("Sentiment Overview"):
                in_tldr = False
                in_events = False
                continue

            if in_tldr and line.strip():
                tldr_section.append(line)
            elif in_events and line.strip().startswith("-"):
                events_section.append(line)

        # Combine TL;DR and up to 3 notable events
        combined = []
        if tldr_section:
            combined.append("**TL;DR**: " + " ".join(tldr_section).strip())
        if events_section:
            combined.append("\n**Notable Events**:")
            combined.extend(events_section[:3])

        return {
            "slug": slug,
            "summary_text": "\n".join(combined) if combined else summary_text[:300],
            "report_path": str(report_path),
        }
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Failed to read ticker report %s: %s", report_path, exc)
        return None


def collect_ticker_news_summaries_node(state: dict) -> dict:
    """Collect 7-day news summaries from individual ticker reports.

    Expected state:
      - settings: Settings
      - basket: { id, label, slug }
      - metrics: basket metrics with instruments list

    Returns state with '_ticker_news_summaries': list of { ticker, slug, summary_text, report_path }
    """
    settings = state["settings"]
    metrics = state.get("metrics", {})
    instruments = metrics.get("instruments", [])

    stock_output_dir = Path(settings.output_dir) / "stocks"
    summaries = []

    for inst in instruments:
        ticker = inst.get("primary_ticker", "")
        iid = inst.get("instrument_id", "")

        # Derive slug from instrument_id
        from ..utils.slug import instrument_id_to_slug

        slug = instrument_id_to_slug(iid) if iid else ticker

        summary = _read_ticker_report_summary(slug, stock_output_dir)
        if summary:
            summary["ticker"] = ticker
            summaries.append(summary)

    logger.info("collect_ticker_news: collected %d summaries", len(summaries))
    return {**state, "_ticker_news_summaries": summaries}
