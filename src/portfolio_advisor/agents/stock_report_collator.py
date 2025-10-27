"""ReportCollatorAgent: combine news summary + technical metrics into 7d per-stock update.

Inputs (from state/disk):
- news_summary: { markdown: str, json: dict | None }
- analysis returns/volatility JSONs via `StockPaths` for the slug

Outputs:
- report.md at `paths.report_dir(slug)/7d/report.md`
- metrics.json at `paths.report_dir(slug)/7d/metrics.json`
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from ..llm import get_llm

logger = logging.getLogger(__name__)


REPORT_SYSTEM = (
    "You produce a concise per-stock trailing 7-day update. "
    "Inputs: (a) news summary markdown and optional JSON metrics; (b) technical metrics including "
    "recent returns windows and optional volatility histogram summary. "
    "Output a cohesive Markdown report with sections: TL;DR, Notable News & Events, "
    "Sentiment Overview (7d), Performance Context (7d). "
    "If a metric is missing, omit it without inventing values. Keep it concise and high-signal."
)


def _read_json_if_exists(path: Path) -> dict[str, Any] | None:
    try:
        if path.exists():
            with path.open("r", encoding="utf-8") as fh:
                return json.load(fh)
    except Exception:  # pragma: no cover - defensive
        logger.warning("collator: failed reading %s", path, exc_info=True)
    return None


def _build_user_prompt(
    ticker: str,
    slug: str,
    news_md: str,
    news_json: dict[str, Any] | None,
    returns_doc: dict[str, Any] | None,
    vol_hist_doc: dict[str, Any] | None,
) -> str:
    parts: list[str] = []
    parts.append(f"Ticker: {ticker}\nSlug: {slug}")
    if returns_doc:
        try:
            as_of = returns_doc.get("as_of")
            wins = returns_doc.get("windows") or {}
            parts.append(
                f"Returns as_of: {as_of} windows: {json.dumps(wins, separators=(',', ':'))}"
            )
        except Exception:
            pass
    if vol_hist_doc:
        try:
            spectrum = (vol_hist_doc.get("variance_spectrum") or {}).get("per_level")
            summary = "present" if spectrum else "missing"
            parts.append(f"Volatility histogram: {summary}")
        except Exception:
            pass
    if news_json:
        try:
            nj = json.dumps(news_json, separators=(",", ":"))
            parts.append(f"News JSON: {nj}")
        except Exception:
            pass
    parts.append("News Highlights Markdown:\n" + (news_md or ""))
    parts.append("Return ONLY the final Markdown report; do not include extra commentary.")
    return "\n\n".join(parts)


def collate_report_node(state: dict) -> dict:
    settings = state["settings"]
    instrument = state.get("instrument") or {}
    ticker = instrument.get("primary_ticker") or ""
    slug = state.get("_slug") or ""
    paths = state.get("_paths")
    if not paths or not slug:
        return state

    include = bool(getattr(settings, "include_news_report", False))
    if not include:
        return state

    news = state.get("news_summary") or {}
    news_md = news.get("markdown") or ""
    news_json = news.get("json")
    if not news_md and not news_json:
        # Nothing to collate
        return state

    # Load existing analysis artifacts
    returns_doc = _read_json_if_exists(paths.analysis_returns_json(slug))
    vol_hist_doc = _read_json_if_exists(paths.analysis_wavelet_hist_json(slug))

    prompt = _build_user_prompt(ticker, slug, news_md, news_json, returns_doc, vol_hist_doc)
    try:
        llm = get_llm(settings)
        resp = llm.invoke(REPORT_SYSTEM + "\n\n" + prompt)
        content = getattr(resp, "content", str(resp))
    except Exception as exc:  # pragma: no cover - provider/network specific
        logger.warning("collator: LLM failed, using fallback: %s", exc)
        content = (
            f"# {ticker or slug} — 7‑Day Update\n\n"
            "TL;DR: [placeholder due to LLM error]\n\n" + (news_md or "")
        )

    # Write outputs
    out_dir = paths.report_dir(slug) / "7d"
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "report.md"
    metrics_path = out_dir / "metrics.json"

    try:
        report_path.write_text(content, encoding="utf-8")
    except Exception:  # pragma: no cover - defensive
        logger.warning("collator: failed writing report.md", exc_info=True)

    metrics: dict[str, Any] = {
        "ticker": ticker,
        "slug": slug,
        "has_news": bool(news_md or news_json),
    }
    try:
        # Extract a few fields for tests/downstream usage, if available
        if news_json:
            so = news_json.get("sentiment_overview") or {}
            counts = so.get("counts") or {}
            metrics.update(
                {
                    "sentiment_overall": so.get("overall_label"),
                    "sentiment_counts": counts,
                    "notable_events": len(news_json.get("notable_events") or []),
                }
            )
        if returns_doc:
            metrics["returns_as_of"] = returns_doc.get("as_of")
            metrics["returns_windows"] = returns_doc.get("windows")
        if vol_hist_doc:
            metrics["volatility_histogram_present"] = bool(
                (vol_hist_doc.get("variance_spectrum") or {}).get("per_level")
            )
    except Exception:  # pragma: no cover - defensive
        pass

    try:
        metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    except Exception:  # pragma: no cover - defensive
        logger.warning("collator: failed writing metrics.json", exc_info=True)

    out = dict(state)
    out.setdefault("artifacts", {})["news_report_7d"] = {
        "report_path": str(report_path),
        "metrics_path": str(metrics_path),
    }
    return out
