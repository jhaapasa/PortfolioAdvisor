"""Analyst agent: composes a concise narrative summary from graph results."""

from __future__ import annotations

import logging

from ..llm import get_llm

logger = logging.getLogger(__name__)


# LLM prompt template is defined at module level for discoverability.
ANALYST_PROMPT_TEMPLATE = (
    "You are a portfolio analyst. "
    "Summarize portfolio holdings extracted from input documents. "
    "Provide a concise, professional markdown summary covering: accounts, baskets, "
    "identifier coverage, resolver outcomes (resolved vs unresolved), and notable "
    "missing data. "
    "Include market comparison insights if available "
    "(portfolio risk metrics, performance vs benchmarks). "
    "Do not make recommendations.\n\n"
    "Files: {files}\n"
    "Plan steps: {plan_steps}\n"
    "Resolver: {resolver_summary}\n\n"
    "Basket Highlights:\n{basket_highlights}\n\n"
    "Market Comparison:\n{market_comparison}\n\n"
    "Holdings (JSON lines):\n{candidates}"
)


def analyst_node(state: dict) -> dict:
    settings = state["settings"]
    llm = get_llm(settings)
    logger.info("Analyst agent start")

    # Produce a brief summary based on parsed candidates and plan
    raw_docs = state.get("raw_docs", []) or []
    file_descriptions = [
        f"{d.get('name','')} ({d.get('mime_type','unknown')}, {d.get('source_bytes',0)} bytes)"
        for d in raw_docs
    ]
    plan = state.get("plan", {})
    # Prefer resolved holdings but fallback to parsed candidates
    resolved = state.get("resolved_holdings", []) or []
    parsed = state.get("parsed_holdings", []) or []
    unresolved = state.get("unresolved_entities", []) or []
    candidates = resolved or parsed
    # compact JSONL to avoid token bloat
    if candidates:
        try:
            import json

            cand_lines = "\n".join(json.dumps(c, separators=(",", ":")) for c in candidates[:200])
        except Exception:  # pragma: no cover - defensive
            cand_lines = "\n".join(str(c) for c in candidates[:200])
    else:
        cand_lines = "[none]"
    resolver_summary = (
        f"resolved={len(resolved)}, unresolved={len(unresolved)}"
        if (resolved or unresolved)
        else "no resolver data"
    )
    # Basket summaries (if present)
    basket_reports = state.get("basket_reports", []) or []
    highlights_lines: list[str] = []
    for rep in basket_reports[:10]:
        try:
            label = rep.get("label")
            slug = rep.get("slug")
            av = rep.get("averages") or {}
            d1 = av.get("d1")
            d5 = av.get("d5")
            d1s = f"{d1:+.2%}" if isinstance(d1, int | float) else "n/a"
            d5s = f"{d5:+.2%}" if isinstance(d5, int | float) else "n/a"
            text = rep.get("summary_text") or ""
            highlights_lines.append(f"{label} ({slug}) — avg d1 {d1s}, d5 {d5s}\n{text}")
        except Exception:  # pragma: no cover - defensive
            continue
    basket_highlights = "\n\n".join(highlights_lines) if highlights_lines else "[none]"

    # Extract market comparison data
    market_context = state.get("market_context")
    market_comparison_lines = []

    if market_context and market_context.portfolio_metrics:
        pm = market_context.portfolio_metrics
        market_comparison_lines.append("Portfolio Risk Metrics:")

        # Average betas
        if pm.average_beta_vs_benchmarks:
            beta_str = ", ".join(
                [f"{k}: {v:.2f}" for k, v in pm.average_beta_vs_benchmarks.items()]
            )
            market_comparison_lines.append(f"- Average Beta: {beta_str}")

        # Sharpe ratios
        market_comparison_lines.append(f"- Portfolio Sharpe (1yr): {pm.portfolio_sharpe:.2f}")
        market_comparison_lines.append(f"- Average Stock Sharpe: {pm.average_stock_sharpe:.2f}")

        # Stocks outperforming
        if pm.stocks_outperforming:
            for benchmark, count in pm.stocks_outperforming.items():
                pct = (count / pm.total_stocks * 100) if pm.total_stocks > 0 else 0
                market_comparison_lines.append(
                    f"- Stocks outperforming {benchmark}: {count}/{pm.total_stocks} ({pct:.0f}%)"
                )

        # Top contributors
        if pm.top_contributors:
            market_comparison_lines.append("\nTop Contributors (vs SPY):")
            for i, contrib in enumerate(pm.top_contributors[:3]):
                ticker = contrib["ticker"]
                excess = contrib.get("excess_sharpe", 0)
                market_comparison_lines.append(f"{i+1}. {ticker}: Excess Sharpe {excess:+.2f}")

    market_comparison = (
        "\n".join(market_comparison_lines)
        if market_comparison_lines
        else "[no market comparison data available]"
    )

    prompt = ANALYST_PROMPT_TEMPLATE.format(
        files=", ".join(file_descriptions) if file_descriptions else "none",
        plan_steps=", ".join(plan.get("steps", [])),
        resolver_summary=resolver_summary,
        basket_highlights=basket_highlights,
        market_comparison=market_comparison,
        candidates=cand_lines,
    )

    try:
        response = llm.invoke(prompt)
        content = getattr(response, "content", str(response))
    except Exception as exc:  # pragma: no cover - network/LLM errors vary
        logger.warning("LLM call failed, using fallback: %s", exc)
        content = "Placeholder analysis due to LLM error."

    logger.info("Analyst agent finished (%d chars)", len(content or ""))
    return {**state, "analysis": content}
