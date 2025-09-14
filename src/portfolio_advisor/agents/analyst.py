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
    "Do not make recommendations.\n\n"
    "Files: {files}\n"
    "Plan steps: {plan_steps}\n"
    "Resolver: {resolver_summary}\n\n"
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
    prompt = ANALYST_PROMPT_TEMPLATE.format(
        files=", ".join(file_descriptions) if file_descriptions else "none",
        plan_steps=", ".join(plan.get("steps", [])),
        resolver_summary=resolver_summary,
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
