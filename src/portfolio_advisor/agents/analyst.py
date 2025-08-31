from __future__ import annotations

import logging

from ..llm import get_llm

logger = logging.getLogger(__name__)


# LLM prompt template is defined at module level for discoverability.
ANALYST_PROMPT_TEMPLATE = (
    "You are a portfolio analyst. "
    "Summarize candidate portfolio holdings extracted from input documents. "
    "Provide a concise, professional markdown summary covering: accounts, baskets, "
    "identifier coverage, and notable missing data. "
    "Do not make recommendations.\n\n"
    "Files: {files}\n"
    "Plan steps: {plan_steps}\n\n"
    "Candidate holdings (JSON lines):\n{candidates}"
)


def analyst_node(state: dict) -> dict:
    settings = state["settings"]
    llm = get_llm(settings)

    # Produce a brief summary based on parsed candidates and plan
    raw_docs = state.get("raw_docs", []) or []
    file_descriptions = [
        f"{d.get('name','')} ({d.get('mime_type','unknown')}, {d.get('source_bytes',0)} bytes)"
        for d in raw_docs
    ]
    plan = state.get("plan", {})
    candidates = state.get("parsed_holdings", []) or []
    # compact JSONL to avoid token bloat
    if candidates:
        try:
            import json

            cand_lines = "\n".join(json.dumps(c, separators=(",", ":")) for c in candidates[:200])
        except Exception:  # pragma: no cover - defensive
            cand_lines = "\n".join(str(c) for c in candidates[:200])
    else:
        cand_lines = "[none]"
    prompt = ANALYST_PROMPT_TEMPLATE.format(
        files=", ".join(file_descriptions) if file_descriptions else "none",
        plan_steps=", ".join(plan.get("steps", [])),
        candidates=cand_lines,
    )

    try:
        response = llm.invoke(prompt)
        content = getattr(response, "content", str(response))
    except Exception as exc:  # pragma: no cover - network/LLM errors vary
        logger.warning("LLM call failed, using fallback: %s", exc)
        content = "Placeholder analysis due to LLM error."

    return {**state, "analysis": content}
