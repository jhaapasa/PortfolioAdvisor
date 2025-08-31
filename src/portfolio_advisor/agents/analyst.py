from __future__ import annotations

import logging

from ..llm import get_llm

logger = logging.getLogger(__name__)


# LLM prompt template is defined at module level for discoverability.
ANALYST_PROMPT_TEMPLATE = (
    "You are a portfolio analyst. Given the input files and a simple plan, produce a short "
    "placeholder analysis suitable for a markdown report."
    "\nFiles: {files}"
    "\nPlan steps: {plan_steps}"
)


def analyst_node(state: dict) -> dict:
    settings = state["settings"]
    llm = get_llm(settings)

    # In the MVP, produce a brief summary based on ingested docs and plan
    raw_docs = state.get("raw_docs", []) or []
    file_descriptions = [
        f"{d.get('name','')} ({d.get('mime_type','unknown')}, {d.get('source_bytes',0)} bytes)"
        for d in raw_docs
    ]
    plan = state.get("plan", {})
    prompt = ANALYST_PROMPT_TEMPLATE.format(
        files=", ".join(file_descriptions) if file_descriptions else "none",
        plan_steps=", ".join(plan.get("steps", [])),
    )

    try:
        response = llm.invoke(prompt)
        content = getattr(response, "content", str(response))
    except Exception as exc:  # pragma: no cover - network/LLM errors vary
        logger.warning("LLM call failed, using fallback: %s", exc)
        content = "Placeholder analysis due to LLM error."

    return {**state, "analysis": content}
