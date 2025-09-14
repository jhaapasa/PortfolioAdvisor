from __future__ import annotations

import logging
from typing import TypedDict

logger = logging.getLogger(__name__)


class Plan(TypedDict):
    steps: list[str]
    rationale: str


def planner_node(state: dict) -> dict:
    # Use ingested documents from state for awareness/logging
    raw_docs = state.get("raw_docs", []) or []
    file_names = [str(d.get("name", "")) for d in raw_docs]
    logger.info("Planner agent start: %d documents", len(file_names))
    logger.debug("Planner received %d documents: %s", len(file_names), ", ".join(file_names))
    plan: Plan = {
        "steps": [
            "Read portfolio inputs",
            "Compute simple summary statistics",
            "Draft analysis narrative",
        ],
        "rationale": "Minimal viable plan for portfolio analysis",
    }
    logger.info("Planner agent finished: %d steps", len(plan["steps"]))
    return {**state, "plan": plan}
