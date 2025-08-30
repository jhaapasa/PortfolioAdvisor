from __future__ import annotations

import logging
from typing import TypedDict

logger = logging.getLogger(__name__)


class Plan(TypedDict):
    steps: list[str]
    rationale: str


def planner_node(state: dict) -> dict:
    files = state.get("files", [])
    file_names = [str(p) for p in files]
    logger.debug("Planner received %d files", len(file_names))
    plan: Plan = {
        "steps": [
            "Read portfolio inputs",
            "Compute simple summary statistics",
            "Draft analysis narrative",
        ],
        "rationale": "Minimal viable plan for portfolio analysis",
    }
    return {**state, "plan": plan}
