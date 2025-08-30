from __future__ import annotations

from typing import Any

from langgraph.graph import END, StateGraph

from .agents.analyst import analyst_node
from .agents.planner import planner_node


def build_graph() -> Any:
    graph = StateGraph(dict)
    graph.add_node("planner", planner_node)
    graph.add_node("analyst", analyst_node)

    graph.set_entry_point("planner")
    graph.add_edge("planner", "analyst")
    graph.add_edge("analyst", END)

    return graph.compile()
