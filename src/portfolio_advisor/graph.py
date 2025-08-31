from __future__ import annotations

import operator
from typing import Any, Annotated, TypedDict

from langgraph.graph import END, StateGraph
from langgraph.types import Send

from .agents.analyst import analyst_node
from .agents.ingestion import ingestion_node
from .agents.parser import parse_one_node
from .agents.planner import planner_node


class GraphState(TypedDict, total=False):
    settings: Any
    raw_docs: list[dict]
    plan: dict
    parsed_holdings: Annotated[list[dict], operator.add]
    errors: Annotated[list[str], operator.add]
    analysis: str


def _dispatch_parse_tasks(state: GraphState):
    docs = state.get("raw_docs", []) or []
    settings = state["settings"]
    return [Send("parse_one", {"settings": settings, "doc": d}) for d in docs]


def _join_after_parse(_state: GraphState) -> dict:
    # No-op node used as a deferred barrier to ensure downstream runs once.
    return {}


def build_graph() -> Any:
    graph = StateGraph(GraphState)
    graph.add_node("ingestion", ingestion_node)
    graph.add_node("planner", planner_node)
    def _noop(_state: GraphState) -> dict:
        return {}

    graph.add_node("dispatch_parse", _noop)
    graph.add_node("parse_one", parse_one_node)
    graph.add_node("join_after_parse", _join_after_parse, defer=True)
    graph.add_node("analyst", analyst_node, defer=True)

    graph.set_entry_point("ingestion")
    graph.add_edge("ingestion", "planner")
    graph.add_edge("planner", "dispatch_parse")
    graph.add_conditional_edges("dispatch_parse", _dispatch_parse_tasks)
    graph.add_edge("parse_one", "join_after_parse")
    graph.add_edge("join_after_parse", "analyst")
    graph.add_edge("analyst", END)

    return graph.compile()
