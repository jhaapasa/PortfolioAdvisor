"""LangGraph application graph definition.

The graph uses explicit fan-out via conditional edges for parsing and resolving.
We insert lightweight "barrier" nodes marked with defer=True so downstream stages
run exactly once after their respective fan-out stages complete.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, StateGraph
from langgraph.types import Send

from .agents.analyst import analyst_node
from .agents.ingestion import ingestion_node
from .agents.parser import parse_one_node
from .agents.planner import planner_node
from .agents.resolver import resolve_one_node


class GraphState(TypedDict, total=False):
    settings: Any
    raw_docs: list[dict]
    plan: dict
    parsed_holdings: Annotated[list[dict], operator.add]
    resolved_holdings: Annotated[list[dict], operator.add]
    unresolved_entities: Annotated[list[dict], operator.add]
    errors: Annotated[list[str], operator.add]
    analysis: str


def _dispatch_parse_tasks(state: GraphState):
    """Fan-out: produce one parse task per ingested document."""
    docs = state.get("raw_docs", []) or []
    settings = state["settings"]
    return [Send("parse_one", {"settings": settings, "doc": d}) for d in docs]


def _join_after_parse(_state: GraphState) -> dict:
    """Barrier: no-op node used with defer=True so downstream runs once."""
    return {}


def _dispatch_resolve_tasks(state: GraphState):
    """Fan-out: produce one resolve task per parsed holding."""
    settings = state["settings"]
    holdings = state.get("parsed_holdings", []) or []
    return [Send("resolve_one", {"settings": settings, "holding": h}) for h in holdings]


def build_graph() -> Any:
    """Construct and compile the PortfolioAdvisor LangGraph.

    The graph structure favors clarity: explicit dispatch nodes for fan-out and
    explicit join barriers using defer=True to ensure single execution of
    subsequent stages.
    """
    graph = StateGraph(GraphState)
    graph.add_node("ingestion", ingestion_node)
    graph.add_node("planner", planner_node)

    def _noop(_state: GraphState) -> dict:
        return {}

    graph.add_node("dispatch_parse", _noop)
    graph.add_node("parse_one", parse_one_node)
    graph.add_node("join_after_parse", _join_after_parse, defer=True)
    graph.add_node("dispatch_resolve", _noop)
    graph.add_node("resolve_one", resolve_one_node)
    graph.add_node("join_after_resolve", _join_after_parse, defer=True)
    graph.add_node("analyst", analyst_node, defer=True)

    graph.set_entry_point("ingestion")
    graph.add_edge("ingestion", "planner")
    graph.add_edge("planner", "dispatch_parse")
    graph.add_conditional_edges("dispatch_parse", _dispatch_parse_tasks)
    graph.add_edge("parse_one", "join_after_parse")
    graph.add_edge("join_after_parse", "dispatch_resolve")
    graph.add_conditional_edges("dispatch_resolve", _dispatch_resolve_tasks)
    graph.add_edge("resolve_one", "join_after_resolve")
    graph.add_edge("join_after_resolve", "analyst")
    graph.add_edge("analyst", END)

    return graph.compile()
