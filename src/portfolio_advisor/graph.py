"""LangGraph application graph definition.

The graph uses explicit fan-out via conditional edges for parsing and resolving.
We insert lightweight "barrier" nodes marked with defer=True so downstream stages
run exactly once after their respective fan-out stages complete.
"""

from __future__ import annotations

import operator
from pathlib import Path
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, StateGraph
from langgraph.types import Send

from .agents.analyst import analyst_node
from .agents.ingestion import ingestion_node
from .agents.parser import parse_one_node
from .agents.planner import planner_node
from .agents.resolver import resolve_one_node
from .graphs.baskets import build_basket_graph
from .graphs.stocks import update_all_for_instruments
from .portfolio.persistence import (
    PortfolioPaths,
    append_history_diffs,
    write_baskets_views,
    write_current_holdings,
    write_portfolio_header,
)


class GraphState(TypedDict, total=False):
    settings: Any
    raw_docs: list[dict]
    plan: dict
    parsed_holdings: Annotated[list[dict], operator.add]
    resolved_holdings: Annotated[list[dict], operator.add]
    unresolved_entities: Annotated[list[dict], operator.add]
    errors: Annotated[list[str], operator.add]
    analysis: str
    portfolio_persisted: bool
    basket_reports: Annotated[list[dict], operator.add]


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

    def _commit_portfolio_node(state: GraphState) -> dict:
        settings = state["settings"]
        holdings = state.get("resolved_holdings", []) or []
        if not holdings:
            return {"portfolio_persisted": False}
        # Read previous snapshot if exists

        p = PortfolioPaths(root=Path(getattr(settings, "portfolio_dir")))
        prev = []
        try:
            import json

            if p.holdings_json().exists():
                with p.holdings_json().open("r", encoding="utf-8") as fh:
                    prev = json.load(fh)
        except Exception:
            prev = []
        # Write current state
        write_current_holdings(getattr(settings, "portfolio_dir"), holdings)
        write_portfolio_header(getattr(settings, "portfolio_dir"), holdings)
        write_baskets_views(getattr(settings, "portfolio_dir"), holdings)
        # Derive instruments and baskets for downstream
        instruments = []
        seen_iids: set[str] = set()
        for h in holdings:
            iid = str(h.get("instrument_id") or "")
            if not iid or iid in seen_iids:
                continue
            seen_iids.add(iid)
            instruments.append(
                {
                    "instrument_id": iid,
                    "primary_ticker": h.get("primary_ticker"),
                }
            )
        # Load baskets index
        import json as _json

        baskets_index_path = p.baskets_index_json()
        baskets: list[dict] = []
        if baskets_index_path.exists():
            try:
                with baskets_index_path.open("r", encoding="utf-8") as fh:
                    idx = _json.load(fh)
                for b in idx:
                    label = b.get("label")
                    slug = b.get("slug")

                    # instruments belonging to this basket
                    def _slugify_local(text: str) -> str:
                        s = (text or "").lower()
                        out = []
                        prev_dash = False
                        for ch in s:
                            if ch.isalnum():
                                out.append(ch)
                                prev_dash = False
                            else:
                                if not prev_dash:
                                    out.append("-")
                                prev_dash = True
                        joined = "".join(out).strip("-")
                        while "--" in joined:
                            joined = joined.replace("--", "-")
                        return joined or "none"

                    binstruments = []
                    seen: set[str] = set()
                    for h in holdings:
                        if _slugify_local(str(h.get("basket") or "")) != slug:
                            continue
                        iid = str(h.get("instrument_id") or "")
                        if not iid or iid in seen:
                            continue
                        seen.add(iid)
                        binstruments.append(
                            {
                                "instrument_id": iid,
                                "primary_ticker": h.get("primary_ticker"),
                            }
                        )
                    baskets.append(
                        {
                            "id": b.get("id"),
                            "label": label,
                            "slug": slug,
                            "instruments": binstruments,
                        }
                    )
            except Exception:
                baskets = []

        # Append history diffs
        dates = [str(h.get("as_of")) for h in holdings if h.get("as_of")]
        as_of = max(dates) if dates else None
        append_history_diffs(getattr(settings, "portfolio_dir"), prev, holdings, as_of)

        # Kick off stocks updates (sequential for now)
        try:
            update_all_for_instruments(settings, instruments)
        except Exception:
            # Non-fatal; continue
            pass

        # Fan out baskets using a compiled baskets graph
        compiled_basket = build_basket_graph()
        reports: list[dict] = []
        for b in baskets:
            try:
                out = compiled_basket.invoke({"settings": settings, "basket": b})
                rep = out.get("basket_report")
                if rep:
                    reports.append(rep)
            except Exception:
                continue
        return {
            "portfolio_persisted": True,
            "instruments": instruments,
            "baskets": baskets,
            "basket_reports": reports,
        }

    graph.add_node("commit_portfolio", _commit_portfolio_node)
    graph.add_node("analyst", analyst_node, defer=True)

    graph.set_entry_point("ingestion")
    graph.add_edge("ingestion", "planner")
    graph.add_edge("planner", "dispatch_parse")
    graph.add_conditional_edges("dispatch_parse", _dispatch_parse_tasks)
    graph.add_edge("parse_one", "join_after_parse")
    graph.add_edge("join_after_parse", "dispatch_resolve")
    graph.add_conditional_edges("dispatch_resolve", _dispatch_resolve_tasks)
    graph.add_edge("resolve_one", "join_after_resolve")
    graph.add_edge("join_after_resolve", "commit_portfolio")
    graph.add_edge("commit_portfolio", "analyst")
    graph.add_edge("analyst", END)

    return graph.compile()
