from __future__ import annotations

from pathlib import Path
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from ..config import Settings


class BasketState(TypedDict, total=False):
    settings: Any
    basket: dict
    metrics: dict
    report_md: str
    basket_report: dict


def _collect_inputs_node(state: BasketState) -> dict:
    s: Settings = state["settings"]
    basket = state["basket"]
    sl: str = basket.get("slug", "")
    # Prefer persisted positions for an authoritative list of tickers in the basket
    derived_tickers: list[str] = []
    try:
        portfolio_dir = getattr(s, "portfolio_dir")
        pos_path = Path(str(portfolio_dir)) / "baskets" / sl / "positions.json"
        if pos_path.exists():
            import json as _json

            items = _json.loads(pos_path.read_text(encoding="utf-8"))
            cand = [str(it.get("primary_ticker") or "").strip() for it in items]
            derived_tickers = sorted({t for t in cand if t})
    except Exception:  # pragma: no cover - best effort
        derived_tickers = []

    tickers: list[str] = derived_tickers or (basket.get("tickers", []) or [])
    base = Path(s.output_dir) / "stocks" / "tickers"
    rows: list[dict[str, Any]] = []
    as_of = None
    for t in tickers:
        ret_path = base / t / "analysis" / "returns.json"
        d1 = None
        d5 = None
        try:
            if ret_path.exists():
                import json

                with ret_path.open("r", encoding="utf-8") as fh:
                    data = json.load(fh)
                w = data.get("windows", {})
                d5 = w.get("d5")
                d1 = w.get("d1")
                as_of = as_of or data.get("as_of")
        except Exception:  # pragma: no cover - best effort
            pass
        # Fallback: derive d1/d5 from primary OHLC if needed
        if d1 is None or d5 is None:
            ohlc_path = base / t / "primary" / "ohlc_daily.json"
            try:
                if ohlc_path.exists():
                    import json as _json

                    with ohlc_path.open("r", encoding="utf-8") as fh:
                        ohlc = _json.load(fh)
                    closes = [float(r.get("close", 0.0)) for r in ohlc.get("data", [])]
                    as_of = as_of or (ohlc.get("coverage") or {}).get("end_date")
                    def trailing(n: int) -> float | None:
                        if len(closes) <= n:
                            return None
                        c0 = closes[-n - 1]
                        ct = closes[-1]
                        if c0 == 0:
                            return None
                        return (ct / c0) - 1.0
                    if d1 is None:
                        d1 = trailing(1)
                    if d5 is None:
                        d5 = trailing(5)
            except Exception:  # pragma: no cover - best effort
                pass
        # Only include row if we have at least one metric
        if d1 is not None or d5 is not None:
            rows.append({"ticker": t, "d1": d1, "d5": d5})
    return {"_collected": {"rows": rows, "as_of": as_of, "slug": sl}}


def _compute_metrics_node(state: BasketState) -> dict:
    basket = state["basket"]
    collected = state.get("_collected", {})
    rows: list[dict[str, Any]] = collected.get("rows", [])
    as_of = collected.get("as_of")
    # Compute equal-weight averages ignoring None values
    def _avg(key: str) -> float | None:
        vals = [float(r[key]) for r in rows if r.get(key) is not None]
        if not vals:
            return None
        return sum(vals) / len(vals)

    d1_avg = _avg("d1")
    d5_avg = _avg("d5")

    # Top movers up/down by key
    def _top(key: str, reverse: bool) -> list[str]:
        present = [r for r in rows if r.get(key) is not None]
        present.sort(key=lambda r: float(r[key]), reverse=reverse)
        return [r["ticker"] for r in present[:3]]

    metrics = {
        "basket": {"id": basket.get("id"), "label": basket.get("label"), "slug": basket.get("slug")},
        "as_of": as_of,
        "tickers": rows,
        "averages": {"d1": d1_avg, "d5": d5_avg},
        "top_movers": {
            "d1_up": _top("d1", True),
            "d1_down": _top("d1", False)[-3:][::-1] if rows else [],
            "d5_up": _top("d5", True),
            "d5_down": _top("d5", False)[-3:][::-1] if rows else [],
        },
        "depends_on": ["stocks.analysis.returns"],
    }
    return {"metrics": metrics}


def _summarize_with_llm_node(state: BasketState) -> dict:
    # Keep simple: format a small markdown summary without calling LLM for now
    m = state["metrics"]
    label = m["basket"]["label"]
    d1 = m["averages"].get("d1")
    d5 = m["averages"].get("d5")
    bullets = [
        f"- {label} average d1: {d1:+.3%}" if d1 is not None else "- d1: n/a",
        f"- {label} average d5: {d5:+.3%}" if d5 is not None else "- d5: n/a",
    ]
    report_md = "\n".join(bullets)
    return {"report_md": report_md}


def _write_outputs_node(state: BasketState) -> dict:
    s: Settings = state["settings"]
    m = state["metrics"]
    slug = m["basket"]["slug"]
    out_dir = Path(s.output_dir) / "baskets" / slug
    out_dir.mkdir(parents=True, exist_ok=True)
    import json

    metrics_path = out_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as fh:
        json.dump({**m, "generated_at": _utcnow_iso()}, fh, ensure_ascii=False, indent=2)
    report_path = out_dir / "report.md"
    report_path.write_text(state.get("report_md") or "", encoding="utf-8")
    return {
        "basket_report": {
            "id": m["basket"]["id"],
            "label": m["basket"]["label"],
            "slug": slug,
            "metrics_path": str(metrics_path),
            "report_path": str(report_path),
            "summary_text": state.get("report_md") or "",
            "averages": m.get("averages", {}),
        }
    }


def build_basket_graph():
    graph = StateGraph(BasketState)
    graph.add_node("collect_inputs", _collect_inputs_node)
    graph.add_node("compute_metrics", _compute_metrics_node)
    graph.add_node("summarize", _summarize_with_llm_node)
    graph.add_node("write_outputs", _write_outputs_node)
    graph.set_entry_point("collect_inputs")
    graph.add_edge("collect_inputs", "compute_metrics")
    graph.add_edge("compute_metrics", "summarize")
    graph.add_edge("summarize", "write_outputs")
    graph.add_edge("write_outputs", END)
    return graph.compile()


def _utcnow_iso() -> str:
    from datetime import UTC, datetime

    return datetime.now(UTC).isoformat()


