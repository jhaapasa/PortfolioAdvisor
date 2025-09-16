"""Resolver agent: normalizes parsed holdings using provider data when available."""

from __future__ import annotations

import logging
from typing import Any

from ..models.canonical import CanonicalHolding
from ..services.polygon_client import PolygonClient
from ..tools.symbol_resolver import ResolverConfig, SymbolResolver

logger = logging.getLogger(__name__)


def _build_resolver(settings: Any) -> SymbolResolver:
    api_key = getattr(settings, "polygon_api_key", None)
    base_url = getattr(settings, "polygon_base_url", None)
    timeout_s = int(getattr(settings, "polygon_timeout_s", 10))

    provider = None
    if api_key:
        provider = PolygonClient(api_key=api_key, base_url=base_url, timeout_s=timeout_s)
    else:
        logger.info("Polygon API key not set; resolver will operate in offline mode")

    # Parse preferred MICs from comma-separated string or sequence
    raw_mics = getattr(settings, "resolver_preferred_mics", "XNAS,XNYS,ARCX")
    if isinstance(raw_mics, str):
        mic_list = tuple(m.strip() for m in raw_mics.split(",") if m.strip())
    else:  # fallback if provided as list in code
        try:
            mic_list = tuple(str(m) for m in raw_mics)
        except Exception:
            mic_list = ("XNAS", "XNYS", "ARCX")

    cfg = ResolverConfig(
        default_locale=str(getattr(settings, "resolver_default_locale", "us")),
        preferred_mics=mic_list,
        confidence_threshold=float(getattr(settings, "resolver_confidence_threshold", 0.6)),
    )
    return SymbolResolver(provider=provider, config=cfg)


def resolve_one_node(state: dict) -> dict:
    """Resolve a single parsed holding into a canonical form or mark unresolved.

    Inputs: expects `settings` and `holding` in state.
    Outputs: returns either `resolved_holdings` (list of CanonicalHolding as dicts)
    or `unresolved_entities` for aggregation.
    """
    settings = state["settings"]
    holding = state.get("holding") or {}
    resolver = _build_resolver(settings)

    try:
        result = resolver.resolve_one(holding)
        if isinstance(result, CanonicalHolding):
            logger.info("Resolved holding: %s", result.primary_ticker)
            return {"resolved_holdings": [result.model_dump()]}
        # Unresolved path (dict)
        logger.info("Unresolved holding: %s", holding.get("name"))
        return {"unresolved_entities": [result]}
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Resolver error for %r: %s", holding.get("name"), exc)
        return {
            "unresolved_entities": [
                {
                    "name": str(holding.get("name") or ""),
                    "source_doc_id": str(holding.get("source_doc_id") or ""),
                    "reason": f"resolver_exception: {exc}",
                    "input": holding,
                }
            ]
        }
