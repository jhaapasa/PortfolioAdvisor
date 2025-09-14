from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from ..models.canonical import CanonicalHolding, InstrumentKey

logger = logging.getLogger(__name__)


@dataclass
class ResolverConfig:
    default_locale: str = "us"
    preferred_mics: tuple[str, ...] = ("XNAS", "XNYS", "ARCX")
    confidence_threshold: float = 0.6


class SymbolResolver:
    """Scaffold for a ticker/identifier normalization resolver.

    This class will call a provider (e.g., Polygon) to validate inputs and
    produce a CanonicalHolding. Implementation will be filled in step 2.
    """

    def __init__(self, provider: Any, config: ResolverConfig | None = None) -> None:
        self._provider = provider
        self._config = config or ResolverConfig()

    # Public API ---------------------------------------------------------------
    def resolve_one(self, parsed: dict[str, Any]) -> CanonicalHolding | dict[str, Any]:
        """Resolve a single parsed holding into a CanonicalHolding or return an unresolved record.

        Basic strategy:
          1) If primary_ticker present, validate via provider and use details.
          2) Else try strong IDs (cusip/figi via provider filters if supported).
          3) Else try name search via provider and rank simple heuristics.
          4) If no provider (offline), mark unresolved.
        """
        name = str(parsed.get("name") or "")
        source_doc_id = str(parsed.get("source_doc_id") or "")
        currency = parsed.get("currency")
        primary_ticker = parsed.get("primary_ticker")

        if self._provider is None:
            return {
                "name": name,
                "source_doc_id": source_doc_id,
                "reason": "no_provider",
                "input": parsed,
            }

        # 1) Validate explicit primary ticker
        if primary_ticker:
            try:
                details = self._provider.get_ticker_details(primary_ticker)
                return self._to_canonical(parsed, details, confidence=1.0, note="exact_ticker")
            except Exception as exc:  # pragma: no cover - provider/network specific
                logger.debug("Ticker details failed for %s: %s", primary_ticker, exc)

        # 2) Strong IDs search (CUSIP/FIGI)
        for key in ("cusip", "figi"):
            value = parsed.get(key)
            if value:
                try:
                    candidates = list(self._provider.list_tickers(**{key: value, "active": True}))
                    best = self._rank_candidates(candidates, currency)
                    if best is not None:
                        return self._to_canonical(parsed, best, confidence=0.95, note=f"id:{key}")
                except Exception as exc:  # pragma: no cover
                    logger.debug("ID search failed for %s=%s: %s", key, value, exc)

        # 3) Name search fallback
        if name:
            try:
                candidates = list(
                    self._provider.list_tickers(search=name, active=True, market="stocks")
                )
                best = self._rank_candidates(candidates, currency)
                if best is not None:
                    return self._to_canonical(parsed, best, confidence=0.7, note="name_search")
            except Exception as exc:  # pragma: no cover
                logger.debug("Name search failed for %r: %s", name, exc)

        # Unresolved
        return {
            "name": name,
            "source_doc_id": source_doc_id,
            "reason": "no_match",
            "input": parsed,
        }

    # Helpers -----------------------------------------------------------------
    @staticmethod
    def build_instrument_id(asset_class: str, locale: str, mic: str, symbol: str) -> str:
        key = InstrumentKey(asset_class=asset_class, locale=locale, mic=mic, symbol=symbol)
        return str(key)

    # Internal helpers ---------------------------------------------------------
    def _rank_candidates(
        self, items: list[dict[str, Any]], currency: str | None
    ) -> dict[str, Any] | None:
        if not items:
            return None

        # Simple scoring: active + preferred MIC + currency match
        def score(item: dict[str, Any]) -> int:
            s = 0
            if item.get("active"):
                s += 3
            # Polygon fields: primary_exchange or primary_exchange_mic may vary by version
            mic = item.get("primary_exchange_mic") or item.get("primary_exchange") or ""
            if mic in self._config.preferred_mics:
                s += 2
            if currency and item.get("currency_name") == currency:
                s += 1
            market = item.get("market")
            if market == "stocks":
                s += 1
            return s

        items_sorted = sorted(items, key=score, reverse=True)
        return items_sorted[0] if items_sorted else None

    def _to_canonical(
        self,
        parsed: dict[str, Any],
        poly: dict[str, Any],
        confidence: float,
        note: str,
    ) -> CanonicalHolding:
        # Extract fields from Polygon response; adapt to possible key names across versions
        ticker = str(poly.get("ticker") or poly.get("symbol") or parsed.get("primary_ticker") or "")
        name = str(poly.get("name") or poly.get("company_name") or parsed.get("company_name") or "")
        market = str(poly.get("market") or "stocks")
        asset_class = self._map_market_to_asset_class(market)
        locale = str(poly.get("locale") or self._config.default_locale)
        mic = str(poly.get("primary_exchange_mic") or poly.get("primary_exchange") or "composite")
        currency = parsed.get("currency") or poly.get("currency_name")

        symbol = ticker
        instrument_id = self.build_instrument_id(asset_class, locale, mic, symbol)

        identifiers = {}
        for key in ("composite_figi", "share_class_figi", "cik", "cusip", "figi"):
            if poly.get(key):
                identifiers[key] = poly.get(key)

        return CanonicalHolding(
            instrument_id=instrument_id,
            asset_class=asset_class,
            locale=locale,
            mic=mic,
            primary_ticker=ticker,
            symbol=symbol,
            polygon_ticker=ticker,
            composite_figi=poly.get("composite_figi"),
            share_class_figi=poly.get("share_class_figi"),
            cik=poly.get("cik"),
            company_name=name or None,
            currency=currency or None,
            as_of=parsed.get("as_of"),
            quantity=parsed.get("quantity"),
            weight=parsed.get("weight"),
            account=parsed.get("account"),
            basket=parsed.get("basket"),
            source_doc_id=parsed.get("source_doc_id"),
            resolution_confidence=float(confidence),
            resolution_notes=note,
            identifiers=identifiers or None,
        )

    @staticmethod
    def _map_market_to_asset_class(market: str) -> str:
        m = market.lower()
        if "option" in m:
            return "options"
        if m in {"crypto", "cryptocurrency"}:
            return "crypto"
        if m in {"fx", "forex", "currencies"}:
            return "fx"
        return "stocks"
