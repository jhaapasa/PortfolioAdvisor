from __future__ import annotations

from typing import Any

from portfolio_advisor.tools.symbol_resolver import ResolverConfig, SymbolResolver


class DummyProvider:
    def __init__(
        self,
        tickers_by_ticker: dict[str, dict[str, Any]] | None = None,
        search_index: list[dict[str, Any]] | None = None,
    ) -> None:
        self._by_ticker = tickers_by_ticker or {}
        self._search_index = search_index or []

    def get_ticker_details(self, ticker: str) -> dict[str, Any]:
        if ticker not in self._by_ticker:
            raise ValueError("not found")
        return self._by_ticker[ticker]

    def list_tickers(self, **filters: Any):
        # Very small emulation of polygon filters used in resolver
        if "cusip" in filters:
            val = filters["cusip"]
            return (i for i in self._search_index if i.get("cusip") == val)
        if "figi" in filters:
            val = filters["figi"]
            return (i for i in self._search_index if i.get("figi") == val)
        if "search" in filters:
            term = str(filters["search"]).lower()
            return (i for i in self._search_index if term in str(i.get("name", "")).lower())
        return iter(())


def test_resolver_validates_primary_ticker():
    provider = DummyProvider(
        tickers_by_ticker={
            "AAPL": {
                "ticker": "AAPL",
                "name": "Apple Inc.",
                "market": "stocks",
                "locale": "us",
                "primary_exchange_mic": "XNAS",
                "composite_figi": "BBG000B9XRY4",
            }
        }
    )
    resolver = SymbolResolver(provider=provider, config=ResolverConfig())
    parsed = {
        "name": "Apple Inc.",
        "source_doc_id": "x.csv",
        "primary_ticker": "AAPL",
        "currency": "USD",
    }
    result = resolver.resolve_one(parsed)
    assert hasattr(result, "instrument_id")
    assert result.primary_ticker == "AAPL"
    assert result.asset_class == "stocks"
    assert result.mic == "XNAS"


def test_resolver_ranks_by_mic_and_active():
    provider = DummyProvider(
        search_index=[
            {
                "ticker": "ACME",
                "name": "Acme Corp",
                "market": "stocks",
                "active": False,
                "primary_exchange_mic": "OTCX",
            },
            {
                "ticker": "ACME",
                "name": "Acme Corporation",
                "market": "stocks",
                "active": True,
                "primary_exchange_mic": "XNAS",
            },
        ]
    )
    resolver = SymbolResolver(provider=provider, config=ResolverConfig(preferred_mics=("XNAS",)))
    parsed = {"name": "Acme", "source_doc_id": "y.csv"}
    result = resolver.resolve_one(parsed)
    assert hasattr(result, "instrument_id")
    assert result.mic == "XNAS"


def test_resolver_offline_marks_unresolved():
    resolver = SymbolResolver(provider=None, config=ResolverConfig())
    parsed = {"name": "Unknown", "source_doc_id": "z.csv"}
    result = resolver.resolve_one(parsed)
    assert isinstance(result, dict)
    assert result.get("reason") == "no_provider"
