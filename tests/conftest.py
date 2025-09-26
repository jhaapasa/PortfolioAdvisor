from __future__ import annotations

from collections.abc import Callable
from types import SimpleNamespace

import pytest


@pytest.fixture
def llm_stub_factory() -> Callable[[Callable[[str], str]], object]:
    """Return a factory that builds simple synchronous LLM stubs."""

    def factory(handler: Callable[[str], str]) -> object:
        class _LLM:
            def __init__(self, fn: Callable[[str], str]) -> None:
                self._fn = fn

            def invoke(self, prompt: str):  # noqa: D401 - mimic langchain API
                return SimpleNamespace(content=self._fn(prompt))

        return _LLM(handler)

    return factory


@pytest.fixture
def polygon_stub(monkeypatch):
    from portfolio_advisor.services.polygon_client import PolygonClient

    calls: list[SimpleNamespace] = []

    def install(bars):
        def _list_aggs_daily(
            self, ticker, from_date, to_date, adjusted=True, limit=50000
        ):  # noqa: ANN001
            calls.append(
                SimpleNamespace(
                    ticker=ticker,
                    from_date=from_date,
                    to_date=to_date,
                    adjusted=adjusted,
                    limit=limit,
                )
            )
            return iter(bars)

        monkeypatch.setattr(PolygonClient, "list_aggs_daily", _list_aggs_daily)
        return calls

    return install
