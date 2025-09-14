from __future__ import annotations

from collections.abc import Iterable
from typing import Any


class PolygonClient:
    """Thin wrapper around polygon-api-client's RESTClient with lazy import.

    This avoids importing the dependency unless a method is actually called,
    making tests easy to run with mocks and no external dependency.
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout_s: int | None = 10,
        trace: bool = False,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url
        self._timeout_s = timeout_s
        self._trace = trace
        self._client = None  # created on first use

    # Internal -----------------------------------------------------------------
    def _ensure_client(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            from polygon import RESTClient  # type: ignore
        except Exception as exc:  # pragma: no cover - dependency not installed in CI by default
            msg = (
                "polygon-api-client is required to use PolygonClient. Install with 'pip install "
                "polygon-api-client'. Original error: "
                f"{exc}"
            )
            raise RuntimeError(msg) from exc

        # RESTClient accepts api_key and trace. Base URL/timeout are controlled by environment
        # variables in the upstream client; we keep them here for future customization.
        self._client = RESTClient(api_key=self._api_key, trace=self._trace)
        return self._client

    # Public API ----------------------------------------------------------------
    def get_ticker_details(self, ticker: str) -> dict[str, Any]:
        client = self._ensure_client()
        data = client.get_ticker_details(ticker)
        # The client returns a pydantic-like object or dict depending on version; normalize to dict
        if hasattr(data, "model_dump"):
            return data.model_dump()  # type: ignore[no-any-return]
        if hasattr(data, "__dict__"):
            return dict(data.__dict__)
        return dict(data)

    def list_tickers(self, **filters: Any) -> Iterable[dict[str, Any]]:
        """Yield normalized dicts of tickers from Polygon's list_tickers iterator.

        Accepts filters such as: ticker, market, type, search, active, cik, cusip,
        figi, order, limit.
        """
        client = self._ensure_client()
        for item in client.list_tickers(**filters):
            if hasattr(item, "model_dump"):
                yield item.model_dump()
            elif hasattr(item, "__dict__"):
                yield dict(item.__dict__)
            else:
                yield dict(item)
