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
        self._closed = False

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

    # Lifecycle ----------------------------------------------------------------
    def close(self) -> None:
        """Close underlying HTTP resources if supported by the client.

        Safe to call multiple times.
        """
        if self._closed:
            return
        client = self._client
        self._closed = True
        if client is None:
            return
        try:
            # polygon RESTClient may expose close() depending on version
            close = getattr(client, "close", None)
            if callable(close):
                close()
        except Exception:
            # Best-effort cleanup
            pass

    def __enter__(self) -> PolygonClient:
        self._ensure_client()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # noqa: D401 (concise)
        self.close()

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

    def list_aggs_daily(
        self,
        ticker: str,
        from_date: str,
        to_date: str,
        adjusted: bool = True,
        limit: int = 50000,
    ) -> Iterable[dict[str, Any]]:
        """Yield normalized daily OHLCV bars for a ticker between inclusive dates.

        Parameters use ISO dates (YYYY-MM-DD). Returns dicts with fields:
        {date, open, high, low, close, volume, vwap?}
        """
        client = self._ensure_client()
        # Timespan enum compatible with string "day"
        try:
            from polygon.enums import Timespan  # type: ignore

            timespan = Timespan.DAY
        except Exception:  # pragma: no cover - fallback if enums change
            timespan = "day"

        for bar in client.list_aggs(
            ticker=ticker,
            multiplier=1,
            timespan=timespan,
            from_=from_date,
            to=to_date,
            adjusted=adjusted,
            limit=limit,
        ):
            # Normalize across possible return types
            if hasattr(bar, "model_dump"):
                b = bar.model_dump()
            elif hasattr(bar, "__dict__"):
                b = dict(bar.__dict__)
            else:
                b = dict(bar)

            # Support both raw JSON short keys and client-mapped verbose keys
            # Timestamp (ms)
            ts = b.get("t") or b.get("timestamp")
            date_str: str | None = None
            if ts is not None:
                try:
                    import datetime as _dt

                    date_str = _dt.datetime.fromtimestamp(int(ts) / 1000, _dt.UTC).strftime(
                        "%Y-%m-%d"
                    )
                except Exception:  # pragma: no cover - defensive
                    date_str = None

            open_v = b.get("o") if b.get("o") is not None else b.get("open")
            high_v = b.get("h") if b.get("h") is not None else b.get("high")
            low_v = b.get("l") if b.get("l") is not None else b.get("low")
            close_v = b.get("c") if b.get("c") is not None else b.get("close")
            volume_v = b.get("v") if b.get("v") is not None else b.get("volume")
            vwap_v = b.get("vw") if b.get("vw") is not None else b.get("vwap")

            yield {
                "date": date_str,
                "open": float(open_v or 0.0),
                "high": float(high_v or 0.0),
                "low": float(low_v or 0.0),
                "close": float(close_v or 0.0),
                "volume": int(volume_v or 0),
                "vwap": float(vwap_v) if vwap_v is not None else None,
            }

    def list_ticker_news(
        self,
        ticker: str,
        published_utc_gte: str | None = None,
        published_utc_lte: str | None = None,
        limit: int = 1000,
        order: str = "desc",
    ) -> Iterable[dict[str, Any]]:
        """Yield news articles for a ticker from Polygon's news endpoint.

        Parameters:
            ticker: Stock ticker symbol
            published_utc_gte: Filter for articles published on or after (ISO format)
            published_utc_lte: Filter for articles published on or before (ISO format)
            limit: Max results per page (default 1000)
            order: Sort order - 'asc' or 'desc' by published_utc
        """
        client = self._ensure_client()

        # The polygon client has a list_ticker_news method
        for article in client.list_ticker_news(
            ticker=ticker,
            published_utc_gte=published_utc_gte,
            published_utc_lte=published_utc_lte,
            limit=limit,
            order=order,
        ):
            if hasattr(article, "model_dump"):
                yield article.model_dump()
            elif hasattr(article, "__dict__"):
                yield dict(article.__dict__)
            else:
                yield dict(article)
