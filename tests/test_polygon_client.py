from __future__ import annotations

from portfolio_advisor.services.polygon_client import PolygonClient


class DummyREST:
    def list_aggs(self, ticker, multiplier, timespan, from_, to, adjusted, limit):  # noqa: ANN001
        assert ticker == "AAPL"
        assert multiplier == 1
        assert timespan in ("day", getattr(type("T", (), {"DAY": "day"}), "DAY", "day"))
        assert from_ == "2025-01-01"
        assert to == "2025-01-05"
        assert adjusted is True
        assert limit == 50000
        # two entries: one with short keys, one with verbose keys
        yield {"t": 1735689600000, "o": 10, "h": 11, "l": 9, "c": 10.5, "v": 100, "vw": 10.2}
        yield {
            "timestamp": 1735948800000,
            "open": 10.6,
            "high": 11.2,
            "low": 10.2,
            "close": 11.0,
            "volume": 110,
            "vwap": None,
        }


def test_list_aggs_daily_normalizes_output(monkeypatch):
    pc = PolygonClient(api_key="x")

    def fake_ensure(self):  # noqa: ANN001
        return DummyREST()

    monkeypatch.setattr(PolygonClient, "_ensure_client", fake_ensure)

    rows = list(pc.list_aggs_daily("AAPL", from_date="2025-01-01", to_date="2025-01-05"))

    assert len(rows) == 2
    r0, r1 = rows
    assert r0["date"] == "2025-01-01"
    assert r0["open"] == 10.0 and r0["volume"] == 100 and r0["vwap"] == 10.2
    assert r1["date"] == "2025-01-04"
    assert r1["close"] == 11.0 and r1["vwap"] is None


def test_close_is_idempotent(monkeypatch):
    pc = PolygonClient(api_key="x")

    class Dummy:
        def __init__(self):
            self.closed = 0

        def close(self):
            self.closed += 1

    dummy = Dummy()

    def fake_ensure(self):  # noqa: ANN001
        self._client = dummy
        return dummy

    monkeypatch.setattr(PolygonClient, "_ensure_client", fake_ensure)

    with pc as client:
        assert client is pc
        assert pc._client is dummy
    # First close should call Dummy.close once
    assert dummy.closed == 1
    pc.close()
    # Second call should not increment
    assert dummy.closed == 1


def test_get_ticker_details_handles_model_dump(monkeypatch):
    class DummyData:
        def __init__(self):
            self.model_dump_called = False

        def model_dump(self):
            self.model_dump_called = True
            return {"ticker": "AAPL", "name": "Apple"}

    dummy = DummyData()

    class DummyClient:
        def get_ticker_details(self, ticker):  # noqa: ANN001
            assert ticker == "AAPL"
            return dummy

    pc = PolygonClient(api_key="x")
    monkeypatch.setattr(pc, "_ensure_client", lambda: DummyClient())

    data = pc.get_ticker_details("AAPL")
    assert data["ticker"] == "AAPL"
    assert dummy.model_dump_called


def test_list_tickers_normalizes_items(monkeypatch):
    class DummyItem:
        def __init__(self, ticker):
            self.ticker = ticker

    class DummyClient:
        def list_tickers(self, **filters):  # noqa: ANN001
            assert filters["active"] is True
            return [DummyItem("A"), {"ticker": "B"}]

    pc = PolygonClient(api_key="x")
    monkeypatch.setattr(pc, "_ensure_client", lambda: DummyClient())

    rows = list(pc.list_tickers(active=True))
    assert rows == [{"ticker": "A"}, {"ticker": "B"}]
