from __future__ import annotations

import pytest

from portfolio_advisor.models.canonical import CanonicalHolding, InstrumentKey


def test_instrument_key_str_and_parse_roundtrip():
    key = InstrumentKey(asset_class="stocks", locale="us", mic="XNAS", symbol="AAPL")
    s = str(key)
    assert s == "cid:stocks:us:XNAS:AAPL"
    parsed = InstrumentKey.parse(s)
    assert parsed.asset_class == "stocks"
    assert parsed.locale == "us"
    assert parsed.mic == "XNAS"
    assert parsed.symbol == "AAPL"


def test_canonical_holding_validates_instrument_id_matches_fields():
    instrument_id = "cid:stocks:us:XNYS:BRK.A"
    h = CanonicalHolding(
        instrument_id=instrument_id,
        asset_class="stocks",
        locale="us",
        mic="XNYS",
        primary_ticker="BRK.A",
        symbol="BRK.A",
        polygon_ticker="BRK.A",
        resolution_confidence=0.9,
    )
    assert h.instrument_id == instrument_id


def test_canonical_holding_raises_on_mismatched_identity():
    with pytest.raises(ValueError):
        CanonicalHolding(
            instrument_id="cid:stocks:us:XNAS:AAPL",
            asset_class="stocks",
            locale="us",
            mic="XNYS",  # mismatch
            primary_ticker="AAPL",
            symbol="AAPL",
            polygon_ticker="AAPL",
            resolution_confidence=1.0,
        )
