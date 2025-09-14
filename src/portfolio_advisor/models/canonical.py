from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, model_validator


class InstrumentKey(BaseModel):
    """Human-readable, self-documenting, unambiguous instrument key.

    String form: "cid:{asset_class}:{locale}:{mic}:{symbol}"
    Examples:
      - cid:stocks:us:XNAS:AAPL
      - cid:stocks:us:XNYS:BRK.A
      - cid:crypto:global:composite:BTCUSD
      - cid:fx:global:composite:EURUSD
      - cid:options:us:OPRA:AAPL241220C00150000
    """

    asset_class: str = Field(description="stocks | options | crypto | fx")
    locale: str = Field(description="Market locale, e.g., 'us' or 'global'")
    mic: str = Field(description="ISO 10383 MIC or 'composite'")
    symbol: str = Field(description="Provider-normalized symbol (e.g., AAPL, BRK.A, BTCUSD)")

    def __str__(self) -> str:  # pragma: no cover - trivial
        return f"cid:{self.asset_class}:{self.locale}:{self.mic}:{self.symbol}"

    @classmethod
    def parse(cls, instrument_id: str) -> InstrumentKey:
        if not instrument_id.startswith("cid:"):
            msg = "Invalid instrument_id: missing 'cid:' prefix"
            raise ValueError(msg)
        parts = instrument_id.split(":", 4)
        if len(parts) != 5:
            msg = "Invalid instrument_id: expected 5 colon-separated parts"
            raise ValueError(msg)
        _, asset_class, locale, mic, symbol = parts
        return cls(asset_class=asset_class, locale=locale, mic=mic, symbol=symbol)


class CanonicalHolding(BaseModel):
    """Resolver-normalized holding with canonical instrument identity.

    Carries forward parsed quantitative fields and enriches with provider metadata.
    """

    # Canonical identity
    instrument_id: str = Field(description="Stringified InstrumentKey 'cid:...' format")
    asset_class: str
    locale: str
    mic: str
    primary_ticker: str
    symbol: str = Field(description="Normalized symbol used in instrument_id")

    # Provider references (traceability)
    polygon_ticker: str
    composite_figi: str | None = None
    share_class_figi: str | None = None
    cik: str | None = None

    # Descriptive
    company_name: str | None = None
    currency: str | None = None
    as_of: str | None = None

    # Portfolio context carried from parse stage
    quantity: float | None = None
    weight: float | None = None
    account: str | None = None
    basket: str | None = None
    source_doc_id: str | None = None

    # Resolver diagnostics
    resolution_confidence: float = Field(ge=0.0, le=1.0, default=0.0)
    resolution_notes: str | None = None
    identifiers: dict[str, Any] | None = None

    @model_validator(mode="after")
    def _validate_identity(self) -> CanonicalHolding:  # pragma: no cover - simple guard
        # Ensure instrument_id matches asset_class/locale/mic/symbol
        try:
            key = InstrumentKey.parse(self.instrument_id)
        except Exception as exc:  # re-raise with context
            msg = f"Invalid instrument_id: {exc}"
            raise ValueError(msg) from exc
        if not (
            key.asset_class == self.asset_class
            and key.locale == self.locale
            and key.mic == self.mic
            and key.symbol == self.symbol
        ):
            msg = "instrument_id does not match fields asset_class/locale/mic/symbol"
            raise ValueError(msg)
        return self
