from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator


class ParsedHolding(BaseModel):
    """A single candidate holding extracted from a source document.

    Many fields are optional because the LLM parser may be unable to extract them reliably.
    The ticker resolver downstream can use any available identifiers to normalize.
    """

    # Required core fields
    name: str = Field(description="Name of the security as found in the document")
    confidence: float = Field(ge=0.0, le=1.0, description="Model confidence in this extraction")
    source_doc_id: str = Field(description="Identifier of the source document (e.g., filename)")

    # Optional identifiers and attributes for downstream resolver
    primary_ticker: str | None = Field(default=None, description="Best guess for primary ticker")
    company_name: str | None = Field(default=None, description="Canonical company/issuer name")
    cusip: str | None = None
    isin: str | None = None
    sedol: str | None = None

    quantity: float | None = Field(default=None, description="Units/Shares as parsed")
    weight: float | None = Field(
        default=None,
        description="Portfolio weight as a fraction (0-1) or %/100",
    )
    currency: str | None = Field(default=None, description="Currency code, e.g., USD")
    as_of: str | None = Field(default=None, description="As-of date string as seen in document")

    account: str | None = Field(default=None, description="Account name/identifier if present")
    basket: str | None = Field(
        default=None,
        description='Basket name/identifier. Use "[none]" if not in any basket.',
    )

    identifiers: dict[str, Any] | None = Field(
        default=None,
        description="Additional freeform identifiers helpful for resolver (e.g., FIGI, BBGID)",
    )
    notes: str | None = None

    @field_validator("basket")
    @classmethod
    def _normalize_basket(cls, value: str | None) -> str | None:
        if value is None:
            return value
        v = value.strip()
        return v if v else "[none]"


class ParsedHoldingsResult(BaseModel):
    """Result of parsing a single source document into candidate holdings."""

    source_doc_id: str = Field(description="Identifier of the source document (e.g., filename)")
    as_of: str | None = Field(default=None, description="Document-level as-of date if present")
    holdings: list[ParsedHolding] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
