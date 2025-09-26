"""Tests for slug utilities."""

from __future__ import annotations

from portfolio_advisor.utils.slug import instrument_id_to_slug, slugify


def test_slugify():
    """Test slugify function."""
    # Basic slugification
    assert slugify("Hello World") == "hello-world"
    assert slugify("Test & Example") == "test-example"
    assert slugify("Multiple   Spaces") == "multiple-spaces"
    assert slugify("Special!@#$%Characters") == "special-characters"

    # Empty string returns 'none' as per the function docstring
    assert slugify("") == "none"
    assert slugify(None) == "none"  # type: ignore

    # Already slugified
    assert slugify("already-slugified") == "already-slugified"

    # Numbers
    assert slugify("Test 123") == "test-123"

    # Leading/trailing special chars
    assert slugify("!!!Test!!!") == "test"

    # All special chars
    assert slugify("!!!@@@###") == "none"

    # Collapse multiple dashes
    assert slugify("Test----Multiple----Dashes") == "test-multiple-dashes"


def test_instrument_id_to_slug():
    """Test instrument ID to slug conversion."""
    # Standard instrument ID
    assert instrument_id_to_slug("cid:stocks:us:XNAS:AAPL") == "cid-stocks-us-xnas-aapl"

    # Mixed case
    assert instrument_id_to_slug("CID:STOCKS:US:xnas:aapl") == "cid-stocks-us-xnas-aapl"

    # Empty string returns 'none'
    assert instrument_id_to_slug("") == "none"

    # Non-standard format
    assert instrument_id_to_slug("some:other:format") == "some-other-format"

    # With special characters
    assert instrument_id_to_slug("cid:stocks:us:XNAS:BRK.A") == "cid-stocks-us-xnas-brk-a"
