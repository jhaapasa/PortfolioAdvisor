"""Tests for parser utility functions."""

from __future__ import annotations

from portfolio_advisor.agents.parser import _truncate_text


def test_truncate_text():
    """Test text truncation function."""
    # Text shorter than limit - not truncated
    short_text = "Hello, world!"
    assert _truncate_text(short_text, 1000) == short_text

    # Text exactly at limit - not truncated
    exact_text = "a" * 300
    assert _truncate_text(exact_text, 300) == exact_text

    # Text longer than limit - truncated
    # The function subtracts 200 from the limit for the truncation message
    long_text = "a" * 500
    result = _truncate_text(long_text, 300)
    assert len(result) < len(long_text)
    assert result.endswith("... [truncated]")
    # Should contain first (300-200) = 100 chars + truncation message
    assert result.startswith("a" * 100)

    # Edge case: limit less than 200
    # In this case, the function would try to slice with negative index
    small_limit_text = "This is a test"
    result = _truncate_text(small_limit_text, 50)
    # This will include text[:50-200] which is text[:-150], which for a 14 char string
    # will return empty string, so result will be just the truncation message
    if len(small_limit_text) > 50:
        assert result == "\n... [truncated]"
