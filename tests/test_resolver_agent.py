"""Tests for the resolver agent module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from portfolio_advisor.agents.resolver import _build_resolver, resolve_one_node
from portfolio_advisor.config import Settings
from portfolio_advisor.models.canonical import CanonicalHolding


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """Clean environment variables to avoid interference."""
    # Remove any existing API keys from environment
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("POLYGON_API_KEY", raising=False)


def test_build_resolver_with_api_key():
    """Test building resolver with API key."""
    settings = Settings(
        input_dir="/tmp/in",
        output_dir="/tmp/out",
        polygon_api_key="test-key",
        polygon_base_url="https://api.test.com",
        polygon_timeout_s=20,
        resolver_preferred_mics="XNAS,XNYS",
        resolver_default_locale="us",
        resolver_confidence_threshold=0.7,
    )

    with patch("portfolio_advisor.agents.resolver.PolygonClient") as mock_client:
        resolver = _build_resolver(settings)

        # Check PolygonClient was created with correct params
        mock_client.assert_called_once_with(
            api_key="test-key",
            base_url="https://api.test.com",
            timeout_s=20,
        )

        # Check resolver config
        assert resolver._config.default_locale == "us"
        assert resolver._config.preferred_mics == ("XNAS", "XNYS")
        assert resolver._config.confidence_threshold == 0.7


def test_build_resolver_without_api_key(caplog):
    """Test building resolver without API key (offline mode)."""
    settings = Settings(
        input_dir="/tmp/in",
        output_dir="/tmp/out",
        polygon_api_key=None,  # No API key
    )

    with caplog.at_level("INFO"):
        resolver = _build_resolver(settings)

    assert "resolver will operate in offline mode" in caplog.text
    assert resolver._provider is None


def test_build_resolver_with_mic_list():
    """Test building resolver with MICs as list instead of string."""
    settings = Settings(
        input_dir="/tmp/in",
        output_dir="/tmp/out",
    )
    # Simulate MICs provided as list (e.g., from override)
    settings.resolver_preferred_mics = ["XNAS", "XNYS", "ARCX"]

    resolver = _build_resolver(settings)
    assert resolver._config.preferred_mics == ("XNAS", "XNYS", "ARCX")


def test_build_resolver_with_invalid_mics():
    """Test building resolver with invalid MICs falls back to defaults."""
    settings = Settings(
        input_dir="/tmp/in",
        output_dir="/tmp/out",
    )
    # Simulate invalid MICs that can't be converted to strings
    settings.resolver_preferred_mics = object()  # Not iterable

    resolver = _build_resolver(settings)
    # Should fall back to defaults
    assert resolver._config.preferred_mics == ("XNAS", "XNYS", "ARCX")


def test_resolve_one_node_success():
    """Test successful resolution of a holding."""
    mock_resolver = MagicMock()
    mock_holding = CanonicalHolding(
        instrument_id="cid:stocks:us:XNAS:AAPL",
        asset_class="stocks",
        locale="us",
        mic="XNAS",
        primary_ticker="AAPL",
        symbol="AAPL",
        polygon_ticker="AAPL",
        company_name="Apple Inc.",
        quantity=100.0,
        unit_value=150.0,
        total_value=15000.0,
    )
    mock_resolver.resolve_one.return_value = mock_holding

    state = {
        "settings": Settings(input_dir="/tmp/in", output_dir="/tmp/out"),
        "holding": {
            "symbol": "AAPL",
            "name": "Apple Inc.",
            "quantity": 100,
            "unit_value": 150,
        },
    }

    with patch("portfolio_advisor.agents.resolver._build_resolver", return_value=mock_resolver):
        result = resolve_one_node(state)

    assert "resolved_holdings" in result
    assert len(result["resolved_holdings"]) == 1
    assert result["resolved_holdings"][0]["primary_ticker"] == "AAPL"
    mock_resolver.resolve_one.assert_called_once()


def test_resolve_one_node_unresolved():
    """Test handling of unresolved holding."""
    mock_resolver = MagicMock()
    unresolved_dict = {
        "symbol": "UNKNOWN",
        "name": "Unknown Company",
        "reason": "No match found",
    }
    mock_resolver.resolve_one.return_value = unresolved_dict

    state = {
        "settings": Settings(input_dir="/tmp/in", output_dir="/tmp/out"),
        "holding": {
            "symbol": "UNKNOWN",
            "name": "Unknown Company",
            "quantity": 50,
        },
    }

    with patch("portfolio_advisor.agents.resolver._build_resolver", return_value=mock_resolver):
        result = resolve_one_node(state)

    assert "unresolved_entities" in result
    assert len(result["unresolved_entities"]) == 1
    assert result["unresolved_entities"][0] == unresolved_dict


def test_resolve_one_node_no_holding():
    """Test handling when no holding is provided in state."""
    mock_resolver = MagicMock()
    mock_resolver.resolve_one.return_value = {"reason": "No holding provided"}

    state = {
        "settings": Settings(input_dir="/tmp/in", output_dir="/tmp/out"),
        # No holding key
    }

    with patch("portfolio_advisor.agents.resolver._build_resolver", return_value=mock_resolver):
        result = resolve_one_node(state)

    # Should still work with empty dict
    assert "unresolved_entities" in result
    mock_resolver.resolve_one.assert_called_once_with({})
