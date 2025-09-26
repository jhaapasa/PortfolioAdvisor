"""Tests for the resolver agent module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from portfolio_advisor.agents.resolver import _build_resolver, resolve_one_node
from portfolio_advisor.models.canonical import CanonicalHolding


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """Clean environment variables to avoid interference."""
    # Remove any existing API keys from environment
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("POLYGON_API_KEY", raising=False)


def test_build_resolver_offline_mode(monkeypatch, caplog):
    """Test building resolver in offline mode without API key."""
    import logging

    # Ensure we capture logs from the resolver module
    logging.getLogger("portfolio_advisor.agents.resolver").setLevel(logging.INFO)

    # Mock PolygonClient to ensure it's not created when api_key is None
    mock_polygon_client = MagicMock()
    monkeypatch.setattr("portfolio_advisor.agents.resolver.PolygonClient", mock_polygon_client)

    # Create settings with no API key - explicitly override any env vars
    settings = MagicMock()
    settings.polygon_api_key = None
    settings.polygon_base_url = None
    settings.polygon_timeout_s = 10
    settings.resolver_preferred_mics = "XNAS,XNYS,ARCX"
    settings.resolver_default_locale = "us"
    settings.resolver_confidence_threshold = 0.6

    with caplog.at_level(logging.INFO):
        resolver = _build_resolver(settings)

    # PolygonClient should NOT have been created
    mock_polygon_client.assert_not_called()

    # Check that offline mode was logged
    assert "Polygon API key not set; resolver will operate in offline mode" in caplog.text

    # Resolver should still be created with None provider
    assert resolver is not None
    assert resolver._provider is None
    assert resolver._config.default_locale == "us"
    assert resolver._config.preferred_mics == ("XNAS", "XNYS", "ARCX")


def test_build_resolver_with_api_key(monkeypatch):
    """Test building resolver with API key creates PolygonClient."""
    # Mock PolygonClient
    mock_polygon_instance = MagicMock()
    mock_polygon_class = MagicMock(return_value=mock_polygon_instance)
    monkeypatch.setattr("portfolio_advisor.agents.resolver.PolygonClient", mock_polygon_class)

    # Create settings with API key
    settings = MagicMock()
    settings.polygon_api_key = "test-api-key"
    settings.polygon_base_url = "https://api.test.com"
    settings.polygon_timeout_s = 20
    settings.resolver_preferred_mics = "XNAS,XNYS"
    settings.resolver_default_locale = "us"
    settings.resolver_confidence_threshold = 0.7

    resolver = _build_resolver(settings)

    # PolygonClient should have been created with correct params
    mock_polygon_class.assert_called_once_with(
        api_key="test-api-key", base_url="https://api.test.com", timeout_s=20
    )

    # Resolver should be created with the provider
    assert resolver is not None
    assert resolver._provider == mock_polygon_instance
    assert resolver._config.default_locale == "us"
    assert resolver._config.preferred_mics == ("XNAS", "XNYS")
    assert resolver._config.confidence_threshold == 0.7


def test_build_resolver_with_mic_list(monkeypatch):
    """Test building resolver with MICs as list instead of string."""
    # Mock PolygonClient to avoid real instantiation
    mock_polygon_class = MagicMock()
    monkeypatch.setattr("portfolio_advisor.agents.resolver.PolygonClient", mock_polygon_class)

    settings = MagicMock()
    settings.polygon_api_key = "dummy-key"
    settings.polygon_base_url = None
    settings.polygon_timeout_s = 10
    # Simulate MICs provided as list (e.g., from override)
    settings.resolver_preferred_mics = ["XNAS", "XNYS", "ARCX"]
    settings.resolver_default_locale = "us"
    settings.resolver_confidence_threshold = 0.6

    resolver = _build_resolver(settings)
    assert resolver._config.preferred_mics == ("XNAS", "XNYS", "ARCX")


def test_build_resolver_with_invalid_mics(monkeypatch):
    """Test building resolver with invalid MICs falls back to defaults."""
    # Mock PolygonClient to avoid real instantiation
    mock_polygon_class = MagicMock()
    monkeypatch.setattr("portfolio_advisor.agents.resolver.PolygonClient", mock_polygon_class)

    settings = MagicMock()
    settings.polygon_api_key = "dummy-key"
    settings.polygon_base_url = None
    settings.polygon_timeout_s = 10
    # Simulate invalid MICs that can't be converted to strings
    settings.resolver_preferred_mics = object()  # Not iterable
    settings.resolver_default_locale = "us"
    settings.resolver_confidence_threshold = 0.6

    resolver = _build_resolver(settings)
    # Should fall back to defaults
    assert resolver._config.preferred_mics == ("XNAS", "XNYS", "ARCX")


def test_resolve_one_node_success(monkeypatch):
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

    # Mock the Settings class to avoid environment variable interference
    mock_settings = MagicMock()
    mock_settings.polygon_api_key = "test-key"

    state = {
        "settings": mock_settings,
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


def test_resolve_one_node_unresolved(monkeypatch):
    """Test handling of unresolved holding."""
    mock_resolver = MagicMock()
    unresolved_dict = {
        "symbol": "UNKNOWN",
        "name": "Unknown Company",
        "reason": "No match found",
    }
    mock_resolver.resolve_one.return_value = unresolved_dict

    # Mock the Settings class
    mock_settings = MagicMock()
    mock_settings.polygon_api_key = "test-key"

    state = {
        "settings": mock_settings,
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


def test_resolve_one_node_no_holding(monkeypatch):
    """Test handling when no holding is provided in state."""
    mock_resolver = MagicMock()
    mock_resolver.resolve_one.return_value = {"reason": "No holding provided"}

    # Mock the Settings class
    mock_settings = MagicMock()
    mock_settings.polygon_api_key = "test-key"

    state = {
        "settings": mock_settings,
        # No holding key
    }

    with patch("portfolio_advisor.agents.resolver._build_resolver", return_value=mock_resolver):
        result = resolve_one_node(state)

    # Should still work with empty dict
    assert "unresolved_entities" in result
    mock_resolver.resolve_one.assert_called_once_with({})
