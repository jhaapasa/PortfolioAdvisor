"""Tests for error classes."""

from __future__ import annotations

from portfolio_advisor.errors import ConfigurationError, InputOutputError, PortfolioAdvisorError


def test_input_output_error():
    """Test InputOutputError can be raised and caught."""
    try:
        raise InputOutputError("Test IO error")
    except InputOutputError as e:
        assert str(e) == "Test IO error"
        assert isinstance(e, PortfolioAdvisorError)


def test_portfolio_advisor_error():
    """Test PortfolioAdvisorError base class."""
    try:
        raise PortfolioAdvisorError("Test base error")
    except PortfolioAdvisorError as e:
        assert str(e) == "Test base error"
        assert isinstance(e, Exception)


def test_configuration_error():
    """Test ConfigurationError can be raised and caught."""
    try:
        raise ConfigurationError("Test config error")
    except ConfigurationError as e:
        assert str(e) == "Test config error"
        assert isinstance(e, PortfolioAdvisorError)
