"""Tests for LLM utilities."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from portfolio_advisor.config import Settings
from portfolio_advisor.llm import (
    LoggingLLMProxy,
    _approx_tokens,
    _excerpt,
    build_llm,
    get_llm,
)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """Clean environment variables to avoid interference."""
    # Remove any existing API keys from environment
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("POLYGON_API_KEY", raising=False)
    # Clear the LRU cache between tests
    from portfolio_advisor.llm import _build_llm_cached

    _build_llm_cached.cache_clear()


def test_excerpt():
    """Test text excerpt function."""
    # Test normal string
    assert _excerpt("Hello, world!") == "Hello, world!"

    # Test long string gets truncated
    long_text = "A" * 30
    assert _excerpt(long_text) == "A" * 20

    # Test newline replacement
    assert _excerpt("Hello\nworld\ntest") == "Hello world test"

    # Test non-string input
    assert _excerpt(123) == "123"

    # Test with custom limit
    assert _excerpt("Hello, world!", limit=5) == "Hello"


def test_approx_tokens():
    """Test token approximation function."""
    # Test normal strings
    assert _approx_tokens("test") == 1  # 4 chars = 1 token
    assert _approx_tokens("hello world") == 3  # 11 chars ≈ 3 tokens
    assert _approx_tokens("A" * 20) == 5  # 20 chars = 5 tokens

    # Test empty string
    assert _approx_tokens("") == 1  # min is 1

    # Test non-string input
    assert _approx_tokens(12345) == 2  # "12345" = 5 chars ≈ 2 tokens


class TestLoggingLLMProxy:
    """Test the LoggingLLMProxy class."""

    def test_init_and_delegation(self):
        """Test proxy initialization and attribute delegation."""
        mock_llm = MagicMock()
        mock_llm.some_attribute = "value"
        mock_llm.some_method = MagicMock(return_value="result")

        proxy = LoggingLLMProxy(mock_llm)

        # Test attribute delegation
        assert proxy.some_attribute == "value"
        assert proxy.some_method() == "result"

    def test_model_name_extraction(self):
        """Test model name extraction from various attributes."""
        # Test with model attribute
        mock_llm = MagicMock()
        mock_llm.model = "gpt-4"
        proxy = LoggingLLMProxy(mock_llm)
        assert proxy._model_name() == "gpt-4"

        # Test with model_name attribute
        mock_llm = MagicMock()
        mock_llm.model_name = "claude-3"
        proxy = LoggingLLMProxy(mock_llm)
        assert proxy._model_name() == "claude-3"

        # Test fallback to class name
        mock_llm = MagicMock()
        mock_llm.__class__.__name__ = "CustomLLM"
        proxy = LoggingLLMProxy(mock_llm)
        assert proxy._model_name() == "CustomLLM"

    def test_invoke_logging(self, caplog):
        """Test synchronous invoke with logging."""
        mock_llm = MagicMock()
        mock_llm.model = "test-model"
        mock_response = MagicMock()
        mock_response.content = "Test response"
        mock_llm.invoke.return_value = mock_response

        proxy = LoggingLLMProxy(mock_llm)

        with caplog.at_level("INFO"):
            result = proxy.invoke("Test prompt")

        assert result == mock_response
        mock_llm.invoke.assert_called_once_with("Test prompt")

        # Check logging
        assert "LLM invoke start" in caplog.text
        assert "model=test-model" in caplog.text
        assert "prompt_excerpt='Test prompt'" in caplog.text
        assert "LLM invoke end" in caplog.text
        assert "response_excerpt='Test response'" in caplog.text

    async def test_ainvoke_logging(self, caplog):
        """Test asynchronous invoke with logging."""
        mock_llm = AsyncMock()
        mock_llm.model = "test-model"
        mock_response = MagicMock()
        mock_response.content = "Async response"
        mock_llm.ainvoke.return_value = mock_response

        proxy = LoggingLLMProxy(mock_llm)

        with caplog.at_level("INFO"):
            result = await proxy.ainvoke("Async prompt")

        assert result == mock_response
        mock_llm.ainvoke.assert_called_once_with("Async prompt")

        # Check logging
        assert "LLM ainvoke start" in caplog.text
        assert "model=test-model" in caplog.text
        assert "prompt_excerpt='Async prompt'" in caplog.text
        assert "LLM ainvoke end" in caplog.text
        assert "response_excerpt='Async response'" in caplog.text


def test_build_llm_with_api_key():
    """Test building LLM with API key."""
    with patch("langchain_openai.ChatOpenAI") as mock_openai:
        mock_llm_instance = MagicMock()
        mock_openai.return_value = mock_llm_instance

        result = build_llm(
            model="gpt-4",
            temperature=0.5,
            max_tokens=1000,
            api_key="test-key",
            api_base="https://api.test.com",
            request_timeout_s=30,
        )

        # Check ChatOpenAI was called with correct params
        mock_openai.assert_called_once_with(
            model="gpt-4",
            temperature=0.5,
            max_tokens=1000,
            api_key="test-key",
            timeout=30,
            max_retries=0,
            base_url="https://api.test.com",
        )

        # Result should be wrapped in LoggingLLMProxy
        assert isinstance(result, LoggingLLMProxy)


def test_build_llm_without_api_key(caplog):
    """Test building LLM without API key returns dummy LLM."""
    with caplog.at_level("WARNING"):
        result = build_llm(
            model="gpt-4",
            temperature=0.5,
            max_tokens=None,
            api_key=None,
            api_base=None,
            request_timeout_s=None,
        )

    assert isinstance(result, LoggingLLMProxy)
    assert "OPENAI_API_KEY not set" in caplog.text

    # Test dummy LLM behavior
    inner_llm = result._inner
    assert inner_llm._llm_type == "dummy"

    # Test dummy LLM generates placeholder response
    chat_result = inner_llm._generate([])
    assert isinstance(chat_result, ChatResult)
    assert len(chat_result.generations) == 1
    assert isinstance(chat_result.generations[0], ChatGeneration)
    assert isinstance(chat_result.generations[0].message, AIMessage)
    assert "Placeholder analysis" in chat_result.generations[0].message.content


def test_get_llm():
    """Test get_llm with caching."""
    settings = Settings(
        input_dir="/tmp/in",
        output_dir="/tmp/out",
        openai_model="gpt-3.5-turbo",
        temperature=0.3,
        max_tokens=500,
        openai_api_key="test-key",
        openai_base_url=None,
        request_timeout_s=60,
    )

    with patch("portfolio_advisor.llm.build_llm") as mock_build:
        mock_llm = MagicMock()
        mock_build.return_value = mock_llm

        # First call
        result1 = get_llm(settings)
        assert result1 == mock_llm

        # Second call should use cache
        result2 = get_llm(settings)
        assert result2 == mock_llm

        # build_llm should only be called once due to caching
        mock_build.assert_called_once_with(
            model="gpt-3.5-turbo",
            temperature=0.3,
            max_tokens=500,
            api_key="test-key",
            api_base=None,
            request_timeout_s=60,
        )


def test_get_llm_different_settings():
    """Test get_llm with different settings creates different instances."""
    settings1 = Settings(
        input_dir="/tmp/in",
        output_dir="/tmp/out",
        openai_model="gpt-3.5-turbo",
        temperature=0.3,
        openai_api_key="key1",
    )

    settings2 = Settings(
        input_dir="/tmp/in",
        output_dir="/tmp/out",
        openai_model="gpt-4",  # Different model
        temperature=0.3,
        openai_api_key="key1",
    )

    with patch("portfolio_advisor.llm.build_llm") as mock_build:
        mock_build.side_effect = [MagicMock(), MagicMock()]

        result1 = get_llm(settings1)
        result2 = get_llm(settings2)

        # Should create two different instances
        assert result1 != result2
        assert mock_build.call_count == 2
