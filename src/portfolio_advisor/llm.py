"""LLM utilities: build and share chat model instances with logging wrappers."""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Protocol

from langchain_core.language_models.chat_models import BaseChatModel

if TYPE_CHECKING:  # pragma: no cover - for typing only
    from .config import Settings

logger = logging.getLogger(__name__)


def _excerpt(text: str, limit: int = 20) -> str:
    if not isinstance(text, str):
        text = str(text)
    return text[:limit].replace("\n", " ")


def _approx_tokens(text: str) -> int:
    # Very rough heuristic: ~4 characters per token on average
    try:
        length = len(text)
    except Exception:
        length = len(str(text))
    return max(1, (length + 3) // 4)


class _SupportsInvoke(Protocol):  # pragma: no cover - typing helper
    def invoke(self, input: Any, *args: Any, **kwargs: Any) -> Any: ...
    async def ainvoke(self, input: Any, *args: Any, **kwargs: Any) -> Any: ...


class LoggingLLMProxy:
    """Lightweight proxy that logs before/after LLM invocations.

    Delegates all attributes to the wrapped LLM instance.
    """

    def __init__(self, inner: _SupportsInvoke) -> None:
        self._inner = inner

    def __getattr__(self, name: str) -> Any:  # delegate everything else
        return getattr(self._inner, name)

    def _model_name(self) -> str:
        # Best-effort model identification
        for attr in ("model", "model_name", "_model", "_model_name"):
            val = getattr(self._inner, attr, None)
            if isinstance(val, str) and val:
                return val
        return type(self._inner).__name__

    def invoke(self, input: Any, *args: Any, **kwargs: Any) -> Any:
        prompt_str = input if isinstance(input, str) else str(input)
        logger.info(
            "LLM invoke start: model=%s prompt_excerpt=%r approx_prompt_tokens=%d",
            self._model_name(),
            _excerpt(prompt_str),
            _approx_tokens(prompt_str),
        )
        import time as _time

        t0 = _time.perf_counter()
        resp = self._inner.invoke(input, *args, **kwargs)
        elapsed_ms = int((_time.perf_counter() - t0) * 1000)
        content = getattr(resp, "content", str(resp))
        content_str = content if isinstance(content, str) else str(content)
        logger.info(
            "LLM invoke end: model=%s response_excerpt=%r approx_response_tokens=%d elapsed_ms=%d",
            self._model_name(),
            _excerpt(content_str),
            _approx_tokens(content_str),
            elapsed_ms,
        )
        return resp

    async def ainvoke(self, input: Any, *args: Any, **kwargs: Any) -> Any:
        prompt_str = input if isinstance(input, str) else str(input)
        logger.info(
            "LLM ainvoke start: model=%s prompt_excerpt=%r approx_prompt_tokens=%d",
            self._model_name(),
            _excerpt(prompt_str),
            _approx_tokens(prompt_str),
        )
        import time as _time

        t0 = _time.perf_counter()
        resp = await self._inner.ainvoke(input, *args, **kwargs)
        elapsed_ms = int((_time.perf_counter() - t0) * 1000)
        content = getattr(resp, "content", str(resp))
        content_str = content if isinstance(content, str) else str(content)
        logger.info(
            "LLM ainvoke end: model=%s response_excerpt=%r approx_response_tokens=%d elapsed_ms=%d",
            self._model_name(),
            _excerpt(content_str),
            _approx_tokens(content_str),
            elapsed_ms,
        )
        return resp


def build_llm(
    model: str,
    temperature: float,
    max_tokens: int | None,
    api_key: str | None,
    api_base: str | None,
    request_timeout_s: int | None,
) -> BaseChatModel:
    """Build an LLM instance for direct, uncached use.

    Prefer `get_llm(settings)` for shared, cached instances across agents.
    """
    from langchain_openai import ChatOpenAI

    if not api_key:
        logger.warning("OPENAI_API_KEY not set; using placeholder LLM output.")

        # Create a dummy LLM-like object that returns canned responses
        class DummyLLM(BaseChatModel):
            @property
            def _llm_type(self) -> str:  # type: ignore[override]
                return "dummy"

            def _generate(self, messages: Any, stop: Any | None = None, **kwargs: Any):  # type: ignore[override]
                from langchain_core.messages import AIMessage
                from langchain_core.outputs import ChatGeneration, ChatResult

                text = "Placeholder analysis: no API key provided."
                return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])

        return LoggingLLMProxy(DummyLLM())

    llm = ChatOpenAI(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        api_key=api_key,
        timeout=request_timeout_s,
        max_retries=0,
        **({"base_url": api_base} if api_base else {}),
    )
    return LoggingLLMProxy(llm)


@lru_cache(maxsize=8)
def _build_llm_cached(
    model: str,
    temperature: float,
    max_tokens: int | None,
    api_key: str | None,
    api_base: str | None,
    request_timeout_s: int | None,
) -> BaseChatModel:
    return build_llm(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        api_key=api_key,
        api_base=api_base,
        request_timeout_s=request_timeout_s,
    )


def get_llm(settings: Settings) -> BaseChatModel:
    """Return a shared, cached LLM instance configured from `Settings`.

    Multiple agents can call this to reuse the same underlying model client.
    """
    return _build_llm_cached(
        model=settings.openai_model,
        temperature=settings.temperature,
        max_tokens=settings.max_tokens,
        api_key=settings.openai_api_key,
        api_base=settings.openai_base_url,
        request_timeout_s=settings.request_timeout_s,
    )
