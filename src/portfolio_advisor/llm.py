from __future__ import annotations

import logging
from functools import lru_cache
from typing import TYPE_CHECKING, Any

from langchain_core.language_models.chat_models import BaseChatModel

if TYPE_CHECKING:  # pragma: no cover - for typing only
    from .config import Settings

logger = logging.getLogger(__name__)


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
    # Lazy import to keep optional dependency path clean
    import httpx
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

        return DummyLLM()

    # Build a dedicated httpx client with strict total timeout and sane limits
    http_timeout = None
    if request_timeout_s is not None:
        try:
            http_timeout = httpx.Timeout(float(request_timeout_s))
        except Exception:  # pragma: no cover - defensive
            http_timeout = httpx.Timeout(60.0)
    http_client = httpx.Client(timeout=http_timeout) if http_timeout is not None else None

    return ChatOpenAI(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        api_key=api_key,
        timeout=request_timeout_s,
        max_retries=0,
        **({"http_client": http_client} if http_client is not None else {}),
        **({"base_url": api_base} if api_base else {}),
    )


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
