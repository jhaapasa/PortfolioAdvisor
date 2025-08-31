from __future__ import annotations

import logging
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel

logger = logging.getLogger(__name__)


def build_llm(
    model: str,
    temperature: float,
    max_tokens: int | None,
    api_key: str | None,
    api_base: str | None,
) -> BaseChatModel:
    # Lazy import to keep optional dependency path clean
    from langchain_google_genai import ChatGoogleGenerativeAI

    if not api_key:
        logger.warning("GEMINI_API_KEY not set; analyst will use placeholder output.")

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

    return ChatGoogleGenerativeAI(
        model=model,
        temperature=temperature,
        max_output_tokens=max_tokens,
        google_api_key=api_key,
        **({"base_url": api_base} if api_base else {}),
    )


def analyst_node(state: dict) -> dict:
    settings = state["settings"]
    llm = build_llm(
        model=settings.gemini_model,
        temperature=settings.temperature,
        max_tokens=settings.max_tokens,
        api_key=settings.gemini_api_key,
        api_base=settings.gemini_api_base,
    )

    # In the MVP, produce a brief summary based on ingested docs and plan
    raw_docs = state.get("raw_docs", []) or []
    file_descriptions = [
        f"{d.get('name','')} ({d.get('mime_type','unknown')}, {d.get('source_bytes',0)} bytes)"
        for d in raw_docs
    ]
    plan = state.get("plan", {})
    prompt = (
        "You are a portfolio analyst. Given the input files and a simple plan, produce a short "
        "placeholder analysis suitable for a markdown report."
        f"\nFiles: {', '.join(file_descriptions) if file_descriptions else 'none'}"
        f"\nPlan steps: {', '.join(plan.get('steps', []))}"
    )

    try:
        response = llm.invoke(prompt)
        content = getattr(response, "content", str(response))
    except Exception as exc:  # pragma: no cover - network/LLM errors vary
        logger.warning("LLM call failed, using fallback: %s", exc)
        content = "Placeholder analysis due to LLM error."

    return {**state, "analysis": content}
