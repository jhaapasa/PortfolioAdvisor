from __future__ import annotations

import json
import logging
import time
from typing import Any

from ..llm import get_llm
from ..models.parsed import ParsedHolding, ParsedHoldingsResult

logger = logging.getLogger(__name__)


# Module-level prompt templates for discoverability and reuse.
PARSER_SYSTEM_PROMPT = (
    "You are a precise information extraction system for portfolio holdings. "
    "Extract a draft list of candidate holdings from the given document. "
    "Return ONLY valid JSON that conforms to the provided JSON Schema. "
    "When information is missing, set the field to null. "
    "For basket, use \"[none]\" if not part of a basket. "
    "Do not include any text outside the JSON object."
)

PARSER_USER_PROMPT_TEMPLATE = (
    "JSON Schema for output (strict):\n{schema}\n\n"
    "Instructions:\n"
    "- Identify holdings with best-effort fields: name, primary_ticker, company_name, "
    "  cusip, isin, sedol, quantity, weight, currency, account, basket, as_of, "
    "  confidence (0-1), source_doc_id.\n"
    "- Use \"[none]\" for basket if not part of any basket.\n"
    "- Prefer known identifiers (ticker, CUSIP, ISIN, SEDOL); include extras under 'identifiers'.\n"
    "- If a value is unknown, use null.\n"
    "- Ensure the root object contains 'source_doc_id' and an array 'holdings'.\n\n"
    "Document (source_doc_id={source_doc_id}):\n{doc_text}\n\n"
    "Output: A single JSON object conforming to the schema."
)

PARSER_REPAIR_PROMPT_PREFIX = (
    "The previous output did not validate against the schema. "
    "Please FIX the JSON to conform exactly to the schema and the instructions. "
    "Validation error:\n{error}\n\n"
)


def _truncate_text(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    head = text[: limit - 200]
    return head + "\n... [truncated]"


def _invoke_llm_json(llm: Any, prompt: str) -> str:
    """Call the LLM synchronously and return its textual content."""
    resp = llm.invoke(prompt)
    content = getattr(resp, "content", str(resp))
    if not isinstance(content, str):
        content = str(content)
    return content


def _validate_to_model(raw_json: str) -> ParsedHoldingsResult:
    data = json.loads(raw_json)
    return ParsedHoldingsResult.model_validate(data)


def _parse_one_doc(
    unit: dict[str, Any],
    settings: Any,
    llm: Any,
) -> ParsedHoldingsResult:
    source_doc_id = unit.get("name") or unit.get("path") or "unknown"
    doc_text = unit.get("as_text", "")
    schema = ParsedHoldingsResult.model_json_schema()
    schema_str = json.dumps(schema, ensure_ascii=False, separators=(",", ":"))

    max_chars = int(getattr(settings, "parser_max_doc_chars", 20000))
    retries = int(getattr(settings, "parser_max_retries", 2))

    base_prompt = "\n\n".join([
        PARSER_SYSTEM_PROMPT,
        PARSER_USER_PROMPT_TEMPLATE.format(
            schema=schema_str,
            source_doc_id=source_doc_id,
            doc_text=_truncate_text(doc_text, max_chars),
        ),
    ])

    error_text = None
    for attempt in range(retries + 1):
        prompt = base_prompt if error_text is None else (
            PARSER_SYSTEM_PROMPT
            + "\n\n"
            + PARSER_REPAIR_PROMPT_PREFIX.format(error=error_text)
            + PARSER_USER_PROMPT_TEMPLATE.format(
                schema=schema_str,
                source_doc_id=source_doc_id,
                doc_text=_truncate_text(doc_text, max_chars),
            )
        )
        try:
            raw = _invoke_llm_json(llm, prompt)
            result = _validate_to_model(raw)
            return result
        except Exception as exc:  # pragma: no cover - error paths validated via unit tests
            error_text = str(exc)
            logger.debug(
                "Validation/parsing failed for %s (attempt %d/%d): %s",
                source_doc_id,
                attempt + 1,
                retries + 1,
                error_text,
            )
            # brief jittered backoff before retry
            time.sleep(min(0.25 * (attempt + 1), 1.0))

    # All attempts failed; return an empty result with errors
    return ParsedHoldingsResult(
        source_doc_id=source_doc_id,
        holdings=[],
        errors=[error_text or "unknown error"],
    )


def parse_one_node(state: dict) -> dict:
    """
    Parse exactly one document unit and return incremental state updates.

    Inputs: expects `settings` and `doc` in state.
    Outputs: returns `parsed_holdings` (list of ParsedHolding as dicts) and `errors` for aggregation.
    """
    settings = state["settings"]
    unit = state.get("doc") or {}
    llm = get_llm(settings)
    result = _parse_one_doc(unit, settings, llm)

    holdings_dicts = [h.model_dump() for h in result.holdings]
    logger.debug(
        "Parsed %d holdings from %s", len(holdings_dicts), unit.get("name") or unit.get("path")
    )
    return {"parsed_holdings": holdings_dicts, "errors": list(result.errors)}


