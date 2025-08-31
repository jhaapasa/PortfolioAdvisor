from __future__ import annotations

import json
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
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


def _gather_threaded(
    units: list[dict[str, Any]],
    settings: Any,
) -> list[ParsedHoldingsResult]:
    max_conc = max(int(getattr(settings, "parser_max_concurrency", 4)), 1)
    max_rpm = max(int(getattr(settings, "parser_max_rpm", 60)), 1)
    min_interval = 60.0 / float(max_rpm)

    last_call_ts = 0.0
    ts_lock = threading.Lock()

    def worker(unit: dict[str, Any]) -> ParsedHoldingsResult:
        nonlocal last_call_ts
        # cooperative rate limiting across threads
        with ts_lock:
            now = time.monotonic()
            wait_for = last_call_ts + min_interval - now
            if wait_for > 0:
                time.sleep(wait_for)
            last_call_ts = time.monotonic()
        llm = get_llm(settings)
        return _parse_one_doc(unit, settings, llm)

    out: list[ParsedHoldingsResult] = []
    with ThreadPoolExecutor(max_workers=max_conc) as pool:
        future_to_idx = {pool.submit(worker, u): i for i, u in enumerate(units)}
        for fut in as_completed(future_to_idx):
            idx = future_to_idx[fut]
            try:
                res = fut.result()
            except Exception as exc:  # pragma: no cover - defensive
                unit = units[idx]
                doc_id = unit.get("name") or unit.get("path") or f"doc_{idx}"
                logger.warning("Parser task failed for %s: %s", doc_id, exc)
                out.append(
                    ParsedHoldingsResult(source_doc_id=doc_id, holdings=[], errors=[str(exc)])
                )
            else:
                out.append(res)
    return out


def parser_node(state: dict) -> dict:
    """LangGraph node: fan-out per document to parse candidate holdings, then concatenate.

    Inputs: expects `raw_docs` from ingestion and `settings`.
    Outputs: adds `parsed_holdings` (list of ParsedHolding as dicts) and appends any errors.
    """
    settings = state["settings"]
    raw_docs = state.get("raw_docs", []) or []

    if not raw_docs:
        return {**state, "parsed_holdings": []}

    # Run threaded parsing with concurrency and rate limits
    results = _gather_threaded(raw_docs, settings)

    concatenated: list[ParsedHolding] = []
    errors: list[str] = list(state.get("errors", []) or [])
    for res in results:
        concatenated.extend(res.holdings)
        errors.extend(res.errors)

    # Convert models to plain dicts for graph state
    holdings_dicts = [h.model_dump() for h in concatenated]
    new_state = {**state, "parsed_holdings": holdings_dicts, "errors": errors}
    logger.info(
        "Parser produced %d candidate holdings from %d documents",
        len(holdings_dicts),
        len(raw_docs),
    )
    return new_state


