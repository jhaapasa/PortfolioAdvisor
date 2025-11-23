"""Ingestion agent: discovers files and extracts normalized plain text units."""

from __future__ import annotations

import csv
import logging
import mimetypes
import os
from dataclasses import dataclass
from email import policy
from email.parser import BytesParser
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

from ..io_utils import list_input_files
from ..models.market import MarketContext

logger = logging.getLogger(__name__)


# Hard cap to protect memory; can be lifted into Settings later if needed
MAX_INGEST_BYTES = 2 * 1024 * 1024  # 2 MiB

# Common OS/artifact files to ignore during ingestion
IGNORED_FILENAMES = {".DS_Store", "Thumbs.db", "desktop.ini"}


def _is_ignored_os_artifact(name: str) -> bool:
    if name in IGNORED_FILENAMES:
        return True
    if name.startswith("._"):
        return True
    # Covers macOS special Icon files like "Icon\r"
    if name.startswith("Icon") and len(name) <= 8:
        return True
    return False


@dataclass
class DocUnit:
    path: str
    name: str
    mime_type: str
    source_bytes: int
    as_text: str
    metadata: dict[str, Any]


class _HTMLTextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._chunks: list[str] = []

    def handle_data(self, data: str) -> None:  # pragma: no cover - trivial
        if data:
            self._chunks.append(data)

    def get_text(self) -> str:
        return " ".join(part.strip() for part in self._chunks if part.strip())


def _guess_mime_type(path: Path) -> str:
    # Extension overrides for common textual types
    ext = path.suffix.lower()
    if ext == ".md":
        return "text/markdown"
    if ext in {".txt", ".log"}:
        return "text/plain"
    if ext in {".htm", ".html"}:
        return "text/html"
    if ext == ".csv":
        return "text/csv"
    if ext == ".eml":
        return "message/rfc822"
    guessed, _ = mimetypes.guess_type(str(path))
    return guessed or "application/octet-stream"


def _read_bytes(path: Path, cap: int) -> tuple[bytes, bool, int]:
    size = path.stat().st_size
    truncated = False
    to_read = min(size, cap + 1)
    with path.open("rb") as fh:
        data = fh.read(to_read)
    if len(data) > cap:
        data = data[:cap]
        truncated = True
    return data, truncated, size


def _decode_text(raw: bytes) -> str:
    return raw.decode("utf-8", errors="replace")


def _extract_html_text(raw: bytes) -> str:
    text = _decode_text(raw)
    parser = _HTMLTextExtractor()
    try:
        parser.feed(text)
    except Exception:  # pragma: no cover - defensive
        return text
    return parser.get_text()


def _extract_csv_text(raw: bytes) -> str:
    text = _decode_text(raw)
    lines = text.splitlines()
    output_rows: list[str] = []
    reader = csv.reader(lines)
    for row in reader:
        output_rows.append(" | ".join(row))
    return "\n".join(output_rows)


def _extract_eml(path: Path, raw: bytes) -> tuple[str, dict[str, Any]]:
    meta: dict[str, Any] = {}
    try:
        msg = BytesParser(policy=policy.default).parsebytes(raw)
        meta["subject"] = msg["subject"]
        meta["from"] = msg["from"]
        meta["to"] = msg["to"]
        meta["date"] = msg["date"]
        # Prefer text/plain parts
        if msg.is_multipart():
            text_part = None
            html_part = None
            for part in msg.walk():
                ctype = part.get_content_type()
                if ctype == "text/plain" and text_part is None:
                    text_part = part
                elif ctype == "text/html" and html_part is None:
                    html_part = part
            if text_part is not None:
                return text_part.get_content(), meta
            if html_part is not None:
                html_content = html_part.get_content()
                html_bytes = html_content.encode("utf-8", errors="ignore")
                return _extract_html_text(html_bytes), meta
        # Not multipart
        ctype = msg.get_content_type()
        if ctype == "text/plain":
            return msg.get_content(), meta
        if ctype == "text/html":
            msg_bytes = msg.get_content().encode("utf-8", errors="ignore")
            return _extract_html_text(msg_bytes), meta
        payload = msg.get_body(preferencelist=("plain", "html"))
        if payload is not None:
            if payload.get_content_type() == "text/html":
                payload_bytes = payload.get_content().encode("utf-8", errors="ignore")
                return _extract_html_text(payload_bytes), meta
            return payload.get_content(), meta
    except Exception as exc:  # pragma: no cover - edge encodings
        logger.warning("Failed parsing eml %s: %s", path.name, exc)
    return _decode_text(raw), meta


def _to_doc_unit(path: Path) -> DocUnit | None:
    try:
        mime = _guess_mime_type(path)
        raw, truncated, size = _read_bytes(path, MAX_INGEST_BYTES)
        metadata: dict[str, Any] = {
            "truncated": truncated,
            "discovery_time": int(os.path.getmtime(path)),
        }

        if mime in {"text/plain", "text/markdown"}:
            as_text = _decode_text(raw)
        elif mime == "text/html":
            as_text = _extract_html_text(raw)
        elif mime == "text/csv":
            as_text = _extract_csv_text(raw)
        elif mime == "message/rfc822":
            as_text, eml_meta = _extract_eml(path, raw)
            metadata.update({k: v for k, v in eml_meta.items() if v})
        else:
            as_text = _decode_text(raw)
            metadata["unsupported"] = True

        return DocUnit(
            path=str(path.resolve()),
            name=path.name,
            mime_type=mime,
            source_bytes=size,
            as_text=as_text,
            metadata=metadata,
        )
    except OSError as exc:
        logger.warning("Skipping unreadable file %s: %s", path, exc)
        return None


def ingestion_node(state: dict) -> dict:
    settings = state["settings"]
    base = settings.input_dir
    logger.info("Ingestion agent start: base=%s", base)
    paths = list_input_files(base)

    docs: list[DocUnit] = []
    type_counts: dict[str, int] = {}

    for p in paths:
        if _is_ignored_os_artifact(p.name):
            logger.debug("Skipping OS artifact: %s", p.name)
            continue
        unit = _to_doc_unit(p)
        if unit is None:
            continue
        docs.append(unit)
        type_counts[unit.mime_type] = type_counts.get(unit.mime_type, 0) + 1

    logger.info("Ingestion discovered %d files", len(docs))
    for mt, count in sorted(type_counts.items()):  # pragma: no cover - logging
        logger.info("- %s: %d", mt, count)

    # Convert dataclasses to plain dicts for graph state
    raw_docs = [
        {
            "path": d.path,
            "name": d.name,
            "mime_type": d.mime_type,
            "source_bytes": d.source_bytes,
            "as_text": d.as_text,
            "metadata": d.metadata,
        }
        for d in docs
    ]

    logger.info("Ingestion agent finished: %d documents ready", len(raw_docs))

    # Initialize MarketContext if not already present
    market_context = state.get("market_context")
    if market_context is None:
        market_context = MarketContext()
        logger.info("Initialized MarketContext for market comparison analysis")

    return {**state, "raw_docs": raw_docs, "market_context": market_context}
