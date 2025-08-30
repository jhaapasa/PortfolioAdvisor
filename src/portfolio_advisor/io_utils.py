from __future__ import annotations

import logging
import os
from collections.abc import Iterable
from pathlib import Path

from .errors import InputOutputError

logger = logging.getLogger(__name__)


def list_input_files(input_dir: str) -> list[Path]:
    base = Path(input_dir)
    if not base.is_dir():
        msg = f"Input directory does not exist: {input_dir}"
        raise InputOutputError(msg)
    files = [p for p in base.iterdir() if p.is_file()]
    files.sort()
    logger.debug("Discovered %d input files", len(files))
    return files


def read_files_preview(paths: Iterable[Path], max_bytes: int = 2048) -> list[tuple[Path, str]]:
    preview: list[tuple[Path, str]] = []
    for p in paths:
        try:
            with p.open("rb") as fh:
                raw = fh.read(max_bytes)
            try:
                text = raw.decode("utf-8", errors="replace")
            except Exception:  # pragma: no cover - fallback
                text = str(raw[:64])
            preview.append((p, text))
        except OSError as exc:
            logger.warning("Failed reading %s: %s", p, exc)
    return preview


def write_output_text(output_dir: str, filename: str, content: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    target = Path(output_dir) / filename
    try:
        target.write_text(content, encoding="utf-8")
    except OSError as exc:
        msg = f"Failed writing output file {target}: {exc}"
        raise InputOutputError(msg) from exc
    logger.info("Wrote output: %s", target)
    return str(target)
