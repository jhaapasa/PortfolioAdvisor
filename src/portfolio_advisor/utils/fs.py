from __future__ import annotations

import contextlib
import datetime as dt
import json
import os
from pathlib import Path
from typing import Any, Iterator


def write_json_atomic(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def utcnow_iso() -> str:
    # Use explicit UTC and Z-suffix for consistency across modules
    return dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat()


@contextlib.contextmanager
def dir_lock(lock_path: Path, timeout_s: int = 10) -> Iterator[None]:  # pragma: no cover
    import time

    lock_path.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.time() + timeout_s
    while True:
        try:
            os.mkdir(lock_path)
            break
        except FileExistsError:
            if time.time() > deadline:
                raise TimeoutError(f"Timed out acquiring lock: {lock_path}")
            time.sleep(0.05)
    try:
        yield
    finally:
        with contextlib.suppress(Exception):
            os.rmdir(lock_path)
