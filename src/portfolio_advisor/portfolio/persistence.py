from __future__ import annotations

import contextlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Reuse atomic write and helper patterns from stocks/db.py within the package
from ..utils.fs import utcnow_iso, write_json_atomic
from ..utils.slug import slugify


@dataclass
class PortfolioPaths:
    root: Path

    def holdings_json(self) -> Path:
        return self.root / "holdings.json"

    def portfolio_json(self) -> Path:
        return self.root / "portfolio.json"

    def baskets_dir(self) -> Path:
        return self.root / "baskets"

    def baskets_index_json(self) -> Path:
        return self.baskets_dir() / "index.json"

    def basket_dir(self, slug: str) -> Path:
        return self.baskets_dir() / slug

    def basket_positions_json(self, slug: str) -> Path:
        return self.basket_dir(slug) / "positions.json"

    def basket_meta_json(self, slug: str) -> Path:
        return self.basket_dir(slug) / "basket.json"

    def history_dir(self) -> Path:
        return self.root / "history"

    def changes_log(self) -> Path:
        return self.history_dir() / "portfolio_changes.jsonl"

    def lock_dir(self) -> Path:
        return self.root / ".lock"


def _read_json(path: Path) -> Any:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _ensure_dirs(paths: PortfolioPaths) -> None:
    paths.root.mkdir(parents=True, exist_ok=True)
    paths.baskets_dir().mkdir(parents=True, exist_ok=True)
    paths.history_dir().mkdir(parents=True, exist_ok=True)


def _slugify(label: str) -> str:
    return slugify(label)


def _basket_id_from_slug(slug: str) -> str:
    return slug.replace("-", "_")


def _stable_holding_view(h: dict[str, Any]) -> dict[str, Any]:
    return {
        "instrument_id": h.get("instrument_id"),
        "primary_ticker": h.get("primary_ticker"),
        "asset_class": h.get("asset_class"),
        "locale": h.get("locale"),
        "mic": h.get("mic"),
        "symbol": h.get("symbol"),
        "company_name": h.get("company_name"),
        "currency": h.get("currency"),
        "as_of": h.get("as_of"),
        "quantity": h.get("quantity"),
        "weight": h.get("weight"),
        "account": h.get("account"),
        "basket": h.get("basket"),
        "source_doc_id": h.get("source_doc_id"),
    }


def write_current_holdings(portfolio_dir: str, holdings: list[dict[str, Any]]) -> str:
    paths = PortfolioPaths(root=Path(portfolio_dir))
    _ensure_dirs(paths)
    # Use stable subset and sort by instrument_id
    subset = [_stable_holding_view(h) for h in holdings]
    subset.sort(key=lambda x: str(x.get("instrument_id") or ""))
    write_json_atomic(paths.holdings_json(), subset)
    return str(paths.holdings_json())


def _derive_baskets(holdings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[str, dict[str, Any]] = {}
    for h in holdings:
        label = str(h.get("basket") or "[none]")
        if label.strip().lower() in {"[none]", "none", ""}:
            # Skip non-basket items for explicit baskets list
            continue
        slug = _slugify(label)
        bid = _basket_id_from_slug(slug)
        b = buckets.setdefault(slug, {"id": bid, "label": label, "slug": slug, "size": 0})
        b["size"] += 1
    # Add last_updated placeholder
    now = utcnow_iso()
    out = []
    for b in buckets.values():
        out.append({**b, "last_updated": now})
    out.sort(key=lambda x: x["slug"])  # deterministic
    return out


def write_portfolio_header(portfolio_dir: str, holdings: list[dict[str, Any]]) -> str:
    paths = PortfolioPaths(root=Path(portfolio_dir))
    _ensure_dirs(paths)
    baskets = _derive_baskets(holdings)
    # as_of: prefer max non-empty as_of across holdings
    dates = [str(h.get("as_of")) for h in holdings if h.get("as_of")]
    as_of = max(dates) if dates else None
    header = {
        "id": "default",
        "as_of": as_of,
        "valuation_ccy": "USD",
        "num_holdings": len(holdings),
        "num_baskets": len(baskets),
        "baskets": baskets,
    }
    write_json_atomic(paths.portfolio_json(), header)
    return str(paths.portfolio_json())


def write_baskets_views(portfolio_dir: str, holdings: list[dict[str, Any]]) -> list[str]:
    paths = PortfolioPaths(root=Path(portfolio_dir))
    _ensure_dirs(paths)
    baskets = _derive_baskets(holdings)
    write_json_atomic(paths.baskets_index_json(), baskets)
    written: list[str] = [str(paths.baskets_index_json())]
    # Write positions per basket
    for b in baskets:
        slug = b["slug"]
        # denormalized subset for this basket (match by slug for robust normalization)
        positions = [
            _stable_holding_view(h)
            for h in holdings
            if _slugify(str(h.get("basket") or "")) == slug
        ]
        positions.sort(key=lambda x: str(x.get("instrument_id") or ""))
        write_json_atomic(paths.basket_positions_json(slug), positions)
        # Write basket meta if not present
        if not paths.basket_meta_json(slug).exists():
            meta = {"id": b["id"], "label": b["label"], "slug": slug}
            write_json_atomic(paths.basket_meta_json(slug), meta)
        written.append(str(paths.basket_positions_json(slug)))
    return written


def _index_by_instrument(holdings: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    m: dict[str, dict[str, Any]] = {}
    for h in holdings:
        iid = str(h.get("instrument_id") or "")
        if iid:
            m[iid] = h
    return m


def append_history_diffs(
    portfolio_dir: str,
    old_holdings: list[dict[str, Any]] | None,
    new_holdings: list[dict[str, Any]],
    as_of: str | None,
) -> str:
    paths = PortfolioPaths(root=Path(portfolio_dir))
    _ensure_dirs(paths)

    old_map = _index_by_instrument(old_holdings or [])
    new_map = _index_by_instrument(new_holdings)

    ts = utcnow_iso()
    # Compute diffs
    lines: list[str] = []
    # Added or updated
    for iid, nxt in new_map.items():
        prv = old_map.get(iid)
        if prv is None:
            entry = {
                "ts": ts,
                "as_of": as_of,
                "op": "add",
                "instrument_id": iid,
                "primary_ticker": nxt.get("primary_ticker"),
                "next": {
                    "weight": nxt.get("weight"),
                    "quantity": nxt.get("quantity"),
                    "basket": nxt.get("basket"),
                    "account": nxt.get("account"),
                },
            }
            lines.append(json.dumps(entry, separators=(",", ":")))
        else:
            # Check meaningful fields
            changed: dict[str, Any] = {}
            for key in ("quantity", "weight", "basket", "account"):
                if prv.get(key) != nxt.get(key):
                    changed[key] = (prv.get(key), nxt.get(key))
            if changed:
                entry = {
                    "ts": ts,
                    "as_of": as_of,
                    "op": "update",
                    "instrument_id": iid,
                    "primary_ticker": nxt.get("primary_ticker"),
                    "prev": {k: v[0] for k, v in changed.items()},
                    "next": {k: v[1] for k, v in changed.items()},
                }
                lines.append(json.dumps(entry, separators=(",", ":")))

    # Removed
    for iid, prv in old_map.items():
        if iid not in new_map:
            entry = {
                "ts": ts,
                "as_of": as_of,
                "op": "remove",
                "instrument_id": iid,
                "primary_ticker": prv.get("primary_ticker"),
            }
            lines.append(json.dumps(entry, separators=(",", ":")))

    # Append lines atomically under a simple file lock
    paths.history_dir().mkdir(parents=True, exist_ok=True)
    lock_dir = paths.history_dir() / ".lock"
    # Simple lock using directory creation (single-process mostly)
    with _simple_lock(lock_dir):
        if lines:
            tmp = paths.changes_log().with_suffix(paths.changes_log().suffix + ".tmp")
            # Write existing content + new lines to tmp, then replace
            existing = ""
            if paths.changes_log().exists():
                with paths.changes_log().open("r", encoding="utf-8") as fh:
                    existing = fh.read()
            with tmp.open("w", encoding="utf-8") as fh:
                if existing:
                    fh.write(existing)
                    if not existing.endswith("\n"):
                        fh.write("\n")
                fh.write("\n".join(lines))
                fh.write("\n")
            os.replace(tmp, paths.changes_log())
    return str(paths.changes_log())


@contextlib.contextmanager
def _simple_lock(lock_path: Path):  # pragma: no cover - straightforward
    import time

    lock_path.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.time() + 10
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
