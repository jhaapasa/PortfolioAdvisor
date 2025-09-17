from __future__ import annotations

import json
from pathlib import Path

from portfolio_advisor.portfolio.persistence import (
    append_history_diffs,
    write_baskets_views,
    write_current_holdings,
    write_portfolio_header,
)


def _h(iid: str, ticker: str, basket: str | None, weight: float | None, qty: float | None):
    return {
        "instrument_id": iid,
        "primary_ticker": ticker,
        "asset_class": "stocks",
        "locale": "us",
        "mic": "XNAS",
        "symbol": ticker,
        "company_name": ticker,
        "currency": "USD",
        "as_of": "2025-09-15",
        "quantity": qty,
        "weight": weight,
        "account": "IRA",
        "basket": basket,
        "source_doc_id": "positions.csv",
    }


def test_persistence_writes_and_diffs(tmp_path: Path):
    pdir = tmp_path / "portfolio"
    pdir.mkdir()

    old_holdings = [
        _h("cid:stocks:us:XNAS:AAPL", "AAPL", "Growth Tech", 0.08, 100.0),
        _h("cid:stocks:us:XNAS:PYPL", "PYPL", "Payments", 0.02, 10.0),
    ]
    new_holdings = [
        _h("cid:stocks:us:XNAS:AAPL", "AAPL", "Growth Tech", 0.085, 120.0),  # update
        _h("cid:stocks:us:XNAS:NVDA", "NVDA", "Growth Tech", 0.04, 20.0),  # add
    ]

    # Write current snapshot and header/baskets
    h_path = write_current_holdings(str(pdir), new_holdings)
    header_path = write_portfolio_header(str(pdir), new_holdings)
    written = write_baskets_views(str(pdir), new_holdings)

    assert Path(h_path).exists()
    assert Path(header_path).exists()
    assert any(w.endswith("baskets/index.json") for w in written)
    assert (pdir / "baskets" / "growth-tech" / "positions.json").exists()

    # Append diffs
    log_path = append_history_diffs(str(pdir), old_holdings, new_holdings, as_of="2025-09-15")
    logp = Path(log_path)
    assert logp.exists()
    lines = [l for l in logp.read_text(encoding="utf-8").splitlines() if l.strip()]
    ops = [json.loads(l)["op"] for l in lines]
    assert set(ops) == {"add", "update", "remove"}


