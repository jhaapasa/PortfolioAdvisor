from __future__ import annotations

import math
from typing import Any


def compute_trailing_returns(ohlc: dict[str, Any]) -> dict[str, Any]:
    rows = ohlc.get("data", []) or []
    closes = [float(r.get("close", 0.0)) for r in rows]

    def trailing(n: int) -> float | None:
        if len(closes) <= n:
            return None
        c0 = closes[-n - 1]
        ct = closes[-1]
        if c0 == 0:
            return None
        return (ct / c0) - 1.0

    return {
        "instrument_id": ohlc.get("instrument_id"),
        "primary_ticker": ohlc.get("primary_ticker"),
        "as_of": ohlc.get("coverage", {}).get("end_date"),
        "windows": {
            "d1": trailing(1),
            "d5": trailing(5),
            "d21": trailing(21),
            "d252": trailing(252),
        },
        "method": "simple_total_return",
        "depends_on": ["primary.ohlc_daily"],
    }


def compute_volatility_annualized(ohlc: dict[str, Any], window: int = 21) -> dict[str, Any]:
    rows = ohlc.get("data", []) or []
    closes = [float(r.get("close", 0.0)) for r in rows]
    if len(closes) < window + 1:
        vol = None
    else:
        import math as _math

        returns = []
        for i in range(1, len(closes)):
            prev = closes[i - 1]
            curr = closes[i]
            if prev > 0 and curr > 0:
                returns.append(_math.log(curr / prev))
        if len(returns) >= window:
            tail = returns[-window:]
            mean = sum(tail) / len(tail)
            var = sum((x - mean) ** 2 for x in tail) / (len(tail) - 1) if len(tail) > 1 else 0.0
            vol = math.sqrt(var) * math.sqrt(252)
        else:
            vol = None
    return {
        "instrument_id": ohlc.get("instrument_id"),
        "primary_ticker": ohlc.get("primary_ticker"),
        "as_of": ohlc.get("coverage", {}).get("end_date"),
        "window": window,
        "annualization_factor": "sqrt(252)",
        "volatility": vol,
        "method": "std(log_returns) * annualization_factor",
        "depends_on": ["primary.ohlc_daily"],
    }


def compute_sma_series(ohlc: dict[str, Any], windows: list[int] | None = None) -> dict[str, Any]:
    win = windows or [20, 50, 100, 200]
    rows = ohlc.get("data", []) or []
    closes = [float(r.get("close", 0.0)) for r in rows]
    dates = [r.get("date") for r in rows]
    out_rows: list[dict[str, Any]] = []
    for i in range(len(closes)):
        row: dict[str, Any] = {"date": dates[i]}
        for w in win:
            if i + 1 >= w:
                window_vals = closes[i + 1 - w : i + 1]
                row[f"sma{w}"] = sum(window_vals) / w
        out_rows.append(row)
    start = None
    for r in out_rows:
        if any(k.startswith("sma") for k in r.keys() if k != "date"):
            start = r.get("date")
            break
    return {
        "instrument_id": ohlc.get("instrument_id"),
        "primary_ticker": ohlc.get("primary_ticker"),
        "windows": win,
        "data": out_rows,
        "coverage": {
            "start_date": start,
            "end_date": ohlc.get("coverage", {}).get("end_date"),
        },
        "depends_on": ["primary.ohlc_daily"],
    }
