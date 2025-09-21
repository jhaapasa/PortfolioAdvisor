from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any

_logger = logging.getLogger(__name__)

# Use a non-interactive backend suitable for headless/threaded execution
try:  # pragma: no cover - backend selection is environment dependent
    import matplotlib  # type: ignore

    matplotlib.use("Agg", force=True)
except Exception:  # pragma: no cover - defensive
    pass

_plot_lock = threading.Lock()


def _ohlc_to_dataframe(ohlc: dict[str, Any]):
    """Convert primary OHLC dict to a pandas DataFrame indexed by date.

    Expects keys: data: list[{date, open, high, low, close, volume}], coverage.end_date.
    """
    try:
        import pandas as pd  # type: ignore
    except Exception as exc:  # pragma: no cover - dependency missing is a runtime error
        raise RuntimeError("pandas is required for plotting") from exc

    rows = ohlc.get("data", []) or []
    if not rows:
        return None
    df = pd.DataFrame(rows)
    # Normalize columns and index
    expected = ["date", "open", "high", "low", "close", "volume"]
    for col in expected:
        if col not in df.columns:
            df[col] = None
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.set_index("date").sort_index()
    # Ensure numeric types
    for c in ("open", "high", "low", "close"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0).astype(int)
    return df


def render_candlestick_ohlcv_1y(output_dir: Path, ohlc: dict[str, Any]) -> Path | None:
    """Render a 1-year candlestick chart with volume to output_dir/report.

    Returns the written path, or None if data insufficient.
    """
    df = _ohlc_to_dataframe(ohlc)
    if df is None or len(df) < 180:  # require ~6 months minimum
        _logger.info("plotting.skip: insufficient data rows=%s", 0 if df is None else len(df))
        return None

    # Slice the last 252 trading days when available
    tail = df.tail(252)
    report_dir = output_dir / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    out_path = report_dir / "candle_ohlcv_1y.png"

    try:
        import mplfinance as mpf  # type: ignore
    except Exception as exc:  # pragma: no cover - dependency missing
        raise RuntimeError("mplfinance is required for plotting") from exc

    style = "yahoo"
    savefig = dict(fname=str(out_path), dpi=150, bbox_inches="tight")
    # Guard plotting with a lock to avoid global state races in Matplotlib
    with _plot_lock:
        mpf.plot(
            tail,
            type="candle",
            volume=True,
            style=style,
            figsize=(12, 6),
            tight_layout=True,
            savefig=savefig,
        )
    return out_path
