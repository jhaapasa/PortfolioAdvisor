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
        try:
            # Explicitly close current figure to release Matplotlib resources
            import matplotlib.pyplot as _plt  # type: ignore

            _plt.close("all")
        except Exception:  # pragma: no cover - defensive cleanup
            pass
    return out_path


def plot_wavelet_variance_spectrum(
    output_dir: Path,
    normalized: dict[str, float],
    title: str | None = None,
    subtitle: str | None = None,
) -> Path:
    """Render normalized wavelet variance spectrum bar chart into report directory.

    Bars are expected for keys D1..D5 and S5; missing keys are skipped.
    The chart is saved as wavelet_variance_spectrum.png next to the candle chart.
    """
    # Prepare output
    report_dir = output_dir / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    out_path = report_dir / "wavelet_variance_spectrum.png"

    # Stable order and human labels with day ranges
    order = ["D1", "D2", "D3", "D4", "D5", "S5"]
    scale_labels = {
        "D1": "D1 (2–4d)",
        "D2": "D2 (4–8d)",
        "D3": "D3 (8–16d)",
        "D4": "D4 (16–32d)",
        "D5": "D5 (32–64d)",
        "S5": "S5 (>64d)",
    }
    xs = [scale_labels[k] for k in order if k in normalized]
    ys = [float(normalized[k]) for k in order if k in normalized]

    # Guard plotting with a lock
    with _plot_lock:
        try:
            import matplotlib.pyplot as plt  # type: ignore
        except Exception as exc:  # pragma: no cover - dependency missing
            raise RuntimeError("matplotlib is required for plotting") from exc

        # Match general look-and-feel (yahoo style used for candles). We'll keep a clean style.
        fig, ax = plt.subplots(figsize=(12, 4), constrained_layout=True)
        bars = ax.bar(xs, ys, color="#1f77b4")
        ax.set_ylabel("Percent of variance")
        ax.set_ylim(0, max(100.0, max(ys) * 1.15 if ys else 100.0))
        # Annotate bars with values
        for rect, val in zip(bars, ys):
            ax.annotate(
                f"{val:.1f}%",
                xy=(rect.get_x() + rect.get_width() / 2, rect.get_height()),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        # Titles
        if title or subtitle:
            if title and subtitle:
                ax.set_title(f"{title}\n{subtitle}")
            elif title:
                ax.set_title(title)
            else:
                ax.set_title(subtitle)

        ax.set_xlabel("Wavelet bands (sym4, daily log-returns)")
        # Rotate labels a bit for readability
        for tick in ax.get_xticklabels():
            tick.set_rotation(0)

        fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
        plt.close(fig)

    return out_path
