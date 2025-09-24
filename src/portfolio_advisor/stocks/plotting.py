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
    # Guard plotting with a lock to avoid global state races in Matplotlib
    with _plot_lock:
        mpf.plot(
            tail,
            type="candle",
            volume=True,
            style=style,
            figsize=(12, 6),
            tight_layout=True,
            savefig=dict(fname=str(out_path), dpi=150, bbox_inches="tight"),
        )
        try:
            # Explicitly close current figure to release Matplotlib resources
            import matplotlib.pyplot as _plt  # type: ignore

            _plt.close("all")
        except Exception:  # pragma: no cover - defensive cleanup
            pass
    return out_path


def render_candlestick_ohlcv_2y_wavelet_trends(
    output_dir: Path, ohlc: dict[str, Any], recon_doc: dict[str, Any] | None
) -> Path | None:
    """Render a 2-year candlestick chart with volume and wavelet trend overlays.

    Overlays are reconstructed price series from wavelet bands contained in
    recon_doc["reconstructions"], if available. Missing or malformed overlays are
    ignored and the function gracefully falls back to plain candles.

    Returns the written path, or None if data insufficient.
    """
    df = _ohlc_to_dataframe(ohlc)
    if df is None or len(df) < 180:  # require ~6 months minimum
        _logger.info("plotting.skip: insufficient data rows=%s", 0 if df is None else len(df))
        return None

    # Slice the last ~504 trading days (approx. 2 years)
    tail = df.tail(504)
    report_dir = output_dir / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    out_path = report_dir / "candle_ohlcv_2y_wavelet_trends.png"

    try:
        import mplfinance as mpf  # type: ignore
    except Exception as exc:  # pragma: no cover - dependency missing
        raise RuntimeError("mplfinance is required for plotting") from exc

    # Prepare overlays when reconstruction doc is provided
    addplots: list[Any] = []
    if recon_doc and isinstance(recon_doc, dict):
        try:
            import pandas as pd  # type: ignore
        except Exception as exc:  # pragma: no cover - dependency missing is a runtime error
            raise RuntimeError("pandas is required for plotting") from exc

        recon_all = (recon_doc.get("reconstructions") or {}) if isinstance(recon_doc, dict) else {}
        meta = recon_doc.get("metadata") or {}
        # Determine J from metadata or infer from available keys (highest S?)
        J = None
        try:
            J = int(meta.get("level")) if meta.get("level") is not None else None
        except Exception:
            J = None
        if J is None:
            try:
                skeys = [
                    int(k[1:])
                    for k in recon_all.keys()
                    if isinstance(k, str) and k.startswith("S") and k[1:].isdigit()
                ]
                J = max(skeys) if skeys else 5
            except Exception:
                J = 5

        # Preferred keys in increasing detail; only plot those present to avoid clutter.
        # Build: S{J}, S{J}_D{J}, and a mid-level composite like S{J}_D{J}_..._D{max(2, J-3)}
        preferred_keys: list[str] = []
        preferred_keys.append(f"S{J}")
        preferred_keys.append(f"S{J}_D{J}")
        # Build a composite down to max(2, J-3)
        parts = [f"S{J}"] + [f"D{i}" for i in range(J, max(1, J - 3), -1)]
        composite = "_".join(parts)
#        preferred_keys.append(composite)

        # Color palette (consistent with matplotlib default palette order)
        # Assign colors deterministically based on key structure
        colors = {
            f"S{J}": "#d62728",
            f"S{J}_D{J}": "#ff7f0e",
            composite: "#9467bd",
        }

        # Build at most 4 overlays (to keep the chart readable)
        overlays_built = 0
        for key in preferred_keys:
            if key not in recon_all:
                continue
            series_rows = recon_all.get(key) or []
            if not series_rows:
                continue
            try:
                s_df = pd.DataFrame(series_rows)
                if "date" not in s_df.columns or "value" not in s_df.columns:
                    continue
                s_df["date"] = pd.to_datetime(s_df["date"], errors="coerce")
                s_df = s_df.set_index("date").sort_index()
                s = pd.to_numeric(s_df["value"], errors="coerce")
                # Align to tail index; restrict to index intersection
                s = s.reindex(tail.index).dropna(how="all")
                if s.notna().sum() < max(30, int(0.1 * len(tail))):
                    # Skip nearly-empty overlays
                    continue
                ap = mpf.make_addplot(
                    s,
                    panel=0,
                    color=colors.get(key, "#333333"),
                    width=1.2,
                )
                addplots.append(ap)
                overlays_built += 1
                if overlays_built >= 4:
                    break
            except Exception:  # pragma: no cover - robust to bad data
                _logger.debug(
                    "plotting.overlay.skip: failed to build series for %s", key, exc_info=True
                )
                continue

    style = "yahoo"
    # Guard plotting with a lock to avoid global state races in Matplotlib
    with _plot_lock:
        fig = None
        try:
            if addplots:
                fig, _axes = mpf.plot(  # type: ignore[assignment]
                    tail,
                    type="candle",
                    volume=True,
                    style=style,
                    addplot=addplots,
                    figsize=(12, 6),
                    tight_layout=True,
                    returnfig=True,
                )
            else:
                fig, _axes = mpf.plot(  # type: ignore[assignment]
                    tail,
                    type="candle",
                    volume=True,
                    style=style,
                    figsize=(12, 6),
                    tight_layout=True,
                    returnfig=True,
                )
            if fig is not None:
                fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
        finally:
            try:
                # Explicitly close current figure to release Matplotlib resources
                if fig is not None:
                    import matplotlib.pyplot as _plt  # type: ignore

                    _plt.close(fig)
                else:
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

    # Determine J from provided keys; default to 5 if ambiguous
    max_detail = 0
    for k in normalized.keys():
        if isinstance(k, str) and k.startswith("D") and k[1:].isdigit():
            try:
                max_detail = max(max_detail, int(k[1:]))
            except Exception:
                pass
    s_level = 0
    for k in normalized.keys():
        if isinstance(k, str) and k.startswith("S") and k[1:].isdigit():
            try:
                s_level = max(s_level, int(k[1:]))
            except Exception:
                pass
    J = max(max_detail, s_level)
    if J <= 0:
        J = 5
    order = [f"D{i}" for i in range(1, J + 1)] + [f"S{J}"]

    # Human labels: Dk ~ [2^{k-1}, 2^{k}] days, S_J > 2^{J}
    def _range_label(k: int) -> str:
        low = 2 ** (k - 1)
        high = 2 ** k
        return f"D{k} ({low}–{high}d)"

    scale_labels = {f"D{i}": _range_label(i) for i in range(1, J + 1)}
    scale_labels[f"S{J}"] = f"S{J} (> {2 ** J}d)"
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
