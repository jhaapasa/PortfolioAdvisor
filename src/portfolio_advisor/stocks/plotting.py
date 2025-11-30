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


def render_candlestick_ohlcv_1y(
    output_dir: Path, ohlc: dict[str, Any], extension_metadata: dict[str, Any] | None = None
) -> Path | None:
    """Render a 1-year candlestick chart with volume to output_dir/report.

    Optionally overlays boundary extension if extension_metadata is provided.

    Args:
        output_dir: Directory for output (report/ subfolder will be created)
        ohlc: OHLC dictionary with price data
        extension_metadata: Optional boundary extension metadata for overlay

    Returns:
        Path to written chart, or None if data insufficient
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
        import pandas as pd  # type: ignore
    except Exception as exc:  # pragma: no cover - dependency missing
        raise RuntimeError("mplfinance and pandas are required for plotting") from exc

    # Process and extend with boundary extension if provided
    last_real_date = None
    extension_line = None
    addplots: list[Any] = []

    if extension_metadata and isinstance(extension_metadata, dict):
        try:
            extension_data = extension_metadata.get("extension") or []
            last_real_date_str = extension_metadata.get("last_real_date")

            if extension_data and last_real_date_str:
                last_real_date = pd.to_datetime(last_real_date_str)

                # Create extension DataFrame with future dates and prices
                ext_dates = pd.DatetimeIndex(
                    [pd.to_datetime(item["date"]) for item in extension_data]
                )
                ext_prices = [item["price"] for item in extension_data]

                # Extend the tail DataFrame to include future dates
                # Create dummy OHLC rows for the extension (we'll only show the line)
                ext_df = pd.DataFrame(
                    {
                        "open": ext_prices,
                        "high": ext_prices,
                        "low": ext_prices,
                        "close": ext_prices,
                        "volume": [0] * len(ext_prices),
                    },
                    index=ext_dates,
                )

                # Combine tail with extension
                tail = pd.concat([tail, ext_df])

                # Create the extension line series (includes connection from last real point)
                if last_real_date in tail.index:
                    last_real_price = df.loc[last_real_date, "close"]  # Get from original df

                    # Build extension line: start from last real price, then follow forecast
                    extension_line = pd.Series(index=tail.index, dtype=float)
                    extension_line.loc[last_real_date] = last_real_price

                    for date, price in zip(ext_dates, ext_prices):
                        extension_line.loc[date] = price

                    # Add the extension line as an overlay (black dotted)
                    addplots.append(
                        mpf.make_addplot(
                            extension_line,
                            panel=0,
                            color="black",
                            width=2.0,
                            linestyle=":",
                            alpha=0.7,
                        )
                    )

                    _logger.debug(
                        "plotting: added boundary extension line (%d forecast points)",
                        len(ext_dates),
                    )
        except Exception:  # pragma: no cover - robust to malformed extension data
            _logger.debug("plotting: failed to create boundary extension overlay", exc_info=True)

    style = "yahoo"
    # Guard plotting with a lock to avoid global state races in Matplotlib
    with _plot_lock:
        fig = None
        try:
            # Always return figure to add markers
            if addplots:
                fig, axes = mpf.plot(  # type: ignore[assignment]
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
                fig, axes = mpf.plot(  # type: ignore[assignment]
                    tail,
                    type="candle",
                    volume=True,
                    style=style,
                    figsize=(12, 6),
                    tight_layout=True,
                    returnfig=True,
                )

            # Add boundary extension markers if present
            if last_real_date is not None and extension_line is not None and fig is not None:
                try:
                    ax = axes[0] if isinstance(axes, list | tuple) else axes

                    # Add vertical dashed line at the boundary
                    if last_real_date in tail.index:
                        ax.axvline(
                            x=last_real_date,
                            color="gray",
                            linestyle="--",
                            alpha=0.5,
                            linewidth=1.5,
                            zorder=10,
                        )

                        # Add legend for the extension line
                        import matplotlib.lines as mlines  # type: ignore

                        forecast_line = mlines.Line2D(
                            [],
                            [],
                            color="black",
                            linewidth=2,
                            linestyle=":",
                            alpha=0.7,
                            label="Forecast (boundary extension)",
                        )
                        separator_line = mlines.Line2D(
                            [],
                            [],
                            color="gray",
                            linewidth=1.5,
                            linestyle="--",
                            alpha=0.5,
                            label="Last real data",
                        )
                        ax.legend(
                            handles=[forecast_line, separator_line],
                            loc="upper left",
                            framealpha=0.9,
                            fontsize=9,
                        )
                except Exception:  # pragma: no cover - markers are optional
                    _logger.debug("Failed to add boundary extension markers", exc_info=True)

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


def _create_coi_plot_segments(
    series, coi_start: int, coi_end: int, color: str, width: float, alpha: float, panel: int = 0
) -> list:
    """Create plot segments for a wavelet series with cone of influence visualization.

    Splits the series into reliable (solid line) and COI (dotted line) regions.

    Parameters
    ----------
    series : pandas.Series
        The time series data to plot (indexed by date)
    coi_start : int
        Start index of reliable data region (0-based)
    coi_end : int
        End index of reliable data region (exclusive)
    color : str
        Line color
    width : float
        Line width
    alpha : float
        Line opacity
    panel : int
        Panel number for mplfinance

    Returns
    -------
    list
        List of mplfinance addplot objects (1-3 segments depending on COI boundaries)
    """
    try:
        import mplfinance as mpf  # type: ignore
        import numpy as np
    except Exception as exc:  # pragma: no cover - dependency missing
        raise RuntimeError("mplfinance is required for plotting") from exc

    plots = []
    series_len = len(series)

    # Ensure boundaries are within valid range
    coi_start = max(0, min(coi_start, series_len))
    coi_end = max(coi_start, min(coi_end, series_len))

    # Left COI region (dotted, more transparent)
    if coi_start > 0:
        # Create full-length series with NaN outside the segment
        left_segment = series.copy()
        left_segment.iloc[coi_start + 1 :] = np.nan
        # Only plot if we have valid data
        if left_segment.notna().sum() > 0:
            plots.append(
                mpf.make_addplot(
                    left_segment,
                    panel=panel,
                    color=color,
                    width=width,
                    alpha=alpha * 0.6,
                    linestyle=":",
                )
            )

    # Reliable data region (solid line)
    if coi_end > coi_start:
        reliable_segment = series.copy()
        reliable_segment.iloc[:coi_start] = np.nan
        reliable_segment.iloc[coi_end:] = np.nan
        if reliable_segment.notna().sum() > 0:
            plots.append(
                mpf.make_addplot(
                    reliable_segment,
                    panel=panel,
                    color=color,
                    width=width,
                    alpha=alpha,
                    linestyle="-",
                )
            )

    # Right COI region (dotted, more transparent)
    if coi_end < series_len:
        right_segment = series.copy()
        right_segment.iloc[: coi_end - 1] = np.nan
        if right_segment.notna().sum() > 0:
            plots.append(
                mpf.make_addplot(
                    right_segment,
                    panel=panel,
                    color=color,
                    width=width,
                    alpha=alpha * 0.6,
                    linestyle=":",
                )
            )

    return plots


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
    legend_labels: list[str] = []
    if recon_doc and isinstance(recon_doc, dict):
        try:
            import pandas as pd  # type: ignore
        except Exception as exc:  # pragma: no cover - dependency missing is a runtime error
            raise RuntimeError("pandas is required for plotting") from exc

        recon_all = (recon_doc.get("reconstructions") or {}) if isinstance(recon_doc, dict) else {}
        meta = recon_doc.get("metadata") or {}
        # Extract COI boundaries from metadata if available
        coi_boundaries_raw = meta.get("coi_boundaries") or {}
        coi_boundaries = (
            {k: tuple(v) for k, v in coi_boundaries_raw.items()} if coi_boundaries_raw else {}
        )

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

        # Prefer plotting all smooth S-levels S1..SJ if present; otherwise fall back.
        s_level_keys = [
            k
            for k in recon_all.keys()
            if isinstance(k, str) and k.startswith("S") and k[1:].isdigit()
        ]
        s_levels_present = sorted({int(k[1:]) for k in s_level_keys})

        if s_levels_present:
            # Plot S_J, S_{J-1}, and S_{J-2} for progressive trend detail
            max_level = max(s_levels_present)
            target_levels = [max_level]  # Always include S_J (coarsest)

            # Add S_{J-1} and S_{J-2} if available
            if max_level >= 2:
                target_levels.append(max_level - 1)  # S_{J-1}
            if max_level >= 3:
                target_levels.append(max_level - 2)  # S_{J-2}

            # Color palette for the three levels
            colors = ["#d62728", "#ff7f0e", "#2ca02c"]  # Red, Orange, Green
            widths = [2.0, 1.7, 1.4]  # Decreasing thickness
            alphas = [0.8, 0.7, 0.6]  # Decreasing opacity

            # Helper function to get day range for wavelet level
            def get_day_range(level: int) -> str:
                """Get the day range for a wavelet decomposition level assuming daily data."""
                # For MODWT, level j corresponds to periods of approximately 2^j days
                period = 2**level
                return f"~{period} days"

            for i, k in enumerate(target_levels):
                if k not in s_levels_present:
                    continue
                key = f"S{k}"
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
                    s = s.reindex(tail.index)  # Ensure alignment with main plot x-axis
                    if s.notna().sum() < max(30, int(0.1 * len(tail))):
                        continue

                    # Visual encoding with distinct colors and decreasing thickness
                    color = colors[i % len(colors)]
                    width = widths[i % len(widths)]
                    alpha = alphas[i % len(alphas)]

                    # Create legend label with day range
                    day_range = get_day_range(k)
                    legend_label = f"S_{k} ({day_range})"
                    legend_labels.append(legend_label)

                    # Use COI visualization if boundaries are available
                    if key in coi_boundaries:
                        coi_start, coi_end = coi_boundaries[key]
                        coi_plots = _create_coi_plot_segments(
                            s, coi_start, coi_end, color, width, alpha, panel=0
                        )
                        addplots.extend(coi_plots)
                    else:
                        # Fallback to simple line if no COI data
                        ap = mpf.make_addplot(
                            s,
                            panel=0,
                            color=color,
                            width=width,
                            alpha=alpha,
                        )
                        addplots.append(ap)
                except Exception:  # pragma: no cover - robust to bad data
                    _logger.debug(
                        "plotting.overlay.skip: failed to build S-level series for %s",
                        key,
                        exc_info=True,
                    )
                    continue
        else:
            # Fallback: plot a small set of representative reconstructions as before
            preferred_keys: list[str] = []
            preferred_keys.append(f"S{J}")
            preferred_keys.append(f"S{J}_D{J}")
            parts = [f"S{J}"] + [f"D{i}" for i in range(J, max(1, J - 3), -1)]
            composite = "_".join(parts)
            # preferred_keys.append(composite)

            colors = {
                f"S{J}": "#d62728",
                f"S{J}_D{J}": "#ff7f0e",
                composite: "#9467bd",
            }

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
                    s = s.reindex(tail.index)  # Ensure alignment with main plot x-axis
                    if s.notna().sum() < max(30, int(0.1 * len(tail))):
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
                # Add legend for wavelet trend overlays if we have any
                if legend_labels:
                    try:
                        # Get the main price axis (usually the first one)
                        ax = _axes[0] if isinstance(_axes, list | tuple) else _axes

                        # Create legend entries for the overlays
                        import matplotlib.lines as mlines  # type: ignore

                        legend_elements = []
                        colors = ["#d62728", "#ff7f0e", "#2ca02c"]  # Match the overlay colors

                        for i, label in enumerate(legend_labels):
                            color = colors[i % len(colors)]
                            line = mlines.Line2D([], [], color=color, linewidth=2, label=label)
                            legend_elements.append(line)

                        # Add COI indicator if boundaries are present
                        if coi_boundaries:
                            # Add a dotted line to indicate COI regions
                            coi_line = mlines.Line2D(
                                [],
                                [],
                                color="gray",
                                linewidth=1.5,
                                linestyle=":",
                                alpha=0.6,
                                label="COI region (boundary effects)",
                            )
                            legend_elements.append(coi_line)

                        # Add legend in upper left corner
                        ax.legend(
                            handles=legend_elements, loc="upper left", framealpha=0.9, fontsize=9
                        )
                    except Exception:  # pragma: no cover - legend is optional
                        _logger.debug("Failed to add legend to wavelet trends plot", exc_info=True)

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
        high = 2**k
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


def render_l1_trend_chart(
    output_dir: Path,
    ohlc: dict[str, Any],
    trend_data: dict[str, Any],
) -> Path | None:
    """Render L1 trend extraction visualization with 3 stacked panels.

    Panel 1: Price & Structural Trend (candlesticks + piecewise linear trend + knot markers)
    Panel 2: Trend Velocity (step plot showing regime state)
    Panel 3: Residuals (price - trend)

    Args:
        output_dir: Directory for output (report/ subfolder will be created)
        ohlc: OHLC dictionary with price data
        trend_data: L1 trend JSON document with trend, knots, velocity data

    Returns:
        Path to written chart, or None if data insufficient
    """
    df = _ohlc_to_dataframe(ohlc)
    if df is None or len(df) < 60:
        _logger.info(
            "plotting.l1_trend.skip: insufficient data rows=%s", 0 if df is None else len(df)
        )
        return None

    # Extract trend data
    trend_series_data = trend_data.get("trend", [])
    knot_dates = trend_data.get("knots", [])
    lambda_used = trend_data.get("lambda", 0.0)

    if not trend_series_data:
        _logger.info("plotting.l1_trend.skip: no trend data")
        return None

    try:
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
    except Exception as exc:
        raise RuntimeError("matplotlib, numpy, pandas required for plotting") from exc

    # Build trend and velocity series
    trend_dates = [pd.to_datetime(item["date"]) for item in trend_series_data]
    trend_values = [float(item["value"]) for item in trend_series_data]
    trend_series = pd.Series(trend_values, index=pd.DatetimeIndex(trend_dates), name="trend")

    # Align OHLC data with trend dates (use last 2 years like wavelet)
    target_len = min(504, len(df))
    tail = df.tail(target_len)

    # Align trend to tail dates
    common_dates = tail.index.intersection(trend_series.index)
    if len(common_dates) < 30:
        _logger.info("plotting.l1_trend.skip: insufficient overlapping dates")
        return None

    # Use aligned data
    aligned_df = tail.loc[common_dates]
    aligned_trend = trend_series.loc[common_dates]

    # Compute velocity (first difference of trend)
    velocity = aligned_trend.diff()
    velocity.iloc[0] = velocity.iloc[1] if len(velocity) > 1 else 0.0

    # Compute residuals
    residuals = aligned_df["close"] - aligned_trend

    # Identify knot positions
    knot_dates_dt = [pd.to_datetime(d) for d in knot_dates]
    knot_mask = aligned_trend.index.isin(knot_dates_dt)

    # Prepare output
    report_dir = output_dir / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    out_path = report_dir / "candle_ohlcv_2y_l1_trends.png"

    with _plot_lock:
        fig = None
        try:
            # Create figure with 3 subplots sharing x-axis
            fig, axes = plt.subplots(
                3,
                1,
                figsize=(14, 10),
                sharex=True,
                gridspec_kw={"height_ratios": [3, 1, 1]},
            )

            # ========== Panel 1: Price & Structural Trend ==========
            ax1 = axes[0]

            # Plot candlesticks (simplified as close line with high/low range)
            ax1.fill_between(
                aligned_df.index,
                aligned_df["low"],
                aligned_df["high"],
                alpha=0.2,
                color="gray",
                label="High-Low range",
            )
            ax1.plot(
                aligned_df.index,
                aligned_df["close"],
                color="gray",
                alpha=0.6,
                linewidth=0.8,
                label="Close price",
            )

            # Plot L1 trend as TRUE piecewise linear segments
            # Connect only knot points + endpoints to avoid solver micro-noise

            # Build list of segment endpoints: start, knots, end
            segment_indices = []
            if len(aligned_trend) > 0:
                segment_indices.append(0)  # First point
            for i, is_knot in enumerate(knot_mask):
                if is_knot and i not in segment_indices:
                    segment_indices.append(i)
            if len(aligned_trend) > 0 and (len(aligned_trend) - 1) not in segment_indices:
                segment_indices.append(len(aligned_trend) - 1)  # Last point

            # Extract segment points
            segment_dates = aligned_trend.index[segment_indices]
            segment_values = aligned_trend.values[segment_indices]

            # Plot the piecewise linear trend (connecting segment points only)
            ax1.plot(
                segment_dates,
                segment_values,
                color="#1f77b4",
                linewidth=2.5,
                label="L1 Trend",
            )

            # Mark knots (orange diamonds)
            knot_trend_values = aligned_trend[knot_mask]
            if len(knot_trend_values) > 0:
                ax1.scatter(
                    knot_trend_values.index,
                    knot_trend_values.values,
                    color="#ff7f0e",
                    marker="D",
                    s=60,
                    zorder=5,
                    label=f"Knots ({len(knot_trend_values)})",
                )

            ax1.set_ylabel("Price ($)")
            ax1.set_title(f"L1 Trend Structure (λ={lambda_used:.1f})")
            ax1.legend(loc="upper left", framealpha=0.9, fontsize=9)
            ax1.grid(True, alpha=0.3)

            # ========== Panel 2: Trend Velocity (Regime State) ==========
            ax2 = axes[1]

            # Create step plot with color coding
            positive_velocity = velocity.copy()
            negative_velocity = velocity.copy()
            positive_velocity[velocity < 0] = np.nan
            negative_velocity[velocity >= 0] = np.nan

            ax2.fill_between(
                velocity.index,
                0,
                positive_velocity,
                step="mid",
                alpha=0.6,
                color="#2ca02c",
                label="Uptrend",
            )
            ax2.fill_between(
                velocity.index,
                0,
                negative_velocity,
                step="mid",
                alpha=0.6,
                color="#d62728",
                label="Downtrend",
            )

            ax2.axhline(y=0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
            ax2.set_ylabel("Velocity ($/day)")
            ax2.set_title("Trend Velocity (Regime State)")
            ax2.legend(loc="upper left", framealpha=0.9, fontsize=9)
            ax2.grid(True, alpha=0.3)

            # ========== Panel 3: Residuals ==========
            ax3 = axes[2]

            ax3.bar(
                residuals.index,
                residuals.values,
                width=1.0,
                color="gray",
                alpha=0.6,
                label="Residuals",
            )

            # Add standard deviation bands
            std_residual = residuals.std()
            ax3.axhline(y=std_residual, color="#ff7f0e", linestyle="--", linewidth=1, alpha=0.7)
            ax3.axhline(y=-std_residual, color="#ff7f0e", linestyle="--", linewidth=1, alpha=0.7)
            ax3.axhline(y=0, color="black", linestyle="-", linewidth=0.8, alpha=0.5)

            ax3.set_ylabel("Residual ($)")
            ax3.set_xlabel("Date")
            ax3.set_title(f"Residuals (Noise) — σ={std_residual:.2f}")
            ax3.grid(True, alpha=0.3)

            # Format x-axis
            fig.autofmt_xdate()
            plt.tight_layout()

            fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
            _logger.info("plotting.l1_trend: saved to %s", out_path)

        except Exception:
            _logger.warning("plotting.l1_trend: failed", exc_info=True)
            return None
        finally:
            try:
                if fig is not None:
                    plt.close(fig)
                else:
                    plt.close("all")
            except Exception:
                pass

    return out_path
