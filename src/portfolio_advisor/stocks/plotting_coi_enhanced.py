"""Enhanced COI plotting with confidence bands based on filter-weighted error estimates."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def create_coi_plot_segments_with_confidence(
    series: pd.Series,
    coi_start: int,
    coi_end: int,
    level: int,
    wavelet: str,
    color: str,
    width: float,
    alpha: float,
    panel: int = 0,
    signal_std: float | None = None,
    confidence_alpha: float = 0.2,
) -> list[dict[str, Any]]:
    """Create plot segments for a wavelet series with COI visualization and confidence bands.

    Splits the series into reliable (solid line) and COI (dotted line with confidence
    bands) regions.

    Parameters
    ----------
    series : pandas.Series
        The time series data to plot (indexed by date)
    coi_start : int
        Start index of reliable data region (0-based)
    coi_end : int
        End index of reliable data region (exclusive)
    level : int
        Wavelet decomposition level for error calculation
    wavelet : str
        Wavelet family name
    color : str
        Line color
    width : float
        Line width
    alpha : float
        Line opacity
    panel : int
        Panel number for mplfinance
    signal_std : float, optional
        Standard deviation of the signal for error scaling. If None, estimated from data.
    confidence_alpha : float
        Alpha value for confidence band shading

    Returns
    -------
    list
        List of plot dictionaries containing both line plots and confidence bands
    """
    try:
        import mplfinance as mpf  # type: ignore

        from portfolio_advisor.stocks.coi_distortion_advanced import calculate_weighted_distortion
    except Exception as exc:  # pragma: no cover - dependency missing
        raise RuntimeError("mplfinance and COI modules are required for plotting") from exc

    plots = []
    series_len = len(series)

    # Ensure boundaries are within valid range
    coi_start = max(0, min(coi_start, series_len))
    coi_end = max(coi_start, min(coi_end, series_len))

    # Estimate signal standard deviation if not provided
    if signal_std is None:
        # Use log returns for financial data
        log_returns = np.log(series / series.shift(1)).dropna()
        signal_std = log_returns.std() if len(log_returns) > 20 else 0.015  # Default 1.5%

    # Calculate confidence bands for COI regions
    def get_confidence_bounds(
        segment_series: pd.Series, start_idx: int, is_left_coi: bool
    ) -> tuple[pd.Series, pd.Series]:
        """Calculate upper and lower confidence bounds for a COI segment."""
        bounds_upper = segment_series.copy()
        bounds_lower = segment_series.copy()

        for i in range(len(segment_series)):
            if segment_series.iloc[i] == segment_series.iloc[i]:  # Check for NaN
                # Calculate distance from boundary
                if is_left_coi:
                    distance_from_boundary = start_idx + i
                else:
                    distance_from_boundary = series_len - (start_idx + i)

                # Get worst-case error for this distance
                _, worst_err, _ = calculate_weighted_distortion(
                    distance_from_boundary, level, wavelet, signal_std**2
                )

                # Convert log-space error to price-space bounds
                # For log-normal: upper = value * exp(error), lower = value * exp(-error)
                current_value = segment_series.iloc[i]
                if worst_err > 0:
                    bounds_upper.iloc[i] = current_value * np.exp(worst_err)
                    bounds_lower.iloc[i] = current_value * np.exp(-worst_err)

        return bounds_upper, bounds_lower

    # Left COI region (dotted with confidence bands)
    if coi_start > 0:
        left_segment = series.copy()
        left_segment.iloc[coi_start + 1 :] = np.nan

        if left_segment.notna().sum() > 0:
            # Main dotted line
            plots.append(
                {
                    "type": "line",
                    "data": mpf.make_addplot(
                        left_segment,
                        panel=panel,
                        color=color,
                        width=width,
                        alpha=alpha * 0.6,
                        linestyle=":",
                    ),
                }
            )

            # Calculate and add confidence bands
            bounds_upper, bounds_lower = get_confidence_bounds(left_segment, 0, is_left_coi=True)

            # Create fill_between data
            # Note: mplfinance doesn't directly support fill_between, so we'll return the bounds
            # for custom processing
            plots.append(
                {
                    "type": "confidence_band",
                    "upper": bounds_upper,
                    "lower": bounds_lower,
                    "color": color,
                    "alpha": confidence_alpha,
                    "panel": panel,
                }
            )

    # Reliable data region (solid line, no confidence bands)
    if coi_end > coi_start:
        reliable_segment = series.copy()
        reliable_segment.iloc[:coi_start] = np.nan
        reliable_segment.iloc[coi_end:] = np.nan

        if reliable_segment.notna().sum() > 0:
            plots.append(
                {
                    "type": "line",
                    "data": mpf.make_addplot(
                        reliable_segment,
                        panel=panel,
                        color=color,
                        width=width,
                        alpha=alpha,
                        linestyle="-",
                    ),
                }
            )

    # Right COI region (dotted with confidence bands)
    if coi_end < series_len:
        right_segment = series.copy()
        right_segment.iloc[: coi_end - 1] = np.nan

        if right_segment.notna().sum() > 0:
            # Main dotted line
            plots.append(
                {
                    "type": "line",
                    "data": mpf.make_addplot(
                        right_segment,
                        panel=panel,
                        color=color,
                        width=width,
                        alpha=alpha * 0.6,
                        linestyle=":",
                    ),
                }
            )

            # Calculate and add confidence bands
            bounds_upper, bounds_lower = get_confidence_bounds(
                right_segment, coi_end - 1, is_left_coi=False
            )

            plots.append(
                {
                    "type": "confidence_band",
                    "upper": bounds_upper,
                    "lower": bounds_lower,
                    "color": color,
                    "alpha": confidence_alpha,
                    "panel": panel,
                }
            )

    return plots


def apply_confidence_bands_to_axes(ax, plots_with_bands: list[dict], df_index) -> None:
    """Apply confidence bands to matplotlib axes after mplfinance plot creation.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to add confidence bands to
    plots_with_bands : list[dict]
        List of plot dictionaries containing confidence band data
    df_index : pandas.DatetimeIndex
        The datetime index of the main dataframe for x-axis alignment
    """
    for plot_dict in plots_with_bands:
        if plot_dict["type"] == "confidence_band":
            upper = plot_dict["upper"]
            lower = plot_dict["lower"]

            # Align with the main dataframe index
            upper_aligned = upper.reindex(df_index)
            lower_aligned = lower.reindex(df_index)

            # Find valid (non-NaN) regions
            valid_mask = upper_aligned.notna() & lower_aligned.notna()
            if valid_mask.sum() > 0:
                # Get x positions (mplfinance uses integer positions)
                x_positions = np.arange(len(df_index))

                # Fill between the bounds
                ax.fill_between(
                    x_positions[valid_mask],
                    lower_aligned[valid_mask].values,
                    upper_aligned[valid_mask].values,
                    color=plot_dict["color"],
                    alpha=plot_dict["alpha"],
                    linewidth=0,
                )
