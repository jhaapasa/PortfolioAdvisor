"""Quantification of progressive distortion within the cone of influence (COI).

The cone of influence represents a gradual degradation of wavelet coefficient
reliability near time series boundaries, not a hard cutoff. This module provides
methods to quantify distortion as a function of distance from the boundary.
"""

from __future__ import annotations

import numpy as np
import pywt


def calculate_distortion_linear(
    distance_from_boundary: int | np.ndarray, coi_width: int
) -> float | np.ndarray:
    """Calculate distortion using simple linear distance model.

    Distortion decreases linearly from 1.0 (at boundary) to 0.0 (at COI edge).

    Parameters
    ----------
    distance_from_boundary : int | np.ndarray
        Distance in samples from the nearest time series boundary
    coi_width : int
        COI width in samples for this decomposition level

    Returns
    -------
    float | np.ndarray
        Distortion factor: 0.0 (no distortion) to 1.0 (maximum distortion)

    Examples
    --------
    >>> calculate_distortion_linear(0, 112)  # At boundary
    1.0
    >>> calculate_distortion_linear(56, 112)  # Halfway to COI
    0.5
    >>> calculate_distortion_linear(112, 112)  # At COI boundary
    0.0
    >>> calculate_distortion_linear(224, 112)  # Beyond COI
    0.0
    """
    if coi_width <= 0:
        return (
            0.0
            if isinstance(distance_from_boundary, int)
            else np.zeros_like(distance_from_boundary)
        )

    distortion = 1.0 - (distance_from_boundary / coi_width)
    return np.clip(distortion, 0.0, 1.0)


def calculate_distortion_exponential(
    distance_from_boundary: int | np.ndarray, coi_width: int, decay_factor: float = 3.0
) -> float | np.ndarray:
    """Calculate distortion using exponential decay model.

    Models progressive "fade" of reliability with smoother transition than linear.
    Uses exp(-d / (COI_width / decay_factor)) form.

    Parameters
    ----------
    distance_from_boundary : int | np.ndarray
        Distance in samples from the nearest time series boundary
    coi_width : int
        COI width in samples for this decomposition level
    decay_factor : float
        Controls decay rate (default 3.0 gives ~95% at edge, ~5% at COI)

    Returns
    -------
    float | np.ndarray
        Distortion factor: 0.0 (no distortion) to 1.0 (maximum distortion)

    Examples
    --------
    >>> calculate_distortion_exponential(0, 112)  # At boundary
    1.0
    >>> calculate_distortion_exponential(112, 112)  # At COI boundary
    0.05...  # ~5% residual distortion
    """
    if coi_width <= 0:
        return (
            0.0
            if isinstance(distance_from_boundary, int)
            else np.zeros_like(distance_from_boundary)
        )

    decay_constant = coi_width / decay_factor
    distortion = np.exp(-distance_from_boundary / decay_constant)
    return np.clip(distortion, 0.0, 1.0)


def calculate_distortion_filter_support(
    distance_from_boundary: int | np.ndarray, level: int, wavelet: str = "sym4"
) -> float | np.ndarray:
    """Calculate distortion based on actual wavelet filter support overlap.

    More accurate than simple distance models - calculates the fraction of
    filter taps operating on padded (artificial) vs real data.

    Parameters
    ----------
    distance_from_boundary : int | np.ndarray
        Distance in samples from the nearest time series boundary
    level : int
        Wavelet decomposition level
    wavelet : str
        Wavelet family name

    Returns
    -------
    float | np.ndarray
        Distortion factor: 0.0 (no distortion) to 1.0 (maximum distortion)

    Examples
    --------
    >>> calculate_distortion_filter_support(0, 5, "sym4")  # At boundary
    1.0
    >>> calculate_distortion_filter_support(200, 5, "sym4")  # Beyond filter support
    0.0
    """
    w = pywt.Wavelet(wavelet)
    filter_len = w.dec_len

    # Effective filter support at this level
    effective_support = filter_len * (2**level)

    # Fraction of filter operating on padded data
    if isinstance(distance_from_boundary, int | float):
        if distance_from_boundary >= effective_support:
            return 0.0
        return (effective_support - distance_from_boundary) / effective_support
    else:
        distortion = (effective_support - distance_from_boundary) / effective_support
        return np.clip(distortion, 0.0, 1.0)


def get_distortion_map(
    n_samples: int,
    level: int,
    wavelet: str = "sym4",
    method: str = "linear",
) -> np.ndarray:
    """Get distortion map for all samples in a time series.

    Parameters
    ----------
    n_samples : int
        Number of samples in the time series
    level : int
        Wavelet decomposition level
    wavelet : str
        Wavelet family name
    method : str
        Distortion calculation method: "linear", "exponential", or "filter_support"

    Returns
    -------
    np.ndarray
        Array of distortion values for each sample (0.0 to 1.0)

    Examples
    --------
    >>> distortion = get_distortion_map(504, 5, "sym4", "linear")
    >>> distortion.shape
    (504,)
    >>> distortion[0]  # At left boundary
    1.0
    >>> distortion[503]  # At right boundary
    1.0
    >>> distortion[252]  # At center
    0.0
    """
    w = pywt.Wavelet(wavelet)
    filter_len = w.dec_len
    coi_width = (filter_len - 1) * (2 ** (level - 1))

    # Calculate distance from nearest boundary for each sample
    distances = np.minimum(np.arange(n_samples), np.arange(n_samples - 1, -1, -1))

    if method == "linear":
        return calculate_distortion_linear(distances, coi_width)
    elif method == "exponential":
        return calculate_distortion_exponential(distances, coi_width)
    elif method == "filter_support":
        return calculate_distortion_filter_support(distances, level, wavelet)
    else:
        raise ValueError(
            f"Unknown method: {method}. Use 'linear', 'exponential', or 'filter_support'"
        )


def classify_distortion_severity(distortion: float) -> str:
    """Classify distortion severity into categories.

    Parameters
    ----------
    distortion : float
        Distortion factor (0.0 to 1.0)

    Returns
    -------
    str
        Severity classification: "minimal", "mild", "moderate", or "severe"
    """
    if distortion < 0.25:
        return "minimal"
    elif distortion < 0.50:
        return "mild"
    elif distortion < 0.75:
        return "moderate"
    else:
        return "severe"
