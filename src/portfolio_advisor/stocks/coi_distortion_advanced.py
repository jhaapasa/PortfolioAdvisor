"""Advanced filter-weighted distortion quantification for wavelet COI.

This module implements sophisticated distortion metrics that consider the actual
wavelet filter coefficients and their energy distribution, not just overlap counts.
"""

from __future__ import annotations

import numpy as np
import pywt


def get_filter_energy_distribution(wavelet: str = "sym4") -> tuple[np.ndarray, np.ndarray]:
    """Get the energy distribution of a wavelet filter.

    Parameters
    ----------
    wavelet : str
        Wavelet family name

    Returns
    -------
    h_coeffs : np.ndarray
        Decomposition (highpass) filter coefficients
    energy_cumsum : np.ndarray
        Cumulative energy distribution (normalized to 1.0)
    """
    w = pywt.Wavelet(wavelet)
    # Get decomposition (highpass) filter coefficients
    h_coeffs = np.array(w.dec_hi)

    # Calculate energy (squared coefficients)
    energy = h_coeffs**2
    total_energy = np.sum(energy)

    # Cumulative energy distribution
    energy_cumsum = np.cumsum(energy) / total_energy

    return h_coeffs, energy_cumsum


def calculate_weighted_distortion(
    distance_from_boundary: int,
    level: int,
    wavelet: str = "sym4",
    signal_variance: float = 1.0,
) -> tuple[float, float, float]:
    """Calculate filter-weighted distortion metrics.

    This advanced metric weights boundary contamination by actual filter
    coefficient magnitudes, providing three measures:

    1. Energy-weighted distortion: Fraction of filter ENERGY on padding
    2. Worst-case coefficient error: Maximum possible coefficient deviation
    3. Expected coefficient error: Typical coefficient deviation

    Parameters
    ----------
    distance_from_boundary : int
        Distance in samples from the nearest time series boundary
    level : int
        Wavelet decomposition level
    wavelet : str
        Wavelet family name
    signal_variance : float
        Variance of the signal (for error estimation)

    Returns
    -------
    energy_distortion : float
        Fraction of filter energy operating on padded data (0.0 to 1.0)
    worst_case_error : float
        Maximum possible coefficient error due to padding
    expected_error : float
        Expected (RMS) coefficient error due to padding
    """
    w = pywt.Wavelet(wavelet)
    filter_len = w.dec_len
    h_coeffs = np.array(w.dec_hi)

    # At level j, filter taps are spaced 2^j apart
    spacing = 2**level
    effective_support = filter_len * spacing
    half_support = effective_support // 2

    if distance_from_boundary >= half_support:
        # No contamination
        return 0.0, 0.0, 0.0

    # Determine which filter taps operate on padded data
    # For symmetric padding at left boundary (distance d from start):
    # Filter tap k operates on position: d - k*spacing
    # If this is negative, it uses padded data

    padding_mask = np.zeros(filter_len, dtype=bool)
    filter_energies = h_coeffs**2

    for k in range(filter_len):
        sample_position = distance_from_boundary - k * spacing
        if sample_position < 0:
            padding_mask[k] = True

    # Energy-weighted distortion
    energy_on_padding = np.sum(filter_energies[padding_mask])
    total_energy = np.sum(filter_energies)
    energy_distortion = energy_on_padding / total_energy if total_energy > 0 else 0.0

    # Worst-case error estimation
    # Maximum error occurs when padded values maximally differ from "true" values
    # For reflection padding, worst case is when true signal would have opposite trend

    # Sum of absolute filter coefficients on padding (L1 norm)
    worst_case_error = np.sum(np.abs(h_coeffs[padding_mask])) * np.sqrt(signal_variance)

    # Expected error estimation (RMS)
    # Assumes padding introduces uncorrelated error with same variance
    # Error variance = sum of squared coefficients on padding
    expected_error = np.sqrt(np.sum(h_coeffs[padding_mask] ** 2)) * np.sqrt(signal_variance)

    return energy_distortion, worst_case_error, expected_error


def analyze_filter_characteristics(wavelet: str = "sym4") -> dict:
    """Analyze wavelet filter characteristics relevant to boundary effects.

    Parameters
    ----------
    wavelet : str
        Wavelet family name

    Returns
    -------
    dict
        Filter characteristics including energy concentration, effective width, etc.
    """
    w = pywt.Wavelet(wavelet)
    h_coeffs = np.array(w.dec_hi)

    # Energy distribution
    energy = h_coeffs**2
    total_energy = np.sum(energy)
    normalized_energy = energy / total_energy

    # Find energy concentration
    # How many coefficients contain X% of energy?
    cumsum_energy = np.cumsum(normalized_energy)
    energy_50_idx = np.argmax(cumsum_energy >= 0.5) + 1
    energy_90_idx = np.argmax(cumsum_energy >= 0.9) + 1
    energy_95_idx = np.argmax(cumsum_energy >= 0.95) + 1

    # Center of mass (first moment)
    indices = np.arange(len(h_coeffs))
    center_of_mass = np.sum(indices * normalized_energy)

    # Effective support (second moment - spread)
    second_moment = np.sum((indices - center_of_mass) ** 2 * normalized_energy)
    effective_width = 2 * np.sqrt(second_moment)

    return {
        "filter_length": len(h_coeffs),
        "filter_coeffs": h_coeffs.tolist(),
        "energy_distribution": normalized_energy.tolist(),
        "center_of_mass": center_of_mass,
        "effective_width": effective_width,
        "energy_50_percent_width": energy_50_idx,
        "energy_90_percent_width": energy_90_idx,
        "energy_95_percent_width": energy_95_idx,
        "max_coeff_magnitude": float(np.max(np.abs(h_coeffs))),
        "l1_norm": float(np.sum(np.abs(h_coeffs))),
        "l2_norm": float(np.sqrt(np.sum(h_coeffs**2))),
    }


def worst_case_signal_deviation(
    distance_from_boundary: int,
    level: int,
    wavelet: str = "sym4",
    signal_range: tuple[float, float] = (-1.0, 1.0),
) -> float:
    """Calculate worst-case signal deviation due to boundary effects.

    This estimates the maximum possible deviation of the reconstructed signal
    from its "true" value at a given distance from the boundary.

    Parameters
    ----------
    distance_from_boundary : int
        Distance in samples from the nearest boundary
    level : int
        Wavelet decomposition level
    wavelet : str
        Wavelet family name
    signal_range : Tuple[float, float]
        Expected range of signal values [min, max]

    Returns
    -------
    float
        Maximum possible signal deviation at this position
    """
    # For signal reconstruction, we need synthesis filters
    w = pywt.Wavelet(wavelet)
    g_coeffs = np.array(w.rec_hi)  # Synthesis highpass filter

    spacing = 2**level
    filter_len = len(g_coeffs)

    # Determine contaminated coefficients
    max_deviation = 0.0
    signal_amplitude = (signal_range[1] - signal_range[0]) / 2

    for k in range(filter_len):
        coeff_position = distance_from_boundary - k * spacing
        if coeff_position < 0:
            # This coefficient is contaminated
            # Worst case: coefficient has maximum error
            # Signal deviation = |synthesis_filter_coeff| × coefficient_error
            _, worst_error, _ = calculate_weighted_distortion(
                abs(coeff_position), level, wavelet, signal_amplitude**2
            )
            max_deviation += abs(g_coeffs[k]) * worst_error

    return max_deviation


def get_progressive_distortion_map(
    n_samples: int,
    level: int,
    wavelet: str = "sym4",
    signal_variance: float = 1.0,
) -> dict[str, np.ndarray]:
    """Get complete distortion maps for all samples in a time series.

    Parameters
    ----------
    n_samples : int
        Number of samples in the time series
    level : int
        Wavelet decomposition level
    wavelet : str
        Wavelet family name
    signal_variance : float
        Signal variance for error estimation

    Returns
    -------
    dict
        Arrays of distortion metrics for each sample:
        - 'energy_distortion': Filter energy on padding (0-1)
        - 'worst_case_error': Maximum coefficient error
        - 'expected_error': Expected (RMS) coefficient error
        - 'traditional_coi': Traditional COI metric for comparison
    """
    # Calculate for all positions
    energy_dist = np.zeros(n_samples)
    worst_error = np.zeros(n_samples)
    expected_error = np.zeros(n_samples)
    traditional_coi = np.zeros(n_samples)

    # Traditional COI for comparison
    w = pywt.Wavelet(wavelet)
    coi_width = (w.dec_len - 1) * (2 ** (level - 1))

    for i in range(n_samples):
        # Distance from nearest boundary
        distance = min(i, n_samples - 1 - i)

        # Advanced metrics
        e_dist, w_err, exp_err = calculate_weighted_distortion(
            distance, level, wavelet, signal_variance
        )
        energy_dist[i] = e_dist
        worst_error[i] = w_err
        expected_error[i] = exp_err

        # Traditional COI
        traditional_coi[i] = max(0, 1 - distance / coi_width) if coi_width > 0 else 0

    return {
        "energy_distortion": energy_dist,
        "worst_case_error": worst_error,
        "expected_error": expected_error,
        "traditional_coi": traditional_coi,
    }
