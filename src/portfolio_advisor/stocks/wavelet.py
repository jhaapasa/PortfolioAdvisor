from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pywt


@dataclass
class WaveletTransformResult:
    # Details ordered from level 1..J
    details: list[np.ndarray]
    scaling: np.ndarray
    # Dates for the aligned analysis window (ISO strings)
    dates: list[str]
    # Transform metadata
    meta: dict[str, Any]


def _compute_log_returns(closes: Iterable[float]) -> np.ndarray:
    arr = np.asarray(list(closes), dtype=float)
    good = np.isfinite(arr) & (arr > 0)
    if not np.all(good):
        # Replace non-positive or non-finite with forward fill of last valid
        for i in range(arr.size):
            if not (arr[i] > 0 and math.isfinite(arr[i])):
                arr[i] = arr[i - 1] if i > 0 else np.nan
        # backfill head if needed
        if not (arr[0] > 0 and math.isfinite(arr[0])):
            first_valid = next((x for x in arr if x > 0 and math.isfinite(x)), np.nan)
            arr[0] = first_valid
            for i in range(1, arr.size):
                if not (arr[i] > 0 and math.isfinite(arr[i])):
                    arr[i] = arr[i - 1]
    # compute log returns
    rets = np.diff(np.log(arr), prepend=np.nan)
    return rets[1:]


def _tail_pad_to_power_of_two_length(x: np.ndarray, power: int) -> tuple[np.ndarray, int]:
    # Ensure len(x) % 2^power == 0 by symmetric tail padding
    m = 1 << power
    n = int(x.size)
    pad = (m - (n % m)) % m
    if pad == 0:
        return x, 0
    # symmetric tail padding per pywt pad modes
    padded = pywt.pad(x, pad_widths=(0, pad), mode="symmetric")
    return padded, pad


def compute_modwt_logreturns(
    dates: list[str],
    closes: list[float],
    level: int = 5,
    wavelet: str = "sym4",
) -> WaveletTransformResult:
    if len(closes) < (level + 34):
        raise ValueError("Insufficient close prices for stable SWT at requested level")

    # Compute log-returns aligned to dates (len = len(closes) - 1)
    log_returns = _compute_log_returns(closes)
    ret_dates = dates[1:]

    # Tail mirror-pad to multiple of 2^level
    padded, pad_len = _tail_pad_to_power_of_two_length(log_returns, power=level)

    # SWT with energy normalization for variance properties
    coeffs = pywt.swt(data=padded, wavelet=wavelet, level=level, trim_approx=False, norm=True)
    # coeffs: list of (cA_j, cD_j) for j=1..level
    details = [pair[1] for pair in coeffs]
    scaling = coeffs[-1][0]

    # Remove tail padding
    if pad_len:
        details = [d[:-pad_len] for d in details]
        scaling = scaling[:-pad_len]

    # Align to last 2 years (approx 504 trading days); keep min(len, 504)
    target_len = min(504, len(ret_dates))
    start_idx = len(ret_dates) - target_len
    aligned_dates = ret_dates[start_idx:]
    details = [d[start_idx:] for d in details]
    scaling = scaling[start_idx:]

    meta = {
        "wavelet": wavelet,
        "level": level,
        "series": "log_returns",
        "padding": "symmetric_tail_only",
        "norm": True,
        "trim_approx": False,
        "original_returns_length": int(len(ret_dates)),
        "padded_length": int(padded.size),
        "pad_len": int(pad_len),
        "analysis_window_length": int(target_len),
    }
    return WaveletTransformResult(
        details=details,
        scaling=scaling,
        dates=list(aligned_dates),
        meta=meta,
    )


def compute_modwt_logprice(
    dates: list[str],
    closes: list[float],
    level: int = 5,
    wavelet: str = "sym4",
) -> WaveletTransformResult:
    if len(closes) < (level + 34):
        raise ValueError("Insufficient close prices for stable SWT at requested level")

    # Sanitize close prices: forward/back-fill non-finite or non-positive
    arr = np.asarray(list(closes), dtype=float)
    good = np.isfinite(arr) & (arr > 0)
    if not np.all(good):
        for i in range(arr.size):
            if not (arr[i] > 0 and math.isfinite(arr[i])):
                arr[i] = arr[i - 1] if i > 0 else np.nan
        if not (arr[0] > 0 and math.isfinite(arr[0])):
            first_valid = next((x for x in arr if x > 0 and math.isfinite(x)), np.nan)
            arr[0] = first_valid
            for i in range(1, arr.size):
                if not (arr[i] > 0 and math.isfinite(arr[i])):
                    arr[i] = arr[i - 1]

    log_price = np.log(arr)

    # Tail mirror-pad to multiple of 2^level
    padded, pad_len = _tail_pad_to_power_of_two_length(log_price, power=level)

    # SWT with energy normalization; keep all approximation levels for stable return type
    coeffs = pywt.swt(data=padded, wavelet=wavelet, level=level, trim_approx=False, norm=True)
    # coeffs: list of (cA_j, cD_j) for j=1..level
    details = [pair[1] for pair in coeffs]
    scaling = coeffs[-1][0]

    # Remove tail padding
    if pad_len:
        details = [d[:-pad_len] for d in details]
        scaling = scaling[:-pad_len]

    # Align to last 2 years (approx 504 trading days); keep min(len, 504)
    target_len = min(504, len(dates))
    start_idx = len(dates) - target_len
    aligned_dates = dates[start_idx:]
    details = [d[start_idx:] for d in details]
    scaling = scaling[start_idx:]

    meta = {
        "wavelet": wavelet,
        "level": level,
        "series": "log_price",
        "padding": "symmetric_tail_only",
        "norm": True,
        "trim_approx": False,
        "original_length": int(len(dates)),
        "padded_length": int(padded.size),
        "pad_len": int(pad_len),
        "analysis_window_length": int(target_len),
    }
    return WaveletTransformResult(
        details=details,
        scaling=scaling,
        dates=list(aligned_dates),
        meta=meta,
    )


def compute_variance_spectrum(
    result: WaveletTransformResult, baseline_returns: np.ndarray
) -> dict[str, Any]:
    # baseline_returns should be aligned to result.dates
    def _var_pop(x: np.ndarray) -> float:
        if x.size == 0:
            return 0.0
        m = float(np.mean(x))
        # population variance (energy per sample) to align with energy preservation
        return float(np.sum((x - m) ** 2) / x.size)

    per_level = {f"D{idx+1}": _var_pop(arr) for idx, arr in enumerate(result.details)}
    s_key = f"S{len(result.details)}"
    per_level[s_key] = _var_pop(result.scaling)
    total = _var_pop(baseline_returns)
    # For SWT with norm=True, the sum of detail variances approximates total variance.
    # Including the scaling variance double-counts energy in practice.
    sum_details = float(sum(per_level[k] for k in per_level.keys() if k.startswith("D")))
    rel_err = float((sum_details - total) / total) if total != 0 else 0.0
    spectrum = {
        "per_level": per_level,
        "total_variance": total,
        "sum_details": sum_details,
        "relative_error": rel_err,
    }
    return spectrum


def compute_histograms(details: list[np.ndarray], bins: int = 50) -> dict[str, Any]:
    histos: dict[str, Any] = {}
    # Determine global bin edges per level independently for interpretability
    for i, arr in enumerate(details):
        key = f"D{i+1}"
        counts, bin_edges = np.histogram(arr, bins=bins)
        histos[key] = {
            "bin_edges": bin_edges.tolist(),
            "counts": counts.astype(int).tolist(),
        }
    return histos


def to_coefficients_json(
    ticker: str,
    result: WaveletTransformResult,
) -> dict[str, Any]:
    coeffs: dict[str, Any] = {}
    for i, arr in enumerate(result.details):
        key = f"D{i+1}"
        coeffs[key] = [{"date": d, "value": float(v)} for d, v in zip(result.dates, arr.tolist())]
    s_key = f"S{len(result.details)}"
    coeffs[s_key] = [
        {"date": d, "value": float(v)} for d, v in zip(result.dates, result.scaling.tolist())
    ]
    return {
        "ticker": ticker,
        "metadata": result.meta,
        "coefficients": coeffs,
    }


def to_volatility_histogram_json(
    ticker: str,
    spectrum: dict[str, Any],
    histos: dict[str, Any],
    meta: dict[str, Any],
) -> dict[str, Any]:
    return {
        "ticker": ticker,
        "metadata": meta,
        "variance_spectrum": spectrum,
        "histograms": histos,
    }


def normalize_variance_spectrum(
    per_level: dict[str, Any], order: list[str] | None = None
) -> dict[str, float]:
    """Normalize per-level variances to percentages that sum to 100.

    Parameters
    ----------
    per_level: mapping like {"D1": var1, ..., "S5": varS}
    order: explicit ordering of bands. Defaults to [D1..D5, S5].

    Returns
    -------
    Ordered dict-like mapping (by iteration) of band -> percent (0..100).
    Missing bands are treated as 0.0; if total is 0, returns 0.0 for all.
    """
    default_order = [f"D{i}" for i in range(1, 6)] + ["S5"]
    bands = order or default_order
    raw_values: list[float] = []
    prepared: dict[str, float] = {}
    for key in bands:
        try:
            val = float(per_level.get(key, 0.0) or 0.0)
        except Exception:
            val = 0.0
        prepared[key] = val
        raw_values.append(val)
    total = float(sum(raw_values))
    if total <= 0.0:
        return {k: 0.0 for k in bands}
    return {k: (prepared[k] / total) * 100.0 for k in bands}


def reconstruct_logprice_series(
    dates: list[str],
    closes: list[float],
    level: int = 5,
    wavelet: str = "sym4",
) -> tuple[list[str], dict[str, list[float]], dict[str, Any]]:
    """Reconstruct price series from log-price SWT by progressively adding bands.

    Padding/analysis strategy mirrors log-returns pipeline:
    - Compute on full history for proper context (fills the beginning of the window)
    - Tail-pad with symmetric mirroring to multiple of 2^level
    - Reconstruct on padded arrays, then unpad and slice the last ~504 days
    - Exponentiate reconstructed log-price back to price units
    """
    if len(closes) < (level + 34):
        raise ValueError("Insufficient close prices for stable SWT at requested level")

    # Sanitize close prices
    arr = np.asarray(list(closes), dtype=float)
    good = np.isfinite(arr) & (arr > 0)
    if not np.all(good):
        for i in range(arr.size):
            if not (arr[i] > 0 and math.isfinite(arr[i])):
                arr[i] = arr[i - 1] if i > 0 else np.nan
        if not (arr[0] > 0 and math.isfinite(arr[0])):
            first_valid = next((x for x in arr if x > 0 and math.isfinite(x)), np.nan)
            arr[0] = first_valid
            for i in range(1, arr.size):
                if not (arr[i] > 0 and math.isfinite(arr[i])):
                    arr[i] = arr[i - 1]

    log_price = np.log(arr)

    # Tail-pad to multiple of 2^level
    padded, pad_len = _tail_pad_to_power_of_two_length(log_price, power=level)

    # SWT on padded series with norm=False to enable faithful inverse reconstruction
    coeffs = pywt.swt(data=padded, wavelet=wavelet, level=level, trim_approx=False, norm=False)
    J = len(coeffs)

    # Reconstruction sets: start with S_J, then progressively add D_J, D_{J-1}, ...
    recon_sets: list[tuple[str, set[str]]] = []
    # S_J only
    recon_sets.append((f"S{J}", {f"S{J}"}))
    # S_J + D_J + D_{J-1} + ...
    accum: set[str] = {f"S{J}"}
    for j in range(J, 0, -1):
        band = f"D{j}"
        accum = set(accum) | {band}
        name = "_".join(sorted(accum, key=lambda k: (k[0] != "S", -int(k[1:]))))
        recon_sets.append((name, set(accum)))

    def _zero_like(x: np.ndarray) -> np.ndarray:
        return np.zeros_like(x)

    recon: dict[str, list[float]] = {}
    for name, include in recon_sets:
        # Build coeffs list for iswt: list of (cA_j, cD_j) from j=1..J
        pairs: list[tuple[np.ndarray, np.ndarray]] = []
        for j in range(1, J + 1):
            cA_j, cD_j = coeffs[j - 1]
            # Always keep approximation coefficients at all levels;
            # zero-out only the detail coefficients not included in this reconstruction.
            a = cA_j
            d = cD_j if (f"D{j}" in include) else (_zero_like(cD_j))
            pairs.append((a, d))
        recon_log = pywt.iswt(pairs, wavelet)
        if pad_len:
            recon_log = recon_log[:-pad_len]
        # Slice to last ~504 days
        target_len = min(504, len(dates))
        start_idx = len(dates) - target_len
        recon_log = recon_log[start_idx:]
        # Back to price units
        recon[name] = np.exp(recon_log).astype(float).tolist()

    # Aligned dates for window
    target_len = min(504, len(dates))
    start_idx = len(dates) - target_len
    aligned_dates = dates[start_idx:]

    meta = {
        "wavelet": wavelet,
        "level": level,
        "series": "price",
        "reconstructed_from": "log_price",
        "method": "iswt",
        "padding": "symmetric_tail_only",
        "norm": False,
        "analysis_window_length": int(target_len),
    }
    return list(aligned_dates), recon, meta


def to_reconstructed_prices_json(
    ticker: str,
    dates: list[str],
    recon: dict[str, list[float]],
    meta: dict[str, Any],
) -> dict[str, Any]:
    recon_series: dict[str, list[dict[str, float]]] = {}
    for name, values in recon.items():
        recon_series[name] = [{"date": d, "value": float(v)} for d, v in zip(dates, list(values))]
    return {
        "ticker": ticker,
        "metadata": meta,
        "reconstructions": recon_series,
    }
