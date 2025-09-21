from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable

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
    return WaveletTransformResult(details=details, scaling=scaling, dates=list(aligned_dates), meta=meta)


def compute_variance_spectrum(result: WaveletTransformResult, baseline_returns: np.ndarray) -> dict[str, Any]:
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
    coeffs["S5"] = [{"date": d, "value": float(v)} for d, v in zip(result.dates, result.scaling.tolist())]
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


