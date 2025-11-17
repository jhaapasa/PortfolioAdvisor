from __future__ import annotations

import math
from datetime import date, timedelta

import numpy as np
import pytest

from portfolio_advisor.stocks.wavelet import (
    WaveletTransformResult,
    compute_histograms,
    compute_modwt_logprice,
    compute_modwt_logreturns,
    compute_variance_spectrum,
    normalize_variance_spectrum,
    reconstruct_logprice_series,
    to_coefficients_json,
    to_reconstructed_prices_json,
    to_volatility_histogram_json,
)


def _make_synthetic_closes(n: int) -> tuple[list[str], list[float]]:
    # Generate business-day-like sequence of dates
    start = date(2020, 1, 1)
    dates: list[str] = []
    d = start
    while len(dates) < n:
        if d.weekday() < 5:
            dates.append(d.isoformat())
        d += timedelta(days=1)
    # Create synthetic log returns with mixed frequencies and mild noise
    t = np.arange(len(dates), dtype=float)
    lr = 0.001 * np.sin(2 * math.pi * t / 10.0) + 0.0005 * np.sin(2 * math.pi * t / 40.0)
    # Build close prices from returns
    log_price = np.cumsum(lr) + math.log(100.0)
    closes = np.exp(log_price).tolist()
    return dates, closes


@pytest.mark.parametrize("level", [3, 5])
def test_wavelet_energy_partition_and_alignment(level):
    dates, closes = _make_synthetic_closes(800)
    result = compute_modwt_logreturns(dates=dates, closes=closes, level=level, wavelet="sym4")

    # Result lengths match dates (2 years ~= 504 trading days)
    assert len(result.dates) == min(504, len(dates) - 1)
    assert all(isinstance(x, str) for x in result.dates)
    for d in result.details:
        assert d.shape[0] == len(result.dates)
    assert result.scaling.shape[0] == len(result.dates)

    # Energy check: Var(r) ~= sum Var(Dj) + Var(SJ)
    arr = np.array(closes, dtype=float)
    base_lr = np.diff(np.log(arr))
    base_aligned = base_lr[-len(result.dates) :]
    spectrum = compute_variance_spectrum(result, base_aligned)
    rel_err = float(spectrum.get("relative_error", 1.0))
    tolerance = 0.05 if level >= 5 else 0.25
    assert abs(rel_err) < tolerance

    # Histograms
    histos = compute_histograms(result.details, bins=32)
    bands = {f"D{i}" for i in range(1, level + 1)}
    assert set(histos.keys()) == bands
    for v in histos.values():
        assert len(v["bin_edges"]) == 33
        assert len(v["counts"]) == 32

    # Normalized spectrum percentages sum to 100
    normalized = normalize_variance_spectrum(spectrum["per_level"])
    assert pytest.approx(sum(normalized.values()), abs=1e-6) == 100.0


def test_logprice_transform_and_reconstruction_behaviors():
    dates, closes = _make_synthetic_closes(800)
    # Log-price transform aligns to window and yields D1..D5 + S5
    price_result = compute_modwt_logprice(dates=dates, closes=closes, level=5, wavelet="sym4")
    assert len(price_result.dates) == min(504, len(dates))
    assert len(price_result.details) == 5
    for d in price_result.details:
        assert d.shape[0] == len(price_result.dates)
    assert price_result.scaling.shape[0] == len(price_result.dates)

    # Reconstructions: full sum should approximate original prices over window
    recon_dates, recon_map, recon_meta, coi_boundaries = reconstruct_logprice_series(
        dates=dates, closes=closes, level=5, wavelet="sym4"
    )
    assert len(recon_dates) == min(504, len(dates))
    # Verify COI boundaries are returned
    assert coi_boundaries is not None
    assert "S5" in coi_boundaries
    assert isinstance(coi_boundaries["S5"], tuple)
    assert len(coi_boundaries["S5"]) == 2
    # Ensure required keys
    assert "S5" in recon_map
    assert "S5_D5_D4_D3_D2_D1" in recon_map
    # Verify that we have all S1..S5 smooth coefficients
    for j in range(1, 6):
        assert f"S{j}" in recon_map, f"Missing S{j} smooth coefficients"
    full = np.array(recon_map["S5_D5_D4_D3_D2_D1"], dtype=float)
    window_orig = np.array(closes[-len(recon_dates) :], dtype=float)
    # Compare relative RMSE to avoid scale dependence
    denom = max(1e-9, float(np.mean(np.abs(window_orig))))
    rmse = float(np.sqrt(np.mean((full - window_orig) ** 2))) / denom
    assert rmse < 0.02  # within 2%
    # Variance decreases as we remove detail bands
    var_full = float(np.var(full))
    var_s5 = float(np.var(np.array(recon_map["S5"], dtype=float)))
    assert var_s5 < var_full

    # Verify MRA additivity: S_J + D_1 + ... + D_J should equal original signal
    # Check if we have the full reconstruction (should be very close to original)
    full_recon_key = "S5_D5_D4_D3_D2_D1"  # This should equal S_1 in MRA theory
    if full_recon_key in recon_map and "S1" in recon_map:
        full_recon = np.array(recon_map[full_recon_key], dtype=float)
        s1_signal = np.array(recon_map["S1"], dtype=float)
        # S1 and full reconstruction should be identical (both = original signal)
        mra_diff = np.abs(full_recon - s1_signal)
        max_diff = float(np.max(mra_diff))
        rel_diff = max_diff / float(np.mean(np.abs(s1_signal)))
        assert (
            rel_diff < 0.01
        ), f"S1 and full reconstruction differ by {rel_diff:.6f} (should be ~0)"

    # Verify smooth signal ordering: S1 should have highest variance, S5 lowest
    if all(f"S{j}" in recon_map for j in range(1, 6)):
        variances = [float(np.var(np.array(recon_map[f"S{j}"], dtype=float))) for j in range(1, 6)]
        # Generally, variance should decrease as j increases (smoother signals)
        # Allow some tolerance for numerical precision
        for j in range(1, 4):  # Check S1 > S2, S2 > S3, S3 > S4 (S5 might be very small)
            assert (
                variances[j - 1] >= variances[j] * 0.95
            ), f"S{j} variance should be >= S{j+1} variance"


def test_to_coefficients_and_histogram_serialization():
    n = 120
    dates = [f"2025-01-{i:02d}" for i in range(1, n + 1)]
    details = [np.linspace(-1, 1, n), np.linspace(1, -1, n)]
    scaling = np.zeros(n)
    result = WaveletTransformResult(
        details=[np.asarray(d) for d in details],
        scaling=np.asarray(scaling),
        dates=dates,
        meta={"level": 2, "wavelet": "sym2"},
    )

    coeffs_doc = to_coefficients_json("aapl", result)
    assert coeffs_doc["metadata"]["wavelet"] == "sym2"
    assert coeffs_doc["coefficients"]["cD1"][0]["date"] == dates[0]
    assert coeffs_doc["coefficients"]["cA2"][-1]["value"] == pytest.approx(0.0)

    spectrum = {"per_level": {"D1": 1.0, "D2": 2.0, "S2": 3.0}, "total_variance": 0.5}
    histos = {"D1": {"bin_edges": [0, 1], "counts": [10]}}
    meta = {"analysis_start": dates[0], "analysis_end": dates[-1]}
    hist_doc = to_volatility_histogram_json("aapl", spectrum, histos, meta)
    assert hist_doc["metadata"]["analysis_end"] == dates[-1]
    assert hist_doc["variance_spectrum"]["per_level"]["S2"] == 3.0

    recon_doc = to_reconstructed_prices_json("aapl", dates, {"S1": [1.0] * n}, meta)
    assert recon_doc["reconstructions"]["S1"][0]["value"] == pytest.approx(1.0)
