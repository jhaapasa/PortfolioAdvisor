from __future__ import annotations

import math
from datetime import date, timedelta

import numpy as np

from portfolio_advisor.stocks.wavelet import (
    compute_histograms,
    compute_modwt_logreturns,
    compute_variance_spectrum,
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
    lr = 0.001 * np.sin(2 * math.pi * t / 10.0) + 0.0005 * np.sin(
        2 * math.pi * t / 40.0
    )
    rng = np.random.default_rng(42)
    lr += rng.normal(0.0, 0.0002, size=lr.shape)
    # Build close prices from returns
    log_price = np.cumsum(lr) + math.log(100.0)
    closes = np.exp(log_price).tolist()
    return dates, closes


def test_wavelet_energy_partition_and_alignment():
    dates, closes = _make_synthetic_closes(800)
    result = compute_modwt_logreturns(dates=dates, closes=closes, level=5, wavelet="sym4")

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
    assert abs(rel_err) < 0.05  # within 5%

    # Histograms
    histos = compute_histograms(result.details, bins=32)
    assert set(histos.keys()) == {"D1", "D2", "D3", "D4", "D5"}
    for v in histos.values():
        assert len(v["bin_edges"]) == 33
        assert len(v["counts"]) == 32


