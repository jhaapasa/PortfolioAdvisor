"""Test advanced filter-weighted COI distortion calculations."""

import numpy as np

from portfolio_advisor.stocks.coi_distortion_advanced import (
    analyze_filter_characteristics,
    calculate_weighted_distortion,
    get_filter_energy_distribution,
    get_progressive_distortion_map,
    worst_case_signal_deviation,
)


class TestFilterEnergyDistribution:
    """Test filter energy distribution calculations."""

    def test_get_filter_energy_distribution_basic(self):
        """Test basic energy distribution for known wavelets."""
        h_coeffs, energy_cumsum = get_filter_energy_distribution("db4")

        # db4 has 8 coefficients
        assert len(h_coeffs) == 8
        assert len(energy_cumsum) == 8

        # Energy cumsum should be monotonic and end at 1.0
        assert np.all(np.diff(energy_cumsum) >= 0)  # monotonic
        assert np.isclose(energy_cumsum[-1], 1.0)  # normalized

        # First coefficient should have some energy
        assert energy_cumsum[0] > 0

    def test_filter_energy_concentration(self):
        """Test that filter energy is concentrated in first few coefficients."""
        for wavelet in ["db4", "sym4", "db2"]:
            h_coeffs, energy_cumsum = get_filter_energy_distribution(wavelet)

            # Most wavelets have >60% energy in first half of coefficients
            # (sym4 has about 74%, db4 has about 90%)
            half_idx = len(h_coeffs) // 2
            assert energy_cumsum[half_idx] > 0.6  # 60% in first half


class TestWeightedDistortion:
    """Test weighted distortion calculations."""

    def test_zero_distance_maximum_distortion(self):
        """Test that zero distance gives maximum distortion."""
        energy_dist, worst_err, expected_err = calculate_weighted_distortion(
            0, level=6, wavelet="db4"
        )

        # At distance 0, most energy is on padding
        assert energy_dist > 0.9  # >90% energy on padding
        assert worst_err > 0  # positive error
        assert expected_err > 0  # positive error
        assert worst_err >= expected_err  # worst case >= expected

    def test_large_distance_no_distortion(self):
        """Test that large distance gives no distortion."""
        # Distance beyond filter support
        energy_dist, worst_err, expected_err = calculate_weighted_distortion(
            1000, level=6, wavelet="db4"
        )

        assert energy_dist == 0.0
        assert worst_err == 0.0
        assert expected_err == 0.0

    def test_distortion_decreases_with_distance(self):
        """Test that distortion metrics decrease with distance."""
        distances = [0, 50, 100, 150, 200, 250, 300]
        energy_dists = []
        worst_errs = []

        for d in distances:
            e_dist, w_err, _ = calculate_weighted_distortion(d, 6, "db4")
            energy_dists.append(e_dist)
            worst_errs.append(w_err)

        # Should be non-increasing
        assert all(energy_dists[i] >= energy_dists[i + 1] for i in range(len(energy_dists) - 1))
        assert all(worst_errs[i] >= worst_errs[i + 1] for i in range(len(worst_errs) - 1))

    def test_energy_weighted_less_than_simple_overlap(self):
        """Test that energy-weighted distortion is less than simple overlap."""
        # At intermediate distances, energy weighting should give lower distortion
        for d in [100, 150, 200]:
            energy_dist, _, _ = calculate_weighted_distortion(d, 6, "db4")

            # Calculate simple overlap
            spacing = 2**6
            half_support = 8 * spacing // 2
            simple_overlap = max(0, (half_support - d) / half_support)

            # Energy-weighted should be less (due to small tail coefficients)
            if simple_overlap > 0:
                assert energy_dist <= simple_overlap


class TestFilterCharacteristics:
    """Test filter characteristic analysis."""

    def test_analyze_filter_characteristics_structure(self):
        """Test that filter analysis returns expected structure."""
        analysis = analyze_filter_characteristics("db4")

        # Check required fields
        assert "filter_length" in analysis
        assert "filter_coeffs" in analysis
        assert "energy_distribution" in analysis
        assert "center_of_mass" in analysis
        assert "effective_width" in analysis
        assert "energy_50_percent_width" in analysis
        assert "energy_90_percent_width" in analysis
        assert "l1_norm" in analysis
        assert "l2_norm" in analysis

        # Verify values
        assert analysis["filter_length"] == 8  # db4 has 8 coefficients
        assert len(analysis["filter_coeffs"]) == 8
        assert len(analysis["energy_distribution"]) == 8

    def test_energy_concentration_metrics(self):
        """Test energy concentration width metrics."""
        analysis = analyze_filter_characteristics("db4")

        # Energy concentration widths should increase
        assert (
            analysis["energy_50_percent_width"]
            <= analysis["energy_90_percent_width"]
            <= analysis["energy_95_percent_width"]
        )

        # All should be within filter length
        assert analysis["energy_95_percent_width"] <= analysis["filter_length"]


class TestProgressiveDistortionMap:
    """Test full distortion map generation."""

    def test_distortion_map_structure(self):
        """Test distortion map returns expected arrays."""
        n_samples = 100
        maps = get_progressive_distortion_map(n_samples, level=3, wavelet="db2")

        # Check all arrays present and correct size
        assert "energy_distortion" in maps
        assert "worst_case_error" in maps
        assert "expected_error" in maps
        assert "traditional_coi" in maps

        for key in maps:
            assert len(maps[key]) == n_samples
            assert isinstance(maps[key], np.ndarray)

    def test_distortion_map_symmetry(self):
        """Test that distortion is symmetric at boundaries."""
        n_samples = 100
        maps = get_progressive_distortion_map(n_samples, level=3)

        # Check symmetry (allowing small numerical differences)
        for key in ["energy_distortion", "worst_case_error", "expected_error"]:
            left_half = maps[key][: n_samples // 2]
            right_half = maps[key][n_samples // 2 :][::-1]
            assert np.allclose(left_half, right_half, rtol=1e-10)

    def test_traditional_vs_energy_weighted(self):
        """Test relationship between traditional and energy-weighted metrics."""
        maps = get_progressive_distortion_map(200, level=4)

        # Energy-weighted distortion should be reasonable compared to traditional
        # At the boundaries they can differ significantly
        # Look at the middle region where both metrics are moderate
        middle_region = slice(50, 150)  # Away from boundaries

        # In the middle region, both should show decreasing distortion
        energy_mid = maps["energy_distortion"][middle_region]
        trad_mid = maps["traditional_coi"][middle_region]

        # Both should have some variation in the middle
        assert np.std(energy_mid) > 0.01
        assert np.std(trad_mid) > 0.01


class TestWorstCaseSignalDeviation:
    """Test worst-case signal deviation calculations."""

    def test_signal_deviation_basic(self):
        """Test basic signal deviation calculation."""
        # At boundary, should have maximum deviation
        dev0 = worst_case_signal_deviation(0, level=4, signal_range=(-1, 1))
        assert dev0 > 0

        # Far from boundary, should have no deviation
        dev_far = worst_case_signal_deviation(500, level=4, signal_range=(-1, 1))
        assert dev_far == 0

    def test_signal_deviation_scales_with_range(self):
        """Test that deviation scales with signal range."""
        d = 100  # intermediate distance

        dev1 = worst_case_signal_deviation(d, 4, signal_range=(-1, 1))
        dev2 = worst_case_signal_deviation(d, 4, signal_range=(-10, 10))

        # Larger signal range should give proportionally larger deviation
        if dev1 > 0:
            assert np.isclose(dev2 / dev1, 10, rtol=0.1)
