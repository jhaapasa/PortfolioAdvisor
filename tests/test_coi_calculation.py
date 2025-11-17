"""Tests for cone of influence calculation in wavelet analysis."""

from portfolio_advisor.stocks.wavelet import calculate_cone_of_influence


def test_calculate_coi_basic():
    """Test basic COI calculation for standard parameters."""
    # For sym4 wavelet with filter length 8, at level 5 with 504 samples
    coi = calculate_cone_of_influence(504, 5, "sym4")

    # Verify all levels are present
    assert "S1" in coi
    assert "S2" in coi
    assert "S3" in coi
    assert "S4" in coi
    assert "S5" in coi

    # Verify tuples have correct structure
    for level in range(1, 6):
        key = f"S{level}"
        assert isinstance(coi[key], tuple)
        assert len(coi[key]) == 2
        start, end = coi[key]
        assert isinstance(start, int)
        assert isinstance(end, int)
        assert start >= 0
        assert end <= 504
        assert start <= end


def test_calculate_coi_levels_increase():
    """Test that COI width increases with decomposition level."""
    coi = calculate_cone_of_influence(504, 5, "sym4")

    # Higher levels should have wider COI (smaller reliable region)
    start1, end1 = coi["S1"]
    start2, end2 = coi["S2"]
    start3, end3 = coi["S3"]
    start4, end4 = coi["S4"]
    start5, end5 = coi["S5"]

    # Start indices should increase with level
    assert start1 < start2 < start3 < start4 < start5
    # End indices should decrease with level
    assert end1 > end2 > end3 > end4 > end5
    # Reliable region shrinks with higher levels
    reliable1 = end1 - start1
    reliable2 = end2 - start2
    reliable5 = end5 - start5
    assert reliable1 > reliable2 > reliable5


def test_calculate_coi_exponential_growth():
    """Test that COI width grows exponentially (by factor of 2) across levels."""
    coi = calculate_cone_of_influence(504, 5, "sym4")

    # sym4 has filter length 8, so COI width at level j = (8-1) * 2^(j-1) = 7 * 2^(j-1)
    expected_widths = {
        "S1": 7 * (2**0),  # 7
        "S2": 7 * (2**1),  # 14
        "S3": 7 * (2**2),  # 28
        "S4": 7 * (2**3),  # 56
        "S5": 7 * (2**4),  # 112
    }

    for level in range(1, 6):
        key = f"S{level}"
        start, end = coi[key]
        actual_width = start  # COI width on left boundary
        expected_width = expected_widths[key]
        assert (
            actual_width == expected_width
        ), f"Level {level}: expected {expected_width}, got {actual_width}"


def test_calculate_coi_different_wavelets():
    """Test COI calculation with different wavelet families."""
    wavelets = ["haar", "db4", "sym4", "coif2"]

    for wavelet in wavelets:
        coi = calculate_cone_of_influence(504, 5, wavelet)
        # All wavelets should produce valid boundaries
        assert len(coi) == 5
        for level in range(1, 6):
            key = f"S{level}"
            assert key in coi
            start, end = coi[key]
            assert start >= 0
            assert end <= 504
            assert start <= end


def test_calculate_coi_small_series():
    """Test COI with very small time series."""
    # Small series with 100 samples
    coi = calculate_cone_of_influence(100, 3, "sym4")

    assert len(coi) == 3
    # Even with small series, boundaries should be valid
    for level in range(1, 4):
        key = f"S{level}"
        start, end = coi[key]
        assert start >= 0
        assert end <= 100
        # At higher levels, COI might cover entire series
        # but start should never exceed end
        assert start <= end


def test_calculate_coi_extreme_level():
    """Test COI when level is high relative to series length."""
    # Level 5 with only 200 samples - COI at S5 is 112 samples on each side
    coi = calculate_cone_of_influence(200, 5, "sym4")

    start5, end5 = coi["S5"]
    # COI width is 112, so reliable region is max(0, 200 - 2*112) = 0
    # But implementation ensures start <= end
    assert start5 == 112
    assert end5 == max(112, 200 - 112)  # = max(112, 88) = 112
    # When COI covers everything, start == end
    if start5 == end5:
        assert True  # No reliable data region


def test_calculate_coi_boundary_conditions():
    """Test COI calculation at boundary conditions."""
    # Test with exact power of 2
    coi = calculate_cone_of_influence(512, 5, "sym4")
    assert coi["S5"] == (112, 512 - 112)

    # Test with level 1 (smallest COI)
    coi = calculate_cone_of_influence(100, 1, "sym4")
    assert coi["S1"] == (7, 93)  # sym4 filter length is 8, so COI width = 7


def test_calculate_coi_validates_inputs():
    """Test that COI calculation handles edge cases gracefully."""
    # Very small series - COI might exceed series length
    coi = calculate_cone_of_influence(10, 2, "sym4")
    assert len(coi) == 2
    # Function returns calculated COI boundaries even if they exceed series bounds
    # Downstream code (plotting) is responsible for clipping to actual data range
    for level in range(1, 3):
        start, end = coi[f"S{level}"]
        # Start and end should be valid integers with start <= end
        assert isinstance(start, int)
        assert isinstance(end, int)
        assert start <= end


def test_calculate_coi_consistent_with_reconstruction():
    """Test that COI boundaries align with reconstruction window length."""
    # Standard 2-year window is 504 days
    window_length = 504
    level = 5
    coi = calculate_cone_of_influence(window_length, level, "sym4")

    # Verify boundaries make sense for the window
    for j in range(1, level + 1):
        start, end = coi[f"S{j}"]
        # Reliable region should be non-empty for reasonable parameters
        reliable_width = end - start
        # At minimum level (S1), should have most of the data reliable
        if j == 1:
            assert reliable_width > 0.9 * window_length
        # At maximum level (S5), reliable region is smaller
        if j == level:
            # With sym4 (filter=8), S5 COI = 7 * 16 = 112 on each side
            # Reliable region = 504 - 224 = 280
            assert reliable_width == 504 - 2 * 112
