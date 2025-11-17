# Confidence Bands for S6 Wavelet Trend in COI Regions (Removed)

**Status: Removed Feature**

**When Implemented**: November 2024
**When Removed**: November 2024 (same sprint)
**Why Removed**: See "Why This Was Removed" section below

## What This Feature Was

The confidence bands were designed to visualize the worst-case price uncertainty around the S6 wavelet trend line within the Cone of Influence (COI) regions. These bands were based on filter-weighted error calculations that account for how much each wavelet filter coefficient is affected by boundary padding.

## Implementation Approach

The confidence bands used advanced filter-weighted distortion calculations from `coi_distortion_advanced.py` to provide quantitative uncertainty estimates.

## What the Visualization Shows

When viewing the candlestick chart with wavelet trends:

1. **S6 Trend Line** (~64 days):
   - **Solid red line**: Reliable data region (center of the time series)
   - **Dotted red line**: COI regions (near time series boundaries)
   - **Light red shaded area**: Confidence bands showing worst-case price deviation

2. **Confidence Band Width**:
   - **Narrowest** at the COI boundary (±0.39% or ~$1.18 for a $150 stock)
   - **Widest** at the time series edges (±2.48% or ~$7.35 for a $150 stock)
   - The bands progressively widen as you move closer to the edges

## Interpreting the Confidence Bands

The confidence bands represent the maximum possible error in the wavelet reconstruction due to boundary effects:

| Location | Distance from Edge | Confidence Band | Interpretation |
|----------|-------------------|-----------------|----------------|
| Edge | 0-50 samples | ±2.48% | Very uncertain, use with extreme caution |
| Inner COI | 100 samples | ±1.39% | Moderately uncertain |
| COI Boundary | 224 samples | ±0.39% | Minimal uncertainty |
| Reliable Region | >256 samples | No bands | Fully reliable |

## Technical Details

The confidence bands are calculated using:

1. **Filter-weighted distortion**: Accounts for the actual wavelet filter coefficient magnitudes
2. **Worst-case error estimation**: `error = Σ|h[k]| × σ` for filter taps on padding
3. **Log-normal transformation**: Bands = `price × exp(±error)` for accurate price-space representation

## Benefits

1. **Visual Risk Assessment**: Immediately see where the S6 trend is less reliable
2. **Quantified Uncertainty**: Know exactly how much the price could deviate
3. **Better Decision Making**: Avoid relying on trends in high-uncertainty regions
4. **Progressive Nature**: Shows that COI effects gradually increase toward boundaries

## Example Use Cases

- **Trading Signals**: Don't base decisions on S6 trends where bands are wide
- **Trend Analysis**: Focus on the solid line region for reliable long-term trends
- **Risk Management**: Use band width to gauge confidence in wavelet-based predictions

## Implementation Note

The confidence bands were only displayed for the S6 level because:
- S6 has the largest COI regions (most affected by boundaries)
- It represents the longest-term trend (~64 days)
- Visual clarity - too many bands would clutter the chart

For other wavelet levels, the dotted line style still indicates COI regions without explicit confidence bands.

## Why This Was Removed

### Visual Clarity Issues

**Primary Problem**: The confidence bands were too subtle on full price charts
- Band width at COI boundary: ±0.39% (±$1.18 for $150 stock)
- Band width at edge: ±2.48% (±$7.35 for $150 stock)
- These ranges were **barely visible** when plotted alongside candlesticks showing daily volatility of 1-3%

**Example**: On a $150 stock chart with daily moves of $2-5:
- COI confidence bands: ±$1-7 width
- Daily price range: $2-5 per bar
- **Result**: Bands lost in the noise of normal price movement

### User Experience Concerns

**Complexity vs Value**:
- Added visual elements that users needed to understand
- Documentation burden (explaining filter-weighted distortion)
- Minimal additional insight beyond "this region less reliable"

**Simple Alternative Works Better**:
- Dotted vs solid lines: Immediately clear
- No numbers to interpret
- Binary distinction: reliable vs less reliable
- No learning curve

### Testing Results

**What We Tried**:
1. **Solid bands**: Too faint
2. **Increased alpha**: Became cluttered
3. **Color coding**: Confusing with existing line colors
4. **Gradient alpha**: Subtle but still unclear value

**Conclusion**: None of the visual treatments made the bands valuable enough to justify the complexity.

### Technical Considerations

**Implementation Was Sound**:
- Filter-weighted calculations were correct
- Worst-case error estimates were accurate
- Code quality was good

**But**:
- Calculations were expensive (not performance critical, but non-trivial)
- Added complexity to plotting code
- Harder to maintain multiple visualization modes

### Decision Rationale

**Keep It Simple**:
- COI visualization goal: Show where data is less reliable
- Dotted lines achieve this goal effectively
- Quantitative uncertainty estimates overkill for visual assessment
- Can always add back later if user demand emerges

**What We Kept**:
- Advanced distortion calculation code (`coi_distortion_advanced.py`)
- Theory documentation (valuable reference)
- Potential for future use in:
  - Programmatic analysis (not visualization)
  - Confidence scoring systems
  - Advanced user features

### Alternative Approaches Considered

**After Removal**:
1. **Annotations**: Text labels showing distortion %
   - Rejected: Too cluttered
   
2. **Hover tooltips**: Show distortion on mouse-over
   - Rejected: Requires interactive plots (future feature)
   
3. **Separate uncertainty plot**: Distortion on second axis
   - Rejected: Users want single integrated view
   
4. **Color intensity**: Fade line color by distortion
   - Partially implemented: Subtle alpha gradient within COI
   - Kept as minor enhancement

### Lessons Learned

1. **Visual effectiveness ≠ Mathematical sophistication**
   - Advanced calculations don't always make better visualizations
   
2. **Test with real data early**
   - Confidence bands looked good in theory
   - Real stock charts revealed the visibility problem
   
3. **Simplicity often wins**
   - Dotted lines: 5 minutes to understand
   - Confidence bands: Need to read documentation
   
4. **Keep good code even when removing features**
   - Distortion calculations still useful for other purposes
   - Documentation valuable for understanding theory

## Related Features

**What Replaced It**: Simple dotted line visualization
- Implementation: `plotting.py:_create_coi_plot_segments()`
- Solid lines in reliable region
- Dotted lines in COI regions
- No quantitative uncertainty shown

**What Remains**: Advanced distortion analysis available
- Code: `stocks/coi_distortion_advanced.py`
- Can be used programmatically
- Potential future application: Auto-filtering unreliable predictions

## TODO: Potential Future Uses

Even though removed from visualization, the confidence band calculations could be valuable for:

- [ ] Programmatic reliability scoring
- [ ] Auto-filtering trade signals in COI regions
- [ ] Risk adjustment factors for predictions
- [ ] Quality metrics for backtesting
- [ ] Advanced user mode (opt-in complex visualization)
