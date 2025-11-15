# Confidence Bands for S6 Wavelet Trend in COI Regions

## Overview

The confidence bands visualize the worst-case price uncertainty around the S6 wavelet trend line within the Cone of Influence (COI) regions. These bands are based on filter-weighted error calculations that account for how much each wavelet filter coefficient is affected by boundary padding.

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

The confidence bands are only displayed for the S6 level because:
- S6 has the largest COI regions (most affected by boundaries)
- It represents the longest-term trend (~64 days)
- Visual clarity - too many bands would clutter the chart

For other wavelet levels, the dotted line style still indicates COI regions without explicit confidence bands.
