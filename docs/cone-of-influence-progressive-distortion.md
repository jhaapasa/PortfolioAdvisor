# Progressive Distortion in the Cone of Influence

## Theory

The cone of influence (COI) isn't a binary boundary - it represents a **gradual degradation** of coefficient reliability as you approach the time series edges.

### Mathematical Foundation

For a wavelet at decomposition level `j` with filter length `L`:

1. **Effective Filter Support**: The wavelet filter at level j has an effective support (region of influence) of approximately:
   ```
   L_eff(j) = L × 2^j samples
   ```

2. **Filter Overlap with Padding**: At distance `d` from the boundary:
   - If `d < L_eff(j)`, some filter taps operate on padded (artificial) data
   - The **distortion factor** can be defined as the fraction of filter taps on padding vs real data

### Distortion Quantification Methods

#### Method 1: Linear Distance Model
Simple normalized distance from COI boundary:
```
distortion(d, j) = max(0, 1 - d / COI_width(j))
```
Where:
- `d` = distance from nearest boundary (in samples)
- `COI_width(j) = (L - 1) × 2^(j-1)`
- Returns 0.0 (no distortion) to 1.0 (maximum distortion)

#### Method 2: Filter Support Fraction
More accurate - based on actual filter overlap:
```
distortion(d, j) = (L_eff(j) - d) / L_eff(j)  if d < L_eff(j)
                   0                           otherwise
```

#### Method 3: Exponential Decay Model
Models the progressive "fade" of reliability:
```
distortion(d, j) = exp(-d / decay_constant(j))

where decay_constant(j) = COI_width(j) / 3
```
This gives ~95% distortion at the edge, ~5% at the COI boundary.

#### Method 4: Energy-Based Distortion
Calculate what fraction of the filter's "energy" operates on padded vs real data.

For symmetric reflection padding, the distortion is lower than for periodic padding because reflections create smoother continuations.

## Practical Implications

### Distortion Thresholds
Based on wavelet theory literature:
- **d < 0.25 × COI_width**: Severe distortion (>75%), avoid for critical analysis
- **0.25 × COI_width ≤ d < 0.5 × COI_width**: Moderate distortion (50-75%), use with caution
- **0.5 × COI_width ≤ d < COI_width**: Mild distortion (25-50%), acceptable for most uses
- **d ≥ COI_width**: Minimal distortion (<25%), reliable

### Example: S6 with db4 (COI_width = 224)
```
Distance from edge | Distortion (Method 1) | Reliability
-------------------|----------------------|------------
0 samples          | 100%                | 0%
56 samples         | 75%                 | 25%
112 samples        | 50%                 | 50%
168 samples        | 25%                 | 75%
224 samples (COI)  | 0%                  | 100%
```

## Visualization Approaches

### 1. Gradient Transparency
Replace hard solid/dotted boundary with progressive alpha blending:
```python
alpha(d) = base_alpha × (1 - distortion(d))
```

### 2. Color Gradient
Use color to indicate distortion level:
- Blue → Green → Yellow → Red (reliable → unreliable)

### 3. Shaded Regions
Add semi-transparent bands showing distortion zones:
- Inner band: d < 0.5 × COI (moderate)
- Outer band: 0.5 × COI ≤ d < COI (mild)

### 4. Confidence Intervals
Show expanding confidence bands near boundaries.

## References

1. Torrence & Compo (1998): "A Practical Guide to Wavelet Analysis" - original COI formulation
2. Percival & Walden (2000): "Wavelet Methods for Time Series Analysis" - MODWT boundary effects
3. Grinsted et al. (2004): "Application of cross wavelet transform" - progressive COI visualization

