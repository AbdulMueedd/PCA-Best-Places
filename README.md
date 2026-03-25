# PCA on Best Places to Live

Applies PCA to the Places Rated Almanac dataset (329 US cities rated on 9 livability factors) to identify patterns and outliers.

## What It Does

1. Reads `places.txt` — 329 cities × 9 factors (climate, housing, healthcare, crime, transportation, education, arts, recreation, economy)
2. **Log-transformed path:** applies log₁₀, centers data, runs SVD, extracts principal components, projects onto top 2 PCs, identifies outliers
3. **Z-score path:** normalizes raw data with z-scores, centers, runs SVD, repeats the same analysis
4. Compares how the two preprocessing methods change the PCA results

## Files Needed

```
places.txt    # must be in the same directory
```

## Run

```bash
python3 ml4Problem3.py
```

## Output

Prints PC vectors, interpretations, variance explained, and outlier lists. Saves:

| File | Description |
|------|-------------|
| `p3_log_scatter.png` | PCA scatter plot (log-transformed) |
| `p3_zscore_scatter.png` | PCA scatter plot (z-score normalized) |

## Key Results

**Log-transformed:** PC1 is dominated by Arts (0.874) — separates culturally rich metros from small towns. Top outliers: New York, Chicago, LA, San Francisco on one end; Texarkana, Sharon PA on the other.

**Z-score:** PC1 loads more evenly across all factors — represents overall city quality. New York becomes a massive outlier (PC1 ≈ 12.4). Z-score normalization prevents high-variance factors from dominating.

## Dependencies

```
numpy, matplotlib
```
