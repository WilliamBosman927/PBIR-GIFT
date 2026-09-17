# Stage-D statistical appendix

## Statistical unit and uncertainty

The primary summary uses six dataset-window cell means. Within each bootstrap replicate, cells are resampled hierarchically, seed runs are resampled within selected cells, and aligned daily returns are sampled with circular blocks of length 20. Intervals use 10,000 bootstrap replicates.

Exact one-sided sign-flip tests operate on the six cell-mean Sharpe differences. Holm values adjust the family of six reported contrasts. Hedges g is the small-sample corrected standardized paired effect across the same six cell means.

## Multiplicity and decision rule

The primary support rule requires the 95% block-bootstrap interval lower bound to exceed zero, both dataset means to be positive, and at least two of three window means to be positive. The confidence-interval condition failed. Exact and Holm-adjusted p-values are supporting diagnostics, not substitutes for the predeclared decision rule.

## Reproducibility

Machine-readable estimates are stored in `stage_d_inference.json`; cell-level contrasts and absolute cell means are stored in the two CSV files in this bundle. Equal-weight results are identical across seeds by construction and are collapsed descriptively at cell level.
