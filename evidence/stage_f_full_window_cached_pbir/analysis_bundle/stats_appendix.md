# Stage-F statistical appendix

The primary unit is the dataset-window cell mean: 12 cells with three paired seeds each. The exact one-sided sign-flip test operates on these 12 cell-mean deltas. Holm correction covers the five reported outcomes.

The hierarchical bootstrap resamples cells, seeds within cells, and then aligned daily returns using circular blocks. It therefore preserves within-series dependence more appropriately than a day-level t-test. This run uses 10,000 replicates and block length 20.

W2/W4/W6 overlap the odd-window schedule. The all-window analysis is a full robustness map; the odd-versus-even summary is reported explicitly so generalization beyond the original Stage-C windows remains visible.
