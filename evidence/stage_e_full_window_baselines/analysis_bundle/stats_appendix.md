# Stage-E statistical appendix

The primary unit is the dataset-window cell mean (12 cells, three seeds per cell). The hierarchical bootstrap resamples cells, then seeds within cells, then aligned daily returns with circular blocks. This preserves time dependence more faithfully than treating all days as independent.

Bootstrap replicates: 10,000; block length: 20. The sign-flip test operates on the 12 cell-mean Sharpe differences. Effect size is paired Hedges g across the same cells.

The strict rule was declared in code before Stage-E results existed. Light Mix directional replication is reported separately because it targets correspondence with the original six-window experiment rather than universal superiority across both portfolios.
