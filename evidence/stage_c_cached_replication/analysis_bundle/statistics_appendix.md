# Statistical appendix: Stage-C fixed-code controlled replication

## Analysis units and direction

- Primary analysis unit: mean of the three matched random-seed deltas in one dataset-window cell.
- Confirmatory cells: D1/D2 × W1/W3/W5 = 6 non-overlapping cells.
- Positive directions: PBIR − Pure GIFT for Sharpe, Sortino, and total return; Pure GIFT − PBIR for maximum drawdown, so positive means lower PBIR drawdown.

## Inference

- Exact one-sided sign-flip sensitivity test across the six cell means, with Holm adjustment across four endpoints.
- Paired Hedges g is computed from the six cell deltas with small-sample correction.
- Primary uncertainty is a hierarchical circular moving-block bootstrap: cells resampled with replacement, seeds resampled within cells, daily returns resampled in 20-day circular blocks, 10,000 resamples, RNG seed 20260823.
- This bootstrap retains serial dependence approximately; it is the basis of the stated 95% confidence intervals.

## Result reproducibility

- Raw pair registry: `../aggregate/stage_b_all_cells.csv`.
- Collapsed paired deltas: `cell_level_paired_deltas.csv`.
- Machine-readable inference: `confirmatory_inference.json`.
- Run command: `python analyze_final_results.py --results-dir results/stage_c_cached_replication --expected-seeds 3 --resamples 10000 --block-length 20 --analysis-role controlled_replication --output-dir results/stage_c_cached_replication/analysis_bundle`.

## Interpretation constraint

The test asks whether changing only the PBIR shaping transformation produces a stable improvement conditional on the Stage-B-selected shared artifact. It is not a fresh out-of-sample confirmation because these windows were already observed.
