# Stage-C fixed-code PBIR-GIFT vs Pure GIFT: replication summary

## Bottom line

The controlled-replication result **DOES NOT MEET the controlled-replication decision gate**. Across the six non-overlapping dataset-window cells (D1/D2 × W1/W3/W5), the mean paired ΔSharpe was **+0.368** (PBIR-GIFT − Pure GIFT), with a 20-day hierarchical moving-block-bootstrap 95% CI of **[-0.053, +0.785]**. The interval crosses zero.

This is a mixed result, not evidence that PBIR reliably improves the primary risk-adjusted objective. It is also not evidence of equivalence or of no effect.

## Completion and protocol checks

- Completed paired runs: **18/18** (6 cells × 3 seeds; windows W1, W3, W5).
- Seeds in every cell: **42, 123, 456**; mismatches: **0**.
- Frozen Stage-A selection hash count: **1**; PPO parameter-set count: **1**.
- Final daily-return artifacts missing: **0**; invalid status entries: **0**.
- Confirmatory units are W1/W3/W5. W2/W4/W6 are overlapping robustness windows only.

## Confirmatory decision rule

| Criterion | Observed result | Pass? |
|---|---:|:---:|
| Complete primary cells and three seeds | 6 cells, 3 seeds each | Yes |
| ΔSharpe bootstrap CI entirely above zero | [-0.053, +0.785] | No |
| Both dataset ΔSharpe means positive | D1 +0.234; D2 +0.502 | Yes |
| At least 2/3 non-overlapping windows positive | W1, W3, W5 | Yes |
| Worst cell MDD deterioration ≤ 5 pp | 2.375 pp | Yes |

## Confirmatory effect estimates

| Endpoint (PBIR direction positive) | Mean cell delta | Cell win rate | Hedges g (paired) | exact p | Holm p | 20-day block-bootstrap 95% CI |
|---|---:|---:|---:|---:|---:|---:|
| sharpe | +0.368 | 83.3% | +1.227 | 0.0312 | 0.1250 | [-0.053, +0.785] |
| sortino | +0.371 | 83.3% | +1.064 | 0.0312 | 0.1250 | [-0.128, +0.829] |
| max_drawdown_improvement | +0.406 | 66.7% | +0.183 | 0.3125 | 0.3125 | [-2.025, +2.216] |
| total_return | +3.173 | 83.3% | +0.586 | 0.0469 | 0.1250 | [-1.285, +10.083] |

Sharpe has an exact one-sided sign-flip p-value of 0.0312 (Holm-adjusted 0.1250); the small number of independent cells makes this test intentionally coarse.

## Cell-level ΔSharpe

| Dataset | Window | Confirmatory? | ΔSharpe | Seed SD | ΔReturn (pp) | MDD improvement (pp) |
|---|---|:---:|---:|---:|---:|---:|
| D1 | W1 | Yes | -0.062 | 0.132 | -0.306 | -1.562 |
| D1 | W3 | Yes | +0.252 | 0.569 | +1.551 | +1.270 |
| D1 | W5 | Yes | +0.513 | 1.072 | +11.780 | -2.375 |
| D2 | W1 | Yes | +0.642 | 0.357 | +0.007 | +1.904 |
| D2 | W3 | Yes | +0.525 | 0.527 | +4.601 | +1.483 |
| D2 | W5 | Yes | +0.340 | 0.543 | +1.406 | +1.713 |

## Interpretation and next action

- ΔSharpe is positive in 5/6 cells and all pooled window means are positive. However, the primary confidence interval still crosses zero, so the result is directionally favorable but inconclusive rather than a reliable general improvement.
- Return has a positive mean delta (+3.173 pp) and 5/6 positive confirmatory cells, but its 95% interval crosses zero. Drawdown has no reliable aggregate change (mean improvement +0.406 pp).
- Preserve this as a controlled replication on previously observed windows; it does not overwrite the locked Stage-B confirmatory result.
- The next decision is governed by the replication gate: only a stable positive result justifies the six-method and component-ablation expansion.

## Important limitations

- D1 and D2 share AMZN, MSFT, and JNJ; they are not independent markets.
- The W2/W4/W6 windows overlap with the three primary windows and must not be counted as six additional independent tests.
- Both methods use the same verified code/rule artifact in each cell. Because these artifacts were selected in Stage B and the windows have already been observed, this is a controlled mechanism replication rather than a fresh confirmatory holdout.
- Each saved evaluation return series contains 105 observations after feature lookback. The confidence intervals correctly reflect sizable uncertainty; they do not establish equivalence.
