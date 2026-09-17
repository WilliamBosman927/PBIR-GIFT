# Stage-B PBIR-GIFT vs Pure GIFT: results summary

## Bottom line

The predeclared confirmatory claim is **NOT SUPPORTED under the predeclared rule**. Across the six non-overlapping dataset-window cells (D1/D2 × W1/W3/W5), the mean paired ΔSharpe was **+0.056** (PBIR-GIFT − Pure GIFT), with a 20-day hierarchical moving-block-bootstrap 95% CI of **[-0.331, +0.385]**. The interval crosses zero.

This is a mixed result, not evidence that PBIR reliably improves the primary risk-adjusted objective under the current pipeline. It is also not evidence of equivalence or of no effect.

## Completion and protocol checks

- Completed paired runs: **36/36** (12 cells × 3 seeds).
- Seeds in every cell: **42, 123, 456**; mismatches: **0**.
- Frozen Stage-A selection hash count: **1**; PPO parameter-set count: **1**.
- Final daily-return artifacts missing: **0**; invalid status entries: **0**.
- Confirmatory units are W1/W3/W5. W2/W4/W6 are overlapping robustness windows only.

## Confirmatory decision rule

| Criterion | Observed result | Pass? |
|---|---:|:---:|
| Complete primary cells and three seeds | 6 cells, 3 seeds each | Yes |
| ΔSharpe bootstrap CI entirely above zero | [-0.331, +0.385] | No |
| Both dataset ΔSharpe means positive | D1 +0.077; D2 +0.036 | Yes |
| At least 2/3 non-overlapping windows positive | W3 | No |
| Worst cell MDD deterioration ≤ 5 pp | 2.907 pp | Yes |

## Confirmatory effect estimates

| Endpoint (PBIR direction positive) | Mean cell delta | Cell win rate | Hedges g (paired) | exact p | Holm p | 20-day block-bootstrap 95% CI |
|---|---:|---:|---:|---:|---:|---:|
| sharpe | +0.056 | 50.0% | +0.261 | 0.2500 | 0.7500 | [-0.331, +0.385] |
| sortino | +0.036 | 50.0% | +0.158 | 0.3594 | 0.7500 | [-0.422, +0.382] |
| max_drawdown_improvement | +0.104 | 83.3% | +0.056 | 0.4531 | 0.7500 | [-1.683, +2.316] |
| total_return | +1.686 | 83.3% | +0.590 | 0.0781 | 0.3125 | [-1.906, +6.603] |

Sharpe has an exact one-sided sign-flip p-value of 0.2500 (Holm-adjusted 0.7500); the small number of independent cells makes this test intentionally coarse.

## Cell-level ΔSharpe

| Dataset | Window | Confirmatory? | ΔSharpe | Seed SD | ΔReturn (pp) | MDD improvement (pp) |
|---|---|:---:|---:|---:|---:|---:|
| D1 | W1 | Yes | +0.076 | 0.140 | +1.177 | +0.819 |
| D1 | W2 | No | -0.087 | 0.143 | -2.390 | -0.557 |
| D1 | W3 | Yes | +0.279 | 0.391 | +2.751 | +1.619 |
| D1 | W4 | No | -0.461 | 0.163 | -3.804 | -2.610 |
| D1 | W5 | Yes | -0.125 | 0.395 | +5.143 | -2.907 |
| D1 | W6 | No | +0.554 | 0.580 | +3.549 | +2.096 |
| D2 | W1 | Yes | -0.102 | 0.404 | -1.774 | +0.277 |
| D2 | W2 | No | +0.668 | 0.345 | +3.532 | +1.149 |
| D2 | W3 | Yes | +0.267 | 0.772 | +2.725 | +0.792 |
| D2 | W4 | No | -0.150 | 0.538 | -0.991 | -0.162 |
| D2 | W5 | Yes | -0.057 | 0.987 | +0.091 | +0.023 |
| D2 | W6 | No | -0.159 | 0.178 | -1.072 | -0.046 |

## Interpretation and next action

- W3 is positive in both datasets (D1 +0.279; D2 +0.267), but W1 and W5 have negative pooled effects. The apparent mean advantage is therefore regime/window-specific rather than stable.
- Return has a positive mean delta (+1.686 pp) and 5/6 positive confirmatory cells, but its 95% interval crosses zero. Drawdown has no reliable aggregate change (mean improvement +0.104 pp).
- Preserve these outputs as the locked primary result. Do not retune PBIR from W1/W3/W5 outcomes and present the revised result as confirmatory.
- The next economical, scientifically useful study is a *code-cached paired replication* on W1/W3/W5: generate the LLM code once per seed, reuse exactly that code for PBIR and Pure GIFT, and counterbalance method order. This isolates the PBIR transformation from LLM sampling/order noise.

## Important limitations

- D1 and D2 share AMZN, MSFT, and JNJ; they are not independent markets.
- The W2/W4/W6 windows overlap with the three primary windows and must not be counted as six additional independent tests.
- The current final runs generate code separately for the two methods, and the observed method order is recorded in the validation manifest. This means LLM-generation variability can remain confounded with method; the comparison is a pipeline comparison, not a fully isolated PBIR-only causal test.
- Each evaluation segment is 125 trading days. The confidence intervals correctly reflect sizable uncertainty; they do not establish equivalence.
