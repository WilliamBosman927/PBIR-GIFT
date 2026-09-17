# Stage-D baseline analysis report

## Decision

Primary GIFT-vs-strong-PPO claim: **NOT SUPPORTED**.
This is a statistical decision, not a program failure.

The Stage-B Pure-GIFT pipeline had a directionally favorable mean Sharpe contrast against PPO-Train+FineTune (+0.470); all six cell means were positive and Hedges g was +1.002. However, the predeclared 95% hierarchical block-bootstrap interval [-0.038, +1.124] includes zero. The data therefore do not support a confirmatory superiority claim.

## Audit and estimands

- 18/18 dataset-window-seed cells completed.
- Each method has 105 aligned out-of-sample daily returns per cell.
- Stage-D pure-PPO and equal-weight baselines made zero LLM calls.
- Inference is based on six dataset-window cell means; daily returns and   seeds are not treated as independent studies.

## Contrast results

| Contrast | Mean ΔSharpe | Cell wins | Seed-run wins | Hedges g | Raw p | Holm p | 95% block-bootstrap CI |
|---|---:|---:|---:|---:|---:|---:|---:|
| gift_pipeline_vs_ppo_train_plus_test | +0.470 | 100.0% | 66.7% | +1.002 | 0.0156 | 0.0938 | [-0.038, +1.124] |
| gift_pipeline_vs_ppo_test_only | +0.305 | 66.7% | 55.6% | +0.479 | 0.1562 | 0.4688 | [-0.215, +1.012] |
| pbir_fixed_vs_ppo_train_plus_test | +0.526 | 100.0% | 77.8% | +1.166 | 0.0156 | 0.0938 | [-0.083, +1.140] |
| pbir_fixed_vs_gift_fixed | +0.368 | 83.3% | 72.2% | +1.227 | 0.0312 | 0.1250 | [-0.052, +0.784] |
| gift_pipeline_vs_equal_weight | -0.155 | 16.7% | 33.3% | -0.716 | 0.9688 | 1.0000 | [-0.463, +0.151] |
| pbir_fixed_vs_equal_weight | -0.099 | 33.3% | 44.4% | -0.270 | 0.7656 | 1.0000 | [-0.474, +0.214] |

## Absolute performance check

| Method | Mean Sharpe across 6 cells | SD across cells | Positive cells | Best cells |
|---|---:|---:|---:|---:|
| Pure GIFT (Stage B pipeline) | +0.965 | 1.801 | 4/6 | 0/6 |
| PBIR-GIFT (Stage C fixed code) | +1.021 | 1.626 | 4/6 | 1/6 |
| Pure GIFT (Stage C fixed code) | +0.653 | 1.505 | 4/6 | 0/6 |
| PPO (train + fine-tune) | +0.495 | 1.713 | 3/6 | 0/6 |
| PPO (test-period training) | +0.659 | 1.725 | 3/6 | 1/6 |
| Equal Weight | +1.120 | 1.873 | 4/6 | 4/6 |

Equal Weight is a material application-level challenge: Pure GIFT's mean cell contrast is -0.155 (17% of cells positive), and PBIR-GIFT's is -0.099 (33% positive). Thus evidence of an advantage over PPO must not be rewritten as an advantage over simple portfolio allocation.

## Interpretation and claim boundary

- Pure GIFT versus strong PPO: large positive point estimate and 6/6   positive cell means, but uncertainty still touches zero; directional,   not confirmatory.
- PBIR versus Pure GIFT: the locked Stage-C result remains directionally   favorable (+0.368; 5/6 cells), but its 95% interval also crosses zero.
- A confidence interval crossing zero is inconclusive; it is neither   evidence of equivalence nor proof of no effect.
- With the current evidence, valid wording is: GIFT/PBIR showed favorable   contrasts to PPO in these selected cells, with substantial seed and   cell uncertainty. Claims of universal superiority are not supported.

## Recommended next action

Lock this completed Stage-D evidence. Do not add seeds merely to force significance. If another confirmatory run is affordable, predeclare one genuinely unseen time period or independent asset universe, keep all configurations fixed, and include Equal Weight as a mandatory baseline.
