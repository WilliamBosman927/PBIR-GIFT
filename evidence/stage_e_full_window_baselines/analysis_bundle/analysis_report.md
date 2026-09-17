# Stage-E full-window reproduction report

## Decisions

- Strict six-window GIFT superiority: **NOT SUPPORTED**.
- Light Mix directional reproduction of the paper: **NOT SUPPORTED**.

These are empirical decisions, not indicators that the program did or did not run correctly.

## Primary comparison

The primary comparison is Stage-B Pure GIFT against PPO-TestOnly, which matches the paper's main PPO baseline protocol more closely than the additional PPO-Train+FineTune stress test.

Across 12 dataset-window cell means, ΔSharpe was +0.112; 8/12 cells favored GIFT. The 95% hierarchical block-bootstrap interval was [-0.342, +0.623], and the exact one-sided sign-flip p-value was 0.2698.

## Light Mix reproduction rule

Mean ΔSharpe: +0.065; positive Sharpe windows: 3/6; windows where GIFT won at least four of five paper metrics: 3/6.

## Contrast summary

| Contrast | Mean ΔSharpe | Positive cells | Hedges g | Exact p | 95% bootstrap CI |
|---|---:|---:|---:|---:|---:|
| pure_gift_vs_ppo_test_only | +0.112 | 8/12 | +0.166 | 0.2698 | [-0.342, +0.623] |
| pure_gift_vs_ppo_train_plus_test | +0.450 | 11/12 | +0.981 | 0.0017 | [+0.049, +0.894] |
| pure_gift_vs_equal_weight | -0.130 | 2/12 | -0.353 | 0.8906 | [-0.416, +0.142] |

## Claim boundary

- Stage-E completes the missing even-window PPO and equal-weight controls; it does not rerun or alter locked Stage-B/Stage-D evidence.
- The original-paper overlay is a descriptive pattern check. Differences in software, sampled LLM code, random seeds, and environment can prevent exact numerical reproduction.
- GIFT-vs-PPO evidence does not establish PBIR-vs-GIFT superiority. That question remains governed by the locked fixed-code Stage-C comparison.
- D2 overlaps D1 by three assets, so the two portfolios are not fully independent replications.
