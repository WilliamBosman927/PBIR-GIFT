# Stage-F full-window fixed-code PBIR replication

## Primary decision

The predeclared strict PBIR superiority gate is **NOT SUPPORTED**.

Across 12 dataset-window cell means, PBIR-GIFT minus Pure GIFT Sharpe was +0.278; 10/12 cells favored PBIR-GIFT. The hierarchical circular-block bootstrap 95% interval was [-0.010, +0.558], with positive probability 97.0%. Paired Hedges g was 0.9401697585748268.

The decision rule requires a positive bootstrap lower bound, positive means in both datasets, at least 8/12 positive cells, and no cell with more than 5 percentage points of MDD deterioration.

## Odd-to-even generalization

- Locked Stage-C odd windows: mean ΔSharpe +0.368; 5/6 positive cells.
- New Stage-F even windows: mean ΔSharpe +0.188; 5/6 positive cells.

## Metric contrasts

| Metric | Mean delta | Positive cells | Hedges g | Exact p | Holm p | 95% block-bootstrap CI |
|---|---:|---:|---:|---:|---:|---:|
| sharpe | +0.278 | 10/12 | +0.940 | 0.0044 | 0.0220 | [-0.010, +0.558] |
| sortino | +0.276 | 10/12 | +0.845 | 0.0059 | 0.0220 | [-0.051, +0.586] |
| total_return | +2.205 | 8/12 | +0.578 | 0.0129 | 0.0259 | [-0.511, +6.046] |
| max_drawdown_improvement | +0.647 | 9/12 | +0.400 | 0.0813 | 0.0813 | [-0.866, +1.998] |
| calmar | +0.675 | 9/12 | +0.673 | 0.0054 | 0.0220 | [-0.238, +1.811] |

## Claim boundary

- This is the core PBIR ablation: both methods receive the exact same cached LLM code and reward rules; only potential-based shaping is toggled.
- A supported result permits a claim of robust benefit under these two portfolios and rolling windows, not universal superiority in all markets.
- A non-supported result should be reported as mixed or regime-dependent; it must not be rewritten as equivalence or hidden.
- Stage-C is verified against its cryptographic lock and is never modified by this analysis. All new files live in Stage-F.
