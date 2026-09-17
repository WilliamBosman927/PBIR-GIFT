# Stage-G independently tuned PPO analysis

## Decision

- Strict Pure-GIFT superiority over PPO-Tuned: **NOT SUPPORTED**.

## Primary contrast

Across 12 dataset-window cell means, ΔSharpe was -0.144; 3/12 cells favored Pure GIFT. The hierarchical block-bootstrap 95% interval was [-0.437, +0.132].

## Claim candidates

- Claim: Pure GIFT outperforms an independently tuned PPO baseline.
  - Decision: weaken.
  - Allowed wording: Report the observed mean effect, interval, and cell wins.
  - Forbidden wording: GIFT universally dominates optimally tuned PPO.
  - Uncertainty: two overlapping stock panels and rolling-window dependence.
