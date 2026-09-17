# PBIR-GIFT Fixed-Code Controlled Replication

- Readiness: `{"complete_primary_cells": true, "expected_primary_cells": 6, "expected_seeds": 3, "seed_ready": true, "daily_returns_available": true}`
- Decision: **DOES NOT MEET the controlled-replication decision gate**
- Mean ΔSharpe: 0.3683
- Dataset means: `{'portfolio_5stocks': 0.23428341114995158, 'portfolio_5stocks2': 0.5022593835292943}`
- Non-overlapping window means: `{'W1': 0.289858938033727, 'W3': 0.388345063618976, 'W5': 0.4266101903661659}`
- Worst cell MDD deterioration: 2.375 pp

## Metric table

| Metric | Mean delta | Win rate | Hedges g | Exact p | Holm p | Block-bootstrap 95% CI |
|---|---:|---:|---:|---:|---:|---|
| sharpe | 0.3683 | 83.3% | 1.2266816951963464 | 0.0312 | 0.1250 | [-0.05278320870954665, 0.7850472454553507] |
| sortino | 0.3714 | 83.3% | 1.0642692128359406 | 0.0312 | 0.1250 | [-0.1276173973301311, 0.8289094363247749] |
| max_drawdown_improvement | 0.4056 | 66.7% | 0.18275285739917427 | 0.3125 | 0.3125 | [-2.0249789468334862, 2.2163387571273536] |
| total_return | 3.1730 | 83.3% | 0.5859056063797908 | 0.0469 | 0.1250 | [-1.2846000631217358, 10.083387038548356] |

## Caveats

- W2/W4/W6 are overlapping robustness windows, not independent units.
- D1 and D2 share AMZN, MSFT, and JNJ.
- A non-significant result is not evidence of equivalence.
