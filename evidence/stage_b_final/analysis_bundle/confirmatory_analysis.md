# PBIR-GIFT Confirmatory Analysis

- Readiness: `{"complete_primary_cells": true, "expected_primary_cells": 6, "expected_seeds": 3, "seed_ready": true, "daily_returns_available": true}`
- Decision: **NOT SUPPORTED under the predeclared rule**
- Mean ΔSharpe: 0.0563
- Dataset means: `{'portfolio_5stocks': 0.07654535680181065, 'portfolio_5stocks2': 0.03597853970341509}`
- Non-overlapping window means: `{'W1': -0.012869469450925472, 'W3': 0.2725479990907645, 'W5': -0.09089268488200042}`
- Worst cell MDD deterioration: 2.907 pp

## Metric table

| Metric | Mean delta | Win rate | Hedges g | Exact p | Holm p | Block-bootstrap 95% CI |
|---|---:|---:|---:|---:|---:|---|
| sharpe | 0.0563 | 50.0% | 0.26100682011837645 | 0.2500 | 0.7500 | [-0.33075682640879545, 0.3846946780370701] |
| sortino | 0.0355 | 50.0% | 0.15763726491154906 | 0.3594 | 0.7500 | [-0.4215090607393657, 0.3824687724492473] |
| max_drawdown_improvement | 0.1040 | 83.3% | 0.055658853797495585 | 0.4531 | 0.7500 | [-1.6832199691986807, 2.3159647505120255] |
| total_return | 1.6856 | 83.3% | 0.5899913343957361 | 0.0781 | 0.3125 | [-1.9064764512670742, 6.602949765750009] |

## Caveats

- W2/W4/W6 are overlapping robustness windows, not independent units.
- D1 and D2 share AMZN, MSFT, and JNJ.
- A non-significant result is not evidence of equivalence.
