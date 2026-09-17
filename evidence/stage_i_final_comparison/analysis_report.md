# Stage-I final comparison analysis

## Evidence hierarchy

1. Primary innovation: cached-code PBIR-GIFT versus Pure GIFT.
2. Mechanism control: Pure GIFT versus PPO-Matched.
3. Fairness control: Pure GIFT versus independently tuned PPO.
4. External context: SMA, Bollinger Bands, XGBoost, and supplementary baselines.

## Key contrast decisions

- PBIR-GIFT − Pure GIFT: ΔSharpe=+0.278, 95% CI [-0.010, +0.558], Holm p=0.0264, supported=False.
- Pure GIFT − PPO-Matched: ΔSharpe=+0.112, 95% CI [-0.342, +0.623], Holm p=0.5396, supported=False.
- Pure GIFT − PPO-Tuned: ΔSharpe=-0.144, 95% CI [-0.437, +0.132], Holm p=0.9775, supported=False.
- Pure GIFT − SMA: ΔSharpe=+0.281, 95% CI [-0.367, +0.898], Holm p=0.5369, supported=False.
- Pure GIFT − Bollinger Bands: ΔSharpe=+0.270, 95% CI [-0.252, +1.074], Holm p=0.5186, supported=False.
- Pure GIFT − XGBoost: ΔSharpe=+0.706, 95% CI [+0.075, +1.481], Holm p=0.1367, supported=False.

## Claim boundary

Report every planned contrast, including null or negative results. A method may be described as superior only for contrasts whose adjusted test and bootstrap interval both support the direction. Universal dominance is not tested.
