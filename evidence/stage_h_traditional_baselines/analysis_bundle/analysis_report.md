# Stage-H traditional baseline analysis

## Scope

All six pre-specified traditional/ML baselines are reported; no method is removed according to its observed result.

## Pre-declared main-text representatives

- SMA: trend-following rule.
- Bollinger Bands: volatility-band breakout rule.
- XGBoost: supervised machine-learning forecaster.

## Sharpe contrast decisions

- Pure GIFT − SMA: mean +0.281, 7/12 positive cells, 95% block-bootstrap CI [-0.367, +0.898], strict=False.
- Pure GIFT − WMA: mean +0.434, 7/12 positive cells, 95% block-bootstrap CI [-0.163, +1.014], strict=False.
- Pure GIFT − ATR: mean +0.446, 9/12 positive cells, 95% block-bootstrap CI [-0.132, +1.085], strict=False.
- Pure GIFT − Bollinger Bands: mean +0.270, 8/12 positive cells, 95% block-bootstrap CI [-0.252, +1.074], strict=False.
- Pure GIFT − Turn-of-the-Month: mean +0.185, 7/12 positive cells, 95% block-bootstrap CI [-0.599, +1.037], strict=False.
- Pure GIFT − XGBoost: mean +0.706, 7/12 positive cells, 95% block-bootstrap CI [+0.075, +1.481], strict=False.
- PBIR-GIFT − SMA: mean +0.340, 7/12 positive cells, 95% block-bootstrap CI [-0.395, +1.040], strict=False.
- PBIR-GIFT − WMA: mean +0.493, 7/12 positive cells, 95% block-bootstrap CI [-0.168, +1.120], strict=False.
- PBIR-GIFT − ATR: mean +0.505, 8/12 positive cells, 95% block-bootstrap CI [-0.074, +1.126], strict=False.
- PBIR-GIFT − Bollinger Bands: mean +0.328, 7/12 positive cells, 95% block-bootstrap CI [-0.253, +1.149], strict=False.
- PBIR-GIFT − Turn-of-the-Month: mean +0.243, 8/12 positive cells, 95% block-bootstrap CI [-0.440, +0.973], strict=False.
- PBIR-GIFT − XGBoost: mean +0.765, 9/12 positive cells, 95% block-bootstrap CI [+0.133, +1.492], strict=True.

## Claim candidates

- Allowed: method-specific effects with cell counts, intervals, and corrected tests.
- Forbidden: selecting only baselines that GIFT beats or claiming universal dominance.
