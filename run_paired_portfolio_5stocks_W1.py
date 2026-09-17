"""Paired PBIR-GIFT/Pure GIFT sweep: portfolio_5stocks, W1."""

from paired_hparam_sweep_runner import run_paired_sweep


if __name__ == "__main__":
    raise SystemExit(run_paired_sweep("config_W1.yaml", "portfolio_5stocks", "W1"))
