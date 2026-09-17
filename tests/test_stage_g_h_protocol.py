from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from hparam_sweep_runner import PARAMETER_GROUPS
from run_stage_g_tuned_ppo import METHOD, _candidate_dir, _select_from_results
from analyze_stage_g_tuned_ppo import _make_figures as make_stage_g_figures
from analyze_stage_h_traditional_baselines import _make_figures as make_stage_h_figures
from analyze_stage_i_final_comparison import (
    MAIN_METHODS,
    _make_figures as make_stage_i_figures,
)
from supplemental_experiment_utils import (
    PROJECT_DIR,
    ensure_isolated_output,
    holm_adjust,
    write_json,
)
from traditional_baselines import rule_weights, xgboost_features


def _frame() -> pd.DataFrame:
    dates = pd.bdate_range("2020-01-01", periods=100).strftime("%Y-%m-%d")
    close = np.linspace(10.0, 20.0, len(dates))
    return pd.DataFrame({
        "close": close,
        "high": close * 1.01,
        "low": close * 0.99,
        "volume": np.linspace(1000, 2000, len(dates)),
    }, index=dates)


def test_supplemental_output_cannot_overlap_locked_evidence(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="protected evidence"):
        ensure_isolated_output(PROJECT_DIR / "results" / "stage_b_final")
    ensure_isolated_output(tmp_path / "stage_g")


def test_rule_weights_are_long_only_and_lookahead_safe() -> None:
    tickers = ["A", "B"]
    frames = {ticker: _frame() for ticker in tickers}
    date = frames["A"].index[70]
    parameters = {"short_window": 20, "long_window": 60}
    before = rule_weights("sma", frames, tickers, date, parameters)
    assert before.shape == (3,)
    assert np.all(before >= 0)
    assert before.sum() == pytest.approx(1.0)

    changed = {ticker: frame.copy() for ticker, frame in frames.items()}
    for frame in changed.values():
        frame.loc[frame.index > date, "close"] *= 100.0
    after = rule_weights("sma", changed, tickers, date, parameters)
    np.testing.assert_allclose(before, after)


def test_xgboost_features_do_not_read_future_rows() -> None:
    frame = _frame()
    date = frame.index[70]
    before = xgboost_features(frame, date, 0, 2)
    changed = frame.copy()
    changed.loc[changed.index > date, ["close", "volume"]] *= 1000.0
    after = xgboost_features(changed, date, 0, 2)
    np.testing.assert_allclose(before, after)


def test_stage_g_selector_aggregates_seeds_and_freezes_prior(tmp_path: Path) -> None:
    seeds = [42, 123]
    selected_prior = {}
    for parameter, values in PARAMETER_GROUPS.items():
        for level, value in enumerate(values, start=1):
            score = 100.0 if level == 3 else float(level)
            for seed in seeds:
                path = _candidate_dir(tmp_path, parameter, level, seed) / "result.json"
                write_json(path, {
                    "status": "completed",
                    "method": METHOD,
                    "parameter": parameter,
                    "level": level,
                    "value": value,
                    "seed": seed,
                    "frozen_prior_ppo": dict(selected_prior),
                    "test_result": {
                        "test_sharpe": score,
                        "test_sortino": score,
                        "test_total_return": score,
                        "test_max_drawdown": 10.0,
                    },
                })
        selected_prior[parameter] = values[2]
    selected_path = _select_from_results(
        tmp_path, seeds, PROJECT_DIR / "config_tuning_D1_W1.yaml")
    artifact = yaml.safe_load(selected_path.read_text(encoding="utf-8"))
    assert artifact["ppo"] == {
        parameter: values[2] for parameter, values in PARAMETER_GROUPS.items()
    }
    assert artifact["protocol"]["selection_seeds"] == seeds


def test_holm_adjustment_is_monotone_and_bounded() -> None:
    adjusted = holm_adjust({"a": 0.01, "b": 0.02, "c": 0.5})
    assert adjusted["a"] == pytest.approx(0.03)
    assert adjusted["b"] == pytest.approx(0.04)
    assert adjusted["c"] == pytest.approx(0.5)
    assert all(0 <= value <= 1 for value in adjusted.values())


def test_supplemental_figure_builders_create_png_and_pdf(tmp_path: Path) -> None:
    datasets = ("portfolio_5stocks", "portfolio_5stocks2")
    windows = ("W1", "W2", "W3", "W4", "W5", "W6")
    all_methods = (
        "pbir", "pure_gift", "sma", "wma", "atr", "bollinger",
        "turn_of_month", "xgboost",
    )
    h_values = {
        (dataset, window, method, "test_sharpe"): float(index + method_index / 10)
        for dataset in datasets
        for index, window in enumerate(windows)
        for method_index, method in enumerate(all_methods)
    }
    make_stage_h_figures(tmp_path / "h", h_values)

    g_values = {
        (dataset, window, method, "test_sharpe"): [1.0, 1.1, 0.9]
        for dataset in datasets for window in windows
        for method in ("pure_gift", "ppo_tuned")
    }
    g_cells = [
        {"dataset": dataset, "window": window, "delta_sharpe": 0.2,
         "seed_sd_delta_sharpe": 0.1}
        for dataset in datasets for window in windows
    ]
    make_stage_g_figures(tmp_path / "g", g_values, g_cells)

    i_values = {
        (dataset, window, method): float(index + method_index / 10)
        for dataset in datasets
        for index, window in enumerate(windows)
        for method_index, method in enumerate(MAIN_METHODS)
    }
    contrasts = [
        {
            "label": f"contrast {index}", "mean_cell_delta_sharpe": 0.2,
            "bootstrap_ci95_low": -0.1, "bootstrap_ci95_high": 0.5,
            "confirmatory_family": index < 3,
        }
        for index in range(6)
    ]
    make_stage_i_figures(tmp_path / "i", i_values, contrasts)
    for stage, names in {
        "h": ("stage_h_sharpe_heatmap", "stage_h_method_mean_sharpe"),
        "g": ("stage_g_six_window_sharpe", "stage_g_sharpe_delta_forest"),
        "i": ("stage_i_main_method_rank_heatmap", "stage_i_key_contrast_forest"),
    }.items():
        for name in names:
            for suffix in ("png", "pdf"):
                path = tmp_path / stage / "figures" / f"{name}.{suffix}"
                assert path.is_file() and path.stat().st_size > 0
