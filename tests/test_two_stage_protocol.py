import csv
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from aggregate_stage_b_results import _load_rows, generate_aggregate
from analyze_final_results import (
    _exact_sign_flip,
    collapse_cells,
    hierarchical_block_bootstrap,
)
from final_paired_runner import prepare_frozen_configs
from hparam_sweep_runner import PARAMETER_GROUPS, PROJECT_DIR
from stage_a_selection import select_stage_a_configuration


def _base_config(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _write_complete_stage_a_csv(path: Path) -> None:
    rows = []
    selected_prior = {}
    for parameter, values in PARAMETER_GROUPS.items():
        for level, value in enumerate(values, start=1):
            # Level 3 wins every group under the method-neutral mean objective.
            score = 10.0 if level == 3 else float(level)
            rows.append({
                "dataset": "portfolio_5stocks",
                "window": "W1",
                "trial_name": f"{parameter}_{level}",
                "parameter": parameter,
                "level": level,
                "value": value,
                "seed": 42,
                "status": "completed",
                "frozen_prior_ppo": json.dumps(
                    selected_prior, sort_keys=True, separators=(",", ":")),
                "pbir_test_sharpe": score + 0.2,
                "pure_gift_test_sharpe": score - 0.2,
                "pbir_test_sortino": score + 0.1,
                "pure_gift_test_sortino": score - 0.1,
                "pbir_test_total_return": score / 100,
                "pure_gift_test_total_return": score / 100,
                "pbir_test_max_drawdown": -0.2,
                "pure_gift_test_max_drawdown": -0.2,
            })
        selected_prior[parameter] = values[2]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_stage_a_selects_all_five_shared_values(tmp_path):
    summary = tmp_path / "paired_sweep_summary.csv"
    _write_complete_stage_a_csv(summary)
    selected_path = select_stage_a_configuration(
        summary,
        PROJECT_DIR / "config_tuning_D1_W1.yaml",
        tmp_path / "selection",
    )
    artifact = _base_config(selected_path)
    assert set(artifact["ppo"]) == set(PARAMETER_GROUPS)
    assert artifact["ppo"] == {
        parameter: values[2] for parameter, values in PARAMETER_GROUPS.items()
    }
    assert artifact["protocol"]["requires_combined_confirmation"] is False
    assert artifact["protocol"]["combined_confirmation"]["status"] == (
        "embedded_in_sequential_ofat")


def test_final_run_rejects_unconfirmed_selection():
    base = _base_config(PROJECT_DIR / "config_W1.yaml")
    artifact = {
        "protocol": {
            "stage": "A_selected",
            "validation_period": ["2020-01-01", "2020-06-30"],
            "requires_combined_confirmation": True,
        },
        "ppo": {key: values[2] for key, values in PARAMETER_GROUPS.items()},
    }
    with pytest.raises(ValueError, match="has not completed"):
        prepare_frozen_configs(base, artifact, "final")


def test_frozen_pair_differs_only_in_pbir_switch():
    base = _base_config(PROJECT_DIR / "config_W1.yaml")
    selected = {key: values[2] for key, values in PARAMETER_GROUPS.items()}
    artifact = {
        "protocol": {
            "stage": "A_selected",
            "validation_period": ["2020-01-01", "2020-06-30"],
            "combined_confirmation": {"status": "completed"},
        },
        "ppo": selected,
    }
    configs, frozen = prepare_frozen_configs(base, artifact, "final")
    assert frozen == selected
    assert configs["pbir"]["ppo"] == configs["pure_gift"]["ppo"]
    assert configs["pbir"]["experiment"]["run_ppo_baselines"] is False
    assert configs["pure_gift"]["experiment"]["run_ppo_baselines"] is False
    assert configs["pbir"]["pbir"]["enabled"] is True
    assert configs["pure_gift"]["pbir"]["enabled"] is False


def test_optional_ppo_baselines_run_once_in_pure_gift_branch():
    base = _base_config(PROJECT_DIR / "config_W1.yaml")
    artifact = {
        "protocol": {
            "stage": "A_selected",
            "validation_period": ["2020-01-01", "2020-06-30"],
            "combined_confirmation": {"status": "embedded_in_sequential_ofat"},
        },
        "ppo": {key: values[2] for key, values in PARAMETER_GROUPS.items()},
    }
    configs, _ = prepare_frozen_configs(
        base, artifact, "final", include_ppo_baselines=True)
    assert configs["pbir"]["experiment"]["run_ppo_baselines"] is False
    assert configs["pure_gift"]["experiment"]["run_ppo_baselines"] is True


def test_stage_a_validation_precedes_all_twelve_final_cells():
    tuning = _base_config(PROJECT_DIR / "config_tuning_D1_W1.yaml")
    validation_end = tuning["experiment"]["test_period"][1]
    paths = [
        PROJECT_DIR / f"config_W{index}.yaml" for index in range(1, 7)
    ] + [
        PROJECT_DIR / f"config2_W{index}.yaml" for index in range(1, 7)
    ]
    for path in paths:
        final_start = _base_config(path)["experiment"]["test_period"][0]
        assert validation_end < final_start, path.name


def test_stage_a_selection_rejects_incomplete_group(tmp_path):
    summary = tmp_path / "paired_sweep_summary.csv"
    _write_complete_stage_a_csv(summary)
    rows = list(csv.DictReader(summary.open(encoding="utf-8")))
    rows = [row for row in rows
            if not (row["parameter"] == "actor_lr" and row["level"] == "5")]
    with summary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="incomplete"):
        select_stage_a_configuration(
            summary, PROJECT_DIR / "config_tuning_D1_W1.yaml", tmp_path / "out")


def test_stage_b_aggregate_builds_png_and_pdf(tmp_path):
    for dataset in ("portfolio_5stocks", "portfolio_5stocks2"):
        for index in range(1, 7):
            root = tmp_path / dataset / f"W{index}" / "seed_42"
            root.mkdir(parents=True)
            row = {
                "dataset": dataset,
                "window": f"W{index}",
                "seed": 42,
                "status": "completed",
            }
            for method, offset in (("pbir", 0.2), ("pure_gift", 0.0)):
                row.update({
                    f"{method}_test_sharpe": index + offset,
                    f"{method}_test_sortino": index + 0.5 + offset,
                    f"{method}_test_total_return": index / 100 + offset / 100,
                    f"{method}_test_max_drawdown": -index / 100 + offset / 100,
                })
            (root / "final_pair_summary.json").write_text(
                json.dumps(row), encoding="utf-8")
    rows = _load_rows(tmp_path)
    assert len(rows) == 12
    figures = generate_aggregate(rows, tmp_path / "aggregate")
    assert {path.suffix for path in figures} == {".png", ".pdf"}
    assert all(path.stat().st_size > 0 for path in figures)


def test_exact_sign_flip_and_cell_collapse():
    rows = []
    for dataset in ("portfolio_5stocks", "portfolio_5stocks2"):
        for window in ("W1", "W3", "W5"):
            for seed in (42, 123):
                rows.append({
                    "dataset": dataset, "window": window, "seed": seed,
                    "pbir_test_sharpe": 1.5,
                    "pure_gift_test_sharpe": 1.0,
                    "pbir_test_sortino": 1.7,
                    "pure_gift_test_sortino": 1.1,
                    "pbir_test_max_drawdown": 8.0,
                    "pure_gift_test_max_drawdown": 10.0,
                    "pbir_test_total_return": 12.0,
                    "pure_gift_test_total_return": 10.0,
                })
    cells = collapse_cells(rows)
    assert len(cells) == 6
    assert all(cell["delta_sharpe"] == pytest.approx(0.5) for cell in cells)
    assert _exact_sign_flip(
        [cell["delta_sharpe"] for cell in cells]) == pytest.approx(1 / 64)


def test_hierarchical_block_bootstrap_reads_daily_returns(tmp_path):
    rows = []
    base_noise = np.tile(np.array([-0.01, 0.004, 0.008, -0.002, 0.006]), 8)
    for dataset in ("portfolio_5stocks", "portfolio_5stocks2"):
        for window in ("W1", "W3", "W5"):
            for seed in (42, 123):
                pbir_dir = tmp_path / dataset / window / str(seed) / "pbir"
                gift_dir = tmp_path / dataset / window / str(seed) / "gift"
                pbir_dir.mkdir(parents=True)
                gift_dir.mkdir(parents=True)
                (pbir_dir / "final_comparison.json").write_text(json.dumps({
                    "daily_returns": {
                        "method": (base_noise + 0.001).tolist()},
                }), encoding="utf-8")
                (gift_dir / "final_comparison.json").write_text(json.dumps({
                    "daily_returns": {"method": base_noise.tolist()},
                }), encoding="utf-8")
                rows.append({
                    "dataset": dataset, "window": window, "seed": seed,
                    "pbir_experiment_dir": str(pbir_dir),
                    "pure_gift_experiment_dir": str(gift_dir),
                })
    result = hierarchical_block_bootstrap(
        rows, "sharpe", resamples=200, block_length=5, random_seed=42)
    assert result is not None
    assert result["resamples"] == 200
    assert result["ci95"][0] > 0
