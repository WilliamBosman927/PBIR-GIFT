from __future__ import annotations

import tempfile
import unittest
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import yaml

from analyze_stage_f_full_window_pbir import (
    DATASETS,
    SEEDS,
    WINDOWS,
    _exact_sign_flip,
    _hierarchical_bootstrap,
    main as analyze_main,
)
from lock_stage_c_results import build as build_stage_c_lock
from lock_stage_f_results import (
    build as build_stage_f_lock,
    validate_stage_f_cells,
    verify as verify_stage_f_lock,
)
from metrics import calmar_ratio, max_drawdown, sharpe_ratio, sortino_ratio
from run_stage_f_even_window_cached_pbir import _guard_output_root


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_mock_pair(
        root: Path, dataset: str, window: str, seed: int,
) -> None:
    cell = root / dataset / window / f"seed_{seed}"
    artifact_dir = cell / "shared_artifact"
    artifact_dir.mkdir(parents=True)
    code_path = artifact_dir / "code.py"
    reward_path = artifact_dir / "reward_config.json"
    code_path.write_text(
        "def revise_state(s):\n    return s\n\n"
        "def intrinsic_reward(updated_s):\n    return 0.0\n",
        encoding="utf-8")
    reward_path.write_text(json.dumps({"reward_rules": [], "lambda": 0.5}),
                           encoding="utf-8")
    artifact = {
        "code_file": code_path.name,
        "code_sha256": _digest(code_path),
        "reward_config_file": reward_path.name,
        "reward_config_sha256": _digest(reward_path),
        "source": {
            "dataset": dataset, "window": window, "seed": seed,
            "method": "pbir",
        },
    }
    artifact_path = artifact_dir / "shared_artifact.json"
    artifact_path.write_text(json.dumps(artifact), encoding="utf-8")
    artifact_sha = _digest(artifact_path)
    fixed = {
        "manifest_sha256": artifact_sha,
        "code_sha256": artifact["code_sha256"],
        "reward_config_sha256": artifact["reward_config_sha256"],
    }

    base = np.linspace(-0.001, 0.001, 100)
    method_returns = {
        "pure_gift": base + 0.0002,
        "pbir": base + 0.0007,
    }
    pair = {
        "dataset": dataset,
        "window": window,
        "seed": seed,
        "status": "completed",
        "evaluation_role": "controlled_replication",
        "shared_artifact_sha256": artifact_sha,
        "shared_code_sha256": artifact["code_sha256"],
        "shared_reward_config_sha256": artifact["reward_config_sha256"],
    }
    for method in ("pbir", "pure_gift"):
        method_dir = cell / "methods" / method
        method_dir.mkdir(parents=True)
        enabled = method == "pbir"
        config = {
            "experiment": {"max_iterations": 1, "sample_count": 1},
            "ppo": {"gamma": 0.99},
            "pbir": {"enabled": enabled, "gamma": 0.99, "scale": 1.0},
        }
        config_path = method_dir / "resolved_config.yaml"
        config_path.write_text(yaml.safe_dump(config, sort_keys=False),
                               encoding="utf-8")
        config_sha = hashlib.sha256(
            yaml.safe_dump(config, sort_keys=True).encode("utf-8")).hexdigest()
        values = method_returns[method]
        result = {
            "test_sharpe": sharpe_ratio(values),
            "test_sortino": sortino_ratio(values),
            "test_max_drawdown": max_drawdown(values),
            "test_total_return": (float(np.prod(1 + values)) - 1.0) * 100,
            "test_calmar": calmar_ratio(values),
        }
        (method_dir / "summary.json").write_text(json.dumps({
            "iterations": 1,
            "llm_call_attempts": 0,
            "llm_call_successes": 0,
            "pbir": {"enabled": enabled},
            "fixed_shared_artifact": fixed,
        }), encoding="utf-8")
        (method_dir / "final_comparison.json").write_text(json.dumps({
            "pbir": {"enabled": enabled},
            "fixed_shared_artifact": fixed,
            "eval_period": ["2020-01-01", "2020-06-30"],
            "test_result": result,
            "daily_returns": {"method": values.tolist()},
        }), encoding="utf-8")
        (method_dir / "run_protocol.json").write_text(
            json.dumps({"config_sha256": config_sha}), encoding="utf-8")
        (method_dir / "best_model.pt").write_bytes(b"mock-model")
        pair[f"{method}_config_sha256"] = config_sha
        pair[f"{method}_llm_call_attempts"] = 0
        pair[f"{method}_llm_call_successes"] = 0
        for key, value in result.items():
            if key != "test_calmar":
                pair[f"{method}_{key}"] = value
    (cell / "final_pair_summary.json").write_text(
        json.dumps(pair), encoding="utf-8")


class StageFFullWindowPBIRTests(unittest.TestCase):
    def test_stage_f_output_is_disjoint_from_prior_evidence(self) -> None:
        project = Path(__file__).resolve().parents[1]
        stage_b = (project / "results" / "stage_b_final").resolve()
        with self.assertRaisesRegex(ValueError, "isolated"):
            _guard_output_root(stage_b, stage_b)
        with self.assertRaisesRegex(ValueError, "isolated"):
            _guard_output_root((project / "results").resolve(), stage_b)
        with tempfile.TemporaryDirectory() as directory:
            _guard_output_root(Path(directory), stage_b)

    def test_incomplete_stage_f_cannot_be_locked(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "Stage-F grid"):
                validate_stage_f_cells(Path(directory))

    def test_exact_sign_flip_uses_twelve_cell_means(self) -> None:
        values = np.ones(12, dtype=float)
        self.assertAlmostEqual(_exact_sign_flip(values), 1.0 / 4096.0)

    def test_hierarchical_bootstrap_preserves_positive_effect(self) -> None:
        pairs = []
        for dataset in DATASETS:
            for window in WINDOWS:
                for seed in SEEDS:
                    gift = np.full(100, 0.0002)
                    pbir = np.full(100, 0.0010)
                    # Small deterministic variation avoids a zero Sharpe denominator.
                    variation = np.linspace(-0.0001, 0.0001, 100)
                    pairs.append({
                        "dataset": dataset,
                        "window": window,
                        "seed": seed,
                        "returns": {
                            "pbir": pbir + variation,
                            "pure_gift": gift + variation * 2.0,
                        },
                    })
        result = _hierarchical_bootstrap(
            pairs, "total_return", resamples=100, block_length=10,
            random_seed=123)
        self.assertGreater(result["ci95"][0], 0.0)
        self.assertEqual(result["probability_positive"], 1.0)

    def test_analysis_and_lock_complete_synthetic_grid(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            stage_c = base / "stage_c"
            stage_f = base / "stage_f"
            for dataset in DATASETS:
                for window in ("W1", "W3", "W5"):
                    for seed in SEEDS:
                        _write_mock_pair(stage_c, dataset, window, seed)
                for window in ("W2", "W4", "W6"):
                    for seed in SEEDS:
                        _write_mock_pair(stage_f, dataset, window, seed)

            stage_c_lock = base / "locks" / "stage_c.json"
            build_stage_c_lock(stage_c, stage_c_lock)
            output = stage_f / "analysis_bundle"
            arguments = [
                "analyze_stage_f_full_window_pbir.py",
                "--stage-c-dir", str(stage_c),
                "--stage-c-lock", str(stage_c_lock),
                "--stage-f-dir", str(stage_f),
                "--output-dir", str(output),
                "--resamples", "100",
                "--block-length", "10",
            ]
            with patch("sys.argv", arguments):
                self.assertEqual(analyze_main(), 0)
            inference = json.loads(
                (output / "stage_f_inference.json").read_text(encoding="utf-8"))
            self.assertEqual(inference["complete_seed_pairs"], 36)
            self.assertEqual(inference["dataset_window_cells"], 12)
            self.assertTrue(
                inference["primary"]["strict_full_window_supported"])

            stage_f_lock = base / "locks" / "stage_f.json"
            manifest = build_stage_f_lock(stage_f, stage_f_lock)
            self.assertEqual(manifest["completed_even_window_paired_cells"], 18)
            verify_stage_f_lock(stage_f_lock)


if __name__ == "__main__":
    unittest.main()
