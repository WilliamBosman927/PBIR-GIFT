import json
from pathlib import Path

import pytest

from final_paired_runner import materialize_shared_code_artifact
from gift_controller import GIFTController
from lock_stage_b_results import build_lock, verify_lock


SHARED_CODE = """import numpy as np
def revise_state(s):
    return np.concatenate([s, [float(np.mean(s))]])
def intrinsic_reward(updated_s):
    return float(np.tanh(updated_s[-1]))
"""


def _write_stage_b_source(root: Path) -> Path:
    cell = root / "portfolio_5stocks" / "W1" / "seed_42"
    method = cell / "methods" / "pbir"
    iteration = method / "iteration_2"
    iteration.mkdir(parents=True)
    (cell / "final_pair_summary.json").write_text(json.dumps({
        "status": "completed",
        "selection_sha256": "selection-hash",
    }), encoding="utf-8")
    (method / "summary.json").write_text(json.dumps({
        "best_iteration": 2,
    }), encoding="utf-8")
    (iteration / "code.py").write_text(SHARED_CODE, encoding="utf-8")
    (iteration / "config.json").write_text(json.dumps({
        "reward_rules": [],
        "lambda": 0.5,
    }), encoding="utf-8")
    return cell


def test_stage_c_materializes_one_verified_shared_artifact(tmp_path):
    source = tmp_path / "stage_b"
    _write_stage_b_source(source)
    manifest_path, manifest, manifest_sha = materialize_shared_code_artifact(
        source,
        tmp_path / "stage_c" / "shared_artifact",
        "portfolio_5stocks",
        "W1",
        42,
    )

    assert manifest_path.is_file()
    assert manifest["source"]["method"] == "pbir"
    assert manifest["source"]["best_iteration"] == 2
    assert len(manifest_sha) == 64
    assert (manifest_path.parent / "code.py").read_text(
        encoding="utf-8") == SHARED_CODE

    controller = GIFTController({
        "experiment": {
            "fixed_artifact_path": str(manifest_path),
            "fixed_artifact_sha256": manifest_sha,
            "max_iterations": 5,
            "sample_count": 3,
        },
        "portfolio": {"default_lambda": 0.5},
        "ppo": {"gamma": 0.99},
        "pbir": {"enabled": True},
        "data": {"tickers": ["AAA"]},
    }, str(tmp_path / "controller"), seed=42)
    assert controller.fixed_artifact is not None
    assert controller.max_iterations == 1
    assert controller.sample_count == 1
    assert controller.llm_call_attempts == 0
    assert controller.fixed_artifact["manifest_sha256"] == manifest_sha


def test_fixed_artifact_rejects_tampering(tmp_path):
    source = tmp_path / "stage_b"
    _write_stage_b_source(source)
    manifest_path, _, manifest_sha = materialize_shared_code_artifact(
        source, tmp_path / "artifact", "portfolio_5stocks", "W1", 42)
    (manifest_path.parent / "code.py").write_text(
        SHARED_CODE + "\n# changed\n", encoding="utf-8")

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        GIFTController({
            "experiment": {
                "fixed_artifact_path": str(manifest_path),
                "fixed_artifact_sha256": manifest_sha,
            },
            "portfolio": {"default_lambda": 0.5},
            "ppo": {"gamma": 0.99},
            "data": {"tickers": ["AAA"]},
        }, str(tmp_path / "tampered"), seed=42)


def test_stage_b_lock_detects_changed_evidence(tmp_path):
    root = tmp_path / "stage_b"
    for dataset in ("portfolio_5stocks", "portfolio_5stocks2"):
        for window in range(1, 7):
            for seed in (42, 123, 456):
                cell = root / dataset / f"W{window}" / f"seed_{seed}"
                cell.mkdir(parents=True)
                (cell / "final_pair_summary.json").write_text(json.dumps({
                    "status": "completed",
                    "dataset": dataset,
                    "window": f"W{window}",
                    "seed": seed,
                }), encoding="utf-8")
    output = tmp_path / "lock" / "manifest.json"
    manifest = build_lock(root, output)
    assert manifest["completed_paired_cells"] == 36
    verify_lock(output)

    first = root / "portfolio_5stocks" / "W1" / "seed_42" / "final_pair_summary.json"
    first.write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="changed"):
        verify_lock(output)
