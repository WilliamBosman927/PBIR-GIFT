"""Run the missing even-window baselines for the full-window replication.

Stage-E is deliberately isolated from the locked Stage-B/C/D trees.  It reads
the frozen PPO configuration and evaluation split from locked Stage-B cells,
then trains only PPO-TestOnly and PPO-Train+FineTune and evaluates Equal
Weight for W2/W4/W6.  It never calls an LLM and never retrains GIFT or PBIR.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import yaml

from hparam_sweep_runner import PROJECT_DIR
from lock_stage_b_results import verify_lock as verify_stage_b_lock
from run_stage_d_ppo_baselines import (
    _canonical_config_sha,
    _complete,
    _equal_weight_result,
    _method_record,
    _normalize_result,
    _read_json,
    _split_period,
    _write_json,
)


DATASETS = ("portfolio_5stocks", "portfolio_5stocks2")
EVEN_WINDOWS = ("W2", "W4", "W6")
DEFAULT_SEEDS = (42, 123, 456)
PROTECTED_RESULT_DIRS = (
    "results/stage_b_final",
    "results/stage_c_cached_replication",
    "results/stage_d_ppo_baselines",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument(
        "--windows", nargs="+", choices=EVEN_WINDOWS,
        default=list(EVEN_WINDOWS))
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--stage-b-dir", default="results/stage_b_final")
    parser.add_argument(
        "--stage-b-lock",
        default="results/stage_b_lock/stage_b_lock_manifest.json")
    parser.add_argument(
        "--results-dir", default="results/stage_e_full_window_baselines")
    parser.add_argument(
        "--device", choices=("cuda", "auto", "cpu"), default="cuda")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument(
        "--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _resolve(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_DIR / path).resolve()


def _assert_isolated_output(results_root: Path, stage_b_root: Path) -> None:
    protected = {stage_b_root}
    protected.update(_resolve(path) for path in PROTECTED_RESULT_DIRS)
    for path in protected:
        if (results_root == path or path in results_root.parents
                or results_root in path.parents):
            raise ValueError(
                "Stage-E output must be disjoint from every locked result tree: "
                f"{path}")


def _load_locked_stage_b(
        stage_b_root: Path, lock_path: Path,
) -> dict[str, Any]:
    if not lock_path.is_file():
        raise FileNotFoundError(f"Stage-B lock manifest not found: {lock_path}")
    verify_stage_b_lock(lock_path)
    manifest = _read_json(lock_path)
    if Path(str(manifest["source_root"])).resolve() != stage_b_root:
        raise ValueError("Stage-B lock manifest points to a different source root")
    if manifest.get("completed_paired_cells") != 36:
        raise ValueError("Stage-B lock does not contain all 36 paired cells")
    return manifest


def _source_config(
        stage_b_root: Path, dataset: str, window: str, seed: int,
) -> tuple[dict[str, Any], dict[str, Any], Path, Path]:
    cell = stage_b_root / dataset / window / f"seed_{seed}"
    pair_path = cell / "final_pair_summary.json"
    if not pair_path.is_file():
        raise FileNotFoundError(f"Incomplete locked Stage-B source: {cell}")
    pair = _read_json(pair_path)
    if pair.get("status") not in {"completed", "skipped_existing"}:
        raise ValueError(f"Incomplete Stage-B pair: {pair_path}")
    candidates = [
        cell / "methods" / "pure_gift" / "resolved_config.yaml",
        Path(str(pair.get("pure_gift_resolved_config_path", ""))),
        Path(str(pair.get("pure_gift_config_path", ""))),
    ]
    config_path = next((
        path if path.is_absolute() else (PROJECT_DIR / path)
        for path in candidates if str(path) not in {"", "."}
        and (path if path.is_absolute() else (PROJECT_DIR / path)).is_file()
    ), None)
    if config_path is None:
        raise FileNotFoundError(
            f"Stage-B Pure-GIFT config is missing for {dataset}/{window}/seed_{seed}")
    config_path = config_path.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(config, dict):
        raise TypeError(f"Stage-B config must be a mapping: {config_path}")
    observed_sha = _canonical_config_sha(config)
    expected_sha = pair.get("pure_gift_config_sha256")
    if observed_sha != expected_sha:
        raise ValueError(
            f"Locked Stage-B config hash mismatch: {config_path}\n"
            f"expected={expected_sha}\nobserved={observed_sha}")
    return pair, config, pair_path, config_path


def _write_registry(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = _parse_args()
    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        os.environ["GIFT_TORCH_DEVICE"] = "cpu"
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        os.environ["GIFT_TORCH_DEVICE"] = "cuda"

    stage_b_root = _resolve(args.stage_b_dir)
    stage_b_lock = _resolve(args.stage_b_lock)
    results_root = _resolve(args.results_dir)
    _assert_isolated_output(results_root, stage_b_root)
    lock_manifest = _load_locked_stage_b(stage_b_root, stage_b_lock)

    import torch
    from gift_controller import GIFTController

    usable_cuda = torch.cuda.is_available() and torch.cuda.device_count() > 0
    if args.device == "cuda" and not usable_cuda:
        raise RuntimeError("CUDA was requested but is not available")
    if args.device == "auto" and not usable_cuda:
        os.environ["GIFT_TORCH_DEVICE"] = "cpu"
    device_label = "cuda" if args.device != "cpu" and usable_cuda else "cpu"

    cells = [
        (dataset, window, seed)
        for dataset in args.datasets
        for window in args.windows
        for seed in args.seeds
    ]
    job_key = hashlib.sha256("|".join(
        f"{dataset}/{window}/{seed}" for dataset, window, seed in cells
    ).encode("utf-8")).hexdigest()[:12]
    registry_path = results_root / "registries" / f"stage_e_{job_key}.csv"
    registry: list[dict[str, Any]] = []

    print("=" * 78)
    print("STAGE-E FULL-WINDOW BASELINE COMPLETION")
    print(f"Cells: {len(cells)}; device={device_label}; physical_gpu={args.gpu}")
    print("Windows: W2/W4/W6 only; locked Stage-B GIFT/PBIR are read-only")
    print("Methods: PPO-TestOnly, PPO-Train+FineTune, Equal Weight")
    print("LLM calls: 0")
    print(f"Stage-B lock: {lock_manifest['aggregate_sha256']}")
    print(f"Results: {results_root}")
    print("=" * 78)

    for index, (dataset, window, seed) in enumerate(cells, start=1):
        print(f"\n[{index}/{len(cells)}] {dataset}/{window}/seed_{seed}",
              flush=True)
        pair, source_config, pair_path, config_path = _source_config(
            stage_b_root, dataset, window, seed)
        config = json.loads(json.dumps(source_config))
        experiment = config.setdefault("experiment", {})
        experiment["no_llm"] = True
        experiment["run_ppo_baselines"] = False
        experiment.pop("fixed_artifact_path", None)
        experiment.pop("fixed_artifact_sha256", None)
        experiment.pop("protocol_role", None)
        config.setdefault("pbir", {})["enabled"] = False
        baseline_config_sha = _canonical_config_sha(config)
        retrain_period, eval_period = _split_period(config)

        cell_root = results_root / dataset / window / f"seed_{seed}"
        cell_root.mkdir(parents=True, exist_ok=True)
        resolved_path = cell_root / "resolved_config.yaml"
        resolved_path.write_text(
            yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        manifest = {
            "protocol": "stage-e-even-window-baseline-completion",
            "dataset": dataset,
            "window": window,
            "seed": seed,
            "source_stage_b_pair": str(pair_path),
            "source_stage_b_config": str(config_path),
            "source_stage_b_lock_sha256": lock_manifest["aggregate_sha256"],
            "source_pure_gift_config_sha256": pair["pure_gift_config_sha256"],
            "baseline_config_canonical_sha256": baseline_config_sha,
            "retrain_period": list(retrain_period),
            "eval_period": list(eval_period),
            "llm_calls": 0,
            "dry_run": args.dry_run,
        }
        _write_json(cell_root / "manifest.json", manifest)
        if args.dry_run:
            registry.append({**manifest, "status": "validated"})
            _write_registry(registry, registry_path)
            continue

        controller = GIFTController(config, str(cell_root / "controller"), seed=seed)
        method_paths = {
            "ppo_test_only": cell_root / "methods" / "ppo_test_only.json",
            "ppo_train_plus_test": (
                cell_root / "methods" / "ppo_train_plus_test.json"),
        }
        if not (args.resume and _complete(
                method_paths["ppo_test_only"], "ppo_test_only")):
            started = time.perf_counter()
            raw = controller._run_baseline_comparison_v1(
                retrain_period, eval_period)
            result = _normalize_result(raw, "baseline")
            _write_json(method_paths["ppo_test_only"], _method_record(
                "ppo_test_only", result, dataset, window, seed,
                baseline_config_sha, retrain_period, eval_period,
                time.perf_counter() - started))
        else:
            print("PPO-TestOnly: resume -> existing complete result")

        if not (args.resume and _complete(
                method_paths["ppo_train_plus_test"],
                "ppo_train_plus_test")):
            started = time.perf_counter()
            raw = controller._run_baseline_train_then_test(
                retrain_period, eval_period)
            result = _normalize_result(raw, "base2")
            _write_json(method_paths["ppo_train_plus_test"], _method_record(
                "ppo_train_plus_test", result, dataset, window, seed,
                baseline_config_sha, retrain_period, eval_period,
                time.perf_counter() - started))
        else:
            print("PPO-Train+FineTune: resume -> existing complete result")

        equal_path = cell_root / "methods" / "equal_weight.json"
        if not (args.resume and _complete(equal_path, "equal_weight")):
            started = time.perf_counter()
            result = _equal_weight_result(config, eval_period)
            _write_json(equal_path, _method_record(
                "equal_weight", result, dataset, window, seed,
                baseline_config_sha, retrain_period, eval_period,
                time.perf_counter() - started))
        else:
            print("Equal Weight: resume -> existing complete result")

        methods = {name: _read_json(path) for name, path in {
            **method_paths, "equal_weight": equal_path}.items()}
        _write_json(cell_root / "baseline_summary.json", {
            "status": "completed",
            **manifest,
            "methods": methods,
        })
        row: dict[str, Any] = {
            "dataset": dataset,
            "window": window,
            "seed": seed,
            "status": "completed",
        }
        for method, record in methods.items():
            for metric, value in record["test_result"].items():
                if metric != "test_avg_weights":
                    row[f"{method}_{metric}"] = value
        registry.append(row)
        _write_registry(registry, registry_path)
        if device_label == "cuda":
            torch.cuda.empty_cache()

    print("\n" + "=" * 78)
    print(f"Stage-E processed {len(cells)} cells")
    print(f"Registry: {registry_path}")
    print(f"Results: {results_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
