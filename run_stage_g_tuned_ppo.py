"""Tune a strong Pure-PPO baseline, freeze it, and evaluate all 12 cells.

Stage-G is a prospective robustness study added after Stage-F.  It never calls
an LLM and never writes to Stage-A--F.  Hyperparameters are selected only on
the dedicated pre-W1 validation period in ``config_tuning_D1_W1.yaml``.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from hparam_sweep_runner import PARAMETER_GROUPS
from run_stage_d_ppo_baselines import _normalize_result, _split_period
from supplemental_experiment_utils import (
    DATASETS,
    METRICS,
    PROJECT_DIR,
    SEEDS,
    WINDOWS,
    canonical_yaml_sha,
    completed_result,
    ensure_isolated_output,
    load_base_config,
    read_json,
    resolve_path,
    write_csv,
    write_json,
)


METHOD = "ppo_tuned"
DEFAULT_ROOT = "results/stage_g_tuned_ppo"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase", choices=("tune", "select", "evaluate", "all"), default="all")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--datasets", nargs="+", choices=DATASETS,
                        default=list(DATASETS))
    parser.add_argument("--windows", nargs="+", choices=WINDOWS,
                        default=list(WINDOWS))
    parser.add_argument("--config", default="config_tuning_D1_W1.yaml")
    parser.add_argument("--results-dir", default=DEFAULT_ROOT)
    parser.add_argument("--device", choices=("cuda", "auto", "cpu"),
                        default="cuda")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument(
        "--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _device(args: argparse.Namespace) -> str:
    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        os.environ["GIFT_TORCH_DEVICE"] = "cpu"
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        os.environ["GIFT_TORCH_DEVICE"] = "cuda"
    import torch
    usable = torch.cuda.is_available() and torch.cuda.device_count() > 0
    if args.device == "cuda" and not usable:
        raise RuntimeError("CUDA was requested but is not available")
    if args.device == "auto" and not usable:
        os.environ["GIFT_TORCH_DEVICE"] = "cpu"
    return "cuda" if args.device != "cpu" and usable else "cpu"


def _validation_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    config = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(config, dict):
        raise TypeError(f"Config must be a mapping: {path}")
    experiment = config.get("experiment", {})
    if experiment.get("protocol_role") != "hyperparameter_validation":
        raise ValueError("Stage-G requires the dedicated Stage-A validation config")
    if experiment.get("test_period") != ["2020-01-01", "2020-06-30"]:
        raise ValueError("Stage-G validation period must remain 2020-H1")
    config.setdefault("experiment", {})["no_llm"] = True
    config["experiment"]["run_ppo_baselines"] = False
    config.setdefault("pbir", {})["enabled"] = False
    return config


def _result_record(
    result: dict[str, Any], *, seed: int, config_sha: str,
    retrain_period: tuple[str, str], eval_period: tuple[str, str],
    wall_time: float, role: str, extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    values = dict(result)
    returns = values.pop("daily_returns")
    record: dict[str, Any] = {
        "status": "completed",
        "method": METHOD,
        "evaluation_role": role,
        "seed": seed,
        "config_sha256": config_sha,
        "retrain_period": list(retrain_period),
        "eval_period": list(eval_period),
        "wall_time_seconds": wall_time,
        "llm_call_attempts": 0,
        "test_result": values,
        "daily_returns": returns,
    }
    if extra:
        record.update(extra)
    return record


def _run_one(
    config: dict[str, Any], output_dir: Path, seed: int, *, role: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    from gift_controller import GIFTController

    retrain_period, eval_period = _split_period(config)
    started = time.perf_counter()
    controller = GIFTController(config, str(output_dir / "controller"), seed=seed)
    raw = controller._run_baseline_comparison_v1(retrain_period, eval_period)
    result = _normalize_result(raw, "baseline")
    return _result_record(
        result,
        seed=seed,
        config_sha=canonical_yaml_sha(config),
        retrain_period=retrain_period,
        eval_period=eval_period,
        wall_time=time.perf_counter() - started,
        role=role,
        extra=extra,
    )


def _score_candidate(records: list[dict[str, Any]]) -> tuple[float, ...]:
    def mean(metric: str) -> float:
        return float(np.mean([
            float(record["test_result"][metric]) for record in records]))
    return (
        mean("test_sharpe"),
        mean("test_sortino"),
        mean("test_total_return"),
        -abs(mean("test_max_drawdown")),
    )


def _candidate_dir(root: Path, parameter: str, level: int, seed: int) -> Path:
    return root / "tuning" / parameter / f"level_{level}" / f"seed_{seed}"


def _select_from_results(
    root: Path, seeds: list[int], source_config: Path,
) -> Path:
    selected: dict[str, Any] = {}
    selection_rows: list[dict[str, Any]] = []
    selected_rows: dict[str, dict[str, Any]] = {}
    for parameter, values in PARAMETER_GROUPS.items():
        expected_prior = dict(selected)
        candidates: list[tuple[tuple[float, ...], dict[str, Any]]] = []
        for level, value in enumerate(values, start=1):
            records: list[dict[str, Any]] = []
            for seed in seeds:
                path = _candidate_dir(root, parameter, level, seed) / "result.json"
                record = read_json(path)
                if record.get("method") != METHOD or record.get("status") != "completed":
                    raise ValueError(f"Incomplete Stage-G tuning result: {path}")
                if record.get("frozen_prior_ppo") != expected_prior:
                    raise ValueError(f"Sequential Stage-G prior mismatch: {path}")
                if record.get("parameter") != parameter or record.get("level") != level:
                    raise ValueError(f"Stage-G candidate identity mismatch: {path}")
                records.append(record)
            score = _score_candidate(records)
            row = {
                "parameter": parameter,
                "level": level,
                "value": value,
                "selection_seeds": ",".join(str(seed) for seed in seeds),
                "frozen_prior_ppo": json.dumps(expected_prior, sort_keys=True),
                "mean_validation_sharpe": score[0],
                "mean_validation_sortino": score[1],
                "mean_validation_total_return": score[2],
                "mean_validation_max_drawdown": -score[3],
            }
            selection_rows.append(row)
            candidates.append((score + (-level,), row))
        winner = max(candidates, key=lambda item: item[0])[1]
        selected[parameter] = winner["value"]
        selected_rows[parameter] = winner

    selection_dir = root / "selection"
    write_csv(selection_dir / "selection_scores.csv", selection_rows)
    artifact = {
        "protocol": {
            "name": "stage-g-independently-tuned-ppo",
            "version": 1,
            "stage": "G_selected",
            "status": "prospective_supplement_after_stage_f",
            "method": "PPO-TestOnly",
            "selection_dataset": "portfolio_5stocks",
            "selection_window": "D1-W1-internal-validation",
            "train_period": ["2019-01-01", "2019-12-31"],
            "validation_period": ["2020-01-01", "2020-06-30"],
            "selection_seeds": seeds,
            "selection_rule": (
                "Fixed-order sequential OFAT; maximize seed-averaged held-out "
                "validation Sharpe of Pure PPO, with Sortino, total return, "
                "absolute max drawdown, and lower level as deterministic ties."
            ),
            "parameter_order": list(PARAMETER_GROUPS),
            "source_config": str(source_config),
            "source_results_root": str(root / "tuning"),
            "selected_candidates": selected_rows,
            "created_utc": datetime.now(timezone.utc).isoformat(),
        },
        "ppo": selected,
    }
    path = selection_dir / "selected_ppo_tuned.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(artifact, sort_keys=False), encoding="utf-8")
    write_json(selection_dir / "selection_manifest.json", artifact)
    return path


def tune(
    args: argparse.Namespace, root: Path, base: dict[str, Any], config_path: Path,
) -> Path | None:
    if args.dry_run:
        plans = []
        for parameter, values in PARAMETER_GROUPS.items():
            for level, value in enumerate(values, start=1):
                for seed in args.seeds:
                    plans.append({
                        "parameter": parameter, "level": level,
                        "value": value, "seed": seed,
                    })
        write_json(root / "dry_run_tuning_plan.json", {
            "status": "validated", "llm_calls": 0,
            "candidate_seed_runs": len(plans), "runs": plans,
        })
        print(f"Stage-G tuning dry-run validated {len(plans)} PPO runs")
        return None

    selected: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    for parameter, values in PARAMETER_GROUPS.items():
        print("=" * 78)
        print(f"STAGE-G TUNING GROUP: {parameter}")
        prior = dict(selected)
        for level, value in enumerate(values, start=1):
            for seed in args.seeds:
                run_dir = _candidate_dir(root, parameter, level, seed)
                config = copy.deepcopy(base)
                config.setdefault("ppo", {}).update(prior)
                config["ppo"][parameter] = value
                config_sha = canonical_yaml_sha(config)
                resolved = run_dir / "resolved_config.yaml"
                result_path = run_dir / "result.json"
                run_dir.mkdir(parents=True, exist_ok=True)
                resolved.write_text(
                    yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
                if args.resume and completed_result(
                        result_path, METHOD, config_sha):
                    print(f"  {parameter} level={level} seed={seed}: resume")
                    record = read_json(result_path)
                else:
                    print(f"  {parameter} level={level} seed={seed}: running")
                    record = _run_one(
                        config, run_dir, seed,
                        role="ppo_hyperparameter_validation",
                        extra={
                            "parameter": parameter,
                            "level": level,
                            "value": value,
                            "frozen_prior_ppo": prior,
                        },
                    )
                    write_json(result_path, record)
                rows.append({
                    "parameter": parameter, "level": level, "value": value,
                    "seed": seed, "status": record["status"],
                    "frozen_prior_ppo": json.dumps(prior, sort_keys=True),
                    **{metric: record["test_result"][metric] for metric in METRICS},
                    "wall_time_seconds": record["wall_time_seconds"],
                    "config_sha256": record["config_sha256"],
                    "result_path": str(result_path),
                })
                write_csv(root / "tuning" / "tuning_registry.csv", rows)
        # Select this group before constructing the next group.
        group_records: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for level, _ in enumerate(values, start=1):
            for seed in args.seeds:
                group_records[level].append(read_json(
                    _candidate_dir(root, parameter, level, seed) / "result.json"))
        winner_level = max(
            group_records,
            key=lambda level: _score_candidate(group_records[level]) + (-level,),
        )
        selected[parameter] = values[winner_level - 1]
        print(f"  selected {parameter}={selected[parameter]}")
    selection = _select_from_results(root, args.seeds, config_path)
    print(f"Stage-G tuning complete; frozen selection: {selection}")
    return selection


def select(args: argparse.Namespace, root: Path, config_path: Path) -> Path:
    path = _select_from_results(root, args.seeds, config_path)
    print(f"Stage-G selection rebuilt and verified: {path}")
    return path


def _load_selection(root: Path) -> tuple[dict[str, Any], Path]:
    path = root / "selection" / "selected_ppo_tuned.yaml"
    if not path.is_file():
        raise FileNotFoundError(
            f"Stage-G selection is missing; run --phase tune or select first: {path}")
    artifact = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if artifact.get("protocol", {}).get("stage") != "G_selected":
        raise ValueError(f"Invalid Stage-G selection artifact: {path}")
    if set(artifact.get("ppo", {})) != set(PARAMETER_GROUPS):
        raise ValueError(f"Stage-G selection is missing PPO parameters: {path}")
    return artifact, path


def evaluate(args: argparse.Namespace, root: Path) -> None:
    artifact, selection_path = _load_selection(root)
    selected = artifact["ppo"]
    selection_sha = canonical_yaml_sha(artifact)
    cells = [
        (dataset, window, seed)
        for dataset in args.datasets for window in args.windows for seed in args.seeds
    ]
    rows: list[dict[str, Any]] = []
    for index, (dataset, window, seed) in enumerate(cells, start=1):
        config, source_path = load_base_config(dataset, window)
        config = copy.deepcopy(config)
        config.setdefault("ppo", {}).update(selected)
        experiment = config.setdefault("experiment", {})
        experiment["no_llm"] = True
        experiment["run_ppo_baselines"] = False
        experiment["protocol_role"] = "stage_g_tuned_ppo_final"
        experiment.pop("fixed_artifact_path", None)
        experiment.pop("fixed_artifact_sha256", None)
        config.setdefault("pbir", {})["enabled"] = False
        config_sha = canonical_yaml_sha(config)
        cell = root / "evaluation" / dataset / window / f"seed_{seed}"
        result_path = cell / "methods" / "ppo_tuned.json"
        resolved = cell / "resolved_config.yaml"
        cell.mkdir(parents=True, exist_ok=True)
        resolved.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        retrain_period, eval_period = _split_period(config)
        manifest = {
            "protocol": "stage-g-tuned-ppo-final-evaluation",
            "dataset": dataset,
            "window": window,
            "seed": seed,
            "source_config": str(source_path),
            "source_config_sha256": canonical_yaml_sha(
                yaml.safe_load(source_path.read_text(encoding="utf-8")) or {}),
            "selection_path": str(selection_path),
            "selection_sha256": selection_sha,
            "config_sha256": config_sha,
            "retrain_period": list(retrain_period),
            "eval_period": list(eval_period),
            "llm_calls": 0,
            "dry_run": args.dry_run,
        }
        write_json(cell / "manifest.json", manifest)
        if args.dry_run:
            record: dict[str, Any] | None = None
            status = "validated"
            print(f"[{index}/{len(cells)}] {dataset}/{window}/seed_{seed}: validated")
        elif args.resume and completed_result(result_path, METHOD, config_sha):
            record = read_json(result_path)
            status = "skipped_existing"
            print(f"[{index}/{len(cells)}] {dataset}/{window}/seed_{seed}: resume")
        else:
            print(f"[{index}/{len(cells)}] {dataset}/{window}/seed_{seed}: running")
            record = _run_one(
                config, cell, seed, role="tuned_ppo_final_test",
                extra={
                    "dataset": dataset,
                    "window": window,
                    "selection_sha256": selection_sha,
                },
            )
            write_json(result_path, record)
            status = "completed"
        row = {
            "dataset": dataset, "window": window, "seed": seed,
            "status": status, "config_sha256": config_sha,
            "result_path": str(result_path),
        }
        if record is not None:
            row.update({metric: record["test_result"][metric] for metric in METRICS})
            row["wall_time_seconds"] = record["wall_time_seconds"]
        rows.append(row)
        write_csv(root / "evaluation" / "evaluation_registry.csv", rows)
    print(f"Stage-G evaluation processed {len(cells)} cells; LLM calls: 0")


def main() -> int:
    args = _parse_args()
    args.seeds = list(dict.fromkeys(args.seeds))
    if not args.seeds:
        raise ValueError("At least one seed is required")
    formal_root = resolve_path(args.results_dir)
    root = (
        resolve_path("results/_dryrun/stage_g_tuned_ppo")
        if args.dry_run and args.results_dir == DEFAULT_ROOT
        else formal_root
    )
    ensure_isolated_output(root)
    config_path = resolve_path(args.config)
    base = _validation_config(config_path)
    print("=" * 78)
    print("STAGE-G INDEPENDENT PURE-PPO TUNING")
    print(f"phase={args.phase}; seeds={args.seeds}; LLM calls=0")
    print(f"results={root}")
    print("=" * 78)

    if args.phase in {"tune", "all"}:
        if not args.dry_run:
            print(f"device={_device(args)}; physical_gpu={args.gpu}")
        tune(args, root, base, config_path)
    if args.phase == "select":
        select(args, root, config_path)
    if args.phase in {"evaluate", "all"}:
        if args.dry_run and args.phase == "all":
            print("Dry-run all stops after validating the tuning matrix; "
                  "run a real tune before evaluation dry-run.")
        else:
            if args.phase == "evaluate" and not args.dry_run:
                print(f"device={_device(args)}; physical_gpu={args.gpu}")
            evaluate(args, root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
