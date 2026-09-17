"""Paired 25-Trial runner: PBIR-GIFT and Pure GIFT share every Trial."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from hparam_sweep_runner import (
    PARAMETER_GROUPS,
    PROJECT_DIR,
    _check_child_python,
    _experiment_complete,
    _normalize_pbir,
    _resolve_project_path,
    _validate_data,
    build_trials,
    collect_metrics,
)
from paired_trial_visualization import generate_paired_overview


METHODS = (
    ("pbir", "enabled", "PBIR-GIFT"),
    ("pure_gift", "disabled", "Pure GIFT"),
)
METRIC_NAMES = (
    "best_train_sharpe", "best_iteration", "test_sharpe", "test_sortino",
    "test_max_drawdown", "test_total_return", "wall_time_seconds",
    "llm_call_attempts", "llm_call_successes", "valid_code_rate",
    "failed_iterations",
)


def _parse_args(default_base_config: str, dataset: str, window: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            f"Run one paired 25-Trial PBIR-GIFT/Pure GIFT sweep for "
            f"{dataset} {window}."))
    parser.add_argument("--base-config", default=default_base_config)
    parser.add_argument("--dataset-name", default=dataset)
    parser.add_argument("--window-name", default=window)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--results-dir", default="results/paired_hparam_sweep",
        help="Root for paired dataset/window results.")
    parser.add_argument(
        "--python-executable", default=sys.executable,
        help="Python used to invoke main.py.")
    parser.add_argument(
        "--device", choices=["cuda", "auto", "cpu"], default="cuda")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--only-param", choices=list(PARAMETER_GROUPS))
    parser.add_argument(
        "--sequential-ofat", action="store_true",
        help=("Select each five-level group on held-out performance and carry "
              "its winner into every subsequent group. The last winner is "
              "therefore a directly evaluated joint configuration."))
    parser.add_argument(
        "--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-trials", type=int)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--include-ppo-baselines", action="store_true",
        help=("Also train the two Pure-PPO baselines inside every method run. "
              "Disabled by default because a paired PBIR/Pure-GIFT sweep "
              "does not need 100 duplicated baseline trainings per cell."))
    parser.add_argument(
        "--no-llm", action="store_true",
        help="Deterministic offline fallback; do not use for formal results.")
    return parser.parse_args()


def _write_summary(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "dataset", "window", "trial_name", "parameter", "level", "value",
        "seed", "status", "data_file", "frozen_prior_ppo", "method_order",
    ]
    for method, _, _ in METHODS:
        columns.extend([
            f"{method}_status", f"{method}_return_code",
            *(f"{method}_{metric}" for metric in METRIC_NAMES),
            f"{method}_config_path", f"{method}_config_sha256",
            f"{method}_experiment_dir",
        ])
    columns.extend([
        "test_sharpe_delta_pbir_minus_gift",
        "test_sortino_delta_pbir_minus_gift",
        "test_total_return_delta_pbir_minus_gift",
        "test_max_drawdown_delta_pbir_minus_gift",
        "test_max_drawdown_improvement_pbir_minus_gift",
    ])
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _add_deltas(row: dict[str, Any]) -> None:
    for metric in ("test_sharpe", "test_sortino", "test_total_return",
                   "test_max_drawdown"):
        pbir = row.get(f"pbir_{metric}")
        gift = row.get(f"pure_gift_{metric}")
        if all(isinstance(value, (int, float)) and np.isfinite(value)
               for value in (pbir, gift)):
            row[f"{metric}_delta_pbir_minus_gift"] = float(pbir - gift)
    pbir_mdd = row.get("pbir_test_max_drawdown")
    gift_mdd = row.get("pure_gift_test_max_drawdown")
    if all(isinstance(value, (int, float)) and np.isfinite(value)
           for value in (pbir_mdd, gift_mdd)):
        row["test_max_drawdown_improvement_pbir_minus_gift"] = float(
            gift_mdd - pbir_mdd)


def _select_sequential_winner(
        rows: list[dict[str, Any]], parameter: str) -> dict[str, Any]:
    candidates = [
        row for row in rows
        if row.get("parameter") == parameter
        and row.get("status") in {"completed", "skipped_existing"}
    ]
    expected = len(PARAMETER_GROUPS[parameter])
    if len(candidates) != expected:
        raise RuntimeError(
            f"Sequential OFAT group {parameter} needs {expected} complete "
            f"paired Trials; found {len(candidates)}")
    for row in candidates:
        for key in ("pbir_test_sharpe", "pure_gift_test_sharpe"):
            value = row.get(key)
            if not isinstance(value, (int, float)) or not np.isfinite(value):
                raise RuntimeError(
                    f"Sequential OFAT cannot select {parameter}: {key} is "
                    f"not finite in {row.get('trial_name')}")
    return max(
        candidates,
        key=lambda row: (
            (row["pbir_test_sharpe"] + row["pure_gift_test_sharpe"]) / 2.0,
            -int(row["level"]),
        ),
    )


def _config_sha256(config: dict[str, Any]) -> str:
    payload = yaml.safe_dump(config, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _update_overview(summary_path: Path, results_root: Path) -> None:
    try:
        paths = generate_paired_overview(summary_path, results_root / "figures")
        if paths:
            print(f"Paired overview updated: {paths[0]}")
    except Exception as exc:
        print(f"WARNING: paired overview skipped: {exc}", file=sys.stderr)


def run_paired_sweep(
        default_base_config: str,
        default_dataset: str,
        default_window: str,
) -> int:
    args = _parse_args(default_base_config, default_dataset, default_window)
    if not re.fullmatch(r"W[1-6]", args.window_name, flags=re.IGNORECASE):
        raise ValueError("--window-name must be W1 through W6")
    window = args.window_name.upper()
    base_config_path = _resolve_project_path(args.base_config)
    if not base_config_path.is_file():
        raise FileNotFoundError(f"Base config not found: {base_config_path}")
    with base_config_path.open("r", encoding="utf-8") as handle:
        base_config = yaml.safe_load(handle) or {}
    if not isinstance(base_config, dict):
        raise TypeError("The base YAML config must contain a mapping")

    # Validate both method variants before any API call.
    pbir_config, _ = _normalize_pbir(base_config, "enabled")
    gift_config, _ = _normalize_pbir(base_config, "disabled")
    for config in (pbir_config, gift_config):
        config.setdefault("experiment", {})["run_ppo_baselines"] = bool(
            args.include_ppo_baselines)
    experiment = base_config.setdefault("experiment", {})
    if args.no_llm:
        experiment["no_llm"] = True
        pbir_config.setdefault("experiment", {})["no_llm"] = True
        gift_config.setdefault("experiment", {})["no_llm"] = True
    if args.smoke:
        for config in (base_config, pbir_config, gift_config):
            config.setdefault("experiment", {})["max_iterations"] = 1
            config.setdefault("ppo", {})["max_episodes"] = 2

    data_info = _validate_data(base_config)
    child_env = os.environ.copy()
    if args.device in {"cuda", "auto"}:
        child_env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    else:
        child_env["CUDA_VISIBLE_DEVICES"] = ""
    device_info: dict[str, Any] = {
        "requested": args.device,
        "physical_gpu": args.gpu if args.device != "cpu" else None,
    }
    if not args.dry_run:
        device_info.update(_check_child_python(
            args.python_executable, child_env,
            require_cuda=args.device == "cuda"))
        provider = str(base_config.get("llm", {}).get("provider", "openai")).lower()
        if (provider == "bailian" and not args.no_llm
                and not os.environ.get("DASHSCOPE_API_KEY")):
            raise RuntimeError(
                "DASHSCOPE_API_KEY is not visible to this launcher process")

    result_base = _resolve_project_path(args.results_dir)
    results_root = result_base / args.dataset_name / window / f"seed_{args.seed}"
    configs_root = (
        PROJECT_DIR / "configs" / "_paired_hparam_sweep"
        / args.dataset_name / window / f"seed_{args.seed}")
    results_root.mkdir(parents=True, exist_ok=True)
    configs_root.mkdir(parents=True, exist_ok=True)

    trials = build_trials(base_config)
    if args.only_param:
        trials = [trial for trial in trials
                  if trial["parameter"] == args.only_param]
    limit = 1 if args.smoke and args.max_trials is None else args.max_trials
    if limit is not None:
        if limit < 1:
            raise ValueError("--max-trials must be positive")
        trials = trials[:limit]

    manifest = {
        "protocol": "paired-within-trial",
        "dataset": args.dataset_name,
        "window": window,
        "methods": [method for method, _, _ in METHODS],
        "method_order_within_trial": "alternating_by_trial_index",
        "seed": args.seed,
        "base_config": str(base_config_path),
        "data": data_info,
        "base_ppo": base_config.get("ppo", {}),
        "parameter_groups": PARAMETER_GROUPS,
        "trial_count": len(trials),
        "device": device_info,
        "resume": args.resume,
        "dry_run": args.dry_run,
        "no_llm": args.no_llm,
        "include_ppo_baselines": args.include_ppo_baselines,
        "search_strategy": (
            "sequential_ofat" if args.sequential_ofat else "independent_ofat"),
        "parameter_order": list(PARAMETER_GROUPS),
    }
    (results_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")

    print("=" * 78)
    print("PAIRED PBIR-GIFT vs PURE GIFT SWEEP")
    print(f"Dataset/window: {args.dataset_name} / {window}")
    print(f"Trials: {len(trials)} (two matched method runs inside each Trial)")
    print(f"Device: {args.device}; physical GPU="
          f"{args.gpu if args.device != 'cpu' else 'N/A'}")
    print(f"Results: {results_root}")
    print("=" * 78)

    rows: list[dict[str, Any]] = []
    failed_pairs = 0
    summary_path = results_root / "paired_sweep_summary.csv"
    method_templates = {"pbir": pbir_config, "pure_gift": gift_config}
    selected_so_far: dict[str, Any] = {}

    for index, trial in enumerate(trials, start=1):
        trial_name = trial["trial_name"]
        print(f"\n[{index:02d}/{len(trials):02d}] PAIRED TRIAL "
              f"{trial['parameter']}={trial['value']} -> {trial_name}")
        row: dict[str, Any] = {
            "dataset": args.dataset_name,
            "window": window,
            "trial_name": trial_name,
            "parameter": trial["parameter"],
            "level": trial["level"],
            "value": trial["value"],
            "seed": args.seed,
            "status": "generated" if args.dry_run else "pending",
            "data_file": data_info["path"],
            "frozen_prior_ppo": json.dumps(
                selected_so_far, sort_keys=True, separators=(",", ":")),
        }
        methods_for_trial = METHODS if index % 2 else tuple(reversed(METHODS))
        row["method_order"] = ">".join(item[0] for item in methods_for_trial)

        method_complete: list[bool] = []
        method_resumed: list[bool] = []
        for method, mode, label in methods_for_trial:
            method_config = copy.deepcopy(method_templates[method])
            if args.sequential_ofat:
                method_config.setdefault("ppo", {}).update(selected_so_far)
            method_config.setdefault("ppo", {})[trial["parameter"]] = trial["value"]
            config_path = configs_root / f"{trial_name}_{method}.yaml"
            experiment_dir = results_root / "trials" / trial_name / method
            with config_path.open("w", encoding="utf-8") as handle:
                yaml.safe_dump(method_config, handle, sort_keys=False)
            config_hash = _config_sha256(method_config)
            row[f"{method}_config_path"] = str(config_path)
            row[f"{method}_config_sha256"] = config_hash
            row[f"{method}_experiment_dir"] = str(experiment_dir)

            protocol_path = experiment_dir / "run_protocol.json"
            existing_protocol: dict[str, Any] = {}
            if protocol_path.is_file():
                try:
                    existing_protocol = json.loads(
                        protocol_path.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    existing_protocol = {}
            resume_compatible = (
                existing_protocol.get("config_sha256") == config_hash
                and existing_protocol.get("method") == method
                and existing_protocol.get("seed") == args.seed
                and existing_protocol.get("search_strategy") == (
                    "sequential_ofat" if args.sequential_ofat
                    else "independent_ofat"))
            experiment_dir.mkdir(parents=True, exist_ok=True)
            protocol_path.write_text(json.dumps({
                "config_sha256": config_hash,
                "method": method,
                "seed": args.seed,
                "search_strategy": (
                    "sequential_ofat" if args.sequential_ofat
                    else "independent_ofat"),
                "parameter": trial["parameter"],
                "value": trial["value"],
                "frozen_prior_ppo": selected_so_far,
            }, indent=2), encoding="utf-8")

            if args.dry_run:
                row[f"{method}_status"] = "generated"
                row[f"{method}_return_code"] = None
                method_complete.append(False)
                method_resumed.append(False)
                continue

            if (args.resume and resume_compatible
                    and _experiment_complete(experiment_dir)):
                print(f"  {label}: resume -> existing result")
                row[f"{method}_status"] = "skipped_existing"
                row[f"{method}_return_code"] = 0
                metrics = collect_metrics(experiment_dir)
                row.update({f"{method}_{key}": value
                            for key, value in metrics.items()})
                method_complete.append(True)
                method_resumed.append(True)
                continue
            if args.resume and _experiment_complete(experiment_dir):
                print(
                    f"  {label}: existing result has a different protocol/"
                    "config hash; it will not be reused")

            print(f"  {label}: running")
            command = [
                args.python_executable, str(PROJECT_DIR / "main.py"),
                "--config", str(config_path),
                "--output-dir", str(experiment_dir),
                "--seed", str(args.seed),
                "--pbir-mode", mode,
            ]
            completed = subprocess.run(command, cwd=PROJECT_DIR, env=child_env)
            row[f"{method}_return_code"] = completed.returncode
            metrics = collect_metrics(experiment_dir)
            row.update({f"{method}_{key}": value
                        for key, value in metrics.items()})
            complete = completed.returncode == 0 and _experiment_complete(
                experiment_dir)
            row[f"{method}_status"] = (
                "completed" if complete else
                "failed" if completed.returncode else "incomplete")
            method_complete.append(complete)
            method_resumed.append(False)

        if args.dry_run:
            row["status"] = "generated"
        elif all(method_complete):
            row["status"] = (
                "skipped_existing" if all(method_resumed) else "completed")
            _add_deltas(row)
        else:
            row["status"] = "failed"
            failed_pairs += 1
        rows.append(row)
        _write_summary(rows, summary_path)

        if args.sequential_ofat:
            next_parameter = (
                trials[index]["parameter"] if index < len(trials) else None)
            if next_parameter != trial["parameter"]:
                if args.dry_run:
                    print(
                        "Sequential selection deferred: dry-run has no "
                        "validation metrics.")
                    if next_parameter is not None:
                        print("Dry-run stops at the first group boundary.")
                        break
                else:
                    try:
                        winner = _select_sequential_winner(
                            rows, trial["parameter"])
                    except RuntimeError as exc:
                        print(f"ERROR: {exc}", file=sys.stderr)
                        failed_pairs += 1
                        break
                    selected_so_far[trial["parameter"]] = winner["value"]
                    mean_sharpe = (
                        winner["pbir_test_sharpe"]
                        + winner["pure_gift_test_sharpe"]) / 2.0
                    print(
                        f"  Sequential winner: {trial['parameter']}="
                        f"{winner['value']} (shared mean validation Sharpe="
                        f"{mean_sharpe:.4f})")

    _write_summary(rows, summary_path)
    if not args.dry_run:
        _update_overview(summary_path, results_root)
    print(f"\nPaired summary: {summary_path}")
    if args.dry_run:
        print("Dry run complete; no LLM calls or training were made.")
    else:
        print(f"Paired sweep complete: {len(rows) - failed_pairs} complete, "
              f"{failed_pairs} failed/incomplete.")
    return 1 if failed_pairs else 0
