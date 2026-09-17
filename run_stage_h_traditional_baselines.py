"""Run six low-cost traditional/ML baselines on all dataset-window cells."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import yaml

from run_stage_d_ppo_baselines import _split_period
from supplemental_experiment_utils import (
    DATASETS,
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
from traditional_baselines import (
    DETERMINISTIC_METHODS,
    METHODS,
    run_traditional_baseline,
)


DEFAULT_ROOT = "results/stage_h_traditional_baselines"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--methods", nargs="+", choices=METHODS,
                        default=list(METHODS))
    parser.add_argument("--datasets", nargs="+", choices=DATASETS,
                        default=list(DATASETS))
    parser.add_argument("--windows", nargs="+", choices=WINDOWS,
                        default=list(WINDOWS))
    parser.add_argument("--xgboost-seeds", nargs="+", type=int,
                        default=list(SEEDS))
    parser.add_argument("--strategy-config", default="config_traditional_baselines.yaml")
    parser.add_argument("--results-dir", default=DEFAULT_ROOT)
    parser.add_argument("--device", choices=("cpu",), default="cpu")
    parser.add_argument(
        "--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _load_strategy_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if value.get("protocol", {}).get("name") != (
            "stage-h-traditional-baseline-specification"):
        raise ValueError(f"Invalid Stage-H strategy config: {path}")
    configured = value.get("methods", {})
    if set(configured) != set(METHODS):
        raise ValueError("Stage-H strategy config must define all six methods")
    return value


def _result_path(cell: Path, method: str, seed: int | None) -> Path:
    if method == "xgboost":
        if seed is None:
            raise ValueError("XGBoost result requires a seed")
        return cell / "methods" / "xgboost" / f"seed_{seed}.json"
    return cell / "methods" / f"{method}.json"


def _record(
    *, method: str, dataset: str, window: str, seed: int | None,
    config_sha: str, adaptation_period: tuple[str, str],
    evaluation_period: tuple[str, str], wall_time: float,
    raw: dict[str, Any],
) -> dict[str, Any]:
    return {
        "status": "completed",
        "method": method,
        "dataset": dataset,
        "window": window,
        "seed": seed,
        "deterministic": method in DETERMINISTIC_METHODS,
        "config_sha256": config_sha,
        "adaptation_period": list(adaptation_period),
        "eval_period": list(evaluation_period),
        "wall_time_seconds": wall_time,
        "llm_call_attempts": 0,
        **raw,
    }


def _existing_cell_summary(
    cell: Path, dataset: str, window: str, xgb_seeds: list[int],
) -> dict[str, Any]:
    method_files: dict[str, list[str]] = {}
    complete = True
    for method in METHODS:
        seeds = xgb_seeds if method == "xgboost" else [None]
        files = []
        for seed in seeds:
            path = _result_path(cell, method, seed)
            if path.is_file():
                try:
                    record = read_json(path)
                    if record.get("status") == "completed":
                        files.append(str(path))
                    else:
                        complete = False
                except Exception:
                    complete = False
            else:
                complete = False
        method_files[method] = files
    return {
        "status": "completed" if complete else "partial",
        "protocol": "stage-h-six-traditional-baselines",
        "dataset": dataset,
        "window": window,
        "deterministic_methods": list(DETERMINISTIC_METHODS),
        "xgboost_seeds": xgb_seeds,
        "method_files": method_files,
        "llm_calls": 0,
    }


def main() -> int:
    args = _parse_args()
    args.methods = list(dict.fromkeys(args.methods))
    args.xgboost_seeds = list(dict.fromkeys(args.xgboost_seeds))
    if not args.xgboost_seeds:
        raise ValueError("At least one XGBoost seed is required")
    formal_root = resolve_path(args.results_dir)
    root = (
        resolve_path("results/_dryrun/stage_h_traditional_baselines")
        if args.dry_run and args.results_dir == DEFAULT_ROOT
        else formal_root
    )
    ensure_isolated_output(root)
    strategy_path = resolve_path(args.strategy_config)
    strategy_config = _load_strategy_config(strategy_path)
    strategy_sha = canonical_yaml_sha(strategy_config)
    root.mkdir(parents=True, exist_ok=True)
    frozen_strategy_path = root / "frozen_strategy_config.yaml"
    frozen_strategy_path.write_text(
        yaml.safe_dump(strategy_config, sort_keys=False), encoding="utf-8")
    print("=" * 78)
    print("STAGE-H TRADITIONAL BASELINES")
    print(f"methods={args.methods}; XGBoost seeds={args.xgboost_seeds}")
    print("device=cpu; LLM calls=0")
    print(f"results={root}")
    print("=" * 78)

    registry: list[dict[str, Any]] = []
    cells = [(dataset, window) for dataset in args.datasets for window in args.windows]
    for cell_index, (dataset, window) in enumerate(cells, start=1):
        base, source_path = load_base_config(dataset, window)
        adaptation_period, evaluation_period = _split_period(base)
        cell = root / dataset / window
        cell.mkdir(parents=True, exist_ok=True)
        resolved_market_config = cell / "resolved_market_config.yaml"
        resolved_market_config.write_text(
            yaml.safe_dump(base, sort_keys=False), encoding="utf-8")
        manifest = {
            "protocol": "stage-h-six-traditional-baselines",
            "dataset": dataset,
            "window": window,
            "source_config": str(source_path),
            "source_config_sha256": canonical_yaml_sha(base),
            "resolved_market_config": str(resolved_market_config),
            "strategy_config": str(frozen_strategy_path),
            "strategy_source_config": str(strategy_path),
            "strategy_config_sha256": strategy_sha,
            "adaptation_period": list(adaptation_period),
            "evaluation_period": list(evaluation_period),
            "requested_methods": args.methods,
            "xgboost_seeds": args.xgboost_seeds,
            "llm_calls": 0,
            "dry_run": args.dry_run,
        }
        write_json(cell / "manifest.json", manifest)
        for method in args.methods:
            method_parameters = strategy_config["methods"][method]
            seeds: list[int | None] = (
                list(args.xgboost_seeds) if method == "xgboost" else [None])
            for seed in seeds:
                protocol_config = {
                    "base_config": base,
                    "strategy_protocol": strategy_config.get("protocol"),
                    "portfolio_protocol": strategy_config.get("portfolio"),
                    "method": method,
                    "method_parameters": method_parameters,
                    "seed": seed,
                }
                config_sha = canonical_yaml_sha(protocol_config)
                result_path = _result_path(cell, method, seed)
                label = f"{dataset}/{window}/{method}"
                if seed is not None:
                    label += f"/seed_{seed}"
                if args.dry_run:
                    status = "validated"
                    record = None
                    print(f"[{cell_index}/{len(cells)}] {label}: validated")
                elif args.resume and completed_result(result_path, method, config_sha):
                    status = "skipped_existing"
                    record = read_json(result_path)
                    print(f"[{cell_index}/{len(cells)}] {label}: resume")
                else:
                    print(f"[{cell_index}/{len(cells)}] {label}: running")
                    started = time.perf_counter()
                    raw = run_traditional_baseline(
                        base, method, method_parameters, evaluation_period,
                        adaptation_period, seed)
                    record = _record(
                        method=method, dataset=dataset, window=window, seed=seed,
                        config_sha=config_sha,
                        adaptation_period=adaptation_period,
                        evaluation_period=evaluation_period,
                        wall_time=time.perf_counter() - started,
                        raw=raw,
                    )
                    write_json(result_path, record)
                    status = "completed"
                row = {
                    "dataset": dataset, "window": window, "method": method,
                    "seed": seed if seed is not None else "deterministic",
                    "status": status, "config_sha256": config_sha,
                    "result_path": str(result_path),
                }
                if record is not None:
                    for metric, value in record["test_result"].items():
                        if metric != "test_avg_weights":
                            row[metric] = value
                    row["wall_time_seconds"] = record["wall_time_seconds"]
                registry.append(row)
                write_csv(root / "stage_h_registry.csv", registry)
        if not args.dry_run:
            write_json(
                cell / "baseline_summary.json",
                _existing_cell_summary(cell, dataset, window, args.xgboost_seeds),
            )

    print(f"Stage-H processed {len(cells)} dataset-window cells")
    if args.dry_run:
        total = sum(
            len(args.xgboost_seeds) if method == "xgboost" else 1
            for method in args.methods) * len(cells)
        print(f"Dry-run validated {total} method runs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
