"""Run only the missing PPO baselines for the locked Stage-B/Stage-C study.

This economical Stage-D launcher never regenerates or retrains GIFT/PBIR.  It
uses the frozen PPO configuration and the exact adaptation/evaluation split
from each completed Stage-C cell, then saves two pure-PPO protocols plus an
equal-weight reference under a separate results root.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import pickle
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml


PROJECT_DIR = Path(__file__).resolve().parent
DATASETS = ("portfolio_5stocks", "portfolio_5stocks2")
WINDOWS = ("W1", "W3", "W5")
DEFAULT_SEEDS = (42, 123, 456)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS,
                        default=list(DATASETS))
    parser.add_argument("--windows", nargs="+", choices=WINDOWS,
                        default=list(WINDOWS))
    parser.add_argument("--seeds", nargs="+", type=int,
                        default=list(DEFAULT_SEEDS))
    parser.add_argument(
        "--stage-c-dir", default="results/stage_c_cached_replication")
    parser.add_argument(
        "--stage-c-lock",
        default="results/stage_c_lock/stage_c_lock_manifest.json")
    parser.add_argument(
        "--results-dir", default="results/stage_d_ppo_baselines")
    parser.add_argument("--device", choices=("cuda", "auto", "cpu"),
                        default="cuda")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument(
        "--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _resolve(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_DIR / path).resolve()


def _canonical_config_sha(config: dict) -> str:
    return hashlib.sha256(
        yaml.safe_dump(config, sort_keys=True).encode("utf-8")).hexdigest()


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")


def _complete(path: Path, expected_method: str) -> bool:
    if not path.is_file():
        return False
    try:
        value = _read_json(path)
    except (json.JSONDecodeError, OSError):
        return False
    result = value.get("test_result", {})
    returns = value.get("daily_returns", [])
    return bool(
        value.get("status") == "completed"
        and value.get("method") == expected_method
        and isinstance(result.get("test_sharpe"), (int, float))
        and isinstance(returns, list) and returns)


def _normalize_result(raw: dict, prefix: str) -> dict[str, Any]:
    return {
        "test_sharpe": float(raw[f"{prefix}_sharpe"]),
        "test_sortino": float(raw[f"{prefix}_sortino"]),
        "test_max_drawdown": float(raw[f"{prefix}_max_drawdown"]),
        "test_calmar": float(raw[f"{prefix}_calmar"]),
        "test_total_return": float(raw[f"{prefix}_total_return"]),
        "test_avg_weights": raw[f"{prefix}_avg_weights"],
        "daily_returns": [float(value) for value in raw[f"{prefix}_returns"]],
    }


def _equal_weight_result(config: dict, evaluation_period: tuple[str, str]) -> dict:
    from metrics import calmar_ratio, max_drawdown, sharpe_ratio, sortino_ratio
    from portfolio_env import PortfolioEnv

    data_path = Path(config["data"]["pickle_file"])
    if not data_path.is_absolute():
        data_path = PROJECT_DIR / data_path
    env = PortfolioEnv(
        str(data_path), config, train_period=evaluation_period,
        transaction_cost=float(config.get("portfolio", {}).get(
            "transaction_cost", 0.001)))
    env.reset()
    risky_count = len(config["data"]["tickers"])
    weights = np.concatenate([
        np.full(risky_count, 1.0 / risky_count), np.array([0.0])])
    returns: list[float] = []
    done = False
    while not done:
        _, _, done, info = env.step(weights)
        if info:
            returns.append(float(info.get("portfolio_return", 0.0)))
    return {
        "test_sharpe": sharpe_ratio(returns),
        "test_sortino": sortino_ratio(returns),
        "test_max_drawdown": max_drawdown(returns),
        "test_calmar": calmar_ratio(returns),
        "test_total_return": (env.portfolio_value - 1.0) * 100,
        "test_avg_weights": {
            **{ticker: 1.0 / risky_count for ticker in config["data"]["tickers"]},
            "CASH": 0.0,
        },
        "daily_returns": returns,
    }


def _split_period(config: dict) -> tuple[tuple[str, str], tuple[str, str]]:
    data_path = Path(config["data"]["pickle_file"])
    if not data_path.is_absolute():
        data_path = PROJECT_DIR / data_path
    with data_path.open("rb") as handle:
        raw = pickle.load(handle)
    start, end = config["experiment"]["test_period"]
    dates = sorted(day for day in raw if start <= day <= end)
    if len(dates) < 2:
        raise ValueError(f"Insufficient test dates in {data_path}")
    split = len(dates) // 2
    return (dates[0], dates[split - 1]), (dates[split], dates[-1])


def _method_record(
        method: str, result: dict, dataset: str, window: str, seed: int,
        config_sha: str, retrain_period: tuple[str, str],
        eval_period: tuple[str, str], wall_time: float,
) -> dict:
    returns = result.pop("daily_returns")
    return {
        "status": "completed",
        "method": method,
        "dataset": dataset,
        "window": window,
        "seed": seed,
        "config_sha256": config_sha,
        "retrain_period": list(retrain_period),
        "eval_period": list(eval_period),
        "wall_time_seconds": wall_time,
        "llm_call_attempts": 0,
        "test_result": result,
        "daily_returns": returns,
    }


def main() -> int:
    args = _parse_args()
    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        os.environ["GIFT_TORCH_DEVICE"] = "cpu"
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        os.environ["GIFT_TORCH_DEVICE"] = "cuda"

    from lock_stage_c_results import verify as verify_stage_c_lock

    stage_c_root = _resolve(args.stage_c_dir)
    stage_c_lock = _resolve(args.stage_c_lock)
    if not args.dry_run:
        locked = verify_stage_c_lock(stage_c_lock)
        if Path(locked["source_root"]).resolve() != stage_c_root:
            raise ValueError("Stage-C lock points to a different source root")

    import torch
    from gift_controller import GIFTController

    usable_cuda = torch.cuda.is_available() and torch.cuda.device_count() > 0
    if args.device == "cuda" and not usable_cuda:
        raise RuntimeError("CUDA was requested but is not available")
    if args.device == "auto" and not usable_cuda:
        os.environ["GIFT_TORCH_DEVICE"] = "cpu"
    device_label = "cuda" if args.device != "cpu" and usable_cuda else "cpu"
    print(f"Stage-D PPO baselines | device={device_label} | physical_gpu={args.gpu}")
    print("LLM calls: 0; locked GIFT/PBIR results will not be modified")

    results_root = _resolve(args.results_dir)
    registry: list[dict[str, Any]] = []
    cells = [
        (dataset, window, seed)
        for dataset in args.datasets for window in args.windows
        for seed in args.seeds
    ]
    job_key = hashlib.sha256("|".join(
        f"{dataset}/{window}/{seed}" for dataset, window, seed in cells
    ).encode("utf-8")).hexdigest()[:12]
    registry_path = results_root / "registries" / f"stage_d_{job_key}.csv"
    for index, (dataset, window, seed) in enumerate(cells, start=1):
        print("=" * 78)
        print(f"CELL {index}/{len(cells)}: {dataset}/{window}/seed_{seed}")
        stage_c_cell = stage_c_root / dataset / window / f"seed_{seed}"
        pair_path = stage_c_cell / "final_pair_summary.json"
        config_path = (
            stage_c_cell / "methods" / "pure_gift" / "resolved_config.yaml")
        if not pair_path.is_file() or not config_path.is_file():
            raise FileNotFoundError(f"Incomplete locked Stage-C source: {stage_c_cell}")
        pair = _read_json(pair_path)
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        config_sha = _canonical_config_sha(config)
        if config_sha != pair["pure_gift_config_sha256"]:
            raise ValueError(f"Config hash mismatch: {config_path}")
        experiment = config.setdefault("experiment", {})
        experiment["no_llm"] = True
        experiment["run_ppo_baselines"] = False
        experiment.pop("fixed_artifact_path", None)
        experiment.pop("fixed_artifact_sha256", None)
        config.setdefault("pbir", {})["enabled"] = False
        baseline_config_sha = _canonical_config_sha(config)
        retrain_period, eval_period = _split_period(config)

        cell_root = results_root / dataset / window / f"seed_{seed}"
        cell_root.mkdir(parents=True, exist_ok=True)
        resolved = cell_root / "resolved_config.yaml"
        resolved.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        manifest = {
            "protocol": "stage-d-missing-ppo-baselines-only",
            "dataset": dataset,
            "window": window,
            "seed": seed,
            "source_stage_c_pair": str(pair_path),
            "source_stage_c_artifact_sha256": pair["shared_artifact_sha256"],
            "source_config_canonical_sha256": config_sha,
            "baseline_config_canonical_sha256": baseline_config_sha,
            "retrain_period": list(retrain_period),
            "eval_period": list(eval_period),
            "llm_calls": 0,
            "dry_run": args.dry_run,
        }
        _write_json(cell_root / "manifest.json", manifest)
        if args.dry_run:
            registry.append({**manifest, "status": "validated"})
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
                method_paths["ppo_train_plus_test"], "ppo_train_plus_test")):
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

        methods = {name: _read_json(path) for name, path in {
            **method_paths, "equal_weight": equal_path}.items()}
        summary = {
            "status": "completed",
            **manifest,
            "methods": methods,
        }
        _write_json(cell_root / "baseline_summary.json", summary)
        row: dict[str, Any] = {
            "dataset": dataset, "window": window, "seed": seed,
            "status": "completed",
        }
        for method, record in methods.items():
            for metric, value in record["test_result"].items():
                if metric != "test_avg_weights":
                    row[f"{method}_{metric}"] = value
        registry.append(row)
        if device_label == "cuda":
            torch.cuda.empty_cache()

        registry_path.parent.mkdir(parents=True, exist_ok=True)
        columns = sorted({key for item in registry for key in item})
        with registry_path.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns,
                                    extrasaction="ignore")
            writer.writeheader()
            writer.writerows(registry)

    print("=" * 78)
    print(f"Stage-D processed {len(cells)} cells")
    print(f"Results: {results_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
