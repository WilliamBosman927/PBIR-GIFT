"""Shared, collision-safe hyperparameter sweep runner for PBIR-GIFT.

The formal Stage-A entry point binds this implementation to the dedicated
Dataset-1, Window-1 validation configuration. Keeping the implementation here
guarantees consistent validation, PBIR semantics, naming, and result schema.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import pickle
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from hparam_visualization import generate_sweep_figures


PROJECT_DIR = Path(__file__).resolve().parent

PARAMETER_GROUPS: dict[str, list[Any]] = {
    "epochs_per_update": [1, 3, 5, 8, 10],
    "max_episodes": [20, 35, 50, 75, 100],
    "actor_lr": [5e-5, 1e-4, 3e-4, 5e-4, 1e-3],
    "clip_epsilon": [0.05, 0.10, 0.15, 0.20, 0.30],
    "entropy_coef": [0.0, 0.001, 0.005, 0.01, 0.02],
}


def _value_tag(value: Any) -> str:
    text = f"{value:.8g}" if isinstance(value, float) else str(value)
    return text.replace("-", "m").replace(".", "p").replace("+", "")


def build_trials(base_config: dict[str, Any]) -> list[dict[str, Any]]:
    """Build 25 one-factor-at-a-time trial configs."""
    trials: list[dict[str, Any]] = []
    for parameter, values in PARAMETER_GROUPS.items():
        for level, value in enumerate(values, start=1):
            config = copy.deepcopy(base_config)
            config.setdefault("ppo", {})[parameter] = value
            trials.append({
                "trial_name": f"{parameter}_g{level}_{_value_tag(value)}",
                "parameter": parameter,
                "level": level,
                "value": value,
                "config": config,
            })
    return trials


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _first_number(mapping: dict[str, Any], keys: list[str]) -> Any:
    for key in keys:
        value = mapping.get(key)
        if isinstance(value, (int, float)) and np.isfinite(value):
            return value
    return None


def collect_metrics(experiment_dir: Path) -> dict[str, Any]:
    summary = _load_json(experiment_dir / "summary.json")
    comparison = _load_json(experiment_dir / "final_comparison.json")
    test_result = comparison.get("test_result", {})
    if not isinstance(test_result, dict):
        test_result = {}
    return {
        "best_train_sharpe": _first_number(
            summary, ["best_train_sharpe", "best_sharpe"]),
        "best_iteration": summary.get("best_iteration"),
        "test_sharpe": _first_number(
            test_result, ["test_sharpe", "val_sharpe", "sharpe"]),
        "test_sortino": _first_number(
            test_result, ["test_sortino", "val_sortino", "sortino"]),
        "test_max_drawdown": _first_number(
            test_result,
            ["test_max_drawdown", "test_mdd", "val_max_drawdown",
             "max_drawdown"],
        ),
        "test_total_return": _first_number(
            test_result, ["test_total_return", "test_return", "total_return"]),
        "wall_time_seconds": _first_number(summary, ["wall_time_seconds"]),
        "llm_call_attempts": _first_number(summary, ["llm_call_attempts"]),
        "llm_call_successes": _first_number(summary, ["llm_call_successes"]),
        "valid_code_rate": _first_number(summary, ["valid_code_rate"]),
        "failed_iterations": _first_number(summary, ["failed_iterations"]),
    }


def _experiment_complete(experiment_dir: Path) -> bool:
    metrics = collect_metrics(experiment_dir)
    return (
        isinstance(metrics.get("best_iteration"), int)
        and metrics.get("test_sharpe") is not None
        and metrics.get("test_total_return") is not None
    )


def write_summary(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "dataset", "window", "method", "data_file", "trial_name", "parameter",
        "level", "value", "seed", "status", "return_code",
        "best_train_sharpe", "best_iteration", "test_sharpe",
        "test_sortino", "test_max_drawdown", "test_total_return",
        "config_path", "experiment_dir",
    ]
    with output_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _update_figures(
        summary_path: Path,
        results_root: Path,
        base_config: dict[str, Any],
        dataset: str,
        window: str,
        method: str,
        seed: int,
) -> None:
    """Regenerate available figures without risking the training run."""
    try:
        paths = generate_sweep_figures(
            summary_path,
            results_root / "figures",
            base_ppo=base_config.get("ppo", {}),
            dataset=dataset,
            window=window,
            method=method,
            seed=seed,
        )
        if paths:
            print(f"Updated {len(paths) // 2} hyperparameter figure(s) in "
                  f"{results_root / 'figures'}")
    except Exception as exc:
        print(f"WARNING: figure generation skipped: {exc}", file=sys.stderr)


def _resolve_project_path(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_DIR / path


def _normalize_pbir(config: dict[str, Any], mode: str) -> tuple[dict[str, Any], str]:
    """Normalize legacy/current PBIR keys and enforce PPO gamma as truth."""
    config = copy.deepcopy(config)
    legacy = config.pop("potential_based_intrinsic_reward", {}) or {}
    pbir = dict(legacy)
    pbir.update(config.get("pbir", {}) or {})
    if mode != "config":
        pbir["enabled"] = mode == "enabled"
    pbir.setdefault("enabled", True)
    pbir.setdefault("scale", 1.0)
    pbir.setdefault("potential_clip", 1.0)
    pbir.setdefault("shaping_clip", 0.0)
    pbir.setdefault("terminal_zero", True)

    ppo_gamma = float(config.setdefault("ppo", {}).get("gamma", 0.99))
    configured_gamma = pbir.pop("gamma", None)
    if configured_gamma is not None and not np.isclose(
            float(configured_gamma), ppo_gamma):
        raise ValueError(
            f"PBIR gamma ({configured_gamma}) must equal ppo.gamma ({ppo_gamma})")
    config["pbir"] = pbir
    return config, "pbir" if pbir["enabled"] else "pure_gift"


def _validate_data(config: dict[str, Any]) -> dict[str, Any]:
    data_cfg = config.get("data", {})
    data_path = _resolve_project_path(str(data_cfg.get("pickle_file", "")))
    tickers = list(data_cfg.get("tickers", []))
    if not data_path.is_file():
        raise FileNotFoundError(f"Pickle data not found: {data_path}")
    if not tickers:
        raise ValueError("config.data.tickers must not be empty")

    with data_path.open("rb") as handle:
        raw_data = pickle.load(handle)
    if not isinstance(raw_data, dict) or not raw_data:
        raise ValueError(f"Pickle must contain a non-empty date mapping: {data_path}")
    dates = sorted(raw_data)
    expected = set(tickers)
    for day in dates:
        prices = raw_data[day].get("price", {})
        missing = expected.difference(prices)
        if missing:
            raise ValueError(f"{day} is missing tickers: {sorted(missing)}")
        for ticker in tickers:
            point = prices[ticker]
            price = point.get("adjusted_close", point.get("close", 0.0))
            if not isinstance(price, (int, float)) or not np.isfinite(price) or price <= 0:
                raise ValueError(f"Invalid price for {ticker} on {day}: {price}")

    experiment = config.get("experiment", {})
    period_counts: dict[str, int] = {}
    for name in ("train_period", "val_period", "test_period"):
        period = experiment.get(name)
        if not isinstance(period, (list, tuple)) or len(period) != 2:
            raise ValueError(f"experiment.{name} must be [start, end]")
        selected = [day for day in dates if period[0] <= day <= period[1]]
        if len(selected) <= 22:
            raise ValueError(
                f"{name} has only {len(selected)} days in {data_path.name}")
        period_counts[name] = len(selected)

    return {
        "path": str(data_path),
        "days": len(dates),
        "first_date": dates[0],
        "last_date": dates[-1],
        "tickers": tickers,
        "period_counts": period_counts,
    }


def _check_child_python(
        python_executable: str,
        child_env: dict[str, str],
        require_cuda: bool,
) -> dict[str, Any]:
    result_prefix = "__GIFT_PREFLIGHT_JSON__="
    command = [
        python_executable,
        "-c",
        ("import json, torch, numpy, yaml, openai, scipy; "
         "ok=torch.cuda.is_available(); "
         "info={'torch':torch.__version__,'cuda_runtime':torch.version.cuda,"
         "'cuda_available':ok,'device_count':torch.cuda.device_count(),"
         "'device_name':torch.cuda.get_device_name(0) if ok else None}; "
         f"print({result_prefix!r} + json.dumps(info), flush=True); "
         f"raise SystemExit(3 if {require_cuda!r} and not ok else 0)"),
    ]
    completed = subprocess.run(
        command, cwd=PROJECT_DIR, env=child_env,
        capture_output=True, text=True)
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()
        if completed.returncode == 3:
            raise RuntimeError(
                "CUDA was requested but is unavailable in the experiment "
                f"Python environment: {python_executable}\n{detail}")
        raise RuntimeError(
            f"Experiment Python is missing dependencies: {python_executable}\n{detail}")

    result_line = next(
        (line for line in reversed(completed.stdout.splitlines())
         if line.startswith(result_prefix)),
        None,
    )
    if result_line is None:
        stdout = completed.stdout.strip() or "<empty>"
        stderr = completed.stderr.strip() or "<empty>"
        raise RuntimeError(
            "Experiment Python preflight finished without a readable result.\n"
            f"stdout:\n{stdout}\n"
            f"stderr:\n{stderr}"
        )
    try:
        device_info = json.loads(result_line[len(result_prefix):])
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            "Experiment Python returned malformed preflight JSON: "
            f"{result_line[len(result_prefix):]!r}"
        ) from exc
    print(f"Python preflight: {device_info}")
    return device_info


def _parse_args(default_base_config: str, dataset_name: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(f"Run PBIR-GIFT OFAT sweep for {dataset_name}: "
                     "5 PPO parameters x 5 values."))
    parser.add_argument("--base-config", default=default_base_config)
    parser.add_argument("--dataset-name", default=dataset_name)
    parser.add_argument(
        "--window-name",
        help="Rolling-window label (W1-W6); used for isolated configs/results.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--results-dir", default=f"results/hparam_sweep/{dataset_name}",
        help="Result base; method and seed subdirectories are added automatically.")
    parser.add_argument(
        "--python-executable", default=sys.executable,
        help="Python used to invoke main.py; it must provide PyTorch.")
    parser.add_argument(
        "--device", choices=["cuda", "auto", "cpu"], default="cuda",
        help="Execution device. Default cuda fails fast instead of silently using CPU.")
    parser.add_argument(
        "--gpu", type=int, default=0,
        help="Physical CUDA device id exposed to the child process.")
    parser.add_argument("--only-param", choices=list(PARAMETER_GROUPS))
    parser.add_argument(
        "--pbir-mode", choices=["config", "enabled", "disabled"],
        default="enabled")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--max-trials", type=int,
        help="Run/generate only the first N selected trials (useful for smoke tests).")
    parser.add_argument(
        "--smoke", action="store_true",
        help="Use 1 GIFT iteration, 2 PPO episodes, and at most 1 trial.")
    parser.add_argument(
        "--no-llm", action="store_true",
        help="Use deterministic fallback factors; intended only for offline checks.")
    return parser.parse_args()


def run_sweep(default_base_config: str, dataset_name: str) -> int:
    args = _parse_args(default_base_config, dataset_name)
    base_config_path = _resolve_project_path(args.base_config)
    if not base_config_path.is_file():
        raise FileNotFoundError(f"Base config not found: {base_config_path}")
    with base_config_path.open("r", encoding="utf-8") as handle:
        base_config = yaml.safe_load(handle) or {}
    if not isinstance(base_config, dict):
        raise TypeError("The base YAML config must contain a mapping at the root")

    base_config, method = _normalize_pbir(base_config, args.pbir_mode)
    inferred_window = re.search(r"(?:^|_)W([1-6])$", base_config_path.stem,
                                flags=re.IGNORECASE)
    window_name = args.window_name or (
        f"W{inferred_window.group(1)}" if inferred_window else "single")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", window_name):
        raise ValueError("--window-name may contain only letters, digits, _ and -")
    experiment = base_config.setdefault("experiment", {})
    if args.no_llm:
        experiment["no_llm"] = True
    if args.smoke:
        experiment["max_iterations"] = 1
        base_config.setdefault("ppo", {})["max_episodes"] = 2

    data_info = _validate_data(base_config)
    child_env = os.environ.copy()
    if args.device in {"cuda", "auto"}:
        child_env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    elif args.device == "cpu":
        child_env["CUDA_VISIBLE_DEVICES"] = ""
    device_info: dict[str, Any] = {
        "requested": args.device,
        "physical_gpu": args.gpu if args.device in {"cuda", "auto"} else None,
    }
    if not args.dry_run:
        device_info.update(_check_child_python(
            args.python_executable, child_env, require_cuda=args.device == "cuda"))
        provider = str(base_config.get("llm", {}).get("provider", "openai")).lower()
        no_llm = bool(experiment.get("no_llm", False))
        if provider == "bailian" and not no_llm and not os.environ.get(
                "DASHSCOPE_API_KEY"):
            raise RuntimeError(
                "DASHSCOPE_API_KEY is required for the live LLM sweep. "
                "Set it in the current shell or use --no-llm for a smoke test.")

    results_base = _resolve_project_path(args.results_dir)
    results_root = results_base / method / f"seed_{args.seed}"
    try:
        results_root.relative_to(PROJECT_DIR / "results")
    except ValueError as exc:
        raise ValueError(
            "--results-dir must be inside the project's results directory "
            "because main.py writes beneath that fixed root") from exc
    generated_root = (
        PROJECT_DIR / "configs" / "_hparam_sweep" / args.dataset_name
        / window_name / method / f"seed_{args.seed}"
    )
    results_root.mkdir(parents=True, exist_ok=True)
    generated_root.mkdir(parents=True, exist_ok=True)

    trials = build_trials(base_config)
    if args.only_param:
        trials = [t for t in trials if t["parameter"] == args.only_param]
    limit = 1 if args.smoke and args.max_trials is None else args.max_trials
    if limit is not None:
        if limit < 1:
            raise ValueError("--max-trials must be positive")
        trials = trials[:limit]

    print(f"Dataset: {args.dataset_name}")
    print(f"Data: {data_info['path']}")
    print(f"Dates: {data_info['first_date']} ~ {data_info['last_date']} "
          f"({data_info['days']} days)")
    print(f"Tickers: {data_info['tickers']}")
    print(f"Periods: {data_info['period_counts']}")
    print(f"Method: {method}; seed={args.seed}")
    print(f"Rolling window: {window_name}")
    print(f"Device: {args.device}; physical GPU="
          f"{args.gpu if args.device in {'cuda', 'auto'} else 'N/A'}")
    print(f"Base config: {base_config_path}")
    print(f"Trials: {len(trials)}")
    print(f"Results: {results_root}")

    manifest = {
        "dataset": args.dataset_name,
        "window": window_name,
        "method": method,
        "reward_semantics": (
            "LLM state potential with gamma*Phi(s_next)-Phi(s_current)"
            if method == "pbir" else
            "LLM direct intrinsic reward (original Pure GIFT control)"
        ),
        "seed": args.seed,
        "base_config": str(base_config_path),
        "data": data_info,
        "parameter_groups": PARAMETER_GROUPS,
        "base_ppo": base_config.get("ppo", {}),
        "trial_count": len(trials),
        "python_executable": args.python_executable,
        "device": device_info,
        "dry_run": args.dry_run,
        "smoke": args.smoke,
        "no_llm": args.no_llm,
    }
    with (results_root / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    rows: list[dict[str, Any]] = []
    failures = 0
    for index, trial in enumerate(trials, start=1):
        trial_name = trial["trial_name"]
        config_path = generated_root / f"{trial_name}.yaml"
        experiment_dir = results_root / trial_name
        with config_path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(trial["config"], handle, sort_keys=False)

        row = {
            "dataset": args.dataset_name,
            "window": window_name,
            "method": method,
            "data_file": data_info["path"],
            "trial_name": trial_name,
            "parameter": trial["parameter"],
            "level": trial["level"],
            "value": trial["value"],
            "seed": args.seed,
            "status": "generated" if args.dry_run else "pending",
            "return_code": None,
            "config_path": str(config_path),
            "experiment_dir": str(experiment_dir),
        }
        print(f"[{index:02d}/{len(trials):02d}] {trial['parameter']}="
              f"{trial['value']} -> {trial_name}")

        if args.dry_run:
            rows.append(row)
            continue
        if args.resume and _experiment_complete(experiment_dir):
            row["status"] = "skipped_existing"
            row["return_code"] = 0
            row.update(collect_metrics(experiment_dir))
            rows.append(row)
            write_summary(rows, results_root / "sweep_summary.csv")
            continue

        experiment_name = str(experiment_dir.relative_to(PROJECT_DIR / "results"))
        command = [
            args.python_executable, str(PROJECT_DIR / "main.py"),
            "--config", str(config_path),
            "--experiment_name", experiment_name,
            "--seed", str(args.seed),
            "--pbir-mode", "enabled" if method == "pbir" else "disabled",
        ]
        completed = subprocess.run(
            command, cwd=PROJECT_DIR, env=child_env)
        row["return_code"] = completed.returncode
        row.update(collect_metrics(experiment_dir))
        if completed.returncode == 0 and _experiment_complete(experiment_dir):
            row["status"] = "completed"
        else:
            row["status"] = "failed" if completed.returncode else "incomplete"
            failures += 1
        rows.append(row)
        summary_path = results_root / "sweep_summary.csv"
        write_summary(rows, summary_path)
        _update_figures(
            summary_path, results_root, base_config,
            args.dataset_name, window_name, method, args.seed)

    summary_path = results_root / "sweep_summary.csv"
    write_summary(rows, summary_path)
    if not args.dry_run:
        _update_figures(
            summary_path, results_root, base_config,
            args.dataset_name, window_name, method, args.seed)
    if args.dry_run:
        print("Dry run complete: configs and manifest generated.")
    else:
        print(f"Sweep complete: {len(rows) - failures} succeeded, {failures} failed.")
    print(f"Summary: {results_root / 'sweep_summary.csv'}")
    return 1 if failures else 0
