"""Run the one-off Stage-A sweep, freeze PPO values, and confirm the joint set."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import yaml

from hparam_sweep_runner import PROJECT_DIR
from stage_a_selection import select_stage_a_configuration


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the only 25-Trial tuning sweep in the protocol.")
    parser.add_argument("--seed", type=int, default=42)
    default_results = str(
        Path(os.environ.get("GIFT_RESULTS_ROOT", PROJECT_DIR / "results"))
        / "stage_a_tuning")
    parser.add_argument("--results-dir", default=default_results)
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument(
        "--device", choices=["cuda", "auto", "cpu"], default="cuda")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument(
        "--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-llm", action="store_true")
    parser.add_argument("--max-trials", type=int)
    parser.add_argument(
        "--only-param",
        choices=["epochs_per_update", "max_episodes", "actor_lr",
                 "clip_epsilon", "entropy_coef"])
    parser.add_argument(
        "--confirm", action=argparse.BooleanOptionalAction, default=False,
        help=("Run one extra paired validation cell using the combined five "
              "selected values. Optional because sequential OFAT already "
              "evaluates the joint setting in its final group."))
    return parser.parse_args()


def _forward_common(args: argparse.Namespace) -> list[str]:
    values = [
        "--seed", str(args.seed),
        "--python-executable", args.python_executable,
        "--device", args.device,
        "--gpu", str(args.gpu),
        "--resume" if args.resume else "--no-resume",
    ]
    if args.dry_run:
        values.append("--dry-run")
    if args.no_llm:
        values.append("--no-llm")
    return values


def main() -> int:
    args = _parse_args()
    tuning_config = PROJECT_DIR / "config_tuning_D1_W1.yaml"
    results_base = Path(args.results_dir)
    if not results_base.is_absolute():
        results_base = PROJECT_DIR / results_base
    tuning_root = (
        results_base / "portfolio_5stocks" / "W1" / f"seed_{args.seed}")

    sweep_command = [
        args.python_executable,
        str(PROJECT_DIR / "run_paired_portfolio_5stocks_W1.py"),
        "--base-config", str(tuning_config),
        "--results-dir", str(results_base),
        "--sequential-ofat",
        *_forward_common(args),
    ]
    if args.max_trials is not None:
        sweep_command.extend(["--max-trials", str(args.max_trials)])
    if args.only_param:
        sweep_command.extend(["--only-param", args.only_param])

    print("STAGE A/1: one paired 25-Trial validation sweep", flush=True)
    completed = subprocess.run(sweep_command, cwd=PROJECT_DIR)
    if completed.returncode:
        return completed.returncode

    # Partial and dry runs are useful operational checks, but are deliberately
    # prevented from creating a supposedly final frozen configuration.
    if args.dry_run or args.max_trials is not None or args.only_param:
        print("Partial/dry Stage-A run finished; selection was not produced.")
        return 0

    print("STAGE A/2: select one shared value from each OFAT group", flush=True)
    selection_path = select_stage_a_configuration(
        tuning_root / "paired_sweep_summary.csv",
        tuning_config,
        tuning_root / "selection",
    )
    print(f"Selected PPO artifact: {selection_path}")
    if not args.confirm:
        print(
            "The last sequential-OFAT winner is the directly evaluated joint "
            "configuration; no extra confirmation pair was requested.")
        return 0

    print("STAGE A/3: confirm the combined selected setting on validation", flush=True)
    confirmation_command = [
        args.python_executable,
        str(PROJECT_DIR / "final_paired_runner.py"),
        "--base-config", str(tuning_config),
        "--dataset-name", "portfolio_5stocks",
        "--window-name", "VALIDATION",
        "--selection", str(selection_path),
        "--results-dir", str(results_base.parent / "stage_a_confirmation"),
        "--evaluation-role", "validation_confirmation",
        *_forward_common(args),
    ]
    confirmation = subprocess.run(confirmation_command, cwd=PROJECT_DIR)
    if confirmation.returncode:
        return confirmation.returncode

    artifact = yaml.safe_load(selection_path.read_text(encoding="utf-8")) or {}
    artifact.setdefault("protocol", {})["requires_combined_confirmation"] = False
    artifact["protocol"]["combined_confirmation"] = {
        "status": "completed",
        "seed": args.seed,
        "result": str(
            results_base.parent / "stage_a_confirmation"
            / "portfolio_5stocks" / "VALIDATION" / f"seed_{args.seed}"
            / "final_pair_summary.json"),
    }
    selection_path.write_text(
        yaml.safe_dump(artifact, sort_keys=False), encoding="utf-8")
    print("Stage A complete: selected PPO values are frozen and confirmed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
