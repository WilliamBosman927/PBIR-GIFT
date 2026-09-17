"""One-click Stage-G -> Stage-H -> Stage-I launcher with safe resume behavior."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
LOCKS = {
    "f": PROJECT_DIR / "results" / "stage_f_lock" / "stage_f_lock_manifest.json",
    "g": PROJECT_DIR / "results" / "stage_g_lock" / "stage_g_lock_manifest.json",
    "h": PROJECT_DIR / "results" / "stage_h_lock" / "stage_h_lock_manifest.json",
    "i": PROJECT_DIR / "results" / "stage_i_lock" / "stage_i_lock_manifest.json",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-stage", choices=("g", "h", "i"), default="g")
    parser.add_argument("--selection-seeds", nargs="+", type=int,
                        default=[42, 123, 456])
    parser.add_argument("--evaluation-seeds", nargs="+", type=int,
                        default=[42, 123, 456])
    parser.add_argument("--device", choices=("cuda", "auto", "cpu"),
                        default="cuda")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument(
        "--resume", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def _run(arguments: list[str]) -> None:
    command = [sys.executable, *arguments]
    print("\n" + "=" * 78, flush=True)
    print("RUN:", " ".join(command), flush=True)
    print("=" * 78, flush=True)
    completed = subprocess.run(command, cwd=PROJECT_DIR)
    if completed.returncode:
        raise SystemExit(
            f"Command failed with return code {completed.returncode}: "
            + " ".join(command))


def _lock_is_valid(stage: str, script: str) -> bool:
    if not LOCKS[stage].is_file():
        return False
    completed = subprocess.run(
        [sys.executable, script, "--verify"], cwd=PROJECT_DIR,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if completed.returncode:
        raise RuntimeError(
            f"Stage-{stage.upper()} has a lock manifest but verification failed. "
            "Refusing to overwrite locked evidence.")
    return True


def _stage_g(args: argparse.Namespace) -> None:
    if _lock_is_valid("g", "lock_stage_g_results.py"):
        print("Stage-G already has a valid lock; skipping it.")
        return
    resume = "--resume" if args.resume else "--no-resume"
    _run([
        "run_stage_g_tuned_ppo.py", "--phase", "tune", "--seeds",
        *map(str, args.selection_seeds), "--device", args.device,
        "--gpu", str(args.gpu), resume,
    ])
    _run([
        "run_stage_g_tuned_ppo.py", "--phase", "evaluate", "--seeds",
        *map(str, args.evaluation_seeds), "--device", args.device,
        "--gpu", str(args.gpu), resume,
    ])
    _run(["analyze_stage_g_tuned_ppo.py", "--resamples", "10000",
          "--block-length", "20"])
    _run(["lock_stage_g_results.py"])
    _run(["lock_stage_g_results.py", "--verify"])


def _stage_h(args: argparse.Namespace) -> None:
    if _lock_is_valid("h", "lock_stage_h_results.py"):
        print("Stage-H already has a valid lock; skipping it.")
        return
    resume = "--resume" if args.resume else "--no-resume"
    _run([
        "run_stage_h_traditional_baselines.py",
        "--methods", "sma", "wma", "atr", "bollinger", "turn_of_month", "xgboost",
        "--datasets", "portfolio_5stocks", "portfolio_5stocks2",
        "--windows", "W1", "W2", "W3", "W4", "W5", "W6",
        "--xgboost-seeds", "42", "123", "456",
        "--device", "cpu", resume,
    ])
    _run(["analyze_stage_h_traditional_baselines.py", "--resamples", "10000",
          "--block-length", "20"])
    _run(["lock_stage_h_results.py"])
    _run(["lock_stage_h_results.py", "--verify"])


def _stage_i() -> None:
    if _lock_is_valid("i", "lock_stage_i_results.py"):
        print("Stage-I already has a valid lock; skipping it.")
        return
    _run([
        "analyze_stage_i_final_comparison.py", "--resamples", "10000",
        "--block-length", "20", "--primary-metric", "sharpe",
        "--correction", "holm",
    ])
    _run(["lock_stage_i_results.py"])
    _run(["lock_stage_i_results.py", "--verify"])


def main() -> int:
    args = _parse_args()
    if not _lock_is_valid("f", "lock_stage_f_results.py"):
        raise RuntimeError(
            "Stage-F is not locked or its contents changed. Run "
            "lock_stage_f_results.py and --verify before supplemental experiments.")
    order = ("g", "h", "i")
    start = order.index(args.start_stage)
    for stage in order[start:]:
        if stage == "g":
            _stage_g(args)
        elif stage == "h":
            _stage_h(args)
        else:
            _stage_i()
    print("\nAll requested supplemental stages completed and locked.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
