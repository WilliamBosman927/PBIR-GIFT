"""Sequential launcher for the 18-cell fixed-code Stage-C replication."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
DATASETS = ("portfolio_5stocks", "portfolio_5stocks2")
PRIMARY_WINDOWS = ("W1", "W3", "W5")
CONFIGS = {
    (dataset, window): (
        f"config{'2' if dataset == 'portfolio_5stocks2' else ''}_{window}.yaml")
    for dataset in DATASETS for window in PRIMARY_WINDOWS
}


def main() -> int:
    parser = argparse.ArgumentParser(
        description=("Run fixed-code PBIR-GIFT/Pure-GIFT replication cells "
                     "sequentially on one GPU."))
    parser.add_argument("--datasets", nargs="+", choices=DATASETS,
                        default=list(DATASETS))
    parser.add_argument("--windows", nargs="+", choices=PRIMARY_WINDOWS,
                        default=list(PRIMARY_WINDOWS))
    parser.add_argument("--seeds", nargs="+", type=int,
                        default=[42, 123, 456])
    parser.add_argument("--device", choices=["cuda", "auto", "cpu"],
                        default="cuda")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--results-dir",
                        default="results/stage_c_cached_replication")
    parser.add_argument("--source-results-dir", default="results/stage_b_final")
    parser.add_argument("--generated-config-dir",
                        default="configs/_stage_c_cached")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction,
                        default=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cells = [
        (dataset, window, seed)
        for dataset in args.datasets
        for window in args.windows
        for seed in args.seeds
    ]
    print("=" * 78)
    print("STAGE-C FIXED-CODE CONTROLLED REPLICATION")
    print(f"Cells: {len(cells)}; device={args.device}; gpu={args.gpu}")
    print("Each cell runs PBIR-GIFT and Pure GIFT sequentially.")
    print("LLM calls during replay: 0")
    print("=" * 78)

    for index, (dataset, window, seed) in enumerate(cells, start=1):
        print(f"\n[{index}/{len(cells)}] {dataset}/{window}/seed_{seed}",
              flush=True)
        command = [
            args.python_executable,
            str(PROJECT_DIR / "final_paired_runner.py"),
            "--base-config", CONFIGS[(dataset, window)],
            "--dataset-name", dataset,
            "--window-name", window,
            "--seed", str(seed),
            "--results-dir", args.results_dir,
            "--shared-code-source-root", args.source_results_dir,
            "--generated-config-dir", args.generated_config_dir,
            "--evaluation-role", "controlled_replication",
            "--device", args.device,
            "--gpu", str(args.gpu),
            "--python-executable", args.python_executable,
            "--resume" if args.resume else "--no-resume",
        ]
        if args.dry_run:
            command.append("--dry-run")
        completed = subprocess.run(command, cwd=PROJECT_DIR)
        if completed.returncode != 0:
            print(
                f"FAILED: {dataset}/{window}/seed_{seed} returned "
                f"{completed.returncode}; stopping so --resume can continue.",
                file=sys.stderr,
            )
            return completed.returncode
    print("\nStage-C requested cells completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
