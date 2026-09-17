"""Run the 18 even-window fixed-code PBIR-GIFT/Pure-GIFT Stage-F cells.

Stage-F completes the controlled PBIR ablation started in Stage-C.  Each cell
materializes the PBIR winner selected in the locked Stage-B run exactly once,
then gives the same code and reward-rule payload to PBIR-GIFT and Pure GIFT.
The only permitted method-level intervention is ``pbir.enabled``; no LLM is
called during this replay.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from hparam_sweep_runner import PROJECT_DIR
from lock_stage_b_results import verify_lock


DATASETS = ("portfolio_5stocks", "portfolio_5stocks2")
EVEN_WINDOWS = ("W2", "W4", "W6")
DEFAULT_SEEDS = (42, 123, 456)
CONFIGS = {
    (dataset, window): (
        f"config{'2' if dataset == 'portfolio_5stocks2' else ''}_{window}.yaml")
    for dataset in DATASETS for window in EVEN_WINDOWS
}
PROTECTED_RESULT_ROOTS = (
    "results/stage_b_final",
    "results/stage_c_cached_replication",
    "results/stage_d_ppo_baselines",
    "results/stage_e_full_window_baselines",
)


def _resolve(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_DIR / path).resolve()


def _overlaps(left: Path, right: Path) -> bool:
    left = left.resolve()
    right = right.resolve()
    return left == right or left in right.parents or right in left.parents


def _verify_stage_b_source(source: Path, lock_path: Path) -> dict:
    if not lock_path.is_file():
        raise FileNotFoundError(
            f"Stage-B lock not found: {lock_path}. Refusing an unlocked replay.")
    verify_lock(lock_path)
    manifest = json.loads(lock_path.read_text(encoding="utf-8"))
    locked_root = Path(str(manifest["source_root"])).resolve()
    if locked_root != source.resolve():
        raise ValueError(
            "--source-results-dir does not match the Stage-B lock source: "
            f"{source} != {locked_root}")
    expected = {
        (dataset, window, seed)
        for dataset in DATASETS for window in EVEN_WINDOWS
        for seed in DEFAULT_SEEDS
    }
    observed = set()
    for dataset, window, seed in expected:
        pair_path = source / dataset / window / f"seed_{seed}" / "final_pair_summary.json"
        if not pair_path.is_file():
            raise FileNotFoundError(f"Missing Stage-B source cell: {pair_path}")
        pair = json.loads(pair_path.read_text(encoding="utf-8"))
        if pair.get("status") not in {"completed", "skipped_existing"}:
            raise ValueError(f"Incomplete Stage-B source cell: {pair_path}")
        observed.add((dataset, window, seed))
    if observed != expected:
        raise ValueError("Stage-B even-window source grid is incomplete")
    return manifest


def _guard_output_root(output: Path, source: Path) -> None:
    protected = [source, *(_resolve(value) for value in PROTECTED_RESULT_ROOTS)]
    for root in protected:
        if _overlaps(output, root):
            raise ValueError(
                "Stage-F output must be isolated from prior evidence roots: "
                f"{output} overlaps {root}")
    unexpected = list(output.glob("*/W[135]/seed_*/final_pair_summary.json"))
    if unexpected:
        raise ValueError(
            "Stage-F output contains odd-window cells; use a clean directory: "
            f"{unexpected[0]}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS,
                        default=list(DATASETS))
    parser.add_argument("--windows", nargs="+", choices=EVEN_WINDOWS,
                        default=list(EVEN_WINDOWS))
    parser.add_argument("--seeds", nargs="+", type=int, choices=DEFAULT_SEEDS,
                        default=list(DEFAULT_SEEDS))
    parser.add_argument("--device", choices=["cuda", "auto", "cpu"],
                        default="cuda")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument(
        "--selection",
        default=("results/stage_a_tuning/portfolio_5stocks/W1/seed_42/"
                 "selection/selected_ppo.yaml"),
        help="Frozen Stage-A PPO selection used by Stage-B and Stage-C.")
    parser.add_argument(
        "--results-dir", default="results/stage_f_full_window_cached_pbir")
    parser.add_argument("--source-results-dir", default="results/stage_b_final")
    parser.add_argument(
        "--stage-b-lock",
        default="results/stage_b_lock/stage_b_lock_manifest.json")
    parser.add_argument(
        "--generated-config-dir",
        default="configs/_stage_f_full_window_cached_pbir")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction,
                        default=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if len(args.seeds) != len(set(args.seeds)):
        raise ValueError("--seeds contains duplicates")
    source = _resolve(args.source_results_dir)
    output = _resolve(args.results_dir)
    generated = _resolve(args.generated_config_dir)
    lock = _resolve(args.stage_b_lock)
    source_manifest = _verify_stage_b_source(source, lock)
    _guard_output_root(output, source)

    cells = [
        (dataset, window, seed)
        for dataset in args.datasets
        for window in args.windows
        for seed in args.seeds
    ]
    print("=" * 78)
    print("STAGE-F EVEN-WINDOW FIXED-CODE PBIR CONTROLLED REPLICATION")
    print(f"Cells: {len(cells)}; device={args.device}; gpu={args.gpu}")
    print("Each cell runs PBIR-GIFT and Pure GIFT sequentially.")
    print("Same cached LLM code/rules inside every pair; LLM calls: 0")
    print(f"Locked Stage-B SHA-256: {source_manifest['aggregate_sha256']}")
    print(f"Results: {output}")
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
            "--selection", args.selection,
            "--seed", str(seed),
            "--results-dir", str(output),
            "--shared-code-source-root", str(source),
            "--generated-config-dir", str(generated),
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
                f"{completed.returncode}. Fix the cause and rerun with --resume.",
                file=sys.stderr,
            )
            return completed.returncode

    state = "generated by dry-run" if args.dry_run else "completed"
    print(f"\nStage-F requested cells {state} successfully.")
    if not args.dry_run and len(cells) == 18:
        print("Next: python analyze_stage_f_full_window_pbir.py "
              "--resamples 10000 --block-length 20")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
