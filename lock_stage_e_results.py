"""Create or verify a strict cryptographic lock for Stage-E evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from hparam_sweep_runner import PROJECT_DIR


DATASETS = ("portfolio_5stocks", "portfolio_5stocks2")
WINDOWS = ("W2", "W4", "W6")
SEEDS = (42, 123, 456)
EXPECTED_METHODS = ("ppo_test_only", "ppo_train_plus_test", "equal_weight")
REQUIRED_ANALYSIS_FILES = (
    "analysis_bundle/analysis_report.md",
    "analysis_bundle/stage_e_analysis.md",
    "analysis_bundle/stats_appendix.md",
    "analysis_bundle/figure_catalog.md",
    "analysis_bundle/stage_e_inference.json",
    "analysis_bundle/stage_e_seed_metrics.csv",
    "analysis_bundle/stage_e_cell_metrics.csv",
    "analysis_bundle/stage_e_paper_style_wins.csv",
    "analysis_bundle/figures/stage_e_six_window_sharpe.png",
    "analysis_bundle/figures/stage_e_six_window_sharpe.pdf",
    "analysis_bundle/figures/stage_e_light_mix_paper_replication.png",
    "analysis_bundle/figures/stage_e_light_mix_paper_replication.pdf",
)


def _resolve(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_DIR / path).resolve()


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _inventory(root: Path) -> list[dict[str, object]]:
    return [
        {
            "path": str(path.relative_to(root)).replace("\\", "/"),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in sorted(item for item in root.rglob("*") if item.is_file())
    ]


def _aggregate(files: list[dict[str, object]]) -> str:
    digest = hashlib.sha256()
    for item in files:
        digest.update(
            f"{item['path']}\0{item['bytes']}\0{item['sha256']}\n".encode("utf-8"))
    return digest.hexdigest()


def _validate_cells(root: Path) -> tuple[list[Path], list[int]]:
    expected = {
        (dataset, window, seed)
        for dataset in DATASETS for window in WINDOWS for seed in SEEDS
    }
    summaries = sorted(root.glob("*/W*/seed_*/baseline_summary.json"))
    observed = {
        (
            path.parents[2].name,
            path.parents[1].name,
            int(path.parent.name.removeprefix("seed_")),
        )
        for path in summaries
    }
    if observed != expected or len(summaries) != 18:
        raise ValueError(
            "Refusing to lock incomplete Stage-E results: "
            f"missing={sorted(expected - observed)}, "
            f"unexpected={sorted(observed - expected)}")

    lengths: list[int] = []
    for path in summaries:
        summary = _read(path)
        if summary.get("status") != "completed" or summary.get("llm_calls") != 0:
            raise ValueError(f"Invalid Stage-E summary status/LLM count: {path}")
        if summary.get("protocol") != "stage-e-even-window-baseline-completion":
            raise ValueError(f"Unexpected Stage-E protocol: {path}")
        if not summary.get("source_stage_b_lock_sha256"):
            raise ValueError(f"Missing Stage-B lock provenance: {path}")
        cell_lengths = []
        for method in EXPECTED_METHODS:
            method_path = path.parent / "methods" / f"{method}.json"
            if not method_path.is_file():
                raise ValueError(f"Missing Stage-E method result: {method_path}")
            value = _read(method_path)
            if value.get("status") != "completed" or value.get("method") != method:
                raise ValueError(f"Incomplete/mislabeled Stage-E result: {method_path}")
            if value.get("llm_call_attempts") != 0:
                raise ValueError(f"Unexpected Stage-E LLM call: {method_path}")
            returns = value.get("daily_returns")
            if not isinstance(returns, list) or len(returns) < 80:
                raise ValueError(f"Too few Stage-E daily returns: {method_path}")
            if not all(isinstance(item, (int, float)) and math.isfinite(item)
                       for item in returns):
                raise ValueError(f"Non-finite Stage-E daily return: {method_path}")
            result = value.get("test_result", {})
            required = (
                "test_total_return", "test_sharpe", "test_sortino",
                "test_max_drawdown", "test_calmar")
            if not all(isinstance(result.get(key), (int, float))
                       and math.isfinite(result[key]) for key in required):
                raise ValueError(f"Missing/non-finite Stage-E metric: {method_path}")
            cell_lengths.append(len(returns))
        if len(set(cell_lengths)) != 1:
            raise ValueError(f"Unaligned Stage-E method returns: {path.parent}")
        lengths.append(cell_lengths[0])

    for relative in REQUIRED_ANALYSIS_FILES:
        path = root / relative
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"Missing/empty Stage-E analysis artifact: {path}")
    inference = _read(root / "analysis_bundle" / "stage_e_inference.json")
    if inference.get("complete_seed_runs") != 36:
        raise ValueError("Stage-E analysis does not cover all 36 six-window runs")
    if inference.get("independent_dataset_window_cells") != 12:
        raise ValueError("Stage-E analysis does not cover all 12 cells")
    return summaries, lengths


def build(root: Path, output: Path) -> dict[str, object]:
    if not root.is_dir():
        raise FileNotFoundError(root)
    summaries, lengths = _validate_cells(root)
    files = _inventory(root)
    manifest: dict[str, object] = {
        "schema_version": 1,
        "role": "locked-stage-e-full-window-baseline-completion",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_root": str(root),
        "completed_even_window_cells": len(summaries),
        "full_analysis_seed_runs": 36,
        "full_analysis_dataset_window_cells": 12,
        "methods_per_even_window_cell": list(EXPECTED_METHODS),
        "llm_calls": 0,
        "daily_return_lengths": sorted(set(lengths)),
        "file_count": len(files),
        "total_bytes": sum(int(item["bytes"]) for item in files),
        "aggregate_sha256": _aggregate(files),
        "files": files,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def verify(output: Path) -> dict[str, object]:
    if not output.is_file():
        raise FileNotFoundError(output)
    manifest = _read(output)
    root = Path(str(manifest["source_root"])).resolve()
    _validate_cells(root)
    current = _inventory(root)
    recorded = manifest["files"]
    if not isinstance(recorded, list):
        raise TypeError("Invalid Stage-E lock inventory")
    current_by_path = {str(item["path"]): item for item in current}
    recorded_by_path = {str(item["path"]): item for item in recorded}
    problems = []
    for path in sorted(recorded_by_path.keys() - current_by_path.keys()):
        problems.append(f"missing: {path}")
    for path in sorted(current_by_path.keys() - recorded_by_path.keys()):
        problems.append(f"unexpected: {path}")
    for path in sorted(recorded_by_path.keys() & current_by_path.keys()):
        if current_by_path[path] != recorded_by_path[path]:
            problems.append(f"changed: {path}")
    if _aggregate(current) != manifest.get("aggregate_sha256"):
        problems.append("aggregate SHA-256 mismatch")
    if problems:
        raise RuntimeError(
            "Stage-E lock verification failed:\n" + "\n".join(problems[:30]))
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir", default="results/stage_e_full_window_baselines")
    parser.add_argument(
        "--output", default="results/stage_e_lock/stage_e_lock_manifest.json")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    output = _resolve(args.output)
    if args.verify:
        manifest = verify(output)
        print(f"Stage-E lock verified: {manifest['file_count']} files unchanged")
    else:
        root = _resolve(args.results_dir)
        if output == root or root in output.parents:
            raise ValueError("Lock manifest must be stored outside the Stage-E root")
        manifest = build(root, output)
        print(
            "Stage-E locked: "
            f"{manifest['completed_even_window_cells']} even-window cells")
        print(f"Manifest: {output}")
    print(f"Aggregate SHA-256: {manifest['aggregate_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
