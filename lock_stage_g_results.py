"""Validate, cryptographically lock, or verify Stage-G evidence."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from hparam_sweep_runner import PARAMETER_GROUPS
from supplemental_experiment_utils import (
    DATASETS,
    SEEDS,
    WINDOWS,
    aggregate_inventory,
    canonical_yaml_sha,
    inventory,
    read_json,
    resolve_path,
    validate_result_record,
)


REQUIRED_ANALYSIS = (
    "analysis_bundle/analysis_report.md",
    "analysis_bundle/stage_g_analysis.md",
    "analysis_bundle/stats_appendix.md",
    "analysis_bundle/figure_catalog.md",
    "analysis_bundle/stage_g_inference.json",
    "analysis_bundle/stage_g_seed_metrics.csv",
    "analysis_bundle/stage_g_cell_metrics.csv",
    "analysis_bundle/figures/stage_g_six_window_sharpe.png",
    "analysis_bundle/figures/stage_g_six_window_sharpe.pdf",
    "analysis_bundle/figures/stage_g_sharpe_delta_forest.png",
    "analysis_bundle/figures/stage_g_sharpe_delta_forest.pdf",
)


def validate(root: Path, require_analysis: bool = True) -> dict[str, Any]:
    selection_path = root / "selection" / "selected_ppo_tuned.yaml"
    if not selection_path.is_file():
        raise FileNotFoundError(selection_path)
    selection = yaml.safe_load(selection_path.read_text(encoding="utf-8")) or {}
    if selection.get("protocol", {}).get("stage") != "G_selected":
        raise ValueError("Invalid Stage-G selection artifact")
    if set(selection.get("ppo", {})) != set(PARAMETER_GROUPS):
        raise ValueError("Stage-G selection does not contain all tuned parameters")
    if selection.get("protocol", {}).get("validation_period") != [
            "2020-01-01", "2020-06-30"]:
        raise ValueError("Stage-G selection used the wrong validation period")
    selection_seeds = selection.get("protocol", {}).get("selection_seeds")
    if not isinstance(selection_seeds, list) or not selection_seeds:
        raise ValueError("Stage-G selection seeds are missing")

    tuned = 0
    for parameter, values in PARAMETER_GROUPS.items():
        for level, _ in enumerate(values, start=1):
            for seed in selection_seeds:
                path = (root / "tuning" / parameter / f"level_{level}"
                        / f"seed_{seed}" / "result.json")
                record = validate_result_record(path, "ppo_tuned")
                resolved = path.parent / "resolved_config.yaml"
                config = yaml.safe_load(resolved.read_text(encoding="utf-8")) or {}
                if canonical_yaml_sha(config) != record.get("config_sha256"):
                    raise ValueError(f"Stage-G tuning config hash mismatch: {resolved}")
                if record.get("evaluation_role") != "ppo_hyperparameter_validation":
                    raise ValueError(f"Wrong Stage-G tuning role: {path}")
                if record.get("parameter") != parameter or record.get("level") != level:
                    raise ValueError(f"Stage-G tuning identity mismatch: {path}")
                if record.get("llm_call_attempts") != 0:
                    raise ValueError(f"Unexpected LLM call in Stage-G: {path}")
                tuned += 1

    evaluated = 0
    for dataset in DATASETS:
        for window in WINDOWS:
            lengths = []
            for seed in SEEDS:
                path = (root / "evaluation" / dataset / window / f"seed_{seed}"
                        / "methods" / "ppo_tuned.json")
                record = validate_result_record(path, "ppo_tuned")
                resolved = path.parents[1] / "resolved_config.yaml"
                config = yaml.safe_load(resolved.read_text(encoding="utf-8")) or {}
                if canonical_yaml_sha(config) != record.get("config_sha256"):
                    raise ValueError(f"Stage-G final config hash mismatch: {resolved}")
                for key, value in selection["ppo"].items():
                    if config.get("ppo", {}).get(key) != value:
                        raise ValueError(f"Stage-G selected PPO mismatch: {resolved}/{key}")
                if record.get("evaluation_role") != "tuned_ppo_final_test":
                    raise ValueError(f"Wrong Stage-G final role: {path}")
                if (record.get("dataset"), record.get("window"), record.get("seed")) != (
                        dataset, window, seed):
                    raise ValueError(f"Stage-G final identity mismatch: {path}")
                if record.get("llm_call_attempts") != 0:
                    raise ValueError(f"Unexpected LLM call in Stage-G: {path}")
                lengths.append(len(record["daily_returns"]))
                evaluated += 1
            if len(set(lengths)) != 1:
                raise ValueError(f"Unaligned Stage-G return lengths: {dataset}/{window}")

    if require_analysis:
        for relative in REQUIRED_ANALYSIS:
            path = root / relative
            if not path.is_file() or path.stat().st_size == 0:
                raise ValueError(f"Missing/empty Stage-G artifact: {path}")
        inference = read_json(root / "analysis_bundle" / "stage_g_inference.json")
        if inference.get("complete_seed_runs") != 36:
            raise ValueError("Stage-G analysis does not cover 36 final runs")
        if inference.get("dataset_window_cells") != 12:
            raise ValueError("Stage-G analysis does not cover 12 cells")
    return {
        "selection_seeds": [int(seed) for seed in selection_seeds],
        "tuning_runs": tuned,
        "final_runs": evaluated,
        "selected_ppo": selection["ppo"],
    }


def build(root: Path, output: Path) -> dict[str, Any]:
    validated = validate(root, require_analysis=True)
    files = inventory(root)
    manifest = {
        "schema_version": 1,
        "role": "locked-stage-g-independently-tuned-ppo",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_root": str(root),
        **validated,
        "llm_calls": 0,
        "file_count": len(files),
        "total_bytes": sum(int(item["bytes"]) for item in files),
        "aggregate_sha256": aggregate_inventory(files),
        "files": files,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def verify(output: Path) -> dict[str, Any]:
    if not output.is_file():
        raise FileNotFoundError(output)
    manifest = read_json(output)
    root = Path(str(manifest["source_root"])).resolve()
    validate(root, require_analysis=True)
    current = inventory(root)
    recorded = manifest.get("files")
    if not isinstance(recorded, list):
        raise TypeError("Invalid Stage-G lock inventory")
    current_by_path = {item["path"]: item for item in current}
    recorded_by_path = {item["path"]: item for item in recorded}
    problems = []
    for path in sorted(recorded_by_path.keys() - current_by_path.keys()):
        problems.append(f"missing: {path}")
    for path in sorted(current_by_path.keys() - recorded_by_path.keys()):
        problems.append(f"unexpected: {path}")
    for path in sorted(recorded_by_path.keys() & current_by_path.keys()):
        if current_by_path[path] != recorded_by_path[path]:
            problems.append(f"changed: {path}")
    if aggregate_inventory(current) != manifest.get("aggregate_sha256"):
        problems.append("aggregate SHA-256 mismatch")
    if problems:
        raise RuntimeError("Stage-G lock verification failed:\n" + "\n".join(problems[:30]))
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results/stage_g_tuned_ppo")
    parser.add_argument(
        "--output", default="results/stage_g_lock/stage_g_lock_manifest.json")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    output = resolve_path(args.output)
    if args.verify:
        manifest = verify(output)
        print(f"Stage-G lock verified: {manifest['file_count']} files unchanged")
    else:
        root = resolve_path(args.results_dir)
        if output == root or root in output.parents:
            raise ValueError("Lock manifest must stay outside the Stage-G root")
        manifest = build(root, output)
        print(f"Stage-G locked: {manifest['final_runs']} final PPO runs")
        print(f"Manifest: {output}")
    print(f"Aggregate SHA-256: {manifest['aggregate_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
