"""Validate, cryptographically lock, or verify Stage-H evidence."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

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
from traditional_baselines import DETERMINISTIC_METHODS, METHODS


REQUIRED_ANALYSIS = (
    "analysis_bundle/analysis_report.md",
    "analysis_bundle/stage_h_analysis.md",
    "analysis_bundle/stats_appendix.md",
    "analysis_bundle/figure_catalog.md",
    "analysis_bundle/stage_h_inference.json",
    "analysis_bundle/stage_h_run_metrics.csv",
    "analysis_bundle/stage_h_cell_metrics.csv",
    "analysis_bundle/stage_h_contrasts.csv",
    "analysis_bundle/figures/stage_h_sharpe_heatmap.png",
    "analysis_bundle/figures/stage_h_sharpe_heatmap.pdf",
    "analysis_bundle/figures/stage_h_method_mean_sharpe.png",
    "analysis_bundle/figures/stage_h_method_mean_sharpe.pdf",
)


def validate(root: Path, require_analysis: bool = True) -> dict[str, Any]:
    frozen_strategy = root / "frozen_strategy_config.yaml"
    if not frozen_strategy.is_file() or frozen_strategy.stat().st_size == 0:
        raise FileNotFoundError(frozen_strategy)
    strategy = yaml.safe_load(frozen_strategy.read_text(encoding="utf-8")) or {}
    counts = {method: 0 for method in METHODS}
    lengths: set[int] = set()
    for dataset in DATASETS:
        for window in WINDOWS:
            resolved = root / dataset / window / "resolved_market_config.yaml"
            base_config = yaml.safe_load(resolved.read_text(encoding="utf-8")) or {}
            summary_path = root / dataset / window / "baseline_summary.json"
            summary = read_json(summary_path)
            if summary.get("status") != "completed" or summary.get("llm_calls") != 0:
                raise ValueError(f"Incomplete Stage-H cell: {summary_path}")
            cell_lengths = []
            for method in DETERMINISTIC_METHODS:
                path = root / dataset / window / "methods" / f"{method}.json"
                record = validate_result_record(path, method)
                if record.get("deterministic") is not True or record.get("seed") is not None:
                    raise ValueError(f"Invalid deterministic provenance: {path}")
                if record.get("lookahead_safe") is not True:
                    raise ValueError(f"Look-ahead audit flag missing: {path}")
                if record.get("llm_call_attempts") != 0:
                    raise ValueError(f"Unexpected LLM call: {path}")
                expected_sha = canonical_yaml_sha({
                    "base_config": base_config,
                    "strategy_protocol": strategy.get("protocol"),
                    "portfolio_protocol": strategy.get("portfolio"),
                    "method": method,
                    "method_parameters": strategy["methods"][method],
                    "seed": None,
                })
                if record.get("config_sha256") != expected_sha:
                    raise ValueError(f"Stage-H config hash mismatch: {path}")
                counts[method] += 1
                cell_lengths.append(len(record["daily_returns"]))
            for seed in SEEDS:
                path = (root / dataset / window / "methods" / "xgboost"
                        / f"seed_{seed}.json")
                record = validate_result_record(path, "xgboost")
                if record.get("seed") != seed or record.get("deterministic") is not False:
                    raise ValueError(f"Invalid XGBoost provenance: {path}")
                if record.get("lookahead_safe") is not True:
                    raise ValueError(f"Look-ahead audit flag missing: {path}")
                if record.get("llm_call_attempts") != 0:
                    raise ValueError(f"Unexpected LLM call: {path}")
                expected_sha = canonical_yaml_sha({
                    "base_config": base_config,
                    "strategy_protocol": strategy.get("protocol"),
                    "portfolio_protocol": strategy.get("portfolio"),
                    "method": "xgboost",
                    "method_parameters": strategy["methods"]["xgboost"],
                    "seed": seed,
                })
                if record.get("config_sha256") != expected_sha:
                    raise ValueError(f"Stage-H config hash mismatch: {path}")
                counts["xgboost"] += 1
                cell_lengths.append(len(record["daily_returns"]))
            if len(set(cell_lengths)) != 1:
                raise ValueError(f"Unaligned Stage-H daily returns: {dataset}/{window}")
            lengths.add(cell_lengths[0])
    if any(counts[method] != 12 for method in DETERMINISTIC_METHODS):
        raise ValueError(f"Incomplete deterministic Stage-H grid: {counts}")
    if counts["xgboost"] != 36:
        raise ValueError(f"Incomplete XGBoost grid: {counts['xgboost']}")

    if require_analysis:
        for relative in REQUIRED_ANALYSIS:
            path = root / relative
            if not path.is_file() or path.stat().st_size == 0:
                raise ValueError(f"Missing/empty Stage-H artifact: {path}")
        inference = read_json(root / "analysis_bundle" / "stage_h_inference.json")
        if inference.get("dataset_window_cells") != 12:
            raise ValueError("Stage-H analysis does not cover all cells")
        if set(inference.get("all_methods_reported", [])) != set(METHODS):
            raise ValueError("Stage-H analysis does not report all six baselines")
    return {"method_run_counts": counts, "daily_return_lengths": sorted(lengths)}


def build(root: Path, output: Path) -> dict[str, Any]:
    validated = validate(root, require_analysis=True)
    files = inventory(root)
    manifest = {
        "schema_version": 1,
        "role": "locked-stage-h-traditional-baselines",
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
        raise TypeError("Invalid Stage-H lock inventory")
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
        raise RuntimeError("Stage-H lock verification failed:\n" + "\n".join(problems[:30]))
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results/stage_h_traditional_baselines")
    parser.add_argument(
        "--output", default="results/stage_h_lock/stage_h_lock_manifest.json")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    output = resolve_path(args.output)
    if args.verify:
        manifest = verify(output)
        print(f"Stage-H lock verified: {manifest['file_count']} files unchanged")
    else:
        root = resolve_path(args.results_dir)
        if output == root or root in output.parents:
            raise ValueError("Lock manifest must stay outside the Stage-H root")
        manifest = build(root, output)
        print("Stage-H locked: six traditional baselines across 12 cells")
        print(f"Manifest: {output}")
    print(f"Aggregate SHA-256: {manifest['aggregate_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
