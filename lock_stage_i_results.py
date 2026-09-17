"""Validate, cryptographically lock, or verify the final Stage-I bundle."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from supplemental_experiment_utils import (
    aggregate_inventory,
    inventory,
    read_json,
    resolve_path,
)


REQUIRED = (
    "analysis_report.md",
    "stage_i_analysis.md",
    "stats_appendix.md",
    "figure_catalog.md",
    "stage_i_inference.json",
    "stage_i_method_summary.csv",
    "stage_i_key_contrasts.csv",
    "figures/stage_i_main_method_rank_heatmap.png",
    "figures/stage_i_main_method_rank_heatmap.pdf",
    "figures/stage_i_key_contrast_forest.png",
    "figures/stage_i_key_contrast_forest.pdf",
)


def _csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def validate(root: Path) -> dict[str, Any]:
    for relative in REQUIRED:
        path = root / relative
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"Missing/empty Stage-I artifact: {path}")
    inference = read_json(root / "stage_i_inference.json")
    if inference.get("analysis_role") != "stage-i-final-locked-synthesis":
        raise ValueError("Invalid Stage-I inference role")
    if len(inference.get("key_contrasts", [])) != 6:
        raise ValueError("Stage-I must report all six pre-specified contrasts")
    if len(_csv_rows(root / "stage_i_method_summary.csv")) != 12:
        raise ValueError("Stage-I method summary must contain 12 methods")
    if len(_csv_rows(root / "stage_i_key_contrasts.csv")) != 6:
        raise ValueError("Stage-I contrast table must contain six contrasts")

    upstream = inference.get("source_lock_hashes", {})
    expected_names = [f"stage_{letter}" for letter in "bcdefgh"]
    if set(upstream) != set(expected_names):
        raise ValueError("Stage-I source-lock set is incomplete")
    for name in expected_names:
        path = resolve_path(
            f"results/{name}_lock/{name}_lock_manifest.json")
        manifest = read_json(path)
        if manifest.get("aggregate_sha256") != upstream[name]:
            raise ValueError(f"Upstream lock changed after Stage-I analysis: {name}")
    return {"source_lock_hashes": upstream, "key_contrasts": 6, "methods": 12}


def build(root: Path, output: Path) -> dict[str, Any]:
    validated = validate(root)
    files = inventory(root)
    manifest = {
        "schema_version": 1,
        "role": "locked-stage-i-final-analysis",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_root": str(root),
        **validated,
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
    validate(root)
    current = inventory(root)
    recorded = manifest.get("files")
    if not isinstance(recorded, list):
        raise TypeError("Invalid Stage-I lock inventory")
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
        raise RuntimeError("Stage-I lock verification failed:\n" + "\n".join(problems[:30]))
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results/stage_i_final_comparison")
    parser.add_argument(
        "--output", default="results/stage_i_lock/stage_i_lock_manifest.json")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    output = resolve_path(args.output)
    if args.verify:
        manifest = verify(output)
        print(f"Stage-I lock verified: {manifest['file_count']} files unchanged")
    else:
        root = resolve_path(args.results_dir)
        if output == root or root in output.parents:
            raise ValueError("Lock manifest must stay outside the Stage-I root")
        manifest = build(root, output)
        print("Stage-I final analysis locked")
        print(f"Manifest: {output}")
    print(f"Aggregate SHA-256: {manifest['aggregate_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
