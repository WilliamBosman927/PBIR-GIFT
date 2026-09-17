"""Create or verify a strict cryptographic lock for Stage-D evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from hparam_sweep_runner import PROJECT_DIR


EXPECTED_METHODS = ("ppo_test_only", "ppo_train_plus_test", "equal_weight")


def _resolve(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_DIR / path).resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _inventory(root: Path) -> list[dict[str, object]]:
    records = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        records.append({
            "path": str(path.relative_to(root)).replace("\\", "/"),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        })
    return records


def _aggregate(files: list[dict[str, object]]) -> str:
    digest = hashlib.sha256()
    for item in files:
        digest.update(
            f"{item['path']}\0{item['bytes']}\0{item['sha256']}\n".encode("utf-8"))
    return digest.hexdigest()


def _validate_cells(root: Path) -> list[Path]:
    summaries = sorted(root.glob("*/W[135]/seed_*/baseline_summary.json"))
    if len(summaries) != 18:
        raise ValueError(f"Expected 18 completed Stage-D cells, found {len(summaries)}")
    for path in summaries:
        summary = json.loads(path.read_text(encoding="utf-8"))
        if summary.get("status") != "completed" or summary.get("llm_calls") != 0:
            raise ValueError(f"Invalid Stage-D summary status/LLM count: {path}")
        for method in EXPECTED_METHODS:
            method_path = path.parent / "methods" / f"{method}.json"
            if not method_path.is_file():
                raise ValueError(f"Missing Stage-D method result: {method_path}")
            value = json.loads(method_path.read_text(encoding="utf-8"))
            if value.get("status") != "completed":
                raise ValueError(f"Incomplete Stage-D method result: {method_path}")
            if value.get("llm_call_attempts") != 0:
                raise ValueError(f"Unexpected Stage-D LLM call: {method_path}")
            returns = value.get("daily_returns", [])
            if len(returns) != 105:
                raise ValueError(
                    f"Expected 105 daily returns, found {len(returns)}: {method_path}")
    return summaries


def build(root: Path, output: Path) -> dict[str, object]:
    summaries = _validate_cells(root)
    files = _inventory(root)
    manifest = {
        "schema_version": 1,
        "role": "locked-stage-d-ppo-and-equal-weight-baselines",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_root": str(root),
        "completed_cells": len(summaries),
        "methods_per_cell": list(EXPECTED_METHODS),
        "llm_calls": 0,
        "daily_returns_per_method_cell": 105,
        "file_count": len(files),
        "total_bytes": sum(int(item["bytes"]) for item in files),
        "aggregate_sha256": _aggregate(files),
        "files": files,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def verify(output: Path) -> dict[str, object]:
    manifest = json.loads(output.read_text(encoding="utf-8"))
    root = Path(manifest["source_root"])
    _validate_cells(root)
    current = _inventory(root)
    recorded = manifest["files"]
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
    if _aggregate(current) != manifest["aggregate_sha256"]:
        problems.append("aggregate SHA-256 mismatch")
    if problems:
        raise RuntimeError(
            "Stage-D lock verification failed:\n" + "\n".join(problems[:30]))
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results/stage_d_ppo_baselines")
    parser.add_argument(
        "--output", default="results/stage_d_lock/stage_d_lock_manifest.json")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    output = _resolve(args.output)
    if args.verify:
        manifest = verify(output)
        print(f"Stage-D lock verified: {manifest['file_count']} files unchanged")
    else:
        root = _resolve(args.results_dir)
        if output == root or root in output.parents:
            raise ValueError("Lock manifest must be stored outside the Stage-D root")
        manifest = build(root, output)
        print(f"Stage-D locked: {manifest['completed_cells']} cells")
        print(f"Manifest: {output}")
    print(f"Aggregate SHA-256: {manifest['aggregate_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
