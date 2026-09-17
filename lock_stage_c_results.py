"""Create or verify a strict cryptographic lock for Stage-C evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from hparam_sweep_runner import PROJECT_DIR


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


def build(root: Path, output: Path) -> dict[str, object]:
    pair_files = sorted(root.glob("*/W[135]/seed_*/final_pair_summary.json"))
    if len(pair_files) != 18:
        raise ValueError(f"Expected 18 completed Stage-C cells, found {len(pair_files)}")
    for path in pair_files:
        pair = json.loads(path.read_text(encoding="utf-8"))
        if pair.get("status") not in {"completed", "skipped_existing"}:
            raise ValueError(f"Incomplete Stage-C cell: {path}")
    files = _inventory(root)
    manifest = {
        "schema_version": 1,
        "role": "locked-stage-c-fixed-code-controlled-replication",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_root": str(root),
        "completed_paired_cells": len(pair_files),
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
    current = _inventory(root)
    recorded = manifest["files"]
    problems = []
    current_paths = {item["path"] for item in current}
    recorded_paths = {item["path"] for item in recorded}
    if current_paths != recorded_paths:
        problems.extend(
            f"missing: {path}" for path in sorted(recorded_paths - current_paths))
        problems.extend(
            f"unexpected: {path}" for path in sorted(current_paths - recorded_paths))
    current_by_path = {item["path"]: item for item in current}
    for item in recorded:
        observed = current_by_path.get(item["path"])
        if observed is not None and (
                observed["bytes"] != item["bytes"]
                or observed["sha256"] != item["sha256"]):
            problems.append(f"changed: {item['path']}")
    if _aggregate(current) != manifest["aggregate_sha256"]:
        problems.append("aggregate SHA-256 mismatch")
    if problems:
        raise RuntimeError(
            "Stage-C lock verification failed:\n" + "\n".join(problems[:30]))
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir", default="results/stage_c_cached_replication")
    parser.add_argument(
        "--output", default="results/stage_c_lock/stage_c_lock_manifest.json")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    output = _resolve(args.output)
    if args.verify:
        manifest = verify(output)
        print(f"Stage-C lock verified: {manifest['file_count']} files unchanged")
    else:
        root = _resolve(args.results_dir)
        if output == root or root in output.parents:
            raise ValueError("Lock manifest must be stored outside the Stage-C root")
        manifest = build(root, output)
        print(f"Stage-C locked: {manifest['completed_paired_cells']} paired cells")
        print(f"Manifest: {output}")
    print(f"Aggregate SHA-256: {manifest['aggregate_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
