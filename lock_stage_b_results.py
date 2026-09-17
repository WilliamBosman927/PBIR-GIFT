"""Create or verify a lightweight cryptographic lock for Stage-B evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_DIR = Path(__file__).resolve().parent
LOCKED_SUFFIXES = {".json", ".csv", ".yaml", ".yml", ".py", ".txt", ".md"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve(path: str) -> Path:
    value = Path(path)
    return value.resolve() if value.is_absolute() else (PROJECT_DIR / value).resolve()


def build_lock(root: Path, output: Path) -> dict[str, Any]:
    pair_files = sorted(root.glob("*/W[1-6]/seed_*/final_pair_summary.json"))
    if len(pair_files) != 36:
        raise ValueError(
            f"Refusing to lock incomplete Stage-B: expected 36 cells, found {len(pair_files)}")
    for path in pair_files:
        row = json.loads(path.read_text(encoding="utf-8"))
        if row.get("status") not in {"completed", "skipped_existing"}:
            raise ValueError(f"Incomplete Stage-B cell: {path}")

    files = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if path.suffix.lower() not in LOCKED_SUFFIXES:
            continue
        files.append({
            "path": str(path.relative_to(root)).replace("\\", "/"),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        })
    aggregate = hashlib.sha256()
    for item in files:
        aggregate.update(
            f"{item['path']}\0{item['bytes']}\0{item['sha256']}\n".encode("utf-8"))
    manifest = {
        "schema_version": 1,
        "role": "locked-stage-b-pipeline-comparison",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_root": str(root),
        "completed_paired_cells": len(pair_files),
        "locked_suffixes": sorted(LOCKED_SUFFIXES),
        "excluded_binary_artifacts": [".pt", ".png", ".pdf"],
        "file_count": len(files),
        "aggregate_sha256": aggregate.hexdigest(),
        "files": files,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def verify_lock(manifest_path: Path) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    root = Path(manifest["source_root"])
    problems = []
    aggregate = hashlib.sha256()
    for item in manifest["files"]:
        path = root / item["path"]
        if not path.is_file():
            problems.append(f"missing: {item['path']}")
            continue
        observed_size = path.stat().st_size
        observed_sha = _sha256(path)
        if observed_size != item["bytes"] or observed_sha != item["sha256"]:
            problems.append(f"changed: {item['path']}")
        aggregate.update(
            f"{item['path']}\0{observed_size}\0{observed_sha}\n".encode("utf-8"))
    if aggregate.hexdigest() != manifest["aggregate_sha256"]:
        problems.append("aggregate SHA-256 mismatch")
    if problems:
        raise RuntimeError("Stage-B lock verification failed:\n" + "\n".join(problems[:20]))
    print(f"Stage-B lock verified: {len(manifest['files'])} files unchanged")
    print(f"Aggregate SHA-256: {manifest['aggregate_sha256']}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results/stage_b_final")
    parser.add_argument("--output",
                        default="results/stage_b_lock/stage_b_lock_manifest.json")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    output = _resolve(args.output)
    if args.verify:
        verify_lock(output)
        return 0
    root = _resolve(args.results_dir)
    if not root.is_dir():
        raise FileNotFoundError(root)
    if output == root or root in output.parents:
        raise ValueError("Lock manifest must be stored outside the Stage-B root")
    manifest = build_lock(root, output)
    print(f"Stage-B locked: {manifest['completed_paired_cells']} paired cells")
    print(f"Manifest: {output}")
    print(f"Aggregate SHA-256: {manifest['aggregate_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
