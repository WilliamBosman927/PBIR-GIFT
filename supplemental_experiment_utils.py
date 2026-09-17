"""Shared utilities for the post-Stage-F supplemental experiments.

The helpers in this module intentionally do not mutate any Stage-A--F result
tree.  Stage-G/H/I use separate output roots and record canonical hashes for
every resolved configuration and result artifact.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml


PROJECT_DIR = Path(__file__).resolve().parent
DATASETS = ("portfolio_5stocks", "portfolio_5stocks2")
WINDOWS = ("W1", "W2", "W3", "W4", "W5", "W6")
SEEDS = (42, 123, 456)
METRICS = (
    "test_total_return",
    "test_sharpe",
    "test_sortino",
    "test_max_drawdown",
    "test_calmar",
)
PROTECTED_RESULT_ROOTS = (
    "results/stage_a_tuning",
    "results/stage_b_final",
    "results/stage_b_lock",
    "results/stage_c_cached_replication",
    "results/stage_c_lock",
    "results/stage_d_ppo_baselines",
    "results/stage_d_lock",
    "results/stage_e_full_window_baselines",
    "results/stage_e_lock",
    "results/stage_f_full_window_cached_pbir",
    "results/stage_f_lock",
)


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_DIR / path).resolve()


def ensure_isolated_output(path: Path, extra: Iterable[Path] = ()) -> None:
    """Refuse output paths that overlap locked or historical evidence."""
    target = path.resolve()
    protected = [resolve_path(item) for item in PROTECTED_RESULT_ROOTS]
    protected.extend(item.resolve() for item in extra)
    for root in protected:
        if target == root or root in target.parents or target in root.parents:
            raise ValueError(
                f"Supplemental output {target} overlaps protected evidence {root}")


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, default=str, allow_nan=False),
        encoding="utf-8",
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def canonical_yaml_sha(value: dict[str, Any]) -> str:
    return hashlib.sha256(
        yaml.safe_dump(value, sort_keys=True).encode("utf-8")
    ).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inventory(root: Path) -> list[dict[str, Any]]:
    return [
        {
            "path": str(path.relative_to(root)).replace("\\", "/"),
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }
        for path in sorted(item for item in root.rglob("*") if item.is_file())
    ]


def aggregate_inventory(files: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for item in files:
        digest.update(
            f"{item['path']}\0{item['bytes']}\0{item['sha256']}\n".encode("utf-8")
        )
    return digest.hexdigest()


def finite(value: object) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def validate_result_record(
    path: Path,
    expected_method: str,
    *,
    minimum_returns: int = 22,
) -> dict[str, Any]:
    record = read_json(path)
    if record.get("status") != "completed":
        raise ValueError(f"Incomplete result: {path}")
    if record.get("method") != expected_method:
        raise ValueError(f"Method mismatch in {path}")
    result = record.get("test_result")
    if not isinstance(result, dict):
        raise ValueError(f"Missing test_result: {path}")
    for metric in METRICS:
        if not finite(result.get(metric)):
            raise ValueError(f"Missing/non-finite {metric}: {path}")
    returns = record.get("daily_returns")
    if (
        not isinstance(returns, list)
        or len(returns) < minimum_returns
        or not all(finite(value) for value in returns)
    ):
        raise ValueError(f"Invalid daily_returns: {path}")
    return record


def completed_result(
    path: Path,
    expected_method: str,
    expected_config_sha: str | None = None,
) -> bool:
    if not path.is_file():
        return False
    try:
        record = validate_result_record(path, expected_method)
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return False
    return expected_config_sha is None or record.get("config_sha256") == expected_config_sha


def base_config_path(dataset: str, window: str) -> Path:
    if dataset not in DATASETS or window not in WINDOWS:
        raise ValueError(f"Unknown dataset/window: {dataset}/{window}")
    prefix = "config" if dataset == "portfolio_5stocks" else "config2"
    return PROJECT_DIR / f"{prefix}_{window}.yaml"


def load_base_config(dataset: str, window: str) -> tuple[dict[str, Any], Path]:
    path = base_config_path(dataset, window)
    if not path.is_file():
        raise FileNotFoundError(path)
    value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(value, dict):
        raise TypeError(f"Config must be a mapping: {path}")
    configured = Path(str(value.get("data", {}).get("pickle_file", "")))
    data_path = configured if configured.is_absolute() else PROJECT_DIR / configured
    if not data_path.is_file():
        raise FileNotFoundError(f"Configured pickle does not exist: {data_path}")
    return value, path


def metric_summary(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=float)
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError("Metric summary requires at least one finite value")
    return {
        "n": int(array.size),
        "mean": float(np.mean(array)),
        "sd": float(np.std(array, ddof=1)) if array.size > 1 else 0.0,
        "median": float(np.median(array)),
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
    }


def holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    """Holm family-wise adjusted p-values, preserving input keys."""
    ordered = sorted(p_values.items(), key=lambda item: item[1])
    adjusted: dict[str, float] = {}
    running = 0.0
    total = len(ordered)
    for rank, (name, value) in enumerate(ordered):
        candidate = min(1.0, float(value) * (total - rank))
        running = max(running, candidate)
        adjusted[name] = running
    return {name: adjusted[name] for name in p_values}
