"""Validate, cryptographically lock, or verify Stage-F evidence."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from hparam_sweep_runner import PROJECT_DIR


DATASETS = ("portfolio_5stocks", "portfolio_5stocks2")
WINDOWS = ("W2", "W4", "W6")
SEEDS = (42, 123, 456)
METHODS = ("pbir", "pure_gift")
METRICS = (
    "test_sharpe", "test_sortino", "test_max_drawdown",
    "test_total_return", "test_calmar",
)
REQUIRED_ANALYSIS_FILES = (
    "analysis_bundle/analysis_report.md",
    "analysis_bundle/stage_f_analysis.md",
    "analysis_bundle/stats_appendix.md",
    "analysis_bundle/figure_catalog.md",
    "analysis_bundle/stage_f_inference.json",
    "analysis_bundle/stage_f_seed_metrics.csv",
    "analysis_bundle/stage_f_cell_metrics.csv",
    "analysis_bundle/stage_f_metric_contrasts.csv",
    "analysis_bundle/figures/stage_f_six_window_sharpe.png",
    "analysis_bundle/figures/stage_f_six_window_sharpe.pdf",
    "analysis_bundle/figures/stage_f_sharpe_delta_forest.png",
    "analysis_bundle/figures/stage_f_sharpe_delta_forest.pdf",
)


def _resolve(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_DIR / path).resolve()


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _config_sha256(config: dict[str, Any]) -> str:
    return hashlib.sha256(
        yaml.safe_dump(config, sort_keys=True).encode("utf-8")).hexdigest()


def _without_pbir_switch(config: dict[str, Any]) -> dict[str, Any]:
    value = copy.deepcopy(config)
    value.setdefault("pbir", {}).pop("enabled", None)
    return value


def _finite(value: object) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


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
            f"{item['path']}\0{item['bytes']}\0{item['sha256']}\n".encode(
                "utf-8"))
    return digest.hexdigest()


def validate_stage_f_cells(
        root: Path, require_analysis: bool = False,
) -> tuple[list[Path], list[int]]:
    """Strictly audit the 18 even-window fixed-code paired cells."""
    root = root.resolve()
    expected = {
        (dataset, window, seed)
        for dataset in DATASETS for window in WINDOWS for seed in SEEDS
    }
    summaries = sorted(root.glob("*/W*/seed_*/final_pair_summary.json"))
    observed: set[tuple[str, str, int]] = set()
    for path in summaries:
        try:
            observed.add((
                path.parents[2].name,
                path.parents[1].name,
                int(path.parent.name.removeprefix("seed_")),
            ))
        except ValueError as exc:
            raise ValueError(f"Invalid Stage-F cell path: {path}") from exc
    if observed != expected or len(summaries) != 18:
        raise ValueError(
            "Stage-F grid is incomplete or contaminated: "
            f"missing={sorted(expected - observed)}, "
            f"unexpected={sorted(observed - expected)}")

    return_lengths: list[int] = []
    for pair_path in summaries:
        dataset = pair_path.parents[2].name
        window = pair_path.parents[1].name
        seed = int(pair_path.parent.name.removeprefix("seed_"))
        cell = pair_path.parent
        pair = _read(pair_path)
        if pair.get("status") not in {"completed", "skipped_existing"}:
            raise ValueError(f"Incomplete Stage-F pair: {pair_path}")
        if pair.get("evaluation_role") != "controlled_replication":
            raise ValueError(f"Wrong evaluation role: {pair_path}")
        if (pair.get("dataset"), pair.get("window"), pair.get("seed")) != (
                dataset, window, seed):
            raise ValueError(f"Cell identity mismatch: {pair_path}")

        artifact_path = cell / "shared_artifact" / "shared_artifact.json"
        artifact = _read(artifact_path)
        artifact_sha = _sha256(artifact_path)
        code_path = artifact_path.parent / str(artifact.get("code_file"))
        reward_path = artifact_path.parent / str(
            artifact.get("reward_config_file"))
        if not code_path.is_file() or _sha256(code_path) != artifact.get(
                "code_sha256"):
            raise ValueError(f"Shared code hash mismatch: {code_path}")
        if not reward_path.is_file() or _sha256(reward_path) != artifact.get(
                "reward_config_sha256"):
            raise ValueError(f"Shared reward-config hash mismatch: {reward_path}")
        source = artifact.get("source", {})
        if (source.get("dataset"), source.get("window"), source.get("seed"),
                source.get("method")) != (dataset, window, seed, "pbir"):
            raise ValueError(f"Shared artifact provenance mismatch: {artifact_path}")
        for key, value in (
            ("shared_artifact_sha256", artifact_sha),
            ("shared_code_sha256", artifact["code_sha256"]),
            ("shared_reward_config_sha256", artifact["reward_config_sha256"]),
        ):
            if pair.get(key) != value:
                raise ValueError(f"{key} mismatch: {pair_path}")

        configs: dict[str, dict[str, Any]] = {}
        comparisons: dict[str, dict[str, Any]] = {}
        cell_lengths = []
        for method in METHODS:
            method_dir = cell / "methods" / method
            summary_path = method_dir / "summary.json"
            comparison_path = method_dir / "final_comparison.json"
            model_path = method_dir / "best_model.pt"
            protocol_path = method_dir / "run_protocol.json"
            config_path = method_dir / "resolved_config.yaml"
            for required in (summary_path, comparison_path, protocol_path,
                             config_path, model_path):
                if not required.is_file() or required.stat().st_size == 0:
                    raise FileNotFoundError(f"Missing/empty Stage-F output: {required}")

            summary = _read(summary_path)
            comparison = _read(comparison_path)
            protocol = _read(protocol_path)
            enabled = method == "pbir"
            if summary.get("iterations") != 1:
                raise ValueError(f"Expected one cached-code iteration: {summary_path}")
            if summary.get("pbir", {}).get("enabled") is not enabled:
                raise ValueError(f"Wrong PBIR switch: {summary_path}")
            if comparison.get("pbir", {}).get("enabled") is not enabled:
                raise ValueError(f"Wrong PBIR switch: {comparison_path}")
            for field in ("llm_call_attempts", "llm_call_successes"):
                if summary.get(field) != 0 or pair.get(f"{method}_{field}") != 0:
                    raise ValueError(f"Non-zero {field}: {method_dir}")

            for fixed_record in (
                    summary.get("fixed_shared_artifact", {}),
                    comparison.get("fixed_shared_artifact", {})):
                if fixed_record.get("manifest_sha256") != artifact_sha:
                    raise ValueError(f"Method artifact mismatch: {method_dir}")
                if fixed_record.get("code_sha256") != artifact["code_sha256"]:
                    raise ValueError(f"Method code mismatch: {method_dir}")
                if (fixed_record.get("reward_config_sha256")
                        != artifact["reward_config_sha256"]):
                    raise ValueError(f"Method reward-config mismatch: {method_dir}")

            config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            if not isinstance(config, dict):
                raise TypeError(f"Invalid resolved config: {config_path}")
            if _config_sha256(config) != pair.get(f"{method}_config_sha256"):
                raise ValueError(f"Resolved config hash mismatch: {config_path}")
            if protocol.get("config_sha256") != pair.get(
                    f"{method}_config_sha256"):
                raise ValueError(f"Run protocol config hash mismatch: {protocol_path}")

            result = comparison.get("test_result", {})
            for metric in METRICS:
                if not _finite(result.get(metric)):
                    raise ValueError(f"Missing/non-finite {metric}: {comparison_path}")
                if metric != "test_calmar":
                    pair_value = pair.get(f"{method}_{metric}")
                    if (not _finite(pair_value) or not math.isclose(
                            float(pair_value), float(result[metric]),
                            rel_tol=1e-10, abs_tol=1e-12)):
                        raise ValueError(
                            f"Pair/comparison metric mismatch: {method_dir}/{metric}")
            returns = comparison.get("daily_returns", {}).get("method")
            if (not isinstance(returns, list) or len(returns) < 80
                    or not all(_finite(value) for value in returns)):
                raise ValueError(f"Invalid daily returns: {comparison_path}")
            cell_lengths.append(len(returns))
            configs[method] = config
            comparisons[method] = comparison

        if _without_pbir_switch(configs["pbir"]) != _without_pbir_switch(
                configs["pure_gift"]):
            raise ValueError(f"Configs differ beyond pbir.enabled: {cell}")
        if configs["pbir"].get("pbir", {}).get("enabled") is not True:
            raise ValueError(f"PBIR config is not enabled: {cell}")
        if configs["pure_gift"].get("pbir", {}).get("enabled") is not False:
            raise ValueError(f"Pure-GIFT PBIR switch is not disabled: {cell}")
        if len(set(cell_lengths)) != 1:
            raise ValueError(f"Unaligned paired daily returns: {cell}")
        if comparisons["pbir"].get("eval_period") != comparisons[
                "pure_gift"].get("eval_period"):
            raise ValueError(f"Paired evaluation period mismatch: {cell}")
        return_lengths.append(cell_lengths[0])

    if require_analysis:
        for relative in REQUIRED_ANALYSIS_FILES:
            path = root / relative
            if not path.is_file() or path.stat().st_size == 0:
                raise ValueError(f"Missing/empty Stage-F analysis artifact: {path}")
        inference = _read(root / "analysis_bundle" / "stage_f_inference.json")
        if inference.get("complete_seed_pairs") != 36:
            raise ValueError("Stage-F analysis does not cover all 36 seed pairs")
        if inference.get("dataset_window_cells") != 12:
            raise ValueError("Stage-F analysis does not cover all 12 cells")
        if inference.get("stage_f_even_window_pairs") != 18:
            raise ValueError("Stage-F analysis does not identify all 18 new pairs")
    return summaries, return_lengths


def build(root: Path, output: Path) -> dict[str, object]:
    if not root.is_dir():
        raise FileNotFoundError(root)
    summaries, lengths = validate_stage_f_cells(root, require_analysis=True)
    files = _inventory(root)
    manifest: dict[str, object] = {
        "schema_version": 1,
        "role": "locked-stage-f-full-window-fixed-code-pbir-replication",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_root": str(root),
        "completed_even_window_paired_cells": len(summaries),
        "full_analysis_seed_pairs": 36,
        "full_analysis_dataset_window_cells": 12,
        "methods_per_cell": list(METHODS),
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
    validate_stage_f_cells(root, require_analysis=True)
    current = _inventory(root)
    recorded = manifest.get("files")
    if not isinstance(recorded, list):
        raise TypeError("Invalid Stage-F lock inventory")
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
            "Stage-F lock verification failed:\n" + "\n".join(problems[:30]))
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir", default="results/stage_f_full_window_cached_pbir")
    parser.add_argument(
        "--output", default="results/stage_f_lock/stage_f_lock_manifest.json")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    output = _resolve(args.output)
    if args.verify:
        manifest = verify(output)
        print(f"Stage-F lock verified: {manifest['file_count']} files unchanged")
    else:
        root = _resolve(args.results_dir)
        if output == root or root in output.parents:
            raise ValueError("Lock manifest must be stored outside the Stage-F root")
        manifest = build(root, output)
        print(
            "Stage-F locked: "
            f"{manifest['completed_even_window_paired_cells']} even-window cells")
        print(f"Manifest: {output}")
    print(f"Aggregate SHA-256: {manifest['aggregate_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
