"""Analyze the complete six-window fixed-code PBIR-GIFT replication.

Odd windows come only from the cryptographically locked Stage-C evidence;
even windows come only from Stage-F.  The primary inferential unit is the
dataset-window cell mean (12 cells, three seeds per cell).  Daily returns are
used inside a hierarchical circular-block bootstrap, not treated as
independent observations.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import itertools
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

from hparam_sweep_runner import PROJECT_DIR
from lock_stage_c_results import verify as verify_stage_c_lock
from lock_stage_f_results import validate_stage_f_cells
from metrics import calmar_ratio, max_drawdown, sharpe_ratio, sortino_ratio


DATASETS = ("portfolio_5stocks", "portfolio_5stocks2")
WINDOWS = tuple(f"W{index}" for index in range(1, 7))
ODD_WINDOWS = ("W1", "W3", "W5")
EVEN_WINDOWS = ("W2", "W4", "W6")
SEEDS = (42, 123, 456)
METHODS = ("pbir", "pure_gift")
METHOD_LABELS = {"pbir": "PBIR-GIFT", "pure_gift": "Pure GIFT"}
METRICS: dict[str, tuple[Callable[[np.ndarray], float], str]] = {
    "sharpe": (sharpe_ratio, "PBIR-GIFT − Pure GIFT"),
    "sortino": (sortino_ratio, "PBIR-GIFT − Pure GIFT"),
    "total_return": (
        lambda values: (float(np.prod(1.0 + np.asarray(values))) - 1.0) * 100,
        "PBIR-GIFT − Pure GIFT (percentage points)"),
    "max_drawdown_improvement": (
        max_drawdown, "Pure GIFT − PBIR-GIFT (percentage points)"),
    "calmar": (calmar_ratio, "PBIR-GIFT − Pure GIFT"),
}
RESULT_KEYS = {
    "sharpe": "test_sharpe",
    "sortino": "test_sortino",
    "total_return": "test_total_return",
    "max_drawdown_improvement": "test_max_drawdown",
    "calmar": "test_calmar",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage-c-dir", default="results/stage_c_cached_replication")
    parser.add_argument(
        "--stage-c-lock",
        default="results/stage_c_lock/stage_c_lock_manifest.json")
    parser.add_argument(
        "--stage-f-dir", default="results/stage_f_full_window_cached_pbir")
    parser.add_argument(
        "--output-dir",
        default="results/stage_f_full_window_cached_pbir/analysis_bundle")
    parser.add_argument("--resamples", type=int, default=10000)
    parser.add_argument("--block-length", type=int, default=20)
    parser.add_argument("--random-seed", type=int, default=20260903)
    return parser.parse_args()


def _resolve(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_DIR / path).resolve()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _without_pbir_switch(config: dict[str, Any]) -> dict[str, Any]:
    value = copy.deepcopy(config)
    value.setdefault("pbir", {}).pop("enabled", None)
    return value


def _finite(value: object) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _verify_stage_c_source(root: Path, lock_path: Path) -> dict[str, Any]:
    manifest = verify_stage_c_lock(lock_path)
    locked_root = Path(str(manifest["source_root"])).resolve()
    if locked_root != root.resolve():
        raise ValueError(
            f"--stage-c-dir does not match its lock: {root} != {locked_root}")
    return manifest


def _validate_grid(root: Path, windows: tuple[str, ...], label: str) -> None:
    expected = {
        (dataset, window, seed)
        for dataset in DATASETS for window in windows for seed in SEEDS
    }
    paths = sorted(root.glob("*/W*/seed_*/final_pair_summary.json"))
    observed = {
        (path.parents[2].name, path.parents[1].name,
         int(path.parent.name.removeprefix("seed_")))
        for path in paths
    }
    if observed != expected or len(paths) != len(expected):
        raise ValueError(
            f"{label} grid mismatch: missing={sorted(expected - observed)}, "
            f"unexpected={sorted(observed - expected)}")


def _load_pair(
        root: Path, dataset: str, window: str, seed: int, source_stage: str,
) -> dict[str, Any]:
    cell = root / dataset / window / f"seed_{seed}"
    pair_path = cell / "final_pair_summary.json"
    pair = _read_json(pair_path)
    if pair.get("status") not in {"completed", "skipped_existing"}:
        raise ValueError(f"Incomplete pair: {pair_path}")
    if pair.get("evaluation_role") != "controlled_replication":
        raise ValueError(f"Not a controlled replication: {pair_path}")
    if (pair.get("dataset"), pair.get("window"), pair.get("seed")) != (
            dataset, window, seed):
        raise ValueError(f"Pair identity mismatch: {pair_path}")

    artifact_path = cell / "shared_artifact" / "shared_artifact.json"
    artifact = _read_json(artifact_path)
    artifact_sha = _sha256(artifact_path)
    if pair.get("shared_artifact_sha256") != artifact_sha:
        raise ValueError(f"Shared artifact mismatch: {pair_path}")
    code_path = artifact_path.parent / str(artifact.get("code_file"))
    reward_path = artifact_path.parent / str(artifact.get("reward_config_file"))
    if _sha256(code_path) != artifact.get("code_sha256"):
        raise ValueError(f"Shared code mismatch: {code_path}")
    if _sha256(reward_path) != artifact.get("reward_config_sha256"):
        raise ValueError(f"Shared reward-config mismatch: {reward_path}")

    configs: dict[str, dict[str, Any]] = {}
    results: dict[str, dict[str, float]] = {}
    returns: dict[str, np.ndarray] = {}
    periods = []
    for method in METHODS:
        method_dir = cell / "methods" / method
        summary = _read_json(method_dir / "summary.json")
        comparison = _read_json(method_dir / "final_comparison.json")
        if summary.get("llm_call_attempts") != 0:
            raise ValueError(f"LLM call found in fixed-code replay: {method_dir}")
        if pair.get(f"{method}_llm_call_attempts") != 0:
            raise ValueError(f"Pair reports LLM call: {pair_path}")
        enabled = method == "pbir"
        if summary.get("pbir", {}).get("enabled") is not enabled:
            raise ValueError(f"Wrong PBIR switch: {method_dir}")
        if comparison.get("pbir", {}).get("enabled") is not enabled:
            raise ValueError(f"Wrong PBIR switch: {method_dir}")
        fixed = comparison.get("fixed_shared_artifact", {})
        if (fixed.get("manifest_sha256") != artifact_sha
                or fixed.get("code_sha256") != artifact.get("code_sha256")
                or fixed.get("reward_config_sha256")
                != artifact.get("reward_config_sha256")):
            raise ValueError(f"Method does not share cached artifact: {method_dir}")
        config = yaml.safe_load(
            (method_dir / "resolved_config.yaml").read_text(encoding="utf-8"))
        if not isinstance(config, dict):
            raise TypeError(f"Invalid resolved config: {method_dir}")
        configs[method] = config

        result = comparison.get("test_result", {})
        result_values = {}
        for metric, key in RESULT_KEYS.items():
            if not _finite(result.get(key)):
                raise ValueError(f"Missing/non-finite {key}: {method_dir}")
            result_values[metric] = float(result[key])
        results[method] = result_values
        values = comparison.get("daily_returns", {}).get("method")
        if (not isinstance(values, list) or len(values) < 80
                or not all(_finite(value) for value in values)):
            raise ValueError(f"Invalid daily returns: {method_dir}")
        returns[method] = np.asarray(values, dtype=float)
        periods.append(comparison.get("eval_period"))

    if _without_pbir_switch(configs["pbir"]) != _without_pbir_switch(
            configs["pure_gift"]):
        raise ValueError(f"Configs differ beyond pbir.enabled: {cell}")
    if len(returns["pbir"]) != len(returns["pure_gift"]):
        raise ValueError(f"Unaligned paired returns: {cell}")
    if periods[0] != periods[1]:
        raise ValueError(f"Evaluation-period mismatch: {cell}")
    return {
        "dataset": dataset,
        "window": window,
        "seed": seed,
        "source_stage": source_stage,
        "artifact_sha256": artifact_sha,
        "code_sha256": artifact["code_sha256"],
        "reward_config_sha256": artifact["reward_config_sha256"],
        "results": results,
        "returns": returns,
        "eval_period": periods[0],
    }


def _delta(pair: dict[str, Any], metric: str) -> float:
    pbir = pair["results"]["pbir"][metric]
    gift = pair["results"]["pure_gift"][metric]
    return float(gift - pbir if metric == "max_drawdown_improvement"
                 else pbir - gift)


def _exact_sign_flip(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    observed = float(np.mean(values))
    exceed = 0
    total = 0
    for signs in itertools.product((-1.0, 1.0), repeat=len(values)):
        permuted = float(np.mean(values * np.asarray(signs)))
        exceed += int(permuted >= observed - 1e-15)
        total += 1
    return float(exceed / total)


def _holm(p_values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(p_values, key=p_values.get)
    adjusted: dict[str, float] = {}
    running = 0.0
    count = len(ordered)
    for rank, key in enumerate(ordered):
        value = min(1.0, (count - rank) * p_values[key])
        running = max(running, value)
        adjusted[key] = running
    return adjusted


def _hedges_g_paired(values: np.ndarray) -> float | None:
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return None
    sd = float(np.std(values, ddof=1))
    if sd <= 0:
        return None
    correction = 1.0 - 3.0 / (4.0 * len(values) - 5.0)
    return float(np.mean(values) / sd * correction)


def _circular_indices(
        n: int, block_length: int, rng: np.random.Generator,
) -> np.ndarray:
    starts = rng.integers(0, n, size=math.ceil(n / block_length))
    return np.concatenate([
        (start + np.arange(block_length)) % n for start in starts])[:n]


def _return_delta(
        metric: str, pbir: np.ndarray, gift: np.ndarray,
) -> float:
    statistic = METRICS[metric][0]
    if metric == "max_drawdown_improvement":
        return float(statistic(gift) - statistic(pbir))
    return float(statistic(pbir) - statistic(gift))


def _hierarchical_bootstrap(
        pairs: list[dict[str, Any]], metric: str, resamples: int,
        block_length: int, random_seed: int,
) -> dict[str, Any]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for pair in pairs:
        groups[(pair["dataset"], pair["window"])].append(pair)
    expected = {(dataset, window) for dataset in DATASETS for window in WINDOWS}
    if set(groups) != expected or any(len(items) != 3 for items in groups.values()):
        raise ValueError("Bootstrap requires exactly three seeds in all 12 cells")

    rng = np.random.default_rng(random_seed)
    cell_keys = sorted(groups)
    boot = np.empty(resamples, dtype=float)
    for sample_index in range(resamples):
        sampled_cells = rng.integers(0, len(cell_keys), size=len(cell_keys))
        cell_effects = []
        for cell_index in sampled_cells:
            members = groups[cell_keys[int(cell_index)]]
            seed_effects = []
            for _ in range(len(members)):
                pair = members[int(rng.integers(0, len(members)))]
                pbir = pair["returns"]["pbir"]
                gift = pair["returns"]["pure_gift"]
                indices = _circular_indices(len(pbir), block_length, rng)
                seed_effects.append(_return_delta(
                    metric, pbir[indices], gift[indices]))
            cell_effects.append(float(np.mean(seed_effects)))
        boot[sample_index] = float(np.mean(cell_effects))
    return {
        "resamples": resamples,
        "block_length": block_length,
        "ci95": [float(np.percentile(boot, 2.5)),
                 float(np.percentile(boot, 97.5))],
        "probability_positive": float(np.mean(boot > 0)),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _make_figures(
        cell_rows: list[dict[str, Any]], inference: dict[str, Any], output: Path,
) -> None:
    colors = {"pbir": "#1261A0", "pure_gift": "#D1495B"}
    x = np.arange(len(WINDOWS))
    figure, axes = plt.subplots(1, 2, figsize=(13.0, 5.0), sharey=True)
    for axis, dataset in zip(axes, DATASETS):
        for method in METHODS:
            means = [next(
                row[f"{method}_mean_sharpe"] for row in cell_rows
                if row["dataset"] == dataset and row["window"] == window)
                     for window in WINDOWS]
            sds = [next(
                row[f"{method}_seed_sd_sharpe"] for row in cell_rows
                if row["dataset"] == dataset and row["window"] == window)
                   for window in WINDOWS]
            axis.errorbar(
                x, means, yerr=sds, marker="o", linewidth=2.0, capsize=3,
                label=METHOD_LABELS[method], color=colors[method])
        axis.axhline(0, color="#59636D", linewidth=1.0, alpha=0.75)
        axis.set_xticks(x, WINDOWS)
        axis.set_xlabel("Rolling evaluation window")
        axis.set_title(
            "D1 · portfolio_5stocks" if dataset == DATASETS[0]
            else "D2 · portfolio_5stocks2", fontweight="bold")
        axis.grid(alpha=0.20)
    axes[0].set_ylabel("Out-of-sample Sharpe (mean ± seed SD)")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=2,
                  bbox_to_anchor=(0.5, -0.02), frameon=False)
    figure.suptitle("Six-window fixed-code PBIR controlled replication",
                    fontweight="bold")
    figure.tight_layout(rect=(0, 0.08, 1, 0.95))
    for suffix in ("png", "pdf"):
        figure.savefig(output / f"stage_f_six_window_sharpe.{suffix}",
                       dpi=260, bbox_inches="tight", facecolor="white")
    plt.close(figure)

    ordered = sorted(cell_rows, key=lambda row: (
        DATASETS.index(row["dataset"]), WINDOWS.index(row["window"])))
    labels = [
        f"{'D1' if row['dataset'] == DATASETS[0] else 'D2'}-{row['window']}"
        for row in ordered]
    values = np.asarray([row["delta_sharpe"] for row in ordered])
    errors = np.asarray([row["seed_sd_delta_sharpe"] for row in ordered])
    primary = inference["primary"]
    low, high = primary["bootstrap"]["ci95"]
    overall = primary["mean_cell_delta_sharpe"]
    y = np.arange(len(ordered))
    figure, axis = plt.subplots(figsize=(9.0, 7.0))
    axis.errorbar(values, y, xerr=errors, fmt="o", color="#2F5D8C",
                  ecolor="#9BB3C9", capsize=3,
                  label="cell mean ± seed SD")
    overall_y = len(ordered) + 0.5
    axis.hlines(
        overall_y, low, high, color="#B5653A", linewidth=2.2,
        label="overall hierarchical block-bootstrap 95% CI")
    axis.plot([low, high], [overall_y, overall_y], "|", color="#B5653A",
              markersize=10)
    axis.plot(overall, overall_y, "D", color="#B5653A", markersize=6)
    axis.axvline(0, color="#59636D", linewidth=1.1)
    axis.set_yticks(np.append(y, overall_y), labels + ["Overall"])
    axis.set_xlabel("Δ Sharpe (PBIR-GIFT − Pure GIFT)")
    axis.set_title("Paired PBIR effect by dataset-window cell",
                   loc="left", fontweight="bold")
    axis.grid(axis="x", alpha=0.22)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False, loc="lower right")
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(output / f"stage_f_sharpe_delta_forest.{suffix}",
                       dpi=260, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def main() -> int:
    args = _parse_args()
    if args.resamples < 100:
        raise ValueError("--resamples must be at least 100")
    if args.block_length < 1:
        raise ValueError("--block-length must be positive")
    stage_c = _resolve(args.stage_c_dir)
    stage_f = _resolve(args.stage_f_dir)
    output = _resolve(args.output_dir)
    if not (output == stage_f or stage_f in output.parents):
        raise ValueError("Analysis output must stay inside the Stage-F root")
    if output == stage_c or stage_c in output.parents:
        raise ValueError("Analysis must never write into locked Stage-C")

    stage_c_manifest = _verify_stage_c_source(
        stage_c, _resolve(args.stage_c_lock))
    _validate_grid(stage_c, ODD_WINDOWS, "Stage-C")
    validate_stage_f_cells(stage_f, require_analysis=False)
    _validate_grid(stage_f, EVEN_WINDOWS, "Stage-F")
    output.mkdir(parents=True, exist_ok=True)
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    pairs: list[dict[str, Any]] = []
    for dataset in DATASETS:
        for window in WINDOWS:
            source_root = stage_c if window in ODD_WINDOWS else stage_f
            source_stage = "Stage-C" if window in ODD_WINDOWS else "Stage-F"
            for seed in SEEDS:
                pairs.append(_load_pair(
                    source_root, dataset, window, seed, source_stage))

    seed_rows: list[dict[str, Any]] = []
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for pair in pairs:
        groups[(pair["dataset"], pair["window"])].append(pair)
        for method in METHODS:
            row: dict[str, Any] = {
                "dataset": pair["dataset"],
                "window": pair["window"],
                "seed": pair["seed"],
                "method": method,
                "method_label": METHOD_LABELS[method],
                "source_stage": pair["source_stage"],
                "shared_artifact_sha256": pair["artifact_sha256"],
                "shared_code_sha256": pair["code_sha256"],
            }
            for metric in METRICS:
                row[metric] = pair["results"][method][metric]
            seed_rows.append(row)

    cell_rows: list[dict[str, Any]] = []
    cell_deltas: dict[str, list[float]] = {metric: [] for metric in METRICS}
    for dataset in DATASETS:
        for window in WINDOWS:
            members = groups[(dataset, window)]
            row: dict[str, Any] = {
                "dataset": dataset,
                "window": window,
                "source_stage": "Stage-C" if window in ODD_WINDOWS else "Stage-F",
                "n_seeds": len(members),
                "seeds": ",".join(str(seed) for seed in SEEDS),
            }
            for metric in METRICS:
                deltas = np.asarray([_delta(pair, metric) for pair in members])
                row[f"delta_{metric}"] = float(np.mean(deltas))
                row[f"seed_sd_delta_{metric}"] = float(np.std(deltas, ddof=1))
                cell_deltas[metric].append(float(np.mean(deltas)))
                for method in METHODS:
                    values = np.asarray([
                        pair["results"][method][metric] for pair in members])
                    row[f"{method}_mean_{metric}"] = float(np.mean(values))
                    row[f"{method}_seed_sd_{metric}"] = float(
                        np.std(values, ddof=1))
            cell_rows.append(row)

    raw_p = {
        metric: _exact_sign_flip(np.asarray(values))
        for metric, values in cell_deltas.items()
    }
    adjusted = _holm(raw_p)
    metric_summaries: dict[str, dict[str, Any]] = {}
    contrast_rows: list[dict[str, Any]] = []
    for index, metric in enumerate(METRICS):
        values = np.asarray(cell_deltas[metric], dtype=float)
        bootstrap = _hierarchical_bootstrap(
            pairs, metric, args.resamples, args.block_length,
            args.random_seed + index)
        summary = {
            "direction": METRICS[metric][1],
            "mean_cell_delta": float(np.mean(values)),
            "median_cell_delta": float(np.median(values)),
            "positive_cells": int(np.sum(values > 0)),
            "cell_win_rate": float(np.mean(values > 0)),
            "seed_pair_win_rate": float(np.mean([
                _delta(pair, metric) > 0 for pair in pairs])),
            "hedges_g_paired": _hedges_g_paired(values),
            "exact_one_sided_sign_flip_p": raw_p[metric],
            "holm_adjusted_p": adjusted[metric],
            "bootstrap": bootstrap,
        }
        metric_summaries[metric] = summary
        contrast_rows.append({
            "metric": metric,
            "direction": summary["direction"],
            "mean_cell_delta": summary["mean_cell_delta"],
            "median_cell_delta": summary["median_cell_delta"],
            "positive_cells": summary["positive_cells"],
            "cell_win_rate": summary["cell_win_rate"],
            "seed_pair_win_rate": summary["seed_pair_win_rate"],
            "hedges_g_paired": summary["hedges_g_paired"],
            "exact_one_sided_sign_flip_p": summary[
                "exact_one_sided_sign_flip_p"],
            "holm_adjusted_p": summary["holm_adjusted_p"],
            "bootstrap_ci95_low": bootstrap["ci95"][0],
            "bootstrap_ci95_high": bootstrap["ci95"][1],
            "bootstrap_probability_positive": bootstrap[
                "probability_positive"],
        })

    sharpe_by_cell = {
        (row["dataset"], row["window"]): row["delta_sharpe"]
        for row in cell_rows
    }
    dataset_means = {
        dataset: float(np.mean([
            value for (name, _), value in sharpe_by_cell.items()
            if name == dataset]))
        for dataset in DATASETS
    }
    window_means = {
        window: float(np.mean([
            value for (_, name), value in sharpe_by_cell.items()
            if name == window]))
        for window in WINDOWS
    }
    phase_summary = {}
    for phase, windows in (("odd_stage_c", ODD_WINDOWS),
                           ("even_stage_f", EVEN_WINDOWS)):
        values = np.asarray([
            value for (_, window), value in sharpe_by_cell.items()
            if window in windows])
        phase_summary[phase] = {
            "mean_delta_sharpe": float(np.mean(values)),
            "positive_cells": int(np.sum(values > 0)),
            "cells": len(values),
        }
    primary_metric = metric_summaries["sharpe"]
    primary_bootstrap = primary_metric["bootstrap"]
    worst_mdd_deterioration = float(max(
        0.0, max(-value for value in cell_deltas[
            "max_drawdown_improvement"])))
    strict_supported = bool(
        primary_bootstrap["ci95"][0] > 0
        and all(value > 0 for value in dataset_means.values())
        and primary_metric["positive_cells"] >= 8
        and worst_mdd_deterioration <= 5.0)

    inference = {
        "analysis_role": "stage-f-full-window-fixed-code-pbir-replication",
        "complete_seed_pairs": len(pairs),
        "dataset_window_cells": len(cell_rows),
        "primary_inferential_cell_means": len(cell_rows),
        "seeds_per_cell": len(SEEDS),
        "stage_c_odd_window_pairs": sum(
            pair["source_stage"] == "Stage-C" for pair in pairs),
        "stage_f_even_window_pairs": sum(
            pair["source_stage"] == "Stage-F" for pair in pairs),
        "llm_calls_during_controlled_replay": 0,
        "source_lock_hashes": {
            "stage_c": stage_c_manifest["aggregate_sha256"],
        },
        "primary": {
            "comparison": "PBIR-GIFT versus Pure GIFT",
            "metric": "out-of-sample Sharpe",
            "mean_cell_delta_sharpe": primary_metric["mean_cell_delta"],
            "positive_cells": primary_metric["positive_cells"],
            "dataset_means": dataset_means,
            "window_means": window_means,
            "bootstrap": primary_bootstrap,
            "worst_cell_mdd_deterioration_percentage_points": (
                worst_mdd_deterioration),
            "strict_full_window_supported": strict_supported,
            "strict_rule": (
                "Sharpe hierarchical block-bootstrap 95% CI lower bound > 0; "
                "both dataset mean Sharpe deltas > 0; at least 8/12 cell means "
                "positive; worst cell MDD deterioration <= 5 percentage points."),
        },
        "odd_even_generalization": phase_summary,
        "metrics": metric_summaries,
        "caveats": [
            "W2/W4/W6 overlap adjacent rolling windows and are robustness checks, "
            "not six additional independent market regimes.",
            "D1 and D2 share AMZN, MSFT, and JNJ, so the two portfolios are not "
            "fully independent universes.",
            "The cached code was selected by the Stage-B PBIR branch; this design "
            "isolates shaping semantics but does not estimate a fresh-code LLM "
            "generation effect.",
            "Failure to pass the strict gate is not evidence of equivalence or of "
            "a software failure.",
        ],
    }
    (output / "stage_f_inference.json").write_text(
        json.dumps(inference, indent=2), encoding="utf-8")
    _write_csv(output / "stage_f_seed_metrics.csv", seed_rows)
    _write_csv(output / "stage_f_cell_metrics.csv", cell_rows)
    _write_csv(output / "stage_f_metric_contrasts.csv", contrast_rows)
    _make_figures(cell_rows, inference, figures)

    ci = primary_bootstrap["ci95"]
    decision = "SUPPORTED" if strict_supported else "NOT SUPPORTED"
    report = [
        "# Stage-F full-window fixed-code PBIR replication",
        "",
        "## Primary decision",
        "",
        f"The predeclared strict PBIR superiority gate is **{decision}**.",
        "",
        f"Across 12 dataset-window cell means, PBIR-GIFT minus Pure GIFT "
        f"Sharpe was {primary_metric['mean_cell_delta']:+.3f}; "
        f"{primary_metric['positive_cells']}/12 cells favored PBIR-GIFT. "
        f"The hierarchical circular-block bootstrap 95% interval was "
        f"[{ci[0]:+.3f}, {ci[1]:+.3f}], with positive probability "
        f"{primary_bootstrap['probability_positive']:.1%}. Paired Hedges g was "
        f"{primary_metric['hedges_g_paired'] if primary_metric['hedges_g_paired'] is not None else 'NA'}.",
        "",
        "The decision rule requires a positive bootstrap lower bound, positive "
        "means in both datasets, at least 8/12 positive cells, and no cell with "
        "more than 5 percentage points of MDD deterioration.",
        "",
        "## Odd-to-even generalization",
        "",
        f"- Locked Stage-C odd windows: mean ΔSharpe "
        f"{phase_summary['odd_stage_c']['mean_delta_sharpe']:+.3f}; "
        f"{phase_summary['odd_stage_c']['positive_cells']}/6 positive cells.",
        f"- New Stage-F even windows: mean ΔSharpe "
        f"{phase_summary['even_stage_f']['mean_delta_sharpe']:+.3f}; "
        f"{phase_summary['even_stage_f']['positive_cells']}/6 positive cells.",
        "",
        "## Metric contrasts",
        "",
        "| Metric | Mean delta | Positive cells | Hedges g | Exact p | Holm p | 95% block-bootstrap CI |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for metric, summary in metric_summaries.items():
        metric_ci = summary["bootstrap"]["ci95"]
        effect = summary["hedges_g_paired"]
        report.append(
            f"| {metric} | {summary['mean_cell_delta']:+.3f} | "
            f"{summary['positive_cells']}/12 | "
            f"{'NA' if effect is None else f'{effect:+.3f}'} | "
            f"{summary['exact_one_sided_sign_flip_p']:.4f} | "
            f"{summary['holm_adjusted_p']:.4f} | "
            f"[{metric_ci[0]:+.3f}, {metric_ci[1]:+.3f}] |")
    report.extend([
        "",
        "## Claim boundary",
        "",
        "- This is the core PBIR ablation: both methods receive the exact same "
        "cached LLM code and reward rules; only potential-based shaping is toggled.",
        "- A supported result permits a claim of robust benefit under these two "
        "portfolios and rolling windows, not universal superiority in all markets.",
        "- A non-supported result should be reported as mixed or regime-dependent; "
        "it must not be rewritten as equivalence or hidden.",
        "- Stage-C is verified against its cryptographic lock and is never modified "
        "by this analysis. All new files live in Stage-F.",
    ])
    report_text = "\n".join(report) + "\n"
    (output / "analysis_report.md").write_text(report_text, encoding="utf-8")
    (output / "stage_f_analysis.md").write_text(report_text, encoding="utf-8")

    appendix = [
        "# Stage-F statistical appendix",
        "",
        "The primary unit is the dataset-window cell mean: 12 cells with three "
        "paired seeds each. The exact one-sided sign-flip test operates on these "
        "12 cell-mean deltas. Holm correction covers the five reported outcomes.",
        "",
        "The hierarchical bootstrap resamples cells, seeds within cells, and then "
        "aligned daily returns using circular blocks. It therefore preserves "
        "within-series dependence more appropriately than a day-level t-test. "
        f"This run uses {args.resamples:,} replicates and block length "
        f"{args.block_length}.",
        "",
        "W2/W4/W6 overlap the odd-window schedule. The all-window analysis is a "
        "full robustness map; the odd-versus-even summary is reported explicitly "
        "so generalization beyond the original Stage-C windows remains visible.",
    ]
    (output / "stats_appendix.md").write_text(
        "\n".join(appendix) + "\n", encoding="utf-8")
    catalog = [
        "# Stage-F figure catalog",
        "",
        "## Figure 1 — Six-window Sharpe trajectories",
        "",
        "`figures/stage_f_six_window_sharpe.png` and `.pdf` show PBIR-GIFT "
        "and Pure GIFT in both datasets. Points are three-seed means; error bars "
        "are seed standard deviations.",
        "",
        "## Figure 2 — Paired Sharpe-effect forest",
        "",
        "`figures/stage_f_sharpe_delta_forest.png` and `.pdf` show each of the "
        "12 cell-mean PBIR effects with seed SD, plus the overall hierarchical "
        "block-bootstrap 95% interval.",
    ]
    (output / "figure_catalog.md").write_text(
        "\n".join(catalog) + "\n", encoding="utf-8")

    print(f"Analysis: {output / 'stage_f_analysis.md'}")
    print(f"Figure: {figures / 'stage_f_six_window_sharpe.png'}")
    print(f"Complete paired seed runs: {len(pairs)} (Stage-C 18 + Stage-F 18)")
    print(f"Strict PBIR full-window supported: {strict_supported}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
