"""Analyze GIFT/PBIR against all six pre-specified traditional baselines."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

from analyze_final_results import _exact_sign_flip, _hedges_g_paired
from analyze_stage_e_full_window_reproduction import _bootstrap_sharpe
from lock_stage_b_results import verify_lock as verify_stage_b_lock
from supplemental_experiment_utils import (
    DATASETS,
    METRICS,
    SEEDS,
    WINDOWS,
    holm_adjust,
    read_json,
    resolve_path,
    validate_result_record,
    write_csv,
    write_json,
)
from traditional_baselines import DETERMINISTIC_METHODS, METHODS


ANCHORS = ("pure_gift", "pbir")
METHOD_LABELS = {
    "pbir": "PBIR-GIFT",
    "pure_gift": "Pure GIFT",
    "sma": "SMA",
    "wma": "WMA",
    "atr": "ATR",
    "bollinger": "Bollinger Bands",
    "turn_of_month": "Turn-of-the-Month",
    "xgboost": "XGBoost",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-b-dir", default="results/stage_b_final")
    parser.add_argument(
        "--stage-b-lock", default="results/stage_b_lock/stage_b_lock_manifest.json")
    parser.add_argument("--stage-h-dir", default="results/stage_h_traditional_baselines")
    parser.add_argument(
        "--strategy-config", default="config_traditional_baselines.yaml")
    parser.add_argument(
        "--output-dir",
        default="results/stage_h_traditional_baselines/analysis_bundle")
    parser.add_argument("--resamples", type=int, default=10000)
    parser.add_argument("--block-length", type=int, default=20)
    return parser.parse_args()


def _stage_b_method(
    root: Path, dataset: str, window: str, seed: int, method: str,
) -> tuple[dict[str, Any], np.ndarray, list[str]]:
    path = root / dataset / window / f"seed_{seed}" / "methods" / method / "final_comparison.json"
    value = read_json(path)
    result = value.get("test_result", {})
    returns = value.get("daily_returns", {}).get("method")
    period = value.get("eval_period")
    if not isinstance(returns, list) or not isinstance(period, list):
        raise ValueError(f"Invalid Stage-B method output: {path}")
    for metric in METRICS:
        if not isinstance(result.get(metric), (int, float)):
            raise ValueError(f"Missing {metric}: {path}")
    return result, np.asarray(returns, dtype=float), [str(item) for item in period]


def _traditional_records(
    root: Path, dataset: str, window: str, method: str,
) -> list[dict[str, Any]]:
    cell = root / dataset / window / "methods"
    paths = (
        [cell / method / f"seed_{seed}.json" for seed in SEEDS]
        if method == "xgboost" else [cell / f"{method}.json"])
    return [validate_result_record(path, method) for path in paths]


def _validate_grid(root: Path) -> None:
    for dataset in DATASETS:
        for window in WINDOWS:
            summary = root / dataset / window / "baseline_summary.json"
            value = read_json(summary)
            if value.get("status") != "completed":
                raise ValueError(f"Incomplete Stage-H cell: {summary}")
            for method in METHODS:
                _traditional_records(root, dataset, window, method)


def _delta(anchor: dict[str, Any], baseline: dict[str, Any], metric: str) -> float:
    if metric == "test_max_drawdown":
        return float(baseline[metric]) - float(anchor[metric])
    return float(anchor[metric]) - float(baseline[metric])


def _make_figures(
    output: Path,
    cell_values: dict[tuple[str, str, str, str], float],
) -> None:
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    methods = list(METHOD_LABELS)
    cells = [(dataset, window) for dataset in DATASETS for window in WINDOWS]
    matrix = np.asarray([
        [cell_values[(dataset, window, method, "test_sharpe")] for method in methods]
        for dataset, window in cells
    ])
    figure, axis = plt.subplots(figsize=(12.4, 7.0))
    image = axis.imshow(matrix, aspect="auto", cmap="RdYlBu", interpolation="nearest")
    axis.set_xticks(np.arange(len(methods)), [METHOD_LABELS[m] for m in methods],
                    rotation=35, ha="right")
    axis.set_yticks(np.arange(len(cells)), [
        f"{'D1' if dataset == DATASETS[0] else 'D2'}-{window}"
        for dataset, window in cells])
    axis.set_title("Out-of-sample Sharpe across methods and cells", fontweight="bold")
    colorbar = figure.colorbar(image, ax=axis, shrink=0.85)
    colorbar.set_label("Sharpe ratio")
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(figures / f"stage_h_sharpe_heatmap.{suffix}",
                       dpi=240, bbox_inches="tight")
    plt.close(figure)

    rng = np.random.default_rng(20260904)
    means, lower, upper = [], [], []
    for method in methods:
        values = np.asarray([
            cell_values[(dataset, window, method, "test_sharpe")]
            for dataset, window in cells])
        boot = np.asarray([
            np.mean(rng.choice(values, size=len(values), replace=True))
            for _ in range(5000)])
        means.append(float(np.mean(values)))
        lower.append(max(0.0, float(np.mean(values) - np.percentile(boot, 2.5))))
        upper.append(max(0.0, float(np.percentile(boot, 97.5) - np.mean(values))))
    x = np.arange(len(methods))
    figure, axis = plt.subplots(figsize=(12.4, 5.4))
    axis.bar(x, means, color=["#6F2DBD", "#1261A0"] + ["#6C8EAD"] * len(METHODS),
             alpha=0.9)
    axis.errorbar(x, means, yerr=np.asarray([lower, upper]), fmt="none",
                  ecolor="#27313A", capsize=4)
    for index, method in enumerate(methods):
        cell = np.asarray([
            cell_values[(dataset, window, method, "test_sharpe")]
            for dataset, window in cells])
        jitter = rng.normal(0, 0.045, size=len(cell))
        axis.scatter(np.full(len(cell), index) + jitter, cell, s=18,
                     color="#1F2933", alpha=0.55)
    axis.axhline(0, color="#5E6872", linewidth=1)
    axis.set_xticks(x, [METHOD_LABELS[m] for m in methods], rotation=30, ha="right")
    axis.set_ylabel("Mean cell Sharpe; bars show cell-bootstrap 95% CI")
    axis.set_title("Cross-window Sharpe comparison", fontweight="bold")
    axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(figures / f"stage_h_method_mean_sharpe.{suffix}",
                       dpi=240, bbox_inches="tight")
    plt.close(figure)


def main() -> int:
    args = _parse_args()
    if args.resamples < 100 or args.block_length < 1:
        raise ValueError("Need at least 100 resamples and positive block length")
    stage_b = resolve_path(args.stage_b_dir)
    stage_h = resolve_path(args.stage_h_dir)
    output = resolve_path(args.output_dir)
    if not (output == stage_h or stage_h in output.parents):
        raise ValueError("Stage-H analysis must stay inside Stage-H")
    lock_path = resolve_path(args.stage_b_lock)
    verify_stage_b_lock(lock_path)
    b_lock = read_json(lock_path)
    if Path(str(b_lock["source_root"])).resolve() != stage_b:
        raise ValueError("Stage-B lock points to another root")
    _validate_grid(stage_h)
    frozen_strategy = stage_h / "frozen_strategy_config.yaml"
    if not frozen_strategy.is_file():
        raise FileNotFoundError(frozen_strategy)
    strategy = yaml.safe_load(frozen_strategy.read_text(encoding="utf-8")) or {}
    requested_strategy = yaml.safe_load(
        resolve_path(args.strategy_config).read_text(encoding="utf-8")) or {}
    if strategy != requested_strategy:
        raise ValueError(
            "Current Stage-H strategy config differs from the frozen run protocol")
    declared_primary = list(
        strategy.get("protocol", {}).get("declared_primary_representatives", []))
    output.mkdir(parents=True, exist_ok=True)

    seed_rows: list[dict[str, Any]] = []
    raw_values: dict[tuple[str, str, str, str], list[float]] = defaultdict(list)
    anchor_daily: dict[tuple[str, str, str], list[np.ndarray]] = defaultdict(list)
    baseline_daily: dict[tuple[str, str, str], list[np.ndarray]] = defaultdict(list)
    periods: dict[tuple[str, str], list[str]] = {}

    for dataset in DATASETS:
        for window in WINDOWS:
            for anchor in ANCHORS:
                for seed in SEEDS:
                    result, returns, period = _stage_b_method(
                        stage_b, dataset, window, seed, anchor)
                    periods[(dataset, window)] = period
                    anchor_daily[(dataset, window, anchor)].append(returns)
                    row = {"dataset": dataset, "window": window, "method": anchor,
                           "seed": seed, "method_label": METHOD_LABELS[anchor]}
                    for metric in METRICS:
                        value = float(result[metric])
                        row[metric] = value
                        raw_values[(dataset, window, anchor, metric)].append(value)
                    seed_rows.append(row)
            for method in METHODS:
                records = _traditional_records(stage_h, dataset, window, method)
                for record in records:
                    if list(record["eval_period"]) != periods[(dataset, window)]:
                        raise ValueError(f"Stage-H period mismatch: {dataset}/{window}/{method}")
                    result = record["test_result"]
                    baseline_daily[(dataset, window, method)].append(
                        np.asarray(record["daily_returns"], dtype=float))
                    row = {
                        "dataset": dataset, "window": window, "method": method,
                        "seed": record.get("seed") if method == "xgboost" else "deterministic",
                        "method_label": METHOD_LABELS[method],
                    }
                    for metric in METRICS:
                        value = float(result[metric])
                        row[metric] = value
                        raw_values[(dataset, window, method, metric)].append(value)
                    seed_rows.append(row)

    cell_values = {
        key: float(np.mean(values)) for key, values in raw_values.items()
    }
    cell_rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        for window in WINDOWS:
            for method in METHOD_LABELS:
                values = raw_values[(dataset, window, method, "test_sharpe")]
                row: dict[str, Any] = {
                    "dataset": dataset, "window": window, "method": method,
                    "method_label": METHOD_LABELS[method], "n_runs": len(values),
                    "deterministic": method in DETERMINISTIC_METHODS,
                }
                for metric in METRICS:
                    array = np.asarray(raw_values[(dataset, window, method, metric)])
                    row[f"mean_{metric}"] = float(np.mean(array))
                    row[f"run_sd_{metric}"] = (
                        float(np.std(array, ddof=1)) if len(array) > 1 else 0.0)
                cell_rows.append(row)

    contrast_rows: list[dict[str, Any]] = []
    inference: dict[str, dict[str, Any]] = {}
    for anchor_index, anchor in enumerate(ANCHORS):
        p_values = {}
        pending: dict[str, dict[str, Any]] = {}
        for method_index, method in enumerate(METHODS):
            deltas = []
            pairs: dict[tuple[str, str], list[tuple[np.ndarray, np.ndarray]]] = (
                defaultdict(list))
            dataset_means: dict[str, list[float]] = defaultdict(list)
            for dataset in DATASETS:
                for window in WINDOWS:
                    anchor_sharpe = cell_values[(dataset, window, anchor, "test_sharpe")]
                    baseline_sharpe = cell_values[(dataset, window, method, "test_sharpe")]
                    delta = anchor_sharpe - baseline_sharpe
                    deltas.append(delta)
                    dataset_means[dataset].append(delta)
                    anchors = anchor_daily[(dataset, window, anchor)]
                    baselines = baseline_daily[(dataset, window, method)]
                    if len(baselines) == 1:
                        pairs[(dataset, window)] = [
                            (values, baselines[0]) for values in anchors]
                    else:
                        pairs[(dataset, window)] = list(zip(anchors, baselines))
            array = np.asarray(deltas, dtype=float)
            p_value = _exact_sign_flip(array)
            p_values[method] = p_value
            bootstrap = _bootstrap_sharpe(
                pairs, args.resamples, args.block_length,
                20261000 + anchor_index * 100 + method_index)
            summary = {
                "comparison": f"{METHOD_LABELS[anchor]} versus {METHOD_LABELS[method]}",
                "mean_cell_delta_sharpe": float(np.mean(array)),
                "median_cell_delta_sharpe": float(np.median(array)),
                "positive_cells": int(np.sum(array > 0)),
                "cell_win_rate": float(np.mean(array > 0)),
                "hedges_g_paired": _hedges_g_paired(array),
                "exact_one_sided_sign_flip_p": p_value,
                "dataset_means": {
                    key: float(np.mean(values)) for key, values in dataset_means.items()},
                "bootstrap": bootstrap,
            }
            summary["strict_supported"] = bool(
                bootstrap["ci95"][0] > 0
                and all(value > 0 for value in summary["dataset_means"].values())
                and summary["positive_cells"] >= 8)
            pending[method] = summary
        adjusted = holm_adjust(p_values)
        for method in METHODS:
            pending[method]["holm_adjusted_p_across_six_baselines"] = adjusted[method]
            key = f"{anchor}_vs_{method}"
            inference[key] = pending[method]
            contrast_rows.append({"anchor": anchor, "baseline": method,
                                  **pending[method]})

    bundle = {
        "analysis_role": "stage-h-full-window-traditional-baselines",
        "dataset_window_cells": 12,
        "gift_seeds_per_cell": 3,
        "xgboost_seeds_per_cell": 3,
        "deterministic_baseline_runs_per_cell": 1,
        "declared_primary_representatives": declared_primary,
        "all_methods_reported": list(METHODS),
        "contrasts": inference,
        "source_lock_hashes": {"stage_b": b_lock["aggregate_sha256"]},
        "caveats": [
            "Traditional rule parameters are canonical pre-specified values, not test-tuned optima.",
            "Deterministic strategies contribute one result per dataset-window cell.",
            "D1 and D2 share three stocks and rolling windows overlap.",
            "Holm adjustment is applied across all six baselines separately for each GIFT anchor.",
        ],
    }
    write_json(output / "stage_h_inference.json", bundle)
    write_csv(output / "stage_h_run_metrics.csv", seed_rows)
    write_csv(output / "stage_h_cell_metrics.csv", cell_rows)
    write_csv(output / "stage_h_contrasts.csv", contrast_rows)
    _make_figures(output, cell_values)

    report = [
        "# Stage-H traditional baseline analysis", "",
        "## Scope", "",
        "All six pre-specified traditional/ML baselines are reported; no method "
        "is removed according to its observed result.", "",
        "## Pre-declared main-text representatives", "",
        "- SMA: trend-following rule.",
        "- Bollinger Bands: volatility-band breakout rule.",
        "- XGBoost: supervised machine-learning forecaster.", "",
        "## Sharpe contrast decisions", "",
    ]
    for anchor in ANCHORS:
        for method in METHODS:
            item = inference[f"{anchor}_vs_{method}"]
            ci = item["bootstrap"]["ci95"]
            report.append(
                f"- {METHOD_LABELS[anchor]} − {METHOD_LABELS[method]}: "
                f"mean {item['mean_cell_delta_sharpe']:+.3f}, "
                f"{item['positive_cells']}/12 positive cells, 95% block-bootstrap "
                f"CI [{ci[0]:+.3f}, {ci[1]:+.3f}], strict="
                f"{item['strict_supported']}.")
    report.extend([
        "", "## Claim candidates", "",
        "- Allowed: method-specific effects with cell counts, intervals, and corrected tests.",
        "- Forbidden: selecting only baselines that GIFT beats or claiming universal dominance.",
    ])
    (output / "analysis_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    (output / "stage_h_analysis.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    stats = [
        "# Stage-H statistical appendix", "",
        "- Inferential unit: dataset-window cell (12 cells).",
        "- Seed means are collapsed before the exact directional sign-flip test.",
        "- Deterministic baselines are not duplicated as independent seeds.",
        "- Holm adjustment controls six baseline contrasts within each anchor method.",
        "- Circular block bootstrap preserves short-range daily-return dependence.",
    ]
    (output / "stats_appendix.md").write_text("\n".join(stats) + "\n", encoding="utf-8")
    catalog = [
        "# Stage-H figure catalog", "",
        "## stage_h_sharpe_heatmap", "",
        "- Purpose: expose method-by-window heterogeneity for every reported method.",
        "- Notice: no baseline is hidden when it outperforms GIFT.",
        "", "## stage_h_method_mean_sharpe", "",
        "- Purpose: summarize cross-cell mean Sharpe while retaining individual cell points.",
        "- Error bars: percentile bootstrap over 12 cell means; not seed-level intervals.",
    ]
    (output / "figure_catalog.md").write_text("\n".join(catalog) + "\n", encoding="utf-8")
    print(f"Analysis: {output / 'stage_h_analysis.md'}")
    print(f"Figure: {output / 'figures' / 'stage_h_sharpe_heatmap.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
