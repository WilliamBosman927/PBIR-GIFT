"""Analyze Pure GIFT against an independently tuned Pure-PPO baseline."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

from analyze_final_results import _exact_sign_flip, _hedges_g_paired
from analyze_stage_e_full_window_reproduction import (
    _bootstrap_sharpe,
    _gift_result,
)
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


METHOD_LABELS = {"pure_gift": "Pure GIFT", "ppo_tuned": "PPO-Tuned"}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-b-dir", default="results/stage_b_final")
    parser.add_argument(
        "--stage-b-lock", default="results/stage_b_lock/stage_b_lock_manifest.json")
    parser.add_argument("--stage-g-dir", default="results/stage_g_tuned_ppo")
    parser.add_argument(
        "--output-dir", default="results/stage_g_tuned_ppo/analysis_bundle")
    parser.add_argument("--resamples", type=int, default=10000)
    parser.add_argument("--block-length", type=int, default=20)
    return parser.parse_args()


def _validate_grid(root: Path) -> None:
    expected = {
        (dataset, window, seed)
        for dataset in DATASETS for window in WINDOWS for seed in SEEDS
    }
    observed = set()
    for path in root.glob("evaluation/*/W*/seed_*/methods/ppo_tuned.json"):
        record = validate_result_record(path, "ppo_tuned")
        observed.add((
            str(record.get("dataset")),
            str(record.get("window")),
            int(record.get("seed")),
        ))
    if observed != expected:
        raise ValueError(
            "Stage-G final grid is incomplete or contaminated: "
            f"missing={sorted(expected - observed)}, "
            f"unexpected={sorted(observed - expected)}")


def _metric_delta(gift: dict[str, Any], ppo: dict[str, Any], metric: str) -> float:
    if metric == "test_max_drawdown":
        return float(ppo[metric]) - float(gift[metric])
    return float(gift[metric]) - float(ppo[metric])


def _make_figures(
    output: Path,
    values: dict[tuple[str, str, str, str], list[float]],
    cell_rows: list[dict[str, Any]],
) -> list[Path]:
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    colors = {"pure_gift": "#1261A0", "ppo_tuned": "#D1495B"}
    created: list[Path] = []
    x = np.arange(len(WINDOWS))
    figure, axes = plt.subplots(1, 2, figsize=(12.4, 4.9), sharey=True)
    for axis, dataset in zip(axes, DATASETS):
        for method in METHOD_LABELS:
            means = [
                np.mean(values[(dataset, window, method, "test_sharpe")])
                for window in WINDOWS
            ]
            sds = [
                np.std(values[(dataset, window, method, "test_sharpe")], ddof=1)
                for window in WINDOWS
            ]
            axis.errorbar(
                x, means, yerr=sds, marker="o", linewidth=2.1, capsize=3,
                color=colors[method], label=METHOD_LABELS[method])
        axis.axhline(0, color="#5E6872", linewidth=1)
        axis.set_xticks(x, WINDOWS)
        axis.set_title("D1 Light Mix" if dataset == DATASETS[0]
                       else "D2 Custom Portfolio", fontweight="bold")
        axis.set_xlabel("Rolling evaluation window")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Out-of-sample Sharpe (mean ± seed SD)")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=2,
                  bbox_to_anchor=(0.5, -0.01), frameon=False)
    figure.suptitle("Pure GIFT versus independently tuned PPO", fontweight="bold")
    figure.tight_layout(rect=(0, 0.07, 1, 0.95))
    for suffix in ("png", "pdf"):
        path = figures / f"stage_g_six_window_sharpe.{suffix}"
        figure.savefig(path, dpi=240, bbox_inches="tight")
        created.append(path)
    plt.close(figure)

    ordered = sorted(cell_rows, key=lambda row: (row["dataset"], row["window"]))
    labels = [
        f"{'D1' if row['dataset'] == DATASETS[0] else 'D2'}-{row['window']}"
        for row in ordered
    ]
    deltas = np.asarray([row["delta_sharpe"] for row in ordered], dtype=float)
    errors = np.asarray([row["seed_sd_delta_sharpe"] for row in ordered], dtype=float)
    y = np.arange(len(ordered))
    figure, axis = plt.subplots(figsize=(8.6, 6.2))
    axis.errorbar(deltas, y, xerr=errors, fmt="o", color="#1261A0",
                  ecolor="#7EA6C9", capsize=3)
    axis.axvline(0, color="#5E6872", linewidth=1.2)
    axis.set_yticks(y, labels)
    axis.invert_yaxis()
    axis.set_xlabel("ΔSharpe (Pure GIFT − PPO-Tuned); error = seed SD")
    axis.set_title("Dataset-window effects", fontweight="bold")
    axis.grid(axis="x", alpha=0.2)
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        path = figures / f"stage_g_sharpe_delta_forest.{suffix}"
        figure.savefig(path, dpi=240, bbox_inches="tight")
        created.append(path)
    plt.close(figure)
    return created


def main() -> int:
    args = _parse_args()
    if args.resamples < 100:
        raise ValueError("--resamples must be at least 100")
    if args.block_length < 1:
        raise ValueError("--block-length must be positive")
    stage_b = resolve_path(args.stage_b_dir)
    stage_g = resolve_path(args.stage_g_dir)
    output = resolve_path(args.output_dir)
    if not (output == stage_g or stage_g in output.parents):
        raise ValueError("Stage-G analysis must stay inside the Stage-G result root")
    lock_path = resolve_path(args.stage_b_lock)
    verify_stage_b_lock(lock_path)
    b_lock = read_json(lock_path)
    if Path(str(b_lock["source_root"])).resolve() != stage_b:
        raise ValueError("Stage-B lock points to a different result root")
    _validate_grid(stage_g)
    output.mkdir(parents=True, exist_ok=True)

    metric_rows: list[dict[str, Any]] = []
    values: dict[tuple[str, str, str, str], list[float]] = defaultdict(list)
    delta_values: dict[str, dict[tuple[str, str], list[float]]] = {
        metric: defaultdict(list) for metric in METRICS
    }
    return_pairs: dict[tuple[str, str], list[tuple[np.ndarray, np.ndarray]]] = (
        defaultdict(list))
    for dataset in DATASETS:
        for window in WINDOWS:
            for seed in SEEDS:
                gift_cell = stage_b / dataset / window / f"seed_{seed}"
                gift_result, gift_returns, gift_period = _gift_result(gift_cell)
                ppo_path = (
                    stage_g / "evaluation" / dataset / window / f"seed_{seed}"
                    / "methods" / "ppo_tuned.json")
                ppo_record = validate_result_record(ppo_path, "ppo_tuned")
                ppo_result = ppo_record["test_result"]
                ppo_returns = np.asarray(ppo_record["daily_returns"], dtype=float)
                if list(ppo_record["eval_period"]) != gift_period:
                    raise ValueError(f"Evaluation period mismatch: {dataset}/{window}/{seed}")
                if len(ppo_returns) != len(gift_returns):
                    raise ValueError(f"Daily-return length mismatch: {dataset}/{window}/{seed}")
                return_pairs[(dataset, window)].append((gift_returns, ppo_returns))
                for method, result in (("pure_gift", gift_result),
                                       ("ppo_tuned", ppo_result)):
                    row = {
                        "dataset": dataset, "window": window, "seed": seed,
                        "method": method, "method_label": METHOD_LABELS[method],
                    }
                    for metric in METRICS:
                        value = float(result[metric])
                        row[metric] = value
                        values[(dataset, window, method, metric)].append(value)
                    metric_rows.append(row)
                for metric in METRICS:
                    delta_values[metric][(dataset, window)].append(
                        _metric_delta(gift_result, ppo_result, metric))

    cell_rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        for window in WINDOWS:
            row: dict[str, Any] = {"dataset": dataset, "window": window,
                                   "n_seeds": len(SEEDS)}
            for method in METHOD_LABELS:
                for metric in METRICS:
                    array = np.asarray(values[(dataset, window, method, metric)])
                    row[f"{method}_mean_{metric}"] = float(np.mean(array))
                    row[f"{method}_seed_sd_{metric}"] = float(np.std(array, ddof=1))
            for metric in METRICS:
                array = np.asarray(delta_values[metric][(dataset, window)])
                name = "max_drawdown_improvement" if metric == "test_max_drawdown" else metric.removeprefix("test_")
                row[f"delta_{name}"] = float(np.mean(array))
                row[f"seed_sd_delta_{name}"] = float(np.std(array, ddof=1))
            cell_rows.append(row)

    metric_inference: dict[str, Any] = {}
    raw_p: dict[str, float] = {}
    for metric in METRICS:
        cell = np.asarray([
            np.mean(delta_values[metric][(dataset, window)])
            for dataset in DATASETS for window in WINDOWS
        ])
        key = "max_drawdown_improvement" if metric == "test_max_drawdown" else metric.removeprefix("test_")
        p_value = _exact_sign_flip(cell)
        raw_p[key] = p_value
        metric_inference[key] = {
            "direction": (
                "PPO-Tuned − Pure GIFT" if metric == "test_max_drawdown"
                else "Pure GIFT − PPO-Tuned"),
            "mean_cell_delta": float(np.mean(cell)),
            "median_cell_delta": float(np.median(cell)),
            "positive_cells": int(np.sum(cell > 0)),
            "cell_win_rate": float(np.mean(cell > 0)),
            "seed_run_win_rate": float(np.mean([
                value > 0 for members in delta_values[metric].values()
                for value in members])),
            "hedges_g_paired": _hedges_g_paired(cell),
            "exact_one_sided_sign_flip_p": p_value,
        }
    adjusted = holm_adjust(raw_p)
    for name, value in adjusted.items():
        metric_inference[name]["holm_adjusted_p"] = value

    sharpe_cells = {
        key: float(np.mean(members))
        for key, members in delta_values["test_sharpe"].items()
    }
    sharpe_bootstrap = _bootstrap_sharpe(
        return_pairs, args.resamples, args.block_length, 20260904)
    sharpe = metric_inference["sharpe"]
    sharpe["bootstrap"] = sharpe_bootstrap
    sharpe["dataset_means"] = {
        dataset: float(np.mean([
            value for (seen, _), value in sharpe_cells.items() if seen == dataset]))
        for dataset in DATASETS
    }
    strict = bool(
        sharpe_bootstrap["ci95"][0] > 0
        and all(value > 0 for value in sharpe["dataset_means"].values())
        and sharpe["positive_cells"] >= 8
    )
    selection = yaml.safe_load(
        (stage_g / "selection" / "selected_ppo_tuned.yaml").read_text(
            encoding="utf-8")) or {}
    inference = {
        "analysis_role": "stage-g-independently-tuned-ppo-fairness-check",
        "complete_seed_runs": len(DATASETS) * len(WINDOWS) * len(SEEDS),
        "dataset_window_cells": len(DATASETS) * len(WINDOWS),
        "seeds_per_cell": len(SEEDS),
        "primary_comparison": "Pure GIFT versus independently tuned PPO-TestOnly",
        "primary_metric": "out-of-sample Sharpe",
        "strict_supported": strict,
        "strict_rule": (
            "Sharpe hierarchical block-bootstrap 95% CI lower bound > 0; "
            "both dataset mean deltas > 0; at least 8/12 positive cells."),
        "selected_ppo": selection.get("ppo", {}),
        "selection_protocol": selection.get("protocol", {}),
        "metrics": metric_inference,
        "source_lock_hashes": {"stage_b": b_lock["aggregate_sha256"]},
        "caveats": [
            "Stage-G is a prospective supplemental fairness analysis added after Stage-F.",
            "D1 and D2 share three stocks and are not independent universes.",
            "Dataset-window cell means, not daily observations, are inferential units.",
            "Sequential OFAT may miss hyperparameter interactions.",
        ],
    }
    write_json(output / "stage_g_inference.json", inference)
    write_csv(output / "stage_g_seed_metrics.csv", metric_rows)
    write_csv(output / "stage_g_cell_metrics.csv", cell_rows)
    _make_figures(output, values, cell_rows)

    ci = sharpe_bootstrap["ci95"]
    decision = "SUPPORTED" if strict else "NOT SUPPORTED"
    report = [
        "# Stage-G independently tuned PPO analysis",
        "",
        "## Decision",
        "",
        f"- Strict Pure-GIFT superiority over PPO-Tuned: **{decision}**.",
        "",
        "## Primary contrast",
        "",
        f"Across 12 dataset-window cell means, ΔSharpe was "
        f"{sharpe['mean_cell_delta']:+.3f}; {sharpe['positive_cells']}/12 cells "
        f"favored Pure GIFT. The hierarchical block-bootstrap 95% interval "
        f"was [{ci[0]:+.3f}, {ci[1]:+.3f}].",
        "",
        "## Claim candidates",
        "",
        "- Claim: Pure GIFT outperforms an independently tuned PPO baseline.",
        f"  - Decision: {'keep' if strict else 'weaken'}.",
        "  - Allowed wording: Report the observed mean effect, interval, and cell wins.",
        "  - Forbidden wording: GIFT universally dominates optimally tuned PPO.",
        "  - Uncertainty: two overlapping stock panels and rolling-window dependence.",
    ]
    (output / "analysis_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    (output / "stage_g_analysis.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    stats = [
        "# Stage-G statistical appendix", "",
        "- Inferential unit: dataset-window mean across three seeds (12 cells).",
        "- Directional exact sign-flip tests are reported for all five metrics.",
        "- Holm correction controls the five-metric family-wise error rate.",
        "- Sharpe uncertainty additionally uses hierarchical circular block bootstrap.",
        "- Daily observations are resampling units within cells, not independent tests.",
    ]
    (output / "stats_appendix.md").write_text("\n".join(stats) + "\n", encoding="utf-8")
    catalog = [
        "# Stage-G figure catalog", "",
        "## stage_g_six_window_sharpe", "",
        "- Purpose: compare Pure GIFT and PPO-Tuned across both panels and six windows.",
        "- Error bars: standard deviation across seeds 42, 123, and 456.",
        "- Interpretation: inspect consistency across windows, not only the grand mean.",
        "",
        "## stage_g_sharpe_delta_forest", "",
        "- Purpose: expose heterogeneous dataset-window Sharpe effects.",
        "- Error bars: seed-level standard deviation, not a confidence interval.",
        "- Interpretation: effects crossing zero indicate unstable seed-level direction.",
    ]
    (output / "figure_catalog.md").write_text("\n".join(catalog) + "\n", encoding="utf-8")
    print(f"Analysis: {output / 'stage_g_analysis.md'}")
    print(f"Figure: {output / 'figures' / 'stage_g_six_window_sharpe.png'}")
    print(f"Strict supported: {strict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
