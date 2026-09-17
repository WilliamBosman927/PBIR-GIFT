"""Create the final, source-locked Stage-I comparison bundle."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from lock_stage_b_results import verify_lock as verify_stage_b_lock
from lock_stage_c_results import verify as verify_stage_c_lock
from lock_stage_d_results import verify as verify_stage_d_lock
from lock_stage_e_results import verify as verify_stage_e_lock
from lock_stage_f_results import verify as verify_stage_f_lock
from lock_stage_g_results import verify as verify_stage_g_lock
from lock_stage_h_results import verify as verify_stage_h_lock
from supplemental_experiment_utils import holm_adjust, read_json, resolve_path, write_csv, write_json


MAIN_METHODS = (
    "pbir", "pure_gift", "ppo_test_only", "ppo_tuned",
    "sma", "bollinger", "xgboost",
)
SUPPLEMENTARY_METHODS = (
    "ppo_train_plus_test", "equal_weight", "wma", "atr", "turn_of_month",
)
LABELS = {
    "pbir": "PBIR-GIFT",
    "pure_gift": "Pure GIFT",
    "ppo_test_only": "PPO-Matched",
    "ppo_tuned": "PPO-Tuned",
    "ppo_train_plus_test": "PPO-Train+FineTune",
    "equal_weight": "Equal Weight",
    "sma": "SMA",
    "wma": "WMA",
    "atr": "ATR",
    "bollinger": "Bollinger Bands",
    "turn_of_month": "Turn-of-the-Month",
    "xgboost": "XGBoost",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-e-dir", default="results/stage_e_full_window_baselines")
    parser.add_argument("--stage-f-dir", default="results/stage_f_full_window_cached_pbir")
    parser.add_argument("--stage-g-dir", default="results/stage_g_tuned_ppo")
    parser.add_argument("--stage-h-dir", default="results/stage_h_traditional_baselines")
    parser.add_argument("--output-dir", default="results/stage_i_final_comparison")
    parser.add_argument("--resamples", type=int, default=10000)
    parser.add_argument("--block-length", type=int, default=20)
    parser.add_argument("--primary-metric", choices=("sharpe",), default="sharpe")
    parser.add_argument("--correction", choices=("holm",), default="holm")
    return parser.parse_args()


def _verify_locks() -> dict[str, dict[str, Any]]:
    paths = {
        "stage_b": resolve_path("results/stage_b_lock/stage_b_lock_manifest.json"),
        "stage_c": resolve_path("results/stage_c_lock/stage_c_lock_manifest.json"),
        "stage_d": resolve_path("results/stage_d_lock/stage_d_lock_manifest.json"),
        "stage_e": resolve_path("results/stage_e_lock/stage_e_lock_manifest.json"),
        "stage_f": resolve_path("results/stage_f_lock/stage_f_lock_manifest.json"),
        "stage_g": resolve_path("results/stage_g_lock/stage_g_lock_manifest.json"),
        "stage_h": resolve_path("results/stage_h_lock/stage_h_lock_manifest.json"),
    }
    verify_stage_b_lock(paths["stage_b"])
    verify_stage_c_lock(paths["stage_c"])
    verify_stage_d_lock(paths["stage_d"])
    verify_stage_e_lock(paths["stage_e"])
    verify_stage_f_lock(paths["stage_f"])
    verify_stage_g_lock(paths["stage_g"])
    verify_stage_h_lock(paths["stage_h"])
    return {name: read_json(path) for name, path in paths.items()}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _collect_cell_sharpe(
    stage_e: Path, stage_g: Path, stage_h: Path,
) -> dict[tuple[str, str, str], float]:
    values: dict[tuple[str, str, str], float] = {}
    for row in _read_csv(stage_h / "analysis_bundle" / "stage_h_cell_metrics.csv"):
        method = row["method"]
        if method in {"pbir", "pure_gift", "sma", "wma", "atr", "bollinger",
                      "turn_of_month", "xgboost"}:
            values[(row["dataset"], row["window"], method)] = float(
                row["mean_test_sharpe"])
    for row in _read_csv(stage_e / "analysis_bundle" / "stage_e_cell_metrics.csv"):
        method = row["method"]
        if method in {"ppo_test_only", "ppo_train_plus_test", "equal_weight"}:
            values[(row["dataset"], row["window"], method)] = float(
                row["mean_test_sharpe"])
    for row in _read_csv(stage_g / "analysis_bundle" / "stage_g_cell_metrics.csv"):
        values[(row["dataset"], row["window"], "ppo_tuned")] = float(
            row["ppo_tuned_mean_test_sharpe"])
    expected_methods = set(MAIN_METHODS) | set(SUPPLEMENTARY_METHODS)
    cells = {(key[0], key[1]) for key in values}
    if len(cells) != 12:
        raise ValueError(f"Stage-I expected 12 cells, found {len(cells)}")
    for dataset, window in cells:
        missing = [
            method for method in expected_methods
            if (dataset, window, method) not in values]
        if missing:
            raise ValueError(f"Missing Stage-I methods for {dataset}/{window}: {missing}")
    return values


def _upstream_contrasts(
    stage_e: Path, stage_f: Path, stage_g: Path, stage_h: Path,
) -> list[dict[str, Any]]:
    e = read_json(stage_e / "analysis_bundle" / "stage_e_inference.json")
    f = read_json(stage_f / "analysis_bundle" / "stage_f_inference.json")
    g = read_json(stage_g / "analysis_bundle" / "stage_g_inference.json")
    h = read_json(stage_h / "analysis_bundle" / "stage_h_inference.json")
    sources = [
        ("pbir_vs_pure_gift", "PBIR-GIFT − Pure GIFT",
         f["metrics"]["sharpe"], "Stage-F", True),
        ("pure_gift_vs_ppo_matched", "Pure GIFT − PPO-Matched",
         e["contrasts"]["pure_gift_vs_ppo_test_only"], "Stage-E", True),
        ("pure_gift_vs_ppo_tuned", "Pure GIFT − PPO-Tuned",
         g["metrics"]["sharpe"], "Stage-G", True),
        ("pure_gift_vs_sma", "Pure GIFT − SMA",
         h["contrasts"]["pure_gift_vs_sma"], "Stage-H", False),
        ("pure_gift_vs_bollinger", "Pure GIFT − Bollinger Bands",
         h["contrasts"]["pure_gift_vs_bollinger"], "Stage-H", False),
        ("pure_gift_vs_xgboost", "Pure GIFT − XGBoost",
         h["contrasts"]["pure_gift_vs_xgboost"], "Stage-H", False),
    ]
    rows = []
    for key, label, value, stage, confirmatory in sources:
        mean = value.get("mean_cell_delta", value.get("mean_cell_delta_sharpe"))
        p_value = value["exact_one_sided_sign_flip_p"]
        rows.append({
            "contrast": key,
            "label": label,
            "source_stage": stage,
            "confirmatory_family": confirmatory,
            "mean_cell_delta_sharpe": float(mean),
            "positive_cells": int(value["positive_cells"]),
            "hedges_g_paired": value.get("hedges_g_paired"),
            "exact_one_sided_sign_flip_p": float(p_value),
            "bootstrap_ci95_low": float(value["bootstrap"]["ci95"][0]),
            "bootstrap_ci95_high": float(value["bootstrap"]["ci95"][1]),
            "bootstrap_probability_positive": float(
                value["bootstrap"]["probability_positive"]),
            "upstream_strict_supported": bool(value.get(
                "strict_supported", value.get("strict_full_window_supported", False))),
        })
    adjusted = holm_adjust({row["contrast"]: row["exact_one_sided_sign_flip_p"]
                            for row in rows})
    for row in rows:
        row["holm_adjusted_p_six_contrasts"] = adjusted[row["contrast"]]
    return rows


def _make_figures(
    output: Path, values: dict[tuple[str, str, str], float],
    contrasts: list[dict[str, Any]],
) -> None:
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    cells = sorted({(dataset, window) for dataset, window, _ in values})
    matrix = np.asarray([
        [values[(dataset, window, method)] for method in MAIN_METHODS]
        for dataset, window in cells
    ])
    # Convert Sharpe values to within-cell ranks: 1 is best.
    ranks = np.empty_like(matrix)
    for index, row in enumerate(matrix):
        order = np.argsort(-row)
        ranks[index, order] = np.arange(1, len(row) + 1)
    figure, axis = plt.subplots(figsize=(11.8, 6.8))
    image = axis.imshow(ranks, aspect="auto", cmap="YlGnBu_r", vmin=1,
                        vmax=len(MAIN_METHODS))
    axis.set_xticks(np.arange(len(MAIN_METHODS)),
                    [LABELS[method] for method in MAIN_METHODS],
                    rotation=32, ha="right")
    axis.set_yticks(np.arange(len(cells)), [
        f"{'D1' if dataset == 'portfolio_5stocks' else 'D2'}-{window}"
        for dataset, window in cells])
    axis.set_title("Main-method rank by dataset-window cell", fontweight="bold")
    colorbar = figure.colorbar(image, ax=axis, shrink=0.82)
    colorbar.set_label("Rank (1 = highest Sharpe)")
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(figures / f"stage_i_main_method_rank_heatmap.{suffix}",
                       dpi=240, bbox_inches="tight")
    plt.close(figure)

    ordered = list(reversed(contrasts))
    y = np.arange(len(ordered))
    means = np.asarray([row["mean_cell_delta_sharpe"] for row in ordered])
    low = np.asarray([row["bootstrap_ci95_low"] for row in ordered])
    high = np.asarray([row["bootstrap_ci95_high"] for row in ordered])
    figure, axis = plt.subplots(figsize=(10.2, 5.5))
    colors = ["#1261A0" if row["confirmatory_family"] else "#6C8EAD"
              for row in ordered]
    for index, (mean, low_value, high_value, color) in enumerate(
            zip(means, low, high, colors)):
        axis.errorbar(
            mean, y[index],
            xerr=np.asarray([[mean - low_value], [high_value - mean]]),
            fmt="o", color=color, ecolor=color, capsize=4,
            linewidth=2, markersize=6)
    axis.axvline(0, color="#5E6872", linewidth=1.2)
    axis.set_yticks(y, [row["label"] for row in ordered])
    axis.set_xlabel("Mean ΔSharpe with hierarchical block-bootstrap 95% CI")
    axis.set_title("Pre-specified key contrasts", fontweight="bold")
    axis.grid(axis="x", alpha=0.2)
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(figures / f"stage_i_key_contrast_forest.{suffix}",
                       dpi=240, bbox_inches="tight")
    plt.close(figure)


def main() -> int:
    args = _parse_args()
    if args.resamples != 10000 or args.block_length != 20:
        raise ValueError(
            "Stage-I is a synthesis of locked upstream analyses and requires "
            "the pre-specified --resamples 10000 --block-length 20 settings")
    locks = _verify_locks()
    stage_e = resolve_path(args.stage_e_dir)
    stage_f = resolve_path(args.stage_f_dir)
    stage_g = resolve_path(args.stage_g_dir)
    stage_h = resolve_path(args.stage_h_dir)
    output = resolve_path(args.output_dir)
    for path in (stage_e, stage_f, stage_g, stage_h):
        if output == path or path in output.parents:
            raise ValueError("Stage-I output must not be inside an upstream result root")
    output.mkdir(parents=True, exist_ok=True)
    values = _collect_cell_sharpe(stage_e, stage_g, stage_h)
    contrasts = _upstream_contrasts(stage_e, stage_f, stage_g, stage_h)

    cells = sorted({(dataset, window) for dataset, window, _ in values})
    summary_rows = []
    for reporting_group, methods in (("main", MAIN_METHODS),
                                     ("supplementary", SUPPLEMENTARY_METHODS)):
        for method in methods:
            cell = np.asarray([values[(dataset, window, method)]
                               for dataset, window in cells])
            ranks = []
            comparison_methods = MAIN_METHODS if reporting_group == "main" else (
                MAIN_METHODS + SUPPLEMENTARY_METHODS)
            for dataset, window in cells:
                ordered = sorted(
                    comparison_methods,
                    key=lambda candidate: values[(dataset, window, candidate)],
                    reverse=True)
                ranks.append(ordered.index(method) + 1)
            summary_rows.append({
                "reporting_group": reporting_group,
                "method": method,
                "method_label": LABELS[method],
                "mean_cell_sharpe": float(np.mean(cell)),
                "sd_across_cells": float(np.std(cell, ddof=1)),
                "median_cell_sharpe": float(np.median(cell)),
                "positive_sharpe_cells": int(np.sum(cell > 0)),
                "mean_rank": float(np.mean(ranks)),
                "rank_1_cells": int(np.sum(np.asarray(ranks) == 1)),
            })
    write_csv(output / "stage_i_method_summary.csv", summary_rows)
    write_csv(output / "stage_i_key_contrasts.csv", contrasts)
    _make_figures(output, values, contrasts)

    confirmatory = [row for row in contrasts if row["confirmatory_family"]]
    inference = {
        "analysis_role": "stage-i-final-locked-synthesis",
        "primary_metric": "out-of-sample Sharpe",
        "inferential_unit": "dataset-window cell mean",
        "main_methods": list(MAIN_METHODS),
        "supplementary_methods": list(SUPPLEMENTARY_METHODS),
        "key_contrasts": contrasts,
        "confirmatory_decisions_after_six-contrast_holm": {
            row["contrast"]: bool(
                row["holm_adjusted_p_six_contrasts"] < 0.05
                and row["bootstrap_ci95_low"] > 0)
            for row in confirmatory
        },
        "source_lock_hashes": {
            name: value["aggregate_sha256"] for name, value in locks.items()},
        "caveats": [
            "Stage-I does not retune or rerun any upstream method.",
            "PBIR versus Pure GIFT uses fixed-code Stage-C/F evidence.",
            "External method rankings use full pipeline Stage-B GIFT/PBIR outputs.",
            "Equal Weight remains reported in the supplementary table.",
            "Only method-specific, not universal, superiority claims are permitted.",
        ],
    }
    write_json(output / "stage_i_inference.json", inference)

    report = [
        "# Stage-I final comparison analysis", "",
        "## Evidence hierarchy", "",
        "1. Primary innovation: cached-code PBIR-GIFT versus Pure GIFT.",
        "2. Mechanism control: Pure GIFT versus PPO-Matched.",
        "3. Fairness control: Pure GIFT versus independently tuned PPO.",
        "4. External context: SMA, Bollinger Bands, XGBoost, and supplementary baselines.",
        "", "## Key contrast decisions", "",
    ]
    for row in contrasts:
        supported = bool(
            row["holm_adjusted_p_six_contrasts"] < 0.05
            and row["bootstrap_ci95_low"] > 0)
        report.append(
            f"- {row['label']}: ΔSharpe={row['mean_cell_delta_sharpe']:+.3f}, "
            f"95% CI [{row['bootstrap_ci95_low']:+.3f}, "
            f"{row['bootstrap_ci95_high']:+.3f}], Holm p="
            f"{row['holm_adjusted_p_six_contrasts']:.4f}, supported={supported}.")
    report.extend([
        "", "## Claim boundary", "",
        "Report every planned contrast, including null or negative results. A method may "
        "be described as superior only for contrasts whose adjusted test and bootstrap "
        "interval both support the direction. Universal dominance is not tested.",
    ])
    for filename in ("analysis_report.md", "stage_i_analysis.md"):
        (output / filename).write_text("\n".join(report) + "\n", encoding="utf-8")
    stats = [
        "# Stage-I statistical appendix", "",
        "- Unit: 12 dataset-window cell means; seeds are repeated runs within cells.",
        "- Six pre-specified Sharpe contrasts form the final Holm family.",
        "- Confidence intervals are inherited from locked hierarchical block-bootstrap analyses.",
        "- A final supported claim requires adjusted p < .05 and CI lower bound > 0.",
    ]
    (output / "stats_appendix.md").write_text("\n".join(stats) + "\n", encoding="utf-8")
    catalog = [
        "# Stage-I figure catalog", "",
        "## stage_i_main_method_rank_heatmap", "",
        "- Purpose: show whether rankings are stable across all 12 cells.",
        "- Interpretation: rank 1 means highest Sharpe within the pre-declared main panel.",
        "", "## stage_i_key_contrast_forest", "",
        "- Purpose: show effect direction, magnitude, and bootstrap uncertainty together.",
        "- Interpretation: intervals crossing zero do not support strict superiority.",
    ]
    (output / "figure_catalog.md").write_text("\n".join(catalog) + "\n", encoding="utf-8")
    print(f"Analysis: {output / 'stage_i_analysis.md'}")
    print(f"Figure: {output / 'figures' / 'stage_i_key_contrast_forest.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
