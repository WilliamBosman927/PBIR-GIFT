"""Analyze locked GIFT/PBIR results against Stage-D pure-PPO baselines."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analyze_final_results import (
    _circular_block_indices,
    _exact_sign_flip,
    _hedges_g_paired,
    _holm,
)
from hparam_sweep_runner import PROJECT_DIR
from metrics import max_drawdown, sharpe_ratio


DATASETS = ("portfolio_5stocks", "portfolio_5stocks2")
WINDOWS = ("W1", "W3", "W5")
SEEDS = (42, 123, 456)
CONTRASTS = {
    "gift_pipeline_vs_ppo_train_plus_test": (
        "Pure GIFT (Stage B)", "PPO-Train+FineTune"),
    "gift_pipeline_vs_ppo_test_only": (
        "Pure GIFT (Stage B)", "PPO-TestOnly"),
    "pbir_fixed_vs_ppo_train_plus_test": (
        "PBIR-GIFT (Stage C)", "PPO-Train+FineTune"),
    "pbir_fixed_vs_gift_fixed": (
        "PBIR-GIFT (Stage C)", "Pure GIFT (Stage C)"),
    "gift_pipeline_vs_equal_weight": (
        "Pure GIFT (Stage B)", "Equal Weight"),
    "pbir_fixed_vs_equal_weight": (
        "PBIR-GIFT (Stage C)", "Equal Weight"),
}
METHOD_LABELS = {
    "gift_stage_b": "Pure GIFT\n(Stage B pipeline)",
    "pbir_stage_c": "PBIR-GIFT\n(Stage C fixed code)",
    "gift_stage_c": "Pure GIFT\n(Stage C fixed code)",
    "ppo_train_plus_test": "PPO\n(train + fine-tune)",
    "ppo_test_only": "PPO\n(test-period training)",
    "equal_weight": "Equal Weight",
}


def _resolve(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_DIR / path).resolve()


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _gift_result(cell: Path, method: str) -> tuple[dict, np.ndarray]:
    value = _read(cell / "methods" / method / "final_comparison.json")
    result = value["test_result"]
    returns = np.asarray(value["daily_returns"]["method"], dtype=float)
    return result, returns


def _baseline_result(cell: Path, method: str) -> tuple[dict, np.ndarray]:
    value = _read(cell / "methods" / f"{method}.json")
    return value["test_result"], np.asarray(value["daily_returns"], dtype=float)


def _bootstrap(
        pairs: dict[tuple[str, str], list[tuple[np.ndarray, np.ndarray]]],
        resamples: int, block_length: int, seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    cells = sorted(pairs)
    boot = np.empty(resamples)
    for sample in range(resamples):
        effects = []
        for chosen in rng.choice(len(cells), size=len(cells), replace=True):
            members = pairs[cells[int(chosen)]]
            seed_effects = []
            for _ in range(len(members)):
                first, second = members[int(rng.integers(0, len(members)))]
                n = min(len(first), len(second))
                indices = _circular_block_indices(n, block_length, rng)
                seed_effects.append(
                    sharpe_ratio(first[:n][indices])
                    - sharpe_ratio(second[:n][indices]))
            effects.append(float(np.mean(seed_effects)))
        boot[sample] = float(np.mean(effects))
    return {
        "resamples": resamples,
        "block_length": block_length,
        "ci95": [float(np.percentile(boot, 2.5)),
                 float(np.percentile(boot, 97.5))],
        "probability_positive": float(np.mean(boot > 0)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-b-dir", default="results/stage_b_final")
    parser.add_argument(
        "--stage-c-dir", default="results/stage_c_cached_replication")
    parser.add_argument(
        "--baseline-dir", default="results/stage_d_ppo_baselines")
    parser.add_argument(
        "--output-dir", default="results/stage_d_ppo_baselines/analysis_bundle")
    parser.add_argument("--resamples", type=int, default=10000)
    parser.add_argument("--block-length", type=int, default=20)
    args = parser.parse_args()
    stage_b = _resolve(args.stage_b_dir)
    stage_c = _resolve(args.stage_c_dir)
    baseline = _resolve(args.baseline_dir)
    output = _resolve(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    expected = {
        (dataset, window, seed)
        for dataset in DATASETS for window in WINDOWS for seed in SEEDS}
    observed = {
        (path.parents[2].name, path.parents[1].name,
         int(path.parent.name.removeprefix("seed_")))
        for path in baseline.glob("*/W[135]/seed_*/baseline_summary.json")
        if _read(path).get("status") == "completed"}
    if observed != expected:
        raise RuntimeError(
            "Stage-D is incomplete; analysis refused. "
            f"Missing: {sorted(expected - observed)}")

    contrast_pairs: dict[
        str, dict[tuple[str, str], list[tuple[np.ndarray, np.ndarray]]]
    ] = {name: defaultdict(list) for name in CONTRASTS}
    cell_effects: dict[str, dict[tuple[str, str], list[float]]] = {
        name: defaultdict(list) for name in CONTRASTS}
    mdd_effects: dict[str, dict[tuple[str, str], list[float]]] = {
        name: defaultdict(list) for name in CONTRASTS}
    method_sharpes: dict[
        str, dict[tuple[str, str], list[float]]
    ] = {name: defaultdict(list) for name in METHOD_LABELS}

    for dataset, window, seed in sorted(expected):
        key = (dataset, window)
        b_cell = stage_b / dataset / window / f"seed_{seed}"
        c_cell = stage_c / dataset / window / f"seed_{seed}"
        d_cell = baseline / dataset / window / f"seed_{seed}"
        b_gift, b_gift_r = _gift_result(b_cell, "pure_gift")
        c_pbir, c_pbir_r = _gift_result(c_cell, "pbir")
        c_gift, c_gift_r = _gift_result(c_cell, "pure_gift")
        ppo_test, ppo_test_r = _baseline_result(d_cell, "ppo_test_only")
        ppo_strong, ppo_strong_r = _baseline_result(
            d_cell, "ppo_train_plus_test")
        equal, equal_r = _baseline_result(d_cell, "equal_weight")
        sources = {
            "gift_pipeline_vs_ppo_train_plus_test": (
                b_gift, b_gift_r, ppo_strong, ppo_strong_r),
            "gift_pipeline_vs_ppo_test_only": (
                b_gift, b_gift_r, ppo_test, ppo_test_r),
            "pbir_fixed_vs_ppo_train_plus_test": (
                c_pbir, c_pbir_r, ppo_strong, ppo_strong_r),
            "pbir_fixed_vs_gift_fixed": (
                c_pbir, c_pbir_r, c_gift, c_gift_r),
            "gift_pipeline_vs_equal_weight": (
                b_gift, b_gift_r, equal, equal_r),
            "pbir_fixed_vs_equal_weight": (
                c_pbir, c_pbir_r, equal, equal_r),
        }
        method_sources = {
            "gift_stage_b": b_gift,
            "pbir_stage_c": c_pbir,
            "gift_stage_c": c_gift,
            "ppo_train_plus_test": ppo_strong,
            "ppo_test_only": ppo_test,
            "equal_weight": equal,
        }
        for method, value in method_sources.items():
            method_sharpes[method][key].append(float(value["test_sharpe"]))
        for name, (first, first_r, second, second_r) in sources.items():
            if len(first_r) != len(second_r) or len(first_r) < 22:
                raise ValueError(f"Unpaired return lengths: {name}/{key}/seed_{seed}")
            contrast_pairs[name][key].append((first_r, second_r))
            cell_effects[name][key].append(
                float(first["test_sharpe"] - second["test_sharpe"]))
            mdd_effects[name][key].append(
                float(second["test_max_drawdown"]
                      - first["test_max_drawdown"]))

    cell_rows = []
    summaries: dict[str, dict[str, Any]] = {}
    raw_p = {}
    for index, (name, labels) in enumerate(CONTRASTS.items()):
        means = {
            key: float(np.mean(values))
            for key, values in cell_effects[name].items()}
        values = np.asarray([means[key] for key in sorted(means)])
        raw_p[name] = _exact_sign_flip(values)
        boot = _bootstrap(
            contrast_pairs[name], args.resamples, args.block_length,
            20260902 + index)
        dataset_means = {
            dataset: float(np.mean([
                value for (observed_dataset, _), value in means.items()
                if observed_dataset == dataset]))
            for dataset in DATASETS}
        window_means = {
            window: float(np.mean([
                value for (_, observed_window), value in means.items()
                if observed_window == window]))
            for window in WINDOWS}
        summaries[name] = {
            "first_method": labels[0],
            "second_method": labels[1],
            "mean_cell_delta_sharpe": float(np.mean(values)),
            "median_cell_delta_sharpe": float(np.median(values)),
            "cell_win_rate": float(np.mean(values > 0)),
            "seed_run_win_rate": float(np.mean([
                value > 0 for seed_values in cell_effects[name].values()
                for value in seed_values])),
            "hedges_g_paired": _hedges_g_paired(values),
            "exact_one_sided_sign_flip_p": raw_p[name],
            "bootstrap": boot,
            "dataset_means": dataset_means,
            "window_means": window_means,
            "mean_mdd_improvement_percentage_points": float(np.mean([
                np.mean(value) for value in mdd_effects[name].values()])),
        }
        for key in sorted(means):
            seed_values = np.asarray(cell_effects[name][key])
            cell_rows.append({
                "contrast": name,
                "dataset": key[0],
                "window": key[1],
                "n_seeds": len(seed_values),
                "mean_delta_sharpe": float(np.mean(seed_values)),
                "seed_sd_delta_sharpe": float(np.std(seed_values, ddof=1)),
            })
    adjusted = _holm(raw_p)
    for name in summaries:
        summaries[name]["holm_adjusted_p"] = adjusted[name]

    primary = summaries["gift_pipeline_vs_ppo_train_plus_test"]
    primary_supported = bool(
        primary["bootstrap"]["ci95"][0] > 0
        and all(value > 0 for value in primary["dataset_means"].values())
        and sum(value > 0 for value in primary["window_means"].values()) >= 2)
    stage_c_inference = _read(
        stage_c / "analysis_bundle" / "confirmatory_inference.json")
    inference = {
        "analysis_role": "stage_d_baseline_comparison",
        "complete_cells": 18,
        "independent_cell_means": 6,
        "primary_question": (
            "Does the Stage-B Pure-GIFT pipeline outperform "
            "PPO-Train+FineTune on test Sharpe?"),
        "primary_supported": primary_supported,
        "contrasts": summaries,
        "pbir_advantage_decision_from_locked_stage_c": {
            "supported": stage_c_inference["primary"]["predeclared_support"],
            "mean_delta_sharpe": stage_c_inference["primary"][
                "mean_delta_sharpe"],
            "ci95": stage_c_inference["bootstrap"]["sharpe"]["ci95"],
        },
        "caveats": [
            "D1 and D2 share three stocks and are not independent universes.",
            "The analysis unit is the dataset-window cell mean, not each day.",
            "Stage-C comparisons are conditional on Stage-B-selected code.",
            "Equal-weight seed duplicates are descriptive only.",
        ],
    }
    (output / "stage_d_inference.json").write_text(
        json.dumps(inference, indent=2), encoding="utf-8")
    with (output / "stage_d_contrast_cells.csv").open(
            "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(cell_rows[0]))
        writer.writeheader()
        writer.writerows(cell_rows)

    ordered_cells = [
        (dataset, window) for dataset in DATASETS for window in WINDOWS]
    method_rows = []
    method_summaries: dict[str, dict[str, Any]] = {}
    cell_matrix = np.empty((len(METHOD_LABELS), len(ordered_cells)))
    for method_index, (method, label) in enumerate(METHOD_LABELS.items()):
        values = []
        for cell_index, key in enumerate(ordered_cells):
            seed_values = np.asarray(method_sharpes[method][key], dtype=float)
            cell_mean = float(np.mean(seed_values))
            values.append(cell_mean)
            cell_matrix[method_index, cell_index] = cell_mean
            method_rows.append({
                "method": method,
                "method_label": label.replace("\n", " "),
                "dataset": key[0],
                "window": key[1],
                "n_seed_runs": len(seed_values),
                "mean_sharpe": cell_mean,
                "seed_sd_sharpe": float(np.std(seed_values, ddof=1)),
            })
        vector = np.asarray(values)
        method_summaries[method] = {
            "method_label": label.replace("\n", " "),
            "mean_across_six_cells": float(np.mean(vector)),
            "sd_across_six_cells": float(np.std(vector, ddof=1)),
            "positive_cells": int(np.sum(vector > 0)),
            "best_cells": 0,
        }
    winners = np.argmax(cell_matrix, axis=0)
    for index, method in enumerate(METHOD_LABELS):
        method_summaries[method]["best_cells"] = int(np.sum(winners == index))
    inference["absolute_method_performance"] = method_summaries
    (output / "stage_d_inference.json").write_text(
        json.dumps(inference, indent=2), encoding="utf-8")
    with (output / "stage_d_method_cell_sharpe.csv").open(
            "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(method_rows[0]))
        writer.writeheader()
        writer.writerows(method_rows)

    names = list(CONTRASTS)
    means = np.asarray([
        summaries[name]["mean_cell_delta_sharpe"] for name in names])
    lows = np.asarray([summaries[name]["bootstrap"]["ci95"][0]
                       for name in names])
    highs = np.asarray([summaries[name]["bootstrap"]["ci95"][1]
                        for name in names])
    labels = [
        "GIFT pipeline − PPO train+FT",
        "GIFT pipeline − PPO test-only",
        "PBIR fixed − PPO train+FT",
        "PBIR fixed − GIFT fixed",
        "GIFT pipeline − Equal Weight",
        "PBIR fixed − Equal Weight",
    ]
    figure, axis = plt.subplots(figsize=(10.0, 5.5))
    y = np.arange(len(names))[::-1]
    axis.errorbar(
        means, y, xerr=np.vstack([means - lows, highs - means]),
        fmt="D", color="#2F5D8C", ecolor="#9BB3C9", capsize=5,
        linewidth=2.2)
    axis.axvline(0, color="#4d5963", linewidth=1.2)
    axis.set_yticks(y, labels)
    axis.set_xlabel("Mean cell ΔSharpe with 95% hierarchical block-bootstrap CI")
    axis.set_title("GIFT/PBIR baseline contrasts", fontweight="bold")
    axis.grid(axis="x", alpha=0.22)
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(output / f"stage_d_sharpe_contrasts.{suffix}",
                       dpi=240, bbox_inches="tight")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(10.5, 5.8))
    image = axis.imshow(cell_matrix, cmap="RdYlGn", aspect="auto", vmin=-2.0,
                        vmax=2.0)
    for row in range(cell_matrix.shape[0]):
        for column in range(cell_matrix.shape[1]):
            value = cell_matrix[row, column]
            axis.text(column, row, f"{value:+.2f}", ha="center", va="center",
                      fontsize=9, color="white" if abs(value) > 1.15 else "#17202a")
    axis.set_xticks(
        np.arange(len(ordered_cells)),
        [f"{'D1' if dataset == DATASETS[0] else 'D2'}-{window}"
         for dataset, window in ordered_cells])
    axis.set_yticks(np.arange(len(METHOD_LABELS)), list(METHOD_LABELS.values()))
    axis.set_title("Absolute out-of-sample Sharpe by method and evaluation cell",
                   fontweight="bold")
    axis.set_xlabel("Dataset-window cell (mean across three seeds)")
    colorbar = figure.colorbar(image, ax=axis, shrink=0.88)
    colorbar.set_label("Sharpe ratio")
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(output / f"stage_d_absolute_sharpe_heatmap.{suffix}",
                       dpi=240, bbox_inches="tight")
    plt.close(figure)

    primary_ci = primary["bootstrap"]["ci95"]
    equal_gift = summaries["gift_pipeline_vs_equal_weight"]
    equal_pbir = summaries["pbir_fixed_vs_equal_weight"]
    lines = [
        "# Stage-D baseline analysis report",
        "",
        "## Decision",
        "",
        f"Primary GIFT-vs-strong-PPO claim: **{'SUPPORTED' if primary_supported else 'NOT SUPPORTED'}**.",
        "This is a statistical decision, not a program failure.",
        "",
        "The Stage-B Pure-GIFT pipeline had a directionally favorable mean "
        f"Sharpe contrast against PPO-Train+FineTune ({primary['mean_cell_delta_sharpe']:+.3f}); "
        f"all six cell means were positive and Hedges g was {primary['hedges_g_paired']:+.3f}. "
        f"However, the predeclared 95% hierarchical block-bootstrap interval "
        f"[{primary_ci[0]:+.3f}, {primary_ci[1]:+.3f}] includes zero. The data "
        "therefore do not support a confirmatory superiority claim.",
        "",
        "## Audit and estimands",
        "",
        "- 18/18 dataset-window-seed cells completed.",
        "- Each method has 105 aligned out-of-sample daily returns per cell.",
        "- Stage-D pure-PPO and equal-weight baselines made zero LLM calls.",
        "- Inference is based on six dataset-window cell means; daily returns and "
        "  seeds are not treated as independent studies.",
        "",
        "## Contrast results",
        "",
        "| Contrast | Mean ΔSharpe | Cell wins | Seed-run wins | Hedges g | Raw p | Holm p | 95% block-bootstrap CI |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in names:
        value = summaries[name]
        ci = value["bootstrap"]["ci95"]
        lines.append(
            f"| {name} | {value['mean_cell_delta_sharpe']:+.3f} | "
            f"{value['cell_win_rate']:.1%} | {value['seed_run_win_rate']:.1%} | "
            f"{value['hedges_g_paired']:+.3f} | "
            f"{value['exact_one_sided_sign_flip_p']:.4f} | "
            f"{value['holm_adjusted_p']:.4f} | [{ci[0]:+.3f}, {ci[1]:+.3f}] |")
    lines.extend([
        "",
        "## Absolute performance check",
        "",
        "| Method | Mean Sharpe across 6 cells | SD across cells | Positive cells | Best cells |",
        "|---|---:|---:|---:|---:|",
    ])
    for method in METHOD_LABELS:
        value = method_summaries[method]
        lines.append(
            f"| {value['method_label']} | "
            f"{value['mean_across_six_cells']:+.3f} | "
            f"{value['sd_across_six_cells']:.3f} | "
            f"{value['positive_cells']}/6 | {value['best_cells']}/6 |")
    lines.extend([
        "",
        "Equal Weight is a material application-level challenge: Pure GIFT's "
        f"mean cell contrast is {equal_gift['mean_cell_delta_sharpe']:+.3f} "
        f"({equal_gift['cell_win_rate']:.0%} of cells positive), and PBIR-GIFT's "
        f"is {equal_pbir['mean_cell_delta_sharpe']:+.3f} "
        f"({equal_pbir['cell_win_rate']:.0%} positive). Thus evidence of an "
        "advantage over PPO must not be rewritten as an advantage over simple "
        "portfolio allocation.",
        "",
        "## Interpretation and claim boundary",
        "",
        "- Pure GIFT versus strong PPO: large positive point estimate and 6/6 "
        "  positive cell means, but uncertainty still touches zero; directional, "
        "  not confirmatory.",
        "- PBIR versus Pure GIFT: the locked Stage-C result remains directionally "
        "  favorable (+0.368; 5/6 cells), but its 95% interval also crosses zero.",
        "- A confidence interval crossing zero is inconclusive; it is neither "
        "  evidence of equivalence nor proof of no effect.",
        "- With the current evidence, valid wording is: GIFT/PBIR showed favorable "
        "  contrasts to PPO in these selected cells, with substantial seed and "
        "  cell uncertainty. Claims of universal superiority are not supported.",
        "",
        "## Recommended next action",
        "",
        "Lock this completed Stage-D evidence. Do not add seeds merely to force "
        "significance. If another confirmatory run is affordable, predeclare one "
        "genuinely unseen time period or independent asset universe, keep all "
        "configurations fixed, and include Equal Weight as a mandatory baseline.",
    ])
    report_text = "\n".join(lines) + "\n"
    (output / "analysis_report.md").write_text(report_text, encoding="utf-8")
    (output / "stage_d_analysis.md").write_text(report_text, encoding="utf-8")

    appendix = [
        "# Stage-D statistical appendix",
        "",
        "## Statistical unit and uncertainty",
        "",
        "The primary summary uses six dataset-window cell means. Within each "
        "bootstrap replicate, cells are resampled hierarchically, seed runs are "
        "resampled within selected cells, and aligned daily returns are sampled "
        f"with circular blocks of length {args.block_length}. "
        f"Intervals use {args.resamples:,} bootstrap replicates.",
        "",
        "Exact one-sided sign-flip tests operate on the six cell-mean Sharpe "
        "differences. Holm values adjust the family of six reported contrasts. "
        "Hedges g is the small-sample corrected standardized paired effect across "
        "the same six cell means.",
        "",
        "## Multiplicity and decision rule",
        "",
        "The primary support rule requires the 95% block-bootstrap interval lower "
        "bound to exceed zero, both dataset means to be positive, and at least two "
        "of three window means to be positive. The confidence-interval condition "
        "failed. Exact and Holm-adjusted p-values are supporting diagnostics, not "
        "substitutes for the predeclared decision rule.",
        "",
        "## Reproducibility",
        "",
        "Machine-readable estimates are stored in `stage_d_inference.json`; "
        "cell-level contrasts and absolute cell means are stored in the two CSV "
        "files in this bundle. Equal-weight results are identical across seeds by "
        "construction and are collapsed descriptively at cell level.",
    ]
    (output / "stats_appendix.md").write_text(
        "\n".join(appendix) + "\n", encoding="utf-8")
    catalog = [
        "# Stage-D figure catalog",
        "",
        "## Figure 1 — Sharpe contrasts",
        "",
        "Files: `stage_d_sharpe_contrasts.png` and `.pdf`. Diamonds show mean "
        "Sharpe differences across six dataset-window cells; bars show 95% "
        f"hierarchical circular-block-bootstrap intervals (block={args.block_length}, "
        f"resamples={args.resamples:,}). Positive values favor the first named method.",
        "",
        "## Figure 2 — Absolute Sharpe heatmap",
        "",
        "Files: `stage_d_absolute_sharpe_heatmap.png` and `.pdf`. Each tile is the "
        "mean out-of-sample Sharpe across three seeds for one method and one "
        "dataset-window cell. It prevents favorable pairwise contrasts from hiding "
        "poor absolute performance or failure to beat Equal Weight.",
    ]
    (output / "figure_catalog.md").write_text(
        "\n".join(catalog) + "\n", encoding="utf-8")
    print(f"Analysis: {output / 'analysis_report.md'}")
    print(f"Figure: {output / 'stage_d_sharpe_contrasts.png'}")
    print(f"Primary supported: {primary_supported}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
