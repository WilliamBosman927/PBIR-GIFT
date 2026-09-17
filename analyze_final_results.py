"""Predeclared paired analysis for completed PBIR-GIFT Stage-B results.

The confirmatory set is W1/W3/W5.  W2/W4/W6 are retained in the heatmap and
descriptive tables but are not counted as extra independent time periods.
"""

from __future__ import annotations

import argparse
import csv
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

from hparam_sweep_runner import PROJECT_DIR
from metrics import max_drawdown, sharpe_ratio, sortino_ratio


PRIMARY_WINDOWS = ("W1", "W3", "W5")
DATASETS = ("portfolio_5stocks", "portfolio_5stocks2")
METRICS = {
    "sharpe": (sharpe_ratio, "PBIR − Pure GIFT"),
    "sortino": (sortino_ratio, "PBIR − Pure GIFT"),
    "max_drawdown_improvement": (max_drawdown, "Pure GIFT − PBIR"),
    "total_return": (
        lambda values: (float(np.prod(1.0 + np.asarray(values))) - 1.0) * 100,
        "PBIR − Pure GIFT"),
}


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def load_complete_rows(root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(root.glob("*/W[1-6]/seed_*/final_pair_summary.json")):
        row = _load_json(path)
        if row.get("status") not in {"completed", "skipped_existing"}:
            continue
        required = ("pbir_test_sharpe", "pure_gift_test_sharpe")
        if not all(isinstance(row.get(key), (int, float)) for key in required):
            continue
        row["source_file"] = str(path)
        rows.append(row)
    return rows


def _metric_delta(row: dict[str, Any], metric: str) -> float:
    if metric == "max_drawdown_improvement":
        return float(
            row["pure_gift_test_max_drawdown"]
            - row["pbir_test_max_drawdown"])
    key = "total_return" if metric == "total_return" else metric
    return float(row[f"pbir_test_{key}"] - row[f"pure_gift_test_{key}"])


def collapse_cells(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["dataset"]), str(row["window"]))].append(row)
    output = []
    for (dataset, window), members in sorted(groups.items()):
        item: dict[str, Any] = {
            "dataset": dataset,
            "window": window,
            "n_seeds": len({int(row["seed"]) for row in members}),
            "seeds": ",".join(str(value) for value in sorted(
                {int(row["seed"]) for row in members})),
            "confirmatory": window in PRIMARY_WINDOWS,
        }
        for metric in METRICS:
            values = np.asarray([_metric_delta(row, metric) for row in members])
            item[f"delta_{metric}"] = float(np.mean(values))
            item[f"seed_sd_{metric}"] = (
                float(np.std(values, ddof=1)) if len(values) > 1 else 0.0)
        output.append(item)
    return output


def _exact_sign_flip(values: np.ndarray, alternative: str = "greater") -> float:
    values = np.asarray(values, dtype=float)
    observed = float(np.mean(values))
    permuted = []
    for signs in itertools.product((-1.0, 1.0), repeat=len(values)):
        permuted.append(float(np.mean(values * np.asarray(signs))))
    if alternative == "greater":
        return float(np.mean(np.asarray(permuted) >= observed - 1e-15))
    return float(np.mean(np.abs(permuted) >= abs(observed) - 1e-15))


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
    d_z = float(np.mean(values) / sd)
    correction = 1.0 - 3.0 / (4.0 * len(values) - 5.0)
    return d_z * correction


def _daily_pair(row: dict[str, Any]) -> tuple[np.ndarray, np.ndarray] | None:
    pbir = _load_json(Path(row["pbir_experiment_dir"]) / "final_comparison.json")
    gift = _load_json(
        Path(row["pure_gift_experiment_dir"]) / "final_comparison.json")
    pbir_returns = pbir.get("daily_returns", {}).get("method", [])
    gift_returns = gift.get("daily_returns", {}).get("method", [])
    if not pbir_returns or not gift_returns:
        return None
    n = min(len(pbir_returns), len(gift_returns))
    if n < 22:
        return None
    pbir_array = np.asarray(pbir_returns[:n], dtype=float)
    gift_array = np.asarray(gift_returns[:n], dtype=float)
    if not np.all(np.isfinite(pbir_array)) or not np.all(np.isfinite(gift_array)):
        return None
    return pbir_array, gift_array


def _circular_block_indices(
        n: int, block_length: int, rng: np.random.Generator,
) -> np.ndarray:
    blocks = math.ceil(n / block_length)
    starts = rng.integers(0, n, size=blocks)
    indices = np.concatenate([
        (start + np.arange(block_length)) % n for start in starts])
    return indices[:n]


def hierarchical_block_bootstrap(
        rows: list[dict[str, Any]], metric: str,
        resamples: int, block_length: int, random_seed: int,
) -> dict[str, Any] | None:
    statistic, _ = METRICS[metric]
    groups: dict[tuple[str, str], list[tuple[np.ndarray, np.ndarray]]] = (
        defaultdict(list))
    for row in rows:
        if row["window"] not in PRIMARY_WINDOWS:
            continue
        pair = _daily_pair(row)
        if pair is not None:
            groups[(row["dataset"], row["window"])].append(pair)
    expected = {(dataset, window) for dataset in DATASETS
                for window in PRIMARY_WINDOWS}
    if set(groups) != expected:
        return None

    rng = np.random.default_rng(random_seed)
    cell_keys = sorted(groups)
    boot = np.empty(resamples, dtype=float)
    for sample_idx in range(resamples):
        sampled_cells = rng.choice(len(cell_keys), size=len(cell_keys), replace=True)
        cell_effects = []
        for cell_idx in sampled_cells:
            members = groups[cell_keys[int(cell_idx)]]
            seed_effects = []
            for _ in range(len(members)):
                pbir, gift = members[int(rng.integers(0, len(members)))]
                indices = _circular_block_indices(
                    len(pbir), block_length, rng)
                if metric == "max_drawdown_improvement":
                    effect = statistic(gift[indices]) - statistic(pbir[indices])
                else:
                    effect = statistic(pbir[indices]) - statistic(gift[indices])
                seed_effects.append(float(effect))
            cell_effects.append(float(np.mean(seed_effects)))
        boot[sample_idx] = float(np.mean(cell_effects))
    return {
        "resamples": resamples,
        "block_length": block_length,
        "ci95": [float(np.percentile(boot, 2.5)),
                 float(np.percentile(boot, 97.5))],
        "probability_positive": float(np.mean(boot > 0.0)),
    }


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _make_figures(
        cells: list[dict[str, Any]], inference: dict[str, Any], output: Path,
) -> list[Path]:
    confirmatory = [row for row in cells if row["confirmatory"]]
    labels = [f"{'D1' if row['dataset'] == DATASETS[0] else 'D2'}-{row['window']}"
              for row in confirmatory]
    values = np.asarray([row["delta_sharpe"] for row in confirmatory])
    errors = np.asarray([row["seed_sd_sharpe"] for row in confirmatory])

    fig, ax = plt.subplots(figsize=(8.6, 5.6))
    y = np.arange(len(values))
    ax.errorbar(values, y, xerr=errors, fmt="o", color="#2F5D8C",
                ecolor="#9BB3C9", capsize=3, label="cell mean ± seed SD")
    bootstrap = inference.get("bootstrap", {}).get("sharpe")
    if bootstrap:
        point = inference["primary"]["mean_delta_sharpe"]
        low, high = bootstrap["ci95"]
        ax.errorbar([point], [len(values) + 0.4],
                    xerr=[[point - low], [high - point]], fmt="D",
                    color="#B5653A", capsize=4, label="hierarchical block bootstrap")
        labels.append("Overall")
        y = np.append(y, len(values) + 0.4)
    ax.axvline(0.0, color="#6F7D88", linewidth=1.0)
    ax.set_yticks(y, labels)
    ax.set_xlabel("Δ Sharpe (PBIR-GIFT − Pure GIFT)")
    figure_title = (
        "Fixed-code controlled replication cells"
        if inference.get("analysis_role") == "controlled_replication" else
        "Confirmatory non-overlapping cells")
    ax.set_title(figure_title, loc="left",
                 fontsize=14, fontweight="semibold")
    ax.grid(axis="x", color="#D8DEE4", alpha=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    forest_paths = []
    for suffix, kwargs in (("png", {"dpi": 300}), ("pdf", {})):
        path = output / f"primary_sharpe_forest.{suffix}"
        fig.savefig(path, bbox_inches="tight", facecolor="white", **kwargs)
        forest_paths.append(path)
    plt.close(fig)

    matrix = np.full((2, 6), np.nan)
    for row in cells:
        dataset_idx = DATASETS.index(row["dataset"])
        window_idx = int(row["window"][1:]) - 1
        matrix[dataset_idx, window_idx] = row["delta_sharpe"]
    bound = float(np.nanmax(np.abs(matrix))) or 1.0
    fig, ax = plt.subplots(figsize=(9.0, 3.4))
    image = ax.imshow(matrix, cmap="RdBu", vmin=-bound, vmax=bound, aspect="auto")
    ax.set_xticks(range(6), [f"W{i}" for i in range(1, 7)])
    ax.set_yticks([0, 1], ["D1", "D2"])
    ax.set_title("PBIR-GIFT minus Pure GIFT Sharpe by cell", loc="left",
                 fontsize=13, fontweight="semibold")
    for i in range(2):
        for j in range(6):
            if np.isfinite(matrix[i, j]):
                ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center",
                        color="white" if abs(matrix[i, j]) > bound * 0.55 else "#25313C")
    fig.colorbar(image, ax=ax, label="Δ Sharpe")
    fig.tight_layout()
    heatmap_paths = []
    for suffix, kwargs in (("png", {"dpi": 300}), ("pdf", {})):
        path = output / f"all_window_sharpe_heatmap.{suffix}"
        fig.savefig(path, bbox_inches="tight", facecolor="white", **kwargs)
        heatmap_paths.append(path)
    plt.close(fig)
    return forest_paths + heatmap_paths


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the preregistered paired Stage-B statistical analysis.")
    parser.add_argument("--results-dir", default="results/stage_b_final")
    parser.add_argument("--output-dir")
    parser.add_argument("--expected-seeds", type=int, default=3)
    parser.add_argument("--resamples", type=int, default=10000)
    parser.add_argument("--block-length", type=int, default=20)
    parser.add_argument("--random-seed", type=int, default=20260823)
    parser.add_argument(
        "--analysis-role",
        choices=["confirmatory", "controlled_replication"],
        default="confirmatory")
    args = parser.parse_args()
    root = Path(args.results_dir)
    if not root.is_absolute():
        root = PROJECT_DIR / root
    output = Path(args.output_dir) if args.output_dir else root / "aggregate" / "statistics"
    if not output.is_absolute():
        output = PROJECT_DIR / output
    output.mkdir(parents=True, exist_ok=True)

    rows = load_complete_rows(root)
    if not rows:
        raise FileNotFoundError(f"No complete Stage-B paired results under {root}")
    cells = collapse_cells(rows)
    _write_csv(cells, output / "cell_level_paired_deltas.csv")
    confirmatory = [row for row in cells if row["confirmatory"]]
    expected_cells = len(DATASETS) * len(PRIMARY_WINDOWS)
    complete_primary = len(confirmatory) == expected_cells
    seed_ready = complete_primary and all(
        row["n_seeds"] >= args.expected_seeds for row in confirmatory)

    raw_p = {}
    metric_summary = {}
    for metric in METRICS:
        values = np.asarray([row[f"delta_{metric}"] for row in confirmatory])
        raw_p[metric] = _exact_sign_flip(values, alternative="greater")
        metric_summary[metric] = {
            "mean_cell_delta": float(np.mean(values)),
            "median_cell_delta": float(np.median(values)),
            "cell_win_rate": float(np.mean(values > 0)),
            "hedges_g_paired": _hedges_g_paired(values),
            "exact_one_sided_sign_flip_p": raw_p[metric],
        }
    adjusted = _holm(raw_p)
    for metric in metric_summary:
        metric_summary[metric]["holm_adjusted_p"] = adjusted[metric]

    bootstrap = {}
    for index, metric in enumerate(METRICS):
        value = hierarchical_block_bootstrap(
            rows, metric, args.resamples, args.block_length,
            args.random_seed + index)
        if value is not None:
            bootstrap[metric] = value

    dataset_means = {
        dataset: float(np.mean([
            row["delta_sharpe"] for row in confirmatory
            if row["dataset"] == dataset]))
        for dataset in DATASETS
    }
    window_means = {
        window: float(np.mean([
            row["delta_sharpe"] for row in confirmatory
            if row["window"] == window]))
        for window in PRIMARY_WINDOWS
    }
    worst_mdd_deterioration = max(
        -float(row["delta_max_drawdown_improvement"])
        for row in confirmatory)
    sharpe_ci = bootstrap.get("sharpe", {}).get("ci95")
    support = bool(
        seed_ready
        and sharpe_ci is not None and sharpe_ci[0] > 0
        and all(value > 0 for value in dataset_means.values())
        and sum(value > 0 for value in window_means.values()) >= 2
        and worst_mdd_deterioration <= 5.0)

    interpretation = (
        ("MEETS the controlled-replication decision gate" if support else
         "DOES NOT MEET the controlled-replication decision gate")
        if args.analysis_role == "controlled_replication" else
        ("SUPPORTED under the predeclared rule" if support else
         "NOT SUPPORTED under the predeclared rule"))
    inference = {
        "analysis_role": args.analysis_role,
        "analysis_readiness": {
            "complete_primary_cells": complete_primary,
            "expected_primary_cells": expected_cells,
            "expected_seeds": args.expected_seeds,
            "seed_ready": seed_ready,
            "daily_returns_available": "sharpe" in bootstrap,
        },
        "primary": {
            "mean_delta_sharpe": metric_summary["sharpe"]["mean_cell_delta"],
            "dataset_means": dataset_means,
            "window_means": window_means,
            "worst_cell_mdd_deterioration_percentage_points": (
                worst_mdd_deterioration),
            "predeclared_support": support,
        },
        "metrics": metric_summary,
        "bootstrap": bootstrap,
        "interpretation": interpretation,
        "caveats": [
            "W2/W4/W6 are overlapping robustness windows, not independent units.",
            "D1 and D2 share AMZN, MSFT, and JNJ.",
            "A non-significant result is not evidence of equivalence.",
        ],
    }
    (output / "confirmatory_inference.json").write_text(
        json.dumps(inference, indent=2), encoding="utf-8")

    markdown = [
        ("# PBIR-GIFT Fixed-Code Controlled Replication\n"
         if args.analysis_role == "controlled_replication" else
         "# PBIR-GIFT Confirmatory Analysis\n"),
        f"- Readiness: `{json.dumps(inference['analysis_readiness'])}`",
        f"- Decision: **{inference['interpretation']}**",
        f"- Mean ΔSharpe: {inference['primary']['mean_delta_sharpe']:.4f}",
        f"- Dataset means: `{dataset_means}`",
        f"- Non-overlapping window means: `{window_means}`",
        f"- Worst cell MDD deterioration: {worst_mdd_deterioration:.3f} pp",
        "\n## Metric table\n",
        "| Metric | Mean delta | Win rate | Hedges g | Exact p | Holm p | Block-bootstrap 95% CI |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for metric, values in metric_summary.items():
        ci = bootstrap.get(metric, {}).get("ci95")
        markdown.append(
            f"| {metric} | {values['mean_cell_delta']:.4f} | "
            f"{values['cell_win_rate']:.1%} | "
            f"{values['hedges_g_paired'] if values['hedges_g_paired'] is not None else 'NA'} | "
            f"{values['exact_one_sided_sign_flip_p']:.4f} | "
            f"{values['holm_adjusted_p']:.4f} | {ci if ci else 'unavailable'} |")
    markdown.extend(["\n## Caveats\n"] + [f"- {item}" for item in inference["caveats"]])
    (output / "confirmatory_analysis.md").write_text(
        "\n".join(markdown) + "\n", encoding="utf-8")
    figures = _make_figures(cells, inference, output)
    print(f"Inference: {output / 'confirmatory_inference.json'}")
    print(f"Figures: {len(figures)} files under {output}")
    print(f"Decision: {inference['interpretation']}")
    return 0 if seed_ready else 2


if __name__ == "__main__":
    raise SystemExit(main())
