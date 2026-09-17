"""Analyze a complete six-window GIFT reproduction without modifying Stage-D.

The program combines frozen Pure-GIFT results from Stage-B, odd-window
baselines from locked Stage-D, and even-window baselines from Stage-E.  All
new tables, figures, and reports are written below the Stage-E output tree.
"""

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
import yaml

from analyze_final_results import (
    _circular_block_indices,
    _exact_sign_flip,
    _hedges_g_paired,
)
from hparam_sweep_runner import PROJECT_DIR
from lock_stage_b_results import verify_lock as verify_stage_b_lock
from lock_stage_d_results import verify as verify_stage_d_lock
from metrics import sharpe_ratio
from run_stage_e_full_window_baselines import _source_config


DATASETS = ("portfolio_5stocks", "portfolio_5stocks2")
WINDOWS = ("W1", "W2", "W3", "W4", "W5", "W6")
ODD_WINDOWS = {"W1", "W3", "W5"}
EVEN_WINDOWS = {"W2", "W4", "W6"}
SEEDS = (42, 123, 456)
BASELINE_METHODS = ("ppo_test_only", "ppo_train_plus_test", "equal_weight")
METHOD_LABELS = {
    "pure_gift": "Pure GIFT",
    "ppo_test_only": "PPO-TestOnly",
    "ppo_train_plus_test": "PPO-Train+FineTune",
    "equal_weight": "Equal Weight",
}
METRICS = (
    "test_total_return",
    "test_sharpe",
    "test_sortino",
    "test_max_drawdown",
    "test_calmar",
)
HIGHER_IS_BETTER = {
    "test_total_return": True,
    "test_sharpe": True,
    "test_sortino": True,
    "test_max_drawdown": False,
    "test_calmar": True,
}

# GIFT paper, Table 2, Light Mix.  Values are kept here only as a transparent
# external-reference overlay; all statistical decisions use local raw returns.
PAPER_LIGHT_MIX = {
    "pure_gift": {
        "W1": (2.41, 0.45, 0.45, 10.04, 0.70),
        "W2": (12.16, 2.11, 2.10, 6.74, 4.26),
        "W3": (-13.61, -1.25, -1.13, 21.23, -1.51),
        "W4": (-9.17, -0.72, -0.77, 13.70, -1.43),
        "W5": (20.40, 2.71, 2.98, 10.05, 4.63),
        "W6": (5.80, 1.11, 1.07, 8.89, 1.65),
    },
    "ppo_test_only": {
        "W1": (-1.78, -0.14, -0.14, 11.93, -0.22),
        "W2": (12.13, 2.08, 2.12, 7.65, 3.75),
        "W3": (-32.70, -2.05, -1.72, 35.98, -2.41),
        "W4": (-23.46, -2.30, -2.42, 26.35, -2.32),
        "W5": (13.62, 1.99, 2.32, 11.33, 2.85),
        "W6": (5.28, 0.79, 0.77, 12.28, 1.16),
    },
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-b-dir", default="results/stage_b_final")
    parser.add_argument(
        "--stage-b-lock",
        default="results/stage_b_lock/stage_b_lock_manifest.json")
    parser.add_argument(
        "--stage-d-dir", default="results/stage_d_ppo_baselines")
    parser.add_argument(
        "--stage-d-lock",
        default="results/stage_d_lock/stage_d_lock_manifest.json")
    parser.add_argument(
        "--stage-e-dir", default="results/stage_e_full_window_baselines")
    parser.add_argument(
        "--output-dir",
        default="results/stage_e_full_window_baselines/analysis_bundle")
    parser.add_argument("--resamples", type=int, default=10000)
    parser.add_argument("--block-length", type=int, default=20)
    return parser.parse_args()


def _resolve(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_DIR / path).resolve()


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write an empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(rows[0])
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _verify_locked_source(lock_path: Path, root: Path, stage: str) -> dict:
    if not lock_path.is_file():
        raise FileNotFoundError(f"{stage} lock manifest not found: {lock_path}")
    if stage == "Stage-B":
        verify_stage_b_lock(lock_path)
    else:
        verify_stage_d_lock(lock_path)
    manifest = _read(lock_path)
    if Path(str(manifest["source_root"])).resolve() != root:
        raise ValueError(f"{stage} lock points to a different source root")
    return manifest


def _complete_stage_e_cells(root: Path) -> None:
    expected = {
        (dataset, window, seed)
        for dataset in DATASETS for window in EVEN_WINDOWS for seed in SEEDS
    }
    observed = set()
    for path in root.glob("*/W*/seed_*/baseline_summary.json"):
        value = _read(path)
        if value.get("status") != "completed" or value.get("llm_calls") != 0:
            continue
        observed.add((
            path.parents[2].name,
            path.parents[1].name,
            int(path.parent.name.removeprefix("seed_")),
        ))
    if observed != expected:
        missing = sorted(expected - observed)
        unexpected = sorted(observed - expected)
        raise RuntimeError(
            "Stage-E is incomplete; six-window analysis refused. "
            f"Missing={missing}; unexpected={unexpected}")


def _gift_result(cell: Path) -> tuple[dict[str, Any], np.ndarray, list[str]]:
    value = _read(cell / "methods" / "pure_gift" / "final_comparison.json")
    result = value["test_result"]
    returns = np.asarray(value["daily_returns"]["method"], dtype=float)
    period = [str(item) for item in value["eval_period"]]
    return result, returns, period


def _baseline_result(
        cell: Path, method: str,
) -> tuple[dict[str, Any], np.ndarray, list[str]]:
    value = _read(cell / "methods" / f"{method}.json")
    if value.get("status") != "completed" or value.get("llm_call_attempts") != 0:
        raise ValueError(f"Invalid baseline result: {cell}/{method}")
    result = value["test_result"]
    returns = np.asarray(value["daily_returns"], dtype=float)
    period = [str(item) for item in value["eval_period"]]
    return result, returns, period


def _bootstrap_sharpe(
        pairs: dict[tuple[str, str], list[tuple[np.ndarray, np.ndarray]]],
        resamples: int,
        block_length: int,
        random_seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(random_seed)
    cells = sorted(pairs)
    if len(cells) not in {6, 12}:
        raise ValueError(f"Expected 6 or 12 cells for bootstrap, found {len(cells)}")
    boot = np.empty(resamples, dtype=float)
    for sample in range(resamples):
        cell_effects = []
        for selected in rng.choice(len(cells), size=len(cells), replace=True):
            members = pairs[cells[int(selected)]]
            seed_effects = []
            for _ in range(len(members)):
                first, second = members[int(rng.integers(0, len(members)))]
                n = min(len(first), len(second))
                indices = _circular_block_indices(n, block_length, rng)
                seed_effects.append(
                    sharpe_ratio(first[:n][indices])
                    - sharpe_ratio(second[:n][indices]))
            cell_effects.append(float(np.mean(seed_effects)))
        boot[sample] = float(np.mean(cell_effects))
    return {
        "resamples": resamples,
        "block_length": block_length,
        "ci95": [float(np.percentile(boot, 2.5)),
                 float(np.percentile(boot, 97.5))],
        "probability_positive": float(np.mean(boot > 0)),
    }


def _contrast_summary(
        deltas: dict[tuple[str, str], list[float]],
        pairs: dict[tuple[str, str], list[tuple[np.ndarray, np.ndarray]]],
        resamples: int,
        block_length: int,
        random_seed: int,
) -> dict[str, Any]:
    cell_means = {key: float(np.mean(values)) for key, values in deltas.items()}
    values = np.asarray([cell_means[key] for key in sorted(cell_means)])
    dataset_means = {
        dataset: float(np.mean([
            value for (seen_dataset, _), value in cell_means.items()
            if seen_dataset == dataset]))
        for dataset in DATASETS
    }
    window_means = {
        window: float(np.mean([
            value for (_, seen_window), value in cell_means.items()
            if seen_window == window]))
        for window in WINDOWS
    }
    return {
        "mean_cell_delta_sharpe": float(np.mean(values)),
        "median_cell_delta_sharpe": float(np.median(values)),
        "positive_cells": int(np.sum(values > 0)),
        "cell_win_rate": float(np.mean(values > 0)),
        "seed_run_win_rate": float(np.mean([
            value > 0 for members in deltas.values() for value in members])),
        "hedges_g_paired": _hedges_g_paired(values),
        "exact_one_sided_sign_flip_p": _exact_sign_flip(values),
        "dataset_means": dataset_means,
        "window_means": window_means,
        "bootstrap": _bootstrap_sharpe(
            pairs, resamples, block_length, random_seed),
    }


def _paper_replication(
        cell_metric_means: dict[tuple[str, str, str, str], float],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    composite_wins = 0
    sharpe_wins = 0
    for window in WINDOWS:
        metric_wins = 0
        row: dict[str, Any] = {"dataset": DATASETS[0], "window": window}
        for metric in METRICS:
            gift = cell_metric_means[(DATASETS[0], window, "pure_gift", metric)]
            ppo = cell_metric_means[(
                DATASETS[0], window, "ppo_test_only", metric)]
            won = gift > ppo if HIGHER_IS_BETTER[metric] else gift < ppo
            metric_wins += int(won)
            row[f"pure_gift_{metric}"] = gift
            row[f"ppo_test_only_{metric}"] = ppo
            row[f"gift_wins_{metric}"] = won
        row["gift_metric_wins"] = metric_wins
        row["gift_wins_at_least_4_of_5"] = metric_wins >= 4
        composite_wins += int(metric_wins >= 4)
        sharpe_wins += int(row["gift_wins_test_sharpe"])
        rows.append(row)
    sharpe_deltas = [
        cell_metric_means[(DATASETS[0], window, "pure_gift", "test_sharpe")]
        - cell_metric_means[(
            DATASETS[0], window, "ppo_test_only", "test_sharpe")]
        for window in WINDOWS
    ]
    decision = {
        "scope": "portfolio_5stocks (Light Mix), six windows",
        "comparison": "Pure GIFT versus PPO-TestOnly",
        "mean_delta_sharpe": float(np.mean(sharpe_deltas)),
        "positive_sharpe_windows": sharpe_wins,
        "windows_winning_at_least_4_of_5_metrics": composite_wins,
        "directional_replication_supported": bool(
            np.mean(sharpe_deltas) > 0
            and sharpe_wins >= 4
            and composite_wins >= 4),
        "rule": (
            "mean Sharpe delta > 0, at least 4/6 positive Sharpe windows, "
            "and at least 4/6 windows winning >=4 of 5 reported metrics"),
    }
    return decision, rows


def _paper_similarity(
        cell_metric_means: dict[tuple[str, str, str, str], float],
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for method in ("pure_gift", "ppo_test_only"):
        output[method] = {}
        for metric_index, metric in enumerate(METRICS):
            paper = np.asarray([
                PAPER_LIGHT_MIX[method][window][metric_index]
                for window in WINDOWS], dtype=float)
            local = np.asarray([
                cell_metric_means[(DATASETS[0], window, method, metric)]
                for window in WINDOWS], dtype=float)
            correlation = None
            if np.std(paper) > 0 and np.std(local) > 0:
                correlation = float(np.corrcoef(paper, local)[0, 1])
            output[method][metric] = {
                "pearson_window_pattern": correlation,
                "mean_absolute_error": float(np.mean(np.abs(local - paper))),
                "paper_values": paper.tolist(),
                "local_seed_means": local.tolist(),
            }
    return output


def main() -> int:
    args = _parse_args()
    if args.resamples < 100:
        raise ValueError("--resamples must be at least 100")
    if args.block_length < 1:
        raise ValueError("--block-length must be positive")
    stage_b = _resolve(args.stage_b_dir)
    stage_d = _resolve(args.stage_d_dir)
    stage_e = _resolve(args.stage_e_dir)
    output = _resolve(args.output_dir)
    if stage_d == output or stage_d in output.parents:
        raise ValueError("Stage-E analysis must never write into locked Stage-D")
    if not (output == stage_e or stage_e in output.parents):
        raise ValueError("Stage-E analysis output must stay inside Stage-E root")

    b_lock = _verify_locked_source(
        _resolve(args.stage_b_lock), stage_b, "Stage-B")
    d_lock = _verify_locked_source(
        _resolve(args.stage_d_lock), stage_d, "Stage-D")
    _complete_stage_e_cells(stage_e)
    output.mkdir(parents=True, exist_ok=True)
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    expected = [
        (dataset, window, seed)
        for dataset in DATASETS for window in WINDOWS for seed in SEEDS
    ]
    metric_rows: list[dict[str, Any]] = []
    metric_values: dict[tuple[str, str, str, str], list[float]] = defaultdict(list)
    sharpe_deltas: dict[
        str, dict[tuple[str, str], list[float]]
    ] = {method: defaultdict(list) for method in BASELINE_METHODS}
    return_pairs: dict[
        str, dict[tuple[str, str], list[tuple[np.ndarray, np.ndarray]]]
    ] = {method: defaultdict(list) for method in BASELINE_METHODS}

    for dataset, window, seed in expected:
        b_cell = stage_b / dataset / window / f"seed_{seed}"
        baseline_root = stage_d if window in ODD_WINDOWS else stage_e
        baseline_cell = baseline_root / dataset / window / f"seed_{seed}"
        _, stage_b_config, _, _ = _source_config(
            stage_b, dataset, window, seed)
        baseline_config = yaml.safe_load(
            (baseline_cell / "resolved_config.yaml").read_text(encoding="utf-8"))
        for field in ("ppo", "data"):
            if baseline_config.get(field) != stage_b_config.get(field):
                raise ValueError(
                    f"{field} config mismatch: {dataset}/{window}/seed_{seed}")
        if (baseline_config.get("experiment", {}).get("test_period")
                != stage_b_config.get("experiment", {}).get("test_period")):
            raise ValueError(
                f"Test-period config mismatch: {dataset}/{window}/seed_{seed}")
        gift_result, gift_returns, gift_period = _gift_result(b_cell)
        sources: dict[str, tuple[dict[str, Any], np.ndarray]] = {
            "pure_gift": (gift_result, gift_returns)}
        for method in BASELINE_METHODS:
            result, returns, period = _baseline_result(baseline_cell, method)
            if period != gift_period:
                raise ValueError(
                    f"Evaluation-period mismatch: {dataset}/{window}/seed_{seed}/"
                    f"{method}: GIFT={gift_period}, baseline={period}")
            if len(returns) != len(gift_returns) or len(returns) < 22:
                raise ValueError(
                    f"Unaligned returns: {dataset}/{window}/seed_{seed}/{method}")
            if not np.all(np.isfinite(returns)):
                raise ValueError(f"Non-finite baseline returns: {baseline_cell}/{method}")
            sources[method] = (result, returns)
            key = (dataset, window)
            sharpe_deltas[method][key].append(
                float(gift_result["test_sharpe"] - result["test_sharpe"]))
            return_pairs[method][key].append((gift_returns, returns))
        if not np.all(np.isfinite(gift_returns)):
            raise ValueError(f"Non-finite GIFT returns: {b_cell}")
        for method, (result, _) in sources.items():
            row: dict[str, Any] = {
                "dataset": dataset,
                "window": window,
                "seed": seed,
                "method": method,
                "baseline_source_stage": (
                    "Stage-B" if method == "pure_gift"
                    else "Stage-D" if window in ODD_WINDOWS else "Stage-E"),
            }
            for metric in METRICS:
                value = float(result[metric])
                if not np.isfinite(value):
                    raise ValueError(
                        f"Non-finite metric: {dataset}/{window}/{seed}/{method}/{metric}")
                row[metric] = value
                metric_values[(dataset, window, method, metric)].append(value)
            metric_rows.append(row)

    cell_metric_means = {
        key: float(np.mean(values)) for key, values in metric_values.items()}
    cell_rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        for window in WINDOWS:
            for method in METHOD_LABELS:
                row: dict[str, Any] = {
                    "dataset": dataset,
                    "window": window,
                    "method": method,
                    "method_label": METHOD_LABELS[method],
                    "n_seeds": len(metric_values[(
                        dataset, window, method, "test_sharpe")]),
                }
                for metric in METRICS:
                    values = np.asarray(metric_values[(
                        dataset, window, method, metric)], dtype=float)
                    row[f"mean_{metric}"] = float(np.mean(values))
                    row[f"seed_sd_{metric}"] = float(np.std(values, ddof=1))
                cell_rows.append(row)

    contrast_summaries = {}
    for index, method in enumerate(BASELINE_METHODS):
        contrast_summaries[f"pure_gift_vs_{method}"] = _contrast_summary(
            sharpe_deltas[method], return_pairs[method], args.resamples,
            args.block_length, 20260903 + index)
    primary = contrast_summaries["pure_gift_vs_ppo_test_only"]
    strict_full_window_supported = bool(
        primary["bootstrap"]["ci95"][0] > 0
        and all(value > 0 for value in primary["dataset_means"].values())
        and primary["positive_cells"] >= 8)
    light_mix, win_rows = _paper_replication(cell_metric_means)
    paper_similarity = _paper_similarity(cell_metric_means)

    inference = {
        "analysis_role": "stage-e-six-window-gift-reproduction",
        "complete_seed_runs": len(expected),
        "independent_dataset_window_cells": 12,
        "seeds_per_cell": len(SEEDS),
        "baseline_provenance": {
            "odd_windows": "locked Stage-D",
            "even_windows": "Stage-E",
            "gift_all_windows": "locked Stage-B",
        },
        "source_lock_hashes": {
            "stage_b": b_lock["aggregate_sha256"],
            "stage_d": d_lock["aggregate_sha256"],
        },
        "primary_comparison": "Pure GIFT versus PPO-TestOnly",
        "strict_full_window_superiority_supported": strict_full_window_supported,
        "strict_rule": (
            "95% hierarchical block-bootstrap lower bound > 0, both dataset "
            "mean Sharpe deltas > 0, and at least 8/12 positive cell deltas"),
        "light_mix_directional_replication": light_mix,
        "contrasts": contrast_summaries,
        "paper_table_2_similarity": paper_similarity,
        "caveats": [
            "D1 and D2 share three stocks and are not independent universes.",
            "The inferential unit is the dataset-window cell mean, not each day.",
            "Stage-B Pure-GIFT code generation is pipeline-level and is not a "
            "fixed-code PBIR ablation.",
            "The paper overlay is descriptive; local raw returns drive inference.",
        ],
    }
    _write_json(output / "stage_e_inference.json", inference)
    _write_csv(output / "stage_e_seed_metrics.csv", metric_rows)
    _write_csv(output / "stage_e_cell_metrics.csv", cell_rows)
    _write_csv(output / "stage_e_paper_style_wins.csv", win_rows)

    colors = {
        "pure_gift": "#1261A0",
        "ppo_test_only": "#D1495B",
        "ppo_train_plus_test": "#7A5195",
        "equal_weight": "#4D9078",
    }
    figure, axes = plt.subplots(1, 2, figsize=(13.2, 5.2), sharey=True)
    x = np.arange(len(WINDOWS))
    for axis, dataset in zip(axes, DATASETS):
        for method in METHOD_LABELS:
            means = np.asarray([
                cell_metric_means[(dataset, window, method, "test_sharpe")]
                for window in WINDOWS])
            sds = np.asarray([
                np.std(metric_values[(dataset, window, method, "test_sharpe")],
                       ddof=1)
                for window in WINDOWS])
            axis.errorbar(
                x, means, yerr=sds, marker="o", linewidth=2.0, capsize=3,
                label=METHOD_LABELS[method], color=colors[method])
        axis.axhline(0, color="#55606A", linewidth=1.0, alpha=0.75)
        axis.set_xticks(x, WINDOWS)
        axis.set_title("D1 Light Mix" if dataset == DATASETS[0]
                       else "D2 Custom Portfolio", fontweight="bold")
        axis.set_xlabel("Rolling evaluation window")
        axis.grid(alpha=0.20)
    axes[0].set_ylabel("Out-of-sample Sharpe (mean ± seed SD)")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=4,
                  bbox_to_anchor=(0.5, -0.02))
    figure.suptitle("Complete six-window GIFT reproduction", fontweight="bold")
    figure.tight_layout(rect=(0, 0.08, 1, 0.95))
    for suffix in ("png", "pdf"):
        figure.savefig(figures / f"stage_e_six_window_sharpe.{suffix}",
                       dpi=240, bbox_inches="tight")
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(12.6, 4.8), sharey=True)
    for axis, method in zip(axes, ("pure_gift", "ppo_test_only")):
        paper = [PAPER_LIGHT_MIX[method][window][1] for window in WINDOWS]
        local = [cell_metric_means[(
            DATASETS[0], window, method, "test_sharpe")] for window in WINDOWS]
        axis.plot(x, paper, "--o", linewidth=2.0, color="#6C757D",
                  label="Paper Table 2")
        axis.plot(x, local, "-o", linewidth=2.2, color=colors[method],
                  label="Local reproduction")
        axis.axhline(0, color="#55606A", linewidth=1.0, alpha=0.75)
        axis.set_xticks(x, WINDOWS)
        axis.set_title(METHOD_LABELS[method], fontweight="bold")
        axis.set_xlabel("Light Mix window")
        axis.grid(alpha=0.20)
        axis.legend(frameon=False)
    axes[0].set_ylabel("Out-of-sample Sharpe")
    figure.suptitle("Original-paper pattern versus local Light Mix reproduction",
                    fontweight="bold")
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    for suffix in ("png", "pdf"):
        figure.savefig(figures / f"stage_e_light_mix_paper_replication.{suffix}",
                       dpi=240, bbox_inches="tight")
    plt.close(figure)

    primary_ci = primary["bootstrap"]["ci95"]
    decision_label = "SUPPORTED" if strict_full_window_supported else "NOT SUPPORTED"
    replication_label = (
        "SUPPORTED" if light_mix["directional_replication_supported"]
        else "NOT SUPPORTED")
    report = [
        "# Stage-E full-window reproduction report",
        "",
        "## Decisions",
        "",
        f"- Strict six-window GIFT superiority: **{decision_label}**.",
        f"- Light Mix directional reproduction of the paper: **{replication_label}**.",
        "",
        "These are empirical decisions, not indicators that the program did or "
        "did not run correctly.",
        "",
        "## Primary comparison",
        "",
        "The primary comparison is Stage-B Pure GIFT against PPO-TestOnly, which "
        "matches the paper's main PPO baseline protocol more closely than the "
        "additional PPO-Train+FineTune stress test.",
        "",
        f"Across 12 dataset-window cell means, ΔSharpe was "
        f"{primary['mean_cell_delta_sharpe']:+.3f}; "
        f"{primary['positive_cells']}/12 cells favored GIFT. The 95% hierarchical "
        f"block-bootstrap interval was [{primary_ci[0]:+.3f}, {primary_ci[1]:+.3f}], "
        f"and the exact one-sided sign-flip p-value was "
        f"{primary['exact_one_sided_sign_flip_p']:.4f}.",
        "",
        "## Light Mix reproduction rule",
        "",
        f"Mean ΔSharpe: {light_mix['mean_delta_sharpe']:+.3f}; positive Sharpe "
        f"windows: {light_mix['positive_sharpe_windows']}/6; windows where GIFT "
        f"won at least four of five paper metrics: "
        f"{light_mix['windows_winning_at_least_4_of_5_metrics']}/6.",
        "",
        "## Contrast summary",
        "",
        "| Contrast | Mean ΔSharpe | Positive cells | Hedges g | Exact p | 95% bootstrap CI |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, value in contrast_summaries.items():
        ci = value["bootstrap"]["ci95"]
        effect = value["hedges_g_paired"]
        effect_text = "NA" if effect is None else f"{effect:+.3f}"
        report.append(
            f"| {name} | {value['mean_cell_delta_sharpe']:+.3f} | "
            f"{value['positive_cells']}/12 | {effect_text} | "
            f"{value['exact_one_sided_sign_flip_p']:.4f} | "
            f"[{ci[0]:+.3f}, {ci[1]:+.3f}] |")
    report.extend([
        "",
        "## Claim boundary",
        "",
        "- Stage-E completes the missing even-window PPO and equal-weight controls; "
        "it does not rerun or alter locked Stage-B/Stage-D evidence.",
        "- The original-paper overlay is a descriptive pattern check. Differences "
        "in software, sampled LLM code, random seeds, and environment can prevent "
        "exact numerical reproduction.",
        "- GIFT-vs-PPO evidence does not establish PBIR-vs-GIFT superiority. That "
        "question remains governed by the locked fixed-code Stage-C comparison.",
        "- D2 overlaps D1 by three assets, so the two portfolios are not fully "
        "independent replications.",
    ])
    report_text = "\n".join(report) + "\n"
    (output / "analysis_report.md").write_text(report_text, encoding="utf-8")
    (output / "stage_e_analysis.md").write_text(report_text, encoding="utf-8")

    appendix = [
        "# Stage-E statistical appendix",
        "",
        "The primary unit is the dataset-window cell mean (12 cells, three seeds "
        "per cell). The hierarchical bootstrap resamples cells, then seeds within "
        "cells, then aligned daily returns with circular blocks. This preserves "
        "time dependence more faithfully than treating all days as independent.",
        "",
        f"Bootstrap replicates: {args.resamples:,}; block length: "
        f"{args.block_length}. The sign-flip test operates on the 12 cell-mean "
        "Sharpe differences. Effect size is paired Hedges g across the same cells.",
        "",
        "The strict rule was declared in code before Stage-E results existed. "
        "Light Mix directional replication is reported separately because it "
        "targets correspondence with the original six-window experiment rather "
        "than universal superiority across both portfolios.",
    ]
    (output / "stats_appendix.md").write_text(
        "\n".join(appendix) + "\n", encoding="utf-8")
    catalog = [
        "# Stage-E figure catalog",
        "",
        "## Figure 1 — Complete six-window Sharpe",
        "",
        "`figures/stage_e_six_window_sharpe.png` and `.pdf` show all four "
        "methods across W1-W6 for D1 and D2. Points are three-seed means and "
        "error bars are seed standard deviations.",
        "",
        "## Figure 2 — Paper-pattern reproduction",
        "",
        "`figures/stage_e_light_mix_paper_replication.png` and `.pdf` compare "
        "the paper's Light Mix Sharpe pattern with the local six-window means. "
        "This figure is descriptive and is not used as the significance test.",
    ]
    (output / "figure_catalog.md").write_text(
        "\n".join(catalog) + "\n", encoding="utf-8")

    print(f"Analysis: {output / 'stage_e_analysis.md'}")
    print(f"Figure: {figures / 'stage_e_six_window_sharpe.png'}")
    print(f"Strict supported: {strict_full_window_supported}")
    print("Light Mix directional replication supported: "
          f"{light_mix['directional_replication_supported']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
