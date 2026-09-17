"""Aggregate completed Stage-B cells and create the cross-window figure."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from hparam_sweep_runner import PROJECT_DIR


METRICS = (
    ("test_sharpe", "Sharpe ratio"),
    ("test_sortino", "Sortino ratio"),
    ("test_total_return", "Total return"),
    ("test_max_drawdown", "Maximum drawdown"),
)
METHODS = (("pbir", "PBIR-GIFT", "#2F5D8C"),
           ("pure_gift", "Pure GIFT", "#B5653A"))
DATASETS = (("portfolio_5stocks", "D1", "-"),
            ("portfolio_5stocks2", "D2", "--"))


def _load_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(root.glob("*/W[1-6]/seed_*/final_pair_summary.json")):
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if row.get("status") not in {"completed", "skipped_existing"}:
            continue
        row["source_file"] = str(path)
        rows.append(row)
    return rows


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _finite_values(rows: list[dict[str, Any]], key: str) -> np.ndarray:
    values = []
    for row in rows:
        try:
            value = float(row[key])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(value):
            values.append(value)
    return np.asarray(values, dtype=float)


def generate_aggregate(rows: list[dict[str, Any]], output_dir: Path) -> list[Path]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["dataset"]), str(row["window"]))].append(row)

    figure, axes = plt.subplots(2, 2, figsize=(12.2, 8.2), sharex=True)
    figure.patch.set_facecolor("white")
    figure.suptitle(
        "Frozen-config PBIR-GIFT vs Pure GIFT across rolling windows",
        x=0.075, y=0.975, ha="left", fontsize=16, fontweight="semibold",
        color="#25313C")
    figure.text(
        0.075, 0.94,
        "Lines are seed means; shaded bands show ±1 SD when multiple seeds exist.",
        ha="left", fontsize=9.5, color="#77828C")

    x = np.arange(1, 7)
    for axis, (metric, title) in zip(axes.ravel(), METRICS):
        for dataset, dataset_label, line_style in DATASETS:
            for method, method_label, color in METHODS:
                means: list[float] = []
                deviations: list[float] = []
                for window in (f"W{i}" for i in x):
                    values = _finite_values(
                        grouped.get((dataset, window), []), f"{method}_{metric}")
                    means.append(float(np.mean(values)) if values.size else np.nan)
                    deviations.append(
                        float(np.std(values, ddof=1)) if values.size > 1 else 0.0)
                mean_array = np.asarray(means)
                sd_array = np.asarray(deviations)
                label = f"{method_label} · {dataset_label}"
                axis.plot(
                    x, mean_array, line_style, color=color, marker="o",
                    linewidth=1.8, markersize=4.5, label=label)
                if np.any(sd_array > 0):
                    axis.fill_between(
                        x, mean_array - sd_array, mean_array + sd_array,
                        color=color, alpha=0.10, linewidth=0)
        axis.set_title(title, loc="left", fontsize=11, color="#25313C")
        axis.set_xticks(x, [f"W{i}" for i in x])
        axis.axhline(0.0, color="#9AA4AD", linewidth=0.7)
        axis.grid(axis="y", color="#D8DEE4", linewidth=0.8, alpha=0.8)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
        axis.spines[["left", "bottom"]].set_color("#77828C")
        axis.tick_params(labelsize=9, colors="#25313C")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles, labels, loc="lower center", ncol=4, frameon=False,
        bbox_to_anchor=(0.5, 0.015), fontsize=9)
    figure.subplots_adjust(
        left=0.075, right=0.98, top=0.89, bottom=0.10,
        hspace=0.27, wspace=0.20)

    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for extension, kwargs in (("png", {"dpi": 300}), ("pdf", {})):
        path = output_dir / f"stage_b_cross_window_comparison.{extension}"
        figure.savefig(path, bbox_inches="tight", facecolor="white", **kwargs)
        written.append(path)
    plt.close(figure)
    return written


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Aggregate all completed Stage-B dataset/window/seed cells.")
    parser.add_argument("--results-dir", default="results/stage_b_final")
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("--expected-seeds", type=int, default=1)
    parser.add_argument(
        "--expected-windows", nargs="+",
        choices=[f"W{i}" for i in range(1, 7)],
        default=[f"W{i}" for i in range(1, 7)],
        help=("Windows required by completeness checking. Stage-C should use "
              "--expected-windows W1 W3 W5."))
    args = parser.parse_args()
    root = Path(args.results_dir)
    if not root.is_absolute():
        root = PROJECT_DIR / root
    rows = _load_rows(root)
    if not rows:
        raise FileNotFoundError(f"No completed Stage-B cells found under {root}")

    observed = {(row["dataset"], row["window"], row["seed"]) for row in rows}
    required_rows = [
        row for row in rows if row.get("window") in args.expected_windows]
    required_observed = {
        (row["dataset"], row["window"], row["seed"])
        for row in required_rows}
    expected_cells = len(DATASETS) * len(args.expected_windows) * args.expected_seeds
    if args.require_complete:
        counts: dict[tuple[str, str], set[int]] = defaultdict(set)
        for dataset, window, seed in required_observed:
            counts[(str(dataset), str(window))].add(int(seed))
        problems = []
        for dataset, _, _ in DATASETS:
            for window in args.expected_windows:
                count = len(counts.get((dataset, window), set()))
                if count != args.expected_seeds:
                    problems.append(f"{dataset}/{window}: {count} seeds")
        if len(required_observed) != expected_cells or problems:
            raise RuntimeError(
                f"Expected {expected_cells} unique required cells; found "
                f"{len(required_observed)}. Problems: {problems}")

    aggregate_dir = root / "aggregate"
    aggregate_dir.mkdir(parents=True, exist_ok=True)
    csv_path = aggregate_dir / "stage_b_all_cells.csv"
    _write_csv(rows, csv_path)
    figures = generate_aggregate(rows, aggregate_dir)
    print(f"Aggregated {len(observed)} unique cells: {csv_path}")
    print(f"Figures: {figures[0]} and {figures[1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
