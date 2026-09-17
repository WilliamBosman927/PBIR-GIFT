"""Publication-style chart for one frozen-config Stage-B paired cell."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PBIR_COLOR = "#2F5D8C"
GIFT_COLOR = "#B5653A"
CHARCOAL = "#25313C"
GREY = "#77828C"
GRID = "#D8DEE4"

METRICS = (
    ("test_sharpe", "Sharpe ratio"),
    ("test_sortino", "Sortino ratio"),
    ("test_total_return", "Total return"),
    ("test_max_drawdown", "Maximum drawdown"),
)


def generate_final_pair_figure(
        summary_json: str | Path,
        output_dir: str | Path | None = None,
) -> list[Path]:
    summary_path = Path(summary_json)
    with summary_path.open("r", encoding="utf-8") as handle:
        row = json.load(handle)
    output_root = Path(output_dir) if output_dir else summary_path.parent / "figures"
    output_root.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(2, 2, figsize=(10.4, 7.4))
    figure.patch.set_facecolor("white")
    figure.suptitle(
        "Frozen-config final comparison", x=0.07, y=0.965,
        ha="left", fontsize=16, fontweight="semibold", color=CHARCOAL)
    figure.text(
        0.07, 0.925,
        f"{row['dataset']} | {row['window']} | seed {row['seed']} | "
        "PBIR-GIFT vs Pure GIFT",
        ha="left", fontsize=10, color=GREY)

    for axis, (metric, label) in zip(axes.ravel(), METRICS):
        values = np.array([
            float(row[f"pbir_{metric}"]),
            float(row[f"pure_gift_{metric}"]),
        ])
        bars = axis.bar(
            [0, 1], values, width=0.58,
            color=[PBIR_COLOR, GIFT_COLOR], edgecolor="white")
        axis.set_title(label, loc="left", fontsize=11, color=CHARCOAL)
        axis.set_xticks([0, 1], ["PBIR-GIFT", "Pure GIFT"])
        axis.spines[["top", "right"]].set_visible(False)
        axis.spines[["left", "bottom"]].set_color(GREY)
        axis.grid(axis="y", color=GRID, linewidth=0.8, alpha=0.8)
        axis.set_axisbelow(True)
        axis.tick_params(axis="both", labelsize=9, colors=CHARCOAL)
        axis.margins(y=0.25)
        for bar, value in zip(bars, values):
            vertical = 3 if value >= 0 else -14
            axis.annotate(
                f"{value:.3f}",
                (bar.get_x() + bar.get_width() / 2, value),
                xytext=(0, vertical), textcoords="offset points",
                ha="center", va="bottom" if value >= 0 else "top",
                fontsize=9, color=CHARCOAL)

    figure.text(
        0.07, 0.025,
        "Both methods use the same frozen PPO configuration, dataset window, "
        "seed, and transaction-cost setting.",
        ha="left", fontsize=8.8, color=GREY)
    figure.subplots_adjust(
        left=0.08, right=0.97, top=0.86, bottom=0.09,
        hspace=0.34, wspace=0.24)

    stem = f"{row['dataset']}_{row['window']}_seed_{row['seed']}_final_pair"
    written: list[Path] = []
    for extension, kwargs in (("png", {"dpi": 300}), ("pdf", {})):
        path = output_root / f"{stem}.{extension}"
        figure.savefig(path, bbox_inches="tight", facecolor="white", **kwargs)
        written.append(path)
    plt.close(figure)
    return written
