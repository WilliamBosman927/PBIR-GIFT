"""Six-method comparison chart for one frozen-config final cell."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


METHODS = (
    ("equal_weight", "Equal Weight", "#8D99A6"),
    ("ppo_test_only", "PPO-TestOnly", "#6F7C45"),
    ("ppo_train_plus_test", "PPO-Train+FT", "#4F7C6B"),
    ("state_gift", "State-GIFT", "#8B6F9E"),
    ("pure_gift", "Pure GIFT", "#B5653A"),
    ("pbir", "PBIR-GIFT", "#2F5D8C"),
)
METRICS = (
    ("test_sharpe", "Sharpe ratio"),
    ("test_sortino", "Sortino ratio"),
    ("test_total_return", "Total return"),
    ("test_max_drawdown", "Maximum drawdown ↓"),
)


def generate_full_method_figure(
        summary_json: str | Path,
        output_dir: str | Path | None = None,
) -> list[Path]:
    summary_path = Path(summary_json)
    row = json.loads(summary_path.read_text(encoding="utf-8"))
    available = [item for item in METHODS
                 if f"{item[0]}_test_sharpe" in row]
    if len(available) < 3:
        return []
    output_root = Path(output_dir) if output_dir else summary_path.parent / "figures"
    output_root.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(2, 2, figsize=(13.0, 8.0))
    figure.suptitle(
        "Final method comparison under a frozen protocol",
        x=0.065, y=0.975, ha="left", fontsize=16, fontweight="semibold",
        color="#25313C")
    figure.text(
        0.065, 0.938,
        f"{row['dataset']} | {row['window']} | seed {row['seed']}",
        ha="left", fontsize=9.5, color="#77828C")

    for axis, (metric, title) in zip(axes.ravel(), METRICS):
        labels = [item[1] for item in available]
        values = [float(row[f"{item[0]}_{metric}"]) for item in available]
        colors = [item[2] for item in available]
        bars = axis.bar(range(len(values)), values, color=colors, width=0.72)
        axis.set_title(title, loc="left", fontsize=11, color="#25313C")
        axis.set_xticks(range(len(labels)), labels, rotation=23, ha="right")
        axis.grid(axis="y", color="#D8DEE4", linewidth=0.8, alpha=0.8)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
        axis.spines[["left", "bottom"]].set_color("#77828C")
        axis.tick_params(labelsize=8.5, colors="#25313C")
        axis.margins(y=0.23)
        for bar, value in zip(bars, values):
            axis.annotate(
                f"{value:.2f}",
                (bar.get_x() + bar.get_width() / 2, value),
                xytext=(0, 3 if value >= 0 else -13),
                textcoords="offset points", ha="center",
                va="bottom" if value >= 0 else "top", fontsize=7.8)
    figure.subplots_adjust(
        left=0.065, right=0.98, top=0.89, bottom=0.11,
        hspace=0.35, wspace=0.20)

    stem = f"{row['dataset']}_{row['window']}_seed_{row['seed']}_six_method"
    written = []
    for suffix, kwargs in (("png", {"dpi": 300}), ("pdf", {})):
        path = output_root / f"{stem}.{suffix}"
        figure.savefig(path, bbox_inches="tight", facecolor="white", **kwargs)
        written.append(path)
    plt.close(figure)
    return written
