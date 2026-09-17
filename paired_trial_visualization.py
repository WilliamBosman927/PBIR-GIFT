"""One publication-style figure for a paired PBIR-GIFT/Pure GIFT sweep."""

from __future__ import annotations

from pathlib import Path
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hparam_visualization import METRICS, PARAMETER_LABELS, _format_x


PBIR_COLOR = "#2F5D8C"
GIFT_COLOR = "#B5653A"
CHARCOAL = "#25313C"
GREY = "#77828C"
GRID = "#D8DEE4"


def generate_paired_overview(
        summary_csv: str | Path,
        output_dir: str | Path | None = None,
) -> list[Path]:
    """Render all five OFAT groups and four metrics in one paired figure."""
    summary_path = Path(summary_csv)
    if not summary_path.is_file():
        raise FileNotFoundError(f"Paired summary not found: {summary_path}")
    frame = pd.read_csv(summary_path)
    required = {
        "dataset", "window", "seed", "status", "parameter", "level", "value",
        *(f"pbir_{metric[0]}" for metric in METRICS),
        *(f"pure_gift_{metric[0]}" for metric in METRICS),
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(
            f"Paired summary is missing columns: {sorted(missing)}")

    frame = frame[frame["status"].isin(["completed", "skipped_existing"])].copy()
    if frame.empty:
        return []
    numeric = [
        "value",
        *(f"{method}_{metric[0]}"
          for method in ("pbir", "pure_gift") for metric in METRICS),
    ]
    for column in numeric:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    parameters = [
        parameter for parameter in PARAMETER_LABELS
        if not frame[frame["parameter"] == parameter].empty
    ]
    if not parameters:
        return []

    dataset = str(frame["dataset"].iloc[0])
    window = str(frame["window"].iloc[0])
    seed = str(frame["seed"].iloc[0])
    output_root = Path(output_dir) if output_dir else summary_path.parent / "figures"
    output_root.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(
        len(parameters), len(METRICS),
        figsize=(16.0, max(7.4, 2.75 * len(parameters) + 3.4)),
        squeeze=False, sharey="col",
    )
    figure.patch.set_facecolor("white")
    figure.suptitle(
        "Paired Hyperparameter Comparison",
        x=0.13, y=0.985, ha="left", fontsize=16,
        fontweight="semibold", color=CHARCOAL,
    )
    figure.text(
        0.13, 0.94,
        f"{dataset} | {window} | PBIR-GIFT vs Pure GIFT | "
        f"{len(frame)} completed paired trials | seed {seed}",
        ha="left", fontsize=10, color=GREY,
    )

    for row_index, parameter in enumerate(parameters):
        group = frame[frame["parameter"] == parameter].sort_values(
            ["level", "value"])
        values = group["value"].to_numpy(dtype=float)
        positions = np.arange(len(group), dtype=float)
        for column_index, (metric, metric_label, _, _) in enumerate(METRICS):
            axis = axes[row_index, column_index]
            axis.set_facecolor("white")
            axis.spines[["top", "right"]].set_visible(False)
            axis.spines[["left", "bottom"]].set_color(GREY)
            axis.grid(axis="y", color=GRID, linewidth=0.8, alpha=0.85)
            axis.set_axisbelow(True)
            axis.tick_params(axis="both", labelsize=7.8, colors=CHARCOAL)
            axis.set_xticks(positions)
            axis.set_xticklabels([
                _format_x(value, parameter) for value in values
            ], fontsize=7.2)
            axis.margins(x=0.08, y=0.18)
            if row_index == 0:
                axis.set_title(metric_label, loc="left", fontsize=11,
                               color=CHARCOAL, pad=8)
            for prefix, label, color, marker, linestyle in (
                    ("pbir", "PBIR-GIFT", PBIR_COLOR, "o", "-"),
                    ("pure_gift", "Pure GIFT", GIFT_COLOR, "s", "--")):
                y = group[f"{prefix}_{metric}"].to_numpy(dtype=float)
                finite = np.isfinite(y)
                axis.plot(
                    positions[finite], y[finite], color=color,
                    linewidth=1.9, linestyle=linestyle,
                    marker=marker, markersize=5.2,
                    markerfacecolor="white", markeredgewidth=1.3,
                    label=label, zorder=3,
                )

    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles, labels, loc="upper right", bbox_to_anchor=(0.96, 0.912),
        frameon=False, ncol=2, fontsize=10,
    )
    figure.text(
        0.13, 0.012,
        "Each x-position is one matched OFAT setting; methods use the same "
        "dataset, window, PPO setting, and seed within a Trial.",
        ha="left", fontsize=9, color=GREY,
    )
    figure.subplots_adjust(
        left=0.16, right=0.975, top=0.82, bottom=0.075,
        hspace=0.48, wspace=0.24,
    )
    for row_index, parameter in enumerate(parameters):
        position = axes[row_index, 0].get_position()
        figure.text(
            0.018, (position.y0 + position.y1) / 2,
            textwrap.fill(PARAMETER_LABELS[parameter], width=18),
            ha="left", va="center", fontsize=9.5,
            fontweight="semibold", color=CHARCOAL,
        )

    written: list[Path] = []
    for extension, kwargs in (("png", {"dpi": 300}), ("pdf", {})):
        path = output_root / f"{dataset}_{window}_paired_overview.{extension}"
        figure.savefig(path, bbox_inches="tight", facecolor="white", **kwargs)
        written.append(path)
    plt.close(figure)
    return written
