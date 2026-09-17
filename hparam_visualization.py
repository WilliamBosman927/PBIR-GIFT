"""Publication-ready hyperparameter sensitivity figures for GIFT sweeps."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd
import yaml


PARAMETER_LABELS = {
    "epochs_per_update": "PPO epochs per update",
    "max_episodes": "Training episodes",
    "actor_lr": "Actor learning rate",
    "clip_epsilon": "PPO clipping coefficient",
    "entropy_coef": "Entropy coefficient",
}

METRICS = (
    ("test_sharpe", "Test Sharpe ratio", True, "{:.3f}"),
    ("test_sortino", "Test Sortino ratio", True, "{:.3f}"),
    ("test_total_return", "Test total return (%)", True, "{:.2f}"),
    ("test_max_drawdown", "Test maximum drawdown (%)", False, "{:.2f}"),
)

BLUE = "#2F5D8C"
BLUE_LIGHT = "#DCE8F2"
GOLD = "#C18A20"
CHARCOAL = "#25313C"
GREY = "#77828C"
GRID = "#D8DEE4"


def _format_x(value: float, parameter: str) -> str:
    if parameter == "actor_lr":
        return f"{value:.0e}"
    if parameter in {"epochs_per_update", "max_episodes"}:
        return str(int(value))
    return f"{value:g}"


def _context_from_rows(
        frame: pd.DataFrame,
        dataset: str | None,
        window: str | None,
        method: str | None,
        seed: int | str | None,
) -> tuple[str, str, str, str]:
    def first(column: str, fallback: Any) -> Any:
        if column in frame and frame[column].notna().any():
            return frame.loc[frame[column].notna(), column].iloc[0]
        return fallback

    return (
        str(first("dataset", dataset or "dataset")),
        str(first("window", window or "single")),
        {"pbir": "PBIR-GIFT", "pure_gift": "Pure GIFT"}.get(
            str(first("method", method or "method")),
            str(first("method", method or "method")).upper()),
        str(first("seed", seed if seed is not None else "N/A")),
    )


def generate_sweep_figures(
        summary_csv: str | Path,
        output_dir: str | Path | None = None,
        *,
        base_ppo: dict[str, Any] | None = None,
        dataset: str | None = None,
        window: str | None = None,
        method: str | None = None,
        seed: int | str | None = None,
) -> list[Path]:
    """Generate one four-panel sensitivity figure per parameter group.

    Only completed or resumed-complete trials with finite test metrics are
    plotted.  Every figure is exported as both 300-DPI PNG and vector PDF.
    """
    summary_path = Path(summary_csv)
    if not summary_path.is_file():
        raise FileNotFoundError(f"Sweep summary not found: {summary_path}")
    frame = pd.read_csv(summary_path)
    required = {"parameter", "value", *(metric[0] for metric in METRICS)}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Sweep summary is missing columns: {sorted(missing)}")
    if "status" in frame:
        frame = frame[frame["status"].isin(["completed", "skipped_existing"])]
    for column in ["value", *(metric[0] for metric in METRICS)]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    output_root = Path(output_dir) if output_dir else summary_path.parent / "figures"
    output_root.mkdir(parents=True, exist_ok=True)
    base_ppo = base_ppo or {}
    dataset_name, window_name, method_name, seed_name = _context_from_rows(
        frame, dataset, window, method, seed)
    written: list[Path] = []

    for parameter in PARAMETER_LABELS:
        group = frame[frame["parameter"] == parameter].copy()
        group = group.dropna(subset=["value"]).sort_values(
            ["level", "value"] if "level" in group else ["value"])
        if group.empty:
            continue

        x = group["value"].to_numpy(dtype=float)
        logarithmic_x = parameter == "actor_lr" and np.all(x > 0)
        plot_x = x if logarithmic_x else np.arange(len(x), dtype=float)
        figure, axes = plt.subplots(2, 2, figsize=(11.2, 7.6), constrained_layout=False)
        axes = axes.ravel()
        figure.patch.set_facecolor("white")

        sharpe_values = group["test_sharpe"].to_numpy(dtype=float)
        if np.isfinite(sharpe_values).any():
            best_primary_idx = int(np.nanargmax(sharpe_values))
            primary_note = (
                f"Sharpe-optimal setting: {_format_x(x[best_primary_idx], parameter)} "
                f"(Sharpe {sharpe_values[best_primary_idx]:.3f})")
        else:
            primary_note = "Sharpe-optimal setting unavailable"

        figure.suptitle(
            f"Hyperparameter sensitivity: {PARAMETER_LABELS[parameter]}",
            x=0.07, y=0.965, ha="left", fontsize=16, fontweight="semibold",
            color=CHARCOAL,
        )
        figure.text(
            0.07, 0.925,
            f"{dataset_name} | {window_name} | {method_name} | "
            f"seed {seed_name} | {primary_note}",
            ha="left", va="center", fontsize=9.5, color=GREY,
        )

        for axis, (column, label, higher_is_better, value_format) in zip(axes, METRICS):
            y = group[column].to_numpy(dtype=float)
            finite = np.isfinite(y)
            axis.set_facecolor("white")
            axis.spines[["top", "right"]].set_visible(False)
            axis.spines[["left", "bottom"]].set_color(GREY)
            axis.grid(axis="y", color=GRID, linewidth=0.8, alpha=0.8)
            axis.set_axisbelow(True)
            axis.set_title(label, loc="left", fontsize=11, color=CHARCOAL, pad=8)
            axis.set_xlabel(PARAMETER_LABELS[parameter], fontsize=9, color=CHARCOAL)
            axis.tick_params(axis="both", labelsize=8.5, colors=CHARCOAL)
            axis.set_xticks(plot_x)
            axis.set_xticklabels([_format_x(value, parameter) for value in x])
            if logarithmic_x:
                axis.set_xscale("log")
            if column in {"test_sharpe", "test_sortino", "test_total_return"}:
                axis.axhline(0.0, color=GREY, linewidth=0.9, linestyle="--", zorder=1)

            if not finite.any():
                axis.text(
                    0.5, 0.5, "No completed finite observations",
                    transform=axis.transAxes, ha="center", va="center",
                    color=GREY, fontsize=9,
                )
                continue

            axis.plot(
                plot_x[finite], y[finite], color=BLUE, linewidth=2.0,
                marker="o", markersize=6, markerfacecolor="white",
                markeredgecolor=BLUE, markeredgewidth=1.5, zorder=3,
            )
            valid_indices = np.flatnonzero(finite)
            local = np.argmax(y[finite]) if higher_is_better else np.argmin(y[finite])
            best_idx = int(valid_indices[local])
            axis.scatter(
                [plot_x[best_idx]], [y[best_idx]], marker="*", s=180,
                color=GOLD, edgecolor=CHARCOAL, linewidth=0.6,
                zorder=5, label="Best",
            )

            baseline = base_ppo.get(parameter)
            if isinstance(baseline, (int, float)) and np.isfinite(baseline):
                if logarithmic_x:
                    baseline_x = float(baseline)
                else:
                    matches = np.flatnonzero(np.isclose(x, float(baseline)))
                    baseline_x = float(matches[0]) if len(matches) else None
                if baseline_x is not None:
                    axis.axvline(
                        baseline_x, color=GREY, linewidth=1.1,
                        linestyle=":", zorder=2, label="Base setting",
                    )

            y_span = float(np.nanmax(y[finite]) - np.nanmin(y[finite]))
            offset = y_span * 0.045 if y_span > 0 else max(abs(float(y[finite][0])) * 0.04, 0.03)
            for xi, yi in zip(plot_x[finite], y[finite]):
                axis.text(
                    xi, yi + offset, value_format.format(yi),
                    ha="center", va="bottom", fontsize=7.5, color=CHARCOAL,
                )
            axis.margins(x=0.09, y=0.20)

        handles, labels = axes[0].get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        if unique:
            figure.legend(
                unique.values(), unique.keys(), loc="upper right",
                bbox_to_anchor=(0.94, 0.962), frameon=False, ncol=2,
                fontsize=8.5,
            )
        figure.text(
            0.07, 0.025,
            "Gold star marks the metric-optimal observed setting; dotted vertical line marks the base PPO setting.",
            ha="left", fontsize=8.3, color=GREY,
        )
        figure.subplots_adjust(left=0.075, right=0.96, top=0.865, bottom=0.09,
                               hspace=0.38, wspace=0.25)

        stem = f"{parameter}_sensitivity"
        png_path = output_root / f"{stem}.png"
        pdf_path = output_root / f"{stem}.pdf"
        figure.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
        figure.savefig(pdf_path, bbox_inches="tight", facecolor="white")
        plt.close(figure)
        written.extend([png_path, pdf_path])

    return written


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate GIFT sweep figures.")
    parser.add_argument("--summary", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--base-config")
    parser.add_argument("--dataset")
    parser.add_argument("--window")
    parser.add_argument("--method")
    parser.add_argument("--seed")
    args = parser.parse_args()
    base_ppo: dict[str, Any] = {}
    if args.base_config:
        with Path(args.base_config).open("r", encoding="utf-8") as handle:
            base_ppo = (yaml.safe_load(handle) or {}).get("ppo", {})
    paths = generate_sweep_figures(
        args.summary, args.output_dir, base_ppo=base_ppo,
        dataset=args.dataset, window=args.window,
        method=args.method, seed=args.seed)
    print(f"Generated {len(paths) // 2} parameter figures:")
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
