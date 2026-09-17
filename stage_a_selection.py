"""Select one shared PPO configuration from the Stage-A paired OFAT sweep.

The selector never reads a final-window test result.  Its input must come from
``config_tuning_D1_W1.yaml``, where the controller's ``test_period`` is a
dedicated pre-W1 validation holdout.  One value is selected independently for
each OFAT parameter using the mean validation Sharpe of PBIR-GIFT and Pure
GIFT.  The combined setting is written as a frozen protocol artifact for all
Stage-B cells.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import yaml

from hparam_sweep_runner import PARAMETER_GROUPS, PROJECT_DIR


COMPLETED_STATUSES = {"completed", "skipped_existing"}


def _finite(row: dict[str, str], key: str) -> float:
    try:
        value = float(row[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Stage-A row has no numeric {key}: {row}") from exc
    if not math.isfinite(value):
        raise ValueError(f"Stage-A row has non-finite {key}: {value}")
    return value


def _load_rows(summary_path: Path) -> list[dict[str, str]]:
    if not summary_path.is_file():
        raise FileNotFoundError(f"Stage-A paired summary not found: {summary_path}")
    with summary_path.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    complete = [row for row in rows if row.get("status") in COMPLETED_STATUSES]
    if not complete:
        raise ValueError("Stage-A summary has no completed paired Trial")
    return complete


def select_stage_a_configuration(
        summary_csv: str | Path,
        base_config: str | Path,
        output_dir: str | Path | None = None,
) -> Path:
    """Select and persist the shared Stage-B PPO setting.

    Returns the path to ``selected_ppo.yaml``.
    """
    summary_path = Path(summary_csv)
    base_path = Path(base_config)
    if not base_path.is_file():
        raise FileNotFoundError(f"Stage-A validation config not found: {base_path}")
    with base_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    experiment = config.get("experiment", {})
    if experiment.get("protocol_role") != "hyperparameter_validation":
        raise ValueError(
            "Stage-A selection requires experiment.protocol_role="
            "hyperparameter_validation")
    validation_period = experiment.get("test_period")
    if not isinstance(validation_period, list) or len(validation_period) != 2:
        raise ValueError("Stage-A validation config needs a two-date test_period")
    if experiment.get("search_strategy") != "sequential_ofat":
        raise ValueError(
            "Stage-A selection requires experiment.search_strategy="
            "sequential_ofat so the final joint setting was directly evaluated")

    rows = _load_rows(summary_path)
    scored: list[dict[str, Any]] = []
    selected: dict[str, Any] = {}
    selected_rows: dict[str, dict[str, Any]] = {}

    for parameter, allowed_values in PARAMETER_GROUPS.items():
        group = [row for row in rows if row.get("parameter") == parameter]
        levels = {int(row["level"]) for row in group}
        expected_levels = set(range(1, len(allowed_values) + 1))
        if levels != expected_levels:
            missing = sorted(expected_levels.difference(levels))
            raise ValueError(
                f"Stage-A parameter {parameter} is incomplete; missing levels {missing}")

        expected_prior = {
            key: selected[key]
            for key in PARAMETER_GROUPS
            if key in selected
        }
        for row in group:
            try:
                observed_prior = json.loads(row.get("frozen_prior_ppo", ""))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Stage-A row {row.get('trial_name')} has no valid "
                    "frozen_prior_ppo audit field") from exc
            if observed_prior != expected_prior:
                raise ValueError(
                    f"Sequential OFAT audit failed for {parameter}: expected "
                    f"prior {expected_prior}, observed {observed_prior}")

        candidates: list[dict[str, Any]] = []
        for row in group:
            level = int(row["level"])
            pbir_sharpe = _finite(row, "pbir_test_sharpe")
            gift_sharpe = _finite(row, "pure_gift_test_sharpe")
            pbir_sortino = _finite(row, "pbir_test_sortino")
            gift_sortino = _finite(row, "pure_gift_test_sortino")
            pbir_return = _finite(row, "pbir_test_total_return")
            gift_return = _finite(row, "pure_gift_test_total_return")
            pbir_mdd = _finite(row, "pbir_test_max_drawdown")
            gift_mdd = _finite(row, "pure_gift_test_max_drawdown")
            candidate = {
                "parameter": parameter,
                "level": level,
                "value": allowed_values[level - 1],
                "pbir_validation_sharpe": pbir_sharpe,
                "pure_gift_validation_sharpe": gift_sharpe,
                "mean_validation_sharpe": (pbir_sharpe + gift_sharpe) / 2.0,
                "mean_validation_sortino": (pbir_sortino + gift_sortino) / 2.0,
                "mean_validation_total_return": (pbir_return + gift_return) / 2.0,
                "mean_validation_max_drawdown": (pbir_mdd + gift_mdd) / 2.0,
            }
            candidates.append(candidate)
            scored.append(candidate)

        # One shared value is chosen for both methods.  Mean Sharpe is primary;
        # the remaining keys only provide deterministic, method-neutral ties.
        winner = max(
            candidates,
            key=lambda item: (
                item["mean_validation_sharpe"],
                item["mean_validation_sortino"],
                item["mean_validation_total_return"],
                -abs(item["mean_validation_max_drawdown"]),
                -item["level"],
            ),
        )
        selected[parameter] = winner["value"]
        selected_rows[parameter] = winner

    output_root = Path(output_dir) if output_dir else summary_path.parent / "selection"
    output_root.mkdir(parents=True, exist_ok=True)

    score_path = output_root / "selection_scores.csv"
    with score_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(scored[0]))
        writer.writeheader()
        writer.writerows(scored)

    artifact = {
        "protocol": {
            "name": "two_stage_paired_evaluation",
            "version": 1,
            "stage": "A_selected",
            "selection_dataset": "portfolio_5stocks",
            "selection_window": "D1-W1-internal-validation",
            "validation_period": validation_period,
            "selection_rule": (
                "Fixed-order sequential OFAT. For each five-level group, "
                "maximize the mean held-out validation Sharpe of PBIR-GIFT "
                "and Pure GIFT, then freeze that winner in every subsequent "
                "group. The final group therefore directly evaluates the "
                "joint five-parameter setting."
            ),
            "parameter_order": list(PARAMETER_GROUPS),
            "source_summary": str(summary_path),
            "source_config": str(base_path),
            "requires_combined_confirmation": False,
            "combined_confirmation": {
                "status": "embedded_in_sequential_ofat",
                "final_group": list(PARAMETER_GROUPS)[-1],
                "selected_level": selected_rows[list(PARAMETER_GROUPS)[-1]][
                    "level"],
            },
        },
        "ppo": selected,
    }
    selected_path = output_root / "selected_ppo.yaml"
    with selected_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(artifact, handle, sort_keys=False)

    manifest = {
        **artifact["protocol"],
        "selected_ppo": selected,
        "selected_rows": selected_rows,
        "candidate_score_table": str(score_path),
        "selected_artifact": str(selected_path),
    }
    (output_root / "selection_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    return selected_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Freeze shared PPO values from a complete Stage-A sweep.")
    parser.add_argument("--summary", required=True)
    parser.add_argument("--base-config", default="config_tuning_D1_W1.yaml")
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    selected = select_stage_a_configuration(
        PROJECT_DIR / args.summary if not Path(args.summary).is_absolute() else args.summary,
        PROJECT_DIR / args.base_config if not Path(args.base_config).is_absolute()
        else args.base_config,
        PROJECT_DIR / args.output_dir if args.output_dir and
        not Path(args.output_dir).is_absolute() else args.output_dir,
    )
    print(f"Frozen Stage-A PPO artifact: {selected}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
