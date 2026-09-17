"""Run one Stage-B PBIR-GIFT/Pure-GIFT cell with frozen PPO settings.

One invocation represents exactly one dataset/window/seed cell.  The two
methods share the selected PPO values and all non-PBIR settings.  They execute
sequentially on a single GPU because two simultaneous training processes would
compete for the same device memory; the cell remains a paired comparison.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
import pickle
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from final_pair_visualization import generate_final_pair_figure
from full_method_visualization import generate_full_method_figure
from code_sandbox import validate as sandbox_validate
from hparam_sweep_runner import (
    PARAMETER_GROUPS,
    PROJECT_DIR,
    _check_child_python,
    _experiment_complete,
    _normalize_pbir,
    _resolve_project_path,
    _validate_data,
    collect_metrics,
)
from metrics import max_drawdown, sharpe_ratio, sortino_ratio
from portfolio_env import PortfolioEnv
from reward_rules import REWARD_RULE_REGISTRY


METHODS = (
    ("pbir", "enabled", "PBIR-GIFT"),
    ("pure_gift", "disabled", "Pure GIFT"),
)
STATE_METHOD = ("state_gift", "disabled", "State-only GIFT")
METRIC_NAMES = (
    "best_train_sharpe", "best_iteration", "test_sharpe", "test_sortino",
    "test_max_drawdown", "test_total_return", "wall_time_seconds",
    "llm_call_attempts", "llm_call_successes", "valid_code_rate",
    "failed_iterations",
)


def _parse_args(
        default_base_config: str,
        default_dataset: str,
        default_window: str,
        default_results_dir: str = "results/stage_b_final",
        default_shared_code_source: str | None = None,
        default_generated_config_dir: str = "configs/_stage_b_final",
        default_evaluation_role: str = "final",
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run one frozen-configuration paired PBIR-GIFT/Pure-GIFT cell."))
    parser.add_argument("--base-config", default=default_base_config)
    parser.add_argument("--dataset-name", default=default_dataset)
    parser.add_argument("--window-name", default=default_window)
    parser.add_argument(
        "--selection",
        default=(
            "results/stage_a_tuning/portfolio_5stocks/W1/seed_42/"
            "selection/selected_ppo.yaml"),
        help="Stage-A selected_ppo.yaml artifact.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--results-dir", default=default_results_dir,
        help="Root for frozen-config final results.")
    parser.add_argument(
        "--shared-code-source-root", default=default_shared_code_source,
        help=("Stage-B result root used to materialize one immutable PBIR "
              "code/rule artifact per cell. Supplying this enables the "
              "zero-LLM controlled-replication protocol."))
    parser.add_argument(
        "--generated-config-dir", default=default_generated_config_dir,
        help="Root directory for generated per-method YAML files.")
    parser.add_argument(
        "--python-executable", default=sys.executable,
        help="Python executable used to invoke main.py.")
    parser.add_argument(
        "--device", choices=["cuda", "auto", "cpu"], default="cuda")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument(
        "--evaluation-role",
        choices=["final", "validation_confirmation", "controlled_replication"],
        default=default_evaluation_role)
    parser.add_argument(
        "--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--include-ppo-baselines", action="store_true",
        help=("Also train Pure-PPO baselines inside both method runs. Disabled "
              "by default because they are not part of the PBIR/GIFT pair."))
    parser.add_argument(
        "--include-state-gift", action="store_true",
        help=("Also run an LLM revise_state-only GIFT control with the "
              "generated intrinsic reward disabled."))
    parser.add_argument(
        "--include-equal-weight", action="store_true",
        help="Evaluate a deterministic equal-weight risky-asset baseline.")
    parser.add_argument(
        "--allow-unconfirmed-selection", action="store_true",
        help=("Allow a Stage-B run before the combined five-parameter setting "
              "has passed its validation confirmation. Not recommended."))
    parser.add_argument(
        "--no-llm", action="store_true",
        help="Deterministic offline fallback; never use it for formal results.")
    return parser.parse_args()


def _load_yaml_mapping(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle) or {}
    if not isinstance(value, dict):
        raise TypeError(f"{label} must contain a YAML mapping: {path}")
    return value


def _load_json_mapping(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} is not valid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise TypeError(f"{label} must contain a JSON object: {path}")
    return value


def _selection_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _bytes_sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def materialize_shared_code_artifact(
        source_root: Path,
        destination: Path,
        dataset: str,
        window: str,
        seed: int,
) -> tuple[Path, dict[str, Any], str]:
    """Copy the Stage-B PBIR winner into one verified shared replay artifact.

    The source PBIR run already performed the LLM search and train-only code
    selection. Stage-C does not call the LLM again: it gives this exact code
    and reward-rule payload to both members of the pair.
    """
    source_cell = source_root / dataset / window / f"seed_{seed}"
    pair_summary_path = source_cell / "final_pair_summary.json"
    source_method = source_cell / "methods" / "pbir"
    summary_path = source_method / "summary.json"
    if not pair_summary_path.is_file() or not summary_path.is_file():
        raise FileNotFoundError(
            "Completed Stage-B PBIR source is missing for "
            f"{dataset}/{window}/seed_{seed}: {source_cell}")
    pair_summary = _load_json_mapping(pair_summary_path, "Stage-B pair summary")
    if pair_summary.get("status") not in {"completed", "skipped_existing"}:
        raise ValueError(
            f"Stage-B source cell is not complete: {pair_summary_path}")
    source_summary = _load_json_mapping(summary_path, "Stage-B PBIR summary")
    best_iteration = source_summary.get("best_iteration")
    if not isinstance(best_iteration, int) or best_iteration < 1:
        raise ValueError(
            f"Stage-B PBIR best_iteration is invalid: {best_iteration!r}")
    iteration_dir = source_method / f"iteration_{best_iteration}"
    source_code = iteration_dir / "code.py"
    source_config = iteration_dir / "config.json"
    if not source_code.is_file() or not source_config.is_file():
        raise FileNotFoundError(
            f"Best Stage-B PBIR iteration artifact is incomplete: {iteration_dir}")

    source_code_bytes = source_code.read_bytes()
    code_text = None
    source_encoding = None
    for encoding in ("utf-8", "utf-8-sig", "gb18030"):
        try:
            code_text = source_code_bytes.decode(encoding)
            source_encoding = encoding
            break
        except UnicodeDecodeError:
            continue
    if code_text is None or source_encoding is None:
        raise UnicodeDecodeError(
            "utf-8/gb18030", source_code_bytes, 0,
            min(1, len(source_code_bytes)),
            f"Cannot decode cached LLM code: {source_code}")
    # Normalize the portable Stage-C artifact to UTF-8. The original source
    # byte hash remains in the manifest for traceability.
    code_bytes = code_text.encode("utf-8")
    sandbox = sandbox_validate(code_text)
    if not sandbox["ok"]:
        raise ValueError(
            "Stage-B PBIR winner no longer passes the code sandbox: "
            f"{sandbox['errors']}")
    source_reward = _load_json_mapping(
        source_config, "Stage-B PBIR iteration config")
    reward_rules = source_reward.get("reward_rules", [])
    if not isinstance(reward_rules, list):
        raise ValueError(f"Stage-B reward_rules must be a list: {source_config}")
    for rule in reward_rules:
        if (not isinstance(rule, dict)
                or rule.get("rule") not in REWARD_RULE_REGISTRY):
            raise ValueError(
                f"Invalid Stage-B reward rule in {source_config}: {rule!r}")
    reward_lambda = source_reward.get("lambda")
    if (not isinstance(reward_lambda, (int, float))
            or not np.isfinite(float(reward_lambda))):
        raise ValueError(
            f"Stage-B reward lambda must be finite: {source_config}")
    reward_payload = {
        "reward_rules": reward_rules,
        "lambda": float(reward_lambda),
        "feature_dim": sandbox["feature_dim"],
        "state_dim": sandbox["state_dim"],
        "rationale": (
            "Frozen from the Stage-B PBIR train-selected best iteration; "
            "shared unchanged by PBIR-GIFT and Pure GIFT in the controlled replay."),
    }
    reward_bytes = json.dumps(
        reward_payload, indent=2, sort_keys=True).encode("utf-8")
    code_sha = _bytes_sha256(code_bytes)
    reward_sha = _bytes_sha256(reward_bytes)
    artifact = {
        "schema_version": 1,
        "artifact_role": "fixed-shared-code-controlled-replication",
        "code_file": "code.py",
        "code_sha256": code_sha,
        "reward_config_file": "reward_config.json",
        "reward_config_sha256": reward_sha,
        "feature_dim": sandbox["feature_dim"],
        "state_dim": sandbox["state_dim"],
        "source": {
            "protocol": "stage_b_pipeline_comparison",
            "dataset": dataset,
            "window": window,
            "seed": seed,
            "method": "pbir",
            "best_iteration": best_iteration,
            "selection_sha256": pair_summary.get("selection_sha256"),
            "source_code_encoding": source_encoding,
            "source_code_original_sha256": _bytes_sha256(source_code_bytes),
            "source_code_relative_path": str(
                source_code.relative_to(source_root)).replace("\\", "/"),
            "source_config_relative_path": str(
                source_config.relative_to(source_root)).replace("\\", "/"),
        },
    }
    manifest_bytes = json.dumps(
        artifact, indent=2, sort_keys=True).encode("utf-8")
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "code.py").write_bytes(code_bytes)
    (destination / "reward_config.json").write_bytes(reward_bytes)
    manifest_path = destination / "shared_artifact.json"
    manifest_path.write_bytes(manifest_bytes)
    manifest_sha = _bytes_sha256(manifest_bytes)
    return manifest_path, artifact, manifest_sha


def prepare_frozen_configs(
        base_config: dict[str, Any],
        selection: dict[str, Any],
        evaluation_role: str,
        include_ppo_baselines: bool = False,
        no_llm: bool = False,
        allow_unconfirmed_selection: bool = False,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Validate protocol boundaries and construct the two matched configs."""
    protocol = selection.get("protocol", {})
    selected_ppo = selection.get("ppo", {})
    if protocol.get("stage") != "A_selected":
        raise ValueError("Selection artifact must have protocol.stage=A_selected")
    confirmation = protocol.get("combined_confirmation", {})
    accepted_confirmations = {"completed", "embedded_in_sequential_ofat"}
    locked_evaluation = evaluation_role in {"final", "controlled_replication"}
    if (locked_evaluation and not allow_unconfirmed_selection
            and confirmation.get("status") not in accepted_confirmations):
        raise ValueError(
            "The OFAT-selected joint PPO setting has not completed its Stage-A "
            "validation confirmation. Run run_stage_a_tuning.py with "
            "sequential OFAT, or explicitly pass "
            "--allow-unconfirmed-selection.")
    expected = set(PARAMETER_GROUPS)
    received = set(selected_ppo)
    if received != expected:
        raise ValueError(
            "Selection artifact must contain exactly the five swept PPO keys; "
            f"missing={sorted(expected - received)}, "
            f"unexpected={sorted(received - expected)}")

    validation_period = protocol.get("validation_period")
    experiment = base_config.get("experiment", {})
    evaluation_period = experiment.get("test_period")
    if (not isinstance(validation_period, list) or len(validation_period) != 2
            or not isinstance(evaluation_period, list)
            or len(evaluation_period) != 2):
        raise ValueError("Selection and base config both need two-date periods")
    if locked_evaluation and validation_period[1] >= evaluation_period[0]:
        raise ValueError(
            "Data-leakage guard: Stage-A validation must end before the "
            f"Stage-B test starts ({validation_period[1]} >= "
            f"{evaluation_period[0]}).")

    frozen_base = copy.deepcopy(base_config)
    frozen_base.setdefault("ppo", {}).update(selected_ppo)
    frozen_exp = frozen_base.setdefault("experiment", {})
    frozen_exp["protocol_role"] = evaluation_role
    frozen_exp["run_ppo_baselines"] = bool(include_ppo_baselines)
    if no_llm:
        frozen_exp["no_llm"] = True

    pbir_config, _ = _normalize_pbir(frozen_base, "enabled")
    gift_config, _ = _normalize_pbir(frozen_base, "disabled")
    # Pure-PPO baselines do not depend on PBIR/GIFT reward semantics. When
    # requested, train them once in the Pure-GIFT branch instead of duplicating
    # them in both branches.
    pbir_config.setdefault("experiment", {})["run_ppo_baselines"] = False
    gift_config.setdefault("experiment", {})["run_ppo_baselines"] = bool(
        include_ppo_baselines)
    return {"pbir": pbir_config, "pure_gift": gift_config}, selected_ppo


def _add_deltas(row: dict[str, Any]) -> None:
    for metric in ("test_sharpe", "test_sortino", "test_total_return",
                   "test_max_drawdown"):
        pbir = row.get(f"pbir_{metric}")
        gift = row.get(f"pure_gift_{metric}")
        if all(isinstance(value, (int, float)) and np.isfinite(value)
               for value in (pbir, gift)):
            row[f"{metric}_delta_pbir_minus_gift"] = float(pbir - gift)
    pbir_mdd = row.get("pbir_test_max_drawdown")
    gift_mdd = row.get("pure_gift_test_max_drawdown")
    if all(isinstance(value, (int, float)) and np.isfinite(value)
           for value in (pbir_mdd, gift_mdd)):
        row["test_max_drawdown_improvement_pbir_minus_gift"] = float(
            gift_mdd - pbir_mdd)


def _add_optional_baselines(
        row: dict[str, Any], experiment_dir: Path) -> None:
    path = experiment_dir / "final_comparison.json"
    if not path.is_file():
        return
    try:
        comparison = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return
    mappings = (
        ("ppo_test_only", "baseline1_test_only", "baseline"),
        ("ppo_train_plus_test", "baseline2_train_plus_test", "base2"),
    )
    for output_prefix, section, source_prefix in mappings:
        result = comparison.get(section, {})
        if not isinstance(result, dict):
            continue
        for metric in ("sharpe", "sortino", "max_drawdown", "total_return"):
            value = result.get(f"{source_prefix}_{metric}")
            if isinstance(value, (int, float)) and np.isfinite(value):
                row[f"{output_prefix}_test_{metric}"] = float(value)


def _equal_weight_metrics(
        config: dict[str, Any], data_path: str) -> dict[str, float]:
    test_period = config.get("experiment", {}).get("test_period", [])
    with Path(data_path).open("rb") as handle:
        raw = pickle.load(handle)
    dates = sorted(day for day in raw if test_period[0] <= day <= test_period[1])
    if len(dates) < 2:
        raise ValueError("Equal-weight baseline has no usable test dates")
    split = len(dates) // 2
    evaluation_period = (dates[split], dates[-1])
    env = PortfolioEnv(
        data_path, config, train_period=evaluation_period,
        transaction_cost=float(config.get("portfolio", {}).get(
            "transaction_cost", 0.001)))
    env.reset()
    risky_count = len(config.get("data", {}).get("tickers", []))
    weights = np.concatenate([
        np.full(risky_count, 1.0 / risky_count), np.array([0.0])])
    returns: list[float] = []
    done = False
    while not done:
        _, _, done, info = env.step(weights)
        if info:
            returns.append(float(info.get("portfolio_return", 0.0)))
    return {
        "equal_weight_test_sharpe": sharpe_ratio(returns),
        "equal_weight_test_sortino": sortino_ratio(returns),
        "equal_weight_test_max_drawdown": max_drawdown(returns),
        "equal_weight_test_total_return": (env.portfolio_value - 1.0) * 100,
    }


def _write_single_row_csv(row: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)


def run_final_pair(
        default_base_config: str,
        default_dataset: str,
        default_window: str,
        default_results_dir: str = "results/stage_b_final",
        default_shared_code_source: str | None = None,
        default_generated_config_dir: str = "configs/_stage_b_final",
        default_evaluation_role: str = "final",
) -> int:
    args = _parse_args(
        default_base_config,
        default_dataset,
        default_window,
        default_results_dir=default_results_dir,
        default_shared_code_source=default_shared_code_source,
        default_generated_config_dir=default_generated_config_dir,
        default_evaluation_role=default_evaluation_role,
    )
    if not re.fullmatch(r"W[1-6]|VALIDATION", args.window_name,
                        flags=re.IGNORECASE):
        raise ValueError("--window-name must be W1 through W6 or VALIDATION")
    window = args.window_name.upper()
    if args.evaluation_role == "final" and window == "VALIDATION":
        raise ValueError("VALIDATION window requires --evaluation-role validation_confirmation")
    controlled_replay = bool(args.shared_code_source_root)
    if controlled_replay and args.evaluation_role != "controlled_replication":
        raise ValueError(
            "--shared-code-source-root requires "
            "--evaluation-role controlled_replication")
    if (args.evaluation_role == "controlled_replication"
            and not controlled_replay):
        raise ValueError(
            "controlled_replication requires --shared-code-source-root")
    if controlled_replay and args.no_llm:
        raise ValueError(
            "Do not combine --shared-code-source-root with --no-llm; the "
            "shared artifact already provides the zero-LLM execution path")

    base_path = _resolve_project_path(args.base_config)
    selection_path = _resolve_project_path(args.selection)
    base_config = _load_yaml_mapping(base_path, "Base config")
    selection = _load_yaml_mapping(selection_path, "Stage-A selection artifact")
    method_configs, selected_ppo = prepare_frozen_configs(
        base_config,
        selection,
        args.evaluation_role,
        include_ppo_baselines=args.include_ppo_baselines,
        no_llm=args.no_llm,
        allow_unconfirmed_selection=args.allow_unconfirmed_selection,
    )
    method_specs = list(METHODS)
    if args.include_state_gift:
        state_config = copy.deepcopy(method_configs["pure_gift"])
        state_exp = state_config.setdefault("experiment", {})
        state_exp["disable_intrinsic_reward"] = True
        state_exp["run_ppo_baselines"] = False
        method_configs["state_gift"] = state_config
        method_specs.append(STATE_METHOD)

    # This validates date coverage and pickle schema before an API call.
    data_info = _validate_data(method_configs["pbir"])
    child_env = os.environ.copy()
    child_env["CUDA_VISIBLE_DEVICES"] = (
        str(args.gpu) if args.device in {"cuda", "auto"} else "")
    device_info: dict[str, Any] = {
        "requested": args.device,
        "physical_gpu": args.gpu if args.device != "cpu" else None,
    }
    if not args.dry_run:
        device_info.update(_check_child_python(
            args.python_executable, child_env,
            require_cuda=args.device == "cuda"))
        provider = str(base_config.get("llm", {}).get(
            "provider", "openai")).lower()
        if (provider == "bailian" and not args.no_llm
                and not controlled_replay
                and not os.environ.get("DASHSCOPE_API_KEY")):
            raise RuntimeError(
                "DASHSCOPE_API_KEY is not visible to this launcher process")

    result_base = _resolve_project_path(args.results_dir)
    shared_source_root = (
        _resolve_project_path(args.shared_code_source_root)
        if controlled_replay else None)
    if (shared_source_root is not None
            and result_base.resolve() == shared_source_root.resolve()):
        raise ValueError(
            "Controlled-replay results must use a new root; refusing to write into the "
            "locked Stage-B source directory")
    results_root = (
        result_base / args.dataset_name / window / f"seed_{args.seed}")
    configs_root = (
        _resolve_project_path(args.generated_config_dir) / args.dataset_name
        / window / f"seed_{args.seed}")
    results_root.mkdir(parents=True, exist_ok=True)
    configs_root.mkdir(parents=True, exist_ok=True)

    shared_artifact_path: Path | None = None
    shared_artifact: dict[str, Any] | None = None
    shared_artifact_sha256: str | None = None
    if shared_source_root is not None:
        shared_artifact_path, shared_artifact, shared_artifact_sha256 = (
            materialize_shared_code_artifact(
                shared_source_root,
                results_root / "shared_artifact",
                args.dataset_name,
                window,
                args.seed,
            ))
        for config in method_configs.values():
            experiment = config.setdefault("experiment", {})
            experiment["fixed_artifact_path"] = str(shared_artifact_path)
            experiment["fixed_artifact_sha256"] = shared_artifact_sha256
            experiment["protocol_role"] = "controlled_replication"
            experiment["max_iterations"] = 1
            experiment["sample_count"] = 1

    order_digest = hashlib.sha256(
        f"{args.dataset_name}|{window}|{args.seed}".encode("utf-8")).digest()
    offset = order_digest[0] % len(method_specs)
    methods_for_cell = tuple(method_specs[offset:] + method_specs[:offset])

    manifest = {
        "protocol": (
            "fixed-code-controlled-replication" if controlled_replay else
            "two-stage-frozen-config-paired-evaluation"),
        "evaluation_role": args.evaluation_role,
        "dataset": args.dataset_name,
        "window": window,
        "seed": args.seed,
        "method_order_within_cell": [item[0] for item in methods_for_cell],
        "method_order_policy": "sha256(dataset|window|seed) cyclic rotation",
        "base_config": str(base_path),
        "selection_artifact": str(selection_path),
        "selection_sha256": _selection_hash(selection_path),
        "frozen_ppo": selected_ppo,
        "data": data_info,
        "device": device_info,
        "resume": args.resume,
        "dry_run": args.dry_run,
        "no_llm": args.no_llm,
        "include_ppo_baselines": args.include_ppo_baselines,
        "include_state_gift": args.include_state_gift,
        "include_equal_weight": args.include_equal_weight,
        "allow_unconfirmed_selection": args.allow_unconfirmed_selection,
        "shared_code_source_root": (
            str(shared_source_root) if shared_source_root else None),
        "shared_artifact_path": (
            str(shared_artifact_path) if shared_artifact_path else None),
        "shared_artifact_sha256": shared_artifact_sha256,
        "shared_code_sha256": (
            shared_artifact.get("code_sha256") if shared_artifact else None),
        "shared_reward_config_sha256": (
            shared_artifact.get("reward_config_sha256")
            if shared_artifact else None),
    }
    existing_manifest = _load_yaml_mapping(
        results_root / "manifest.json", "Existing cell manifest"
    ) if (results_root / "manifest.json").is_file() else {}
    suite_fields = (
        "selection_sha256", "include_ppo_baselines", "include_state_gift",
        "include_equal_weight", "shared_artifact_sha256",
    )
    suite_changed = any(
        existing_manifest.get(key) != manifest.get(key) for key in suite_fields)
    existing_complete = any(
        _experiment_complete(results_root / "methods" / method)
        for method in ("pbir", "pure_gift", "state_gift"))
    if args.resume and suite_changed and existing_complete:
        raise RuntimeError(
            "This cell already contains completed results from a different "
            "method suite or selection artifact. Use a new --results-dir for "
            "the six-method study, or pass --no-resume to rerun the whole cell.")
    (results_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")

    print("=" * 78)
    print("FIXED-CODE CONTROLLED REPLICATION" if controlled_replay else
          "FROZEN-CONFIG PAIRED EVALUATION")
    print(f"Dataset/window/seed: {args.dataset_name} / {window} / {args.seed}")
    print(f"Frozen PPO: {selected_ppo}")
    if controlled_replay:
        print("The pair shares the exact LLM code/rules; only PBIR shaping toggles.")
        print(f"Shared artifact SHA-256: {shared_artifact_sha256}")
        print("LLM calls during fixed-code replay: 0")
    else:
        print("The PBIR/Pure core pair shares all settings except reward semantics.")
    if args.include_state_gift:
        print("State-only GIFT is an additional predeclared control.")
    print(f"Results: {results_root}")
    print("=" * 78)

    row: dict[str, Any] = {
        "dataset": args.dataset_name,
        "window": window,
        "seed": args.seed,
        "status": "generated" if args.dry_run else "pending",
        "evaluation_role": args.evaluation_role,
        "selection_artifact": str(selection_path),
        "selection_sha256": manifest["selection_sha256"],
        "method_order": ">".join(item[0] for item in methods_for_cell),
        "shared_artifact_path": (
            str(shared_artifact_path) if shared_artifact_path else None),
        "shared_artifact_sha256": shared_artifact_sha256,
        "shared_code_sha256": (
            shared_artifact.get("code_sha256") if shared_artifact else None),
        "shared_reward_config_sha256": (
            shared_artifact.get("reward_config_sha256")
            if shared_artifact else None),
        "shared_source_best_iteration": (
            shared_artifact.get("source", {}).get("best_iteration")
            if shared_artifact else None),
        **{f"selected_{key}": value for key, value in selected_ppo.items()},
    }
    if args.include_equal_weight:
        row.update(_equal_weight_metrics(
            method_configs["pbir"], data_info["path"]))
    method_complete: list[bool] = []
    method_resumed: list[bool] = []

    for method, mode, label in methods_for_cell:
        config_path = configs_root / f"{method}.yaml"
        experiment_dir = results_root / "methods" / method
        experiment_dir.mkdir(parents=True, exist_ok=True)
        with config_path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(method_configs[method], handle, sort_keys=False)
        resolved_config_path = experiment_dir / "resolved_config.yaml"
        with resolved_config_path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(method_configs[method], handle, sort_keys=False)
        row[f"{method}_config_path"] = str(config_path)
        row[f"{method}_resolved_config_path"] = str(resolved_config_path)
        config_hash = hashlib.sha256(yaml.safe_dump(
            method_configs[method], sort_keys=True).encode("utf-8")).hexdigest()
        row[f"{method}_config_sha256"] = config_hash
        row[f"{method}_experiment_dir"] = str(experiment_dir)

        protocol_path = experiment_dir / "run_protocol.json"
        existing_protocol: dict[str, Any] = {}
        if protocol_path.is_file():
            try:
                existing_protocol = json.loads(
                    protocol_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                existing_protocol = {}
        resume_compatible = (
            existing_protocol.get("config_sha256") == config_hash
            and existing_protocol.get("selection_sha256")
            == manifest["selection_sha256"]
            and existing_protocol.get("seed") == args.seed
            and existing_protocol.get("method") == method
            and existing_protocol.get("shared_artifact_sha256")
            == shared_artifact_sha256)
        protocol_path.write_text(json.dumps({
            "config_sha256": config_hash,
            "selection_sha256": manifest["selection_sha256"],
            "seed": args.seed,
            "method": method,
            "dataset": args.dataset_name,
            "window": window,
            "evaluation_role": args.evaluation_role,
            "shared_artifact_path": (
                str(shared_artifact_path) if shared_artifact_path else None),
            "shared_artifact_sha256": shared_artifact_sha256,
            "shared_code_sha256": (
                shared_artifact.get("code_sha256")
                if shared_artifact else None),
        }, indent=2), encoding="utf-8")

        if args.dry_run:
            row[f"{method}_status"] = "generated"
            row[f"{method}_return_code"] = None
            continue
        if (args.resume and resume_compatible
                and _experiment_complete(experiment_dir)):
            print(f"{label}: resume -> existing complete result")
            row[f"{method}_status"] = "skipped_existing"
            row[f"{method}_return_code"] = 0
            metrics = collect_metrics(experiment_dir)
            row.update({f"{method}_{key}": value
                        for key, value in metrics.items()})
            if method == "pure_gift":
                _add_optional_baselines(row, experiment_dir)
            method_complete.append(True)
            method_resumed.append(True)
            continue
        if args.resume and _experiment_complete(experiment_dir):
            print(
                f"{label}: existing result has a different frozen protocol/"
                "config hash; it will not be reused")

        print(f"{label}: running")
        command = [
            args.python_executable, str(PROJECT_DIR / "main.py"),
            "--config", str(config_path),
            "--output-dir", str(experiment_dir),
            "--seed", str(args.seed),
            "--pbir-mode", mode,
        ]
        if shared_artifact_path is not None:
            command.extend(["--fixed-artifact", str(shared_artifact_path)])
        completed = subprocess.run(command, cwd=PROJECT_DIR, env=child_env)
        metrics = collect_metrics(experiment_dir)
        row.update({f"{method}_{key}": value
                    for key, value in metrics.items()})
        if method == "pure_gift":
            _add_optional_baselines(row, experiment_dir)
        complete = completed.returncode == 0 and _experiment_complete(
            experiment_dir)
        row[f"{method}_return_code"] = completed.returncode
        row[f"{method}_status"] = (
            "completed" if complete else
            "failed" if completed.returncode else "incomplete")
        method_complete.append(complete)
        method_resumed.append(False)

    if not args.dry_run and len(method_complete) == len(method_specs) \
            and all(method_complete):
        row["status"] = (
            "skipped_existing" if all(method_resumed) else "completed")
        _add_deltas(row)
    elif not args.dry_run:
        row["status"] = "failed"

    summary_json = results_root / "final_pair_summary.json"
    summary_json.write_text(json.dumps(row, indent=2), encoding="utf-8")
    _write_single_row_csv(row, results_root / "final_pair_summary.csv")
    if row["status"] in {"completed", "skipped_existing"}:
        try:
            paths = generate_final_pair_figure(summary_json)
            print(f"Cell figures: {paths[0]} and {paths[1]}")
        except Exception as exc:
            print(f"WARNING: final-pair figure skipped: {exc}", file=sys.stderr)
        try:
            paths = generate_full_method_figure(summary_json)
            if paths:
                print(f"Full-method figures: {paths[0]} and {paths[1]}")
        except Exception as exc:
            print(f"WARNING: full-method figure skipped: {exc}", file=sys.stderr)
    print(f"Cell summary: {summary_json}")
    return 0 if row["status"] in {
        "generated", "completed", "skipped_existing"} else 1


def main() -> int:
    return run_final_pair("config_W1.yaml", "portfolio_5stocks", "W1")


if __name__ == "__main__":
    raise SystemExit(main())
