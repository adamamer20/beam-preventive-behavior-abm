"""Run phase 0 (prompt tournament) end-to-end for choice validation."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import time
from pathlib import Path

import pandas as pd

from beam_abm.evaluation.artifacts import (
    PromptTournamentOutputPaths,
    build_run_metadata,
    utc_timestamp,
    utc_timestamp_iso,
    write_run_metadata,
)
from beam_abm.evaluation.choice._canonicalize_ids import normalize_model_slug
from beam_abm.evaluation.utils import prompt_tournament_utils as pt
from beam_abm.evaluation.utils.alignment_utils import (
    merge_alignment_into_row_level,
    summarize_alignment_rows,
)


def _load_local_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _get_arg_value(args: list[str], flag: str) -> str | None:
    if flag in args:
        idx = args.index(flag)
        if idx + 1 < len(args):
            return args[idx + 1]
    return None


def _set_arg_value(args: list[str], flag: str, value: str) -> list[str]:
    if flag in args:
        idx = args.index(flag)
        if idx + 1 < len(args):
            args[idx + 1] = value
            return args
    return [*args, flag, value]


def _has_flag(args: list[str], flag: str) -> bool:
    return flag in args


def _try_load_json(text: str | None) -> object | None:
    if not text:
        return None
    raw = str(text).strip()
    if not raw:
        return None
    # Common case: shell-quoted JSON ends up wrapped in quotes.
    if (raw.startswith("'") and raw.endswith("'")) or (raw.startswith('"') and raw.endswith('"')):
        raw = raw[1:-1].strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def _load_model_options_payload(
    *,
    override_model_options: str | None,
    override_model_options_file: str | None,
    default_model_options: str,
) -> object | None:
    if override_model_options and override_model_options_file:
        raise ValueError("Provide only one of --model-options or --model-options-file")
    if override_model_options_file:
        return json.loads(Path(override_model_options_file).read_text(encoding="utf-8"))
    loaded = _try_load_json(override_model_options)
    if loaded is not None:
        return loaded
    return _try_load_json(default_model_options)


def _extract_reasoning_effort(model_options: object | None) -> str | None:
    if not isinstance(model_options, dict):
        return None
    reasoning = model_options.get("reasoning")
    if isinstance(reasoning, dict):
        effort = reasoning.get("effort")
        if isinstance(effort, str) and effort.strip():
            return effort.strip().lower()
    effort = model_options.get("reasoning_effort")
    if isinstance(effort, str) and effort.strip():
        return effort.strip().lower()
    return None


def _build_default_generate_args(
    *,
    prompts_path: Path,
    strategy: str,
    keep_outcomes: str,
    output_root: Path,
    run_id: str,
    outcome_batch: str,
    keep_outcome_list: list[str],
    include_input: bool,
    build_paired_profiles_step: object,
) -> list[str]:
    args: list[str] = [
        "--out",
        str(prompts_path),
        "--strategy",
        strategy,
        "--dedup-levers",
        "--keep-outcomes",
        keep_outcomes,
    ]
    if strategy == "paired_profile":
        args.extend(["--paired-profile-multi-outcome"])
        if include_input:
            paired_out = _frozen_paired_profiles_path(output_root, run_id, outcome_batch)
            if not paired_out.exists():
                outcome_list = keep_outcome_list or ["vax_willingness_Y"]
                tmp_paths: list[Path] = []
                for outcome in outcome_list:
                    design_path = Path(f"empirical/output/anchors/pe/mutable_engines/reduced/{outcome}/design.parquet")
                    if not design_path.exists():
                        continue
                    tmp_path = paired_out.parent / f"paired_profiles__{outcome}.jsonl"
                    build_args = f"--in {design_path} --out {tmp_path}"
                    pt.run_step(build_paired_profiles_step.main, pt.split_args(build_args))
                    tmp_paths.append(tmp_path)
                paired_out.parent.mkdir(parents=True, exist_ok=True)
                with paired_out.open("w", encoding="utf-8") as dst:
                    for tmp_path in tmp_paths:
                        if not tmp_path.exists():
                            continue
                        dst.write(tmp_path.read_text(encoding="utf-8"))
            args.extend(["--in", str(paired_out)])
    else:
        if include_input:
            args.extend(["--in-dir", "empirical/output/anchors/pe/mutable_engines/reduced"])
    return args


def _normalize_ref_model(name: str) -> str:
    key = str(name).strip().lower()
    if key in {"xgboost", "xgb"}:
        return "xgb"
    if key in {"linear", "lin"}:
        return "linear"
    return key


def _safe_token(value: str | None) -> str:
    text = str(value or "").strip()
    out = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in text).strip("_")
    return out or "unknown"


def _parse_models(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [m.strip() for m in raw.split(",") if m.strip()]


def _copy_if_missing(src: Path, dest: Path) -> None:
    if not src.exists() or dest.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")


def _ref_ice_path(outcome: str, ref_model: str) -> Path:
    ref_key = _normalize_ref_model(ref_model)
    base = Path("empirical/output/anchors/pe/mutable_engines/reduced") / outcome
    if ref_key == "linear":
        return base / "ice_ref.csv"
    return base / f"ice_ref__{ref_key}.csv"


def _get_row_outcome(row: dict) -> str | None:
    meta = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    outcome = meta.get("outcome") if isinstance(meta, dict) else None
    if outcome is None:
        outcome = row.get("outcome")
    if isinstance(outcome, str) and outcome.strip():
        return outcome.strip()
    return None


def _build_baseline_predictions(
    *,
    samples_path: Path,
    out_path: Path,
    strategy: str = "data_only",
) -> Path:
    if not samples_path.exists():
        raise FileNotFoundError(f"Missing samples for baseline: {samples_path}")

    out_rows: list[dict] = []
    total_rows = 0
    matched_strategy = 0
    matched_with_scalars = 0
    with samples_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            total_rows += 1
            row = json.loads(line)
            if str(row.get("strategy") or "data_only") != strategy:
                continue
            matched_strategy += 1
            row_id = row.get("id")
            if not isinstance(row_id, str) or not row_id:
                continue
            samples = row.get("samples")
            if not isinstance(samples, list) or not samples:
                continue
            vals: list[float] = []
            for sample in samples:
                if not isinstance(sample, dict):
                    continue
                scalar = sample.get("prediction_scalar")
                if scalar is None:
                    continue
                try:
                    vals.append(float(scalar))
                except (TypeError, ValueError):
                    continue
            if not vals:
                continue
            matched_with_scalars += 1
            outcome = _get_row_outcome(row)
            out_rows.append(
                {
                    "id": row_id,
                    "outcome": outcome,
                    "y_pred": float(sum(vals) / max(1, len(vals))),
                }
            )

    if not out_rows:
        raise ValueError(
            "No baseline predictions could be built from data_only samples. "
            f"samples_path={samples_path} strategy={strategy} total_rows={total_rows} "
            f"matched_strategy={matched_strategy} matched_with_scalars={matched_with_scalars}. "
            "If you are running multi_expert without generating data_only prompts/samples, "
            "include data_only in --prompt-families or ensure a data_only samples file exists."
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for row in out_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return out_path


def _collect_prompt_ids(prompts_path: Path) -> set[str]:
    ids: set[str] = set()
    if not prompts_path.exists():
        return ids
    with prompts_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            row_id = row.get("id")
            if isinstance(row_id, str) and row_id:
                ids.add(row_id)
    return ids


def _baseline_has_all_ids(baseline_path: Path, required_ids: set[str]) -> bool:
    if not baseline_path.exists() or not required_ids:
        return False
    missing = set(required_ids)
    with baseline_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            row_id = row.get("id")
            if isinstance(row_id, str) and row_id in missing:
                missing.remove(row_id)
                if not missing:
                    return True
    return not missing


def _find_latest_data_only_samples(
    *,
    output_root: Path,
    model_slug: str,
    outcome_batch: str,
    exclude_run_id: str | None = None,
) -> Path | None:
    base_dir = output_root / model_slug / "perturbed" / "data_only"
    if not base_dir.exists():
        return None
    run_dirs = [p for p in base_dir.iterdir() if p.is_dir()]
    run_dirs = sorted(run_dirs, reverse=True)
    for run_dir in run_dirs:
        if exclude_run_id and run_dir.name == exclude_run_id:
            continue
        candidate = run_dir / outcome_batch / "uncalibrated" / "samples.jsonl"
        if candidate.exists():
            return candidate
    return None


def _frozen_prompt_dir(output_root: Path, run_id: str, outcome_batch: str) -> Path:
    return output_root / "frozen_prompts" / run_id / outcome_batch


def _frozen_prompts_path(output_root: Path, run_id: str, outcome_batch: str, strategy: str) -> Path:
    return _frozen_prompt_dir(output_root, run_id, outcome_batch) / f"prompts__{strategy}.jsonl"


def _frozen_paired_profiles_path(output_root: Path, run_id: str, outcome_batch: str) -> Path:
    return _frozen_prompt_dir(output_root, run_id, outcome_batch) / "paired_profiles.jsonl"


def _frozen_unperturbed_prompts_path(
    output_root: Path,
    run_id: str,
    outcome_batch: str,
    family: str,
) -> Path:
    return output_root / "frozen_prompts" / run_id / "unperturbed" / outcome_batch / f"prompts__{family}.jsonl"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prompt-families",
        default="data_only,first_person,third_person,multi_expert,paired_profile",
    )
    parser.add_argument(
        "--models",
        default=None,
        help=(
            "Optional comma-separated model list. When provided, {model} and {model_slug} are expanded "
            "inside *-args templates, and each model is run sequentially."
        ),
    )

    parser.add_argument(
        "--keep-outcomes",
        default="vax_willingness_Y,mask_when_pressure_high,stay_home_when_symptomatic",
        help="Comma-separated outcomes to keep in prompt-tournament generation.",
    )

    parser.add_argument("--build-paired-profiles-args", default=None)
    parser.add_argument("--generate-prompts-args", default=None)
    parser.add_argument("--sample-args", default=None)
    parser.add_argument("--ice-args", default=None)
    parser.add_argument("--align-args", default=None)
    parser.add_argument(
        "--ref-models",
        default="linear",
        help="Comma-separated reference model names for alignment (e.g., linear,xgb).",
    )
    parser.add_argument("--unperturbed-args", default=None)
    parser.add_argument("--calibrate-args", default=None)
    parser.add_argument("--apply-calibration-args", default=None)
    parser.add_argument("--align-calibrated-args", default=None)
    parser.add_argument("--apply-unperturbed-calibration-args", default=None)
    parser.add_argument("--calibrated-metrics-args", default=None)
    parser.add_argument(
        "--pe-ice-metrics",
        action="store_true",
        help="Compute PE/ICE directional agreement diagnostics after ICE is built.",
    )
    parser.add_argument("--pe-ice-metrics-args", default=None)
    parser.add_argument(
        "--run-tag",
        default=None,
        help="Optional subdirectory name to keep outputs isolated (e.g., run id).",
    )
    parser.add_argument(
        "--timestamp-output",
        action="store_true",
        help="Append a UTC timestamp subdirectory to output paths.",
    )
    parser.add_argument(
        "--resume-strategies",
        action="store_true",
        help=(
            "Resume per-strategy outputs (prompts/samples/ice/align). "
            "Missing artifacts are generated per strategy, then prompts are merged."
        ),
    )
    parser.add_argument(
        "--migrate-existing",
        action="store_true",
        help="Split combined prompts/samples into per-strategy files when resuming.",
    )
    parser.add_argument(
        "--skip-posthoc",
        action="store_true",
        help="Skip post-hoc diagnostics after the full run completes.",
    )
    parser.add_argument(
        "--skip-pe-calibration",
        action="store_true",
        help="Skip PE gain calibration on perturbed ICE tables.",
    )
    parser.add_argument("--posthoc-args", default=None)

    args = parser.parse_args()

    families = [f.strip() for f in args.prompt_families.split(",") if f.strip()]
    keep_outcomes = args.keep_outcomes
    keep_outcome_list = [o.strip() for o in keep_outcomes.split(",") if o.strip()]
    ref_models = [_normalize_ref_model(m) for m in args.ref_models.split(",") if m.strip()]
    if not ref_models:
        ref_models = ["linear"]

    from beam_abm.evaluation.choice.prompt_tournament import build_paired_profiles as build_paired_profiles_step
    from beam_abm.evaluation.choice.prompt_tournament import generate_prompts as generate_prompts_step
    from beam_abm.evaluation.common.apply_calibration_to_ice import main as apply_calibration_step
    from beam_abm.evaluation.common.apply_pe_calibration_to_ice import main as apply_pe_calibration_step
    from beam_abm.evaluation.common.calibrate_llm_predictions import main as calibrate_step
    from beam_abm.evaluation.common.calibrate_pe_gains import main as pe_calibrate_step
    from beam_abm.evaluation.common.compute_alignment import main as align_step
    from beam_abm.evaluation.common.compute_ice_from_samples import main as ice_step
    from beam_abm.evaluation.common.llm_sampling import main as sample_step

    project_root = Path(__file__).resolve().parents[5]
    support_dir = Path(__file__).resolve().parents[1] / "support" / "unperturbed"
    unperturbed_script = support_dir / "run_phase.py"
    unperturbed_utils = _load_local_module(support_dir / "unperturbed_utils.py", "unperturbed_utils")
    UNPERTURBED_FAMILIES = unperturbed_utils.SUPPORTED_PROMPT_FAMILIES

    run_id = pt.resolve_run_tag(args.run_tag, timestamp_output=args.timestamp_output) or utc_timestamp()
    output_root = Path(os.getenv("PROMPT_TOURNAMENT_OUTPUT_ROOT", "evaluation/output/prompt_tournament"))
    default_model = os.getenv("PROMPT_TOURNAMENT_MODEL", "openai/gpt-oss-20b")
    default_backend = os.getenv("PROMPT_TOURNAMENT_BACKEND", "vllm")
    default_strategy = os.getenv("PROMPT_TOURNAMENT_STRATEGY", "all")
    default_k = int(os.getenv("PROMPT_TOURNAMENT_K", "4"))
    default_max_model_len = int(os.getenv("PROMPT_TOURNAMENT_MAX_MODEL_LEN", "4096"))
    default_max_concurrency = int(os.getenv("PROMPT_TOURNAMENT_MAX_CONCURRENCY", "50"))
    default_model_options = json.dumps({"max-model-len": default_max_model_len})

    resume = bool(args.resume_strategies)

    if default_strategy.lower() == "all":
        prompt_strategies = list(families)
    else:
        prompt_strategies = [default_strategy]

    if "data_only" in prompt_strategies:
        prompt_strategies = ["data_only", *[s for s in prompt_strategies if s != "data_only"]]
    if "multi_expert" in prompt_strategies:
        prompt_strategies = [*[s for s in prompt_strategies if s != "multi_expert"], "multi_expert"]

    strategy_outcomes = keep_outcome_list or ["vax_willingness_Y"]
    models = _parse_models(args.models) or [default_model]

    # Unperturbed sampling is executed via a separate subprocess runner.
    # To avoid surprising behaviour (e.g. --sample-args backend not applying),
    # extract a few sampling overrides once and reuse them.
    sample_args_overrides: list[str] = []
    if args.sample_args:
        sample_args_overrides = pt.split_args(args.sample_args)
    override_backend = _get_arg_value(sample_args_overrides, "--backend")
    override_k = _get_arg_value(sample_args_overrides, "--k")
    override_max_concurrency = _get_arg_value(sample_args_overrides, "--max-concurrency")
    override_model_options = _get_arg_value(sample_args_overrides, "--model-options")
    override_model_options_file = _get_arg_value(sample_args_overrides, "--model-options-file")

    model_options_payload = _load_model_options_payload(
        override_model_options=override_model_options,
        override_model_options_file=override_model_options_file,
        default_model_options=default_model_options,
    )
    reasoning_effort = _extract_reasoning_effort(model_options_payload)

    for model_name in models:
        model_slug = normalize_model_slug(model_name, reasoning_effort=reasoning_effort)
        start_time_utc = utc_timestamp_iso()
        run_start = time.time()
        outcome_batch = PromptTournamentOutputPaths(
            run_type="perturbed",
            strategy="data_only",
            run_id=run_id,
            outcomes=strategy_outcomes,
            calibrated=False,
            model_slug=model_slug,
            root=output_root,
        ).outcome_batch_name()

        for strategy in prompt_strategies:
            paths_uncal = PromptTournamentOutputPaths(
                run_type="perturbed",
                strategy=strategy,
                run_id=run_id,
                outcomes=strategy_outcomes,
                calibrated=False,
                model_slug=model_slug,
                root=output_root,
            )
            strategy_dir = paths_uncal.base_dir()
            prompts_path = paths_uncal.prompts_path()
            samples_path = paths_uncal.samples_path()
            ice_path = paths_uncal.ice_path()

            strategy_dir.mkdir(parents=True, exist_ok=True)

            frozen_prompts = _frozen_prompts_path(output_root, run_id, outcome_batch, strategy)
            if frozen_prompts.exists() and not prompts_path.exists():
                _copy_if_missing(frozen_prompts, prompts_path)
            if not frozen_prompts.exists() and prompts_path.exists():
                _copy_if_missing(prompts_path, frozen_prompts)

            if not (resume and prompts_path.exists()):
                if not prompts_path.exists():
                    override_args = (
                        pt.split_args(
                            args.generate_prompts_args,
                            model=model_name,
                            model_slug=model_slug,
                            strategy=strategy,
                        )
                        if args.generate_prompts_args
                        else []
                    )
                    include_input = not (_has_flag(override_args, "--in") or _has_flag(override_args, "--in-dir"))
                    base_args = _build_default_generate_args(
                        prompts_path=prompts_path,
                        strategy=strategy,
                        keep_outcomes=keep_outcomes,
                        output_root=output_root,
                        run_id=run_id,
                        outcome_batch=outcome_batch,
                        keep_outcome_list=keep_outcome_list,
                        include_input=include_input,
                        build_paired_profiles_step=build_paired_profiles_step,
                    )
                    generate_args = [*base_args, *override_args]
                    pt.run_step(generate_prompts_step.main, generate_args)
                    if not frozen_prompts.exists():
                        _copy_if_missing(prompts_path, frozen_prompts)

            if not (resume and samples_path.exists()):
                if args.sample_args:
                    sample_args = pt.split_args(
                        args.sample_args,
                        model=model_name,
                        model_slug=model_slug,
                        strategy=strategy,
                    )
                    sample_args = _set_arg_value(sample_args, "--in", str(prompts_path))
                    sample_args = _set_arg_value(sample_args, "--out", str(samples_path))
                else:
                    sample_args = pt.split_args(
                        " ".join(
                            [
                                f"--in {prompts_path}",
                                f"--out {samples_path}",
                                f"--backend {default_backend}",
                                f"--model {model_name}",
                                "--outcome-scale continuous",
                                f"--k {default_k}",
                                "--two-pass",
                                f"--model-options '{default_model_options}'",
                                "--max-model-len-by-strategy-file evaluation/specs/prompt_tournament_max_model_len.json",
                                f"--run-id {run_id}",
                                f"--max-concurrency {default_max_concurrency}",
                            ]
                        ),
                        model=model_name,
                        model_slug=model_slug,
                        strategy=strategy,
                    )

                if "--run-id" not in sample_args:
                    sample_args = _set_arg_value(sample_args, "--run-id", run_id)
                if "--model" not in sample_args:
                    sample_args = _set_arg_value(sample_args, "--model", model_name)

                if strategy == "multi_expert":
                    baseline_path = paths_uncal.base_dir() / "baseline_predictions__data_only.jsonl"
                    data_only_paths = PromptTournamentOutputPaths(
                        run_type="perturbed",
                        strategy="data_only",
                        run_id=run_id,
                        outcomes=strategy_outcomes,
                        calibrated=False,
                        model_slug=model_slug,
                        root=output_root,
                    )
                    data_only_samples = data_only_paths.samples_path()
                    baseline_ids = _collect_prompt_ids(prompts_path)
                    baseline_ok = _baseline_has_all_ids(baseline_path, baseline_ids)
                    if (not baseline_path.exists()) or (data_only_samples.exists() and not baseline_ok):
                        samples_source = data_only_samples
                        if not samples_source.exists():
                            fallback = _find_latest_data_only_samples(
                                output_root=output_root,
                                model_slug=model_slug,
                                outcome_batch=outcome_batch,
                                exclude_run_id=run_id,
                            )
                            if fallback is not None:
                                samples_source = fallback
                        if not samples_source.exists():
                            raise ValueError(
                                "multi_expert requires data_only samples. "
                                "No same-run samples and no prior run with matching outcomes were found."
                            )
                        _build_baseline_predictions(
                            samples_path=samples_source,
                            out_path=baseline_path,
                            strategy="data_only",
                        )
                    sample_args = _set_arg_value(sample_args, "--strategy", "multi_expert")
                    sample_args = _set_arg_value(sample_args, "--baseline-predictions", str(baseline_path))
                    sample_args = _set_arg_value(sample_args, "--baseline-join-key", "id")
                    sample_args = _set_arg_value(sample_args, "--baseline-col", "y_pred")

                pt.run_step(sample_step, sample_args)

            if not (resume and ice_path.exists()):
                if args.ice_args:
                    ice_args = pt.split_args(
                        args.ice_args,
                        model=model_name,
                        model_slug=model_slug,
                        strategy=strategy,
                    )
                    ice_args = _set_arg_value(ice_args, "--in", str(samples_path))
                    ice_args = _set_arg_value(ice_args, "--out", str(ice_path))
                else:
                    ice_args = pt.split_args(
                        f"--in {samples_path} --out {ice_path}",
                        model=model_name,
                        model_slug=model_slug,
                        strategy=strategy,
                    )
                pt.run_step(ice_step, ice_args)

            summary_paths_by_ref: dict[str, list[Path]] = {}
            order_paths_by_ref: dict[str, list[Path]] = {}
            for outcome in strategy_outcomes:
                for ref_model in ref_models:
                    ref_ice = _ref_ice_path(outcome, ref_model)
                    if not ref_ice.exists():
                        print(
                            f"Skipping alignment for {strategy}::{outcome}::{ref_model}: missing ref ICE at {ref_ice}"
                        )
                        continue
                    ref_key = _normalize_ref_model(ref_model)
                    metrics_suffix = "" if ref_key == "linear" else f"__ref_{ref_key}"
                    metrics_out = strategy_dir / f"_summary_metrics__{outcome}{metrics_suffix}.csv"
                    order_out = strategy_dir / f"_order_effects__{outcome}{metrics_suffix}.csv"
                    summary_paths_by_ref.setdefault(ref_key, []).append(metrics_out)
                    order_paths_by_ref.setdefault(ref_key, []).append(order_out)
                    if not (resume and metrics_out.exists() and order_out.exists()):
                        align_args = pt.split_args(
                            " ".join(
                                [
                                    f"--ref {ref_ice}",
                                    f"--llm {ice_path}",
                                    f"--out {metrics_out}",
                                    f"--order-out {order_out}",
                                ]
                            ),
                            model=model_name,
                            model_slug=model_slug,
                            strategy=strategy,
                        )
                        pt.run_step(align_step, align_args)

            for ref_key, summary_paths in summary_paths_by_ref.items():
                summary_frames = [pd.read_csv(path) for path in summary_paths if path.exists()]
                if summary_frames:
                    summary_rows = pd.concat(summary_frames, ignore_index=True)
                    if ref_key == "linear":
                        summary_out = paths_uncal.summary_metrics_path()
                    else:
                        summary_out = paths_uncal.base_dir() / f"summary_metrics__ref_{ref_key}.csv"
                    summarize_alignment_rows(summary_rows).to_csv(summary_out, index=False)

                    row_level_path = paths_uncal.row_level_path()
                    if row_level_path.exists():
                        prefix = "align_" if ref_key == "linear" else f"align_{ref_key}_"
                        for align_path in summary_paths:
                            if not align_path.exists():
                                continue
                            merge_alignment_into_row_level(
                                row_level_path=str(row_level_path),
                                alignment_path=str(align_path),
                                on_cols=[
                                    ("dataset_id", "id"),
                                    ("perturbation_type", "target"),
                                    ("outcome", "outcome"),
                                    ("strategy", "strategy"),
                                ],
                                prefix=prefix,
                            )

            if strategy == "paired_profile" and order_paths_by_ref:
                for ref_key, order_paths in order_paths_by_ref.items():
                    order_frames = [pd.read_csv(path) for path in order_paths if path.exists()]
                    if not order_frames:
                        continue
                    order_out = strategy_dir / (
                        "order_effects.csv" if ref_key == "linear" else f"order_effects__ref_{ref_key}.csv"
                    )
                    pd.concat(order_frames, ignore_index=True).to_csv(order_out, index=False)

            for summary_paths in summary_paths_by_ref.values():
                for path in summary_paths:
                    path.unlink(missing_ok=True)
            for order_paths in order_paths_by_ref.values():
                for path in order_paths:
                    path.unlink(missing_ok=True)

            end_time_utc = utc_timestamp_iso()
            run_meta = build_run_metadata(
                run_id=run_id,
                run_type="perturbed",
                strategy=strategy,
                outcomes=strategy_outcomes,
                start_time_utc=start_time_utc,
                end_time_utc=end_time_utc,
                runtime_seconds=time.time() - run_start,
                config_snapshot={
                    "backend": default_backend,
                    "model": model_name,
                    "model_slug": model_slug,
                    "k": default_k,
                    "model_options": default_model_options,
                    "prompt_families": families,
                    "keep_outcomes": strategy_outcomes,
                },
                dataset_inputs={"anchors_root": "empirical/output/anchors/pe/mutable_engines/reduced"},
                anchors_inputs={
                    "ref_models": ref_models,
                    "ref_ice": "empirical/output/anchors/pe/mutable_engines/reduced/<outcome>/ice_ref.csv",
                    "ref_ice_template": "empirical/output/anchors/pe/mutable_engines/reduced/<outcome>/ice_ref__{ref}.csv",
                },
                perturbation_inputs={"lever_config": "config/levers.json"},
            )
            write_run_metadata(paths_uncal.run_metadata_path(), run_meta)

        unperturbed_families = [family for family in families if family in UNPERTURBED_FAMILIES]
        calibration_by_family: dict[str, Path] = {}
        if unperturbed_families:
            metrics_filename = unperturbed_utils.metrics_filename
            for family in unperturbed_families:
                uncal_paths = PromptTournamentOutputPaths(
                    run_type="unperturbed",
                    strategy=family,
                    run_id=run_id,
                    outcomes=strategy_outcomes,
                    calibrated=False,
                    model_slug=model_slug,
                    root=output_root,
                )
                base_dir = uncal_paths.base_dir()
                base_dir.mkdir(parents=True, exist_ok=True)

                frozen_unperturbed_prompts = _frozen_unperturbed_prompts_path(
                    output_root,
                    run_id,
                    outcome_batch,
                    family,
                )
                if not frozen_unperturbed_prompts.exists() and uncal_paths.prompts_path().exists():
                    _copy_if_missing(uncal_paths.prompts_path(), frozen_unperturbed_prompts)
                if not frozen_unperturbed_prompts.exists():
                    frozen_unperturbed_prompts.parent.mkdir(parents=True, exist_ok=True)
                    gen_args = [
                        "generate-prompts",
                        "--outdir",
                        str(frozen_unperturbed_prompts.parent),
                        "--prompt-family",
                        family,
                        "--outcomes",
                        ",".join(strategy_outcomes),
                    ]
                    pt.run_step_subprocess(unperturbed_script, gen_args, cwd=project_root)

                if not (resume and uncal_paths.prompts_path().exists()):
                    _copy_if_missing(frozen_unperturbed_prompts, uncal_paths.prompts_path())

                if not (resume and uncal_paths.samples_path().exists()):
                    unperturbed_backend = override_backend or str(default_backend)
                    unperturbed_k = override_k or str(default_k)
                    unperturbed_max_concurrency = override_max_concurrency or str(default_max_concurrency)
                    sample_args = [
                        "sample",
                        "--prompts",
                        str(frozen_unperturbed_prompts),
                        "--outdir",
                        str(base_dir),
                        "--prompt-family",
                        family,
                        "--backend",
                        unperturbed_backend,
                        "--models",
                        str(model_name),
                        "--k",
                        unperturbed_k,
                        "--run-id",
                        str(run_id),
                        "--max-concurrency",
                        unperturbed_max_concurrency,
                    ]

                    # Prefer file-based options when provided.
                    if override_model_options_file:
                        sample_args.extend(["--model-options-file", override_model_options_file])
                    else:
                        sample_args.extend(["--model-options", str(override_model_options or default_model_options)])

                    pt.run_step_subprocess(unperturbed_script, sample_args, cwd=project_root)

                metrics_out = base_dir / metrics_filename(family=family, pred_col="y_pred")
                if not (resume and metrics_out.exists()):
                    metrics_args = [
                        "metrics",
                        "--outdir",
                        str(base_dir),
                        "--prompt-family",
                        family,
                    ]
                    pt.run_step_subprocess(unperturbed_script, metrics_args, cwd=project_root)

                if metrics_out.exists():
                    metrics_df = pd.read_csv(metrics_out)
                    metrics_df["calibration_status"] = "uncalibrated"
                    metrics_df["run_id"] = run_id
                    metrics_df["strategy"] = family
                    metrics_df.to_csv(uncal_paths.summary_metrics_path(), index=False)

                uncal_meta = build_run_metadata(
                    run_id=run_id,
                    run_type="unperturbed",
                    strategy=family,
                    outcomes=strategy_outcomes,
                    start_time_utc=start_time_utc,
                    end_time_utc=utc_timestamp_iso(),
                    runtime_seconds=time.time() - run_start,
                    config_snapshot={
                        "backend": default_backend,
                        "model": model_name,
                        "model_slug": model_slug,
                        "k": default_k,
                    },
                    dataset_inputs={"clean_csv": "preprocess/output/clean_processed_survey.csv"},
                    anchors_inputs={"lever_config": "config/levers.json"},
                )
                write_run_metadata(uncal_paths.run_metadata_path(), uncal_meta)

                cal_paths = PromptTournamentOutputPaths(
                    run_type="unperturbed",
                    strategy=family,
                    run_id=run_id,
                    outcomes=strategy_outcomes,
                    calibrated=True,
                    model_slug=model_slug,
                    root=output_root,
                )
                cal_dir = cal_paths.base_dir()
                cal_dir.mkdir(parents=True, exist_ok=True)

                calibration_out = cal_dir / "calibration.json"
                calibrated_predictions_out = cal_dir / "predictions_calibrated.csv"
                if not (resume and calibration_out.exists() and calibrated_predictions_out.exists()):
                    if args.calibrate_args:
                        calibrate_args = pt.split_args(
                            args.calibrate_args,
                            model=model_name,
                            model_slug=model_slug,
                            strategy=family,
                        )
                        calibrate_args = _set_arg_value(calibrate_args, "--in", str(base_dir))
                        calibrate_args = _set_arg_value(calibrate_args, "--out", str(calibrated_predictions_out))
                        calibrate_args = _set_arg_value(calibrate_args, "--calibration-out", str(calibration_out))
                    else:
                        calibrate_args = pt.split_args(
                            " ".join(
                                [
                                    f"--in {base_dir}",
                                    f"--out {calibrated_predictions_out}",
                                    f"--calibration-out {calibration_out}",
                                ]
                            ),
                            model=model_name,
                            model_slug=model_slug,
                            strategy=family,
                        )
                    pt.run_step(calibrate_step, calibrate_args)

                calibration_by_family[family] = calibration_out

                metrics_cal_out = cal_dir / metrics_filename(family=family, pred_col="y_pred_cal")
                if not (resume and metrics_cal_out.exists()):
                    metrics_args = [
                        "metrics",
                        "--predictions",
                        str(calibrated_predictions_out),
                        "--outdir",
                        str(cal_dir),
                        "--prompt-family",
                        family,
                        "--metrics-only",
                        "--pred-col",
                        "y_pred_cal",
                    ]
                    pt.run_step_subprocess(unperturbed_script, metrics_args, cwd=project_root)

                if metrics_cal_out.exists():
                    metrics_df = pd.read_csv(metrics_cal_out)
                    metrics_df["calibration_status"] = "calibrated"
                    metrics_df["run_id"] = run_id
                    metrics_df["strategy"] = family
                    metrics_df.to_csv(cal_paths.summary_metrics_path(), index=False)

                cal_meta = build_run_metadata(
                    run_id=run_id,
                    run_type="unperturbed",
                    strategy=family,
                    outcomes=strategy_outcomes,
                    start_time_utc=start_time_utc,
                    end_time_utc=utc_timestamp_iso(),
                    runtime_seconds=time.time() - run_start,
                    config_snapshot={
                        "backend": default_backend,
                        "model": model_name,
                        "model_slug": model_slug,
                        "k": default_k,
                        "calibration": str(calibration_out),
                    },
                    dataset_inputs={"clean_csv": "preprocess/output/clean_processed_survey.csv"},
                    anchors_inputs={"lever_config": "config/levers.json"},
                )
                write_run_metadata(cal_paths.run_metadata_path(), cal_meta)

        if calibration_by_family:
            default_calibration = next(iter(calibration_by_family.values()))
            for strategy in prompt_strategies:
                cal_source = calibration_by_family.get(strategy, default_calibration)
                if cal_source is None:
                    continue
                paths_uncal = PromptTournamentOutputPaths(
                    run_type="perturbed",
                    strategy=strategy,
                    run_id=run_id,
                    outcomes=strategy_outcomes,
                    calibrated=False,
                    model_slug=model_slug,
                    root=output_root,
                )
                paths_cal = PromptTournamentOutputPaths(
                    run_type="perturbed",
                    strategy=strategy,
                    run_id=run_id,
                    outcomes=strategy_outcomes,
                    calibrated=True,
                    model_slug=model_slug,
                    root=output_root,
                )
                ice_in = paths_uncal.ice_path()
                ice_out = paths_cal.base_dir() / "ice_llm_calibrated.csv"
                if not ice_in.exists():
                    continue
                if not (resume and ice_out.exists()):
                    apply_args = pt.split_args(
                        " ".join(
                            [
                                f"--llm-ice {ice_in}",
                                f"--calibration {cal_source}",
                                f"--out {ice_out}",
                            ]
                        ),
                        model=model_name,
                        model_slug=model_slug,
                        strategy=strategy,
                    )
                    pt.run_step(apply_calibration_step, apply_args)

                summary_paths_by_ref: dict[str, list[Path]] = {}
                order_paths_by_ref: dict[str, list[Path]] = {}
                for outcome in strategy_outcomes:
                    for ref_model in ref_models:
                        ref_ice = _ref_ice_path(outcome, ref_model)
                        if not ref_ice.exists():
                            print(
                                "Skipping calibrated alignment for "
                                f"{strategy}::{outcome}::{ref_model}: missing ref ICE at {ref_ice}"
                            )
                            continue
                        ref_key = _normalize_ref_model(ref_model)
                        metrics_suffix = "" if ref_key == "linear" else f"__ref_{ref_key}"
                        metrics_out = paths_cal.base_dir() / f"_summary_metrics__{outcome}{metrics_suffix}.csv"
                        order_out = paths_cal.base_dir() / f"_order_effects__{outcome}{metrics_suffix}.csv"
                        summary_paths_by_ref.setdefault(ref_key, []).append(metrics_out)
                        order_paths_by_ref.setdefault(ref_key, []).append(order_out)
                        if not (resume and metrics_out.exists() and order_out.exists()):
                            align_args = pt.split_args(
                                " ".join(
                                    [
                                        f"--ref {ref_ice}",
                                        f"--llm {ice_out}",
                                        f"--out {metrics_out}",
                                        f"--order-out {order_out}",
                                    ]
                                ),
                                model=model_name,
                                model_slug=model_slug,
                                strategy=strategy,
                            )
                            pt.run_step(align_step, align_args)

                for ref_key, summary_paths in summary_paths_by_ref.items():
                    summary_frames = [pd.read_csv(path) for path in summary_paths if path.exists()]
                    if summary_frames:
                        summary_rows = pd.concat(summary_frames, ignore_index=True)
                        if ref_key == "linear":
                            summary_out = paths_cal.summary_metrics_path()
                        else:
                            summary_out = paths_cal.base_dir() / f"summary_metrics__ref_{ref_key}.csv"
                        summarize_alignment_rows(summary_rows).to_csv(summary_out, index=False)

                        row_level_path = paths_cal.row_level_path()
                        if row_level_path.exists():
                            prefix = "align_" if ref_key == "linear" else f"align_{ref_key}_"
                            for align_path in summary_paths:
                                if not align_path.exists():
                                    continue
                                merge_alignment_into_row_level(
                                    row_level_path=str(row_level_path),
                                    alignment_path=str(align_path),
                                    on_cols=[
                                        ("dataset_id", "id"),
                                        ("perturbation_type", "target"),
                                        ("outcome", "outcome"),
                                        ("strategy", "strategy"),
                                    ],
                                    prefix=prefix,
                                )

            if not args.skip_pe_calibration:
                pe_dir = strategy_dir / "pe_calibrated"
                pe_dir.mkdir(parents=True, exist_ok=True)

                pe_summary_paths_by_ref: dict[str, list[Path]] = {}
                pe_order_paths_by_ref: dict[str, list[Path]] = {}

                for ref_model in ref_models:
                    ref_key = _normalize_ref_model(ref_model)
                    metrics_suffix = "" if ref_key == "linear" else f"__ref_{ref_key}"
                    ref_paths = [_ref_ice_path(outcome=o, ref_model=ref_model) for o in strategy_outcomes]
                    missing = [p for p in ref_paths if not p.exists()]
                    if missing:
                        print(f"Skipping PE calibration for {strategy}::{ref_model}: missing ref ICE for outcomes.")
                        continue

                    ref_concat = pe_dir / f"ref_ice{metrics_suffix}.csv"
                    if not (resume and ref_concat.exists()):
                        ref_frames = [pd.read_csv(p) for p in ref_paths]
                        pd.concat(ref_frames, ignore_index=True).to_csv(ref_concat, index=False)

                    pe_cal_path = pe_dir / f"pe_calibration{metrics_suffix}.json"
                    pe_ice_out = pe_dir / f"ice_llm_pe_calibrated{metrics_suffix}.csv"

                    if not (resume and pe_cal_path.exists() and pe_ice_out.exists()):
                        pe_cal_args = pt.split_args(
                            " ".join(
                                [
                                    f"--ref-ice {ref_concat}",
                                    f"--llm-ice {ice_path}",
                                    f"--out {pe_cal_path}",
                                    "--group-cols outcome,target,strategy",
                                ]
                            ),
                            model=model_name,
                            model_slug=model_slug,
                            strategy=strategy,
                        )
                        pt.run_step(pe_calibrate_step, pe_cal_args)

                        pe_apply_args = pt.split_args(
                            " ".join(
                                [
                                    f"--llm-ice {ice_path}",
                                    f"--calibration {pe_cal_path}",
                                    f"--out {pe_ice_out}",
                                ]
                            ),
                            model=model_name,
                            model_slug=model_slug,
                            strategy=strategy,
                        )
                        pt.run_step(apply_pe_calibration_step, pe_apply_args)

                    for outcome in strategy_outcomes:
                        ref_ice_outcome = _ref_ice_path(outcome, ref_model)
                        metrics_out = pe_dir / f"_summary_metrics__{outcome}{metrics_suffix}.csv"
                        order_out = pe_dir / f"_order_effects__{outcome}{metrics_suffix}.csv"
                        pe_summary_paths_by_ref.setdefault(ref_key, []).append(metrics_out)
                        pe_order_paths_by_ref.setdefault(ref_key, []).append(order_out)
                        if not (resume and metrics_out.exists() and order_out.exists()):
                            align_args = pt.split_args(
                                " ".join(
                                    [
                                        f"--ref {ref_ice_outcome}",
                                        f"--llm {pe_ice_out}",
                                        f"--out {metrics_out}",
                                        f"--order-out {order_out}",
                                    ]
                                ),
                                model=model_name,
                                model_slug=model_slug,
                                strategy=strategy,
                            )
                            pt.run_step(align_step, align_args)

                for ref_key, summary_paths in pe_summary_paths_by_ref.items():
                    summary_frames = [pd.read_csv(path) for path in summary_paths if path.exists()]
                    if summary_frames:
                        summary_rows = pd.concat(summary_frames, ignore_index=True)
                        if ref_key == "linear":
                            summary_out = pe_dir / "summary_metrics.csv"
                        else:
                            summary_out = pe_dir / f"summary_metrics__ref_{ref_key}.csv"
                        summarize_alignment_rows(summary_rows).to_csv(summary_out, index=False)

                if strategy == "paired_profile" and pe_order_paths_by_ref:
                    for ref_key, order_paths in pe_order_paths_by_ref.items():
                        order_frames = [pd.read_csv(path) for path in order_paths if path.exists()]
                        if not order_frames:
                            continue
                        order_out = pe_dir / (
                            "order_effects.csv" if ref_key == "linear" else f"order_effects__ref_{ref_key}.csv"
                        )
                        pd.concat(order_frames, ignore_index=True).to_csv(order_out, index=False)

                for summary_paths in pe_summary_paths_by_ref.values():
                    for path in summary_paths:
                        path.unlink(missing_ok=True)
                for order_paths in pe_order_paths_by_ref.values():
                    for path in order_paths:
                        path.unlink(missing_ok=True)

                if strategy == "paired_profile" and order_paths_by_ref:
                    for ref_key, order_paths in order_paths_by_ref.items():
                        order_frames = [pd.read_csv(path) for path in order_paths if path.exists()]
                        if not order_frames:
                            continue
                        order_out = paths_cal.base_dir() / (
                            "order_effects.csv" if ref_key == "linear" else f"order_effects__ref_{ref_key}.csv"
                        )
                        pd.concat(order_frames, ignore_index=True).to_csv(order_out, index=False)

                for summary_paths in summary_paths_by_ref.values():
                    for path in summary_paths:
                        path.unlink(missing_ok=True)
                for order_paths in order_paths_by_ref.values():
                    for path in order_paths:
                        path.unlink(missing_ok=True)

                uncal_row_level = paths_uncal.row_level_path()
                cal_row_level = paths_cal.row_level_path()
                if uncal_row_level.exists() and not cal_row_level.exists():
                    cal_df = pd.read_csv(uncal_row_level)
                    cal_df["calibration_status"] = "calibrated"
                    cal_df.to_csv(cal_row_level, index=False)

                cal_meta = build_run_metadata(
                    run_id=run_id,
                    run_type="perturbed",
                    strategy=strategy,
                    outcomes=strategy_outcomes,
                    start_time_utc=start_time_utc,
                    end_time_utc=utc_timestamp_iso(),
                    runtime_seconds=time.time() - run_start,
                    config_snapshot={
                        "backend": default_backend,
                        "model": model_name,
                        "model_slug": model_slug,
                        "k": default_k,
                        "calibration": str(cal_source),
                    },
                    dataset_inputs={"anchors_root": "empirical/output/anchors/pe/mutable_engines/reduced"},
                    anchors_inputs={
                        "ref_models": ref_models,
                        "ref_ice": "empirical/output/anchors/pe/mutable_engines/reduced/<outcome>/ice_ref.csv",
                        "ref_ice_template": "empirical/output/anchors/pe/mutable_engines/reduced/<outcome>/ice_ref__{ref}.csv",
                    },
                    perturbation_inputs={"lever_config": "config/levers.json"},
                )
                write_run_metadata(paths_cal.run_metadata_path(), cal_meta)

        if not args.skip_posthoc:
            posthoc_script = Path(__file__).with_name("posthoc_tests.py")
            if posthoc_script.exists():
                posthoc_args = pt.split_args(
                    args.posthoc_args,
                    model=model_name,
                    model_slug=model_slug,
                )
                posthoc_args = _set_arg_value(posthoc_args, "--output-root", str(output_root))
                posthoc_args = _set_arg_value(posthoc_args, "--run-id", str(run_id))
                posthoc_args = _set_arg_value(posthoc_args, "--model-slug", str(model_slug))
                pt.run_step_subprocess(posthoc_script, posthoc_args, cwd=project_root)


if __name__ == "__main__":
    main()
