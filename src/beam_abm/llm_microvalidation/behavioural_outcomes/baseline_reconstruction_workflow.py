"""Run choice-validation unperturbed pipeline with canonical outputs.

Output contract:

evaluation/output/choice_validation/unperturbed/
  _runs/<run_id>/<outcome>/<model>/<strategy>/{uncalibrated,calibrated}/...
  <outcome>/<model>/<strategy>/{uncalibrated,calibrated}/...
  _summary/...
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd

from beam_abm.cli import parser_compat as cli
from beam_abm.llm.schemas.predictions import extract_scalar_prediction
from beam_abm.llm_microvalidation.artifacts import utc_timestamp
from beam_abm.llm_microvalidation.behavioural_outcomes._canonicalize_ids import normalize_model_slug
from beam_abm.llm_microvalidation.behavioural_outcomes._canonicalize_sample_rebuild import (
    find_samples_jsonl,
    rebuild_canonical_leaf,
)
from beam_abm.llm_microvalidation.behavioural_outcomes._canonicalize_summary import (
    write_models_strategies_summary,
    write_strategies_summary,
)
from beam_abm.llm_microvalidation.behavioural_outcomes.baseline_reconstruction_posthoc import (
    run_cli as run_unperturbed_posthoc,
)
from beam_abm.llm_microvalidation.behavioural_outcomes.run_conditional_knockout import (
    run_cli as run_conditional_knockout,
)
from beam_abm.llm_microvalidation.behavioural_outcomes.support.baseline_reconstruction.run_phase import (
    run_cli as run_unperturbed_phase,
)
from beam_abm.llm_microvalidation.shared.calibrate_llm_predictions import run_cli as calibrate_llm_predictions
from beam_abm.llm_microvalidation.utils.jsonl import numpy_json_default, read_jsonl, write_jsonl


def _parse_csv_list(value: str | None) -> list[str]:
    if not value:
        return []
    return [v.strip() for v in value.split(",") if v.strip()]


def _summaries_root(root: Path) -> Path:
    return root / "_summary"


def _load_model_options_payload(args: cli.Namespace) -> object | None:
    if args.model_options and args.model_options_file:
        raise ValueError("Provide only one of --model-options or --model-options-file")
    if args.model_options_file:
        return json.loads(Path(args.model_options_file).read_text(encoding="utf-8"))
    if args.model_options:
        return json.loads(str(args.model_options))
    return None


def _select_model_options(payload: object | None, *, model: str, models: list[str]) -> dict | None:
    if payload is None:
        return None
    if not isinstance(payload, dict):
        return None

    model_set = set(models)
    payload_keys = {str(k) for k in payload.keys()}
    is_map = payload_keys.issubset(model_set) and all(isinstance(v, dict) for v in payload.values())
    if is_map:
        selected = payload.get(model)
        return selected if isinstance(selected, dict) else None

    # Otherwise treat as a global options object.
    return payload


def _extract_reasoning_effort(model_options: dict | None) -> str | None:
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


def rebuild_summaries(root: Path) -> None:
    global_rows: list[dict] = []

    for outcome_dir in sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith("_")):
        outcome = outcome_dir.name
        outcome_rows: list[dict] = []

        for model_dir in sorted(p for p in outcome_dir.iterdir() if p.is_dir() and not p.name.startswith("_")):
            model = model_dir.name
            strategy_rows: list[dict] = []

            for strategy_dir in sorted(p for p in model_dir.iterdir() if p.is_dir() and not p.name.startswith("_")):
                strategy = strategy_dir.name
                row: dict = {"outcome": outcome, "model": model, "strategy": strategy}
                uncal = strategy_dir / "uncalibrated" / "summary_metrics.csv"
                cal = strategy_dir / "calibrated" / "summary_metrics.csv"

                if uncal.exists():
                    df = pd.read_csv(uncal)
                    if not df.empty:
                        first = df.iloc[0].to_dict()
                        row.update(
                            {
                                f"uncal_{k}": first.get(k)
                                for k in ["type", "n", "mae", "rmse", "spearman", "brier", "auc"]
                            }
                        )
                if cal.exists():
                    df = pd.read_csv(cal)
                    if not df.empty:
                        first = df.iloc[0].to_dict()
                        row.update(
                            {f"cal_{k}": first.get(k) for k in ["type", "n", "mae", "rmse", "spearman", "brier", "auc"]}
                        )

                strategy_rows.append({k: v for k, v in row.items() if k not in {"outcome", "model"}})
                outcome_rows.append(row)
                global_rows.append(row)

            write_strategies_summary(outcome_dir=root / outcome, model=model, strategy_rows=strategy_rows)

        write_models_strategies_summary(outcome_dir=root / outcome, rows=outcome_rows)

    if global_rows:
        out_dir = _summaries_root(root)
        out_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(global_rows).to_csv(out_dir / "all_outcomes_models_strategies.csv", index=False)


def run_behavioural_outcomes_baseline_reconstruction(argv: list[str] | None = None) -> None:
    parser = cli.ArgumentParser()
    parser.add_argument(
        "--output-root",
        default="evaluation/output/choice_validation/unperturbed",
        help="Choice-validation unperturbed root (contains _runs and canonical).",
    )
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--outcomes", default="vax_willingness_Y,mask_when_pressure_high,stay_home_when_symptomatic")
    parser.add_argument("--models", default="openai/gpt-oss-20b")
    parser.add_argument("--prompt-families", default="data_only,first_person,third_person,multi_expert")

    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--backend", choices=["vllm", "azure_openai"], default="vllm")
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--max-concurrency", type=int, default=64)
    parser.add_argument("--model-options", default=None)
    parser.add_argument("--model-options-file", default=None)
    parser.add_argument(
        "--skip-posthoc",
        action="store_true",
        help="Skip post-hoc diagnostics (tail-bias, dispersion, slice stability).",
    )
    parser.add_argument(
        "--repair-only",
        action="store_true",
        help=(
            "Repair an existing run from already-sampled batch outputs (no prompt generation or sampling). "
            "Overwrites per-outcome samples, recomputes metrics, calibration, and canonical leaves."
        ),
    )
    parser.add_argument(
        "--repair-samples-only",
        action="store_true",
        help="Only rewrite per-outcome samples from batch outputs, then exit.",
    )
    parser.add_argument(
        "--skip-block-ablation",
        action="store_true",
        help="Skip canonical unperturbed block-ablation summary generation.",
    )
    parser.add_argument(
        "--block-ablation-output-root",
        default="evaluation/output/choice_validation/unperturbed_block_ablation_canonical",
        help="Output root for canonical unperturbed block-ablation artifacts.",
    )

    args = parser.parse_args(argv)

    if bool(args.repair_samples_only) and bool(args.repair_only):
        raise ValueError("Provide only one of --repair-only or --repair-samples-only")

    root = Path(args.output_root)
    runs_root = root / "_runs"
    locks_root = root / "_locks"
    run_id = args.run_id or utc_timestamp()

    outcomes = _parse_csv_list(args.outcomes)
    families = _parse_csv_list(args.prompt_families)

    repair_mode = bool(args.repair_only) or bool(args.repair_samples_only)
    model_options_payload = _load_model_options_payload(args) if not repair_mode else None

    batch_root = runs_root / run_id / "_batch"
    if repair_mode:
        if not batch_root.exists():
            raise FileNotFoundError(
                f"Missing batch outputs at {batch_root}. "
                "Cannot repair without existing _batch/<model>/<family>/uncalibrated/samples__*.jsonl"
            )
        model_norms = sorted(p.name for p in batch_root.iterdir() if p.is_dir())
        if not model_norms:
            raise FileNotFoundError(f"No model folders under {batch_root}")
    else:
        models = _parse_csv_list(args.models)
        model_options_payload = _load_model_options_payload(args)

    def _row_outcome(row: dict) -> str | None:
        raw = row.get("outcome")
        if isinstance(raw, str) and raw.strip():
            return raw.strip()
        meta = row.get("metadata")
        if isinstance(meta, dict):
            m = meta.get("outcome")
            if isinstance(m, str) and m.strip():
                return m.strip()
        rid = row.get("dataset_id") or row.get("id")
        if isinstance(rid, str) and rid.strip() and "__" in rid:
            return rid.split("__", 1)[0].strip() or None
        return None

    def _as_per_outcome_row(*, row: dict, outcome: str) -> dict | None:
        """Project a multi-outcome samples row into a single-outcome row.

        The downstream unperturbed metrics + canonicalization code expects:
        - row['outcome'] (or metadata['outcome'])
        - row['outcome_type'] (or metadata['outcome_type'])
        - y_true in metadata['y_true'] or row['y_true']
        - numeric sample['prediction_scalar'] values
        """

        current = _row_outcome(row)
        # If the row already represents a specific outcome, never remap it.
        if current is not None and current != outcome:
            return None

        meta = row.get("metadata")
        meta = meta if isinstance(meta, dict) else {}

        y_true_by_outcome = meta.get("y_true_by_outcome")
        outcome_type_by_outcome = meta.get("outcome_type_by_outcome")
        outcomes_list = row.get("outcomes")
        if outcomes_list is None:
            outcomes_list = meta.get("outcomes")

        has_multi_truth = isinstance(y_true_by_outcome, dict) and outcome in y_true_by_outcome
        # Accept both true multi-outcome rows and per-outcome-expanded rows that still
        # carry y_true_by_outcome/outcome_type_by_outcome.
        if not has_multi_truth:
            return None
        if isinstance(outcomes_list, list) and outcome not in outcomes_list:
            return None

        y_true = y_true_by_outcome.get(outcome) if isinstance(y_true_by_outcome, dict) else None
        if y_true is None:
            return None

        outcome_type: str | None = None
        if isinstance(outcome_type_by_outcome, dict):
            raw_t = outcome_type_by_outcome.get(outcome)
            if isinstance(raw_t, str) and raw_t.strip():
                outcome_type = raw_t.strip().lower()

        row_index = meta.get("row_index")
        dataset_id = f"{outcome}__{row_index}" if row_index is not None else None

        samples = row.get("samples")
        if not isinstance(samples, list):
            samples = []

        new_samples: list[dict] = []
        for sample in samples:
            if not isinstance(sample, dict):
                continue
            parsed = sample.get("parsed")
            per_outcome_parsed = None
            if isinstance(parsed, dict):
                per_outcome_parsed = parsed.get(outcome)

            if per_outcome_parsed is None:
                raw_text = sample.get("raw_text")
                if isinstance(raw_text, str) and raw_text:
                    try:
                        payload = json.loads(raw_text)
                    except json.JSONDecodeError:
                        payload = None
                    if isinstance(payload, dict):
                        per_outcome_parsed = payload.get(outcome)

            scalar = extract_scalar_prediction(per_outcome_parsed)
            if scalar is None and isinstance(per_outcome_parsed, int | float):
                scalar = float(per_outcome_parsed)

            new_sample = dict(sample)
            if per_outcome_parsed is not None:
                new_sample["parsed"] = per_outcome_parsed
            if scalar is not None:
                new_sample["prediction_scalar"] = scalar
            new_samples.append(new_sample)

        new_meta = dict(meta)
        new_meta["outcome"] = outcome
        if outcome_type is not None:
            new_meta["outcome_type"] = outcome_type
        new_meta["y_true"] = y_true

        out = dict(row)
        out["metadata"] = new_meta
        out["outcome"] = outcome
        if outcome_type is not None:
            out["outcome_type"] = outcome_type
        out["y_true"] = y_true
        if dataset_id is not None:
            out["dataset_id"] = dataset_id
            out.setdefault("id", dataset_id)
        out["samples"] = new_samples
        return out

    # Iterate models from disk when repairing, otherwise from CLI args.
    if repair_mode:
        model_items: list[tuple[str | None, str]] = [(None, mn) for mn in model_norms]
    else:
        model_items = []
        for model in models:
            per_model_options = _select_model_options(model_options_payload, model=model, models=models)
            reasoning_effort = _extract_reasoning_effort(per_model_options)
            model_norm = normalize_model_slug(model, reasoning_effort=reasoning_effort)
            model_items.append((model, model_norm))

    for model, model_norm in model_items:
        for family in families:
            if repair_mode:
                existing_family_dir = batch_root / model_norm / family
                if not existing_family_dir.exists():
                    continue
            # Multi-outcome by default:
            # - generate prompts once per family (all outcomes)
            # - sample once per model+family (expensive step)
            # - split samples per outcome for per-outcome calibration + canonical leaf updates

            frozen_dir = runs_root / run_id / "_frozen_prompts" / family
            frozen_dir.mkdir(parents=True, exist_ok=True)
            frozen_prompts = frozen_dir / f"prompts__{family}.jsonl"

            if not repair_mode and not frozen_prompts.exists():
                run_unperturbed_phase(
                    [
                        "generate-prompts",
                        "--outdir",
                        str(frozen_dir),
                        "--prompt-family",
                        family,
                        "--outcomes",
                        ",".join(outcomes),
                        "--n",
                        str(int(args.n)),
                        "--seed",
                        str(int(args.seed)),
                    ],
                )

            batch_leaf = runs_root / run_id / "_batch" / model_norm / family
            batch_uncal = batch_leaf / "uncalibrated"
            batch_uncal.mkdir(parents=True, exist_ok=True)

            batch_samples_out = batch_uncal / f"samples__{family}.jsonl"
            if not repair_mode and not batch_samples_out.exists():
                sample_args = [
                    "sample",
                    "--prompts",
                    str(frozen_prompts),
                    "--outdir",
                    str(batch_uncal),
                    "--prompt-family",
                    family,
                    "--backend",
                    str(args.backend),
                    "--models",
                    str(model),
                    "--k",
                    str(int(args.k)),
                    "--run-id",
                    run_id,
                    "--max-concurrency",
                    str(int(args.max_concurrency)),
                ]
                if args.model_options_file:
                    sample_args.extend(["--model-options-file", str(args.model_options_file)])
                elif args.model_options:
                    sample_args.extend(["--model-options", str(args.model_options)])
                run_unperturbed_phase(sample_args)

            if not batch_samples_out.exists():
                if repair_mode:
                    continue
                raise RuntimeError(f"Expected batch samples at {batch_samples_out}")

            batch_rows = read_jsonl(batch_samples_out, dicts_only=True)

            for outcome in outcomes:
                run_leaf = runs_root / run_id / outcome / model_norm / family
                uncal_dir = run_leaf / "uncalibrated"
                cal_dir = run_leaf / "calibrated"
                uncal_dir.mkdir(parents=True, exist_ok=True)
                cal_dir.mkdir(parents=True, exist_ok=True)

                # Materialize per-outcome samples from multi-outcome batch.
                outcome_samples = uncal_dir / "samples.jsonl"
                outcome_samples_named = uncal_dir / f"samples__{family}.jsonl"
                if repair_mode or not outcome_samples.exists():
                    filtered: list[dict] = []
                    for r in batch_rows:
                        projected = _as_per_outcome_row(row=r, outcome=outcome)
                        if projected is not None:
                            filtered.append(projected)
                            continue
                        if _row_outcome(r) == outcome:
                            filtered.append(r)
                    write_jsonl(outcome_samples, filtered, json_default=numpy_json_default)
                    write_jsonl(outcome_samples_named, filtered, json_default=numpy_json_default)
                elif not outcome_samples_named.exists():
                    shutil.copyfile(outcome_samples, outcome_samples_named)

                if bool(args.repair_samples_only):
                    continue

                # Metrics (uncalibrated).
                metrics_uncal = uncal_dir / f"metrics_summary__{family}.csv"
                if repair_mode or not metrics_uncal.exists():
                    run_unperturbed_phase(
                        ["metrics", "--outdir", str(uncal_dir), "--prompt-family", family],
                    )

                # Calibrate.
                calibration_out = cal_dir / "calibration.json"
                calibrated_predictions = cal_dir / "predictions_calibrated.csv"
                if repair_mode or not (calibration_out.exists() and calibrated_predictions.exists()):
                    calibrate_llm_predictions(
                        [
                            "--in",
                            str(uncal_dir),
                            "--out",
                            str(calibrated_predictions),
                            "--calibration-out",
                            str(calibration_out),
                        ],
                    )

                # Metrics (calibrated).
                metrics_cal = cal_dir / f"metrics_summary_calibrated__{family}.csv"
                if repair_mode or not metrics_cal.exists():
                    run_unperturbed_phase(
                        [
                            "metrics",
                            "--predictions",
                            str(calibrated_predictions),
                            "--outdir",
                            str(cal_dir),
                            "--prompt-family",
                            family,
                            "--metrics-only",
                            "--pred-col",
                            "y_pred_cal",
                        ],
                    )

                # Update canonical leaf from existing canonical + this run.
                canonical_leaf = root / outcome / model_norm / family
                new_samples = find_samples_jsonl(uncal_dir)
                if new_samples is None:
                    raise RuntimeError(f"Missing samples for run leaf: {uncal_dir}")
                existing_samples = canonical_leaf / "uncalibrated" / "samples.jsonl"
                sample_inputs = [new_samples]
                if existing_samples.exists():
                    sample_inputs = [existing_samples, new_samples]
                rebuild_canonical_leaf(
                    canonical_leaf=canonical_leaf,
                    samples_paths=sample_inputs,
                    strategy=family,
                    locks_root=locks_root,
                )

            if bool(args.repair_samples_only):
                continue

    rebuild_summaries(root)

    if not bool(args.skip_posthoc):
        run_unperturbed_posthoc(["--output-root", str(root)])

    if not bool(args.skip_block_ablation):
        model_for_ablation = _parse_csv_list(args.models)[0] if _parse_csv_list(args.models) else "openai/gpt-oss-20b"
        run_conditional_knockout(
            [
                "--full-root",
                str(root),
                "--output-root",
                str(args.block_ablation_output_root),
                "--model",
                str(model_for_ablation),
                "--strategy",
                "data_only",
            ]
        )
