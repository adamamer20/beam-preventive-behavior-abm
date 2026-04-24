"""Generate prompts for unperturbed respondent rows."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from beam_abm.cli import parser_compat as cli
from beam_abm.llm_microvalidation.behavioural_outcomes.support.baseline_reconstruction.unperturbed_utils import (
    SUPPORTED_PROMPT_FAMILIES,
    build_predictor_cols,
    build_prompts,
    load_lever_config,
    load_outcome_types,
    parse_list_arg,
    parse_prompt_families,
    stratified_sample_indices,
)
from beam_abm.llm_microvalidation.prompts import (
    PromptTemplateBuilder,
    build_clean_spec_index,
)
from beam_abm.llm_microvalidation.shared.labels import (
    load_clean_spec_question_answer_maps,
    load_display_groups,
    load_display_labels,
    load_ordinal_endpoints_from_clean_spec,
    load_possible_numeric_bounds_from_clean_spec,
)
from beam_abm.llm_microvalidation.utils.jsonl import numpy_json_default, write_jsonl


def run_cli(argv: list[str] | None = None) -> None:
    parser = cli.ArgumentParser()
    parser.add_argument("--input", default="preprocess/output/clean_processed_survey.csv")
    parser.add_argument("--outdir", default="evaluation/output/debug_unperturbed")
    parser.add_argument(
        "--outcomes",
        default="vax_willingness_Y,mask_when_pressure_high,stay_home_when_symptomatic",
        help="Comma-separated outcome column names (default matches prompt-tournament keep-outcomes)",
    )
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--prompt-family", default=None)
    parser.add_argument(
        "--prompt-families",
        default=",".join(sorted(SUPPORTED_PROMPT_FAMILIES)),
        help="Comma-separated prompt families (data_only,first_person,third_person,multi_expert)",
    )
    parser.add_argument("--strata-col", default="country")

    parser.add_argument("--lever-config", default="config/levers.json")
    parser.add_argument("--model-plan", default="empirical/output/modeling/model_plan.json")
    parser.add_argument("--display-labels", default="thesis/specs/display_labels.json")
    parser.add_argument("--clean-spec", default="preprocess/specs/5_clean_transformed_questions.json")
    parser.add_argument("--column-types", default="empirical/specs/column_types.tsv")

    args = parser.parse_args(argv)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    families = parse_prompt_families(args.prompt_family, args.prompt_families)

    outcomes = parse_list_arg(args.outcomes)
    if not outcomes:
        raise ValueError("Provide --outcomes")

    df = pd.read_csv(args.input)
    if args.strata_col not in df.columns:
        raise KeyError(f"Expected a '{args.strata_col}' column for stratified sampling")

    indices = stratified_sample_indices(df, strata_col=str(args.strata_col), n=int(args.n), seed=int(args.seed))

    lever_cfg = load_lever_config(Path(args.lever_config))
    col_types = load_outcome_types(Path(args.column_types))

    display_labels = load_display_labels(Path(args.display_labels))
    display_groups = load_display_groups(Path(args.display_labels))
    clean_spec_path = Path(args.clean_spec)
    clean_questions, clean_answers = load_clean_spec_question_answer_maps(clean_spec_path)
    possible_bounds = load_possible_numeric_bounds_from_clean_spec(clean_spec_path)
    ordinal_endpoints = load_ordinal_endpoints_from_clean_spec(clean_spec_path)
    clean_spec_data = json.loads(clean_spec_path.read_text(encoding="utf-8"))
    encodings, by_id = build_clean_spec_index(clean_spec_data)

    possible_bounds.setdefault("covax_legitimacy_scepticism_idx", (0.0, 1.0))
    if "income_ppp_norm" in df.columns:
        income_vals = pd.to_numeric(df["income_ppp_norm"], errors="coerce").dropna()
        if not income_vals.empty:
            possible_bounds.setdefault(
                "income_ppp_norm",
                (float(income_vals.min()), float(income_vals.max())),
            )

    prompt_builder = PromptTemplateBuilder(
        display_labels=display_labels,
        display_groups=display_groups,
        clean_questions=clean_questions,
        clean_answers=clean_answers,
        encodings=encodings,
        by_id=by_id,
        column_types=col_types,
        possible_bounds=possible_bounds,
        ordinal_endpoints=ordinal_endpoints,
    )

    for family in families:
        rows: list[dict] = []

        for outcome_key in outcomes:
            if outcome_key not in df.columns:
                raise KeyError(f"Outcome column not found in dataset: {outcome_key}")
            outcome_type = col_types.get(outcome_key)
            if outcome_type not in {"binary", "ordinal", "continuous"}:
                raise ValueError(f"Unsupported/missing outcome type for {outcome_key}: {outcome_type}")

            outcome_label = prompt_builder.outcome_label(outcome_key)

            bounds = possible_bounds.get(outcome_key)
            endpoints = ordinal_endpoints.get(outcome_key) if outcome_type == "ordinal" else None

            predictor_cols = build_predictor_cols(
                outcome_key=outcome_key,
                lever_cfg=lever_cfg,
                model_plan=Path(args.model_plan),
            )

            messages_list, y_obs, country_obs, row_idx, attrs_list = build_prompts(
                df=df,
                indices=indices,
                outcome_key=outcome_key,
                predictor_cols=predictor_cols,
                outcome_label=str(outcome_label),
                outcome_type=outcome_type,
                bounds=bounds,
                endpoints=endpoints,
                col_types=col_types,
                display_labels=display_labels,
                clean_questions=clean_questions,
                clean_answers=clean_answers,
                possible_bounds=possible_bounds,
                ordinal_endpoints=ordinal_endpoints,
                prompt_family=family,
                prompt_builder=prompt_builder,
            )

            if not messages_list:
                continue

            for i, (messages, y_true, country, idx, attrs) in enumerate(
                zip(messages_list, y_obs, country_obs, row_idx, attrs_list, strict=False)
            ):
                system_prompt = messages[0].get("value") if messages else ""
                human_prompt = messages[1].get("value") if len(messages) > 1 else ""
                bounds_payload = [float(bounds[0]), float(bounds[1])] if bounds is not None else None
                endpoints_payload = list(endpoints) if endpoints is not None else None

                row: dict = {
                    "id": f"{outcome_key}__{idx}__{i}",
                    "outcome": outcome_key,
                    "outcome_type": outcome_type,
                    "prompt_family": family,
                    "strategy": family,
                    "system_prompt": system_prompt,
                    "human_prompt": human_prompt,
                    "messages": messages,
                    "attributes": attrs,
                    "metadata": {
                        "outcome": outcome_key,
                        "outcome_type": outcome_type,
                        "outcome_label": str(outcome_label),
                        "bounds": bounds_payload,
                        "endpoints": endpoints_payload,
                        "country": country,
                        "y_true": float(y_true),
                        "row_index": int(idx),
                    },
                }

                if family in {"first_person", "third_person"}:
                    row["predict_system_prompt"] = system_prompt
                    row["narrative_system_prompt"] = system_prompt
                    if family == "first_person":
                        narrative_assistant = messages[1].get("value") if len(messages) > 1 else ""
                        narrative_user = messages[2].get("value") if len(messages) > 2 else ""
                        row["narrative_human_prompt"] = narrative_assistant
                        row["narrative_assistant_prompt"] = narrative_assistant
                        row["narrative_user_prompt"] = narrative_user
                    else:
                        row["narrative_user_prompt"] = messages[1].get("value") if len(messages) > 1 else ""

                rows.append(row)

        out_path = outdir / f"prompts__{family}.jsonl"
        write_jsonl(out_path, rows, json_default=numpy_json_default)
