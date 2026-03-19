"""Generate prompts for micro-validation strategies.

This script reads a CSV/JSONL of realized attribute vectors and outputs JSONL prompts.

Input JSONL format (minimal):
- id
- attributes: dict

Output JSONL:
- id
- strategy
- system_prompt / human_prompt (or narrative_* for narrative strategies; attributes for multi_expert)
- metadata
- attributes (only for multi_expert)
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import pandas as pd

from beam_abm.empirical.io import parse_list_arg
from beam_abm.empirical.missingness import SPECIAL_STRUCTURAL_PREFIXES
from beam_abm.empirical.plan_io import extract_predictors_from_model_plan
from beam_abm.evaluation.common.labels import (
    load_clean_spec_question_answer_maps,
    load_display_groups,
    load_display_labels,
    load_ordinal_endpoints_from_clean_spec,
    load_possible_numeric_bounds_from_clean_spec,
)
from beam_abm.evaluation.prompting import (
    PromptTemplateBuilder,
    build_clean_spec_index,
)
from beam_abm.evaluation.utils.jsonl import (
    read_rows as _read_rows,
)
from beam_abm.evaluation.utils.jsonl import (
    write_jsonl as _write_jsonl,
)


def _read_parquet_paths(
    paths: list[Path], *, limit: int | None, shuffle: bool, seed: int
) -> tuple[list[dict], bool, dict[str, tuple[float, float]]]:
    try:
        import polars as pl
    except ImportError:
        scans = []
    else:
        scans = [pl.scan_parquet(p) for p in paths]
    if scans:
        computed_bounds: dict[str, tuple[float, float]] = {}
        # Infer numeric bounds for select variables without a declared scale in the clean spec.
        for col in ("income_ppp_norm",):
            mins: list[float] = []
            maxs: list[float] = []
            for lf in scans:
                if col not in lf.collect_schema().names():
                    continue
                agg = (
                    lf.select(
                        pl.col(col).cast(pl.Float64, strict=False).min().alias("mn"),
                        pl.col(col).cast(pl.Float64, strict=False).max().alias("mx"),
                    )
                    .collect(engine="streaming")
                    .to_dicts()
                )
                if not agg:
                    continue
                mn = agg[0].get("mn")
                mx = agg[0].get("mx")
                if mn is not None:
                    mins.append(float(mn))
                if mx is not None:
                    maxs.append(float(mx))
            if mins and maxs:
                computed_bounds[col] = (min(mins), max(maxs))

        rows: list[dict] = []
        for lf in scans:
            if shuffle:
                lf = lf.with_row_index("__row_idx")
                lf = lf.with_columns(pl.col("__row_idx").hash(seed=int(seed)).alias("__rand"))
                lf = lf.sort("__rand")
                lf = lf.drop(["__row_idx", "__rand"])
            rows.extend(lf.collect(engine="streaming").to_dicts())

        if shuffle:
            import random as _random

            _random.Random(int(seed)).shuffle(rows)
        return rows, True, computed_bounds

    rows: list[dict] = []
    for p in paths:
        rows.extend(_read_rows(p))
    return rows, False, {}


def _fmt_num(x: float) -> str:
    if float(x).is_integer():
        return str(int(x))
    return f"{x:.3f}".rstrip("0").rstrip(".")


def main() -> None:
    task_parser = argparse.ArgumentParser(add_help=False)
    task_parser.add_argument("--mv-task", choices=["choice", "belief_update"], default="choice")
    task_args, remaining_argv = task_parser.parse_known_args()
    if task_args.mv_task == "belief_update":
        from beam_abm.evaluation.belief.prompt_generation import generate_belief_prompts

        belief_parser = argparse.ArgumentParser()
        belief_parser.add_argument("--data", required=True, help="Input CSV/Parquet/JSONL with baseline state rows.")
        belief_parser.add_argument(
            "--pe-ref-p",
            default=None,
            help="Precomputed PE_ref_P table (parquet/csv/jsonl) with one or more rows per (id, signal_id).",
        )
        belief_parser.add_argument(
            "--run-type",
            choices=["perturbed", "unperturbed"],
            default="perturbed",
            help="Belief prompt generation mode.",
        )
        belief_parser.add_argument("--spec", required=True, help="Belief-update spec JSON with signals.")
        belief_parser.add_argument("--out", required=True, help="Output JSONL prompts.")
        belief_parser.add_argument("--limit", type=int, default=None)
        belief_parser.add_argument("--shuffle", action="store_true")
        belief_parser.add_argument("--seed", type=int, default=0)
        belief_parser.add_argument("--profile-col", default="anchor_id")
        belief_parser.add_argument("--per-profile", type=int, default=100)
        belief_parser.add_argument("--anchor-col", default="anchor_id")
        belief_parser.add_argument("--keep-anchors", default=None)
        belief_parser.add_argument("--n-per-anchor-country", type=int, default=0)
        belief_parser.add_argument("--country-col", default="country")
        belief_parser.add_argument("--id-col", default="row_id")
        belief_parser.add_argument("--prompt-families", default="data_only,first_person,third_person,multi_expert")
        belief_parser.add_argument("--display-labels", default="thesis/specs/display_labels.json")
        belief_parser.add_argument("--clean-spec", default="preprocess/specs/5_clean_transformed_questions.json")
        belief_parser.add_argument("--column-types", default="empirical/specs/column_types.tsv")
        belief_parser.add_argument("--model-plan", default="empirical/output/modeling/model_plan.json")
        belief_parser.add_argument("--alpha-secondary", type=float, default=0.33)
        belief_parser.add_argument("--epsilon-ref", type=float, default=0.03)
        belief_parser.add_argument("--top-n-primary", type=int, default=1)
        belief_args = belief_parser.parse_args(remaining_argv)
        generate_belief_prompts(
            data=belief_args.data,
            spec=belief_args.spec,
            out=belief_args.out,
            pe_ref_p=belief_args.pe_ref_p,
            run_type=belief_args.run_type,
            limit=belief_args.limit,
            shuffle=belief_args.shuffle,
            seed=belief_args.seed,
            profile_col=belief_args.profile_col,
            per_profile=belief_args.per_profile,
            anchor_col=belief_args.anchor_col,
            keep_anchors=belief_args.keep_anchors,
            n_per_anchor_country=belief_args.n_per_anchor_country,
            country_col=belief_args.country_col,
            id_col=belief_args.id_col,
            prompt_families=belief_args.prompt_families,
            display_labels=belief_args.display_labels,
            clean_spec=belief_args.clean_spec,
            column_types=belief_args.column_types,
            model_plan=belief_args.model_plan,
            alpha_secondary=belief_args.alpha_secondary,
            epsilon_ref=belief_args.epsilon_ref,
            top_n_primary=belief_args.top_n_primary,
        )
        return

    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="in_path", default=None)
    parser.add_argument(
        "--in-dir",
        dest="in_dir",
        default=None,
        help="Directory containing per-outcome design.parquet files (loads all for multi-outcome dedup).",
    )
    parser.add_argument("--out", dest="out_path", required=True)
    parser.add_argument(
        "--strategy",
        choices=["data_only", "first_person", "third_person", "paired_profile", "multi_expert"],
        default="data_only",
        help="Prompting strategy to use.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on number of rows to emit (debugging).",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle rows before applying --limit (debugging).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="RNG seed used when --shuffle is set.",
    )
    parser.add_argument(
        "--dedup-levers",
        action="store_true",
        default=True,
        help="Deduplicate per (id, lever, grid_point) and request multiple outcomes in one prompt.",
    )
    parser.add_argument(
        "--no-dedup-levers",
        dest="dedup_levers",
        action="store_false",
        help="Disable multi-outcome deduplication (required for paired_profile).",
    )

    parser.add_argument(
        "--paired-profile-multi-outcome",
        action="store_true",
        default=False,
        help=(
            "Enable multi-outcome prompts for paired_profile when --dedup-levers groups multiple outcomes. "
            'Output schema becomes: {"A": {<outcome>: <number>, ...}, "B": {...}}.'
        ),
    )

    # Debug-focused filters / sampling (keeps design logic but shrinks the prompt set).
    parser.add_argument(
        "--keep-outcomes",
        default="vax_willingness_Y,mask_when_pressure_high,stay_home_when_symptomatic",
        help="Optional comma-separated list of outcomes to keep (filters input rows before prompt generation).",
    )
    parser.add_argument(
        "--keep-targets",
        default=(
            "institutional_trust_avg,"
            "vaccine_risk_avg,"
            "covid_perceived_danger_Y,"
            "social_norms_vax_avg,"
            "fivec_low_complacency,"
            "moral_progressive_orientation,"
            "infection_likelihood_avg"
        ),
        help="Optional comma-separated list of lever/target names to keep.",
    )
    parser.add_argument(
        "--keep-grid-points",
        default="low,high",
        help="Optional comma-separated list of grid_point values to keep (e.g. low,high).",
    )
    parser.add_argument(
        "--keep-countries",
        default=None,
        help="Optional comma-separated list of countries to keep (parsed from id like '<anchor>::<country>::<row>').",
    )
    parser.add_argument(
        "--keep-anchors",
        default=None,
        help="Optional comma-separated list of anchor ids to keep (parsed from id like '<anchor>::<country>::<row>').",
    )
    parser.add_argument(
        "--n-per-anchor-country",
        type=int,
        default=5,
        help="Optional number of unique base ids to sample per (anchor_id, country). Uses --seed for determinism.",
    )
    parser.add_argument("--base-prompt", default="You are an expert analyst.")
    parser.add_argument(
        "--task",
        default="Output: JSON only (on the declared outcome scale).",
    )
    parser.add_argument(
        "--prompt-style",
        choices=["psychometrician"],
        default="psychometrician",
        help="Prompt template style to use for data_only and paired_profile outputs.",
    )
    parser.add_argument(
        "--task-template",
        default=None,
        help="Optional format string using {outcome} to build per-row task instructions.",
    )
    parser.add_argument(
        "--driver-cols",
        default=None,
        help="Optional comma-separated list of attribute columns to include in prompts (aligns LLM with ref model inputs).",
    )
    parser.add_argument(
        "--include-missing-indicators",
        action="store_true",
        help="Include <col>_missing predictors from model_plan.json when using model_ref.",
    )
    parser.add_argument(
        "--lever-config",
        default="config/levers.json",
        help="Optional JSON mapping outcome->lever/driver columns (e.g. config/levers.json).",
    )
    parser.add_argument(
        "--lever-key",
        default=None,
        help="Key into --lever-config. If omitted, uses each row's outcome as the key.",
    )
    parser.add_argument(
        "--model-plan",
        default="empirical/output/modeling/model_plan.json",
        help="Model plan JSON used to resolve predictors when lever-config entries use model_ref.",
    )
    parser.add_argument(
        "--include-cols",
        default=None,
        help="Optional comma-separated extra columns to include in prompts in addition to driver cols (e.g. country).",
    )
    parser.add_argument("--all-attrs", action="store_true")
    parser.add_argument(
        "--display-labels",
        default="thesis/specs/display_labels.json",
        help="Central display labels JSON (used for short, human-readable attribute names).",
    )
    parser.add_argument(
        "--clean-spec",
        default="preprocess/specs/5_clean_transformed_questions.json",
        help="Clean preprocessing spec (fallback for attribute labels when missing from --display-labels).",
    )
    parser.add_argument(
        "--column-types",
        default="empirical/specs/column_types.tsv",
        help="Column types TSV (used to tag ordinal items as Likert).",
    )
    args = parser.parse_args(remaining_argv)

    def _parse_id_parts(rid: object) -> tuple[int | None, str | None, int | None]:
        if not isinstance(rid, str) or "::" not in rid:
            return None, None, None
        parts = rid.split("::", 2)
        if len(parts) != 3:
            return None, None, None
        a_raw, country, row_raw = parts
        try:
            anchor_id = int(a_raw)
        except Exception:
            anchor_id = None
        try:
            row_id = int(row_raw)
        except Exception:
            row_id = None
        return anchor_id, (country if country else None), row_id

    parquet_sampled = False
    computed_bounds: dict[str, tuple[float, float]] = {}
    if args.in_dir:
        root = Path(args.in_dir)
        if not root.exists():
            raise FileNotFoundError(f"--in-dir not found: {root}")
        paths = sorted(root.rglob("design.parquet"))
        if not paths:
            raise FileNotFoundError(f"No design.parquet files found under: {root}")
        rows, parquet_sampled, computed_bounds = _read_parquet_paths(
            paths, limit=args.limit, shuffle=args.shuffle, seed=args.seed
        )
    elif args.in_path:
        in_path = Path(args.in_path)
        if in_path.suffix.lower() == ".parquet":
            rows, parquet_sampled, computed_bounds = _read_parquet_paths(
                [in_path], limit=args.limit, shuffle=args.shuffle, seed=args.seed
            )
        else:
            rows = _read_rows(in_path)
    else:
        raise ValueError("Provide either --in or --in-dir")
    if args.shuffle and not parquet_sampled:
        rows = pd.DataFrame(rows).sample(frac=1.0, random_state=int(args.seed)).to_dict(orient="records")

    keep_outcomes = set(parse_list_arg(args.keep_outcomes)) if args.keep_outcomes else set()
    keep_targets = set(parse_list_arg(args.keep_targets)) if args.keep_targets else set()
    keep_grid_points = {str(x) for x in parse_list_arg(args.keep_grid_points)} if args.keep_grid_points else set()
    keep_countries = {str(x) for x in parse_list_arg(args.keep_countries)} if args.keep_countries else set()
    keep_anchors: set[int] = set()
    if args.keep_anchors:
        for x in parse_list_arg(args.keep_anchors):
            try:
                keep_anchors.add(int(str(x)))
            except Exception:
                continue

    if keep_outcomes or keep_targets or keep_grid_points or keep_countries or keep_anchors or args.n_per_anchor_country:
        # Enrich rows with parsed id parts for stable filtering / stratified sampling.
        for r in rows:
            anchor_id, country, row_id = _parse_id_parts(r.get("id"))
            r.setdefault("__anchor_id", anchor_id)
            r.setdefault("__country", country)
            r.setdefault("__row_id", row_id)

    if keep_outcomes:
        rows = [
            r
            for r in rows
            if (
                (isinstance(r.get("outcome"), str) and r.get("outcome") in keep_outcomes)
                or (
                    isinstance(r.get("outcomes"), list)
                    and any(isinstance(o, str) and o in keep_outcomes for o in r.get("outcomes") or [])
                )
            )
        ]
    if keep_targets:
        rows = [r for r in rows if isinstance(r.get("target"), str) and r.get("target") in keep_targets]
    if keep_grid_points and args.strategy != "paired_profile":
        rows = [r for r in rows if str(r.get("grid_point")) in keep_grid_points]
    if keep_countries:
        rows = [r for r in rows if isinstance(r.get("__country"), str) and r.get("__country") in keep_countries]
    if keep_anchors:
        rows = [r for r in rows if isinstance(r.get("__anchor_id"), int) and r.get("__anchor_id") in keep_anchors]

    if args.n_per_anchor_country is not None:
        n = max(0, int(args.n_per_anchor_country))
        if n <= 0:
            rows = []
        else:
            import random as _random
            import zlib

            # Sample ids (base profiles) per (anchor, country) and keep all rows for those ids.
            by_group: dict[tuple[int, str], list[str]] = {}
            for r in rows:
                rid = r.get("id")
                anchor_id = r.get("__anchor_id")
                country = r.get("__country")
                if not isinstance(rid, str) or not isinstance(anchor_id, int) or not isinstance(country, str):
                    continue
                by_group.setdefault((anchor_id, country), []).append(rid)

            selected_ids: set[str] = set()
            for (anchor_id, country), ids in by_group.items():
                uniq = sorted(set(ids))
                country_hash = zlib.crc32(country.encode("utf-8")) & 0xFFFFFFFF
                rng = _random.Random(
                    (int(args.seed) + 1_000_003 * anchor_id + 10_007 * (country_hash & 0xFFFF)) & 0xFFFFFFFF
                )
                rng.shuffle(uniq)
                if len(uniq) >= n:
                    pick = uniq[:n]
                elif uniq:
                    # Fill with replacement to hit the requested n.
                    pick = [uniq[i % len(uniq)] for i in range(n)]
                else:
                    pick = []
                selected_ids.update(pick)

            rows = [r for r in rows if isinstance(r.get("id"), str) and r.get("id") in selected_ids]

    lever_map: dict[str, object] = {}
    if args.lever_config and not args.all_attrs:
        lever_map = json.loads(Path(args.lever_config).read_text(encoding="utf-8"))

    include_cols_override = list(parse_list_arg(args.include_cols)) if args.include_cols else []
    driver_cols_override = list(parse_list_arg(args.driver_cols)) if args.driver_cols else None

    display_labels = load_display_labels(Path(args.display_labels))
    display_groups = load_display_groups(Path(args.display_labels))
    clean_spec_path = Path(args.clean_spec)
    clean_questions, clean_answers = load_clean_spec_question_answer_maps(clean_spec_path)
    clean_spec_data = json.loads(clean_spec_path.read_text(encoding="utf-8"))
    encodings, by_id = build_clean_spec_index(clean_spec_data)
    possible_bounds = load_possible_numeric_bounds_from_clean_spec(Path(args.clean_spec))
    ordinal_endpoints = load_ordinal_endpoints_from_clean_spec(clean_spec_path)
    for k, v in computed_bounds.items():
        possible_bounds.setdefault(k, v)

    column_types = pd.read_csv(args.column_types, sep="\t").set_index("column")["type"].to_dict()

    def _resolve_lever_entry(row: dict) -> dict | None:
        lever_key = args.lever_key or row.get("outcome")
        if not lever_key or not isinstance(lever_map, dict):
            return None
        entry = lever_map.get(str(lever_key))
        return entry if isinstance(entry, dict) else None

    def _resolve_driver_cols(row: dict) -> list[str] | None:
        if driver_cols_override:
            return driver_cols_override
        if args.all_attrs:
            return None
        lever_key = args.lever_key or row.get("outcome")
        if not lever_key:
            return None
        entry = lever_map.get(str(lever_key)) if isinstance(lever_map, dict) else None
        if entry is None:
            return None
        if isinstance(entry, list):
            return [str(x) for x in entry]
        if isinstance(entry, dict):
            model_ref = entry.get("model_ref")
            if model_ref:
                predictors = extract_predictors_from_model_plan(
                    model_plan_path=Path(args.model_plan),
                    model_name=str(model_ref),
                )
                if args.include_missing_indicators:
                    predictors = predictors + [f"{p}_missing" for p in predictors]
                return predictors
            if entry.get("type") == "model_ref":
                predictors = extract_predictors_from_model_plan(
                    model_plan_path=Path(args.model_plan),
                    model_name=str(lever_key),
                )
                if args.include_missing_indicators:
                    predictors = predictors + [f"{p}_missing" for p in predictors]
                return predictors
            levers = entry.get("levers")
            if isinstance(levers, list):
                return [str(x) for x in levers]
            cols = entry.get("columns")
            if isinstance(cols, list):
                return [str(x) for x in cols]
        return None

    memo_base: dict[str, str | None] = {}
    visiting: set[str] = set()

    def _base_id(col_id: str) -> str | None:
        cid = col_id.strip()
        if not cid:
            return None
        if cid in memo_base:
            return memo_base[cid]
        if cid in visiting:
            return None
        visiting.add(cid)
        item = by_id.get(cid)
        if item is None:
            visiting.remove(cid)
            memo_base[cid] = None
            return None
        transforms = item.get("transformations") or []
        if not isinstance(transforms, list):
            transforms = []
        for tr in transforms:
            if not isinstance(tr, dict) or tr.get("type") != "renamed":
                continue
            src = tr.get("rename_from")
            if isinstance(src, str) and src.strip():
                base = _base_id(src)
                visiting.remove(cid)
                memo_base[cid] = base
                return base
        visiting.remove(cid)
        memo_base[cid] = cid
        return cid

    def _encoding_spec_for(col_id: str) -> object | None:
        base = _base_id(col_id)
        if base is None:
            return None
        item = by_id.get(base)
        if item is None:
            return None
        transforms = item.get("transformations") or []
        if not isinstance(transforms, list):
            return None
        for tr in transforms:
            if not isinstance(tr, dict) or tr.get("type") != "encoding":
                continue
            return tr.get("encoding")
        return None

    def _mapping_from_encoding(enc: object | None) -> dict[str, object] | None:
        if enc is None:
            return None
        if isinstance(enc, dict):
            return enc
        if isinstance(enc, str):
            s = enc.strip()
            m = encodings.get(s)
            return m if isinstance(m, dict) else None
        return None

    def _as_numeric(val: object) -> float | None:
        if val is None:
            return None
        if isinstance(val, bool):
            return 1.0 if val else 0.0
        if isinstance(val, int | float) and val == val:
            return float(val)
        try:
            num = float(str(val))
            # Treat NaN or infinite values as missing
            if not math.isfinite(num):
                return None
            return num
        except Exception:
            return None

    def _normalize_value(col: str, val: object) -> str:
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return "missing"
        mapping = _mapping_from_encoding(_encoding_spec_for(col))
        if mapping:
            num_val = _as_numeric(val)
            key = str(val)
            mapped = mapping.get(key)
            if mapped is None and key.lower() != key:
                mapped = mapping.get(key.lower())
            if mapped is None and key.upper() != key:
                mapped = mapping.get(key.upper())
            if mapped is None and "$DEFAULT" in mapping:
                mapped = mapping.get("$DEFAULT")
            if mapped is None:
                if num_val is not None:
                    return _fmt_num(num_val)
                return "missing"
            num = _as_numeric(mapped)
            if num is not None:
                return _fmt_num(num)
            return str(mapped)
        num = _as_numeric(val)
        if num is not None:
            return _fmt_num(num)
        return str(val)

    def _display_label(col: str) -> str:
        return display_labels.get(col) or clean_questions.get(col) or col

    def _outcome_semantics(outcome: str | None) -> tuple[str, str]:
        if not outcome:
            return "Outcome semantics: (unknown)", 'Return ONLY JSON: {"prediction": <float>}.'
        label = _outcome_label(outcome)
        outcome_type = str(column_types.get(outcome, "")).strip().lower()
        if outcome_type == "binary":
            schema = 'Return ONLY JSON: {"p_yes": <float in [0,1]>}.'
        else:
            schema = 'Return ONLY JSON: {"prediction": <float>}. Use the stated scale bounds.'
        return f"Outcome semantics: {label}", schema

    def _outcome_label(outcome: str) -> str:
        label = _display_label(outcome)
        bounds = possible_bounds.get(outcome)
        endpoints = ordinal_endpoints.get(outcome)
        if bounds is not None:
            lo, hi = bounds
            lo_s, hi_s = _fmt_num(float(lo)), _fmt_num(float(hi))
            if endpoints is not None:
                lo_lab, hi_lab = endpoints
                return f"{label} ({lo_s}–{hi_s}: {lo_s} = {lo_lab}, {hi_s} = {hi_lab})"
            return f"{label} ({lo_s}–{hi_s})"
        return label

    def _multi_outcome_semantics(outcomes: list[str]) -> tuple[str, str]:
        lines = ["Outcome semantics:"]
        for outcome in outcomes:
            label = _outcome_label(outcome)
            t = str(column_types.get(outcome, "")).strip().lower()
            if t == "binary":
                label = f"{label} (binary probability in [0,1])"
            lines.append(f"- {outcome}: {label}")
        schema = "Return ONLY JSON with keys: " + ", ".join(outcomes) + "."
        schema += " Each value must stay within the stated scale bounds."
        return "\n".join(lines), schema

    def _psychometrician_system(outcome: str | None) -> str:
        semantics, schema = _outcome_semantics(outcome)
        return (
            "You are an expert survey psychometrician acting as a prediction function.\n\n"
            "Predict the expected self-reported survey response for people in this dataset/country, "
            "not what is socially desirable or recommended.\n\n"
            f"{semantics}\n\n"
            "Do not include reasoning or commentary.\n"
            f"{schema}"
        )

    def _first_person_system_prompt() -> str:
        return (
            "You are answering as the same speaker across the conversation.\n"
            "Maintain strict self-consistency with what you have already stated about yourself, "
            "your beliefs, and your constraints in prior assistant messages.\n"
            "Do not add new biography, motives, or beliefs unless they have been stated already. "
            "If something is not stated, treat it as unknown."
        )

    def _prediction_user_prompt(outcome: str | None) -> str:
        semantics, schema = _outcome_semantics(outcome)
        return (
            "Now answer the prediction task.\n"
            "Predict the expected self-reported survey response for people in this dataset/country, "
            "not what is socially desirable or recommended.\n"
            f"{semantics}\n"
            "Do not include reasoning or commentary.\n"
            f"{schema}"
        )

    def _prediction_user_prompt_first_person(outcome: str | None) -> str:
        semantics, schema = _outcome_semantics(outcome)
        return (
            "Telling me what you really believe and not what is socially desirable or recommended, "
            "what do you think about whether:\n"
            f"{semantics}\n"
            "Do not include reasoning or commentary.\n"
            f"{schema}"
        )

    def _prediction_user_prompt_multi(outcomes: list[str]) -> str:
        semantics, schema = _multi_outcome_semantics(outcomes)
        return (
            "Now answer the prediction task.\n"
            "Predict the expected self-reported survey response for people in this dataset/country, "
            "not what is socially desirable or recommended.\n"
            f"{semantics}\n"
            "Do not include reasoning or commentary.\n"
            f"{schema}"
        )

    def _prediction_user_prompt_first_person_multi(outcomes: list[str]) -> str:
        semantics, schema = _multi_outcome_semantics(outcomes)
        return (
            "Telling me what you really believe and not what is socially desirable or recommended, "
            "what do you think about whether:\n"
            f"{semantics}\n"
            "Do not include reasoning or commentary.\n"
            f"{schema}"
        )

    def _maybe_survey_context(attrs: dict[str, object]) -> str | None:
        for key in ("country", "Country", "country_name"):
            if key in attrs and attrs[key] is not None:
                country_label = _country_label(attrs[key])
                return f"Survey context: Country={country_label}, Year=2024"
        return None

    def _maybe_label_numeric(col: str, val: object) -> str:
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return "missing"
        if str(column_types.get(col, "")).strip().lower() == "binary":
            b = _as_bool(val)
            if b is None:
                return "missing"
            return "Yes" if b else "No"
        raw = _normalize_value(col, val)
        num = _as_numeric(raw)
        if num is None:
            return raw
        return raw

    def _scale_descriptor(col: str) -> tuple[tuple[object, ...], str] | None:
        if str(column_types.get(col, "")).strip().lower() == "binary":
            return None
        if col == "M_education":
            return None
        if col == "covax_legitimacy_scepticism_idx":
            lo_s, hi_s = "0", "1"
            low_lab = "no scepticism reasons endorsed"
            high_lab = "endorsed all listed scepticism reasons"
            key = (lo_s, hi_s, low_lab, high_lab)
            label = f"{lo_s}–{hi_s} ({lo_s} = {low_lab}, {hi_s} = {high_lab})"
            return key, label

        def _normalize_scale_label(label: str) -> str:
            cleaned = label.strip().lower().replace("’", "").replace("'", "").replace("–", "-").replace("—", "-")
            return " ".join(cleaned.split())

        def _canonical_endpoint(label: str) -> str:
            norm = _normalize_scale_label(label)
            synonyms = {
                "strongly disagree": "strongly disagree",
                "strongly agree": "strongly agree",
                "not at all": "not at all",
                "completely": "extremely",
                "extremely": "extremely",
                "not at all relevant": "not at all relevant",
                "extremely relevant": "extremely relevant",
            }
            return synonyms.get(norm, norm)

        def _canonical_scale_label(
            lo_s: str,
            hi_s: str,
            low_raw: str,
            high_raw: str,
        ) -> tuple[tuple[object, ...], str]:
            low_norm = _canonical_endpoint(low_raw)
            high_norm = _canonical_endpoint(high_raw)
            key = (lo_s, hi_s, low_norm, high_norm)
            canonical_labels = {
                ("1", "7", "strongly disagree", "strongly agree"): ("Strongly disagree", "Strongly agree"),
                ("1", "6", "strongly disagree", "strongly agree"): ("Strongly disagree", "Strongly agree"),
                ("1", "6", "not at all relevant", "extremely relevant"): (
                    "Not at all relevant",
                    "Extremely relevant",
                ),
                ("1", "5", "not at all", "extremely"): ("Not at all", "Extremely"),
            }
            display_pair = canonical_labels.get(key)
            if display_pair is None:
                display_pair = (low_raw, high_raw)
            low_disp, high_disp = display_pair
            label = f"{lo_s}–{hi_s} ({lo_s} = {low_disp}, {hi_s} = {high_disp})"
            return key, label

        def _resolve_bits(cid: str, seen: set[str]) -> tuple[tuple[str, str] | None, tuple[float, float] | None]:
            if cid in seen:
                return None, None
            seen.add(cid)

            endpoints = ordinal_endpoints.get(cid)
            bounds = possible_bounds.get(cid)

            item = by_id.get(cid)
            if not isinstance(item, dict):
                return endpoints, bounds
            transforms = item.get("transformations") or []
            if not isinstance(transforms, list):
                return endpoints, bounds
            for tr in transforms:
                if not isinstance(tr, dict):
                    continue
                if tr.get("type") == "renamed":
                    src = tr.get("rename_from")
                    if isinstance(src, str):
                        src_endpoints, src_bounds = _resolve_bits(src, seen)
                        if endpoints is None and src_endpoints is not None:
                            endpoints = src_endpoints
                        if bounds is None and src_bounds is not None:
                            bounds = src_bounds
                if tr.get("type") == "merged":
                    merged_from = tr.get("merged_from")
                    if not isinstance(merged_from, list):
                        continue
                    for src in merged_from:
                        if not isinstance(src, str):
                            continue
                        src_endpoints, src_bounds = _resolve_bits(src, seen)
                        if endpoints is None and src_endpoints is not None:
                            endpoints = src_endpoints
                        if bounds is None and src_bounds is not None:
                            bounds = src_bounds
            return endpoints, bounds

        endpoints, bounds = _resolve_bits(col, set())
        if endpoints is None and bounds is None:
            return None
        if bounds is not None:
            lo, hi = bounds
            lo_s, hi_s = _fmt_num(float(lo)), _fmt_num(float(hi))
            if endpoints is not None:
                low_lab, high_lab = endpoints
                return _canonical_scale_label(lo_s, hi_s, low_lab, high_lab)
            key = (lo_s, hi_s)
            return key, f"{lo_s}–{hi_s}"
        low_lab, high_lab = endpoints
        key = (low_lab, high_lab)
        return key, f"{low_lab} … {high_lab}"

    def _format_attr_lines(
        attrs: dict[str, object],
        driver_cols: list[str] | None,
        *,
        scale_seen: set[tuple[object, ...]],
    ) -> list[str]:
        _ = scale_seen
        if driver_cols is None:
            cols = [k for k in attrs.keys() if not k.startswith(SPECIAL_STRUCTURAL_PREFIXES)]
        else:
            cols = [c for c in driver_cols if c in attrs]
        lines: list[str] = []
        scale_items: dict[tuple[object, ...], dict[str, object]] = {}
        scale_order: list[tuple[object, ...]] = []
        non_scale_lines: list[str] = []
        grouped_cols: set[str] = set()
        for group in display_groups.values():
            members = group.get("members")
            if not isinstance(members, dict):
                continue
            grouped_cols.update(members.keys())

        for group in display_groups.values():
            members = group.get("members")
            if not isinstance(members, dict):
                continue
            member_entries: list[tuple[str, str, str]] = []
            for col, member_label in members.items():
                if col not in cols:
                    continue
                if col.startswith(SPECIAL_STRUCTURAL_PREFIXES):
                    continue
                raw_val = attrs.get(col)
                if raw_val is None or (isinstance(raw_val, float) and pd.isna(raw_val)):
                    continue
                if isinstance(raw_val, str) and raw_val.strip().lower() in {"missing", "nan", "none"}:
                    continue
                val = _maybe_label_numeric(col, raw_val)
                member_entries.append((member_label, val, col))
            if member_entries:
                title = group.get("title")
                if not isinstance(title, str) or not title.strip():
                    continue
                group_title = title.strip()
                by_scale: dict[tuple[object, ...], list[tuple[str, str]]] = {}
                for member_label, val, col in member_entries:
                    scale = _scale_descriptor(col)
                    if scale is None:
                        non_scale_lines.append(f"- {group_title}:")
                        non_scale_lines.append(f"  - {member_label}: {val}")
                        continue
                    key, scale_label = scale
                    by_scale.setdefault((key, scale_label), []).append((member_label, val))

                scale_inline = bool(group.get("scale_inline"))
                if scale_inline and len(by_scale) == 1:
                    (key, _scale_label), entries = next(iter(by_scale.items()))
                    lo_s, hi_s = key[0], key[1]
                    non_scale_lines.append(f"- {group_title} (scale={lo_s}-{hi_s}):")
                    for member_label, val in entries:
                        non_scale_lines.append(f"  - {member_label}: {val}")
                    continue

                for (key, scale_label), entries in by_scale.items():
                    if key not in scale_items:
                        scale_items[key] = {"label": scale_label, "items": []}
                        scale_order.append(key)
                    items = scale_items[key]["items"]
                    existing = next(
                        (item for item in items if item["type"] == "group" and item["title"] == group_title),
                        None,
                    )
                    if existing is None:
                        existing = {"type": "group", "title": group_title, "entries": []}
                        items.append(existing)
                    existing["entries"].extend(entries)

        inline_entries: list[tuple[str, str, str]] = []
        for col in cols:
            if col.startswith(SPECIAL_STRUCTURAL_PREFIXES):
                continue
            if col in grouped_cols:
                continue
            raw_val = attrs.get(col)
            if raw_val is None or (isinstance(raw_val, float) and pd.isna(raw_val)):
                continue
            if isinstance(raw_val, str) and raw_val.strip().lower() in {"missing", "nan", "none"}:
                continue
            label = _display_label(col)
            val = _maybe_label_numeric(col, raw_val)
            inline_entries.append((label, val, col))
        for label, val, col in inline_entries:
            scale = _scale_descriptor(col)
            if scale is None:
                non_scale_lines.append(f"- {label}: {val}")
                continue
            key, scale_label = scale
            if key not in scale_items:
                scale_items[key] = {"label": scale_label, "items": []}
                scale_order.append(key)
            scale_items[key]["items"].append({"type": "item", "label": label, "value": val})

        for key in scale_order:
            scale_label = scale_items[key]["label"]
            items = scale_items[key]["items"]
            total_items = 0
            for item in items:
                if item["type"] == "group":
                    total_items += len(item["entries"])
                else:
                    total_items += 1
            if total_items == 1 and items and items[0]["type"] == "item":
                item = items[0]
                inline_label = f"{item['label']} (scale {scale_label})"
                lines.append(f"- {inline_label}: {item['value']}")
                continue
            lines.append(f"Scale: {scale_label}")
            for item in items:
                if item["type"] == "group":
                    lines.append(f"- {item['title']}:")
                    for member_label, value in item["entries"]:
                        lines.append(f"  - {member_label}: {value}")
                else:
                    lines.append(f"- {item['label']}: {item['value']}")
            lines.append("")

        if non_scale_lines:
            lines.extend(non_scale_lines)
        return lines

    def _format_task(outcome: str | None) -> str:
        if args.task_template and outcome:
            return args.task_template.format(outcome=outcome)
        return args.task

    def _bucket_scale(val: object, lo: int, hi: int) -> int | None:
        num = _as_numeric(val)
        if num is None:
            return None
        rounded = int(round(num))
        return max(lo, min(hi, rounded))

    def _as_bool(val: object) -> bool | None:
        num = _as_numeric(val)
        if num is None:
            return None
        return bool(round(num))

    def _income_quintiles_by_country() -> dict[str, list[float]]:
        if hasattr(_income_quintiles_by_country, "_cache"):
            return _income_quintiles_by_country._cache
        path = Path("preprocess/output/clean_processed_survey.csv")
        if not path.exists():
            _income_quintiles_by_country._cache = {}
            return {}
        df = pd.read_csv(path, usecols=["country", "income_ppp_norm"])
        df = df.dropna(subset=["country", "income_ppp_norm"])
        if df.empty:
            _income_quintiles_by_country._cache = {}
            return {}
        out: dict[str, list[float]] = {}
        for country, g in df.groupby("country", dropna=True):
            arr = pd.to_numeric(g["income_ppp_norm"], errors="coerce").dropna()
            if arr.empty:
                continue
            qs = arr.quantile([0.2, 0.4, 0.6, 0.8]).tolist()
            out[str(country)] = [float(x) for x in qs]
        _income_quintiles_by_country._cache = out
        return out

    def _income_phrase(val: object, country: object, ctx: dict[str, str]) -> str | None:
        num = _as_numeric(val)
        if num is None:
            return None
        cuts = _income_quintiles_by_country().get(str(country))
        if not cuts:
            return None
        labels = [
            "on the very low side",
            "on the lower side",
            "around the middle",
            "on the higher side",
            "on the very high side",
        ]
        idx = 0
        for c in cuts:
            if num > c:
                idx += 1
        idx = min(idx, 4)
        return f"{ctx['possessive'].capitalize()} income is {labels[idx]}."

    def _apply_pronoun_context(text: str, ctx: dict[str, str]) -> str:
        if ctx["subject"] == "I":
            return text
        replacements: list[tuple[str, str]] = [
            (r"(?:(?<=^)|(?<=[.!?]\s))I'm\b", "They are"),
            (r"(?:(?<=^)|(?<=[.!?]\s))I’m\b", "They are"),
            (r"\bI'm\b", "they are"),
            (r"\bI’m\b", "they are"),
            (r"(?:(?<=^)|(?<=[.!?]\s))I've\b", "They have"),
            (r"(?:(?<=^)|(?<=[.!?]\s))I’ve\b", "They have"),
            (r"\bI've\b", "they have"),
            (r"\bI’ve\b", "they have"),
            (r"(?:(?<=^)|(?<=[.!?]\s))I'd\b", "They would"),
            (r"(?:(?<=^)|(?<=[.!?]\s))I’d\b", "They would"),
            (r"\bI'd\b", "they would"),
            (r"\bI’d\b", "they would"),
            (r"(?:(?<=^)|(?<=[.!?]\s))I'll\b", "They will"),
            (r"(?:(?<=^)|(?<=[.!?]\s))I’ll\b", "They will"),
            (r"\bI'll\b", "they will"),
            (r"\bI’ll\b", "they will"),
            (r"(?:(?<=^)|(?<=[.!?]\s))I am\b", "They are"),
            (r"\bI am\b", "they are"),
            (r"(?:(?<=^)|(?<=[.!?]\s))I have\b", "They have"),
            (r"\bI have\b", "they have"),
            (r"(?:(?<=^)|(?<=[.!?]\s))I had\b", "They had"),
            (r"\bI had\b", "they had"),
            (r"(?:(?<=^)|(?<=[.!?]\s))I will\b", "They will"),
            (r"\bI will\b", "they will"),
        ]
        for pattern, repl in replacements:
            text = re.sub(pattern, repl, text)
        text = re.sub(r"\bmyself\b", "themselves", text)
        text = re.sub(r"\bmine\b", "theirs", text)
        text = re.sub(r"\bme\b", "them", text)
        text = re.sub(r"\bmy\b", "their", text)
        text = re.sub(r"\bMy\b", "Their", text)
        text = re.sub(r"(?:(?<=^)|(?<=[.!?]\s))I\b", "They", text)
        text = re.sub(r"\bI\b", "they", text)
        return text

    def _trust_phrase(val: object, obj: str, ctx: dict[str, str]) -> str | None:
        level = _bucket_scale(val, 1, 5)
        if level is None:
            return None
        mapping = {
            1: "I don't really trust {obj} at all.",
            2: "I trust {obj} a little, but I'm cautious.",
            3: "I'm somewhat neutral — I trust {obj} in some cases, but not blindly.",
            4: "I generally trust {obj}.",
            5: "I trust {obj} a lot.",
        }
        return _apply_pronoun_context(mapping[level].format(obj=obj), ctx)

    def _scale_phrase(val: object, mapping: dict[int, str], ctx: dict[str, str]) -> str | None:
        level = _bucket_scale(val, min(mapping), max(mapping))
        if level is None:
            return None
        return _apply_pronoun_context(mapping[level], ctx)

    def _mfq_binding_joint_phrase(
        agreement_val: object,
        relevance_val: object,
        ctx: dict[str, str],
    ) -> str | None:
        agreement_level = _bucket_scale(agreement_val, 1, 6)
        relevance_level = _bucket_scale(relevance_val, 1, 6)
        if agreement_level is None or relevance_level is None:
            return None
        agrees = agreement_level >= 4
        relevance_high = relevance_level >= 5
        relevance_mid = relevance_level in (3, 4)
        label = "loyalty, authority, and purity"
        if agrees and relevance_high:
            s = f"I endorse principles of {label}, and they are important in how I judge right and wrong."
        elif agrees and relevance_mid:
            s = f"I endorse principles of {label}, and they matter somewhat when I judge right and wrong."
        elif agrees:
            s = f"I endorse principles of {label}, but they do not matter much in my moral judgments."
        elif relevance_high:
            s = f"Principles of {label} matter to me, even if I do not fully endorse them."
        elif relevance_mid:
            s = (
                f"I do not fully endorse principles of {label}, but they still matter somewhat when I judge "
                "right and wrong."
            )
        else:
            s = f"I do not endorse principles of {label}, and they are not very relevant to my moral judgments."
        return _apply_pronoun_context(s, ctx)

    def _infection_likelihood_phrase(val: object, ctx: dict[str, str]) -> str | None:
        num = _as_numeric(val)
        if num is None:
            return None
        if num < 1.5:
            s = "I think it's very unlikely I'll catch an infectious disease like COVID or flu."
        elif num < 2.5:
            s = "I think it's unlikely I'll catch an infectious disease like COVID or flu."
        elif num < 3.5:
            s = "I think there's a fair chance I'll catch an infectious disease like COVID or flu."
        elif num < 4.5:
            s = "I think it's likely I'll catch an infectious disease like COVID or flu."
        else:
            s = "I think it's very likely I'll catch an infectious disease like COVID or flu."
        return _apply_pronoun_context(s, ctx)

    def _norms_phrase(val: object, group_label: str, ctx: dict[str, str]) -> str | None:
        level = _bucket_scale(val, 1, 7)
        if level is None:
            return None
        mapping = {
            1: f"My {group_label} generally don’t hold progressive views on these issues.",
            2: f"My {group_label} tend to be more conservative than progressive on these issues.",
            3: f"My {group_label} lean slightly away from progressive positions.",
            4: f"My {group_label} are mixed—no clear progressive or conservative tilt.",
            5: f"My {group_label} lean somewhat progressive on these issues.",
            6: f"My {group_label} are clearly progressive on these issues.",
            7: f"My {group_label} are strongly progressive on these issues.",
        }
        return mapping[level].replace("My ", f"{ctx['possessive'].capitalize()} ")

    def _covax_legitimacy_phrase(val: object, ctx: dict[str, str]) -> str | None:
        num = _as_numeric(val)
        if num is None:
            return None
        if num < 0.2:
            s = "I have almost no doubts about how COVID vaccines are tested and approved."
        elif num < 0.4:
            s = "I have only a few doubts about how COVID vaccines are tested and approved."
        elif num < 0.6:
            s = "I have mixed feelings about how COVID vaccines are tested and approved."
        elif num < 0.8:
            s = "I have quite a lot of doubts about how COVID vaccines are tested and approved."
        else:
            s = "I have very strong doubts about how COVID vaccines are tested and approved."
        return _apply_pronoun_context(s, ctx)

    def _age_phrase(val: object, ctx: dict[str, str]) -> str | None:
        num = _as_numeric(val)
        if num is None:
            return None
        if num < 30:
            band = "in my 20s"
        elif num < 45:
            band = "in my 30s or early 40s"
        elif num < 60:
            band = "in my late 40s or 50s"
        elif num < 75:
            band = "in my 60s or early 70s"
        else:
            band = "in my mid-70s or older"
        if ctx["subject"] == "They":
            band = band.replace("my", ctx["possessive"])
        return band

    def _country_label(country: object) -> str:
        if country is None:
            return ""
        raw = str(country).strip()
        mapping = {
            "IT": "Italy",
            "FR": "France",
            "DE": "Germany",
            "ES": "Spain",
            "HU": "Hungary",
            "UK": "United Kingdom",
        }
        return mapping.get(raw, raw)

    def _build_narrative(attrs: dict[str, object], driver_cols: list[str] | None, strategy: str) -> str:
        cols = set(driver_cols) if driver_cols else set(attrs.keys())
        first = strategy == "first_person"
        ctx = {
            "subject": "I" if first else "They",
            "subject_lc": "I" if first else "they",
            "be": "am" if first else "are",
            "possessive": "my" if first else "their",
        }
        narrative_cols = {
            "sex_female",
            "age",
            "M_education",
            "country",
            "income_ppp_norm",
            "minority",
            "M_religion",
            "health_poor",
            "medical_conditions",
            "medical_conditions_c",
            "covid_num",
            "exper_illness",
            "side_effects",
            "side_effects_other",
            "institutional_trust_avg",
            "group_trust_vax",
            "trust_traditional",
            "trust_onlineinfo",
            "trust_social",
            "covax_legitimacy_scepticism_idx",
            "fivec_pro_vax_idx",
            "fivec_confidence",
            "fivec_low_constraints",
            "fivec_low_complacency",
            "fivec_collective_responsibility",
            "vaccine_risk_avg",
            "vaccine_fear_avg",
            "flu_vaccine_risk",
            "flu_disease_fear",
            "covid_perceived_danger_Y",
            "disease_fear_avg",
            "infection_likelihood_avg",
            "severity_if_infected_avg",
            "social_norms_vax_avg",
            "family_progressive_norms_nonvax_idx",
            "friends_progressive_norms_nonvax_idx",
            "colleagues_progressive_norms_nonvax_idx",
            "moral_progressive_orientation",
            "mfq_agreement_binding_idx",
            "mfq_relevance_binding_idx",
            "flu_vaccinated_pre_pandemic",
            "flu_vaccinated_2020_2021",
            "flu_vaccinated_2021_2022",
            "flu_vaccinated_2022_2023",
            "duration_socialmedia",
            "duration_infomedia",
        }

        def _is_missing(val: object) -> bool:
            if val is None:
                return True
            if isinstance(val, float) and pd.isna(val):
                return True
            if isinstance(val, str) and val.strip().lower() in {"missing", "nan", "none"}:
                return True
            return False

        blocks: list[str] = []

        def _round_half_up(num: float) -> int:
            return int(math.floor(num + 0.5))

        def _bucket_scale_half_up(val: object, lo: int, hi: int) -> int | None:
            num = _as_numeric(val)
            if num is None:
                return None
            rounded = _round_half_up(num)
            return max(lo, min(hi, rounded))

        def _mean(values: list[float]) -> float | None:
            if not values:
                return None
            return sum(values) / len(values)

        def _scale_extreme(val: object, hi: int) -> str | None:
            num = _as_numeric(val)
            if num is None:
                return None
            if hi == 7:
                low_thr, high_thr = 2, 6
            elif hi == 6:
                low_thr, high_thr = 2, 5
            elif hi == 5:
                low_thr, high_thr = 2, 4
            else:
                return None
            if num <= low_thr:
                return "low"
            if num >= high_thr:
                return "high"
            return "mid"

        def _join_sentence(base: str, clauses: list[str]) -> str:
            parts = [base] + [c for c in clauses if c]
            if not parts:
                return ""
            main = parts[0].rstrip(".")
            extras = [p.rstrip(".") for p in parts[1:]]
            if not extras:
                return f"{main}."
            return f"{main}; " + "; ".join(extras) + "."

        def _covax_legitimacy_clause(val: object) -> str | None:
            num = _as_numeric(val)
            if num is None:
                return None
            if num <= 0.2:
                return "I have almost no doubts about how COVID vaccines are approved"
            if num <= 0.4:
                return "I have only a few doubts about how COVID vaccines are approved"
            if num <= 0.6:
                return "I have some doubts about how COVID vaccines are approved"
            if num <= 0.8:
                return "I have many doubts about how COVID vaccines are approved"
            return "I have very strong doubts about how COVID vaccines are approved"

        def _media_time_clause() -> str | None:
            parts: list[str] = []
            if "duration_socialmedia" in cols:
                level = _bucket_scale_half_up(attrs.get("duration_socialmedia"), 1, 5)
                if level is not None:
                    sm_map = {
                        1: "I spend very little time on social media",
                        2: "I spend a little time on social media",
                        3: "I spend some time on social media",
                        4: "I spend quite a lot of time on social media",
                        5: "I spend a lot of time on social media",
                    }
                    parts.append(sm_map[level])
            if "duration_infomedia" in cols:
                level = _bucket_scale_half_up(attrs.get("duration_infomedia"), 1, 5)
                if level is not None:
                    info_map = {
                        1: "I follow news and information very little",
                        2: "I follow news and information a little",
                        3: "I follow news and information sometimes",
                        4: "I follow news and information quite a lot",
                        5: "I follow news and information a lot",
                    }
                    parts.append(info_map[level])
            if not parts:
                return None
            if len(parts) == 2:
                return f"{parts[0]} and {parts[1]}"
            return parts[0]

        def _summarize_trust_info() -> str | None:
            source_cols = [
                ("institutional_trust_avg", "institutions and public authorities"),
                ("trust_traditional", "traditional media"),
                ("trust_onlineinfo", "online sources"),
                ("trust_social", "social media"),
                ("group_trust_vax", "people I know"),
            ]
            contrast_candidates = {
                "trust_traditional",
                "trust_onlineinfo",
                "trust_social",
                "group_trust_vax",
            }
            sources: list[tuple[str, float]] = []
            for col, label in source_cols:
                if col not in cols:
                    continue
                num = _as_numeric(attrs.get(col))
                if num is None:
                    continue
                sources.append((label, num))
            if not sources:
                return None
            mean_val = _mean([v for _, v in sources])
            level = _bucket_scale_half_up(mean_val, 1, 5)
            overall_map = {
                1: "Overall, I do not really trust vaccine information sources",
                2: "Overall, I am cautious about vaccine information",
                3: "Overall, I feel neutral about vaccine information",
                4: "Overall, I generally trust vaccine information sources",
                5: "Overall, I trust vaccine information sources a lot",
            }
            overall = overall_map[level]
            inst_clause = None
            for col, label in source_cols:
                if col != "institutional_trust_avg":
                    continue
                if col not in cols:
                    continue
                inst_level = _bucket_scale_half_up(attrs.get(col), 1, 5)
                if inst_level is not None:
                    inst_map = {
                        1: f"I do not really trust {label}",
                        2: f"I am cautious about {label}",
                        3: f"I feel neutral about {label}",
                        4: f"I generally trust {label}",
                        5: f"I trust {label} a lot",
                    }
                    inst_clause = inst_map[inst_level]
            contrast_sources = [
                (label, val, col)
                for col, label in source_cols
                for (lab, val) in sources
                if lab == label and col in contrast_candidates
            ]
            if contrast_sources:
                top_label, top_val, _ = max(contrast_sources, key=lambda kv: kv[1])
                bottom_label, bottom_val, _ = min(contrast_sources, key=lambda kv: kv[1])
            else:
                top_label, top_val = max(sources, key=lambda kv: kv[1])
                bottom_label, bottom_val = min(sources, key=lambda kv: kv[1])
            contrast = None
            spread = top_val - bottom_val
            if spread < 0.5:
                contrast = "I do not have a clear preference across sources"
            elif spread < 1.0:
                contrast = f"I lean slightly toward {top_label} over {bottom_label}"
            else:
                contrast = f"I trust {top_label} much more than {bottom_label}"
            clauses = [inst_clause, contrast]
            covax_clause = None
            if "covax_legitimacy_scepticism_idx" in cols:
                covax_clause = _covax_legitimacy_clause(attrs.get("covax_legitimacy_scepticism_idx"))
            if covax_clause and overall == "Overall, I feel neutral about vaccine information":
                if contrast == "Trust is fairly similar across sources":
                    clauses = [f"{contrast}, but {covax_clause}"]
                    covax_clause = None
            if covax_clause:
                clauses.append(covax_clause)
            clauses.append(_media_time_clause())
            sentence = _join_sentence(overall, clauses)
            return _apply_pronoun_context(sentence, ctx)

        def _summarize_vaccine_stance() -> str | None:
            core_cols = [
                "fivec_pro_vax_idx",
                "fivec_confidence",
                "fivec_low_complacency",
                "fivec_collective_responsibility",
            ]
            core_vals: list[float] = []
            for col in core_cols:
                if col not in cols:
                    continue
                num = _as_numeric(attrs.get(col))
                if num is not None:
                    core_vals.append(num)
            if not core_vals:
                return None
            mean_val = _mean(core_vals)
            level = _bucket_scale_half_up(mean_val, 1, 7)
            overall_map = {
                1: "Overall, I am strongly against vaccines",
                2: "Overall, I am strongly against vaccines",
                3: "Overall, I lean against vaccines",
                4: "Overall, I feel mixed about vaccines",
                5: "Overall, I lean in favor of vaccines",
                6: "Overall, I am strongly in favor of vaccines",
                7: "Overall, I am strongly in favor of vaccines",
            }
            overall = overall_map[level]
            clauses: list[str] = []
            if "fivec_pro_vax_idx" in cols:
                pro_level = _bucket_scale_half_up(attrs.get("fivec_pro_vax_idx"), 1, 7)
                if pro_level is not None:
                    if pro_level <= 2:
                        clauses.append("In general, I am skeptical of vaccines")
                    elif pro_level == 3:
                        clauses.append("In general, I lean against vaccines")
                    elif pro_level == 4:
                        clauses.append("In general, I am on the fence about vaccines")
                    elif pro_level == 5:
                        clauses.append("In general, I lean toward vaccines")
                    else:
                        clauses.append("In general, I am pro-vaccine")
            if "fivec_confidence" in cols:
                conf_level = _bucket_scale_half_up(attrs.get("fivec_confidence"), 1, 7)
                if conf_level is not None:
                    if conf_level <= 2:
                        clauses.append("I am not very confident vaccines are safe")
                    elif conf_level == 3:
                        clauses.append("I am only a little confident vaccines are safe")
                    elif conf_level == 4:
                        clauses.append("I am somewhat confident vaccines are safe")
                    elif conf_level == 5:
                        clauses.append("I am fairly confident vaccines are safe")
                    else:
                        clauses.append("I am very confident vaccines are safe")
            if "fivec_low_constraints" in cols:
                constraint_level = _bucket_scale_half_up(attrs.get("fivec_low_constraints"), 1, 7)
                if constraint_level is None:
                    pass
                elif constraint_level <= 2:
                    clauses.append("Everyday stress and practical barriers make it hard for me to get vaccinated")
                elif constraint_level == 3:
                    clauses.append("Stress and barriers make it somewhat hard for me to get vaccinated")
                elif constraint_level == 4:
                    clauses.append("Getting vaccinated is neither easy nor hard for me")
                elif constraint_level == 5:
                    clauses.append("Getting vaccinated is fairly manageable for me")
                else:
                    clauses.append("Even when I am busy or stressed, I can follow through on getting vaccinated")
            if "fivec_collective_responsibility" in cols:
                coll_level = _bucket_scale_half_up(attrs.get("fivec_collective_responsibility"), 1, 7)
                if coll_level is None:
                    pass
                elif coll_level <= 2:
                    clauses.append("Protecting others is not a big factor for me here")
                elif coll_level == 3:
                    clauses.append("Protecting others matters only a little to me here")
                elif coll_level == 4:
                    clauses.append("Protecting others matters to me, but it is not the main driver")
                elif coll_level == 5:
                    clauses.append("Protecting others matters a lot to me here")
                else:
                    clauses.append("Protecting others is very important to me here")
            if "vaccine_risk_avg" in cols:
                risk_level = _bucket_scale_half_up(attrs.get("vaccine_risk_avg"), 1, 5)
                if risk_level is not None:
                    risk_map = {
                        1: "I think vaccine risks are very low",
                        2: "I think vaccine risks are low",
                        3: "I think vaccine risks are moderate",
                        4: "I think vaccine risks are high",
                        5: "I think vaccine risks are very high",
                    }
                    clauses.append(risk_map[risk_level])
            if "vaccine_fear_avg" in cols:
                fear_level = _bucket_scale_half_up(attrs.get("vaccine_fear_avg"), 1, 5)
                if fear_level is not None:
                    fear_map = {
                        1: "Emotionally, I feel very calm about side effects",
                        2: "Emotionally, I feel calm about side effects",
                        3: "Emotionally, I feel somewhat uneasy about side effects",
                        4: "Emotionally, I feel afraid about side effects",
                        5: "Emotionally, I feel very afraid about side effects",
                    }
                    clauses.append(fear_map[fear_level])
            if "vaccine_risk_avg" not in cols and "vaccine_fear_avg" not in cols and "flu_vaccine_risk" in cols:
                risk_level = _bucket_scale_half_up(attrs.get("flu_vaccine_risk"), 1, 5)
                if risk_level is not None:
                    flu_map = {
                        1: "I do not see the flu vaccine as risky",
                        2: "I see the flu vaccine as a little risky",
                        3: "I have some concerns about the flu vaccine",
                        4: "I see the flu vaccine as fairly risky",
                        5: "I see the flu vaccine as very risky",
                    }
                    clauses.append(flu_map[risk_level])
            sentence = _join_sentence(overall, clauses)
            return _apply_pronoun_context(sentence, ctx)

        def _summarize_disease_stakes() -> str | None:
            dim_cols = [
                "disease_fear_avg",
                "infection_likelihood_avg",
                "severity_if_infected_avg",
                "covid_perceived_danger_Y",
            ]
            dim_vals: list[tuple[str, float]] = []
            for col in dim_cols:
                if col not in cols:
                    continue
                num = _as_numeric(attrs.get(col))
                if num is None:
                    continue
                dim_vals.append((col, num))
            if not dim_vals:
                return None
            clauses: list[str] = []
            fear_level = None
            severity_level = None
            covid_level = None
            likelihood_clause = None
            comp_level = None
            for col, num in dim_vals:
                if col == "infection_likelihood_avg":
                    likelihood_clause = _infection_likelihood_phrase(num, ctx)
                elif col == "disease_fear_avg":
                    fear_level = _bucket_scale_half_up(num, 1, 5)
                elif col == "severity_if_infected_avg":
                    severity_level = _bucket_scale_half_up(num, 1, 5)
                elif col == "covid_perceived_danger_Y":
                    covid_level = _bucket_scale_half_up(num, 1, 5)
            if "fivec_low_complacency" in cols:
                comp_level = _bucket_scale_half_up(attrs.get("fivec_low_complacency"), 1, 7)

            comp_phrase = None
            if comp_level is not None:
                if comp_level <= 2:
                    comp_phrase = "I do not see vaccine-preventable diseases like COVID, flu, measles, and meningitis as a big deal"
                elif comp_level == 3:
                    comp_phrase = "I see vaccine-preventable diseases like COVID, flu, measles, and meningitis as somewhat serious"
                elif comp_level == 4:
                    comp_phrase = "I see vaccine-preventable diseases like COVID, flu, measles, and meningitis as moderately serious"
                elif comp_level == 5:
                    comp_phrase = (
                        "I see vaccine-preventable diseases like COVID, flu, measles, and meningitis as quite serious"
                    )
                else:
                    comp_phrase = (
                        "I take vaccine-preventable diseases like COVID, flu, measles, and meningitis seriously"
                    )

            if fear_level is not None or severity_level is not None:
                fear_map = {
                    1: "not very worried about these diseases",
                    2: "only a little worried about these diseases",
                    3: "moderately worried about these diseases",
                    4: "quite worried about these diseases",
                    5: "very worried about these diseases",
                }
                severity_map = {
                    1: "I expect an infection would be mild",
                    2: "I expect an infection would be not too serious",
                    3: "I expect an infection would be moderately serious",
                    4: "I expect an infection would be quite serious",
                    5: "I expect an infection would be very serious",
                }
                fear_phrase = fear_map.get(fear_level) if fear_level is not None else None
                severity_phrase = severity_map.get(severity_level) if severity_level is not None else None
                if fear_phrase and severity_phrase:
                    combined = f"I am {fear_phrase}, and {severity_phrase}"
                elif fear_phrase:
                    combined = f"I am {fear_phrase}"
                elif severity_phrase:
                    combined = severity_phrase
                else:
                    combined = None
                if combined:
                    if comp_phrase:
                        clauses.append(f"{comp_phrase}, and {combined}")
                        comp_phrase = None
                    else:
                        clauses.append(combined)
            if comp_phrase:
                clauses.append(comp_phrase)
            if likelihood_clause:
                clauses.append(likelihood_clause.rstrip("."))
            if covid_level is not None:
                covid_map = {
                    1: "COVID does not feel very dangerous to me",
                    2: "COVID feels a bit dangerous to me",
                    3: "COVID feels moderately dangerous to me",
                    4: "COVID feels quite dangerous to me",
                    5: "COVID feels extremely dangerous to me",
                }
                clauses.append(covid_map[covid_level])
            if "flu_disease_fear" in cols:
                flu_level = _bucket_scale_half_up(attrs.get("flu_disease_fear"), 1, 5)
                if flu_level is not None:
                    flu_map = {
                        1: "I am not very afraid of the flu",
                        2: "I am only a little worried about the flu",
                        3: "I feel moderately worried about the flu",
                        4: "I am quite worried about the flu",
                        5: "I am very afraid of the flu",
                    }
                    clauses.append(flu_map[flu_level])
            if not clauses:
                return None
            sentence = _join_sentence(clauses[0], clauses[1:])
            return _apply_pronoun_context(sentence, ctx)

        def _progressive_self_phrase() -> str | None:
            if "moral_progressive_orientation" not in cols:
                return None
            level = _bucket_scale_half_up(attrs.get("moral_progressive_orientation"), 1, 7)
            if level is None:
                return None
            if level <= 2:
                return "I lean away from those progressive positions"
            if level == 3:
                return "I slightly lean away from those progressive positions"
            if level == 4:
                return "I feel mixed about those progressive positions"
            if level == 5:
                return "I lean somewhat toward those progressive positions"
            return "I strongly endorse those progressive positions"

        def _vax_norms_phrase() -> str | None:
            if "social_norms_vax_avg" not in cols:
                return None
            level = _bucket_scale_half_up(attrs.get("social_norms_vax_avg"), 1, 7)
            if level is None:
                return None
            if level <= 2:
                return "People around me generally think vaccines are not important"
            if level == 3:
                return "People around me lean against vaccines"
            if level == 4:
                return "People around me are mixed about vaccines"
            if level == 5:
                return "People around me lean in favor of vaccines"
            return "People around me strongly value vaccines"

        def _progressive_circles_clause() -> str | None:
            group_cols = [
                ("family", "family_progressive_norms_nonvax_idx"),
                ("friends", "friends_progressive_norms_nonvax_idx"),
                ("colleagues", "colleagues_progressive_norms_nonvax_idx"),
            ]
            values: list[tuple[str, float]] = []
            for label, col in group_cols:
                if col not in cols:
                    continue
                num = _as_numeric(attrs.get(col))
                if num is None:
                    continue
                values.append((label, num))
            if not values:
                return None
            mean_val = _mean([v for _, v in values])
            level = _bucket_scale_half_up(mean_val, 1, 7)
            issue_label = "climate harm, LGBT adoption rights, and immigration tolerance"
            overall_map = {
                1: f"My circles are generally conservative on {issue_label}",
                2: f"My circles lean away from progressive positions on {issue_label}",
                3: f"My circles lean slightly away from progressive positions on {issue_label}",
                4: f"My circles are mixed on {issue_label}",
                5: f"My circles lean somewhat progressive on {issue_label}",
                6: f"My circles are clearly progressive on {issue_label}",
                7: f"My circles are strongly progressive on {issue_label}",
            }
            overall = overall_map[level]
            top_label, top_val = max(values, key=lambda kv: kv[1])
            bottom_label, bottom_val = min(values, key=lambda kv: kv[1])
            contrast = None
            if top_val - bottom_val >= 2:
                contrast = f"my {top_label} are more progressive than my {bottom_label}"
            elif level != 4:
                top_extreme = _scale_extreme(top_val, 7)
                bottom_extreme = _scale_extreme(bottom_val, 7)
                if top_extreme == "high" and bottom_extreme != "high":
                    contrast = f"my {top_label} are especially progressive"
                elif bottom_extreme == "low" and top_extreme != "low":
                    contrast = f"my {bottom_label} lean more conservative"
                else:
                    contrast = "views are fairly similar across groups"
            else:
                contrast = None
            if contrast:
                return f"{overall}, {contrast}"
            return overall

        def _summarize_norms_moral() -> str | None:
            clauses: list[str] = []
            self_phrase = _progressive_self_phrase()
            if self_phrase:
                clauses.append(self_phrase)
            vax_norms = _vax_norms_phrase()
            if vax_norms:
                clauses.append(vax_norms)
            circles_clause = _progressive_circles_clause()
            if circles_clause:
                clauses.append(circles_clause)
            mfq_clause = None
            if "mfq_agreement_binding_idx" in cols or "mfq_relevance_binding_idx" in cols:
                mfq_clause = _mfq_binding_joint_phrase(
                    attrs.get("mfq_agreement_binding_idx"),
                    attrs.get("mfq_relevance_binding_idx"),
                    ctx,
                )
            if mfq_clause:
                clauses.append(mfq_clause)
            if not clauses:
                return None
            sentence = _join_sentence(clauses[0], clauses[1:])
            return _apply_pronoun_context(sentence, ctx)

        # Demography / context
        demo_bits: list[str] = []
        if "sex_female" in cols:
            val = _as_bool(attrs.get("sex_female"))
            if val is not None:
                demo_bits.append("a woman" if val else "a man")
        age = _age_phrase(attrs.get("age"), ctx) if "age" in cols else None
        if age:
            demo_bits.append(age)
        if "M_education" in cols:
            edu = _bucket_scale(attrs.get("M_education"), 0, 2)
            if edu is not None:
                edu_map = {
                    0: "with a lower level of formal education",
                    1: "with a medium level of formal education",
                    2: "with a higher level of formal education",
                }
                demo_bits.append(edu_map[edu])
        country = attrs.get("country")
        if country:
            demo_bits.append(f"living in {_country_label(country)}")
        demo_clauses: list[str] = []
        if "minority" in cols:
            val = _as_bool(attrs.get("minority"))
            if val is not None:
                demo_clauses.append(
                    "I consider myself part of a minority group"
                    if val
                    else "I do not consider myself part of a minority group"
                )
        if "M_religion" in cols:
            val = _as_bool(attrs.get("M_religion"))
            if val is not None:
                if val:
                    demo_clauses.append("I am either Christian or atheist/agnostic (not another religion)")
                else:
                    demo_clauses.append(
                        "I am outside the Christianity/atheist-agnostic bucket (e.g., Orthodox, Muslim, Jewish, "
                        "Eastern, or another religion)"
                    )
        blocks.append(_apply_pronoun_context("It is 2024.", ctx))
        if demo_bits:
            base = f"I am {', '.join(demo_bits)}"
            sentence = _join_sentence(base, demo_clauses)
            blocks.append(_apply_pronoun_context(sentence, ctx))
        elif demo_clauses:
            sentence = _join_sentence(demo_clauses[0], demo_clauses[1:])
            blocks.append(_apply_pronoun_context(sentence, ctx))
        income = _income_phrase(attrs.get("income_ppp_norm"), country, ctx) if "income_ppp_norm" in cols else None
        if income:
            blocks.append(income)

        # Health background
        health_clauses: list[str] = []
        if "health_poor" in cols:
            health_map = {
                1: "Overall, my health is very good",
                2: "My health is good",
                3: "My health is okay",
                4: "My health is not great",
                5: "Overall, my health is poor",
            }
            phrase = _scale_phrase(attrs.get("health_poor"), {k: f"{v}." for k, v in health_map.items()}, ctx)
            if phrase:
                health_clauses.append(phrase.rstrip("."))
        if "medical_conditions" in cols:
            val = _as_bool(attrs.get("medical_conditions"))
            if val:
                health_clauses.append("I have at least one underlying medical condition")
        if "medical_conditions_c" in cols:
            val = _as_bool(attrs.get("medical_conditions_c"))
            if val:
                health_clauses.append("Someone in my household has an underlying medical condition")
        if "covid_num" in cols:
            num = _as_numeric(attrs.get("covid_num"))
            if num is not None and num >= 0.5:
                if num < 1.5:
                    health_clauses.append("I have had COVID once")
                elif num < 2.5:
                    health_clauses.append("I have had COVID twice")
                else:
                    health_clauses.append("I have had COVID multiple times")
        if "exper_illness" in cols:
            val = _as_bool(attrs.get("exper_illness"))
            if val:
                health_clauses.append(
                    "I personally know someone who was hospitalized or died due to a serious infectious illness (not COVID-19)"
                )
        side_self = _as_bool(attrs.get("side_effects")) if "side_effects" in cols else None
        side_other = _as_bool(attrs.get("side_effects_other")) if "side_effects_other" in cols else None
        if side_self is not None or side_other is not None:
            if side_self and side_other:
                health_clauses.append(
                    "Vaccine side effects (beyond temporary arm pain) have affected me and people I know"
                )
            elif side_self:
                health_clauses.append("I have experienced vaccine side effects (beyond temporary arm pain)")
            elif side_other:
                health_clauses.append(
                    "I know someone who has experienced vaccine side effects (beyond temporary arm pain)"
                )
            else:
                health_clauses.append(
                    "Vaccine side effects (beyond temporary arm pain) have not been a major issue for me or people around me"
                )
        flu_bits: list[tuple[str, bool]] = []
        single_flu_phrase: str | None = None
        flu_map = {
            "flu_vaccinated_pre_pandemic": "Before 2020, I did get the flu vaccine",
            "flu_vaccinated_2020_2021": "I got a flu shot in the 2020/2021 season",
            "flu_vaccinated_2021_2022": "I got a flu shot in the 2021/2022 season",
            "flu_vaccinated_2022_2023": "I got a flu shot in the 2022/2023 season",
        }
        for key, yes_phrase in flu_map.items():
            if key not in cols:
                continue
            val = _as_bool(attrs.get(key))
            if val is None:
                continue
            flu_bits.append((key, val))
            if len(flu_bits) == 1:
                phrase = (
                    yes_phrase
                    if val
                    else yes_phrase.replace("I got", "I did not get").replace("I did get", "I did not usually get")
                )
                single_flu_phrase = phrase
            else:
                single_flu_phrase = None
        if len(flu_bits) == 1 and single_flu_phrase:
            health_clauses.append(single_flu_phrase)
        elif len(flu_bits) >= 2:
            yes_count = sum(1 for _, v in flu_bits if v)
            no_count = sum(1 for _, v in flu_bits if not v)
            if yes_count >= 2 and yes_count > no_count:
                health_clauses.append("I have been fairly consistent about getting the flu vaccine")
            elif no_count >= 2 and no_count > yes_count:
                health_clauses.append("I rarely get the flu vaccine")
            else:
                health_clauses.append("My flu vaccination is occasional rather than routine")
        if health_clauses:
            sentence = _join_sentence(health_clauses[0], health_clauses[1:])
            blocks.append(_apply_pronoun_context(sentence, ctx))

        trust_sentence = _summarize_trust_info()
        if trust_sentence:
            blocks.append(trust_sentence)

        vacc_sentence = _summarize_vaccine_stance()
        if vacc_sentence:
            blocks.append(vacc_sentence)

        stakes_sentence = _summarize_disease_stakes()
        if stakes_sentence:
            blocks.append(stakes_sentence)

        norms_sentence = _summarize_norms_moral()
        if norms_sentence:
            blocks.append(norms_sentence)

        missing = sorted(
            col
            for col in cols
            if col in attrs
            and not col.startswith(SPECIAL_STRUCTURAL_PREFIXES)
            and not _is_missing(attrs.get(col))
            and col not in narrative_cols
        )
        if missing:
            raise ValueError(f"Unmapped narrative attributes: {', '.join(missing)}")

        return "\n".join(blocks)

    builder = PromptTemplateBuilder(
        display_labels=display_labels,
        display_groups=display_groups,
        clean_questions=clean_questions,
        clean_answers=clean_answers,
        encodings=encodings,
        by_id=by_id,
        column_types=column_types,
        possible_bounds=possible_bounds,
        ordinal_endpoints=ordinal_endpoints,
    )

    _outcome_label = builder.outcome_label
    _outcome_semantics = builder.outcome_semantics
    _multi_outcome_semantics = builder.multi_outcome_semantics
    _psychometrician_system = builder.psychometrician_system
    _first_person_system_prompt = builder.first_person_system_prompt
    _prediction_user_prompt = builder.prediction_user_prompt
    _prediction_user_prompt_first_person = builder.prediction_user_prompt_first_person
    _prediction_user_prompt_multi = builder.prediction_user_prompt_multi
    _prediction_user_prompt_first_person_multi = builder.prediction_user_prompt_first_person_multi
    _maybe_survey_context = builder.maybe_survey_context
    _format_attr_lines = builder.format_attr_lines
    _build_narrative = builder.build_narrative

    def _outcome_metadata_payload(outcome_name: str | None) -> dict[str, object]:
        if not isinstance(outcome_name, str) or not outcome_name.strip():
            return {}
        payload: dict[str, object] = {"outcome_label": _outcome_label(outcome_name)}
        bounds = possible_bounds.get(outcome_name)
        if bounds is not None:
            payload["bounds"] = [float(bounds[0]), float(bounds[1])]
        endpoints = ordinal_endpoints.get(outcome_name)
        if endpoints is not None:
            payload["endpoints"] = [str(endpoints[0]), str(endpoints[1])]
        return payload

    prompts: list[dict] = []

    for row in rows:
        attrs = {k: v for k, v in row.items() if k not in {"attributes"}}
        if isinstance(row.get("attributes"), dict):
            attrs = row.get("attributes")

        driver_cols = _resolve_driver_cols(row)
        include_cols = list(include_cols_override)
        entry = _resolve_lever_entry(row)
        if isinstance(entry, dict):
            extra = entry.get("include_cols")
            if isinstance(extra, list):
                include_cols.extend([str(x) for x in extra])
        if include_cols:
            for col in include_cols:
                if col in row and col not in attrs:
                    attrs[col] = row.get(col)

        outcome = row.get("outcome")
        target = row.get("target")
        grid_point = row.get("grid_point")
        meta = {
            "target": target,
            "grid_point": grid_point,
            "outcome": outcome,
        }
        meta.update(_outcome_metadata_payload(outcome if isinstance(outcome, str) else None))
        for key in ["delta_x", "delta_z", "grid_value", "q25", "q50", "q75", "stratum_level", "stratum_key"]:
            if key in row:
                meta[key] = row.get(key)

        if args.strategy == "multi_expert":
            attr_lines = _format_attr_lines(attrs, driver_cols, scale_seen=set())
            human_lines = []
            ctx = _maybe_survey_context(attrs)
            if ctx:
                human_lines.append(ctx)
            human_lines.append("Attributes:")
            human_lines.extend(attr_lines)
            human_prompt = "\n".join(human_lines)
            prompts.append(
                {
                    "id": row.get("id"),
                    "strategy": "multi_expert",
                    "attributes": attrs,
                    "human_prompt": human_prompt,
                    "metadata": meta,
                    "outcome": outcome,
                }
            )
            continue

        if args.strategy == "paired_profile":
            attrs_low = row.get("attributes_low") or {}
            attrs_high = row.get("attributes_high") or {}
            if not isinstance(attrs_low, dict) or not isinstance(attrs_high, dict):
                raise ValueError("paired_profile requires attributes_low and attributes_high")

            scale_seen: set[tuple[object, ...]] = set()

            def _mk_profile_lines(
                attrs_block: dict,
                driver_cols=driver_cols,
                scale_seen=scale_seen,
            ) -> list[str]:
                return _format_attr_lines(attrs_block, driver_cols, scale_seen=scale_seen)

            # Randomize order for paired-profile bias checks.
            seed = row.get("seed") or 0
            rng = pd.util.hash_pandas_object(pd.Series([seed])).iloc[0]
            order = "A_then_B" if int(rng) % 2 == 0 else "B_then_A"

            if order == "A_then_B":
                a_lines = _mk_profile_lines(attrs_low)
                b_lines = _mk_profile_lines(attrs_high)
            else:
                a_lines = _mk_profile_lines(attrs_high)
                b_lines = _mk_profile_lines(attrs_low)

            def _split_common_and_diff(
                left: list[str],
                right: list[str],
            ) -> tuple[list[str], list[str], list[str]]:
                right_set = set(right)
                left_set = set(left)
                common = [line for line in left if line in right_set]
                left_only = [line for line in left if line not in right_set]
                right_only = [line for line in right if line not in left_set]
                return common, left_only, right_only

            def _reattach_group_headers(
                *,
                common: list[str],
                left_only: list[str],
                right_only: list[str],
                left: list[str],
                right: list[str],
            ) -> tuple[list[str], list[str], list[str]]:
                """Keep group headers with their children.

                `format_attr_lines` emits group headers like "- X:" followed by indented children.
                When paired profiles differ only in the child values, the header string is identical
                and ends up in `common`, leaving an empty-looking header in the Common section.

                If a group header has *no* common children, move the header into both diff blocks
                immediately before the first differing child line.

                If a group header has some common children but also some differing children, keep
                the header in Common (so common children stay grouped), but also include the header
                once in each diff block that has differing children, immediately before the first
                such child. This avoids orphaned child lines in the diff blocks.
                """

                def _child_to_header(lines: list[str]) -> dict[str, str]:
                    out: dict[str, str] = {}
                    current: str | None = None
                    for ln in lines:
                        s = str(ln)
                        if s.startswith("- ") and s.rstrip().endswith(":"):
                            current = s
                            continue
                        if s.startswith("  - ") and current is not None:
                            out[s] = current
                            continue
                        # Reset header when leaving the indented block (blank line or new top-level item).
                        if not s.startswith("  - "):
                            current = None
                    return out

                left_map = _child_to_header(left)
                right_map = _child_to_header(right)

                common_children_by_header: set[str] = set()
                for ln in common:
                    hdr = left_map.get(str(ln))
                    if hdr is not None:
                        common_children_by_header.add(hdr)

                common_out = list(common)
                left_out = list(left_only)
                right_out = list(right_only)

                for ln in list(common_out):
                    s = str(ln)
                    is_header = s.startswith("- ") and s.rstrip().endswith(":")
                    if not is_header:
                        continue
                    # Keep the header in Common if at least one child line is also common.
                    if s in common_children_by_header:
                        continue

                    left_child_idxs = [i for i, x in enumerate(left_out) if left_map.get(str(x)) == s]
                    right_child_idxs = [i for i, x in enumerate(right_out) if right_map.get(str(x)) == s]
                    if not left_child_idxs and not right_child_idxs:
                        continue

                    # Move the header out of Common.
                    common_out = [x for x in common_out if str(x) != s]

                    if left_child_idxs:
                        left_out.insert(min(left_child_idxs), s)
                    if right_child_idxs:
                        right_out.insert(min(right_child_idxs), s)

                # Also include headers in diff blocks when only some children differ.
                # Example:
                #   Common contains "- Trust ...:" + some children.
                #   Diff contains a child like "  - Institutions: 2".
                # Without duplication, the diff child looks context-free.

                def _insert_header_for_diff_children(
                    *,
                    diff_lines: list[str],
                    child_to_header: dict[str, str],
                    common_headers: set[str],
                ) -> list[str]:
                    out_lines = list(diff_lines)
                    # Find first index of each header's diff child.
                    first_idx: dict[str, int] = {}
                    for i, ln in enumerate(out_lines):
                        hdr = child_to_header.get(str(ln))
                        if hdr is None:
                            continue
                        if hdr not in common_headers:
                            continue
                        if hdr not in first_idx:
                            first_idx[hdr] = i

                    # Insert headers in a stable order (left-to-right).
                    inserted: set[str] = set()
                    offset = 0
                    for hdr, idx in sorted(first_idx.items(), key=lambda kv: kv[1]):
                        if hdr in inserted:
                            continue
                        if hdr in out_lines:
                            continue
                        out_lines.insert(idx + offset, hdr)
                        inserted.add(hdr)
                        offset += 1

                    return out_lines

                common_headers = {
                    str(ln) for ln in common_out if str(ln).startswith("- ") and str(ln).rstrip().endswith(":")
                }
                left_out = _insert_header_for_diff_children(
                    diff_lines=left_out,
                    child_to_header=left_map,
                    common_headers=common_headers,
                )
                right_out = _insert_header_for_diff_children(
                    diff_lines=right_out,
                    child_to_header=right_map,
                    common_headers=common_headers,
                )

                return common_out, left_out, right_out

            common_lines, a_only, b_only = _split_common_and_diff(a_lines, b_lines)
            common_lines, a_only, b_only = _reattach_group_headers(
                common=common_lines,
                left_only=a_only,
                right_only=b_only,
                left=a_lines,
                right=b_lines,
            )

            semantics, _ = _outcome_semantics(outcome)
            system_prompt = (
                "You are an expert survey psychometrician acting as a prediction function.\n\n"
                "Predict the expected self-reported survey response for people in this dataset/country, "
                "not what is socially desirable or recommended.\n\n"
                f"{semantics}\n\n"
                "Do not include reasoning or commentary."
            )
            human_lines: list[str] = []
            ctx = _maybe_survey_context(attrs_low) or _maybe_survey_context(attrs_high)
            if ctx:
                human_lines.append(ctx)
            if common_lines:
                human_lines.append("Common attributes:")
                human_lines.extend(common_lines)
                human_lines.append("")
            human_lines.append("Profile A (differences):")
            if a_only:
                human_lines.extend(a_only)
            else:
                human_lines.append("- None")
            human_lines.append("")
            human_lines.append("Profile B (differences):")
            if b_only:
                human_lines.extend(b_only)
            else:
                human_lines.append("- None")
            human_lines.append("")
            human_lines.append("Predict the outcome for Profile A and for Profile B separately.")
            human_lines.append('Output format: {"A": <number>, "B": <number>}')

            meta.update(
                {
                    "paired_order": order,
                    "metadata_low": row.get("metadata_low"),
                    "metadata_high": row.get("metadata_high"),
                }
            )

            prompts.append(
                {
                    "id": row.get("id"),
                    "strategy": "paired_profile",
                    "system_prompt": system_prompt,
                    "human_prompt": "\n".join(human_lines),
                    "metadata": meta,
                    "outcome": outcome,
                }
            )
            continue

        # For remaining strategies, build a shared attribute block.
        attr_lines = _format_attr_lines(attrs, driver_cols, scale_seen=set())

        system_prompt = _psychometrician_system(outcome).strip()

        if args.strategy in {"first_person", "third_person"}:
            q25 = row.get("q25")
            q75 = row.get("q75")
            other_value = None
            if str(grid_point) == "low":
                other_value = q75
            elif str(grid_point) == "high":
                other_value = q25

            narrative_human = _build_narrative(
                attrs,
                driver_cols,
                args.strategy,
                perturbed_target=str(target) if target is not None else None,
                perturbed_grid_point=str(grid_point) if grid_point is not None else None,
                perturbed_other_value=other_value,
            )
            if args.strategy == "first_person":
                narrative_assistant = narrative_human
                narrative_system = _first_person_system_prompt()
                narrative_user = _prediction_user_prompt_first_person(outcome)
            else:
                narrative_assistant = None
                narrative_system = None
                narrative_user = None

            predict_system = "" if args.strategy == "first_person" else system_prompt
            payload = {
                "id": row.get("id"),
                "strategy": args.strategy,
                "narrative_human_prompt": narrative_human,
                "predict_system_prompt": predict_system,
                "metadata": meta,
                "outcome": outcome,
            }

            if args.strategy == "first_person":
                payload.update(
                    {
                        "narrative_system_prompt": narrative_system,
                        "narrative_assistant_prompt": narrative_assistant,
                        "narrative_user_prompt": narrative_user,
                    }
                )
            else:
                payload.update(
                    {
                        "system_prompt": system_prompt,
                        "human_prompt": narrative_human,
                    }
                )

            prompts.append(payload)
            continue

        # Default data_only
        human_lines = []
        ctx = _maybe_survey_context(attrs)
        if ctx:
            human_lines.append(ctx)
        human_lines.append("Attributes:")
        human_lines.extend(attr_lines)
        human_lines.append("")
        prompts.append(
            {
                "id": row.get("id"),
                "strategy": args.strategy,
                "system_prompt": (
                    f"{system_prompt}\n\n"
                    "Fields not listed are unknown; treat them as unknown (do not assume typical values)."
                ),
                "human_prompt": "\n".join(human_lines),
                "metadata": meta,
                "outcome": outcome,
                "attributes": attrs,
            }
        )

    if args.strategy in {"first_person", "third_person"}:
        # Guardrail: ensure low/high variants never collapse to identical narratives.
        # This catches any missed lever-specific override mapping.
        seen: dict[tuple[str, str, str, str], dict[str, str]] = {}
        collisions: list[tuple[str, str, str, str]] = []
        for r in prompts:
            if str(r.get("strategy")) not in {"first_person", "third_person"}:
                continue
            meta = r.get("metadata") if isinstance(r.get("metadata"), dict) else {}
            target = str(meta.get("target")) if meta.get("target") is not None else ""
            grid_point = str(meta.get("grid_point")) if meta.get("grid_point") is not None else ""
            if grid_point not in {"low", "high"}:
                continue

            rid = str(r.get("id"))
            outcome = str(r.get("outcome"))
            strategy = str(r.get("strategy"))
            key = (rid, target, outcome, strategy)

            if strategy == "first_person":
                narrative = str(r.get("narrative_human_prompt") or "")
            else:
                narrative = str(r.get("human_prompt") or "")

            bucket = seen.setdefault(key, {})
            bucket[grid_point] = narrative
            if "low" in bucket and "high" in bucket and bucket["low"] == bucket["high"]:
                collisions.append(key)

        if collisions:
            sample = "; ".join([f"id={k[0]} target={k[1]} outcome={k[2]} strategy={k[3]}" for k in collisions[:10]])
            raise ValueError(
                "Prompt collision detected: low/high narratives are identical for one or more records. "
                f"Examples: {sample}"
            )

    if args.dedup_levers:
        # Deduplicate by (id, lever, grid_point) and request multiple outcomes in one prompt.
        grouped: dict[tuple[str, str, str], dict] = {}
        for r in prompts:
            meta = r.get("metadata") if isinstance(r.get("metadata"), dict) else {}
            key = (str(r.get("id")), str(meta.get("target")), str(meta.get("grid_point")))
            entry = grouped.get(key)
            if entry is None:
                entry = r
                entry["outcomes"] = [r.get("outcome")]
                grouped[key] = entry
            else:
                entry_outcomes = entry.get("outcomes")
                if not isinstance(entry_outcomes, list):
                    entry_outcomes = []
                entry_outcomes.append(r.get("outcome"))
                entry["outcomes"] = entry_outcomes

        prompts = list(grouped.values())

        has_multi_outcome = any(
            isinstance(r.get("outcomes"), list) and len(r.get("outcomes") or []) > 1 for r in prompts
        )
        if has_multi_outcome and args.strategy == "paired_profile" and not args.paired_profile_multi_outcome:
            raise ValueError(
                "Multi-outcome prompts are not supported with paired_profile unless "
                "--paired-profile-multi-outcome is set."
            )

        for r in prompts:
            outcomes = r.get("outcomes")
            if not isinstance(outcomes, list) or len(outcomes) <= 1:
                continue
            outcomes = [str(o) for o in outcomes]
            meta = r.get("metadata") if isinstance(r.get("metadata"), dict) else {}
            meta["outcomes"] = outcomes
            meta["multi_outcome"] = True
            labels_by_outcome: dict[str, str] = {}
            bounds_by_outcome: dict[str, list[float]] = {}
            endpoints_by_outcome: dict[str, list[str]] = {}
            for outcome_name in outcomes:
                payload = _outcome_metadata_payload(outcome_name)
                label = payload.get("outcome_label")
                bounds = payload.get("bounds")
                endpoints = payload.get("endpoints")
                if isinstance(label, str):
                    labels_by_outcome[outcome_name] = label
                if isinstance(bounds, list) and len(bounds) == 2:
                    bounds_by_outcome[outcome_name] = [float(bounds[0]), float(bounds[1])]
                if isinstance(endpoints, list) and len(endpoints) == 2:
                    endpoints_by_outcome[outcome_name] = [str(endpoints[0]), str(endpoints[1])]
            if labels_by_outcome:
                meta["outcome_labels_by_outcome"] = labels_by_outcome
            if bounds_by_outcome:
                meta["bounds_by_outcome"] = bounds_by_outcome
            if endpoints_by_outcome:
                meta["endpoints_by_outcome"] = endpoints_by_outcome
            r["metadata"] = meta

            semantics, schema = _multi_outcome_semantics(outcomes)
            if str(r.get("strategy")) == "paired_profile":
                paired_schema = (
                    "Return ONLY JSON with keys: A, B. "
                    "Each value must be a JSON object with keys: " + ", ".join(outcomes) + "."
                )
                system_prompt = (
                    "You are an expert survey psychometrician acting as a prediction function.\n\n"
                    "Predict the expected self-reported survey response for people in this dataset/country, "
                    "not what is socially desirable or recommended.\n\n"
                    f"{semantics}\n\n"
                    "Do not include reasoning or commentary.\n"
                    f"{paired_schema}"
                )

                r["system_prompt"] = system_prompt

                human_prompt = r.get("human_prompt")
                if isinstance(human_prompt, str):
                    lines = [ln.rstrip() for ln in human_prompt.splitlines()]
                    while lines and not lines[-1].strip():
                        lines.pop()
                    while lines and (
                        lines[-1].strip().startswith("Output format:")
                        or lines[-1].strip().startswith("Predict the outcome")
                    ):
                        lines.pop()
                    lines.append("Predict each outcome for Profile A and for Profile B separately.")
                    lines.append('Output format: {"A": {<outcome>: <number>, ...}, "B": {<outcome>: <number>, ...}}')
                    r["human_prompt"] = "\n".join(lines)
            else:
                system_prompt = (
                    "You are an expert survey psychometrician acting as a prediction function.\n\n"
                    "Predict the expected self-reported survey response for people in this dataset/country, "
                    "not what is socially desirable or recommended.\n\n"
                    f"{semantics}\n\n"
                    "Do not include reasoning or commentary.\n"
                    f"{schema}"
                )

            if r.get("strategy") in {"first_person", "third_person"}:
                if r.get("strategy") == "first_person":
                    r["narrative_user_prompt"] = _prediction_user_prompt_first_person_multi(outcomes)
                else:
                    r["system_prompt"] = system_prompt
                    r["predict_system_prompt"] = system_prompt
                    narrative_human = r.get("narrative_human_prompt")
                    if isinstance(narrative_human, str):
                        r["human_prompt"] = narrative_human
                continue

            if r.get("strategy") == "data_only":
                attrs = r.get("attributes") if isinstance(r.get("attributes"), dict) else {}
                attr_lines = _format_attr_lines(attrs, _resolve_driver_cols(r), scale_seen=set())
                human_lines = []
                ctx = _maybe_survey_context(attrs)
                if ctx:
                    human_lines.append(ctx)
                human_lines.append("Attributes:")
                human_lines.extend(attr_lines)
                human_lines.append("")
                r["system_prompt"] = system_prompt
                r["human_prompt"] = "\n".join(human_lines)

    if args.shuffle:
        import random as _random

        _random.Random(int(args.seed)).shuffle(prompts)
    if args.limit is not None:
        prompts = prompts[: max(0, int(args.limit))]

    _write_jsonl(Path(args.out_path), prompts)


if __name__ == "__main__":
    main()
