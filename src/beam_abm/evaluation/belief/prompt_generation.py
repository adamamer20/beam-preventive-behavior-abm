"""Generate prompts for belief-update micro-validation (Phase 0 prompt tournament)."""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from beam_abm.empirical.plan_io import extract_predictors_from_model_plan
from beam_abm.evaluation.belief.reference import (
    default_ref_state_models,
)
from beam_abm.evaluation.common.labels import (
    load_clean_spec_question_answer_maps,
    load_display_groups,
    load_display_labels,
    load_ordinal_endpoints_from_clean_spec,
    load_possible_numeric_bounds_from_clean_spec,
)
from beam_abm.evaluation.prompting import PromptTemplateBuilder, build_clean_spec_index


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _read_rows(path: Path) -> list[dict]:
    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
        return df.to_dict(orient="records")
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
        return df.to_dict(orient="records")
    return _read_jsonl(path)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _safe_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        val = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(val) or math.isinf(val):
        return None
    return val


def _is_missing(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    if isinstance(value, str) and value.strip().lower() in {"missing", "nan", "none"}:
        return True
    return False


def _parse_list(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]


def _unique_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _resolve_context_predictors(
    *,
    spec: dict[str, object],
    mutable_state: list[str],
    model_plan_path: Path,
) -> tuple[list[str], dict[str, str], dict[str, list[str]]]:
    ref_map_raw = spec.get("ref_state_models")
    ref_map = ref_map_raw if isinstance(ref_map_raw, dict) else {}
    if not ref_map:
        ref_map = default_ref_state_models()

    resolved_models: dict[str, str] = {}
    unresolved_models: dict[str, list[str]] = {}
    predictors: list[str] = []
    for state_var in mutable_state:
        requested_raw = ref_map.get(state_var)
        requested = str(requested_raw).strip() if requested_raw is not None else ""
        if not requested:
            unresolved_models[str(state_var)] = []
            continue

        try:
            found_predictors = extract_predictors_from_model_plan(
                model_plan_path=model_plan_path,
                model_name=requested,
            )
        except (FileNotFoundError, KeyError, ValueError):
            unresolved_models[str(state_var)] = [requested]
            continue
        resolved_models[str(state_var)] = requested
        predictors.extend(found_predictors)

    return _unique_preserve_order(predictors), resolved_models, unresolved_models


def _read_pe_ref_p_table(path: Path) -> pd.DataFrame:
    suf = path.suffix.lower()
    if suf == ".parquet":
        return pd.read_parquet(path)
    if suf == ".csv":
        return pd.read_csv(path)
    if suf == ".jsonl":
        return pd.DataFrame(_read_jsonl(path))
    raise ValueError(f"Unsupported PE_ref_P table format: {path}")


def _derive_signal_meta_from_pe(
    pe_df: pd.DataFrame,
    *,
    alpha_secondary: float,
    epsilon_ref: float,
    top_n_primary: int,
) -> dict[str, dict[str, object]]:
    if pe_df.empty:
        return {}

    grouped = pe_df.groupby(["signal_id", "state_var"], dropna=False)
    med_raw = (
        grouped["pe_ref_p"]
        .apply(lambda s: float(pd.to_numeric(s, errors="coerce").abs().median()))
        .reset_index(name="median_abs_pe_ref_p")
    )
    med_std = (
        grouped["pe_ref_p_std"]
        .apply(lambda s: float(pd.to_numeric(s, errors="coerce").abs().median()))
        .reset_index(name="median_abs_pe_ref_p_std")
    )
    med_df = med_raw.merge(med_std, on=["signal_id", "state_var"], how="inner")

    out: dict[str, dict[str, object]] = {}
    for signal_id, sub in med_df.groupby("signal_id", dropna=False):
        by_var_abs = {str(r["state_var"]): float(r["median_abs_pe_ref_p"]) for _, r in sub.iterrows()}
        by_var_abs_std = {str(r["state_var"]): float(r["median_abs_pe_ref_p_std"]) for _, r in sub.iterrows()}
        ranked = sorted(by_var_abs.items(), key=lambda kv: (-kv[1], kv[0]))
        max_val = ranked[0][1] if ranked else 0.0
        top_n = max(1, int(top_n_primary))
        primary_vars = {v for v, _ in ranked[:top_n]}

        tiers: dict[str, str] = {}
        for var, value in ranked:
            std_val = by_var_abs_std.get(var, float("nan"))
            if math.isfinite(std_val) and std_val < float(epsilon_ref):
                tiers[var] = "none"
            elif var in primary_vars:
                tiers[var] = "primary"
            elif max_val > 0.0 and value >= float(alpha_secondary) * float(max_val):
                tiers[var] = "secondary"
            else:
                tiers[var] = "weak"

        ranking = [
            {
                "var": var,
                "rank": idx + 1,
                "median_abs_pe_ref_p": float(value),
                "median_abs_pe_ref_p_std": float(by_var_abs_std.get(var, float("nan"))),
                "tier": tiers.get(var, "none"),
            }
            for idx, (var, value) in enumerate(ranked)
        ]
        out[str(signal_id)] = {
            "median_abs_pe_ref_p_by_var": by_var_abs,
            "median_abs_pe_ref_p_std_by_var": by_var_abs_std,
            "tier_by_var": tiers,
            "ranking": ranking,
            "alpha_secondary": float(alpha_secondary),
            "epsilon_ref": float(epsilon_ref),
            "top_n_primary": int(top_n),
        }
    return out


def _build_row_signal_from_precomputed(
    pe_df: pd.DataFrame,
    *,
    signal_text_lookup: dict[str, str],
    alpha_secondary: float,
    epsilon_ref: float,
    top_n_primary: int,
) -> tuple[dict[tuple[str, str], dict[str, object]], dict[str, dict[str, object]]]:
    if "id" not in pe_df.columns:
        for alt in ("row_id", "base_id"):
            if alt in pe_df.columns:
                pe_df = pe_df.rename(columns={alt: "id"})
                break
    if "signal_id" not in pe_df.columns:
        for alt in ("signal", "signal_name"):
            if alt in pe_df.columns:
                pe_df = pe_df.rename(columns={alt: "signal_id"})
                break
    required = {"id", "signal_id", "state_var", "pe_ref_p"}
    missing = sorted(required - set(pe_df.columns))
    if missing:
        raise ValueError(f"PE_ref_P table missing required columns: {missing}")

    work = pe_df.copy()
    work["id"] = work["id"].astype(str)
    work["signal_id"] = work["signal_id"].astype(str)
    work["state_var"] = work["state_var"].astype(str)
    if "pe_ref_p_std" not in work.columns:
        work["pe_ref_p_std"] = float("nan")

    signal_meta = _derive_signal_meta_from_pe(
        work,
        alpha_secondary=float(alpha_secondary),
        epsilon_ref=float(epsilon_ref),
        top_n_primary=int(top_n_primary),
    )

    row_signal: dict[tuple[str, str], dict[str, object]] = {}
    for (row_id, signal_id), sub in work.groupby(["id", "signal_id"], dropna=False):
        by_var = {str(r["state_var"]): float(r["pe_ref_p"]) for _, r in sub.iterrows()}
        by_var_std: dict[str, float] = {}
        for _, r in sub.iterrows():
            std_val = _safe_float(r.get("pe_ref_p_std"))
            if std_val is not None:
                by_var_std[str(r["state_var"])] = float(std_val)

        first = sub.iloc[0].to_dict()
        tier_by_var = signal_meta.get(str(signal_id), {}).get("tier_by_var", {})
        ranking = signal_meta.get(str(signal_id), {}).get("ranking", [])
        signal_text = str(first.get("signal_text") or signal_text_lookup.get(str(signal_id), "")).strip()

        shock_var = str(first.get("shock_var") or "").strip()
        shock_type = str(first.get("shock_type") or "").strip()
        shock_direction = str(first.get("shock_direction") or "").strip()
        shock_from_q = _safe_float(first.get("shock_from_quantile"))
        shock_to_q = _safe_float(first.get("shock_to_quantile"))
        shock_low = _safe_float(first.get("shock_low_value"))
        shock_high = _safe_float(first.get("shock_high_value"))

        shock_payload: dict[str, object] | None = None
        if shock_var:
            shock_payload = {
                "var": shock_var,
                "type": shock_type or "continuous",
                "direction": shock_direction or "increase",
            }
            if shock_low is not None:
                shock_payload["from"] = shock_low
            if shock_high is not None:
                shock_payload["to"] = shock_high
            if shock_from_q is not None:
                shock_payload["from_quantile"] = shock_from_q
            if shock_to_q is not None:
                shock_payload["to_quantile"] = shock_to_q

        row_signal[(str(row_id), str(signal_id))] = {
            "signal_id": str(signal_id),
            "signal_text": signal_text,
            "pe_ref_p": by_var,
            "pe_ref_p_std": by_var_std,
            "pe_ref_p_tier_by_var": tier_by_var,
            "pe_ref_p_ranking": ranking,
            "pe_ref_p_delta_z": _safe_float(first.get("delta_z")),
            "pe_ref_p_delta_x": _safe_float(first.get("delta_x")),
            "shock": shock_payload,
            "shocks": [shock_payload] if shock_payload is not None else [],
            "shock_var": shock_var if shock_var else None,
            "shock_base_value": _safe_float(first.get("shock_base_value")),
            "shock_low_value": shock_low,
            "shock_high_value": shock_high,
            "shock_perturbed_value": shock_high,
        }

    return row_signal, signal_meta


def _stratified_sample_rows(
    rows: list[dict],
    *,
    profile_col: str,
    per_profile: int,
    rng: random.Random,
) -> list[dict]:
    groups: dict[str, list[dict]] = {}
    for row in rows:
        profile = row.get(profile_col)
        profile_key = str(profile) if profile is not None else "__missing_profile__"
        groups.setdefault(profile_key, []).append(row)

    sampled_rows: list[dict] = []
    for profile_key in sorted(groups):
        bucket = groups[profile_key]
        if not bucket:
            continue
        if len(bucket) >= per_profile:
            sampled_rows.extend(rng.sample(bucket, per_profile))
            continue

        sampled_rows.extend(bucket)
        needed = per_profile - len(bucket)
        sampled_rows.extend(rng.choices(bucket, k=needed))

    return sampled_rows


def _sample_n_per_anchor_country(
    rows: list[dict],
    *,
    anchor_col: str,
    country_col: str,
    n_per_anchor_country: int,
    rng: random.Random,
) -> list[dict]:
    if n_per_anchor_country <= 0:
        return rows
    buckets: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        anchor = row.get(anchor_col)
        country = row.get(country_col)
        if anchor is None or country is None:
            continue
        buckets.setdefault((str(anchor), str(country)), []).append(row)

    sampled: list[dict] = []
    for key in sorted(buckets.keys()):
        group = buckets[key]
        if len(group) >= n_per_anchor_country:
            sampled.extend(rng.sample(group, n_per_anchor_country))
            continue
        sampled.extend(group)
        needed = n_per_anchor_country - len(group)
        sampled.extend(rng.choices(group, k=needed))
    return sampled


def _extract_signals(spec: dict[str, object]) -> list[dict[str, object]]:
    direct = spec.get("signals")
    if isinstance(direct, list):
        return [entry for entry in direct if isinstance(entry, dict)]
    raise ValueError("Belief-update spec must define `signals` as a list.")


def _belief_update_system_data_only(
    *,
    semantics: str,
    mutable_state: list[str],
) -> str:
    bullets = "\n".join(f"* {line.lstrip('- ').strip()}" for line in semantics.splitlines() if line.startswith("- "))
    key_list = ", ".join(mutable_state)
    return (
        "You are an expert survey psychometrician acting as a prediction function.\n\n"
        "Predict the expected self-reported values of the mutable state variables below from the provided profile "
        "predictors.\n\n"
        "Mutable state variables and scales:\n\n"
        f"{bullets}\n\n"
        "Output rules:\n\n"
        "* Return ONLY valid JSON, with keys from the mutable state.\n"
        f"* Allowed keys (exact): {key_list}.\n"
        "* Use floats for all values.\n"
        "* Include ALL mutable state keys in every response.\n"
        "* Do not include any other keys, reasoning, or commentary.\n"
        "* Fields not provided are unknown; do not assume values.\n"
    )


def _belief_update_task(mutable_state: list[str]) -> str:
    vars_text = ", ".join(mutable_state)
    json_shape = "{\n" + ",\n".join([f'  "{k}": <float>' for k in mutable_state]) + "\n}"
    return (
        "Task: Return ONLY JSON with keys exactly equal to the mutable_state variable names. "
        "Use floats and include all keys.\n"
        f"mutable_state variables: {vars_text}\n"
        f"Output format:\n{json_shape}"
    )


def _format_section(title: str, lines: list[str]) -> list[str]:
    out = [f"{title}:"]
    if lines:
        out.extend(lines)
    else:
        out.append("- None")
    return out


def _build_narrative_safe(
    builder: PromptTemplateBuilder,
    attrs: dict[str, object],
    driver_cols: list[str],
    strategy: str,
    dropped_cols_sink: set[str] | None = None,
) -> str:
    safe_attrs = dict(attrs)
    for _ in range(3):
        try:
            return builder.build_narrative(safe_attrs, driver_cols, strategy)
        except ValueError as exc:
            msg = str(exc)
            if "Unmapped narrative attributes:" not in msg:
                raise
            missing = msg.split("Unmapped narrative attributes:", 1)[-1].strip()
            missing_cols = [m.strip() for m in missing.split(",") if m.strip()]
            if not missing_cols:
                raise
            if dropped_cols_sink is not None:
                dropped_cols_sink.update(missing_cols)
            for col in missing_cols:
                safe_attrs.pop(col, None)
    return builder.build_narrative(safe_attrs, driver_cols, strategy)


def generate_belief_prompts(
    *,
    data: str | Path,
    spec: str | Path,
    out: str | Path,
    pe_ref_p: str | Path | None = None,
    run_type: str = "perturbed",
    limit: int | None = None,
    shuffle: bool = False,
    seed: int = 0,
    profile_col: str = "anchor_id",
    per_profile: int = 100,
    anchor_col: str = "anchor_id",
    keep_anchors: str | None = None,
    n_per_anchor_country: int = 0,
    country_col: str = "country",
    id_col: str = "row_id",
    prompt_families: str = "data_only,first_person,third_person,multi_expert",
    display_labels: str | Path = "thesis/specs/display_labels.json",
    clean_spec: str | Path = "preprocess/specs/5_clean_transformed_questions.json",
    column_types: str | Path = "empirical/specs/column_types.tsv",
    model_plan: str | Path = "empirical/output/modeling/model_plan.json",
    alpha_secondary: float = 0.33,
    epsilon_ref: float = 0.03,
    top_n_primary: int = 1,
) -> None:
    args = SimpleNamespace(
        data=str(data),
        pe_ref_p=str(pe_ref_p) if pe_ref_p is not None else None,
        run_type=str(run_type),
        spec=str(spec),
        out=str(out),
        limit=limit,
        shuffle=bool(shuffle),
        seed=int(seed),
        profile_col=str(profile_col),
        per_profile=int(per_profile),
        anchor_col=str(anchor_col),
        keep_anchors=keep_anchors,
        n_per_anchor_country=int(n_per_anchor_country),
        country_col=str(country_col),
        id_col=str(id_col),
        prompt_families=str(prompt_families),
        display_labels=str(display_labels),
        clean_spec=str(clean_spec),
        column_types=str(column_types),
        model_plan=str(model_plan),
        alpha_secondary=float(alpha_secondary),
        epsilon_ref=float(epsilon_ref),
        top_n_primary=int(top_n_primary),
    )
    rng = random.Random(int(args.seed))

    rows = list(_read_rows(Path(args.data)))

    keep_anchors = set(_parse_list(args.keep_anchors))
    if keep_anchors:
        rows = [r for r in rows if str(r.get(args.anchor_col)) in keep_anchors]

    if int(args.n_per_anchor_country) > 0:
        rows = _sample_n_per_anchor_country(
            rows,
            anchor_col=str(args.anchor_col),
            country_col=str(args.country_col),
            n_per_anchor_country=int(args.n_per_anchor_country),
            rng=rng,
        )

    if args.per_profile > 0 and args.profile_col:
        rows = _stratified_sample_rows(
            rows,
            profile_col=str(args.profile_col),
            per_profile=int(args.per_profile),
            rng=rng,
        )
    if args.shuffle:
        rng.shuffle(rows)
    if args.limit is not None:
        rows = rows[: max(0, int(args.limit))]

    with Path(args.spec).open("r", encoding="utf-8") as f:
        spec = json.load(f)

    always_on_moderators = list(spec.get("always_on_moderators", []) or [])
    mutable_state = list(spec.get("mutable_state", []) or [])
    signals = _extract_signals(spec if isinstance(spec, dict) else {})
    signal_text_by_id: dict[str, str] = {}
    for signal in signals:
        if not isinstance(signal, dict):
            continue
        signal_id = str(signal.get("id") or "").strip()
        if not signal_id:
            continue
        signal_text_by_id[signal_id] = str(signal.get("text") or "").strip()
    shock_defaults = spec.get("shock_defaults") if isinstance(spec, dict) else {}
    if not isinstance(shock_defaults, dict):
        shock_defaults = {}
    alpha_secondary = float(shock_defaults.get("alpha_secondary", args.alpha_secondary))
    epsilon_ref = float(shock_defaults.get("epsilon_ref", args.epsilon_ref))
    top_n_primary = int(shock_defaults.get("top_n_primary", args.top_n_primary))

    prompt_families = set(_parse_list(args.prompt_families))
    run_type = str(args.run_type).strip().lower()
    if run_type == "unperturbed" and "paired_profile" in prompt_families:
        raise ValueError("paired_profile is not supported for belief-update unperturbed prompts.")
    if run_type == "perturbed" and not args.pe_ref_p:
        raise ValueError("--pe-ref-p is required when --run-type=perturbed.")

    display_labels = load_display_labels(Path(args.display_labels))
    display_groups = load_display_groups(Path(args.display_labels))
    clean_questions, clean_answers = load_clean_spec_question_answer_maps(Path(args.clean_spec))
    encodings, by_id = build_clean_spec_index(json.loads(Path(args.clean_spec).read_text(encoding="utf-8")))
    column_types = pd.read_csv(Path(args.column_types), sep="\t").set_index("column")["type"].to_dict()
    possible_bounds = load_possible_numeric_bounds_from_clean_spec(Path(args.clean_spec))
    ordinal_endpoints = load_ordinal_endpoints_from_clean_spec(Path(args.clean_spec))

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

    prompts: list[dict] = []
    missing_columns: set[str] = set()
    narrative_dropped_cols: set[str] = set()
    row_signal_map: dict[tuple[str, str], dict[str, object]] = {}
    signal_meta_map: dict[str, dict[str, object]] = {}
    pe_ref_p_source: str | None = None
    if run_type == "perturbed":
        pe_ref_p_df = _read_pe_ref_p_table(Path(str(args.pe_ref_p)))
        row_signal_map, signal_meta_map = _build_row_signal_from_precomputed(
            pe_ref_p_df,
            signal_text_lookup=signal_text_by_id,
            alpha_secondary=alpha_secondary,
            epsilon_ref=epsilon_ref,
            top_n_primary=top_n_primary,
        )
        pe_ref_p_source = str(args.pe_ref_p)
    by_id_signal: dict[str, list[dict[str, object]]] = {}
    for (row_id, signal_id), payload in row_signal_map.items():
        enriched = dict(payload)
        enriched["signal_id"] = str(signal_id)
        by_id_signal.setdefault(str(row_id), []).append(enriched)

    outcome_semantics, _ = builder.multi_outcome_semantics(mutable_state)
    system_prompt = _belief_update_system_data_only(
        semantics=outcome_semantics,
        mutable_state=mutable_state,
    )
    ref_context_cols, resolved_ref_models, unresolved_ref_models = _resolve_context_predictors(
        spec=spec if isinstance(spec, dict) else {},
        mutable_state=[str(x) for x in mutable_state if str(x).strip()],
        model_plan_path=Path(args.model_plan),
    )
    if unresolved_ref_models:
        print(f"Warning: unresolved ref-model predictors for mutable states: {unresolved_ref_models}")

    rows_without_pe = 0
    for i, row in enumerate(rows):
        base_id = row.get("id") or row.get(args.id_col, i)
        base_id = str(base_id)

        base_country = row.get(args.country_col)
        baseline_state = {k: _safe_float(row.get(k)) for k in mutable_state if k in row}

        moderators = {k: row.get(k) for k in always_on_moderators if k in row}

        if run_type == "perturbed":
            row_entries = by_id_signal.get(base_id, [])
            if not row_entries:
                rows_without_pe += 1
                continue
        else:
            row_entries = [
                {
                    "signal_id": "__unperturbed__",
                    "signal_text": "No additional signal; baseline predictors only.",
                    "pe_ref_p": {},
                    "pe_ref_p_std": {},
                    "pe_ref_p_tier_by_var": {},
                    "pe_ref_p_ranking": [],
                    "pe_ref_p_delta_z": None,
                    "pe_ref_p_delta_x": None,
                    "shocks": [],
                }
            ]

        for row_signal_payload in row_entries:
            signal_id = str(row_signal_payload.get("signal_id") or "").strip()
            if not signal_id:
                continue
            signal_text = str(row_signal_payload.get("signal_text") or signal_text_by_id.get(signal_id, "")).strip()
            signal_ref_meta = signal_meta_map.get(str(signal_id), {})

            shock_payload = (
                row_signal_payload.get("shock") if isinstance(row_signal_payload.get("shock"), dict) else None
            )
            shock_vars: list[str] = []
            if shock_payload is not None:
                shock_var = str(shock_payload.get("var") or "").strip()
                if shock_var:
                    shock_vars.append(shock_var)
            for shock in (
                row_signal_payload.get("shocks", []) if isinstance(row_signal_payload.get("shocks"), list) else []
            ):
                if not isinstance(shock, dict):
                    continue
                shock_var = str(shock.get("var") or "").strip()
                if shock_var:
                    shock_vars.append(shock_var)
            shock_vars = _unique_preserve_order(shock_vars)

            low_row = dict(row)
            high_row = dict(row)
            if shock_payload is not None:
                shock_var = str(shock_payload.get("var") or "").strip()
                shock_low = _safe_float(row_signal_payload.get("shock_low_value"))
                shock_high = _safe_float(row_signal_payload.get("shock_high_value"))
                if shock_var and shock_low is not None:
                    low_row[shock_var] = shock_low
                if shock_var and shock_high is not None:
                    high_row[shock_var] = shock_high
            perturbed_row = high_row

            context_cols = list(dict.fromkeys([*ref_context_cols, *shock_vars]))
            mutable_cols = list(mutable_state)
            allowed_cols = list(dict.fromkeys([*context_cols, *mutable_cols]))

            context_attrs = {
                k: perturbed_row.get(k)
                for k in context_cols
                if k in perturbed_row and not _is_missing(perturbed_row.get(k))
            }
            mutable_attrs = {k: row.get(k) for k in mutable_cols if k in row and not _is_missing(row.get(k))}

            for col in allowed_cols:
                if col not in perturbed_row:
                    missing_columns.add(col)

            context_lines = builder.format_attr_lines(context_attrs, context_cols, scale_seen=set())
            survey_ctx = None
            if base_country:
                survey_ctx = f"Survey context: Country={base_country}, Year=2024"
            meta = {
                "base_id": base_id,
                "row_index": i,
                "country": base_country,
                "run_type": run_type,
                "signal_id": signal_id,
                "signal_text": signal_text,
                "baseline_state": baseline_state,
                "always_on_moderators": moderators,
                "prompt_context_cols": context_cols,
                "prompt_context_mode": (
                    "ref_predictors_perturbed_from_precomputed_pe"
                    if run_type == "perturbed"
                    else "ref_predictors_observed_baseline"
                ),
                "ref_predictor_models": resolved_ref_models,
                "mutable_state": mutable_state,
                "shock": shock_payload,
                "shocks": row_signal_payload.get("shocks", []),
                "pe_ref_p": row_signal_payload.get("pe_ref_p", {}),
                "pe_ref_p_std": row_signal_payload.get("pe_ref_p_std", {}),
                "pe_ref_p_tier_by_var": row_signal_payload.get("pe_ref_p_tier_by_var", {}),
                "pe_ref_p_ranking": row_signal_payload.get("pe_ref_p_ranking", []),
                "pe_ref_p_delta_z": row_signal_payload.get("pe_ref_p_delta_z"),
                "pe_ref_p_delta_x": row_signal_payload.get("pe_ref_p_delta_x"),
                "pe_ref_p_signal_meta": signal_ref_meta,
                "pe_ref_p_source": pe_ref_p_source,
            }

            if "paired_profile" in prompt_families:
                attrs_a = {k: low_row.get(k) for k in context_cols if k in low_row and not _is_missing(low_row.get(k))}
                attrs_b = {
                    k: high_row.get(k) for k in context_cols if k in high_row and not _is_missing(high_row.get(k))
                }
                a_lines = builder.format_attr_lines(attrs_a, context_cols, scale_seen=set())
                b_lines = builder.format_attr_lines(attrs_b, context_cols, scale_seen=set())
                paired_lines = []
                if survey_ctx:
                    paired_lines.append(survey_ctx)
                    paired_lines.append("")
                paired_lines.extend(
                    [
                        "Profile A (shock low predictors):",
                        *a_lines,
                        "",
                        "Profile B (shock high predictors):",
                        *b_lines,
                        "",
                        "Predict mutable state for Profile A and Profile B separately.",
                        'Output format: {"A": {<mutable_state_key>: <float>, ...}, "B": {<mutable_state_key>: <float>, ...}}',
                        "",
                        _belief_update_task(mutable_state),
                    ]
                )
                paired_meta = dict(meta)
                paired_meta["paired_order"] = "A_low_B_high"
                prompts.append(
                    {
                        "id": base_id,
                        "strategy": "paired_profile",
                        "system_prompt": system_prompt,
                        "human_prompt": "\n".join(paired_lines),
                        "metadata": paired_meta,
                    }
                )

            if "multi_expert" in prompt_families:
                human_lines = []
                if survey_ctx:
                    human_lines.append(survey_ctx)
                    human_lines.append("")
                human_lines.extend(_format_section("Predictors", context_lines))
                human_lines.append("")
                human_lines.append(_belief_update_task(mutable_state))
                prompts.append(
                    {
                        "id": base_id,
                        "strategy": "multi_expert",
                        "system_prompt": system_prompt,
                        "human_prompt": "\n".join(human_lines),
                        "metadata": meta,
                        "attributes": {**context_attrs, **mutable_attrs},
                    }
                )

            if "data_only" in prompt_families:
                human_lines = []
                if survey_ctx:
                    human_lines.append(survey_ctx)
                    human_lines.append("")
                human_lines.extend(_format_section("Predictors", context_lines))
                human_lines.append("")
                human_lines.append(_belief_update_task(mutable_state))
                prompts.append(
                    {
                        "id": base_id,
                        "strategy": "data_only",
                        "system_prompt": system_prompt,
                        "human_prompt": "\n".join(human_lines),
                        "metadata": meta,
                        "attributes": {**context_attrs, **mutable_attrs},
                    }
                )

            if "third_person" in prompt_families or "first_person" in prompt_families:
                narrative_context = _build_narrative_safe(
                    builder,
                    context_attrs,
                    context_cols,
                    "third_person",
                    dropped_cols_sink=narrative_dropped_cols,
                )
                if "third_person" in prompt_families:
                    human_lines = [
                        narrative_context,
                        "",
                    ]
                    if survey_ctx:
                        human_lines.extend([survey_ctx, ""])
                    human_lines.append(_belief_update_task(mutable_state))
                    prompts.append(
                        {
                            "id": base_id,
                            "strategy": "third_person",
                            "system_prompt": system_prompt,
                            "human_prompt": "\n".join(human_lines),
                            "metadata": meta,
                        }
                    )

                if "first_person" in prompt_families:
                    narrative_fp = _build_narrative_safe(
                        builder,
                        context_attrs,
                        context_cols,
                        "first_person",
                        dropped_cols_sink=narrative_dropped_cols,
                    )
                    narrative_fp_full = narrative_fp
                    prompt_user = "\n".join(
                        [
                            "Given the information above, predict the expected values of your mutable state variables.",
                            "",
                            _belief_update_task(mutable_state),
                        ]
                    )
                    narrative_system = builder.first_person_system_prompt()
                    prompts.append(
                        {
                            "id": base_id,
                            "strategy": "first_person",
                            "narrative_human_prompt": narrative_fp_full,
                            "predict_system_prompt": "",
                            "narrative_system_prompt": narrative_system,
                            "narrative_assistant_prompt": narrative_fp_full,
                            "narrative_user_prompt": prompt_user,
                            "metadata": meta,
                        }
                    )

    if missing_columns:
        missing_list = ", ".join(sorted(missing_columns))
        print(f"Warning: allowed columns missing in data: {missing_list}")
    if narrative_dropped_cols:
        dropped_list = ", ".join(sorted(narrative_dropped_cols))
        print(f"Warning: unmapped narrative attributes were omitted: {dropped_list}")

    _write_jsonl(Path(args.out), prompts)
    print(f"Wrote {len(prompts)} belief-update prompts to {args.out}")
    if run_type == "perturbed" and rows_without_pe > 0:
        print(f"Warning: skipped {rows_without_pe} sampled rows with no matching PE_ref_P entries.")
