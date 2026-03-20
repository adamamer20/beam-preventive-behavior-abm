"""Run unperturbed block-ablation (prompt knockout) for choice validation.

This script removes block-specific attributes from unperturbed prompts, re-samples
LLM predictions, and reports block-importance as the drop in pooled gini and
MAE skill relative to the full unperturbed run.

Outputs are written to a separate ablation root by default.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from beam_abm.cli import parser_compat as cli
from beam_abm.evaluation.choice._canonicalize_ids import normalize_model_slug
from beam_abm.evaluation.common.labels import (
    load_clean_spec_question_answer_maps,
    load_display_groups,
    load_display_labels,
    load_ordinal_endpoints_from_clean_spec,
    load_possible_numeric_bounds_from_clean_spec,
)
from beam_abm.evaluation.common.metrics import gini_from_cindex
from beam_abm.evaluation.prompting import (
    PromptTemplateBuilder,
    build_clean_spec_index,
)
from beam_abm.evaluation.utils.jsonl import read_jsonl as _read_jsonl
from beam_abm.evaluation.utils.jsonl import write_jsonl as _write_jsonl

EXCLUDED_OUTCOMES = {"flu_vaccinated_2023_2024"}


@dataclass(frozen=True)
class BlockSpec:
    name: str
    columns: list[str]


def _load_json(path: Path) -> dict | list:
    return json.loads(path.read_text(encoding="utf-8"))


def _iter_jsonl_stream(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                yield obj


def _prompt_fingerprint(row: dict[str, Any], *, model_hint: str | None = None) -> str:
    model = row.get("model") or model_hint or ""
    messages = row.get("messages")
    if not isinstance(messages, list) or not messages:
        messages = [
            {"from": "system", "value": str(row.get("system_prompt") or "")},
            {"from": "human", "value": str(row.get("human_prompt") or "")},
        ]
    payload = {"model": str(model), "messages": messages}
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _iter_samples_paths(specs: list[str]) -> list[Path]:
    paths: list[Path] = []
    for raw in specs:
        if not raw:
            continue
        p = Path(raw)
        if p.is_file():
            if p.suffix == ".jsonl" and p.name.startswith("samples"):
                paths.append(p)
            continue
        if p.is_dir():
            for cand in p.rglob("samples*.jsonl"):
                if cand.is_file():
                    paths.append(cand)
    # De-dupe while preserving order
    seen: set[Path] = set()
    out: list[Path] = []
    for p in paths:
        if p in seen:
            continue
        seen.add(p)
        out.append(p)
    return out


def _build_samples_cache(
    *,
    sample_paths: list[Path],
    model_hint: str,
    min_k: int,
) -> dict[str, list[dict[str, Any]]]:
    cache: dict[str, list[dict[str, Any]]] = {}
    for path in sample_paths:
        try:
            for row in _iter_jsonl_stream(path):
                fp = _prompt_fingerprint(row, model_hint=model_hint)
                if fp in cache:
                    continue
                samples = row.get("samples")
                if not isinstance(samples, list) or len(samples) < int(min_k):
                    continue
                kept = [s for s in samples if isinstance(s, dict)]
                if len(kept) < int(min_k):
                    continue
                cache[fp] = kept
        except OSError:
            continue
    return cache


def _safe_token(value: object) -> str:
    text = str(value) if value is not None else ""
    out = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in text).strip("_")
    return out or "unknown"


def _infer_outcome(row: dict) -> str | None:
    outcome = row.get("outcome")
    if isinstance(outcome, str) and outcome:
        return outcome
    meta = row.get("metadata")
    if isinstance(meta, dict):
        outcome = meta.get("outcome")
        if isinstance(outcome, str) and outcome:
            return outcome
    row_id = row.get("id") or row.get("dataset_id")
    if isinstance(row_id, str) and "__" in row_id:
        return row_id.split("__", 1)[0]
    return None


def _infer_prompt_family(row: dict, fallback: str) -> str:
    family = row.get("prompt_family")
    if isinstance(family, str) and family:
        return family
    strategy = row.get("strategy")
    if isinstance(strategy, str) and strategy:
        return strategy
    return fallback


def _load_base_prompts(prompts: Path, prompt_family: str) -> list[dict]:
    if prompts.is_dir():
        candidate = prompts / f"prompts__{prompt_family}.jsonl"
        if not candidate.exists():
            raise FileNotFoundError(f"Missing prompts file: {candidate}")
        return _read_jsonl(candidate)
    if prompts.exists():
        return _read_jsonl(prompts)
    raise FileNotFoundError(f"Missing prompts path: {prompts}")


def _infer_outcomes_from_full_root(*, full_root: Path) -> list[str]:
    if not full_root.exists() or not full_root.is_dir():
        return []
    out: list[str] = []
    for p in full_root.iterdir():
        if not p.is_dir():
            continue
        name = p.name
        if not name or name.startswith("_"):
            continue
        out.append(name)
    return sorted(set(out))


def _load_base_prompts_from_full_samples(
    *,
    full_root: Path,
    outcome: str,
    model_slug: str,
    strategy: str,
) -> list[dict]:
    samples_path = full_root / outcome / model_slug / strategy / "uncalibrated" / "samples.jsonl"
    if not samples_path.exists():
        raise FileNotFoundError("Auto prompt loading requires canonical samples.jsonl. Missing: " + str(samples_path))
    # Stream in and keep the full row dicts (they contain attributes/metadata needed for knockout).
    return [dict(r) for r in _iter_jsonl_stream(samples_path)]


def _block_map_from_model_plan(
    model_plan_path: Path,
    outcome: str,
    *,
    model_ref: str | None = None,
) -> dict[str, list[str]]:
    plan_raw = _load_json(model_plan_path)
    plan = plan_raw if isinstance(plan_raw, list) else plan_raw.get("models") or plan_raw.get("plan") or []
    if not isinstance(plan, list):
        return {}

    def _model_entry_by_name(entries: list[dict], model_name: str) -> dict | None:
        for entry in entries:
            if entry.get("model") == model_name:
                return entry
        return None

    def _select_model_plan_entry(plan_entries: list[dict], *, outcome: str) -> dict | None:
        candidates: list[dict] = []
        for entry in plan_entries:
            if entry.get("outcome") != outcome:
                continue
            if not isinstance(entry.get("predictors"), dict):
                continue
            candidates.append(entry)
        if not candidates:
            return None

        def _score(entry: dict) -> tuple[int, str]:
            model = str(entry.get("model") or "")
            desc = str(entry.get("description") or "").lower()
            if model.endswith("_ENGINES_REF"):
                return (0, model)
            if "ENGINES_REF" in model:
                return (1, model)
            if "ref engines" in desc:
                return (2, model)
            return (10, model)

        return sorted(candidates, key=_score)[0]

    entry = None
    entries = [e for e in plan if isinstance(e, dict)]
    if model_ref:
        entry = _model_entry_by_name(entries, model_ref)
    if entry is None:
        entry = _select_model_plan_entry(entries, outcome=outcome)
    if entry is None:
        return {}
    predictors = entry.get("predictors")
    if not isinstance(predictors, dict):
        return {}
    out: dict[str, list[str]] = {}
    for block, cols in predictors.items():
        if not isinstance(block, str) or not block.strip():
            continue
        if not isinstance(cols, list):
            continue
        out[block] = [str(c) for c in cols if isinstance(c, str) and c.strip()]
    return out


def _filter_blocks(
    block_map: dict[str, list[str]],
    *,
    include_blocks: tuple[str, ...],
    exclude_blocks: tuple[str, ...],
) -> list[BlockSpec]:
    include_set = {b for b in include_blocks if b}
    exclude_tokens = [b for b in exclude_blocks if b]
    out: list[BlockSpec] = []
    for name, cols in block_map.items():
        if include_set and name not in include_set:
            continue
        if any(token in name for token in exclude_tokens):
            continue
        out.append(BlockSpec(name=name, columns=cols))
    return out


def _prompt_builder(
    *,
    display_labels_path: Path,
    clean_spec_path: Path,
    column_types_path: Path,
) -> PromptTemplateBuilder:
    display_labels = load_display_labels(display_labels_path)
    display_groups = load_display_groups(display_labels_path)
    clean_questions, clean_answers = load_clean_spec_question_answer_maps(clean_spec_path)
    possible_bounds = load_possible_numeric_bounds_from_clean_spec(clean_spec_path)
    ordinal_endpoints = load_ordinal_endpoints_from_clean_spec(clean_spec_path)
    clean_spec_data = _load_json(clean_spec_path)
    encodings, by_id = build_clean_spec_index(clean_spec_data if isinstance(clean_spec_data, dict) else {})

    possible_bounds.setdefault("covax_legitimacy_scepticism_idx", (0.0, 1.0))

    col_types = pd.read_csv(column_types_path, sep="\t")
    if "column" in col_types.columns and "type" in col_types.columns:
        col_types_map = {
            str(k): str(v).strip().lower() for k, v in col_types.set_index("column")["type"].to_dict().items()
        }
    else:
        col_types_map = {}

    return PromptTemplateBuilder(
        display_labels=display_labels,
        display_groups=display_groups,
        clean_questions=clean_questions,
        clean_answers=clean_answers,
        encodings=encodings,
        by_id=by_id,
        column_types=col_types_map,
        possible_bounds=possible_bounds,
        ordinal_endpoints=ordinal_endpoints,
    )


def _attribute_block(
    *, prompt_builder: PromptTemplateBuilder, attrs: dict[str, object], predictor_cols: list[str]
) -> str:
    context_line = prompt_builder.maybe_survey_context(attrs)
    attr_lines = prompt_builder.format_attr_lines(attrs, predictor_cols, scale_seen=set())
    lines: list[str] = []
    if context_line:
        lines.append(context_line)
    lines.append("Attributes:")
    lines.extend(attr_lines)
    return "\n".join(lines)


def _messages_for_family(
    *,
    prompt_builder: PromptTemplateBuilder,
    family: str,
    outcome_key: str,
    predictor_cols: list[str],
    attrs: dict[str, object],
) -> tuple[str, str, list[dict[str, str]]]:
    if family == "data_only":
        system_prompt = (
            f"{prompt_builder.psychometrician_system(outcome_key).strip()}\n\n"
            "Fields not listed are unknown; treat them as unknown (do not assume typical values)."
        )
        human_prompt = _attribute_block(prompt_builder=prompt_builder, attrs=attrs, predictor_cols=predictor_cols)
        messages = [
            {"from": "system", "value": system_prompt},
            {"from": "human", "value": human_prompt},
        ]
        return system_prompt, human_prompt, messages
    if family == "first_person":
        narrative = prompt_builder.build_narrative(attrs, predictor_cols, "first_person")
        system_prompt = prompt_builder.first_person_system_prompt()
        human_prompt = prompt_builder.prediction_user_prompt_first_person(outcome_key)
        messages = [
            {"from": "system", "value": system_prompt},
            {"from": "assistant", "value": narrative},
            {"from": "human", "value": human_prompt},
        ]
        return system_prompt, human_prompt, messages
    if family == "third_person":
        narrative = prompt_builder.build_narrative(attrs, predictor_cols, "third_person")
        system_prompt = prompt_builder.psychometrician_system(outcome_key).strip()
        human_prompt = narrative
        messages = [
            {"from": "system", "value": system_prompt},
            {"from": "human", "value": narrative},
        ]
        return system_prompt, human_prompt, messages
    if family == "multi_expert":
        system_prompt = prompt_builder.psychometrician_system(outcome_key).strip()
        human_prompt = _attribute_block(prompt_builder=prompt_builder, attrs=attrs, predictor_cols=predictor_cols)
        messages = [
            {"from": "system", "value": system_prompt},
            {"from": "human", "value": human_prompt},
        ]
        return system_prompt, human_prompt, messages
    raise ValueError(f"Unknown prompt family: {family}")


def _build_knockout_prompts(
    *,
    base_prompts: list[dict],
    block: BlockSpec,
    outcome: str,
    prompt_family: str,
    prompt_builder: PromptTemplateBuilder,
    predictor_cols: list[str],
) -> list[dict]:
    drop_cols = set(block.columns)
    drop_cols.update({f"{c}_missing" for c in block.columns})

    out: list[dict] = []
    for row in base_prompts:
        row_outcome = _infer_outcome(row)
        if row_outcome != outcome:
            continue

        attrs = row.get("attributes") if isinstance(row.get("attributes"), dict) else {}
        attrs = dict(attrs)
        for col in drop_cols:
            attrs.pop(col, None)

        family = _infer_prompt_family(row, prompt_family)
        predictor_cols_clean = [c for c in predictor_cols if c not in drop_cols]
        system_prompt, human_prompt, messages = _messages_for_family(
            prompt_builder=prompt_builder,
            family=family,
            outcome_key=outcome,
            predictor_cols=predictor_cols_clean,
            attrs=attrs,
        )

        meta = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        meta = dict(meta)
        meta["knockout_block"] = block.name
        meta["knockout_cols"] = sorted(block.columns)

        new_row = dict(row)
        new_row["system_prompt"] = system_prompt
        new_row["human_prompt"] = human_prompt
        new_row["messages"] = messages
        new_row["attributes"] = attrs
        new_row["prompt_family"] = family
        new_row["strategy"] = row.get("strategy") or family
        new_row["metadata"] = meta
        if family in {"first_person", "third_person"}:
            new_row["predict_system_prompt"] = system_prompt
            new_row["narrative_system_prompt"] = system_prompt
            if family == "first_person":
                new_row["narrative_human_prompt"] = messages[1]["value"] if len(messages) > 1 else ""
                new_row["narrative_assistant_prompt"] = new_row["narrative_human_prompt"]
                new_row["narrative_user_prompt"] = messages[2]["value"] if len(messages) > 2 else ""
            else:
                new_row["narrative_user_prompt"] = messages[1]["value"] if len(messages) > 1 else ""

        out.append(new_row)

    return out


def _gini_np(x: np.ndarray, y: np.ndarray) -> float | None:
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 2:
        return None
    x = x[mask]
    y = y[mask]

    unique_preds = np.unique(x)
    pred_ranks = np.searchsorted(unique_preds, x) + 1
    n_ranks = int(unique_preds.size)

    order = np.argsort(y, kind="mergesort")
    y = y[order]
    pred_ranks = pred_ranks[order]

    tree = np.zeros(n_ranks + 2, dtype=np.int64)

    def _add(idx: int, value: int) -> None:
        while idx < tree.size:
            tree[idx] += value
            idx += idx & -idx

    def _sum(idx: int) -> int:
        s = 0
        while idx > 0:
            s += tree[idx]
            idx -= idx & -idx
        return s

    total_pairs = 0
    concordant = 0
    ties = 0
    i = 0
    n = int(y.size)
    while i < n:
        j = i + 1
        while j < n and y[j] == y[i]:
            j += 1
        total_prev = _sum(n_ranks)
        for k in range(i, j):
            rank = int(pred_ranks[k])
            less = _sum(rank - 1)
            equal = _sum(rank) - less
            concordant += less
            ties += equal
            total_pairs += total_prev
        for k in range(i, j):
            _add(int(pred_ranks[k]), 1)
        i = j

    if total_pairs == 0:
        return None
    cindex = float((concordant + 0.5 * ties) / total_pairs)
    return gini_from_cindex(cindex)


def _skill_np(y_true: np.ndarray, y_pred: np.ndarray) -> float | None:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if int(mask.sum()) == 0:
        return None
    yt = y_true[mask]
    yp = y_pred[mask]
    mae = float(np.mean(np.abs(yp - yt)))
    baseline = float(np.median(yt))
    baseline_mae = float(np.mean(np.abs(yt - baseline)))
    if baseline_mae <= 0:
        return None
    return float(1.0 - mae / baseline_mae)


def _metrics_from_predictions(df: pd.DataFrame, pred_col: str) -> dict[str, float | None]:
    if df.empty or "y_true" not in df.columns or pred_col not in df.columns:
        return {"n": 0.0, "gini": None, "skill": None}
    y_true = pd.to_numeric(df["y_true"], errors="coerce").to_numpy(dtype=float)
    y_pred = pd.to_numeric(df[pred_col], errors="coerce").to_numpy(dtype=float)
    gini = _gini_np(y_true, y_pred)
    skill = _skill_np(y_true, y_pred)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    return {
        "n": float(int(mask.sum())),
        "gini": float(gini) if gini is not None else None,
        "skill": float(skill) if skill is not None else None,
    }


def _load_predictions(path: Path, *, pred_col: str) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if pred_col not in df.columns and pred_col == "y_pred_cal" and "y_pred" in df.columns:
        df = df.rename(columns={"y_pred": "y_pred_cal"})
    return df


def _parse_top_ks(raw: str) -> tuple[int, ...]:
    vals: list[int] = []
    for token in str(raw).split(","):
        tok = token.strip()
        if not tok:
            continue
        k = int(tok)
        if k <= 0:
            raise ValueError("--top-k values must be positive integers.")
        vals.append(k)
    if not vals:
        raise ValueError("--top-k must contain at least one positive integer.")
    return tuple(sorted(set(vals)))


def _build_reference_rank_table(
    *,
    lobo_df: pd.DataFrame,
    levers: dict[str, Any],
    reference_model_slug: str | None,
) -> pd.DataFrame:
    required = {"model", "outcome", "block", "metric", "kind", "delta"}
    missing = sorted(required.difference(lobo_df.columns))
    if missing:
        raise ValueError(f"LOBO CSV is missing required columns: {missing}")

    mappings: list[dict[str, str]] = []
    for outcome, cfg in levers.items():
        if not isinstance(outcome, str) or not outcome or not isinstance(cfg, dict):
            continue
        model_ref = reference_model_slug or cfg.get("model_ref")
        if isinstance(model_ref, str) and model_ref:
            mappings.append({"outcome": outcome, "model_ref": model_ref})
    if not mappings:
        raise ValueError("No outcome->model_ref mapping found in lever config.")

    ref_models = pd.DataFrame(mappings).drop_duplicates()
    work = lobo_df.merge(
        ref_models,
        left_on=["outcome", "model"],
        right_on=["outcome", "model_ref"],
        how="inner",
    )
    work = work[(work["kind"].astype(str) == "lobo") & (work["metric"].astype(str) == "pseudo_r2")].copy()
    work["delta"] = pd.to_numeric(work["delta"], errors="coerce")
    work = work.dropna(subset=["delta", "block", "outcome"])
    if work.empty:
        raise ValueError("No pseudo_r2 LOBO rows found for configured reference models.")

    work["rank_ref"] = work.groupby(["outcome", "model_ref"], as_index=False)["delta"].rank(
        method="average",
        ascending=False,
    )
    return work[["outcome", "block", "model_ref", "delta", "rank_ref"]].rename(columns={"delta": "delta_ref"})


def _compute_posthoc_rank_error(
    summary_df: pd.DataFrame,
    *,
    lobo_df: pd.DataFrame,
    levers: dict[str, Any],
    reference_model_slug: str | None,
    top_ks: tuple[int, ...],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    required = {"outcome", "model", "strategy", "block", "delta_gini"}
    missing = sorted(required.difference(summary_df.columns))
    if missing:
        raise ValueError(f"Summary is missing required columns: {missing}")

    work = summary_df.copy()
    work["delta_gini"] = pd.to_numeric(work["delta_gini"], errors="coerce")
    work = work.dropna(subset=["outcome", "model", "strategy", "block", "delta_gini"]).copy()
    if work.empty:
        raise ValueError("No valid rows with delta_gini found for posthoc rank analysis.")

    work["rank"] = work.groupby(["outcome", "model", "strategy"], as_index=False)["delta_gini"].rank(
        method="average",
        ascending=False,
    )
    ref_df = _build_reference_rank_table(
        lobo_df=lobo_df,
        levers=levers,
        reference_model_slug=reference_model_slug,
    )
    llm_df = work[["outcome", "model", "strategy", "block", "rank", "delta_gini"]].rename(
        columns={"model": "model_llm", "rank": "rank_llm", "delta_gini": "delta_gini_llm"}
    )
    paired = llm_df.merge(ref_df, on=["outcome", "block"], how="inner")
    if paired.empty:
        raise ValueError("No overlap between LLM and reference rows by outcome/strategy/block for rank error.")
    paired["rank_signed_error"] = paired["rank_llm"] - paired["rank_ref"]
    paired["rank_abs_error"] = paired["rank_signed_error"].abs()
    for k in top_ks:
        paired[f"top{k}_agreement"] = ((paired["rank_llm"] <= float(k)) == (paired["rank_ref"] <= float(k))).astype(
            float
        )

    agg_cols: dict[str, str] = {
        "rank_abs_error": "median",
        "rank_signed_error": "median",
    }
    for k in top_ks:
        agg_cols[f"top{k}_agreement"] = "mean"

    by_block = (
        paired.groupby(["model_llm", "strategy", "block"], as_index=False)
        .agg(agg_cols)
        .rename(
            columns={
                "rank_abs_error": "median_abs_rank_error",
                "rank_signed_error": "median_signed_rank_error",
            }
        )
    )
    by_model_strategy = (
        paired.groupby(["model_llm", "strategy"], as_index=False)
        .agg(agg_cols)
        .rename(
            columns={
                "rank_abs_error": "median_abs_rank_error",
                "rank_signed_error": "median_signed_rank_error",
            }
        )
    )
    return paired, by_block, by_model_strategy


def _write_posthoc_outputs(
    *,
    summary_path: Path,
    lobo_path: Path,
    levers_path: Path,
    output_root: Path,
    reference_model_slug: str | None,
    top_ks: tuple[int, ...],
) -> None:
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary CSV for posthoc analysis: {summary_path}")
    if not lobo_path.exists():
        raise FileNotFoundError(f"Missing LOBO CSV for posthoc analysis: {lobo_path}")
    if not levers_path.exists():
        raise FileNotFoundError(f"Missing lever config for posthoc analysis: {levers_path}")
    summary_df = pd.read_csv(summary_path)
    lobo_df = pd.read_csv(lobo_path)
    levers_raw = _load_json(levers_path)
    if not isinstance(levers_raw, dict):
        raise ValueError(f"Lever config must be a JSON object: {levers_path}")
    paired, by_block, by_model_strategy = _compute_posthoc_rank_error(
        summary_df,
        lobo_df=lobo_df,
        levers=levers_raw,
        reference_model_slug=reference_model_slug,
        top_ks=top_ks,
    )
    out_dir = output_root / "_summary"
    out_dir.mkdir(parents=True, exist_ok=True)
    paired.to_csv(out_dir / "posthoc_rank_error_rows.csv", index=False)
    by_block.to_csv(out_dir / "posthoc_rank_error_by_block.csv", index=False)
    by_model_strategy.to_csv(out_dir / "posthoc_rank_error_by_model_strategy.csv", index=False)


def run_cli(argv: list[str] | None = None) -> None:
    parser = cli.ArgumentParser()
    parser.add_argument(
        "--prompts",
        required=False,
        default=None,
        help=(
            "Optional prompt JSONL path or directory containing prompts__<family>.jsonl. "
            "If omitted, the script auto-loads base prompts from canonical unperturbed samples.jsonl under --full-root "
            "for each outcome/model/strategy (recommended; reuses exactly what was evaluated in the full run)."
        ),
    )
    parser.add_argument(
        "--full-root",
        default="evaluation/output/choice_validation/unperturbed",
        help="Canonical unperturbed root for full metrics.",
    )
    parser.add_argument(
        "--output-root",
        default="evaluation/output/choice_validation/unperturbed_block_ablation",
        help="Output root for block-ablation runs.",
    )
    parser.add_argument("--model", default="openai/gpt-oss-20b")
    parser.add_argument(
        "--model-slug",
        default=None,
        help="Optional model slug used to locate canonical inputs.",
    )
    parser.add_argument("--strategy", default="data_only")
    parser.add_argument(
        "--outcomes",
        default=None,
        help="Comma-separated outcomes to run. Defaults to outcomes present in prompts.",
    )
    parser.add_argument(
        "--blocks",
        default=None,
        help="Comma-separated block names to ablate. Defaults to engine blocks from model_plan.",
    )
    parser.add_argument(
        "--exclude-blocks",
        default="demography,missing,missingness",
        help="Comma-separated block names to exclude when --blocks is not set.",
    )
    parser.add_argument(
        "--model-plan",
        default="empirical/output/modeling/model_plan.json",
        help="Model plan JSON used for block definitions.",
    )
    parser.add_argument(
        "--lever-config",
        default="config/levers.json",
        help="Lever config used to resolve driver cols for prompts.",
    )
    parser.add_argument(
        "--include-cols",
        default=None,
        help="Optional comma-separated extra columns to include in prompts.",
    )
    parser.add_argument(
        "--display-labels",
        default="thesis/specs/display_labels.json",
        help="Display labels JSON.",
    )
    parser.add_argument(
        "--clean-spec",
        default="preprocess/specs/5_clean_transformed_questions.json",
        help="Clean preprocessing spec.",
    )
    parser.add_argument(
        "--column-types",
        default="empirical/specs/column_types.tsv",
        help="Column types TSV.",
    )

    # Sampling controls
    parser.add_argument("--backend", choices=["vllm", "azure_openai"], default="vllm")
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--max-concurrency", type=int, default=None)
    parser.add_argument("--model-options", default=None)
    parser.add_argument("--model-options-file", default=None)
    parser.add_argument("--azure-deployments", default=None)
    parser.add_argument("--azure-endpoint", default=None)
    parser.add_argument("--azure-api-version", default=None)
    parser.add_argument(
        "--force-sampling",
        action="store_true",
        help="Force re-running sampling and metrics even if outputs exist.",
    )
    parser.add_argument(
        "--reuse-only",
        action="store_true",
        help="Reuse existing sampling/metrics outputs; never run sampling.",
    )
    parser.add_argument(
        "--posthoc-only",
        action="store_true",
        help="Skip knockout/sampling; compute rank-error posthoc analysis from existing summary CSV.",
    )
    parser.add_argument(
        "--posthoc-summary-csv",
        default=None,
        help="Optional path to existing unperturbed_block_ablation.csv for --posthoc-only.",
    )
    parser.add_argument(
        "--reference-model-slug",
        default=None,
        help=(
            "Optional global override for reference model slug. "
            "If omitted, uses per-outcome model_ref from --posthoc-levers-json."
        ),
    )
    parser.add_argument(
        "--posthoc-lobo-csv",
        default="empirical/output/modeling/model_lobo_all.csv",
        help="Path to LOBO summary CSV used to compute reference ranks.",
    )
    parser.add_argument(
        "--posthoc-levers-json",
        default="config/levers.json",
        help="Path to lever config JSON with outcome->model_ref mappings.",
    )
    parser.add_argument(
        "--top-k",
        default="1,2",
        help="Comma-separated k values for top-k agreement rates in posthoc analysis (default: 1,2).",
    )

    parser.add_argument(
        "--reuse-samples-from",
        action="append",
        default=[],
        help=(
            "Optional sources for reusing existing samples. Can be a samples*.jsonl file "
            "or a directory; directories are scanned recursively for samples*.jsonl. "
            "Repeat to add multiple sources. Reuse requires exact match on hashed prompt messages."
        ),
    )

    args = parser.parse_args(argv)
    if args.force_sampling and args.reuse_only:
        raise ValueError("--force-sampling and --reuse-only cannot both be set.")
    top_ks = _parse_top_ks(args.top_k)

    if args.posthoc_only:
        summary_csv = (
            Path(args.posthoc_summary_csv)
            if args.posthoc_summary_csv
            else Path(args.output_root) / "_summary" / "unperturbed_block_ablation.csv"
        )
        _write_posthoc_outputs(
            summary_path=summary_csv,
            lobo_path=Path(args.posthoc_lobo_csv),
            levers_path=Path(args.posthoc_levers_json),
            output_root=Path(args.output_root),
            reference_model_slug=args.reference_model_slug,
            top_ks=top_ks,
        )
        return

    from beam_abm.evaluation.choice.support.unperturbed import unperturbed_utils as utils
    from beam_abm.evaluation.choice.support.unperturbed.compute_metrics import run_cli as compute_metrics
    from beam_abm.evaluation.choice.support.unperturbed.run_sampling import run_cli as run_sampling

    base_prompts: list[dict] = []
    prompts_path = Path(args.prompts) if args.prompts else None
    if prompts_path is not None:
        base_prompts = _load_base_prompts(prompts_path, str(args.strategy))
        if not base_prompts:
            raise ValueError("No base prompts found.")

    full_root = Path(args.full_root)
    outcomes = list(utils.parse_list_arg(args.outcomes)) if args.outcomes else []
    if not outcomes and base_prompts:
        outcomes = sorted({o for o in map(_infer_outcome, base_prompts) if o})
    if not outcomes and not base_prompts:
        outcomes = _infer_outcomes_from_full_root(full_root=full_root)
    outcomes = [o for o in outcomes if o not in EXCLUDED_OUTCOMES]
    if not outcomes:
        raise ValueError("No outcomes found to run block ablation.")

    model_plan_path = Path(args.model_plan)
    lever_map = utils.load_lever_config(Path(args.lever_config))
    include_cols = list(utils.parse_list_arg(args.include_cols)) if args.include_cols else []

    prompt_builder = _prompt_builder(
        display_labels_path=Path(args.display_labels),
        clean_spec_path=Path(args.clean_spec),
        column_types_path=Path(args.column_types),
    )

    include_blocks = utils.parse_list_arg(args.blocks) if args.blocks else ()
    exclude_blocks = utils.parse_list_arg(args.exclude_blocks) if args.exclude_blocks else ()

    model_slug = args.model_slug or normalize_model_slug(args.model, reasoning_effort=None)
    model_slug = model_slug or _safe_token(args.model)

    pred_col = "y_pred"

    summary_rows: list[dict] = []
    missing_outputs: list[tuple[str, str, str]] = []

    full_metrics_cache: dict[tuple[str, str, str], dict[str, float | None]] = {}

    cache_specs = [str(Path(args.output_root)), *list(args.reuse_samples_from or [])]
    cache_paths = _iter_samples_paths(cache_specs)
    samples_cache = _build_samples_cache(
        sample_paths=cache_paths,
        model_hint=str(args.model),
        min_k=int(args.k),
    )

    for outcome in outcomes:
        lever_entry = lever_map.get(outcome) if isinstance(lever_map, dict) else None
        model_ref = lever_entry.get("model_ref") if isinstance(lever_entry, dict) else None
        block_map = _block_map_from_model_plan(
            model_plan_path,
            outcome,
            model_ref=model_ref,
        )
        if not block_map:
            continue

        blocks = _filter_blocks(
            block_map,
            include_blocks=include_blocks,
            exclude_blocks=exclude_blocks,
        )
        if not blocks:
            continue

        predictor_cols = utils.build_predictor_cols(
            outcome_key=outcome,
            lever_cfg=lever_map,
            model_plan=model_plan_path,
        )
        if include_cols:
            predictor_cols = list(dict.fromkeys([*predictor_cols, *include_cols]))

        # Load base prompt rows if not supplied explicitly.
        outcome_base_prompts = base_prompts
        if not outcome_base_prompts:
            outcome_base_prompts = _load_base_prompts_from_full_samples(
                full_root=full_root,
                outcome=outcome,
                model_slug=model_slug,
                strategy=str(args.strategy),
            )

        full_key = (outcome, model_slug, args.strategy)
        if full_key not in full_metrics_cache:
            full_leaf = full_root / outcome / model_slug / args.strategy / "uncalibrated"
            full_pred_path = full_leaf / "predictions.csv"
            full_df = _load_predictions(full_pred_path, pred_col=pred_col)
            full_metrics_cache[full_key] = _metrics_from_predictions(full_df, pred_col)

        for block in blocks:
            block_dir = Path(args.output_root) / outcome / model_slug / args.strategy / _safe_token(block.name)
            uncal_dir = block_dir / "uncalibrated"
            uncal_dir.mkdir(parents=True, exist_ok=True)

            prompts_out = uncal_dir / "prompts.jsonl"
            samples_path = uncal_dir / f"samples__{args.strategy}.jsonl"
            predictions_path = uncal_dir / f"predictions__{args.strategy}__all.csv"

            if args.reuse_only and not (samples_path.exists() and predictions_path.exists()):
                missing_outputs.append((outcome, block.name, str(uncal_dir)))
                continue

            knockout_prompts = _build_knockout_prompts(
                base_prompts=outcome_base_prompts,
                block=block,
                outcome=outcome,
                prompt_family=str(args.strategy),
                prompt_builder=prompt_builder,
                predictor_cols=predictor_cols,
            )
            if not knockout_prompts:
                continue

            _write_jsonl(prompts_out, knockout_prompts)

            if not (samples_path.exists() and predictions_path.exists()) or args.force_sampling:
                cached_rows: list[dict] = []
                missing_prompts: list[dict] = []

                local_cache: dict[str, list[dict[str, Any]]] = {}

                if args.force_sampling:
                    missing_prompts = list(knockout_prompts)
                else:
                    # Prefer locally existing samples (resume), then fall back to the global cache.
                    if samples_path.exists():
                        for row in _iter_jsonl_stream(samples_path):
                            fp = _prompt_fingerprint(row, model_hint=str(args.model))
                            samples = row.get("samples")
                            if isinstance(samples, list) and len(samples) >= int(args.k):
                                kept = [s for s in samples if isinstance(s, dict)]
                                if len(kept) >= int(args.k):
                                    local_cache[fp] = kept

                    for row in knockout_prompts:
                        fp = _prompt_fingerprint(row, model_hint=str(args.model))
                        samples = local_cache.get(fp)
                        if samples is None:
                            samples = samples_cache.get(fp)
                        if samples is not None and len(samples) >= int(args.k):
                            cached_rows.append({**row, "samples": samples, "model": str(args.model)})
                        else:
                            missing_prompts.append(row)

                if args.reuse_only and missing_prompts:
                    missing_outputs.append((outcome, block.name, str(uncal_dir)))
                    continue

                newly_sampled_rows: list[dict] = []
                if missing_prompts:
                    tmp_dir = uncal_dir / "_tmp_missing_sampling"
                    tmp_dir.mkdir(parents=True, exist_ok=True)
                    prompts_missing = tmp_dir / "prompts_missing.jsonl"
                    _write_jsonl(prompts_missing, missing_prompts)

                    sample_args = [
                        "--prompts",
                        str(prompts_missing),
                        "--outdir",
                        str(tmp_dir),
                        "--prompt-family",
                        str(args.strategy),
                        "--backend",
                        str(args.backend),
                        "--models",
                        str(args.model),
                        "--k",
                        str(int(args.k)),
                    ]
                    if args.temperature is not None:
                        sample_args.extend(["--temperature", str(float(args.temperature))])
                    if args.top_p is not None:
                        sample_args.extend(["--top-p", str(float(args.top_p))])
                    if args.max_concurrency is not None:
                        sample_args.extend(["--max-concurrency", str(int(args.max_concurrency))])
                    if args.model_options is not None:
                        sample_args.extend(["--model-options", str(args.model_options)])
                    if args.model_options_file is not None:
                        sample_args.extend(["--model-options-file", str(args.model_options_file)])
                    if args.azure_deployments is not None:
                        sample_args.extend(["--azure-deployments", str(args.azure_deployments)])
                    if args.azure_endpoint is not None:
                        sample_args.extend(["--azure-endpoint", str(args.azure_endpoint)])
                    if args.azure_api_version is not None:
                        sample_args.extend(["--azure-api-version", str(args.azure_api_version)])

                    run_sampling(sample_args)

                    missing_samples_path = tmp_dir / f"samples__{args.strategy}.jsonl"
                    if missing_samples_path.exists():
                        newly_sampled_rows = _read_jsonl(missing_samples_path)

                        # Update the global cache so later blocks can reuse within this run.
                        for sampled in newly_sampled_rows:
                            fp = _prompt_fingerprint(sampled, model_hint=str(args.model))
                            samples = sampled.get("samples")
                            if isinstance(samples, list) and len(samples) >= int(args.k):
                                kept = [s for s in samples if isinstance(s, dict)]
                                if len(kept) >= int(args.k) and fp not in samples_cache:
                                    samples_cache[fp] = kept

                merged_rows = [*cached_rows, *newly_sampled_rows]
                if merged_rows:
                    _write_jsonl(samples_path, merged_rows)

                metrics_args = [
                    "--samples",
                    str(samples_path),
                    "--outdir",
                    str(uncal_dir),
                    "--prompt-family",
                    str(args.strategy),
                ]
                compute_metrics(metrics_args)

            if not predictions_path.exists():
                missing_outputs.append((outcome, block.name, str(uncal_dir)))
                continue

            ko_df = _load_predictions(predictions_path, pred_col="y_pred")
            ko_metrics = _metrics_from_predictions(ko_df, "y_pred")
            full_metrics = full_metrics_cache.get(full_key, {"n": 0.0, "gini": None, "skill": None})

            summary_rows.append(
                {
                    "outcome": outcome,
                    "model": model_slug,
                    "strategy": args.strategy,
                    "block": block.name,
                    "gini_full": full_metrics.get("gini"),
                    "gini_ko": ko_metrics.get("gini"),
                    "delta_gini": (
                        None
                        if full_metrics.get("gini") is None or ko_metrics.get("gini") is None
                        else float(full_metrics.get("gini") - ko_metrics.get("gini"))
                    ),
                    "skill_full": full_metrics.get("skill"),
                    "skill_ko": ko_metrics.get("skill"),
                    "delta_skill": (
                        None
                        if full_metrics.get("skill") is None or ko_metrics.get("skill") is None
                        else float(full_metrics.get("skill") - ko_metrics.get("skill"))
                    ),
                    "n_full": full_metrics.get("n"),
                    "n_ko": ko_metrics.get("n"),
                }
            )

    if summary_rows:
        summary_path = Path(args.output_root) / "_summary" / "unperturbed_block_ablation.csv"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(summary_rows).to_csv(summary_path, index=False)

    if missing_outputs:
        missing_path = Path(args.output_root) / "_summary" / "ablation_missing.csv"
        missing_path.parent.mkdir(parents=True, exist_ok=True)
        missing_df = pd.DataFrame(missing_outputs, columns=["outcome", "block", "path"])
        missing_df.to_csv(missing_path, index=False)
