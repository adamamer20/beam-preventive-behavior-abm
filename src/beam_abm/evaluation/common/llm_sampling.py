"""Run LLM sampling for micro-validation prompts.

This module hosts the core sampling CLI logic for reuse in phase runners.
"""

from __future__ import annotations

import asyncio
import atexit
import json
import time
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import Field, ValidationError, create_model

from beam_abm.cli import parser_compat as cli
from beam_abm.common.logging import get_logger
from beam_abm.evaluation.artifacts import (
    build_conversation_id,
    build_prompt_id,
    hash_text,
    infer_run_id_from_path,
    utc_timestamp_iso,
)
from beam_abm.evaluation.common.baseline_lookup import _lookup_baseline_value as _lookup_baseline_value_shared
from beam_abm.evaluation.common.labels import load_possible_numeric_bounds_from_clean_spec
from beam_abm.evaluation.utils.jsonl import read_jsonl as _read_jsonl
from beam_abm.evaluation.utils.jsonl import write_jsonl as _write_jsonl
from beam_abm.evaluation.utils.regex_fallback import regex_extract_prediction
from beam_abm.llm.backends.factory import get_backend
from beam_abm.llm.inference import SampleResult, sample_many_concurrent
from beam_abm.llm.prompts.prompts import BELIEF_UPDATE_EXPERT_PROMPTS, EXPERT_PROMPTS, format_attribute_vector
from beam_abm.llm.schemas.model_config import ModelConfig, ModelOptions
from beam_abm.llm.schemas.predictions import extract_paired_scalar_predictions, extract_scalar_prediction_with_metadata
from beam_abm.llm.utils.conversation_logger import ConversationLogger
from beam_abm.settings import API_CONFIG

logger = get_logger(__name__)


def _load_symbol(path: str):
    # Format: module.submodule:Symbol
    mod, sym = path.split(":", 1)
    m = __import__(mod, fromlist=[sym])
    return getattr(m, sym)


def _messages_to_prompt_text(messages: list[dict[str, str]]) -> str:
    if not messages:
        return ""
    if len(messages) == 1:
        return messages[0].get("value") or messages[0].get("content") or ""
    return "\n\n".join(
        f"[{m.get('from') or m.get('role') or ''}]\n{m.get('value') or m.get('content') or ''}" for m in messages
    )


def _relpath(path: Path | str, base: Path) -> str:
    resolved = Path(path).resolve()
    base_resolved = base.resolve()
    if resolved.is_relative_to(base_resolved):
        return str(resolved.relative_to(base_resolved))
    return str(path)


def _parse_anchor_parts(row_id: object) -> tuple[int | None, str | None, int | None]:
    if not isinstance(row_id, str) or "::" not in row_id:
        return None, None, None
    parts = row_id.split("::", 2)
    if len(parts) != 3:
        return None, None, None
    anchor_raw, country, row_raw = parts
    anchor_id = int(anchor_raw) if anchor_raw.isdigit() else None
    row_idx = int(row_raw) if row_raw.isdigit() else None
    return anchor_id, (country if country else None), row_idx


def _build_row_level_outputs(
    *,
    out_rows: list[dict],
    model_name: str,
    backend: str,
    run_id: str | None,
    calibration_status: str,
    raw_conversations_path: Path,
    prompts_path: Path,
    samples_path: Path,
    base_dir: Path,
) -> tuple[list[dict], list[dict]]:
    raw_conversations: list[dict] = []
    row_level: list[dict] = []
    raw_rel = _relpath(raw_conversations_path, base_dir)
    samples_rel = _relpath(samples_path, base_dir)
    prompts_rel = _relpath(prompts_path, base_dir)
    source_files = json.dumps([raw_rel, samples_rel, prompts_rel])

    for row in out_rows:
        meta = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        prompt_text = row.get("_prompt_text") or row.get("prompt_text") or ""
        prompt_hash = row.get("_prompt_hash") or hash_text(prompt_text)
        prompt_id = row.get("prompt_id") or build_prompt_id(prompt_text)
        row["prompt_id"] = prompt_id
        dataset_id = row.get("id")
        outcome = row.get("outcome")
        if outcome is None and isinstance(meta, dict):
            outcome = meta.get("outcome")
        outcome_label = outcome if isinstance(outcome, str) else None
        strategy = row.get("strategy") or "data_only"

        anchor_id, country, row_idx = _parse_anchor_parts(dataset_id)
        reference_value = None
        reference_method = "unknown"
        if isinstance(row.get("baseline_y0"), int | float):
            reference_value = row.get("baseline_y0")
            reference_method = "baseline"
        elif isinstance(meta, dict) and meta.get("y_true") is not None:
            reference_value = meta.get("y_true")
            reference_method = "ground_truth"

        perturbation_params: dict[str, object] = {}
        if isinstance(meta, dict):
            for key in ("grid_point", "delta_z", "paired_order", "target"):
                if key in meta:
                    perturbation_params[key] = meta.get(key)

        samples = row.get("samples") if isinstance(row.get("samples"), list) else []
        for sample_idx, sample in enumerate(samples):
            response_text = ""
            if isinstance(sample, dict):
                response_text = (
                    sample.get("raw_text_original") or sample.get("raw_text") or sample.get("raw_text_structurer") or ""
                )
            if not response_text and isinstance(sample, dict):
                response_text = json.dumps(sample, ensure_ascii=False)

            response_hash = hash_text(response_text)
            conversation_id = build_conversation_id(
                prompt_id=prompt_id,
                response_text=response_text,
                sample_idx=int(sample_idx),
                model=model_name,
            )
            if isinstance(sample, dict):
                sample["prompt_id"] = prompt_id
                sample["conversation_id"] = conversation_id

            raw_conversations.append(
                {
                    "prompt_id": prompt_id,
                    "conversation_id": conversation_id,
                    "strategy": strategy,
                    "outcome": outcome_label,
                    "prompt_text": prompt_text,
                    "response_text": response_text,
                    "model": model_name,
                    "backend": backend,
                    "model_options": row.get("model_options"),
                    "timestamp_utc": utc_timestamp_iso(),
                    "prompt_hash": prompt_hash,
                    "response_hash": response_hash,
                }
            )

            parsed = sample.get("parsed") if isinstance(sample, dict) else None
            parsed_text = json.dumps(parsed, ensure_ascii=False) if isinstance(parsed, dict) else None

            row_level.append(
                {
                    "dataset_id": dataset_id,
                    "prompt_id": prompt_id,
                    "conversation_id": conversation_id,
                    "outcome": outcome_label,
                    "reference_value": reference_value,
                    "reference_method": reference_method,
                    "anchor_id": anchor_id,
                    "anchor_value": None,
                    "perturbation_id": (meta.get("grid_point") if isinstance(meta, dict) else None),
                    "perturbation_type": (meta.get("target") if isinstance(meta, dict) else None),
                    "perturbation_params": json.dumps(perturbation_params) if perturbation_params else None,
                    "model_prediction_raw": response_text,
                    "model_prediction_parsed": parsed_text,
                    "prediction_scalar": sample.get("prediction_scalar") if isinstance(sample, dict) else None,
                    "prediction_scalar_A": sample.get("prediction_scalar_A") if isinstance(sample, dict) else None,
                    "prediction_scalar_B": sample.get("prediction_scalar_B") if isinstance(sample, dict) else None,
                    "prediction_scalar_delta": (
                        sample.get("prediction_scalar_delta") if isinstance(sample, dict) else None
                    ),
                    "row_metrics_input_tokens": sample.get("input_tokens") if isinstance(sample, dict) else None,
                    "row_metrics_output_tokens": sample.get("output_tokens") if isinstance(sample, dict) else None,
                    "row_metrics_total_tokens": sample.get("total_tokens") if isinstance(sample, dict) else None,
                    "row_metrics_attempts": sample.get("attempts") if isinstance(sample, dict) else None,
                    "row_metrics_error": sample.get("error") if isinstance(sample, dict) else None,
                    "strategy": strategy,
                    "run_id": run_id,
                    "calibration_status": calibration_status,
                    "source_files": source_files,
                    "country": country,
                    "row_index": row_idx,
                }
            )

    return raw_conversations, row_level


def _load_json_arg(raw: str | None, path: str | None, *, label: str) -> Any:
    if raw and path:
        raise ValueError(f"Provide only one of --{label} or --{label}-file")
    if path:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    if raw:
        return json.loads(raw)
    return None


def _coerce_strategy_token_map(payload: Any) -> dict[str, int]:
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise TypeError("strategy token map must be a JSON object")
    out: dict[str, int] = {}
    for key, value in payload.items():
        if value is None:
            continue
        try:
            tokens = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid token value for strategy {key!r}: {value!r}") from exc
        if tokens <= 0:
            raise ValueError(f"Token value must be positive for strategy {key!r}")
        out[str(key)] = tokens
    return out


def _select_model_options(payload: Any, model_name: str) -> ModelOptions | None:
    if payload is None:
        return None
    if not isinstance(payload, dict):
        raise TypeError("model options must be a JSON object")
    if model_name in payload and isinstance(payload[model_name], dict):
        return ModelOptions.model_validate(payload[model_name])
    return ModelOptions.model_validate(payload)


def _infer_outcome_scale_from_column_types(
    *,
    rows: list[dict],
    column_types_path: Path,
) -> str | None:
    outcomes: set[str] = set()
    for r in rows:
        meta = r.get("metadata") if isinstance(r.get("metadata"), dict) else {}
        outcome = None
        if isinstance(meta, dict):
            outcome = meta.get("outcome")
        if outcome is None:
            outcome = r.get("outcome")
        if isinstance(outcome, str) and outcome.strip():
            outcomes.add(outcome.strip())

    if not outcomes:
        return None
    if len(outcomes) > 1:
        raise ValueError(f"Multiple outcomes in input; pass --outcome-scale explicitly: {sorted(outcomes)}")

    outcome = next(iter(outcomes))
    if not column_types_path.exists():
        raise FileNotFoundError(f"Column types file not found: {column_types_path}")

    ct = pd.read_csv(column_types_path, sep="\t")
    if "column" not in ct.columns or "type" not in ct.columns:
        raise ValueError(f"Column types file missing required columns: {column_types_path}")

    m = ct.set_index("column")["type"].astype(str).to_dict()
    t = str(m.get(outcome, "")).strip().lower()
    if not t:
        raise ValueError(f"Outcome column not found in column types spec: {outcome}")

    if t == "binary":
        return "binary"
    if t == "continuous":
        return "continuous"
    if t == "ordinal":
        return "ordinal"
    raise ValueError(f"Unsupported inferred outcome type for structured output: {outcome} -> {t}")


def _load_outcome_types(column_types_path: Path) -> dict[str, str]:
    if not column_types_path.exists():
        raise FileNotFoundError(f"Column types file not found: {column_types_path}")
    ct = pd.read_csv(column_types_path, sep="\t")
    if "column" not in ct.columns or "type" not in ct.columns:
        raise ValueError(f"Column types file missing required columns: {column_types_path}")
    return {str(k): str(v).strip().lower() for k, v in ct.set_index("column")["type"].to_dict().items()}


def _get_single_outcome(rows: list[dict]) -> str:
    outcomes: set[str] = set()
    for r in rows:
        meta = r.get("metadata") if isinstance(r.get("metadata"), dict) else {}
        outcome: str | None = None
        if isinstance(meta, dict):
            raw = meta.get("outcome")
            if isinstance(raw, str) and raw.strip():
                outcome = raw.strip()
        if outcome is None:
            raw = r.get("outcome")
            if isinstance(raw, str) and raw.strip():
                outcome = raw.strip()

        # Back-compat: some generators provide a single outcome via `outcomes` list.
        if outcome is None:
            raw_outcomes = None
            if isinstance(meta, dict):
                raw_outcomes = meta.get("outcomes")
            if raw_outcomes is None:
                raw_outcomes = r.get("outcomes")
            if isinstance(raw_outcomes, list) and len(raw_outcomes) == 1:
                only = raw_outcomes[0]
                if isinstance(only, str) and only.strip():
                    outcome = only.strip()

        if outcome is not None:
            outcomes.add(outcome)

    if not outcomes:
        raise ValueError(
            "Outcome name missing; cannot build structured response model. "
            "Expected `outcome` / `metadata.outcome` or a single-item `outcomes` / `metadata.outcomes`."
        )
    if len(outcomes) > 1:
        raise ValueError(f"Multiple outcomes found; expected one: {sorted(outcomes)}")
    return next(iter(outcomes))


def _numeric_bounds(levels: list[str] | None) -> tuple[float, float] | None:
    if not levels:
        return None
    numeric: list[float] = []
    for lv in levels:
        try:
            numeric.append(float(lv))
        except (TypeError, ValueError):
            return None
    if not numeric:
        return None
    return min(numeric), max(numeric)


def _build_scalar_field(
    *,
    outcome: str,
    outcome_type: str,
    ordinal_levels: list[str] | None,
    bounds: tuple[float, float] | None,
) -> tuple[type, object]:
    if outcome_type == "binary":
        return (
            float,
            Field(..., ge=0.0, le=1.0, description=f"P(Y=1) for {outcome} on [0,1]"),
        )
    if outcome_type == "ordinal":
        resolved_bounds = _numeric_bounds(ordinal_levels) if ordinal_levels else None
        if resolved_bounds is None:
            resolved_bounds = bounds
        if resolved_bounds:
            lo, hi = resolved_bounds
            return (
                float,
                Field(..., ge=float(lo), le=float(hi), description=f"Expected value for {outcome}"),
            )
        return (float, Field(..., description=f"Expected value for {outcome}"))
    if bounds:
        lo, hi = bounds
        return (
            float,
            Field(..., ge=float(lo), le=float(hi), description=f"Point prediction for {outcome}"),
        )
    return (float, Field(..., description=f"Point prediction for {outcome}"))


def _validate_parsed_prediction(response_model: type | None, parsed: object) -> tuple[object | None, str | None]:
    if response_model is None:
        return parsed, None
    try:
        if hasattr(response_model, "model_validate"):
            model = response_model.model_validate(parsed)
            if hasattr(model, "model_dump"):
                return model.model_dump(), None
            return model.dict(), None
        model = response_model.parse_obj(parsed)
        return model.dict(), None
    except ValidationError as exc:
        return None, str(exc)
    except Exception as exc:
        return None, str(exc)


def _build_scalar_outcome_model(
    *,
    outcome: str,
    outcome_type: str,
    ordinal_levels: list[str] | None,
    bounds: tuple[float, float] | None,
) -> type:
    field = _build_scalar_field(
        outcome=outcome,
        outcome_type=outcome_type,
        ordinal_levels=ordinal_levels,
        bounds=bounds,
    )
    return create_model("SingleOutcomePrediction", prediction=field)


def _build_paired_scalar_model(
    *,
    outcome: str,
    outcome_type: str,
    ordinal_levels: list[str] | None,
    bounds: tuple[float, float] | None,
) -> type:
    field = _build_scalar_field(
        outcome=outcome,
        outcome_type=outcome_type,
        ordinal_levels=ordinal_levels,
        bounds=bounds,
    )
    return create_model("PairedScalarPrediction", A=field, B=field)


def _build_multi_outcome_model(
    *,
    outcomes: list[str],
    outcome_types: dict[str, str],
    ordinal_levels: list[str] | None,
    bounds_by_outcome: dict[str, tuple[float, float]],
) -> type:
    fields: dict[str, tuple[type, object]] = {}
    for outcome in outcomes:
        t = outcome_types.get(outcome, "")
        if t in {"continuous", "binary", "ordinal"}:
            fields[outcome] = _build_scalar_field(
                outcome=outcome,
                outcome_type=t,
                ordinal_levels=ordinal_levels,
                bounds=bounds_by_outcome.get(outcome),
            )
        else:
            raise ValueError(f"Unsupported outcome type for multi-outcome structured output: {outcome} -> {t}")

    return create_model("MultiOutcomePrediction", **fields)


def _build_paired_multi_outcome_model(
    *,
    outcomes: list[str],
    outcome_types: dict[str, str],
    ordinal_levels: list[str] | None,
    bounds_by_outcome: dict[str, tuple[float, float]],
) -> type:
    inner = _build_multi_outcome_model(
        outcomes=outcomes,
        outcome_types=outcome_types,
        ordinal_levels=ordinal_levels,
        bounds_by_outcome=bounds_by_outcome,
    )
    return create_model(
        "PairedMultiOutcomePrediction",
        A=(inner, Field(..., description="Profile A multi-outcome predictions")),
        B=(inner, Field(..., description="Profile B multi-outcome predictions")),
    )


def _coerce_bounds(bounds: object) -> tuple[float, float] | None:
    if isinstance(bounds, list | tuple) and len(bounds) == 2:
        try:
            return float(bounds[0]), float(bounds[1])
        except (TypeError, ValueError):
            return None
    return None


def _coerce_endpoints(endpoints: object) -> tuple[str, str] | None:
    if isinstance(endpoints, list | tuple) and len(endpoints) == 2:
        lo = str(endpoints[0]).strip()
        hi = str(endpoints[1]).strip()
        if lo and hi:
            return lo, hi
    return None


def _format_ordinal_desc(
    *,
    bounds: tuple[float, float] | None,
    endpoints: tuple[str, str] | None,
) -> str:
    if bounds is not None:
        lo, hi = bounds
        if endpoints is not None:
            lo_lab, hi_lab = endpoints
            return (
                f"ordinal expected value in [{float(lo)}, {float(hi)}] "
                f"(where {float(lo)} = {lo_lab}, {float(hi)} = {hi_lab})"
            )
        return f"ordinal expected value in [{float(lo)}, {float(hi)}]"
    if endpoints is not None:
        lo_lab, hi_lab = endpoints
        return f"ordinal expected value (low = {lo_lab}, high = {hi_lab})"
    return "ordinal expected value"


def _format_multi_outcome_semantics(
    *,
    outcomes: list[str],
    outcome_types: dict[str, str],
    ordinal_levels: list[str] | None,
    labels_by_outcome: dict[str, str] | None = None,
    bounds_by_outcome: dict[str, tuple[float, float]] | None = None,
    endpoints_by_outcome: dict[str, tuple[str, str]] | None = None,
) -> tuple[str, str]:
    lines = ["Outcome semantics:"]
    ordinal_bounds = _numeric_bounds(ordinal_levels) if ordinal_levels else None
    for outcome in outcomes:
        label = (
            labels_by_outcome.get(outcome)
            if isinstance(labels_by_outcome, dict) and isinstance(labels_by_outcome.get(outcome), str)
            else outcome
        )
        t = str(outcome_types.get(outcome, "")).strip().lower()
        if t == "binary":
            desc = "binary probability in [0,1]"
        elif t == "ordinal":
            row_bounds = bounds_by_outcome.get(outcome) if isinstance(bounds_by_outcome, dict) else None
            row_endpoints = endpoints_by_outcome.get(outcome) if isinstance(endpoints_by_outcome, dict) else None
            resolved_bounds = row_bounds or ordinal_bounds
            desc = _format_ordinal_desc(bounds=resolved_bounds, endpoints=row_endpoints)
        elif t == "continuous":
            desc = "continuous value"
        else:
            desc = "value"
        lines.append(f"- {label}: {desc}")

    schema = "Return ONLY JSON with keys: " + ", ".join(outcomes) + "."
    return "\n".join(lines), schema


def _get_row_outcome(row: dict) -> str | None:
    meta = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    outcome = None
    if isinstance(meta, dict):
        outcome = meta.get("outcome")
    if outcome is None:
        outcome = row.get("outcome")
    if isinstance(outcome, str) and outcome.strip():
        return outcome.strip()
    return None


def _format_single_outcome_semantics(
    *,
    outcome: str | None,
    outcome_type: str | None,
    ordinal_levels: list[str] | None,
    outcome_label: str | None = None,
    bounds: tuple[float, float] | None = None,
    endpoints: tuple[str, str] | None = None,
) -> str:
    label = outcome_label or outcome or "outcome"
    if outcome_type == "binary":
        desc = "binary probability in [0,1]"
    elif outcome_type == "ordinal":
        resolved_bounds = bounds
        if resolved_bounds is None:
            resolved_bounds = _numeric_bounds(ordinal_levels) if ordinal_levels else None
        desc = _format_ordinal_desc(bounds=resolved_bounds, endpoints=endpoints)
    elif outcome_type == "continuous":
        desc = "continuous value"
    else:
        desc = "value"
    return f"Outcome semantics: {label} is a {desc}."


def _build_delta_detail_model(*, outcome: str | None) -> type:
    label = outcome or "outcome"
    return create_model(
        "ExpertDelta",
        delta_raw=(float, Field(..., description=f"Additive adjustment for {label}")),
        mechanism=(str, Field(..., description="Short mechanism summary")),
        fields_used=(list[str], Field(..., description="Attribute keys used for the delta")),
    )


def _build_delta_map_model() -> type:
    return create_model(
        "ExpertDeltaMap",
        delta=(dict[str, float], Field(..., description="Signed delta updates keyed by mutable state")),
        fields_used=(list[str], Field(..., description="Attribute keys used for the delta")),
    )


def _build_multi_outcome_delta_model(*, outcomes: list[str]) -> type:
    delta_model = _build_delta_detail_model(outcome=None)
    fields: dict[str, tuple[type, object]] = {}
    for outcome in outcomes:
        fields[outcome] = (delta_model, Field(..., description=f"Delta for {outcome}"))
    return create_model("MultiOutcomeDelta", **fields)


def _strip_expert_output_format(expert_prompt: str) -> str:
    """Remove hard-coded JSON schema instructions from expert role prompts.

    The per-row schema varies (single outcome vs multi-outcome vs belief-update).
    Keeping schema constraints only in the human prompt prevents contradictory
    instructions that cause structured-output validation failures.
    """

    marker = "Output format:"
    if marker not in expert_prompt:
        return expert_prompt
    return expert_prompt.split(marker, 1)[0].rstrip()


def _clip_prediction(value: float, *, outcome_type: str | None, ordinal_levels: list[str] | None) -> float:
    if outcome_type == "binary":
        return max(0.0, min(1.0, value))
    if outcome_type == "ordinal":
        bounds = _numeric_bounds(ordinal_levels) if ordinal_levels else None
        if bounds is not None:
            lo, hi = bounds
            return max(float(lo), min(float(hi), value))
    return value


def _coerce_float(value: object) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _extract_join_value(row: dict, join_key: str) -> str | None:
    value = row.get(join_key)
    if value is None and isinstance(row.get("metadata"), dict):
        value = row["metadata"].get(join_key)
    # The unperturbed microvalidation runners sometimes generate synthetic
    # per-family ids like `data_only__<join_id>` or `multi__<join_id>` while
    # using `--baseline-join-key join_id` to align baselines across families.
    # If the join key is missing, derive it from the synthetic `id`.
    if value is None and join_key == "join_id":
        synthetic_id = row.get("id")
        if synthetic_id is None and isinstance(row.get("metadata"), dict):
            synthetic_id = row["metadata"].get("id")
        if isinstance(synthetic_id, str) and synthetic_id:
            # Join-id conventions:
            # - Output/sample rows often use:  <outcome>__<join_id>
            # - Family prompt rows sometimes use: <join_id> (no outcome prefix)
            #
            # The baseline builder keeps the family-prefixed join id, e.g.
            #   id=vax_willingness_Y__multi__3793__0 -> join_id=multi__3793__0
            #   id=multi__3793__0                   -> join_id=multi__3793__0
            family_prefixes = (
                "data_only__",
                "first_person__",
                "third_person__",
                "paired_profile__",
                "multi_expert__",
                "multi__",
            )
            if synthetic_id.startswith(family_prefixes):
                value = synthetic_id
            elif "__" in synthetic_id:
                value = synthetic_id.split("__", 1)[1]
            else:
                value = synthetic_id
    if value is None:
        return None
    return str(value)


def _row_log_metadata(row: dict) -> tuple[str | None, str | None, str | None]:
    record_id = row.get("id")
    if record_id is not None:
        record_id = str(record_id)
    strategy = row.get("strategy")
    if strategy is not None:
        strategy = str(strategy)
    else:
        strategy = "data_only"

    meta = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    grid_point = None
    if isinstance(meta, dict):
        grid_point = meta.get("grid_point")
        if grid_point is None and any(k in meta for k in ("grid_point_low", "grid_point_high")):
            grid_point = "paired"
    if grid_point is None:
        grid_point = row.get("grid_point")

    perturbation = str(grid_point) if grid_point is not None else "unperturbed"
    return record_id, strategy, perturbation


def _load_baseline_predictions(
    *,
    path: Path,
    join_key: str,
    baseline_col: str,
) -> tuple[dict[str, float | dict[str, float]], dict[tuple[str, str], float]]:
    if not path.exists():
        raise FileNotFoundError(f"Baseline predictions file not found: {path}")

    by_id: dict[str, float | dict[str, float]] = {}
    by_id_outcome: dict[tuple[str, str], float] = {}

    def _load_rows() -> list[dict]:
        if path.suffix.lower() == ".jsonl":
            return _read_jsonl(path)
        sep = "\t" if path.suffix.lower() == ".tsv" else ","
        df = pd.read_csv(path, sep=sep)
        return df.to_dict(orient="records")

    for row in _load_rows():
        key = _extract_join_value(row, join_key)
        if key is None:
            continue
        outcome = _get_row_outcome(row)
        raw_val = row.get(baseline_col)
        if isinstance(raw_val, str) and raw_val.strip().startswith("{"):
            try:
                raw_val = json.loads(raw_val)
            except json.JSONDecodeError:
                pass
        if isinstance(raw_val, dict):
            coerced = {str(k): float(v) for k, v in raw_val.items() if _coerce_float(v) is not None}
            if coerced:
                by_id[key] = coerced
            continue
        val = _coerce_float(raw_val)
        if val is None:
            continue
        if outcome:
            by_id_outcome[(key, outcome)] = val
        else:
            by_id[key] = val

    return by_id, by_id_outcome


def _lookup_baseline_value(
    *,
    join_value: str | None,
    outcome: str | None,
    outcomes: list[str] | None,
    by_id: dict[str, float | dict[str, float]],
    by_id_outcome: dict[tuple[str, str], float],
) -> float | dict[str, float]:
    return _lookup_baseline_value_shared(
        join_value=join_value,
        outcome=outcome,
        outcomes=outcomes,
        by_id=by_id,
        by_id_outcome=by_id_outcome,
    )


def run_cli(argv: list[str] | None = None) -> None:
    parser = cli.ArgumentParser()
    parser.add_argument("--in", dest="in_path", required=True, help="Input JSONL of prompts")
    parser.add_argument("--out", dest="out_path", required=True, help="Output JSONL of sampled responses")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on number of input rows to sample (debugging).",
    )

    parser.add_argument("--backend", choices=["vllm", "azure_openai"], default="vllm")
    parser.add_argument("--model", required=True, help="Model name (vLLM HF name) or logical name")

    parser.add_argument("--azure-deployment", default=None)
    parser.add_argument("--azure-endpoint", default=None)
    parser.add_argument("--azure-api-version", default=None)
    parser.add_argument("--azure-model-hint", default=None)

    parser.add_argument("--model-options", default=None, help="JSON object of vLLM model options")
    parser.add_argument("--model-options-file", default=None, help="Path to JSON model options")

    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=None,
        help="Optional per-request max_model_len override (vLLM backend only).",
    )
    parser.add_argument(
        "--max-model-len-by-strategy",
        default=None,
        help="JSON map of strategy->max_model_len overrides.",
    )
    parser.add_argument(
        "--max-model-len-by-strategy-file",
        default="evaluation/specs/prompt_tournament_max_model_len.json",
        help="Path to JSON file with strategy->max_model_len overrides.",
    )
    parser.add_argument("--max-retries", type=int, default=int(API_CONFIG.get("llm_max_retries") or 10))
    parser.add_argument(
        "--max-concurrency",
        dest="max_concurrency",
        type=int,
        default=None,
        help="Optional cap on total concurrent sample tasks (samples-in-flight). When set, prompts are processed in batches so in-flight requests <= max_concurrency.",
    )
    parser.add_argument(
        "--two-pass",
        dest="two_pass",
        action="store_true",
        help="Enable two-pass sampling (raw text first, then structurer JSON conversion).",
    )
    parser.add_argument(
        "--no-two-pass",
        dest="two_pass",
        action="store_false",
        help="Disable two-pass sampling (single structured pass).",
    )
    parser.set_defaults(two_pass=True)
    parser.add_argument(
        "--structurer-model",
        default="Qwen/Qwen3-4B-Instruct-2507",
        help="Model used for the JSON structuring pass.",
    )
    parser.add_argument(
        "--structurer-temperature",
        type=float,
        default=0.7,
        help="Temperature for structurer pass.",
    )
    parser.add_argument(
        "--structurer-top-p",
        type=float,
        default=0.8,
        help="Top-p for structurer pass.",
    )
    parser.add_argument(
        "--structurer-top-k",
        type=int,
        default=20,
        help="Top-k for structurer pass.",
    )
    parser.add_argument(
        "--structurer-min-p",
        type=float,
        default=0.0,
        help="Min-p for structurer pass.",
    )

    parser.add_argument(
        "--no-structured",
        dest="structured",
        action="store_false",
        help="Disable structured JSON enforcement (default: enabled).",
    )
    parser.set_defaults(structured=True)
    parser.add_argument(
        "--response-model",
        default=None,
        help="Optional Pydantic schema import path, e.g. 'beam_abm.llm.schemas.survey:SurveyPrediction'",
    )
    parser.add_argument(
        "--outcome-scale",
        choices=["continuous", "binary", "ordinal"],
        default=None,
        help="Advanced override for the inferred outcome scale (normally inferred from --column-types).",
    )
    parser.add_argument(
        "--ordinal-levels",
        default=None,
        help="Comma-separated ordinal level labels (used only when --outcome-scale=ordinal).",
    )
    parser.add_argument(
        "--column-types",
        default="empirical/specs/column_types.tsv",
        help="Column type spec (TSV) used to infer outcome scale when --structured and --outcome-scale is omitted.",
    )
    parser.add_argument(
        "--clean-spec",
        default="preprocess/specs/5_clean_transformed_questions.json",
        help="Clean preprocessing spec (used to infer numeric bounds for structured validation).",
    )
    parser.add_argument(
        "--strategy",
        default="direct",
        choices=["direct", "multi_expert"],
        help="Sampling strategy. 'direct' uses provided prompts; 'multi_expert' builds expert prompts from attributes.",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Optional run id to record in row-level outputs (defaults to inferred output path).",
    )
    parser.add_argument(
        "--baseline-predictions",
        default=None,
        help="Path to data_only predictions (CSV or JSONL) for multi_expert aggregation.",
    )
    parser.add_argument(
        "--baseline-join-key",
        default="id",
        help="Join key used to align baseline predictions with multi_expert rows (default: id).",
    )
    parser.add_argument(
        "--baseline-col",
        default="y_pred",
        help="Column or field in baseline predictions to use (default: y_pred).",
    )
    parser.add_argument("--base-prompt", default="")
    parser.add_argument("--task", default="Return predictions on the declared outcome scale as JSON.")
    parser.add_argument("--method", default="microvalidation")

    args = parser.parse_args(argv)

    if args.backend == "azure_openai" and args.two_pass:
        logger.warning("Azure OpenAI backend does not support two-pass structuring; disabling --two-pass.")
        args.two_pass = False

    # Remote API backends can stall badly when we fan out thousands of requests at once.
    # Use a conservative default cap unless the user explicitly opts out.
    if args.max_concurrency is None and args.backend != "vllm":
        args.max_concurrency = 128
        logger.warning(
            "No --max-concurrency set for backend={}; defaulting to {} to avoid unbounded fan-out. "
            + "Override with --max-concurrency N.",
            str(args.backend),
            int(args.max_concurrency),
        )

    out_path = Path(args.out_path)
    base_dir = out_path.resolve().parent
    run_id = args.run_id or infer_run_id_from_path(out_path)
    raw_conversations_path = base_dir / "raw_conversations.jsonl"
    row_level_path = base_dir / "row_level.csv"
    extra_raw_conversations: list[dict] = []

    if args.backend == "vllm":
        if args.temperature is None:
            args.temperature = 0.4
        if args.top_p is None:
            args.top_p = 0.8

    in_path = Path(args.in_path)
    try:
        in_size_bytes = in_path.stat().st_size
    except FileNotFoundError:
        in_size_bytes = 0
    logger.info(
        "Loading input JSONL: path={} size_mb={:.2f}",
        str(in_path),
        in_size_bytes / (1024.0 * 1024.0),
    )
    t0 = time.perf_counter()
    rows = _read_jsonl(in_path)
    dt = max(1e-9, time.perf_counter() - t0)
    mb = in_size_bytes / (1024.0 * 1024.0)
    logger.info(
        "Loaded input JSONL: rows={} seconds={:.3f} throughput_mb_s={:.2f}",
        len(rows),
        dt,
        (mb / dt) if mb else 0.0,
    )
    if args.limit is not None:
        rows = rows[: max(0, int(args.limit))]

    if not rows:
        logger.warning(
            "No input rows found in {}; writing empty output to {} and exiting.",
            args.in_path,
            args.out_path,
        )
        _write_jsonl(out_path, [])
        return

    options_payload = _load_json_arg(args.model_options, args.model_options_file, label="model-options")
    max_model_len_payload = _load_json_arg(
        args.max_model_len_by_strategy,
        args.max_model_len_by_strategy_file,
        label="max-model-len-by-strategy",
    )
    max_model_len_by_strategy = _coerce_strategy_token_map(max_model_len_payload)
    logger.info(
        "Max model len configured: default={}, by_strategy={}",
        args.max_model_len,
        max_model_len_by_strategy or None,
    )
    model_options = _select_model_options(options_payload, args.model)
    structurer_options = _select_model_options(options_payload, args.structurer_model)
    model_options_payload = model_options.model_dump(exclude_none=True) if model_options is not None else None

    model = ModelConfig(
        name=args.model,
        backend=args.backend,
        options=model_options,
        azure=(
            {
                "deployment": args.azure_deployment,
                "endpoint": args.azure_endpoint,
                "api_version": args.azure_api_version,
                "model_hint": args.azure_model_hint,
            }
            if args.backend == "azure_openai"
            else None
        ),
    )
    structurer_model = ModelConfig(
        name=args.structurer_model,
        backend=args.backend,
        options=structurer_options,
        azure=(
            {
                "deployment": args.azure_deployment,
                "endpoint": args.azure_endpoint,
                "api_version": args.azure_api_version,
                "model_hint": args.azure_model_hint,
            }
            if args.backend == "azure_openai"
            else None
        ),
    )

    conversation_logger = ConversationLogger(
        output_dir=str(out_path.resolve().parent / "conversations"),
        filename=f"conversations_{out_path.stem}.csv",
    )
    atexit.register(conversation_logger.close)
    for active_model in (model, structurer_model):
        backend = get_backend(active_model)
        set_logger = getattr(backend, "set_conversation_logger", None)
        if callable(set_logger):
            set_logger(conversation_logger)

    def _is_belief_update_row(row: dict) -> bool:
        meta = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        return isinstance(meta, dict) and bool(meta.get("mutable_state"))

    backend_key = str(args.backend or "").strip().lower()
    is_responses_backend = backend_key in {"azure_openai", "azure", "openai", "openai_v1"}
    belief_update_rows = sum(1 for r in rows if _is_belief_update_row(r))

    # Provider-enforced structured outputs (Responses API) require strict object
    # schemas where `required` includes *all* properties. Belief-update prompts
    # intentionally emit sparse deltas (missing keys mean "no change"), so we
    # must NOT send `response_format` for those runs.
    use_structured_output_api = bool(args.structured)
    if bool(args.structured) and is_responses_backend and belief_update_rows == len(rows):
        use_structured_output_api = False
        logger.warning(
            "Disabling provider structured outputs for belief-update run (backend={}): Responses json_schema strict mode requires required-all objects; belief updates are sparse deltas. Will still parse/validate JSON locally.",
            backend_key,
        )

    logger.info(
        "Sampling config: backend={} model={} strategy={} rows={} k={} max_concurrency={} two_pass={} structured_local={} structured_api={}",
        str(args.backend),
        str(args.model),
        str(args.strategy),
        len(rows),
        int(args.k),
        int(args.max_concurrency) if args.max_concurrency is not None else None,
        bool(args.two_pass),
        bool(args.structured),
        bool(use_structured_output_api),
    )

    baseline_by_id: dict[str, float | dict[str, float]] = {}
    baseline_by_id_outcome: dict[tuple[str, str], float] = {}
    if args.strategy == "multi_expert":
        if not args.baseline_predictions:
            raise ValueError("--baseline-predictions is required for strategy=multi_expert")
        baseline_by_id, baseline_by_id_outcome = _load_baseline_predictions(
            path=Path(args.baseline_predictions),
            join_key=str(args.baseline_join_key),
            baseline_col=str(args.baseline_col),
        )

    strategies = {str(r.get("strategy")) for r in rows if r.get("strategy") is not None}
    paired_mode = (len(strategies) == 1 and "paired_profile" in strategies) or False

    response_model = _load_symbol(args.response_model) if args.response_model else None
    belief_update_response_model: type | None = None
    belief_update_paired_response_model: type | None = None
    if args.structured and response_model is None:
        belief_update_response_model = _load_symbol("beam_abm.llm.schemas.predictions:BeliefUpdatePrediction")
    if args.structured:
        belief_update_paired_response_model = _load_symbol(
            "beam_abm.llm.schemas.predictions:BeliefUpdatePairedPrediction"
        )
    outcome_types = _load_outcome_types(Path(args.column_types)) if args.structured else {}
    ordinal_levels = (
        [s.strip() for s in (args.ordinal_levels or "").split(",") if s.strip()] if args.ordinal_levels else []
    )
    bounds_by_outcome: dict[str, tuple[float, float]] = {}
    if args.structured:
        try:
            bounds_by_outcome = load_possible_numeric_bounds_from_clean_spec(Path(args.clean_spec))
        except Exception as exc:
            logger.warning(f"Failed to load bounds from clean spec: {exc}")

    def _row_outcomes(row: dict) -> list[str] | None:
        # Many prompt-tournament JSONL rows include both:
        # - a single `outcome` (the target for this row), and
        # - an `outcomes` list (the outcome batch / run context).
        #
        meta = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        # For sampling, only treat rows as multi-outcome when explicitly flagged.
        # This flag is set during prompt generation when --dedup-levers produces
        # a single prompt for multiple outcomes.
        if not bool(meta.get("multi_outcome")):
            if _get_row_outcome(row) is not None:
                return None
        outcomes = meta.get("outcomes") if isinstance(meta, dict) else None
        if outcomes is None:
            outcomes = row.get("outcomes")
        if isinstance(outcomes, list) and outcomes:
            return [str(o) for o in outcomes]
        return None

    has_multi_outcome = any(_row_outcomes(r) for r in rows)
    single_outcomes_in_file = {_get_row_outcome(r) for r in rows}
    single_outcomes_in_file.discard(None)

    if (
        args.structured
        and response_model is None
        and args.outcome_scale is None
        and not has_multi_outcome
        and len(single_outcomes_in_file) <= 1
    ):
        inferred = _infer_outcome_scale_from_column_types(rows=rows, column_types_path=Path(args.column_types))
        if inferred is not None:
            args.outcome_scale = inferred

    # Track whether the file-level response_model was built from the
    # multi-outcome branch so that single-outcome rows in a mixed file
    # can fall through to per-row scalar model construction instead of
    # inheriting a MultiOutcomePrediction schema that doesn't match their
    # prompt.
    _file_level_is_multi_outcome = False
    if args.structured and response_model is None and has_multi_outcome:
        multi_outcome_sets: set[tuple[str, ...]] = set()
        for r in rows:
            row_outcomes = _row_outcomes(r)
            if not row_outcomes:
                continue
            multi_outcome_sets.add(tuple(sorted({str(o).strip() for o in row_outcomes if str(o).strip()})))
        if len(multi_outcome_sets) == 1:
            outcomes_tuple = next(iter(multi_outcome_sets))
            outcomes_list = list(outcomes_tuple)
            outcome_types_for_multi = outcome_types
            if args.outcome_scale:
                outcome_types_for_multi = dict.fromkeys(outcomes_list, args.outcome_scale)
            if paired_mode:
                response_model = _build_paired_multi_outcome_model(
                    outcomes=outcomes_list,
                    outcome_types=outcome_types_for_multi,
                    ordinal_levels=ordinal_levels or None,
                    bounds_by_outcome=bounds_by_outcome,
                )
            else:
                response_model = _build_multi_outcome_model(
                    outcomes=outcomes_list,
                    outcome_types=outcome_types_for_multi,
                    ordinal_levels=ordinal_levels or None,
                    bounds_by_outcome=bounds_by_outcome,
                )
            _file_level_is_multi_outcome = True

    if args.structured and response_model is None and args.outcome_scale and not has_multi_outcome:
        levels = [s.strip() for s in (args.ordinal_levels or "").split(",") if s.strip()] if args.ordinal_levels else []
        outcome_name = _get_single_outcome(rows)
        if paired_mode:
            response_model = _build_paired_scalar_model(
                outcome=outcome_name,
                outcome_type=args.outcome_scale,
                ordinal_levels=levels or None,
                bounds=bounds_by_outcome.get(outcome_name),
            )
        else:
            response_model = _build_scalar_outcome_model(
                outcome=outcome_name,
                outcome_type=args.outcome_scale,
                ordinal_levels=levels or None,
                bounds=bounds_by_outcome.get(outcome_name),
            )

    def _attach_prompt_metadata(row: dict, messages: list[dict[str, str]]) -> None:
        prompt_text = _messages_to_prompt_text(messages)
        prompt_hash = hash_text(prompt_text)
        template_version = None
        meta = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        if isinstance(meta, dict):
            template_version = meta.get("prompt_template_version")
        template_version = template_version or row.get("prompt_template_version")
        prompt_id = build_prompt_id(prompt_text, template_version=str(template_version) if template_version else None)
        row["prompt_id"] = prompt_id
        row["_prompt_text"] = prompt_text
        row["_prompt_hash"] = prompt_hash

    if args.strategy == "direct":

        async def _run_direct() -> list[list[dict]]:
            results_by_index: dict[int, list[dict]] = {}

            # Helper to build per-row prediction messages; handles narrative two-pass logic
            async def _build_prediction_message(_i: int, r: dict) -> tuple[list[dict[str, str]], type | None, str]:
                # Start from file-level model, but SKIP it for single-outcome
                # rows when the file-level model was built from the
                # multi-outcome branch (the schemas are incompatible).
                row_outcomes = _row_outcomes(r)
                if _file_level_is_multi_outcome and not row_outcomes:
                    row_response_model = None
                else:
                    row_response_model = response_model
                meta = r.get("metadata") if isinstance(r.get("metadata"), dict) else {}
                is_belief_update = isinstance(meta, dict) and bool(meta.get("mutable_state"))
                strat = str(r.get("strategy") or "data_only")
                if (
                    args.structured
                    and is_belief_update
                    and strat == "paired_profile"
                    and belief_update_paired_response_model is not None
                ):
                    row_response_model = belief_update_paired_response_model
                if args.structured and row_response_model is None and is_belief_update:
                    row_response_model = belief_update_response_model
                if row_outcomes and args.structured and row_response_model is None:
                    outcome_types_for_multi = outcome_types
                    if args.outcome_scale:
                        outcome_types_for_multi = dict.fromkeys(row_outcomes, args.outcome_scale)
                    if paired_mode:
                        row_response_model = _build_paired_multi_outcome_model(
                            outcomes=row_outcomes,
                            outcome_types=outcome_types_for_multi,
                            ordinal_levels=ordinal_levels or None,
                            bounds_by_outcome=bounds_by_outcome,
                        )
                    else:
                        row_response_model = _build_multi_outcome_model(
                            outcomes=row_outcomes,
                            outcome_types=outcome_types_for_multi,
                            ordinal_levels=ordinal_levels or None,
                            bounds_by_outcome=bounds_by_outcome,
                        )
                elif args.structured and row_response_model is None and args.outcome_scale is None:
                    # Mixed single-outcome files (one outcome per row) should still be structured.
                    # Build a per-row scalar schema inferred from column types.
                    row_outcome = _get_row_outcome(r)
                    if row_outcome is None:
                        raise ValueError(
                            "Outcome name missing; cannot build structured response model. "
                            "Expected `outcome` / `metadata.outcome` or a non-empty `outcomes` list."
                        )
                    row_outcome_type = outcome_types.get(row_outcome, "").strip().lower()
                    if row_outcome_type not in {"continuous", "binary", "ordinal"}:
                        raise ValueError(
                            f"Unsupported inferred outcome type for structured output: {row_outcome} -> {row_outcome_type}"
                        )
                    if paired_mode:
                        row_response_model = _build_paired_scalar_model(
                            outcome=row_outcome,
                            outcome_type=row_outcome_type,
                            ordinal_levels=ordinal_levels or None,
                            bounds=bounds_by_outcome.get(row_outcome),
                        )
                    else:
                        row_response_model = _build_scalar_outcome_model(
                            outcome=row_outcome,
                            outcome_type=row_outcome_type,
                            ordinal_levels=ordinal_levels or None,
                            bounds=bounds_by_outcome.get(row_outcome),
                        )

                # Narrative strategies: pass deterministic narrative directly to prediction.
                if strat in {"first_person", "third_person"}:
                    narr_human = r.get("narrative_human_prompt")
                    pred_sys = r.get("predict_system_prompt")
                    narrative_system = r.get("narrative_system_prompt")
                    narrative_assistant = r.get("narrative_assistant_prompt")
                    narrative_user = r.get("narrative_user_prompt")

                    # Only use the narrative-specific prompt fields when they are actually present.
                    # Otherwise, fall through to the standard system/human (or legacy prompt) handling
                    # below, which uses the exact prompts as stored in the JSONL.
                    has_narrative_chat = isinstance(narrative_user, str)
                    has_predict_chat = isinstance(narr_human, str) and isinstance(pred_sys, str)
                    if not (has_narrative_chat or has_predict_chat):
                        pass
                    elif isinstance(narrative_user, str):
                        messages = []
                        sys_value = narrative_system if isinstance(narrative_system, str) else pred_sys
                        if isinstance(sys_value, str):
                            messages.append({"from": "system", "value": sys_value})
                        if isinstance(narrative_assistant, str):
                            messages.append({"from": "assistant", "value": narrative_assistant})
                        elif isinstance(narr_human, str):
                            messages.append({"from": "assistant", "value": narr_human})
                        messages.append({"from": "human", "value": narrative_user})
                        _attach_prompt_metadata(r, messages)
                        return messages, row_response_model, f"{args.method}::predict::{strat}"
                    else:
                        pred_human = f"Narrative:\n{narr_human.strip()}\n\nReturn only JSON."
                        messages = [{"from": "system", "value": pred_sys}, {"from": "human", "value": pred_human}]
                        _attach_prompt_metadata(r, messages)
                        return messages, row_response_model, f"{args.method}::predict::{strat}"

                # One-shot strategies: system+human only.
                sys_prompt = r.get("system_prompt")
                human_prompt = r.get("human_prompt")
                if not isinstance(sys_prompt, str) or not isinstance(human_prompt, str):
                    # Back-compat: accept legacy combined prompt.
                    legacy = r.get("prompt")
                    if not isinstance(legacy, str):
                        raise ValueError("Input rows must include system_prompt+human_prompt (or legacy prompt)")
                    messages = [{"from": "human", "value": legacy}]
                else:
                    messages = [{"from": "system", "value": sys_prompt}, {"from": "human", "value": human_prompt}]
                _attach_prompt_metadata(r, messages)
                return messages, row_response_model, args.method

            indexed_rows = list(enumerate(rows))
            if max_model_len_by_strategy:
                grouped: dict[int | None, list[tuple[int, dict]]] = {}
                for i, r in indexed_rows:
                    strat = str(r.get("strategy") or "data_only")
                    model_len_cap = int(max_model_len_by_strategy.get(strat, 0)) or None
                    grouped.setdefault(model_len_cap, []).append((i, r))
                group_items = list(grouped.items())
            else:
                group_items = [(args.max_model_len, indexed_rows)]

            for model_len_cap, group_rows in group_items:
                logger.info(
                    "Sampling batch with max_model_len={} (rows={})",
                    model_len_cap,
                    len(group_rows),
                )
                if args.max_concurrency is not None:
                    k = max(1, int(args.k))
                    batch_size = max(1, int(args.max_concurrency) // k)
                else:
                    batch_size = len(group_rows) if group_rows else 1

                for start in range(0, len(group_rows), batch_size):
                    batch = group_rows[start : start + batch_size]
                    prompts = ["" for _ in batch]
                    messages_list: list[list[dict[str, str]]] = []
                    response_models_list: list[type | None] = []
                    methods_list: list[str] = []
                    record_ids: list[str | None] = []
                    prompt_strategies: list[str | None] = []
                    perturbations: list[str | None] = []
                    start_index = batch[0][0] if batch else 0

                    # Prepare messages (narrative generation done serially here for simplicity)
                    for i, r in batch:
                        msgs, rm, method = await _build_prediction_message(i, r)
                        messages_list.append(msgs)
                        response_models_list.append(rm)
                        methods_list.append(method)
                        record_id, strategy, perturbation = _row_log_metadata(r)
                        record_ids.append(record_id)
                        prompt_strategies.append(strategy)
                        perturbations.append(perturbation)

                    # Call sampler for this batch (k samples per prompt concurrently)
                    if args.two_pass and use_structured_output_api:
                        raw_methods_list = [f"{m}::raw" for m in methods_list]
                        raw_batches = await sample_many_concurrent(
                            model=model,
                            prompts=prompts,
                            messages_list=messages_list,
                            response_model=None,
                            use_structured_output=False,
                            temperature=args.temperature,
                            top_p=args.top_p,
                            max_model_len=model_len_cap or args.max_model_len,
                            k=args.k,
                            max_retries=args.max_retries,
                            method=f"{args.method}::raw",
                            methods=raw_methods_list,
                            start_record_index=start_index,
                            max_concurrency=args.max_concurrency,
                            record_ids=record_ids,
                            prompt_strategies=prompt_strategies,
                            perturbations=perturbations,
                        )

                        structurer_prompts: list[str] = []
                        structurer_messages_list: list[list[dict[str, str]]] = []
                        structurer_models: list[type | None] = []
                        structurer_methods: list[str] = []
                        structurer_record_ids: list[str | None] = []
                        structurer_prompt_strategies: list[str | None] = []
                        structurer_perturbations: list[str | None] = []
                        structurer_map: list[tuple[int, int]] = []
                        structurer_raw_texts: list[str] = []
                        regex_failed = 0
                        regex_logged = 0

                        structured_batches: list[list[SampleResult | None]] = [
                            [None for _ in raw_samples] for raw_samples in raw_batches
                        ]

                        system_prompt = (
                            "You are a strict JSON-only parser. "
                            "Return ONLY valid JSON that matches the required schema. "
                            "Do not add any extra keys or commentary."
                        )

                        for prompt_idx, raw_samples in enumerate(raw_batches):
                            rm = response_models_list[prompt_idx]
                            allow_scalar_fallback = True
                            if rm is not None and hasattr(rm, "model_fields"):
                                allow_scalar_fallback = len(rm.model_fields) == 1
                            for sample_idx, raw_sample in enumerate(raw_samples):
                                if raw_sample.error or not raw_sample.raw_text:
                                    structured_batches[prompt_idx][sample_idx] = SampleResult(
                                        raw_text=raw_sample.raw_text,
                                        parsed=None,
                                        error=raw_sample.error or "empty raw_text from pass-1",
                                        attempts=raw_sample.attempts,
                                        input_tokens=raw_sample.input_tokens,
                                        output_tokens=raw_sample.output_tokens,
                                        total_tokens=raw_sample.total_tokens,
                                        raw_text_original=raw_sample.raw_text,
                                    )
                                    continue
                                parsed = regex_extract_prediction(
                                    raw_sample.raw_text,
                                    allow_scalar_fallback=allow_scalar_fallback,
                                )
                                if parsed is not None:
                                    validated, error = _validate_parsed_prediction(rm, parsed)
                                    if validated is None:
                                        if regex_logged < 5:
                                            raw_head = " ".join(raw_sample.raw_text.strip().split())[:200]
                                            logger.warning(
                                                "Regex parse validation failed; record_id=%s strategy=%s perturbation=%s allow_scalar_fallback=%s error=%s raw_text_head=%r",
                                                record_ids[prompt_idx],
                                                prompt_strategies[prompt_idx],
                                                perturbations[prompt_idx],
                                                allow_scalar_fallback,
                                                error,
                                                raw_head,
                                            )
                                            regex_logged += 1
                                    else:
                                        structured_batches[prompt_idx][sample_idx] = SampleResult(
                                            raw_text=raw_sample.raw_text,
                                            parsed=validated,
                                            error=None,
                                            attempts=raw_sample.attempts,
                                            input_tokens=raw_sample.input_tokens,
                                            output_tokens=raw_sample.output_tokens,
                                            total_tokens=raw_sample.total_tokens,
                                            raw_text_original=raw_sample.raw_text,
                                        )
                                        continue
                                regex_failed += 1
                                if regex_logged < 5:
                                    raw_head = " ".join(raw_sample.raw_text.strip().split())[:200]
                                    logger.warning(
                                        "Regex parse failed; record_id=%s strategy=%s perturbation=%s allow_scalar_fallback=%s raw_text_head=%r",
                                        record_ids[prompt_idx],
                                        prompt_strategies[prompt_idx],
                                        perturbations[prompt_idx],
                                        allow_scalar_fallback,
                                        raw_head,
                                    )
                                    regex_logged += 1
                                structurer_map.append((prompt_idx, sample_idx))
                                structurer_raw_texts.append(raw_sample.raw_text)
                                structured_batches[prompt_idx][sample_idx] = SampleResult(
                                    raw_text="",
                                    parsed=None,
                                    error="structurer pending",
                                    attempts=raw_sample.attempts,
                                    input_tokens=raw_sample.input_tokens,
                                    output_tokens=raw_sample.output_tokens,
                                    total_tokens=raw_sample.total_tokens,
                                    raw_text_original=raw_sample.raw_text,
                                )
                                structurer_prompts.append("")
                                structurer_messages_list.append(
                                    [
                                        {"from": "system", "value": system_prompt},
                                        {
                                            "from": "human",
                                            "value": (
                                                "Convert the following model output to JSON that matches the schema:\n\n"
                                                f"{raw_sample.raw_text.strip()}"
                                            ),
                                        },
                                    ]
                                )
                                structurer_models.append(rm)
                                structurer_methods.append(f"{methods_list[prompt_idx]}::structurer")
                                structurer_record_ids.append(record_ids[prompt_idx])
                                structurer_prompt_strategies.append(prompt_strategies[prompt_idx])
                                structurer_perturbations.append(perturbations[prompt_idx])

                        if regex_failed:
                            logger.warning(
                                "Regex parse failures in batch: count=%s structurer_calls=%s",
                                regex_failed,
                                len(structurer_prompts),
                            )

                        if structurer_prompts:
                            backend = get_backend(structurer_model)
                            shutdown_fn = getattr(backend, "shutdown_all_engines", None)
                            if callable(shutdown_fn):
                                logger.info("Forcing vLLM teardown before structurer pass")
                                await shutdown_fn(reason="two-pass-structurer", wait_seconds=0.5)
                            structurer_batches = await sample_many_concurrent(
                                model=structurer_model,
                                prompts=structurer_prompts,
                                messages_list=structurer_messages_list,
                                response_model=structurer_models,
                                use_structured_output=True,
                                temperature=args.structurer_temperature,
                                top_p=args.structurer_top_p,
                                top_k=args.structurer_top_k,
                                min_p=args.structurer_min_p,
                                max_model_len=model_len_cap or args.max_model_len,
                                k=1,
                                max_retries=args.max_retries,
                                method=f"{args.method}::structurer",
                                methods=structurer_methods,
                                start_record_index=start_index,
                                max_concurrency=args.max_concurrency,
                                record_ids=structurer_record_ids,
                                prompt_strategies=structurer_prompt_strategies,
                                perturbations=structurer_perturbations,
                            )
                            for map_idx, ((prompt_idx, sample_idx), structurer_result) in enumerate(
                                zip(structurer_map, structurer_batches, strict=False)
                            ):
                                if structurer_result:
                                    structurer_sample = structurer_result[0]
                                    raw_text_original = structurer_raw_texts[map_idx]
                                    structured_batches[prompt_idx][sample_idx] = SampleResult(
                                        raw_text=structurer_sample.raw_text,
                                        parsed=structurer_sample.parsed,
                                        error=structurer_sample.error,
                                        attempts=structurer_sample.attempts,
                                        input_tokens=structurer_sample.input_tokens,
                                        output_tokens=structurer_sample.output_tokens,
                                        total_tokens=structurer_sample.total_tokens,
                                        raw_text_original=raw_text_original,
                                        raw_text_structurer=structurer_sample.raw_text,
                                    )
                        for prompt_samples in structured_batches:
                            for sample_idx, sample in enumerate(prompt_samples):
                                if sample is None:
                                    prompt_samples[sample_idx] = SampleResult(
                                        raw_text="",
                                        parsed=None,
                                        error="structurer missing",
                                        attempts=0,
                                        input_tokens=0,
                                        output_tokens=0,
                                        total_tokens=0,
                                        raw_text_original="",
                                    )

                        batches = [list(prompt_samples) for prompt_samples in structured_batches]
                    else:
                        batches = await sample_many_concurrent(
                            model=model,
                            prompts=prompts,
                            messages_list=messages_list,
                            response_model=response_models_list,
                            use_structured_output=use_structured_output_api,
                            temperature=args.temperature,
                            top_p=args.top_p,
                            max_model_len=model_len_cap or args.max_model_len,
                            k=args.k,
                            max_retries=args.max_retries,
                            method=args.method,
                            methods=methods_list,
                            start_record_index=start_index,
                            max_concurrency=args.max_concurrency,
                            record_ids=record_ids,
                            prompt_strategies=prompt_strategies,
                            perturbations=perturbations,
                        )

                    for (row_idx, _r), batch_result in zip(batch, batches, strict=False):
                        packed: list[dict] = []
                        for s in batch_result if batch_result else []:
                            d = s.__dict__.copy()
                            if d.get("raw_text_original") is None:
                                d["raw_text_original"] = d.get("raw_text", "")
                            packed.append(d)
                        results_by_index[row_idx] = packed

            return [results_by_index.get(i, []) for i in range(len(rows))]

        packed_batches = asyncio.run(_run_direct())

        out_rows: list[dict] = []
        for row_idx, (row, samples_dicts) in enumerate(zip(rows, packed_batches, strict=False)):
            row["model_options"] = model_options_payload
            row_outcomes = _row_outcomes(row)
            if row_outcomes:
                by_outcome: dict[str, list[dict]] = {o: [] for o in row_outcomes}
                for d0 in samples_dicts:
                    if paired_mode:
                        parsed = d0.get("parsed")
                        parsed_a = parsed.get("A") if isinstance(parsed, dict) else None
                        parsed_b = parsed.get("B") if isinstance(parsed, dict) else None
                        for o in row_outcomes:
                            d = dict(d0)
                            if d.get("raw_text_original") is None:
                                d["raw_text_original"] = d.get("raw_text", "")
                            a_o = parsed_a.get(o) if isinstance(parsed_a, dict) else None
                            b_o = parsed_b.get(o) if isinstance(parsed_b, dict) else None
                            a_scalar, a_meta = extract_scalar_prediction_with_metadata(a_o)
                            b_scalar, b_meta = extract_scalar_prediction_with_metadata(b_o)
                            d["prediction_scalar_A"] = a_scalar
                            d["prediction_scalar_B"] = b_scalar
                            if a_scalar is not None and b_scalar is not None:
                                d["prediction_scalar_delta"] = float(b_scalar) - float(a_scalar)
                            if a_meta is not None:
                                d["prediction_scalar_meta_A"] = a_meta
                            if b_meta is not None:
                                d["prediction_scalar_meta_B"] = b_meta
                            by_outcome[o].append(d)
                    else:
                        for o in row_outcomes:
                            d = dict(d0)
                            if d.get("raw_text_original") is None:
                                d["raw_text_original"] = d.get("raw_text", "")
                            parsed = d.get("parsed", {})
                            parsed_o = parsed.get(o) if isinstance(parsed, dict) else None
                            if parsed_o is None and isinstance(parsed, dict):
                                parsed_o = parsed
                            scalar, meta = extract_scalar_prediction_with_metadata(parsed_o)
                            d["prediction_scalar"] = scalar
                            if meta is not None:
                                d["prediction_scalar_meta"] = meta
                            by_outcome[o].append(d)

                for o in row_outcomes:
                    meta = dict(row.get("metadata") or {})
                    meta["outcome"] = o
                    y_true_by_outcome = meta.get("y_true_by_outcome") if isinstance(meta, dict) else None
                    if isinstance(y_true_by_outcome, dict) and o in y_true_by_outcome:
                        meta["y_true"] = y_true_by_outcome.get(o)
                    outcome_type_by_outcome = meta.get("outcome_type_by_outcome") if isinstance(meta, dict) else None
                    outcome_type = None
                    if isinstance(outcome_type_by_outcome, dict) and o in outcome_type_by_outcome:
                        outcome_type = outcome_type_by_outcome.get(o)
                        meta["outcome_type"] = outcome_type
                    bounds_by_outcome = meta.get("bounds_by_outcome") if isinstance(meta, dict) else None
                    if isinstance(bounds_by_outcome, dict) and o in bounds_by_outcome:
                        meta["bounds"] = bounds_by_outcome.get(o)
                    endpoints_by_outcome = meta.get("endpoints_by_outcome") if isinstance(meta, dict) else None
                    if isinstance(endpoints_by_outcome, dict) and o in endpoints_by_outcome:
                        meta["endpoints"] = endpoints_by_outcome.get(o)

                    base_id = row.get("id")
                    base_id_s = str(base_id) if isinstance(base_id, str) and base_id.strip() else str(row_idx)
                    stable_id = f"{o}__{base_id_s}"
                    out_rows.append(
                        {
                            **row,
                            "id": stable_id,
                            "outcome": o,
                            "outcome_type": outcome_type,
                            "metadata": meta,
                            "samples": by_outcome[o],
                        }
                    )
            else:
                packed = []
                for d0 in samples_dicts:
                    d = dict(d0)
                    if d.get("raw_text_original") is None:
                        d["raw_text_original"] = d.get("raw_text", "")
                    scalar, meta = extract_scalar_prediction_with_metadata(d.get("parsed"))
                    d["prediction_scalar"] = scalar
                    if meta is not None:
                        d["prediction_scalar_meta"] = meta
                    paired = extract_paired_scalar_predictions(d.get("parsed"))
                    if paired is not None:
                        a, b = paired
                        d["prediction_scalar_A"] = a
                        d["prediction_scalar_B"] = b
                        if a is not None and b is not None:
                            d["prediction_scalar_delta"] = float(b) - float(a)
                    packed.append(d)
                out_rows.append({**row, "samples": packed})

        _write_jsonl(out_path, out_rows)
        raw_conversations, row_level = _build_row_level_outputs(
            out_rows=out_rows,
            model_name=str(args.model),
            backend=str(args.backend),
            run_id=run_id,
            calibration_status="uncalibrated",
            raw_conversations_path=raw_conversations_path,
            prompts_path=Path(args.in_path),
            samples_path=out_path,
            base_dir=base_dir,
        )
        if extra_raw_conversations:
            raw_conversations.extend(extra_raw_conversations)
        if raw_conversations:
            _write_jsonl(raw_conversations_path, raw_conversations)
        if row_level:
            row_level_path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(row_level).to_csv(row_level_path, index=False)
        return

    # Multi-expert strategy expects input JSONL rows with `attributes`.
    async def _run_all() -> list[dict]:
        # IMPORTANT: do not `gather` across all rows at once.
        # Each row spawns experts*k async calls. If we schedule all rows simultaneously,
        # we create millions of asyncio Tasks and (worse) any per-call semaphore would
        # be local to that row, making `--max-concurrency` ineffective globally.
        global_semaphore: asyncio.Semaphore | None = None
        if args.max_concurrency is not None:
            global_semaphore = asyncio.Semaphore(int(args.max_concurrency))

        logger.info(
            "multi_expert setup: experts={} k={} max_concurrency={}",
            max(len(EXPERT_PROMPTS), len(BELIEF_UPDATE_EXPERT_PROMPTS)),
            int(args.k),
            int(args.max_concurrency) if args.max_concurrency is not None else None,
        )

        async def _run_one(i: int, row: dict) -> dict:
            row_outcomes = _row_outcomes(row)
            attrs = row.get("attributes")
            if not isinstance(attrs, dict):
                raise ValueError("multi_expert strategy requires 'attributes' dict in input rows")
            meta = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
            is_belief_update = isinstance(meta, dict) and bool(meta.get("mutable_state"))
            expert_prompts = BELIEF_UPDATE_EXPERT_PROMPTS if is_belief_update else EXPERT_PROMPTS
            model_len_cap = None
            if max_model_len_by_strategy:
                strat = str(row.get("strategy") or "data_only")
                model_len_cap = int(max_model_len_by_strategy.get(strat, 0)) or None
            record_id, prompt_strategy, perturbation = _row_log_metadata(row)

            if is_belief_update:
                row_response_model = _build_delta_map_model()
            elif row_outcomes:
                row_response_model = _build_multi_outcome_delta_model(outcomes=row_outcomes)
            else:
                row_response_model = _build_delta_detail_model(outcome=_get_row_outcome(row))

            attr_block = None
            row_human_prompt = row.get("human_prompt")
            if isinstance(row_human_prompt, str) and row_human_prompt.strip():
                attr_block = row_human_prompt.strip()
            if attr_block is None:
                attr_text = format_attribute_vector(attrs)
                attr_block = f"Attributes:\n{attr_text}"
            join_value = _extract_join_value(row, str(args.baseline_join_key))
            baseline_y0 = _lookup_baseline_value(
                join_value=join_value,
                outcome=_get_row_outcome(row),
                outcomes=row_outcomes,
                by_id=baseline_by_id,
                by_id_outcome=baseline_by_id_outcome,
            )

            outcome_name = _get_row_outcome(row)
            outcome_type = args.outcome_scale
            if outcome_type is None and outcome_name is not None:
                outcome_type = outcome_types.get(outcome_name)

            if row_outcomes:
                labels_by_outcome: dict[str, str] = {}
                bounds_meta = meta.get("bounds_by_outcome") if isinstance(meta, dict) else None
                endpoints_meta = meta.get("endpoints_by_outcome") if isinstance(meta, dict) else None
                labels_meta = meta.get("outcome_labels_by_outcome") if isinstance(meta, dict) else None
                bounds_for_semantics: dict[str, tuple[float, float]] = {}
                endpoints_for_semantics: dict[str, tuple[str, str]] = {}
                if isinstance(bounds_meta, dict):
                    for outcome_name_key, outcome_bounds in bounds_meta.items():
                        coerced = _coerce_bounds(outcome_bounds)
                        if coerced is not None:
                            bounds_for_semantics[str(outcome_name_key)] = coerced
                if isinstance(endpoints_meta, dict):
                    for outcome_name_key, outcome_endpoints in endpoints_meta.items():
                        coerced = _coerce_endpoints(outcome_endpoints)
                        if coerced is not None:
                            endpoints_for_semantics[str(outcome_name_key)] = coerced
                if isinstance(labels_meta, dict):
                    for outcome_name_key, outcome_label_value in labels_meta.items():
                        if isinstance(outcome_label_value, str) and outcome_label_value.strip():
                            labels_by_outcome[str(outcome_name_key)] = outcome_label_value.strip()
                semantics, _schema = _format_multi_outcome_semantics(
                    outcomes=row_outcomes,
                    outcome_types=outcome_types,
                    ordinal_levels=ordinal_levels or None,
                    labels_by_outcome=labels_by_outcome or None,
                    bounds_by_outcome=bounds_for_semantics or None,
                    endpoints_by_outcome=endpoints_for_semantics or None,
                )
                if is_belief_update:
                    delta_schema = (
                        'Return ONLY JSON: {"delta": {<mutable_state_key>: <float>, ...}, "fields_used": ["..."]}.'
                    )
                else:
                    delta_schema = (
                        "Return ONLY JSON with keys: "
                        + ", ".join(row_outcomes)
                        + ". Each value must be a JSON object with keys "
                        + "'delta_raw', 'mechanism', 'fields_used'."
                    )
            else:
                row_bounds = _coerce_bounds(meta.get("bounds")) if isinstance(meta, dict) else None
                row_endpoints = _coerce_endpoints(meta.get("endpoints")) if isinstance(meta, dict) else None
                row_outcome_label = None
                if isinstance(meta, dict):
                    raw_outcome_label = meta.get("outcome_label")
                    if isinstance(raw_outcome_label, str) and raw_outcome_label.strip():
                        row_outcome_label = raw_outcome_label.strip()
                semantics = _format_single_outcome_semantics(
                    outcome=outcome_name,
                    outcome_type=outcome_type,
                    ordinal_levels=ordinal_levels or None,
                    outcome_label=row_outcome_label,
                    bounds=row_bounds,
                    endpoints=row_endpoints,
                )
                if is_belief_update:
                    delta_schema = (
                        'Return ONLY JSON: {"delta": {<mutable_state_key>: <float>, ...}, "fields_used": ["..."]}.'
                    )
                else:
                    delta_schema = "Return ONLY JSON with keys: delta_raw, mechanism, fields_used."

            row["model_options"] = model_options_payload
            base_prompt_parts = [attr_block, semantics, delta_schema]
            if args.base_prompt:
                base_prompt_parts.insert(0, str(args.base_prompt))
            base_prompt_text = "\n\n".join([part for part in base_prompt_parts if part])
            row["prompt_id"] = build_prompt_id(base_prompt_text)
            row["_prompt_text"] = base_prompt_text
            row["_prompt_hash"] = hash_text(base_prompt_text)

            # Expert deltas: single-turn per expert (system+human), no synthesis.
            expert_names = list(expert_prompts.keys())
            prompts = ["" for _ in expert_names]
            messages_list: list[list[dict[str, str]]] = []
            methods_list: list[str] = []
            for name, expert_prompt in expert_prompts.items():
                expert_prompt = _strip_expert_output_format(expert_prompt)
                if args.base_prompt:
                    system_prompt = (
                        f"{args.base_prompt}\n\n{expert_prompt}\n\n"
                        "You are one of several experts. Provide ONLY the delta adjustment to the baseline prediction. "
                        "Do NOT output full predictions."
                    )
                else:
                    system_prompt = (
                        f"{expert_prompt}\n\n"
                        "You are one of several experts. Provide ONLY the delta adjustment to the baseline prediction. "
                        "Do NOT output full predictions."
                    )
                # Always include explicit schema + semantics. Many prompt JSONLs provide
                # `human_prompt` as an attribute block only, which otherwise leaves
                # multi-outcome/belief-update rows underspecified and prone to
                # json_validate_failed errors.
                human_prompt_parts = [attr_block]
                if semantics and semantics not in attr_block:
                    human_prompt_parts.append(semantics)
                if delta_schema and delta_schema not in attr_block:
                    human_prompt_parts.append(delta_schema)
                human_prompt = "\n\n".join([p for p in human_prompt_parts if p])
                messages_list.append(
                    [
                        {"from": "system", "value": system_prompt},
                        {"from": "human", "value": human_prompt},
                    ]
                )
                methods_list.append(f"{args.method}::expert::{name}")

            response_models_list = [row_response_model for _ in expert_names]
            if args.two_pass and use_structured_output_api:
                raw_methods_list = [f"{m}::raw" for m in methods_list]
                raw_batches = await sample_many_concurrent(
                    model=model,
                    prompts=prompts,
                    messages_list=messages_list,
                    use_structured_output=False,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    max_model_len=model_len_cap or args.max_model_len,
                    k=args.k,
                    max_retries=args.max_retries,
                    method=f"{args.method}::expert::raw",
                    methods=raw_methods_list,
                    start_record_index=i,
                    max_concurrency=args.max_concurrency,
                    semaphore=global_semaphore,
                    record_ids=[record_id for _ in expert_names],
                    prompt_strategies=[prompt_strategy for _ in expert_names],
                    perturbations=[perturbation for _ in expert_names],
                )

                structurer_prompts: list[str] = []
                structurer_messages_list: list[list[dict[str, str]]] = []
                structurer_models: list[type | None] = []
                structurer_methods: list[str] = []
                structurer_record_ids: list[str | None] = []
                structurer_prompt_strategies: list[str | None] = []
                structurer_perturbations: list[str | None] = []
                structurer_map: list[tuple[int, int]] = []
                structurer_raw_texts: list[str] = []
                regex_failed = 0
                regex_logged = 0

                structured_batches: list[list[SampleResult | None]] = [
                    [None for _ in raw_samples] for raw_samples in raw_batches
                ]

                system_prompt = (
                    "You are a strict JSON-only parser. "
                    "Return ONLY valid JSON that matches the required schema. "
                    "Do not add any extra keys or commentary."
                )

                for prompt_idx, raw_samples in enumerate(raw_batches):
                    rm = response_models_list[prompt_idx]
                    allow_scalar_fallback = True
                    if rm is not None and hasattr(rm, "model_fields"):
                        allow_scalar_fallback = len(rm.model_fields) == 1
                    for sample_idx, raw_sample in enumerate(raw_samples):
                        if raw_sample.error or not raw_sample.raw_text:
                            structured_batches[prompt_idx][sample_idx] = SampleResult(
                                raw_text=raw_sample.raw_text,
                                parsed=None,
                                error=raw_sample.error or "empty raw_text from pass-1",
                                attempts=raw_sample.attempts,
                                input_tokens=raw_sample.input_tokens,
                                output_tokens=raw_sample.output_tokens,
                                total_tokens=raw_sample.total_tokens,
                                raw_text_original=raw_sample.raw_text,
                            )
                            continue
                        parsed = regex_extract_prediction(
                            raw_sample.raw_text,
                            allow_scalar_fallback=allow_scalar_fallback,
                        )
                        if parsed is not None:
                            structured_batches[prompt_idx][sample_idx] = SampleResult(
                                raw_text=raw_sample.raw_text,
                                parsed=parsed,
                                error=None,
                                attempts=raw_sample.attempts,
                                input_tokens=raw_sample.input_tokens,
                                output_tokens=raw_sample.output_tokens,
                                total_tokens=raw_sample.total_tokens,
                                raw_text_original=raw_sample.raw_text,
                            )
                            continue
                        regex_failed += 1
                        if regex_logged < 5:
                            raw_head = " ".join(raw_sample.raw_text.strip().split())[:200]
                            logger.warning(
                                "Regex parse failed (multi_expert); record_id=%s strategy=%s perturbation=%s allow_scalar_fallback=%s raw_text_head=%r",
                                record_id,
                                prompt_strategy,
                                perturbation,
                                allow_scalar_fallback,
                                raw_head,
                            )
                            regex_logged += 1
                        structurer_map.append((prompt_idx, sample_idx))
                        structurer_raw_texts.append(raw_sample.raw_text)
                        structured_batches[prompt_idx][sample_idx] = SampleResult(
                            raw_text="",
                            parsed=None,
                            error="structurer pending",
                            attempts=raw_sample.attempts,
                            input_tokens=raw_sample.input_tokens,
                            output_tokens=raw_sample.output_tokens,
                            total_tokens=raw_sample.total_tokens,
                            raw_text_original=raw_sample.raw_text,
                        )
                        structurer_prompts.append("")
                        structurer_messages_list.append(
                            [
                                {"from": "system", "value": system_prompt},
                                {
                                    "from": "human",
                                    "value": (
                                        "Convert the following model output to JSON that matches the schema:\n\n"
                                        f"{raw_sample.raw_text.strip()}"
                                    ),
                                },
                            ]
                        )
                        structurer_models.append(rm)
                        structurer_methods.append(f"{methods_list[prompt_idx]}::structurer")
                        structurer_record_ids.append(record_id)
                        structurer_prompt_strategies.append(prompt_strategy)
                        structurer_perturbations.append(perturbation)

                if regex_failed:
                    logger.warning(
                        "Regex parse failures in batch (multi_expert): count=%s structurer_calls=%s",
                        regex_failed,
                        len(structurer_prompts),
                    )

                if structurer_prompts:
                    backend = get_backend(structurer_model)
                    shutdown_fn = getattr(backend, "shutdown_all_engines", None)
                    if callable(shutdown_fn):
                        logger.info("Forcing vLLM teardown before structurer pass (multi_expert)")
                        await shutdown_fn(reason="two-pass-structurer", wait_seconds=0.5)
                    structurer_batches = await sample_many_concurrent(
                        model=structurer_model,
                        prompts=structurer_prompts,
                        messages_list=structurer_messages_list,
                        response_model=structurer_models,
                        use_structured_output=True,
                        temperature=args.structurer_temperature,
                        top_p=args.structurer_top_p,
                        top_k=args.structurer_top_k,
                        min_p=args.structurer_min_p,
                        max_model_len=model_len_cap or args.max_model_len,
                        k=1,
                        max_retries=args.max_retries,
                        method=f"{args.method}::structurer",
                        methods=structurer_methods,
                        start_record_index=i,
                        max_concurrency=args.max_concurrency,
                        semaphore=global_semaphore,
                        record_ids=structurer_record_ids,
                        prompt_strategies=structurer_prompt_strategies,
                        perturbations=structurer_perturbations,
                    )
                    for map_idx, ((prompt_idx, sample_idx), structurer_result) in enumerate(
                        zip(structurer_map, structurer_batches, strict=False)
                    ):
                        if structurer_result:
                            structurer_sample = structurer_result[0]
                            raw_text_original = structurer_raw_texts[map_idx]
                            structured_batches[prompt_idx][sample_idx] = SampleResult(
                                raw_text=structurer_sample.raw_text,
                                parsed=structurer_sample.parsed,
                                error=structurer_sample.error,
                                attempts=structurer_sample.attempts,
                                input_tokens=structurer_sample.input_tokens,
                                output_tokens=structurer_sample.output_tokens,
                                total_tokens=structurer_sample.total_tokens,
                                raw_text_original=raw_text_original,
                                raw_text_structurer=structurer_sample.raw_text,
                            )
                for prompt_samples in structured_batches:
                    for sample_idx, sample in enumerate(prompt_samples):
                        if sample is None:
                            prompt_samples[sample_idx] = SampleResult(
                                raw_text="",
                                parsed=None,
                                error="structurer missing",
                                attempts=0,
                                input_tokens=0,
                                output_tokens=0,
                                total_tokens=0,
                                raw_text_original="",
                            )

                batches = [list(prompt_samples) for prompt_samples in structured_batches]
            else:
                batches = await sample_many_concurrent(
                    model=model,
                    prompts=prompts,
                    messages_list=messages_list,
                    response_model=response_models_list,
                    use_structured_output=use_structured_output_api,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    max_model_len=model_len_cap or args.max_model_len,
                    k=args.k,
                    max_retries=args.max_retries,
                    method=f"{args.method}::expert",
                    methods=methods_list,
                    start_record_index=i,
                    max_concurrency=args.max_concurrency,
                    semaphore=global_semaphore,
                    record_ids=[record_id for _ in expert_names],
                    prompt_strategies=[prompt_strategy for _ in expert_names],
                    perturbations=[perturbation for _ in expert_names],
                )

            outcome_label = None
            if row_outcomes:
                outcome_label = "_".join([str(o) for o in row_outcomes])
            else:
                outcome_label = _get_row_outcome(row)

            for expert_idx, _name in enumerate(expert_names):
                prompt_text = _messages_to_prompt_text(messages_list[expert_idx])
                prompt_hash = hash_text(prompt_text)
                prompt_id = build_prompt_id(prompt_text)
                for sample_idx, sample in enumerate(batches[expert_idx] if expert_idx < len(batches) else []):
                    response_text = sample.raw_text_original or sample.raw_text or ""
                    response_hash = hash_text(response_text)
                    conversation_id = build_conversation_id(
                        prompt_id=prompt_id,
                        response_text=response_text,
                        sample_idx=int(sample_idx),
                        model=str(args.model),
                    )
                    extra_raw_conversations.append(
                        {
                            "prompt_id": prompt_id,
                            "conversation_id": conversation_id,
                            "strategy": row.get("strategy", "multi_expert"),
                            "outcome": outcome_label,
                            "prompt_text": prompt_text,
                            "response_text": response_text,
                            "model": str(args.model),
                            "backend": str(args.backend),
                            "model_options": model_options_payload,
                            "timestamp_utc": utc_timestamp_iso(),
                            "prompt_hash": prompt_hash,
                            "response_hash": response_hash,
                        }
                    )

            def _parse_delta(
                parsed: object,
                *,
                outcome: str | None,
                allow_flat_fallback: bool,
            ) -> float | None:
                if not isinstance(parsed, dict):
                    return None
                if is_belief_update:
                    delta_payload = parsed.get("delta")
                    if not isinstance(delta_payload, dict):
                        return None
                    if outcome is None:
                        return None
                    if outcome not in delta_payload:
                        return 0.0
                    return _coerce_float(delta_payload.get(outcome))
                payload: object = parsed
                if outcome is not None:
                    nested = parsed.get(outcome)
                    if isinstance(nested, dict):
                        payload = nested
                    elif allow_flat_fallback:
                        payload = parsed
                    else:
                        return None
                if not isinstance(payload, dict):
                    return None
                return _coerce_float(payload.get("delta_raw"))

            def _per_expert_parsed(
                parsed: object,
                *,
                outcome: str | None,
                allow_flat_fallback: bool,
            ) -> dict[str, Any] | None:
                if not isinstance(parsed, dict):
                    return None
                if is_belief_update:
                    return parsed
                if outcome is None:
                    return parsed
                payload = parsed.get(outcome)
                if isinstance(payload, dict):
                    return payload
                if allow_flat_fallback:
                    return parsed
                return None

            def _mean(values: list[float]) -> float:
                if not values:
                    return float("nan")
                return float(sum(values) / len(values))

            base_payload = {
                **row,
                "strategy": row.get("strategy", "multi_expert"),
            }

            if row_outcomes:
                allow_flat_fallback = len(row_outcomes) == 1
                by_outcome: list[dict] = []
                for outcome in row_outcomes:
                    outcome_type = outcome_types.get(outcome)
                    outcome_samples: list[dict] = []
                    per_outcome_y: list[float] = []
                    per_outcome_expert_deltas: dict[str, list[dict[str, Any] | None]] = {
                        name: [] for name in expert_names
                    }

                    for sample_idx in range(int(args.k)):
                        per_expert: dict[str, dict[str, Any] | None] = {}
                        errors: dict[str, str] = {}
                        deltas: list[float] = []
                        for expert_idx, name in enumerate(expert_names):
                            sample = batches[expert_idx][sample_idx] if sample_idx < len(batches[expert_idx]) else None
                            parsed = sample.parsed if sample is not None else None
                            per_expert[name] = _per_expert_parsed(
                                parsed,
                                outcome=outcome,
                                allow_flat_fallback=allow_flat_fallback,
                            )
                            per_outcome_expert_deltas[name].append(per_expert[name])
                            if sample is None or sample.error:
                                errors[name] = sample.error if sample is not None else "missing"
                                continue
                            delta = _parse_delta(
                                parsed,
                                outcome=outcome,
                                allow_flat_fallback=allow_flat_fallback,
                            )
                            if delta is None:
                                errors[name] = "invalid delta_raw"
                                continue
                            deltas.append(delta)

                        prediction_scalar: float | None = None
                        if len(deltas) == len(expert_names):
                            prediction_scalar = _clip_prediction(
                                float(baseline_y0[outcome]) + sum(deltas),
                                outcome_type=outcome_type,
                                ordinal_levels=ordinal_levels or None,
                            )
                            per_outcome_y.append(prediction_scalar)

                        sample_entry = {
                            "baseline_y0": float(baseline_y0[outcome]),
                            "delta_sum": sum(deltas) if deltas else None,
                            "prediction_scalar": prediction_scalar,
                            "expert_deltas": per_expert,
                        }
                        if errors:
                            sample_entry["error"] = errors
                        outcome_samples.append(sample_entry)

                    meta = dict(row.get("metadata") or {})
                    meta["outcome"] = outcome
                    by_outcome.append(
                        {
                            **base_payload,
                            "outcome": outcome,
                            "metadata": meta,
                            "baseline_y0": float(baseline_y0[outcome]),
                            "expert_deltas": per_outcome_expert_deltas,
                            "y_pred": _mean(per_outcome_y),
                            "samples": outcome_samples,
                        }
                    )
                return {"multi_outcome": by_outcome}

            per_expert_deltas: dict[str, list[dict[str, Any] | None]] = {name: [] for name in expert_names}
            samples: list[dict] = []
            y_values: list[float] = []
            for sample_idx in range(int(args.k)):
                per_expert: dict[str, dict[str, Any] | None] = {}
                errors: dict[str, str] = {}
                deltas: list[float] = []
                allow_flat_fallback = False
                for expert_idx, name in enumerate(expert_names):
                    sample = batches[expert_idx][sample_idx] if sample_idx < len(batches[expert_idx]) else None
                    parsed = sample.parsed if sample is not None else None
                    per_expert[name] = _per_expert_parsed(
                        parsed,
                        outcome=None,
                        allow_flat_fallback=allow_flat_fallback,
                    )
                    per_expert_deltas[name].append(per_expert[name])
                    if sample is None or sample.error:
                        errors[name] = sample.error if sample is not None else "missing"
                        continue
                    delta = _parse_delta(
                        parsed,
                        outcome=None,
                        allow_flat_fallback=allow_flat_fallback,
                    )
                    if delta is None:
                        errors[name] = "invalid delta_raw"
                        continue
                    deltas.append(delta)

                prediction_scalar: float | None = None
                if len(deltas) == len(expert_names):
                    prediction_scalar = _clip_prediction(
                        float(baseline_y0) + sum(deltas),
                        outcome_type=outcome_type,
                        ordinal_levels=ordinal_levels or None,
                    )
                    y_values.append(prediction_scalar)

                sample_entry = {
                    "baseline_y0": float(baseline_y0),
                    "delta_sum": sum(deltas) if deltas else None,
                    "prediction_scalar": prediction_scalar,
                    "expert_deltas": per_expert,
                }
                if errors:
                    sample_entry["error"] = errors
                samples.append(sample_entry)

            return {
                **base_payload,
                "baseline_y0": float(baseline_y0),
                "expert_deltas": per_expert_deltas,
                "y_pred": _mean(y_values),
                "samples": samples,
            }

        total = len(rows)
        # Heuristic batch size: keep task creation bounded. Even with a global semaphore,
        # scheduling every row at once can peg CPU and look "hung".
        batch_size = 50
        if args.max_concurrency is not None:
            denom = max(1, int(args.k) * max(1, max(len(EXPERT_PROMPTS), len(BELIEF_UPDATE_EXPERT_PROMPTS))))
            batch_size = max(1, int(args.max_concurrency) // denom)
            batch_size = min(200, batch_size)

        flattened: list[dict] = []
        for start in range(0, total, batch_size):
            end = min(total, start + batch_size)
            logger.info("multi_expert progress: rows={}/{} batch_size={}", end, total, batch_size)
            results = await asyncio.gather(*[_run_one(i, row) for i, row in enumerate(rows[start:end], start=start)])
            for item in results:
                if isinstance(item, dict) and "multi_outcome" in item:
                    multi = item.get("multi_outcome")
                    if isinstance(multi, list):
                        flattened.extend(multi)
                    continue
                flattened.append(item)
        return flattened

    out_rows = asyncio.run(_run_all())
    _write_jsonl(out_path, out_rows)
    raw_conversations, row_level = _build_row_level_outputs(
        out_rows=out_rows,
        model_name=str(args.model),
        backend=str(args.backend),
        run_id=run_id,
        calibration_status="uncalibrated",
        raw_conversations_path=raw_conversations_path,
        prompts_path=Path(args.in_path),
        samples_path=out_path,
        base_dir=base_dir,
    )
    if extra_raw_conversations:
        raw_conversations.extend(extra_raw_conversations)
    if raw_conversations:
        _write_jsonl(raw_conversations_path, raw_conversations)
    if row_level:
        row_level_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(row_level).to_csv(row_level_path, index=False)
    return
