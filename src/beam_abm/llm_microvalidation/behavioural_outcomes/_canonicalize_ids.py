from __future__ import annotations

import hashlib
import re
from typing import Any

_DATASET_ID_RE = re.compile(r"^(?P<outcome>.+?)__(?:(?:multi|single)__)?(?P<row_index>\d+)(?:__\d+)?$")


def normalize_choice_dataset_id(value: object) -> str | None:
    """Normalize historical dataset_id conventions to a canonical key.

    Canonical choice-validation dataset id is:
      <outcome>__<row_index>

    Historical exports have included:
      - <outcome>__<row_index>__<replicate>
      - <outcome>__multi__<row_index>__<replicate>
      - <outcome>__single__<row_index>__<replicate>

    This normalizer only rewrites when it can *safely* infer (outcome,row_index).
    IDs without an outcome prefix (e.g. multi__<row_index>__<i>) cannot be
    normalized without additional metadata.
    """

    if not isinstance(value, str):
        return None
    s = value.strip()
    if not s:
        return None
    m = _DATASET_ID_RE.match(s)
    if not m:
        return None
    outcome = m.group("outcome").strip()
    row_raw = m.group("row_index")
    try:
        row_index = int(row_raw)
    except ValueError:
        return None
    if not outcome:
        return None
    return f"{outcome}__{row_index}"


def normalize_model_slug(model: str | None, *, reasoning_effort: str | None = None) -> str:
    """Normalize model identifiers for pathing.

    Important: Do not drop provider prefixes (e.g., "openai/") if present.
    We keep them as part of the slug (with '/' sanitized) so output trees
    can disambiguate identical base model names across providers.
    """

    raw = (model or "").strip()
    if not raw:
        return "unknown_model"

    # Preserve explicit effort suffixes already encoded in the model string.
    # This matters when callers pass directory tokens like "openai_gpt-oss-20b_high"
    # without separately providing reasoning_effort.
    inferred_effort: str | None = None
    raw_base = raw
    for suffix in ("_high", "_low", "_medium"):
        if raw_base.endswith(suffix):
            inferred_effort = suffix.lstrip("_")
            raw_base = raw_base[: -len(suffix)]
            break
    if not reasoning_effort and inferred_effort:
        reasoning_effort = inferred_effort

    lowered = raw_base.lower()

    # Project convention: treat OpenAI GPT-OSS identifiers as the same underlying
    # model as local/vLLM GPT-OSS identifiers. Collapse provider prefixes.
    if (lowered.startswith("openai/") or lowered.startswith("openai_")) and "gpt-oss-20b" in lowered:
        base = "gpt-oss-20b"
    elif (lowered.startswith("openai/") or lowered.startswith("openai_")) and "gpt-oss-120b" in lowered:
        base = "gpt-oss-120b"
    else:
        # If a provider prefix is included (e.g., "openai/gpt-oss-20b"), preserve
        # it in the slug while keeping things filesystem-safe.
        if "/" in raw_base:
            base = _safe_token(raw_base)
        # For bare model names, keep stable slugs.
        elif "gpt-oss-20b" in lowered:
            base = "gpt-oss-20b"
        elif "gpt-oss-120b" in lowered:
            base = "gpt-oss-120b"
        else:
            base = _safe_token(raw_base)

    effort = (reasoning_effort or "").strip().lower()
    # Project policy: GPT-OSS 120B uses medium effort in this pipeline.
    if base == "gpt-oss-120b" and effort == "high":
        effort = "medium"
    # Preserve historical default behavior for "medium" (or absent) effort.
    if effort and effort != "medium":
        base = f"{base}_{_safe_token(effort)}"

    return base


def _safe_token(value: object) -> str:
    text = str(value) if value is not None else ""
    out = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in text).strip("_")
    return out or "unknown"


def _row_dataset_id(row: dict[str, Any]) -> str:
    # Prefer a stable (outcome, respondent_row_index) identifier when available.
    # Canonical leaves should use a single consistent key so rows can be joined
    # across prompt families/strategies.
    meta = row.get("metadata")
    if isinstance(meta, dict):
        row_index = meta.get("row_index")
        if row_index is not None:
            try:
                row_index_int = int(row_index)
            except (TypeError, ValueError):
                row_index_int = None
            if row_index_int is not None:
                outcome = row.get("outcome")
                if not (isinstance(outcome, str) and outcome.strip()):
                    outcome = meta.get("outcome")
                if not (isinstance(outcome, str) and outcome.strip()):
                    rid = row.get("dataset_id") or row.get("id")
                    if isinstance(rid, str) and rid.strip() and "__" in rid:
                        outcome = rid.split("__", 1)[0].strip() or None
                if isinstance(outcome, str) and outcome.strip():
                    return f"{outcome.strip()}__{row_index_int}"

    raw_dataset_id = row.get("dataset_id")
    normalized = normalize_choice_dataset_id(raw_dataset_id)
    if normalized is not None:
        return normalized
    if isinstance(raw_dataset_id, str) and raw_dataset_id.strip():
        return raw_dataset_id.strip()

    raw_id = row.get("id")
    normalized = normalize_choice_dataset_id(raw_id)
    if normalized is not None:
        return normalized
    if isinstance(raw_id, str) and raw_id.strip():
        return raw_id.strip()

    if isinstance(meta, dict):
        outcome = meta.get("outcome")
        row_index = meta.get("row_index")
        if outcome and row_index is not None:
            return f"{outcome}__{row_index}"
    raise ValueError("Unable to infer dataset_id for samples row")


def _sample_dedupe_id(
    *,
    row: dict[str, Any],
    sample: dict[str, Any],
    sample_idx: int,
) -> str:
    # Preferred stable id.
    conv_id = sample.get("conversation_id")
    if isinstance(conv_id, str) and conv_id.strip():
        return conv_id.strip()

    prompt_id = row.get("prompt_id")
    raw_text = sample.get("raw_text_original") or sample.get("raw_text")
    scalar = sample.get("prediction_scalar")
    payload = f"{_row_dataset_id(row)}::{prompt_id}::{sample_idx}::{raw_text}::{scalar}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
