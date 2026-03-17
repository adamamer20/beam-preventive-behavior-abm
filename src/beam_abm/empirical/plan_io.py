"""Helpers for reading `model_plan.json` artifacts.

The modeling pipeline writes a materialized plan (e.g.
`empirical/output/modeling/model_plan.json`) which is convenient for
downstream consumers like microvalidation. This module provides small, robust
utilities for extracting predictor lists from that JSON.
"""

from __future__ import annotations

import json
from pathlib import Path


def _unique_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _iter_plan_entries(model_plan: object) -> list[dict]:
    if isinstance(model_plan, list):
        return [x for x in model_plan if isinstance(x, dict)]
    if isinstance(model_plan, dict):
        models = model_plan.get("models") or model_plan.get("plan")
        if isinstance(models, list):
            return [x for x in models if isinstance(x, dict)]
        # Fall back: some exports might be a dict-of-dicts keyed by name.
        if isinstance(models, dict):
            return [x for x in models.values() if isinstance(x, dict)]
        return [model_plan]
    return []


def extract_predictors_from_model_plan(
    *,
    model_plan_path: Path,
    model_name: str,
    drop_suffixes: tuple[str, ...] = ("_missing",),
    drop_cols: tuple[str, ...] = ("country",),
) -> list[str]:
    """Extract flattened predictors for a named model from `model_plan.json`.

    Parameters
    ----------
    model_plan_path:
        Path to a `model_plan.json` file produced by the modeling pipeline.
    model_name:
        The `model` name inside the JSON (e.g. `B_PROTECT_mask_when_pressure_high_ENGINES_REF`).
    drop_suffixes:
        Column name suffixes to exclude (e.g. structural missingness indicators).
    drop_cols:
        Exact column names to exclude (e.g. "country" is handled separately via add_country).
    """
    if not model_plan_path.exists():
        raise FileNotFoundError(f"model_plan.json not found: {model_plan_path}")
    raw = json.loads(model_plan_path.read_text(encoding="utf-8"))
    entries = _iter_plan_entries(raw)
    spec = next((e for e in entries if str(e.get("model") or e.get("name") or "") == model_name), None)
    if spec is None:
        raise KeyError(f"Model not found in {model_plan_path}: {model_name}")

    predictors = spec.get("predictors")
    if not isinstance(predictors, dict):
        raise ValueError(f"Model spec missing predictors dict: {model_name}")

    out: list[str] = []
    for _, cols in predictors.items():
        if not isinstance(cols, list):
            continue
        for c in cols:
            col = str(c)
            if col in drop_cols:
                continue
            if any(col.endswith(suf) for suf in drop_suffixes):
                continue
            out.append(col)
    return _unique_preserve_order(out)
