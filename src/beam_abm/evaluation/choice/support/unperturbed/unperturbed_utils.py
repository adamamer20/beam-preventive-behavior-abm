"""Shared helpers for unperturbed microvalidation pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from beam_abm.empirical.plan_io import extract_predictors_from_model_plan
from beam_abm.evaluation.prompting import PromptTemplateBuilder

Quantiles = tuple[float, ...]
PromptFamily = str

SUPPORTED_PROMPT_FAMILIES: set[PromptFamily] = {
    "data_only",
    "first_person",
    "third_person",
    "multi_expert",
}


def _stable_str_sort_key(x: Any) -> str:
    return str(x)


def parse_list_arg(value: str | None) -> list[str]:
    if value is None:
        return []
    parts = [p.strip() for p in value.split(",")]
    return [p for p in parts if p]


def parse_prompt_families(single: str | None, multi: str | None) -> list[PromptFamily]:
    default_multi = ",".join(sorted(SUPPORTED_PROMPT_FAMILIES))
    if single and multi and multi.strip() != default_multi:
        raise ValueError("Provide only one of --prompt-family or --prompt-families")
    if single:
        families = [single.strip()]
    else:
        families = parse_list_arg(multi) or ["data_only", "first_person", "multi_expert"]
    unknown = [f for f in families if f not in SUPPORTED_PROMPT_FAMILIES]
    if unknown:
        raise ValueError(f"Unsupported prompt families: {unknown}")
    return families


def load_lever_config(path: Path) -> dict[str, Any]:
    obj: Any = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise TypeError(f"Expected object in lever config: {path}")
    return obj


def load_outcome_types(path: Path) -> dict[str, str]:
    ct = pd.read_csv(path, sep="\t")
    if "column" not in ct.columns or "type" not in ct.columns:
        raise ValueError(f"Column types file missing required columns: {path}")
    return {str(k): str(v).strip().lower() for k, v in ct.set_index("column")["type"].to_dict().items()}


def fmt_num(x: float) -> str:
    if float(x).is_integer():
        return str(int(x))
    return f"{x:.3f}".rstrip("0").rstrip(".")


def spearman(x: np.ndarray, y: np.ndarray) -> float | None:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2:
        return None
    sx = pd.Series(x[mask]).rank(method="average").to_numpy(dtype=float)
    sy = pd.Series(y[mask]).rank(method="average").to_numpy(dtype=float)
    vx = sx - sx.mean()
    vy = sy - sy.mean()
    denom = float(np.sqrt((vx * vx).sum()) * np.sqrt((vy * vy).sum()))
    if denom <= 0.0:
        return None
    return float((vx * vy).sum() / denom)


def auc(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    """Compute ROC AUC without sklearn.

    Uses the rank-sum / Mann–Whitney formulation.
    Returns None if AUC is undefined (all positives or all negatives).
    """

    mask = np.isfinite(y_true) & np.isfinite(y_score)
    if mask.sum() < 2:
        return None
    y = y_true[mask].astype(int)
    s = y_score[mask].astype(float)
    if len(np.unique(y)) < 2:
        return None

    ranks = pd.Series(s).rank(method="average").to_numpy(dtype=float)
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return None
    sum_ranks_pos = float(ranks[y == 1].sum())
    u = sum_ranks_pos - n_pos * (n_pos + 1) / 2.0
    return float(u / (n_pos * n_neg))


def quantile_summary(arr: np.ndarray, q: Quantiles) -> dict[str, float | None]:
    mask = np.isfinite(arr)
    if mask.sum() == 0:
        return {f"q{int(qq * 100):02d}": None for qq in q} | {"mean": None}
    out: dict[str, float | None] = {"mean": float(np.mean(arr[mask]))}
    for qq in q:
        out[f"q{int(qq * 100):02d}"] = float(np.quantile(arr[mask], qq))
    return out


def metrics_filename(*, family: str, pred_col: str) -> str:
    if pred_col == "y_pred_cal":
        return f"metrics_summary_calibrated__{family}.csv"
    return f"metrics_summary__{family}.csv"


def stratified_sample_indices(
    df: pd.DataFrame,
    *,
    strata_col: str,
    n: int,
    seed: int,
) -> list[int]:
    if strata_col not in df.columns:
        raise KeyError(f"Missing strata column: {strata_col}")

    groups = {k: v.index.to_list() for k, v in df.groupby(strata_col, dropna=False)}
    keys = sorted(groups.keys(), key=_stable_str_sort_key)
    if not keys:
        return []

    rng = np.random.default_rng(int(seed))
    base = n // len(keys)
    rem = n % len(keys)

    out: list[int] = []
    for i, k in enumerate(keys):
        want = base + (1 if i < rem else 0)
        idxs = groups[k]
        if not idxs or want <= 0:
            continue
        if len(idxs) >= want:
            pick = rng.choice(idxs, size=want, replace=False)
        else:
            pick = rng.choice(idxs, size=want, replace=True)
        out.extend(int(x) for x in pick)

    rng.shuffle(out)
    return out[:n]


def build_prompts(
    *,
    df: pd.DataFrame,
    indices: list[int],
    outcome_key: str,
    predictor_cols: list[str],
    outcome_label: str,
    outcome_type: str,
    bounds: tuple[float, float] | None,
    endpoints: tuple[str, str] | None,
    col_types: dict[str, str],
    display_labels: dict[str, str],
    clean_questions: dict[str, str],
    clean_answers: dict[str, str],
    possible_bounds: dict[str, tuple[float, float]],
    ordinal_endpoints: dict[str, tuple[str, str]],
    prompt_family: PromptFamily,
    prompt_builder: PromptTemplateBuilder,
) -> tuple[list[list[dict[str, str]]], np.ndarray, list[str], list[int], list[dict[str, object]]]:
    messages_list: list[list[dict[str, str]]] = []
    observed: list[float] = []
    countries: list[str] = []
    row_indices: list[int] = []
    attributes_list: list[dict[str, object]] = []

    _ = outcome_label, outcome_type, bounds, endpoints

    for idx in indices:
        row = df.loc[idx]
        y = row.get(outcome_key)
        if pd.isna(y):
            continue
        try:
            y_f = float(y)
        except (TypeError, ValueError):
            continue

        country = row.get("country")
        country_s = "" if pd.isna(country) else str(country)

        attrs: dict[str, object] = {col: row.get(col) for col in predictor_cols if col in df.columns}
        if "country" in df.columns and "country" not in attrs:
            attrs["country"] = row.get("country")

        context_line = prompt_builder.maybe_survey_context(attrs)
        attr_lines = prompt_builder.format_attr_lines(attrs, predictor_cols, scale_seen=set())

        def _attribute_block(context_line=context_line, attr_lines=attr_lines) -> str:
            human_lines = []
            if context_line:
                human_lines.append(context_line)
            human_lines.append("Attributes:")
            human_lines.extend(attr_lines)
            return "\n".join(human_lines)

        if prompt_family == "data_only":
            system_prompt = (
                f"{prompt_builder.psychometrician_system(outcome_key).strip()}\n\n"
                "Fields not listed are unknown; treat them as unknown (do not assume typical values)."
            )
            human_prompt = _attribute_block()
            messages = [
                {"from": "system", "value": system_prompt},
                {"from": "human", "value": human_prompt},
            ]
        elif prompt_family == "first_person":
            narrative = prompt_builder.build_narrative(attrs, predictor_cols, "first_person")
            messages = [
                {"from": "system", "value": prompt_builder.first_person_system_prompt()},
                {"from": "assistant", "value": narrative},
                {"from": "human", "value": prompt_builder.prediction_user_prompt_first_person(outcome_key)},
            ]
        elif prompt_family == "third_person":
            narrative = prompt_builder.build_narrative(attrs, predictor_cols, "third_person")
            messages = [
                {"from": "system", "value": prompt_builder.psychometrician_system(outcome_key).strip()},
                {"from": "human", "value": narrative},
            ]
        elif prompt_family == "multi_expert":
            system_prompt = prompt_builder.psychometrician_system(outcome_key).strip()
            human_prompt = _attribute_block()
            messages = [
                {"from": "system", "value": system_prompt},
                {"from": "human", "value": human_prompt},
            ]
        else:
            raise ValueError(f"Unknown prompt family: {prompt_family}")

        messages_list.append(messages)
        observed.append(y_f)
        countries.append(country_s)
        row_indices.append(int(idx))
        attributes_list.append(attrs)

    return messages_list, np.asarray(observed, dtype=float), countries, row_indices, attributes_list


def compute_metrics_rows(
    df: pd.DataFrame,
    *,
    pred_col: str,
) -> list[dict[str, Any]]:
    q: Quantiles = (0.10, 0.25, 0.50, 0.75, 0.90)
    rows: list[dict[str, Any]] = []

    if pred_col not in df.columns:
        raise KeyError(f"Prediction column not found: {pred_col}")
    if "y_true" not in df.columns:
        raise KeyError("Missing y_true column")

    group_cols = ["model", "outcome", "outcome_type"]
    for (model, outcome, outcome_type), g in df.groupby(group_cols, dropna=False):
        y_true = g["y_true"].to_numpy(dtype=float)
        y_pred = g[pred_col].to_numpy(dtype=float)
        if outcome_type == "binary":
            y_bin = y_true.astype(int)
            mask = np.isfinite(y_pred) & np.isfinite(y_bin)
            brier = float(np.mean((y_pred[mask] - y_bin[mask]) ** 2)) if mask.sum() else None
            auc_val = auc(y_true=y_bin.astype(float), y_score=y_pred)
            rows.append(
                {
                    "model": model,
                    "outcome": outcome,
                    "type": outcome_type,
                    "n": int(mask.sum()),
                    "brier": brier,
                    "auc": auc_val,
                }
            )
            continue

        mask = np.isfinite(y_pred) & np.isfinite(y_true)
        mae = float(np.mean(np.abs(y_pred[mask] - y_true[mask]))) if mask.sum() else None
        rmse = float(np.sqrt(np.mean((y_pred[mask] - y_true[mask]) ** 2))) if mask.sum() else None
        spear = spearman(y_pred, y_true)
        pred_q = quantile_summary(y_pred, q=q)
        true_q = quantile_summary(y_true, q=q)
        rows.append(
            {
                "model": model,
                "outcome": outcome,
                "type": outcome_type,
                "n": int(mask.sum()),
                "mae": mae,
                "rmse": rmse,
                "spearman": spear,
                **{f"pred_{k}": v for k, v in pred_q.items()},
                **{f"true_{k}": v for k, v in true_q.items()},
            }
        )
    return rows


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, default=_json_default) + "\n")


def build_predictor_cols(
    *,
    outcome_key: str,
    lever_cfg: dict[str, Any],
    model_plan: Path,
) -> list[str]:
    entry = lever_cfg.get(outcome_key)
    if entry is None:
        raise KeyError(f"Outcome not found in lever config: {outcome_key}")
    if isinstance(entry, list):
        return [str(c) for c in entry]
    if isinstance(entry, dict):
        model_ref = entry.get("model_ref")
        cols_obj = entry.get("predictors") or entry.get("driver_cols") or entry.get("attributes")
        if isinstance(model_ref, str) and model_ref.strip():
            return extract_predictors_from_model_plan(
                model_plan_path=model_plan,
                model_name=model_ref.strip(),
                drop_suffixes=("_missing",),
                drop_cols=("country",),
            )
        if isinstance(cols_obj, list) and cols_obj:
            return [str(c) for c in cols_obj]
        raise ValueError(f"Unsupported lever-config entry for {outcome_key}: {entry}")
    raise ValueError(f"Unsupported lever-config entry type for {outcome_key}: {type(entry)}")
