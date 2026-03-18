"""Compute post-hoc diagnostics for unperturbed choice-validation outputs.

Reads canonical prediction leaves from:
  evaluation/output/choice_validation/unperturbed/<outcome>/<model>/<strategy>/{uncalibrated,calibrated}/predictions*.csv

Writes:
- per-leaf posthoc_metrics.csv next to the predictions file
- global summary at <output_root>/_summary/posthoc_metrics_all.csv

This is safe to run repeatedly.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from beam_abm.evaluation.choice.posthoc import (
    compute_slice_spearman,
    compute_y_decile_errors_from_df,
    summarize_choice_validation_posthoc_metrics,
)
from beam_abm.evaluation.common.calibration import fit_binary_platt_country, fit_isotonic_country
from beam_abm.evaluation.common.metrics import gini_from_cindex

_DEFAULT_REF_MODEL_BY_OUTCOME: dict[str, str] = {
    "vax_willingness_Y": "B_COVID_MAIN_vax_willingness_Y_ENGINES_STAKESdanger",
    "mask_when_pressure_high": "B_PROTECT_mask_when_pressure_high_ENGINES_REF",
    "stay_home_when_symptomatic": "B_PROTECT_stay_home_when_symptomatic_ENGINES_REF",
}

# For posthoc reference benchmarking we report flu on the no-habit basis.
_FORCED_REF_MODEL_BY_OUTCOME: dict[str, str] = {
    "flu_vaccinated_2023_2024": "B_FLU_vaccinated_2023_2024_ENGINES",
}


def _safe_json_loads(raw: str) -> dict | None:
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def _extract_model_ref_map(payload: dict) -> dict[str, str]:
    out: dict[str, str] = {}
    for k, v in payload.items():
        outcome = str(k).strip()
        if not outcome:
            continue
        if isinstance(v, str) and v.strip():
            out[outcome] = v.strip()
            continue
        if isinstance(v, dict):
            model_ref = v.get("model_ref")
            if isinstance(model_ref, str) and model_ref.strip():
                out[outcome] = model_ref.strip()
    return out


def _pearson(x: pd.Series, y: pd.Series) -> float | None:
    """Pearson correlation with NaN-safe filtering (no SciPy)."""

    x2, y2 = x.align(y, join="inner")
    m = pd.notna(x2) & pd.notna(y2)
    if int(m.sum()) < 2:
        return None
    xv = x2[m].astype(float)
    yv = y2[m].astype(float)
    vx = xv - float(xv.mean())
    vy = yv - float(yv.mean())
    denom = float((vx.pow(2).sum() ** 0.5) * (vy.pow(2).sum() ** 0.5))
    if denom <= 0.0:
        return None
    return float((vx * vy).sum() / denom)


def _spearman_series(x: pd.Series, y: pd.Series) -> float | None:
    """Spearman correlation via rank-then-Pearson (no SciPy)."""

    x2, y2 = x.align(y, join="inner")
    m = pd.notna(x2) & pd.notna(y2)
    if int(m.sum()) < 2:
        return None
    rx = x2[m].astype(float).rank(method="average")
    ry = y2[m].astype(float).rank(method="average")
    return _pearson(rx, ry)


def _is_ignored_path_for_scan(p: Path, *, include_runs: bool) -> bool:
    if include_runs:
        return False
    ignored = {"_runs", "_locks", "_summary"}
    return any(part in ignored for part in p.parts)


def _infer_status(pred_path: Path) -> str:
    name = pred_path.name
    if "calibrated" in name:
        return "calibrated"
    # Fall back to parent directory name.
    if pred_path.parent.name in {"calibrated", "uncalibrated"}:
        return pred_path.parent.name
    return "uncalibrated"


def _infer_pred_col(status: str) -> str:
    return "y_pred_cal" if status == "calibrated" else "y_pred"


def _leaf_metadata(df: pd.DataFrame) -> dict:
    # Prefer file-embedded metadata when present.
    meta: dict = {}
    for col in ["run_id", "model", "outcome", "outcome_type", "strategy", "prompt_family"]:
        if col in df.columns and not df.empty:
            val = df[col].iloc[0]
            meta[col] = val
    return meta


def _path_fallback_metadata(*, pred_path: Path, root: Path) -> dict:
    # Path structure is usually:
    #   <root>/<outcome>/<model>/<strategy>/{uncalibrated,calibrated}/predictions*.csv
    # Or within per-run folders:
    #   <root>/_runs/<run_id>/<outcome>/<model>/<strategy>/...
    outcome = None
    model = None
    strategy = None

    strategy_dir = (
        pred_path.parent.parent if pred_path.parent.name in {"calibrated", "uncalibrated"} else pred_path.parent
    )
    model_dir = strategy_dir.parent
    outcome_dir = model_dir.parent
    strategy = strategy_dir.name
    model = model_dir.name
    outcome = outcome_dir.name

    run_id = None
    try:
        rel = pred_path.relative_to(root)
    except ValueError:
        rel = None
    if rel is not None:
        parts = list(rel.parts)
        if "_runs" in parts:
            idx = parts.index("_runs")
            if idx + 1 < len(parts):
                run_id = parts[idx + 1]
        else:
            run_id = "canonical"

    return {
        "outcome": outcome,
        "model": model,
        "strategy": strategy,
        "run_id": run_id,
    }


def _fill_missing_metadata(*, meta: dict, fallback: dict) -> dict:
    out = dict(meta)
    for k, v in fallback.items():
        if k not in out:
            out[k] = v
            continue
        if out[k] is None:
            out[k] = v
            continue
        if isinstance(out[k], float) and pd.isna(out[k]):
            out[k] = v
            continue
        if isinstance(out[k], str) and out[k].strip() == "":
            out[k] = v
    return out


def _prefer_path_metadata(*, meta: dict, fallback: dict) -> dict:
    """Prefer on-disk leaf metadata over file-embedded metadata.

    For canonical summaries we want model identifiers that match the
    directory naming contract (e.g. including `_high` / `_low` suffixes).
    Prediction CSVs may embed a base model name (e.g. `openai/gpt-oss-20b`)
    that omits effort, so we explicitly take the path-derived values.
    """

    out = dict(meta)
    for key in ["outcome", "model", "strategy", "run_id"]:
        val = fallback.get(key)
        if val is None:
            continue
        if isinstance(val, float) and pd.isna(val):
            continue
        out[key] = val
    return out


def _empty_posthoc_summary(*, slice_col: str) -> dict:
    return {
        "n": 0,
        "pooled_gini": None,
        "mae_skill_cal": None,
        "tail_bias_bottom_decile_mean_error": None,
        "tail_bias_top_decile_mean_error": None,
        "dispersion_ratio_iqr": None,
        "dispersion_ratio_q90q10": None,
        "slice_col": slice_col,
        "n_slices": 0,
        "n_slices_valid": 0,
        "slice_spearman_mean": None,
        "slice_spearman_median": None,
        "slice_spearman_min": None,
        "slice_spearman_p25": None,
        "slice_spearman_p75": None,
        "slice_spearman_max": None,
        "slice_gini_mean": None,
        "slice_gini_median": None,
        "slice_gini_min": None,
        "slice_gini_p25": None,
        "slice_gini_p75": None,
        "slice_gini_max": None,
        # Optional: anchor-neighborhood slice stability (filled when enabled + available).
        "anchor_neighborhoods_path": None,
        "anchor_neighborhood_max_rank": None,
        "anchor_neighborhood_n_slices": 0,
        "anchor_neighborhood_n_slices_valid": 0,
        "anchor_neighborhood_spearman_mean": None,
        "anchor_neighborhood_spearman_median": None,
        "anchor_neighborhood_spearman_min": None,
        "anchor_neighborhood_spearman_p25": None,
        "anchor_neighborhood_spearman_p75": None,
        "anchor_neighborhood_spearman_max": None,
    }


def _extract_row_id(dataset_id: object) -> int | None:
    """Parse a numeric row_id out of dataset_id strings.

    Prefer the canonicalized key (<outcome>__<row_index>) and then extract
    row_index. This supports legacy on-disk ids while keeping the rest of
    the pipeline centered on a single convention.
    """

    from beam_abm.evaluation.choice._canonicalize_ids import normalize_choice_dataset_id

    s = dataset_id.strip() if isinstance(dataset_id, str) else ""
    if not s:
        return None

    norm = normalize_choice_dataset_id(s)
    if isinstance(norm, str) and norm:
        s = norm

    if "__" not in s:
        return None
    parts = s.split("__")
    if len(parts) >= 2 and parts[1].isdigit():
        return int(parts[1])
    return None


def _spearman_np(x: np.ndarray, y: np.ndarray) -> float | None:
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 2:
        return None
    sx = pd.Series(x[mask]).rank(method="average").to_numpy(dtype=float)
    sy = pd.Series(y[mask]).rank(method="average").to_numpy(dtype=float)
    vx = sx - float(np.mean(sx))
    vy = sy - float(np.mean(sy))
    denom = float(np.sqrt(float(np.sum(vx * vx))) * np.sqrt(float(np.sum(vy * vy))))
    if denom <= 0.0:
        return None
    return float(np.sum(vx * vy) / denom)


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


def _iter_canonical_uncalibrated_predictions_csv(*, root: Path, outcome: str) -> list[Path]:
    outcome_dir = root / outcome
    if not outcome_dir.exists():
        return []

    candidates: list[Path] = []
    for p in outcome_dir.rglob("predictions.csv"):
        if "_runs" in p.parts or "_summary" in p.parts or "_locks" in p.parts:
            continue
        if p.parent.name != "uncalibrated":
            continue
        candidates.append(p)
    return sorted(candidates)


def compute_reference_benchmark(
    *,
    output_root: Path,
    coef_path: Path,
    model_plan: Path,
    column_types: Path,
    input_csv: Path,
    missingness_threshold: float,
    standardize: bool,
    ref_model_map: Path | None,
    lever_config: Path | None,
    out: Path | None,
) -> Path:
    """Compute and write the reference-model benchmark correlation summary.

    Returns the written CSV path.
    """

    # Lazy imports: this keeps the base posthoc run lightweight when the
    # reference benchmark is disabled.
    from patsy.highlevel import dmatrices
    from statsmodels.miscmodels.ordinal_model import OrderedModel

    from beam_abm.empirical.io import load_column_types_with_reference
    from beam_abm.empirical.missingness import apply_missingness_plan_pd, build_missingness_plan_pd
    from beam_abm.empirical.modeling import ModelBlock, ModelSpec, _prepare_formula, standardize_columns

    def _mae_skill_np(y_true: np.ndarray, y_pred: np.ndarray) -> float | None:
        m = np.isfinite(y_true) & np.isfinite(y_pred)
        if int(m.sum()) == 0:
            return None
        yt = y_true[m]
        yp = y_pred[m]
        mae = float(np.mean(np.abs(yp - yt)))
        baseline = float(np.median(yt))
        baseline_mae = float(np.mean(np.abs(yt - baseline)))
        if baseline_mae <= 0:
            return None
        return float(1.0 - mae / baseline_mae)

    def _ci_np(arr: np.ndarray, alpha: float = 0.05) -> tuple[float | None, float | None]:
        a = arr[np.isfinite(arr)]
        if a.size == 0:
            return (None, None)
        lo = float(np.quantile(a, alpha / 2.0))
        hi = float(np.quantile(a, 1.0 - alpha / 2.0))
        return (lo, hi)

    if not output_root.exists():
        raise FileNotFoundError(f"Missing output root: {output_root}")
    if not coef_path.exists():
        raise FileNotFoundError(f"Missing coefficients CSV: {coef_path}")
    if not model_plan.exists():
        raise FileNotFoundError(f"Missing model plan JSON: {model_plan}")
    if not column_types.exists():
        raise FileNotFoundError(f"Missing column types TSV: {column_types}")
    if not input_csv.exists():
        raise FileNotFoundError(f"Missing input CSV: {input_csv}")

    def _load_model_plan_entries(model_plan_path: Path) -> list[dict[str, object]]:
        raw = json.loads(model_plan_path.read_text(encoding="utf-8"))
        if isinstance(raw, list):
            return [x for x in raw if isinstance(x, dict)]
        if isinstance(raw, dict):
            models = raw.get("models") or raw.get("plan")
            if isinstance(models, list):
                return [x for x in models if isinstance(x, dict)]
            if isinstance(models, dict):
                return [x for x in models.values() if isinstance(x, dict)]
            return [raw]
        return []

    def _spec_from_model_plan_entry(entry: dict[str, object]) -> ModelSpec:
        name = str(entry.get("model") or entry.get("name") or "").strip()
        if not name:
            raise ValueError("model_plan entry missing model/name")
        outcome = str(entry.get("outcome") or "").strip()
        if not outcome:
            raise ValueError(f"model_plan entry missing outcome for {name}")
        model_type = str(entry.get("model_type") or entry.get("type") or "").strip().lower()
        if model_type not in {"linear", "logit", "ordinal"}:
            raise ValueError(f"Unsupported/missing model_type for {name}: {model_type!r}")
        predictors = entry.get("predictors")
        if not isinstance(predictors, dict):
            raise ValueError(f"model_plan entry missing predictors dict for {name}")

        blocks: list[ModelBlock] = []
        for block_name, cols in predictors.items():
            if not isinstance(cols, list):
                continue
            block_predictors = tuple(str(c) for c in cols if str(c).strip())
            if not block_predictors:
                continue
            blocks.append(ModelBlock(name=str(block_name), predictors=block_predictors))

        interactions_raw = entry.get("interactions")
        interactions: list[tuple[str, ...]] = []
        if isinstance(interactions_raw, list):
            for inter in interactions_raw:
                if isinstance(inter, list) and inter:
                    interactions.append(tuple(str(x) for x in inter if str(x).strip()))

        add_country = bool(entry.get("add_country", True))
        desc = entry.get("description")
        group = entry.get("group")
        return ModelSpec(
            name=name,
            outcome=outcome,
            model_type=model_type,  # type: ignore[arg-type]
            blocks=tuple(blocks),
            interactions=tuple(interactions),
            add_country=add_country,
            description=str(desc) if isinstance(desc, str) else None,
            group=str(group) if isinstance(group, str) else None,
        )

    ref_model_by_outcome = dict(_DEFAULT_REF_MODEL_BY_OUTCOME)
    if lever_config is not None and lever_config.exists():
        lever_mapping = _safe_json_loads(lever_config.read_text(encoding="utf-8"))
        if lever_mapping is not None:
            ref_model_by_outcome.update(_extract_model_ref_map(lever_mapping))
    if ref_model_map is not None:
        mapping = _safe_json_loads(ref_model_map.read_text(encoding="utf-8"))
        if mapping is None:
            raise ValueError("--ref-model-map must be a JSON object mapping outcome -> model")
        ref_model_by_outcome.update(_extract_model_ref_map(mapping))

    # Enforce stable no-habit flu reference in posthoc benchmark outputs.
    ref_model_by_outcome.update(_FORCED_REF_MODEL_BY_OUTCOME)

    coef_all = pd.read_csv(coef_path)
    if "model" not in coef_all.columns or "term" not in coef_all.columns or "estimate" not in coef_all.columns:
        raise ValueError(f"Coefficient table missing required columns: {coef_path}")

    plan_entries = _load_model_plan_entries(model_plan)
    plan_by_name = {str(e.get("model") or e.get("name") or "").strip(): e for e in plan_entries if isinstance(e, dict)}

    needed_ref_models = sorted({str(m).strip() for m in ref_model_by_outcome.values() if str(m).strip()})
    specs_by_model: dict[str, ModelSpec] = {}
    for model_name in needed_ref_models:
        entry = plan_by_name.get(model_name)
        if entry is None:
            raise KeyError(f"Reference model {model_name!r} not found in {model_plan}")
        specs_by_model[model_name] = _spec_from_model_plan_entry(entry)

    col_types, reference_levels = load_column_types_with_reference(column_types)
    df_base = pd.read_csv(input_csv)

    predict_cols: set[str] = set()
    for spec in specs_by_model.values():
        for col in spec.predictor_list():
            if col != "country":
                predict_cols.add(col)
    predict_cols = {c for c in predict_cols if c in df_base.columns}

    plan = build_missingness_plan_pd(
        df_base,
        predictors=sorted(predict_cols),
        threshold=float(missingness_threshold),
        column_types=col_types,
        country_col="country" if "country" in df_base.columns else None,
    )

    df_proc = df_base
    if standardize and predict_cols:
        df_proc = standardize_columns(df_proc, sorted(predict_cols), col_types)
    df_proc = apply_missingness_plan_pd(
        df_proc,
        plan,
        standardize=standardize,
        country_col="country" if "country" in df_proc.columns else None,
    )

    out_rows: list[dict[str, object]] = []
    for outcome, ref_model in sorted(ref_model_by_outcome.items(), key=lambda kv: kv[0]):
        pred_candidates = _iter_canonical_uncalibrated_predictions_csv(root=output_root, outcome=outcome)
        if not pred_candidates:
            out_rows.append(
                {
                    "outcome": outcome,
                    "ref_model": ref_model,
                    "n": 0,
                    "eta_std": float("nan"),
                    "term_in_beta": float("nan"),
                    "term_in_design": float("nan"),
                    "term_missing_in_beta": float("nan"),
                    "ref_spearman_eta": float("nan"),
                    "ref_gini_eta": float("nan"),
                    "ref_spearman_pred": float("nan"),
                    "ref_gini_pred": float("nan"),
                    "ref_skill_pred": float("nan"),
                    "ref_gini_pred_ci_low": float("nan"),
                    "ref_gini_pred_ci_high": float("nan"),
                    "ref_skill_pred_ci_low": float("nan"),
                    "ref_skill_pred_ci_high": float("nan"),
                    "predictions_paths_n": 0,
                }
            )
            continue

        y_true_rows: list[pd.DataFrame] = []
        for pred_path in pred_candidates:
            try:
                leaf_df = pd.read_csv(pred_path, usecols=["dataset_id", "y_true"])
            except (OSError, UnicodeDecodeError, pd.errors.EmptyDataError, pd.errors.ParserError):
                continue
            if leaf_df.empty:
                continue
            leaf_df = leaf_df.copy()
            leaf_df["row_id"] = leaf_df["dataset_id"].map(_extract_row_id)
            leaf_df["y_true"] = pd.to_numeric(leaf_df["y_true"], errors="coerce")
            leaf_df = leaf_df.dropna(subset=["row_id", "y_true"])
            if leaf_df.empty:
                continue
            leaf_df["row_id"] = leaf_df["row_id"].astype(int)
            y_true_rows.append(leaf_df[["row_id", "y_true"]])

        if not y_true_rows:
            out_rows.append(
                {
                    "outcome": outcome,
                    "ref_model": ref_model,
                    "n": 0,
                    "eta_std": float("nan"),
                    "term_in_beta": float("nan"),
                    "term_in_design": float("nan"),
                    "term_missing_in_beta": float("nan"),
                    "ref_spearman_eta": float("nan"),
                    "ref_gini_eta": float("nan"),
                    "ref_spearman_pred": float("nan"),
                    "ref_gini_pred": float("nan"),
                    "ref_skill_pred": float("nan"),
                    "ref_gini_pred_ci_low": float("nan"),
                    "ref_gini_pred_ci_high": float("nan"),
                    "ref_skill_pred_ci_low": float("nan"),
                    "ref_skill_pred_ci_high": float("nan"),
                    "predictions_paths_n": int(len(pred_candidates)),
                }
            )
            continue

        y_true_df = pd.concat(y_true_rows, ignore_index=True)
        y_true_by_row_id = y_true_df.groupby("row_id", dropna=False)["y_true"].mean()

        coef = coef_all.loc[coef_all["model"].astype(str) == str(ref_model), ["term", "estimate"]].copy()
        if coef.empty:
            out_rows.append(
                {
                    "outcome": outcome,
                    "ref_model": ref_model,
                    "n": 0,
                    "eta_std": float("nan"),
                    "term_in_beta": 0,
                    "term_in_design": float("nan"),
                    "term_missing_in_beta": float("nan"),
                    "ref_spearman_eta": float("nan"),
                    "ref_gini_eta": float("nan"),
                    "predictions_paths_n": int(len(pred_candidates)),
                }
            )
            continue

        spec = specs_by_model.get(str(ref_model))
        if spec is None:
            raise KeyError(f"Missing ModelSpec for reference model: {ref_model}")

        formula, _ = _prepare_formula(spec, df_proc, col_types, reference_levels)
        _, X_full = dmatrices(formula, data=df_proc, return_type="dataframe")
        if spec.model_type == "ordinal":
            if "Intercept" in X_full.columns:
                X_full = X_full.drop(columns=["Intercept"])
            constant_cols = [name for name in X_full.columns if np.isclose(X_full[name].std(ddof=0), 0.0)]
            if constant_cols:
                X_full = X_full.drop(columns=constant_cols)

        beta = pd.Series(
            coef["estimate"].to_numpy(dtype=float),
            index=coef["term"].astype(str),
            dtype="float64",
        )

        design_cols = list(X_full.columns)
        missing_in_beta = [c for c in design_cols if c not in beta.index]
        common_cols = [c for c in design_cols if c in beta.index]

        if not common_cols:
            rho = None
            gini = None
            eta_std = float("nan")
            n_used = 0
            joined = pd.DataFrame()
        else:
            eta_series = X_full[common_cols].dot(beta.reindex(common_cols).fillna(0.0))
            row_ids = y_true_by_row_id.index.to_numpy(dtype=int)
            eta_series = eta_series.reindex(row_ids)
            joined = pd.DataFrame(
                {
                    "y_true": y_true_by_row_id.reindex(row_ids),
                    "eta": eta_series,
                }
            ).dropna()
            n_used = int(joined.shape[0])
            eta_std = float(np.nanstd(joined["eta"].to_numpy(dtype=float))) if n_used else float("nan")
            rho = _spearman_np(joined["y_true"].to_numpy(dtype=float), joined["eta"].to_numpy(dtype=float))
            gini = _gini_np(joined["eta"].to_numpy(dtype=float), joined["y_true"].to_numpy(dtype=float))

        # Response-scale predictions (used by thesis reference columns).
        rho_pred = None
        gini_pred = None
        skill_pred = None
        gini_pred_ci = (None, None)
        skill_pred_ci = (None, None)
        try:
            pred_joined: pd.DataFrame | None = None
            if spec.model_type in {"linear", "logit"} and not joined.empty:
                # Reuse coefficient-derived eta; map to response scale.
                y_ref_pred = joined["eta"].to_numpy(dtype=float)
                if spec.model_type == "logit":
                    y_ref_pred = 1.0 / (1.0 + np.exp(-np.clip(y_ref_pred, -40.0, 40.0)))
                pred_joined = pd.DataFrame(
                    {
                        "y_true": joined["y_true"].to_numpy(dtype=float),
                        "y_ref_pred": y_ref_pred,
                    }
                ).dropna()
            elif spec.model_type == "ordinal":
                y_ord, X_ord = dmatrices(formula, data=df_proc, return_type="dataframe")
                if "Intercept" in X_ord.columns:
                    X_ord = X_ord.drop(columns=["Intercept"])
                constant_cols = [name for name in X_ord.columns if np.isclose(X_ord[name].std(ddof=0), 0.0)]
                if constant_cols:
                    X_ord = X_ord.drop(columns=constant_cols)
                ord_model = OrderedModel(
                    y_ord.iloc[:, 0],
                    X_ord,
                    distr="logit",
                    weights=None,
                )
                ord_fit = ord_model.fit(method="bfgs", disp=False)
                probs = np.asarray(ord_fit.model.predict(ord_fit.params, exog=X_ord), dtype=float)
                levels = np.sort(pd.to_numeric(y_ord.iloc[:, 0], errors="coerce").dropna().unique()).astype(float)
                if probs.ndim == 1:
                    pred_vals = probs
                else:
                    if levels.size != probs.shape[1]:
                        levels = np.arange(1, probs.shape[1] + 1, dtype=float)
                    pred_vals = probs @ levels
                pred_series = pd.Series(np.asarray(pred_vals, dtype=float), index=X_ord.index, dtype="float64")
                row_ids = y_true_by_row_id.index.to_numpy(dtype=int)
                pred_joined = pd.DataFrame(
                    {
                        "y_true": y_true_by_row_id.reindex(row_ids),
                        "y_ref_pred": pred_series.reindex(row_ids),
                    }
                ).dropna()

            if pred_joined is not None:
                n_pred = int(pred_joined.shape[0])
                if n_pred > 1:
                    yt = pred_joined["y_true"].to_numpy(dtype=float)
                    yp = pred_joined["y_ref_pred"].to_numpy(dtype=float)
                    rho_pred = _spearman_np(yt, yp)
                    gini_pred = _gini_np(yp, yt)
                    skill_pred = _mae_skill_np(yt, yp)

                    boot_b = 500
                    rng = np.random.default_rng(2026_02_06 + int(abs(hash(outcome)) % 10_000))
                    boot_idx = rng.integers(0, n_pred, size=(boot_b, n_pred), dtype=np.int64)
                    g_boot = np.full(boot_b, np.nan, dtype=float)
                    s_boot = np.full(boot_b, np.nan, dtype=float)
                    for b in range(boot_b):
                        idx = boot_idx[b]
                        g_boot[b] = float(_gini_np(yp[idx], yt[idx]) or np.nan)
                        s_boot[b] = float(_mae_skill_np(yt[idx], yp[idx]) or np.nan)
                    gini_pred_ci = _ci_np(g_boot)
                    skill_pred_ci = _ci_np(s_boot)
        except Exception:
            pass

        out_rows.append(
            {
                "outcome": outcome,
                "ref_model": ref_model,
                "n": int(n_used),
                "eta_std": float(eta_std),
                "term_in_beta": int(len(beta.index)),
                "term_in_design": int(len(design_cols)),
                "term_missing_in_beta": int(len(missing_in_beta)),
                "ref_spearman_eta": float(rho) if rho is not None else float("nan"),
                "ref_gini_eta": float(gini) if gini is not None else float("nan"),
                "ref_spearman_pred": float(rho_pred) if rho_pred is not None else float("nan"),
                "ref_gini_pred": float(gini_pred) if gini_pred is not None else float("nan"),
                "ref_skill_pred": float(skill_pred) if skill_pred is not None else float("nan"),
                "ref_gini_pred_ci_low": float(gini_pred_ci[0]) if gini_pred_ci[0] is not None else float("nan"),
                "ref_gini_pred_ci_high": float(gini_pred_ci[1]) if gini_pred_ci[1] is not None else float("nan"),
                "ref_skill_pred_ci_low": float(skill_pred_ci[0]) if skill_pred_ci[0] is not None else float("nan"),
                "ref_skill_pred_ci_high": float(skill_pred_ci[1]) if skill_pred_ci[1] is not None else float("nan"),
                "predictions_paths_n": int(len(pred_candidates)),
            }
        )

    out_path = out if out is not None else (output_root / "_summary" / "ref_benchmark.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(out_rows).to_csv(out_path, index=False)
    return out_path


def _cv_splits(*, n: int, n_folds: int, seed: int) -> list[np.ndarray]:
    """Reproduce the fold splitting used by cross_fit_predictions (no stratification)."""

    if n <= 0:
        return []
    folds = min(int(n_folds), n)
    if folds <= 1 or n < 2:
        return [np.arange(n, dtype=int)]
    rng = np.random.default_rng(int(seed))
    indices = np.arange(n, dtype=int)
    rng.shuffle(indices)
    return [arr.astype(int) for arr in np.array_split(indices, folds) if arr.size]


def _mae_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float | None, float | None, int]:
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    n = int(m.sum())
    if n == 0:
        return None, None, 0
    err = y_pred[m] - y_true[m]
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err * err)))
    return mae, rmse, n


def _brier(y_true: np.ndarray, y_pred: np.ndarray) -> float | None:
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    if int(m.sum()) == 0:
        return None
    err = y_pred[m] - y_true[m]
    return float(np.mean(err * err))


def _calibration_cv_rows(
    df: pd.DataFrame,
    *,
    country_col: str,
    n_folds: int,
    seed: int,
) -> tuple[list[dict], list[dict]]:
    """Compute per-fold and aggregate CV metrics for calibration.

    Uses the same calibration families as the main calibrate CLI:
    - binary: Platt scaling with country effects
    - non-binary: country-aware isotonic regression
    """

    required = {"model", "outcome", "outcome_type", "y_true", "y_pred", country_col}
    if any(c not in df.columns for c in required):
        return [], []

    fold_rows: list[dict] = []
    summary_rows: list[dict] = []

    group_cols = ["model", "outcome", "outcome_type"]
    for (model, outcome, outcome_type), g in df.groupby(group_cols, dropna=False):
        work = g.dropna(subset=["y_true", "y_pred", country_col]).copy()
        if work.empty:
            continue

        work = work.reset_index(drop=True)
        y_true = pd.to_numeric(work["y_true"], errors="coerce").to_numpy(dtype=float)
        y_pred = pd.to_numeric(work["y_pred"], errors="coerce").to_numpy(dtype=float)

        splits = _cv_splits(n=len(work), n_folds=int(n_folds), seed=int(seed))
        if not splits:
            continue

        oof_pred = np.full(len(work), np.nan, dtype=float)
        for fold_id, test_idx in enumerate(splits):
            train_idx = np.setdiff1d(np.arange(len(work), dtype=int), test_idx, assume_unique=False)
            if train_idx.size == 0:
                continue
            train = work.iloc[train_idx]
            test = work.iloc[test_idx]

            t = str(outcome_type).lower()
            if t == "binary":
                cal = fit_binary_platt_country(train, p_col="y_pred", y_col="y_true", country_col=country_col, eps=1e-6)
                pred_cal = cal.predict(
                    pd.to_numeric(test["y_pred"], errors="coerce").to_numpy(dtype=float),
                    test[country_col].astype(str).to_numpy(dtype=str),
                )
            else:
                lo = float(np.nanmin(pd.to_numeric(train["y_true"], errors="coerce").to_numpy(dtype=float)))
                hi = float(np.nanmax(pd.to_numeric(train["y_true"], errors="coerce").to_numpy(dtype=float)))
                bounds = (lo, hi) if np.isfinite(lo) and np.isfinite(hi) else None
                cal = fit_isotonic_country(
                    train,
                    y_pred_col="y_pred",
                    y_true_col="y_true",
                    country_col=country_col,
                    bounds=bounds,
                )
                pred_cal = cal.predict(
                    pd.to_numeric(test["y_pred"], errors="coerce").to_numpy(dtype=float),
                    test[country_col].astype(str).to_numpy(dtype=str),
                )
            oof_pred[test_idx] = pred_cal

            mae_uncal, rmse_uncal, _ = _mae_rmse(y_true[test_idx], y_pred[test_idx])
            mae_cal, rmse_cal, n_cal = _mae_rmse(y_true[test_idx], pred_cal.astype(float))
            brier_uncal = _brier(y_true[test_idx], y_pred[test_idx]) if t == "binary" else None
            brier_cal = _brier(y_true[test_idx], pred_cal.astype(float)) if t == "binary" else None
            fold_rows.append(
                {
                    "model": model,
                    "outcome": outcome,
                    "outcome_type": t,
                    "fold": int(fold_id),
                    "seed": int(seed),
                    "n_folds": int(n_folds),
                    "n": int(n_cal),
                    "mae_uncal": mae_uncal,
                    "rmse_uncal": rmse_uncal,
                    "mae_cal": mae_cal,
                    "rmse_cal": rmse_cal,
                    "brier_uncal": brier_uncal,
                    "brier_cal": brier_cal,
                }
            )

        # Aggregate across all held-out predictions.
        mae_uncal, rmse_uncal, _ = _mae_rmse(y_true, y_pred)
        mae_cal, rmse_cal, n_cal = _mae_rmse(y_true, oof_pred)
        brier_uncal = _brier(y_true, y_pred) if str(outcome_type).lower() == "binary" else None
        brier_cal = _brier(y_true, oof_pred) if str(outcome_type).lower() == "binary" else None
        summary_rows.append(
            {
                "model": model,
                "outcome": outcome,
                "outcome_type": str(outcome_type).lower(),
                "seed": int(seed),
                "n_folds": int(n_folds),
                "n": int(n_cal),
                "mae_uncal": mae_uncal,
                "rmse_uncal": rmse_uncal,
                "mae_cal": mae_cal,
                "rmse_cal": rmse_cal,
                "brier_uncal": brier_uncal,
                "brier_cal": brier_cal,
            }
        )

    return fold_rows, summary_rows


def _summarize_spearman_slices(*, slice_spears) -> dict:
    vals = pd.Series(
        [float(s.spearman) if s.spearman is not None else float("nan") for s in slice_spears],
        dtype="float64",
    )
    vals = vals[pd.notna(vals)]
    if vals.empty:
        return {
            "n_slices": int(len(slice_spears)),
            "n_slices_valid": 0,
            "spearman_mean": None,
            "spearman_median": None,
            "spearman_min": None,
            "spearman_p25": None,
            "spearman_p75": None,
            "spearman_max": None,
        }
    return {
        "n_slices": int(len(slice_spears)),
        "n_slices_valid": int(vals.size),
        "spearman_mean": float(vals.mean()),
        "spearman_median": float(vals.median()),
        "spearman_min": float(vals.min()),
        "spearman_p25": float(vals.quantile(0.25)),
        "spearman_p75": float(vals.quantile(0.75)),
        "spearman_max": float(vals.max()),
    }


def _anchor_neighborhood_summary(
    *,
    df: pd.DataFrame,
    y_true_col: str,
    y_pred_col: str,
    anchor_neighborhoods_path: Path,
    max_rank: int | None,
) -> tuple[dict, list[dict]]:
    """Compute slice Spearman within (anchor_id, country) neighborhoods.

    Uses a precomputed neighborhoods parquet with columns:
      anchor_id, country, row_id, rank, dist

    Joins predictions via (country, row_id), where row_id is parsed from dataset_id.
    """

    out_empty = {
        "anchor_neighborhoods_path": str(anchor_neighborhoods_path),
        "anchor_neighborhood_max_rank": max_rank,
        "anchor_neighborhood_n_slices": 0,
        "anchor_neighborhood_n_slices_valid": 0,
        "anchor_neighborhood_spearman_mean": None,
        "anchor_neighborhood_spearman_median": None,
        "anchor_neighborhood_spearman_min": None,
        "anchor_neighborhood_spearman_p25": None,
        "anchor_neighborhood_spearman_p75": None,
        "anchor_neighborhood_spearman_max": None,
    }

    if df.empty:
        return out_empty, []
    if "dataset_id" not in df.columns or "country" not in df.columns:
        return out_empty, []
    if y_true_col not in df.columns or y_pred_col not in df.columns:
        return out_empty, []
    if not anchor_neighborhoods_path.exists():
        return out_empty, []

    neighborhoods = pd.read_parquet(anchor_neighborhoods_path)
    required = {"anchor_id", "country", "row_id", "rank"}
    if not required.issubset(set(neighborhoods.columns)):
        return out_empty, []

    if max_rank is not None:
        neighborhoods = neighborhoods.loc[pd.to_numeric(neighborhoods["rank"], errors="coerce") <= int(max_rank)]

    base = df[["dataset_id", "country", y_true_col, y_pred_col]].copy()
    base["row_id"] = base["dataset_id"].map(_extract_row_id)
    base = base.dropna(subset=["row_id", "country", y_true_col, y_pred_col])
    if base.empty:
        return out_empty, []

    base["row_id"] = pd.to_numeric(base["row_id"], errors="coerce").astype("Int64")
    neighborhoods["row_id"] = pd.to_numeric(neighborhoods["row_id"], errors="coerce").astype("Int64")
    neighborhoods["anchor_id"] = pd.to_numeric(neighborhoods["anchor_id"], errors="coerce").astype("Int64")

    merged = base.merge(
        neighborhoods[["anchor_id", "country", "row_id"]],
        on=["country", "row_id"],
        how="inner",
    )
    if merged.empty:
        return out_empty, []

    merged["anchor_slice"] = merged["country"].astype(str) + "::" + merged["anchor_id"].astype(str)
    slice_spears = compute_slice_spearman(
        df=merged,
        y_true_col=y_true_col,
        y_pred_col=y_pred_col,
        slice_col="anchor_slice",
    )
    stats = _summarize_spearman_slices(slice_spears=slice_spears)

    details: list[dict] = []
    for s in slice_spears:
        country = None
        anchor_id = None
        if "::" in s.slice_value:
            left, right = s.slice_value.split("::", 1)
            country = left
            anchor_id = right
        details.append(
            {
                "country": country,
                "anchor_id": anchor_id,
                "n": int(s.n),
                "spearman": s.spearman,
            }
        )

    out = {
        "anchor_neighborhoods_path": str(anchor_neighborhoods_path),
        "anchor_neighborhood_max_rank": max_rank,
        "anchor_neighborhood_n_slices": int(stats["n_slices"]),
        "anchor_neighborhood_n_slices_valid": int(stats["n_slices_valid"]),
        "anchor_neighborhood_spearman_mean": stats["spearman_mean"],
        "anchor_neighborhood_spearman_median": stats["spearman_median"],
        "anchor_neighborhood_spearman_min": stats["spearman_min"],
        "anchor_neighborhood_spearman_p25": stats["spearman_p25"],
        "anchor_neighborhood_spearman_p75": stats["spearman_p75"],
        "anchor_neighborhood_spearman_max": stats["spearman_max"],
    }
    return out, details


def _find_prediction_files(root: Path, *, include_runs: bool) -> list[Path]:
    files: list[Path] = []
    for p in root.rglob("predictions*.csv"):
        if _is_ignored_path_for_scan(p, include_runs=include_runs):
            continue
        if p.name not in {"predictions.csv", "predictions_calibrated.csv"}:
            continue
        files.append(p)
    return sorted(files)


def _predictions_path_relative(*, pred_path: Path, root: Path) -> str:
    try:
        return str(pred_path.relative_to(root))
    except ValueError:
        return str(pred_path)


_POSTHOC_BOOT_KEYS = (
    "slice_gini_mean",
    "dispersion_ratio_iqr",
    "tail_bias_bottom_decile_mean_error",
    "tail_bias_top_decile_mean_error",
    "pooled_gini",
    "mae_skill_cal",
)


def _bootstrap_posthoc_ci(
    *,
    df: pd.DataFrame,
    y_true_col: str,
    y_pred_col: str,
    slice_col: str,
    B: int,
    seed: int,
    keys: tuple[str, ...] = _POSTHOC_BOOT_KEYS,
) -> dict[str, float | None]:
    if B <= 0:
        return {}
    n = int(df.shape[0])
    if n <= 0:
        return {}

    rng = np.random.default_rng(int(seed))
    boot_idx = rng.integers(0, n, size=(int(B), n), dtype=np.int64)
    bs: dict[str, np.ndarray] = {k: np.empty(int(B), dtype=np.float64) for k in keys}

    for b in range(int(B)):
        sub = df.iloc[boot_idx[b]].reset_index(drop=True)
        summary, _ = summarize_choice_validation_posthoc_metrics(
            df=sub,
            y_true_col=y_true_col,
            y_pred_col=y_pred_col,
            slice_col=slice_col,
        )
        # Compute pooled gini and mae_skill_cal for bootstrap (not in posthoc summary).
        yt_b = pd.to_numeric(sub[y_true_col], errors="coerce").to_numpy(dtype=float)
        yp_b = pd.to_numeric(sub[y_pred_col], errors="coerce").to_numpy(dtype=float)
        _g = _gini_np(yt_b, yp_b)
        summary["pooled_gini"] = float(_g) if _g is not None else None
        _m = np.isfinite(yt_b) & np.isfinite(yp_b)
        if np.any(_m):
            _yt = yt_b[_m]
            _yp = yp_b[_m]
            _mae = float(np.mean(np.abs(_yp - _yt)))
            _bm = float(np.median(_yt))
            _bmae = float(np.mean(np.abs(_yt - _bm)))
            summary["mae_skill_cal"] = float(1.0 - _mae / _bmae) if _bmae > 0 else None
        else:
            summary["mae_skill_cal"] = None
        for k in keys:
            val = summary.get(k)
            bs[k][b] = float(val) if val is not None else float("nan")

    out: dict[str, float | None] = {}
    for k, arr in bs.items():
        vals = arr[np.isfinite(arr)]
        if vals.size == 0:
            out[f"{k}_lo"] = None
            out[f"{k}_hi"] = None
            continue
        out[f"{k}_lo"] = float(np.quantile(vals, 0.025))
        out[f"{k}_hi"] = float(np.quantile(vals, 0.975))
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        default="evaluation/output/choice_validation/unperturbed",
        help="Choice-validation unperturbed root.",
    )
    parser.add_argument(
        "--include-runs",
        action="store_true",
        help="Include per-run leaves under _runs/ (default: false).",
    )
    parser.add_argument(
        "--slice-col",
        default="country",
        help="Column to compute within-slice Spearman (default: country).",
    )
    parser.add_argument(
        "--write-slice-details",
        action="store_true",
        help="Write per-slice spearman details next to posthoc_metrics.csv.",
    )
    parser.add_argument(
        "--anchor-neighborhoods-path",
        default="empirical/output/anchors/system/anchor_neighborhoods.parquet",
        help="Parquet file defining (anchor_id,country) neighborhoods for optional slice-stability diagnostics.",
    )
    parser.add_argument(
        "--anchor-neighborhood-max-rank",
        type=int,
        default=None,
        help="Use only neighborhoods rows with rank <= this value (default: use all rows in file).",
    )
    parser.add_argument(
        "--calibration-cv-folds",
        type=int,
        default=5,
        help="Number of folds for calibration CV summaries (default: 5).",
    )
    parser.add_argument(
        "--calibration-cv-seed",
        type=int,
        default=13,
        help="Seed for calibration CV fold assignment (default: 13).",
    )
    parser.add_argument(
        "--skip-calibration-cv",
        action="store_true",
        help="Skip calibration CV summaries (speeds up posthoc runs).",
    )
    parser.add_argument(
        "--skip-reference-benchmark",
        action="store_true",
        help="Skip the reference-model benchmark computation (default: compute it).",
    )
    parser.add_argument(
        "--coef-path",
        default="empirical/output/modeling/model_coefficients_all.csv",
        help="Reference coefficients CSV (model, term, estimate...).",
    )
    parser.add_argument(
        "--model-plan",
        default="empirical/output/modeling/model_plan.json",
        help="Model plan JSON (used to reconstruct patsy formulas).",
    )
    parser.add_argument(
        "--column-types",
        default="empirical/specs/column_types.tsv",
        help="Column types TSV with optional reference levels.",
    )
    parser.add_argument(
        "--input-csv",
        default="preprocess/output/clean_processed_survey.csv",
        help="Clean processed survey CSV (same base used for structural modeling).",
    )
    parser.add_argument(
        "--missingness-threshold",
        type=float,
        default=0.35,
        help="Missingness threshold used when constructing fill/indicator plan.",
    )
    parser.add_argument(
        "--no-standardize",
        action="store_true",
        help="Disable predictor standardization (default mirrors modeling pipeline: standardize on).",
    )
    parser.add_argument(
        "--ref-model-map",
        default=None,
        help="Optional JSON file mapping outcome -> reference model name.",
    )
    parser.add_argument(
        "--lever-config",
        default="config/levers.json",
        help="Lever config used to resolve outcome -> model_ref defaults for reference benchmark.",
    )
    parser.add_argument(
        "--ref-benchmark-out",
        default=None,
        help="Optional explicit output CSV path for reference benchmark (default: <output_root>/_summary/ref_benchmark.csv).",
    )
    parser.add_argument(
        "--bootstrap-b",
        type=int,
        default=0,
        help="Bootstrap iterations for posthoc CI summaries (default: 0 = skip).",
    )
    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=20260201,
        help="Seed for posthoc bootstrap resampling (default: 20260201).",
    )
    # Anchor-neighborhood slice stability is required by default.

    args = parser.parse_args(argv)

    root = Path(args.output_root)
    pred_paths = _find_prediction_files(root, include_runs=bool(args.include_runs))

    rows: list[dict] = []
    bootstrap_rows: list[dict] = []
    residuals_by_group: dict[tuple[str, str, str, str], dict[str, pd.Series]] = {}
    cal_cv_fold_rows: list[dict] = []
    cal_cv_summary_rows: list[dict] = []

    for pred_path in pred_paths:
        try:
            df = pd.read_csv(pred_path)
        except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError):
            df = pd.DataFrame()
        status = _infer_status(pred_path)
        pred_col = _infer_pred_col(status)

        try:
            summary, slice_details = summarize_choice_validation_posthoc_metrics(
                df=df,
                y_true_col="y_true",
                y_pred_col=pred_col,
                slice_col=str(args.slice_col),
            )
        except KeyError:
            summary, slice_details = (_empty_posthoc_summary(slice_col=str(args.slice_col)), [])

        # Pooled gini and MAE skill.
        _pooled_gini = None
        _mae_skill_cal = None
        if not df.empty and "y_true" in df.columns and pred_col in df.columns:
            yt = pd.to_numeric(df["y_true"], errors="coerce").to_numpy(dtype=float)
            yp = pd.to_numeric(df[pred_col], errors="coerce").to_numpy(dtype=float)
            _g = _gini_np(yt, yp)
            if _g is not None:
                _pooled_gini = float(_g)
            _mask = np.isfinite(yt) & np.isfinite(yp)
            if np.any(_mask):
                _yt_v = yt[_mask]
                _yp_v = yp[_mask]
                _mae = float(np.mean(np.abs(_yp_v - _yt_v)))
                _baseline = float(np.median(_yt_v))
                _baseline_mae = float(np.mean(np.abs(_yt_v - _baseline)))
                if _baseline_mae > 0:
                    _mae_skill_cal = float(1.0 - _mae / _baseline_mae)
        summary["pooled_gini"] = _pooled_gini
        summary["mae_skill_cal"] = _mae_skill_cal

        # Anchor-neighborhood slice stability (required by default).
        max_rank = int(args.anchor_neighborhood_max_rank) if args.anchor_neighborhood_max_rank is not None else None
        anchor_summary, anchor_details = _anchor_neighborhood_summary(
            df=df,
            y_true_col="y_true",
            y_pred_col=pred_col,
            anchor_neighborhoods_path=Path(args.anchor_neighborhoods_path),
            max_rank=max_rank,
        )

        fallback_meta = _path_fallback_metadata(pred_path=pred_path, root=root)
        meta = _fill_missing_metadata(meta=_leaf_metadata(df), fallback=fallback_meta)
        meta = _prefer_path_metadata(meta=meta, fallback=fallback_meta)

        row = {
            **meta,
            "calibration_status": status,
            "predictions_path": _predictions_path_relative(pred_path=pred_path, root=root),
            **summary,
            **anchor_summary,
        }

        if int(args.bootstrap_b) > 0 and not df.empty:
            try:
                boot_ci = _bootstrap_posthoc_ci(
                    df=df,
                    y_true_col="y_true",
                    y_pred_col=pred_col,
                    slice_col=str(args.slice_col),
                    B=int(args.bootstrap_b),
                    seed=int(args.bootstrap_seed),
                )
            except KeyError:
                boot_ci = {}
            if boot_ci:
                bootstrap_rows.append(
                    {
                        **meta,
                        "calibration_status": status,
                        **boot_ci,
                    }
                )

        out_dir = pred_path.parent
        out_metrics = out_dir / "posthoc_metrics.csv"
        pd.DataFrame([row]).to_csv(out_metrics, index=False)

        # Quantile/decile MAE + signed bias by y_true.
        try:
            deciles = compute_y_decile_errors_from_df(df=df, y_true_col="y_true", y_pred_col=pred_col, n_bins=10)
        except KeyError:
            deciles = []
        pd.DataFrame(
            [
                {
                    "decile": d.decile,
                    "y_q_lo": d.q_lo,
                    "y_q_hi": d.q_hi,
                    "n": d.n,
                    "mae": d.mae,
                    "mean_error": d.mean_error,
                }
                for d in deciles
            ]
        ).to_csv(out_dir / "y_decile_metrics.csv", index=False)

        if args.write_slice_details and slice_details:
            details_path = out_dir / f"slice_spearman__{args.slice_col}.csv"
            pd.DataFrame([{"slice": s.slice_value, "n": s.n, "spearman": s.spearman} for s in slice_details]).to_csv(
                details_path, index=False
            )

        if args.write_slice_details and anchor_details:
            details_path = out_dir / "slice_spearman__anchor_neighborhoods.csv"
            pd.DataFrame(anchor_details).to_csv(details_path, index=False)

        # Residual correlation across strategies (computed later per group).
        if not df.empty and "dataset_id" in df.columns and "y_true" in df.columns and pred_col in df.columns:
            tmp = df[["dataset_id", "y_true", pred_col]].copy()
            tmp["y_true"] = pd.to_numeric(tmp["y_true"], errors="coerce")
            tmp[pred_col] = pd.to_numeric(tmp[pred_col], errors="coerce")
            tmp["resid"] = tmp[pred_col] - tmp["y_true"]
            tmp["row_id"] = tmp["dataset_id"].map(_extract_row_id)
            tmp = tmp.dropna(subset=["row_id", "resid"])
            if not tmp.empty:
                tmp["row_id"] = tmp["row_id"].astype(int)
            resid = tmp.groupby("row_id", dropna=False)["resid"].mean()

            outcome = str(row.get("outcome") or "")
            model = str(row.get("model") or "")
            run_id = str(row.get("run_id") or "")
            strategy = str(row.get("strategy") or "")
            key = (outcome, model, run_id, status)
            if key not in residuals_by_group:
                residuals_by_group[key] = {}
            if strategy:
                residuals_by_group[key][strategy] = resid

        # Calibration CV summary (computed on uncalibrated predictions only).
        if (
            not bool(args.skip_calibration_cv)
            and status == "uncalibrated"
            and not df.empty
            and "y_pred" in df.columns
            and "y_true" in df.columns
            and "country" in df.columns
        ):
            fold_rows, summary_rows = _calibration_cv_rows(
                df=df,
                country_col="country",
                n_folds=int(args.calibration_cv_folds),
                seed=int(args.calibration_cv_seed),
            )
            for r in fold_rows:
                r2 = {**meta, **r}
                cal_cv_fold_rows.append(r2)
            for r in summary_rows:
                r2 = {**meta, **r}
                cal_cv_summary_rows.append(r2)

        rows.append(row)

    if rows:
        out_summary_dir = root / "_summary"
        out_summary_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).sort_values(
            by=["outcome", "model", "strategy", "calibration_status"],
            kind="mergesort",
            na_position="last",
        ).to_csv(out_summary_dir / "posthoc_metrics_all.csv", index=False)

        if bootstrap_rows:
            pd.DataFrame(bootstrap_rows).sort_values(
                by=["outcome", "model", "strategy", "calibration_status"],
                kind="mergesort",
                na_position="last",
            ).to_csv(out_summary_dir / "posthoc_metrics_bootstrap.csv", index=False)

        # Residual correlation across strategies within each group.
        corr_rows: list[dict] = []
        for (outcome, model, run_id, status), by_strategy in sorted(residuals_by_group.items()):
            strategies = sorted(by_strategy.keys())
            for i, a in enumerate(strategies):
                for b in strategies[i + 1 :]:
                    sa = by_strategy[a]
                    sb = by_strategy[b]
                    sa2, sb2 = sa.align(sb, join="inner")
                    n_common = int((pd.notna(sa2) & pd.notna(sb2)).sum())
                    corr_rows.append(
                        {
                            "outcome": outcome,
                            "model": model,
                            "run_id": run_id,
                            "calibration_status": status,
                            "strategy_a": a,
                            "strategy_b": b,
                            "n_common": n_common,
                            "n_a": int(pd.notna(sa).sum()),
                            "n_b": int(pd.notna(sb).sum()),
                            "overlap_rate_a": float(n_common) / float(int(pd.notna(sa).sum()))
                            if int(pd.notna(sa).sum())
                            else None,
                            "overlap_rate_b": float(n_common) / float(int(pd.notna(sb).sum()))
                            if int(pd.notna(sb).sum())
                            else None,
                            "residual_corr_pearson": _pearson(sa, sb),
                            "residual_corr_spearman": _spearman_series(sa, sb),
                        }
                    )
        if corr_rows:
            corr_df = pd.DataFrame(corr_rows)
            corr_df.to_csv(out_summary_dir / "residual_correlation_by_strategy.csv", index=False)
            # Back-compat / thesis-friendly alias.
            corr_df.to_csv(out_summary_dir / "residual_corr_strategy.csv", index=False)

        if cal_cv_summary_rows:
            pd.DataFrame(cal_cv_summary_rows).to_csv(out_summary_dir / "calibration_cv_summary.csv", index=False)
        if cal_cv_fold_rows:
            pd.DataFrame(cal_cv_fold_rows).to_csv(out_summary_dir / "calibration_cv_folds.csv", index=False)

    # ── Cell-level bootstrap for pooled Gini and MAE skill ────────────────
    #
    # When --bootstrap-b > 0, precompute:
    #   cell_bootstrap_cis.csv     – per (outcome, model, strategy) gini/skill CIs
    #   outcome_bootstrap_cis.csv  – per outcome mean-across-cells gini/skill CIs
    #   strategy_bootstrap_cis.csv – per strategy cross-outcome mean gini/skill CIs
    #
    # These replace the expensive on-the-fly bootstrap in the thesis QMD.
    if int(args.bootstrap_b) > 0 and rows:
        _BOOT_B = int(args.bootstrap_b)
        _BOOT_SEED_BASE = int(args.bootstrap_seed)

        # Gather outcomes, models, strategies from collected rows.
        all_outcomes = sorted({str(r.get("outcome", "")) for r in rows if r.get("outcome")})
        all_models = sorted({str(r.get("model", "")) for r in rows if r.get("model")})
        all_strategies = sorted({str(r.get("strategy", "")) for r in rows if r.get("strategy")})

        cell_boot_rows: list[dict] = []
        # Key -> (BOOT_B,) arrays for gini and skill_cal.
        boot_dist: dict[tuple[str, str, str], dict[str, np.ndarray]] = {}

        for oi, outcome in enumerate(all_outcomes):
            for mi, model in enumerate(all_models):
                for si, strategy in enumerate(all_strategies):
                    # Find the uncalibrated predictions path.
                    uncal_path = root / outcome / model / strategy / "uncalibrated" / "predictions.csv"
                    cal_path = root / outcome / model / strategy / "calibrated" / "predictions_calibrated.csv"
                    if not uncal_path.exists() or not cal_path.exists():
                        continue
                    try:
                        df_uncal = pd.read_csv(uncal_path)
                        df_cal = pd.read_csv(cal_path)
                    except Exception:
                        continue
                    if df_uncal.empty or df_cal.empty:
                        continue

                    # Align on dataset_id.
                    yt = pd.to_numeric(df_uncal.get("y_true", pd.Series(dtype=float)), errors="coerce").to_numpy(
                        dtype=float
                    )
                    yp_raw = pd.to_numeric(df_uncal.get("y_pred", pd.Series(dtype=float)), errors="coerce").to_numpy(
                        dtype=float
                    )
                    cal_col = "y_pred_cal" if "y_pred_cal" in df_cal.columns else "y_pred"
                    yp_cal = pd.to_numeric(df_cal.get(cal_col, pd.Series(dtype=float)), errors="coerce").to_numpy(
                        dtype=float
                    )
                    n = min(yt.size, yp_raw.size, yp_cal.size)
                    if n < 2:
                        continue
                    yt, yp_raw, yp_cal = yt[:n], yp_raw[:n], yp_cal[:n]

                    rng = np.random.default_rng(_BOOT_SEED_BASE + 10_000 * oi + 97 * mi + 7919 * si)
                    boot_idx = rng.integers(0, n, size=(_BOOT_B, n), dtype=np.int64)

                    gini_bs = np.empty(_BOOT_B, dtype=np.float64)
                    skill_bs = np.empty(_BOOT_B, dtype=np.float64)
                    for b in range(_BOOT_B):
                        idx = boot_idx[b]
                        _yt = yt[idx]
                        _g = _gini_np(_yt, yp_raw[idx])
                        gini_bs[b] = float(_g) if _g is not None else float("nan")
                        _ypc = yp_cal[idx]
                        _mask = np.isfinite(_yt) & np.isfinite(_ypc)
                        if np.any(_mask):
                            _mae = float(np.mean(np.abs(_ypc[_mask] - _yt[_mask])))
                            _bm = float(np.median(_yt[_mask]))
                            _bmae = float(np.mean(np.abs(_yt[_mask] - _bm)))
                            skill_bs[b] = float(1.0 - _mae / _bmae) if _bmae > 0 else float("nan")
                        else:
                            skill_bs[b] = float("nan")

                    boot_dist[(outcome, model, strategy)] = {"gini": gini_bs, "skill_cal": skill_bs}

                    # Per-cell CIs.
                    def _q(arr: np.ndarray) -> tuple[float | None, float | None]:
                        vals = arr[np.isfinite(arr)]
                        if vals.size == 0:
                            return (None, None)
                        return (float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975)))

                    g_lo, g_hi = _q(gini_bs)
                    s_lo, s_hi = _q(skill_bs)
                    cell_boot_rows.append(
                        {
                            "outcome": outcome,
                            "model": model,
                            "strategy": strategy,
                            "gini_lo": g_lo,
                            "gini_hi": g_hi,
                            "skill_cal_lo": s_lo,
                            "skill_cal_hi": s_hi,
                        }
                    )

        out_summary_dir = root / "_summary"
        out_summary_dir.mkdir(parents=True, exist_ok=True)

        # cell_bootstrap_cis.csv
        if cell_boot_rows:
            pd.DataFrame(cell_boot_rows).to_csv(out_summary_dir / "cell_bootstrap_cis.csv", index=False)

        # outcome_bootstrap_cis.csv — mean across cells within each outcome.
        outcome_ci_rows: list[dict] = []
        for outcome in all_outcomes:
            keys = [k for k in boot_dist if k[0] == outcome]
            if not keys:
                continue
            gini_stack = np.vstack([boot_dist[k]["gini"] for k in keys])
            skill_stack = np.vstack([boot_dist[k]["skill_cal"] for k in keys])
            mean_gini = np.nanmean(gini_stack, axis=0)
            mean_skill = np.nanmean(skill_stack, axis=0)
            g_vals = mean_gini[np.isfinite(mean_gini)]
            s_vals = mean_skill[np.isfinite(mean_skill)]
            outcome_ci_rows.append(
                {
                    "outcome": outcome,
                    "mean_gini_lo": float(np.quantile(g_vals, 0.025)) if g_vals.size else None,
                    "mean_gini_hi": float(np.quantile(g_vals, 0.975)) if g_vals.size else None,
                    "mean_skill_lo": float(np.quantile(s_vals, 0.025)) if s_vals.size else None,
                    "mean_skill_hi": float(np.quantile(s_vals, 0.975)) if s_vals.size else None,
                }
            )
        if outcome_ci_rows:
            pd.DataFrame(outcome_ci_rows).to_csv(out_summary_dir / "outcome_bootstrap_cis.csv", index=False)

        # strategy_bootstrap_cis.csv — for each strategy, average across models
        # within each outcome, then average across outcomes.
        strategy_ci_rows: list[dict] = []
        for strategy in all_strategies:
            gini_parts: list[np.ndarray] = []
            skill_parts: list[np.ndarray] = []
            for outcome in all_outcomes:
                keys = [k for k in boot_dist if k[0] == outcome and k[2] == strategy]
                if not keys:
                    continue
                gini_mat = np.vstack([boot_dist[k]["gini"] for k in keys])
                skill_mat = np.vstack([boot_dist[k]["skill_cal"] for k in keys])
                gini_parts.append(np.nanmean(gini_mat, axis=0))
                skill_parts.append(np.nanmean(skill_mat, axis=0))
            if not gini_parts or not skill_parts:
                continue
            gini_bs = np.mean(np.vstack(gini_parts), axis=0)
            skill_bs = np.mean(np.vstack(skill_parts), axis=0)
            g_vals = gini_bs[np.isfinite(gini_bs)]
            s_vals = skill_bs[np.isfinite(skill_bs)]
            strategy_ci_rows.append(
                {
                    "strategy": strategy,
                    "mean_gini_lo": float(np.quantile(g_vals, 0.025)) if g_vals.size else None,
                    "mean_gini_hi": float(np.quantile(g_vals, 0.975)) if g_vals.size else None,
                    "mean_skill_lo": float(np.quantile(s_vals, 0.025)) if s_vals.size else None,
                    "mean_skill_hi": float(np.quantile(s_vals, 0.975)) if s_vals.size else None,
                }
            )
        if strategy_ci_rows:
            pd.DataFrame(strategy_ci_rows).to_csv(out_summary_dir / "strategy_bootstrap_cis.csv", index=False)

    # Reference benchmark is conceptually post-hoc, but independent of leaf iteration.
    if not bool(args.skip_reference_benchmark):
        compute_reference_benchmark(
            output_root=root,
            coef_path=Path(args.coef_path),
            model_plan=Path(args.model_plan),
            column_types=Path(args.column_types),
            input_csv=Path(args.input_csv),
            missingness_threshold=float(args.missingness_threshold),
            standardize=not bool(args.no_standardize),
            ref_model_map=Path(args.ref_model_map) if args.ref_model_map else None,
            lever_config=Path(args.lever_config) if args.lever_config else None,
            out=Path(args.ref_benchmark_out) if args.ref_benchmark_out else None,
        )


if __name__ == "__main__":
    main()
