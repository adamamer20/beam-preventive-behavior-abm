from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from beam_abm.evaluation.choice._canonicalize_types import CalibrationStatus
from beam_abm.evaluation.common.metrics import gini_alignment, gini_from_cindex


def _spearman(x: np.ndarray, y: np.ndarray) -> float | None:
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


def _auc(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
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


def _mae_skill_score(y_true: np.ndarray, y_pred: np.ndarray, *, outcome_type: str) -> tuple[float | None, float | None]:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() == 0:
        return None, None
    yt = y_true[mask]
    yp = y_pred[mask]
    model_mae = float(np.mean(np.abs(yp - yt)))

    outcome_type = str(outcome_type).lower()
    if outcome_type == "binary":
        prevalence = float(np.mean(yt))
        baseline = 1.0 if prevalence >= 0.5 else 0.0
    else:
        baseline = float(np.median(yt))

    baseline_mae = float(np.mean(np.abs(yt - baseline)))
    if baseline_mae <= 0:
        return model_mae, None
    return model_mae, float(1.0 - model_mae / baseline_mae)


def compute_metrics_df(
    df: pd.DataFrame, *, pred_col: str, run_id: str, strategy: str, status: CalibrationStatus
) -> pd.DataFrame:
    required = {"model", "outcome", "outcome_type", "y_true", pred_col, "country"}
    missing = [c for c in sorted(required) if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns for metrics: {missing}")

    rows: list[dict[str, Any]] = []
    for (model, outcome, outcome_type), g in df.groupby(["model", "outcome", "outcome_type"], dropna=False):
        y_true = g["y_true"].to_numpy(dtype=float)
        y_pred = g[pred_col].to_numpy(dtype=float)

        if str(outcome_type).lower() == "binary":
            y_bin = y_true.astype(int)
            mask = np.isfinite(y_pred) & np.isfinite(y_bin.astype(float))
            brier = float(np.mean((y_pred[mask] - y_bin[mask]) ** 2)) if mask.sum() else None
            auc_val = _auc(y_true=y_bin.astype(float), y_score=y_pred)
            gini = None
            if auc_val is not None and np.isfinite(auc_val):
                gini = gini_from_cindex(float(auc_val))
            mae, mae_skill = _mae_skill_score(y_true, y_pred, outcome_type=str(outcome_type))
            rows.append(
                {
                    "model": model,
                    "outcome": outcome,
                    "type": str(outcome_type).lower(),
                    "n": int(mask.sum()),
                    "brier": brier,
                    "auc": auc_val,
                    "gini": gini,
                    "mae": mae,
                    "mae_skill": mae_skill,
                    "calibration_status": status,
                    "run_id": run_id,
                    "strategy": strategy,
                }
            )
            continue

        mask = np.isfinite(y_pred) & np.isfinite(y_true)
        mae = float(np.mean(np.abs(y_pred[mask] - y_true[mask]))) if mask.sum() else None
        rmse = float(np.sqrt(np.mean((y_pred[mask] - y_true[mask]) ** 2))) if mask.sum() else None
        spear = _spearman(y_pred, y_true)
        gini = gini_alignment(y_true, y_pred)
        _, mae_skill = _mae_skill_score(y_true, y_pred, outcome_type=str(outcome_type))
        rows.append(
            {
                "model": model,
                "outcome": outcome,
                "type": str(outcome_type).lower(),
                "n": int(mask.sum()),
                "mae": mae,
                "rmse": rmse,
                "spearman": spear,
                "gini": gini,
                "mae_skill": mae_skill,
                "calibration_status": status,
                "run_id": run_id,
                "strategy": strategy,
            }
        )
    return pd.DataFrame(rows)
