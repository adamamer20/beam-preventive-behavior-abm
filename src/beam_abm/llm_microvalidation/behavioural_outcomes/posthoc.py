from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from beam_abm.llm_microvalidation.shared.metrics import gini_from_cindex


@dataclass(frozen=True, slots=True)
class SliceSpearman:
    slice_value: str
    n: int
    spearman: float | None


@dataclass(frozen=True, slots=True)
class SliceGini:
    slice_value: str
    n: int
    gini: float | None


@dataclass(frozen=True, slots=True)
class DecileError:
    decile: int
    q_lo: float | None
    q_hi: float | None
    n: int
    mae: float | None
    mean_error: float | None


def _as_float_array(values: pd.Series) -> np.ndarray:
    return pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)


def _spearman(x: np.ndarray, y: np.ndarray) -> float | None:
    """Spearman correlation (no SciPy dependency).

    Computes Spearman by ranking (average ties) then Pearson on ranks.
    """

    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 2:
        return None

    sx = pd.Series(x[mask]).rank(method="average").to_numpy(dtype=float)
    sy = pd.Series(y[mask]).rank(method="average").to_numpy(dtype=float)

    vx = sx - float(sx.mean())
    vy = sy - float(sy.mean())

    denom = float(np.sqrt((vx * vx).sum()) * np.sqrt((vy * vy).sum()))
    if denom <= 0.0:
        return None

    return float((vx * vy).sum() / denom)


def _cindex(x: np.ndarray, y: np.ndarray) -> float | None:
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 2:
        return None

    x = x[mask]
    y = y[mask]
    unique_preds = np.unique(x)
    pred_ranks = np.searchsorted(unique_preds, x) + 1
    m = int(unique_preds.size)

    order = np.argsort(y, kind="mergesort")
    y = y[order]
    pred_ranks = pred_ranks[order]

    tree = np.zeros(m + 2, dtype=np.int64)

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
        total_prev = _sum(m)
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
    return float((concordant + 0.5 * ties) / total_pairs)


def _nanquantile(x: np.ndarray, q: float) -> float | None:
    if x.size == 0:
        return None
    mask = np.isfinite(x)
    if int(mask.sum()) == 0:
        return None
    return float(np.nanquantile(x[mask], q))


def _iqr(x: np.ndarray) -> float | None:
    q75 = _nanquantile(x, 0.75)
    q25 = _nanquantile(x, 0.25)
    if q75 is None or q25 is None:
        return None
    return float(q75 - q25)


def compute_tail_bias(
    *,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    bottom_q: float = 0.10,
    top_q: float = 0.90,
) -> tuple[float | None, float | None]:
    """Mean signed error in the bottom vs top decile of y_true.

    Returns (bottom_decile_mean_error, top_decile_mean_error) where
    error = y_pred - y_true.
    """

    q_lo = _nanquantile(y_true, bottom_q)
    q_hi = _nanquantile(y_true, top_q)
    if q_lo is None or q_hi is None:
        return (None, None)

    err = y_pred - y_true
    bottom_mask = np.isfinite(err) & np.isfinite(y_true) & (y_true <= q_lo)
    top_mask = np.isfinite(err) & np.isfinite(y_true) & (y_true >= q_hi)

    bottom = float(np.mean(err[bottom_mask])) if int(bottom_mask.sum()) else None
    top = float(np.mean(err[top_mask])) if int(top_mask.sum()) else None
    return (bottom, top)


def compute_dispersion_ratios(*, y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float | None, float | None]:
    """Compute dispersion ratios capturing scale compression.

    Returns (iqr_ratio, q90q10_ratio).
    """

    iqr_true = _iqr(y_true)
    iqr_pred = _iqr(y_pred)

    iqr_ratio: float | None
    if iqr_true is None or iqr_pred is None or iqr_true <= 0.0:
        iqr_ratio = None
    else:
        iqr_ratio = float(iqr_pred / iqr_true)

    q90_true = _nanquantile(y_true, 0.90)
    q10_true = _nanquantile(y_true, 0.10)
    q90_pred = _nanquantile(y_pred, 0.90)
    q10_pred = _nanquantile(y_pred, 0.10)

    q90q10_ratio: float | None
    if None in (q90_true, q10_true, q90_pred, q10_pred):
        q90q10_ratio = None
    else:
        denom = float(q90_true - q10_true)
        numerator = float(q90_pred - q10_pred)
        q90q10_ratio = None if denom <= 0.0 else float(numerator / denom)

    return (iqr_ratio, q90q10_ratio)


def compute_slice_spearman(
    *,
    df: pd.DataFrame,
    y_true_col: str,
    y_pred_col: str,
    slice_col: str,
) -> list[SliceSpearman]:
    if slice_col not in df.columns:
        return []

    out: list[SliceSpearman] = []
    for slice_value, g in df.groupby(slice_col, dropna=False):
        slice_str = "" if slice_value is None else str(slice_value)
        y_true = _as_float_array(g[y_true_col])
        y_pred = _as_float_array(g[y_pred_col])
        rho = _spearman(y_pred, y_true)
        mask = np.isfinite(y_true) & np.isfinite(y_pred)
        out.append(SliceSpearman(slice_value=slice_str, n=int(mask.sum()), spearman=rho))
    return out


def compute_slice_gini(
    *,
    df: pd.DataFrame,
    y_true_col: str,
    y_pred_col: str,
    slice_col: str,
) -> list[SliceGini]:
    if slice_col not in df.columns:
        return []

    out: list[SliceGini] = []
    for slice_value, g in df.groupby(slice_col, dropna=False):
        slice_str = "" if slice_value is None else str(slice_value)
        y_true = _as_float_array(g[y_true_col])
        y_pred = _as_float_array(g[y_pred_col])
        cindex = _cindex(y_pred, y_true)
        gini = None if cindex is None else gini_from_cindex(float(cindex))
        mask = np.isfinite(y_true) & np.isfinite(y_pred)
        out.append(SliceGini(slice_value=slice_str, n=int(mask.sum()), gini=gini))
    return out


def summarize_choice_validation_posthoc_metrics(
    *,
    df: pd.DataFrame,
    y_true_col: str = "y_true",
    y_pred_col: str = "y_pred",
    slice_col: str = "country",
) -> tuple[dict[str, Any], list[SliceSpearman]]:
    """Compute post-hoc diagnostics from canonical predictions.

    The returned dict is a single-row payload (safe to convert to CSV).
    Also returns per-slice Spearman results for optional detail output.
    """

    required = {y_true_col, y_pred_col}
    missing = [c for c in sorted(required) if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    y_true = _as_float_array(df[y_true_col])
    y_pred = _as_float_array(df[y_pred_col])

    valid_mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true_v = y_true[valid_mask]
    y_pred_v = y_pred[valid_mask]

    bottom_err, top_err = compute_tail_bias(y_true=y_true_v, y_pred=y_pred_v)
    iqr_ratio, q90q10_ratio = compute_dispersion_ratios(y_true=y_true_v, y_pred=y_pred_v)

    slice_spears = compute_slice_spearman(
        df=df,
        y_true_col=y_true_col,
        y_pred_col=y_pred_col,
        slice_col=slice_col,
    )

    slice_ginis = compute_slice_gini(
        df=df,
        y_true_col=y_true_col,
        y_pred_col=y_pred_col,
        slice_col=slice_col,
    )

    slice_vals = np.array([np.nan if s.spearman is None else float(s.spearman) for s in slice_spears], dtype=float)
    slice_valid = slice_vals[np.isfinite(slice_vals)]

    gini_vals = np.array([np.nan if s.gini is None else float(s.gini) for s in slice_ginis], dtype=float)
    gini_valid = gini_vals[np.isfinite(gini_vals)]

    summary: dict[str, Any] = {
        "n": int(valid_mask.sum()),
        "tail_bias_bottom_decile_mean_error": bottom_err,
        "tail_bias_top_decile_mean_error": top_err,
        "dispersion_ratio_iqr": iqr_ratio,
        "dispersion_ratio_q90q10": q90q10_ratio,
        "slice_col": slice_col,
        "n_slices": int(len(slice_spears)),
        "n_slices_valid": int(slice_valid.size),
        "slice_spearman_mean": float(np.mean(slice_valid)) if slice_valid.size else None,
        "slice_spearman_median": float(np.median(slice_valid)) if slice_valid.size else None,
        "slice_spearman_min": float(np.min(slice_valid)) if slice_valid.size else None,
        "slice_spearman_p25": float(np.quantile(slice_valid, 0.25)) if slice_valid.size else None,
        "slice_spearman_p75": float(np.quantile(slice_valid, 0.75)) if slice_valid.size else None,
        "slice_spearman_max": float(np.max(slice_valid)) if slice_valid.size else None,
        "slice_gini_mean": float(np.mean(gini_valid)) if gini_valid.size else None,
        "slice_gini_median": float(np.median(gini_valid)) if gini_valid.size else None,
        "slice_gini_min": float(np.min(gini_valid)) if gini_valid.size else None,
        "slice_gini_p25": float(np.quantile(gini_valid, 0.25)) if gini_valid.size else None,
        "slice_gini_p75": float(np.quantile(gini_valid, 0.75)) if gini_valid.size else None,
        "slice_gini_max": float(np.max(gini_valid)) if gini_valid.size else None,
    }

    return summary, slice_spears


def compute_y_decile_errors(
    *,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bins: int = 10,
) -> list[DecileError]:
    """Compute MAE and signed bias by y_true quantile bins.

    This is intentionally robust to discrete y_true (ties / repeated quantiles):
    bins whose quantile boundaries collapse may be empty.
    """

    if int(n_bins) <= 1:
        raise ValueError("n_bins must be > 1")

    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    yt = y_true[mask]
    yp = y_pred[mask]
    if yt.size == 0:
        return []

    qs = np.linspace(0.0, 1.0, int(n_bins) + 1)
    bounds = np.nanquantile(yt, qs)
    out: list[DecileError] = []

    err = yp - yt
    abs_err = np.abs(err)

    for i in range(int(n_bins)):
        q_lo = float(bounds[i]) if np.isfinite(bounds[i]) else None
        q_hi = float(bounds[i + 1]) if np.isfinite(bounds[i + 1]) else None
        if q_lo is None or q_hi is None:
            out.append(DecileError(decile=i + 1, q_lo=q_lo, q_hi=q_hi, n=0, mae=None, mean_error=None))
            continue

        if i < int(n_bins) - 1:
            sel = (yt >= q_lo) & (yt < q_hi)
        else:
            sel = (yt >= q_lo) & (yt <= q_hi)

        n = int(sel.sum())
        if n == 0:
            out.append(DecileError(decile=i + 1, q_lo=q_lo, q_hi=q_hi, n=0, mae=None, mean_error=None))
            continue

        out.append(
            DecileError(
                decile=i + 1,
                q_lo=q_lo,
                q_hi=q_hi,
                n=n,
                mae=float(np.mean(abs_err[sel])),
                mean_error=float(np.mean(err[sel])),
            )
        )

    return out


def compute_y_decile_errors_from_df(
    *,
    df: pd.DataFrame,
    y_true_col: str,
    y_pred_col: str,
    n_bins: int = 10,
) -> list[DecileError]:
    required = {y_true_col, y_pred_col}
    missing = [c for c in sorted(required) if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    y_true = _as_float_array(df[y_true_col])
    y_pred = _as_float_array(df[y_pred_col])
    return compute_y_decile_errors(y_true=y_true, y_pred=y_pred, n_bins=n_bins)
