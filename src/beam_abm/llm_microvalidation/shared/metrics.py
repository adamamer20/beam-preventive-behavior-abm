"""Alignment metric computation (DAR, NMAE, segment diagnostics, block alignment helpers)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import spearmanr


@dataclass(frozen=True)
class CI:
    low: float
    high: float

    def excludes_zero(self) -> bool:
        return (self.low > 0 and self.high > 0) or (self.low < 0 and self.high < 0)


def bootstrap_ci(values: np.ndarray, *, n_boot: int = 1000, alpha: float = 0.05, rng_seed: int = 0) -> CI:
    rng = np.random.default_rng(rng_seed)
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if values.size == 0:
        return CI(low=float("nan"), high=float("nan"))

    idx = rng.integers(0, values.size, size=(n_boot, values.size))
    means = np.mean(values[idx], axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return CI(low=float(lo), high=float(hi))


def perturbation_effect(ice_low: float, ice_high: float) -> float:
    return float(ice_high - ice_low)


def directional_agreement(
    *,
    pe_ref: float,
    pe_llm: float,
    ci_ref: CI,
    ci_llm: CI,
) -> tuple[bool | None, str]:
    """Directional agreement with CI-based no-signal ties.

    Returns (agreement, status) where agreement is:
    - True/False when both CIs exclude 0
    - None when no-signal (tie)
    """

    if not ci_ref.excludes_zero() or not ci_llm.excludes_zero():
        return None, "no_signal"

    s_ref = np.sign(pe_ref)
    s_llm = np.sign(pe_llm)
    return bool(s_ref == s_llm), "scored"


def dar(agreements: list[bool | None]) -> float:
    scored = [a for a in agreements if a is not None]
    if not scored:
        return float("nan")
    return float(np.mean(scored))


def segment_contrasts(ice_low: float, ice_mid: float, ice_high: float) -> tuple[float, float]:
    return float(ice_mid - ice_low), float(ice_high - ice_mid)


def curvature(ice_low: float, ice_mid: float, ice_high: float) -> float:
    return float(ice_high - 2 * ice_mid + ice_low)


def smae(ice_llm: np.ndarray, ice_ref: np.ndarray, *, denom: float) -> float:
    ice_llm = np.asarray(ice_llm, dtype=float)
    ice_ref = np.asarray(ice_ref, dtype=float)
    mask = ~(np.isnan(ice_llm) | np.isnan(ice_ref))
    if not np.any(mask):
        return float("nan")
    mae = float(np.mean(np.abs(ice_llm[mask] - ice_ref[mask])))
    return float(mae / denom) if denom > 0 else float("nan")


def spearman_alignment(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = ~(np.isnan(x) | np.isnan(y))
    if np.sum(mask) < 2:
        return float("nan")
    rho, _ = spearmanr(x[mask], y[mask])
    return float(rho)


def cindex_alignment(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Concordance index for continuous/ordinal outcomes.

    Computes the proportion of concordant pairs with 0.5 credit for ties in
    predictions. Returns NaN when fewer than 2 valid observations are available.
    """

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    if np.sum(mask) < 2:
        return float("nan")
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    unique_preds = np.unique(y_pred)
    pred_ranks = np.searchsorted(unique_preds, y_pred) + 1
    m = int(unique_preds.size)

    order = np.argsort(y_true, kind="mergesort")
    y_true = y_true[order]
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
    n = int(y_true.size)
    while i < n:
        j = i + 1
        while j < n and y_true[j] == y_true[i]:
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
        return float("nan")
    return float((concordant + 0.5 * ties) / total_pairs)


def gini_from_cindex(cindex: float) -> float:
    """Rescale concordance to the Gini coefficient (G = 2C - 1)."""

    if not np.isfinite(cindex):
        return float("nan")
    return float(2.0 * cindex - 1.0)


def gini_alignment(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Gini coefficient derived from concordance alignment (G = 2C - 1)."""

    return gini_from_cindex(cindex_alignment(y_true, y_pred))
