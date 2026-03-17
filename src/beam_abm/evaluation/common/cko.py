"""Conditional Knock-Out (CKO) utilities.

CKO importance for a block b is defined as the expected increase in loss when the
block features x_b are replaced by an on-manifold kNN draw from p(x_b | x_-b).

This module provides the on-manifold replacement mechanism; model-specific
prediction and loss wiring is handled by the calling scripts.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


@dataclass(frozen=True)
class CKODraw:
    donor_index: int
    donor_distance: float


def _zscore_matrix(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean_ = np.nanmean(x, axis=0)
    x = np.where(np.isnan(x), mean_, x)
    sd_ = np.nanstd(x, axis=0, ddof=1)
    sd_ = np.where(sd_ > 0, sd_, 1.0)
    return (x - mean_) / sd_, mean_, sd_


def conditional_knockout_draw(
    *,
    df: pd.DataFrame,
    row_index: int,
    block_cols: list[str],
    conditioning_cols: list[str],
    k: int = 25,
    rng: np.random.Generator | None = None,
) -> CKODraw:
    """Pick a donor row for block replacement using kNN on conditioning columns."""

    if rng is None:
        rng = np.random.default_rng(0)

    x = df[conditioning_cols].to_numpy(dtype=float)
    xz, _, _ = _zscore_matrix(x)

    n = xz.shape[0]
    if n < 2:
        return CKODraw(donor_index=row_index, donor_distance=0.0)

    nbrs = NearestNeighbors(n_neighbors=min(max(2, k), n), metric="euclidean")
    nbrs.fit(xz)

    distances, indices = nbrs.kneighbors(xz[row_index].reshape(1, -1), return_distance=True)
    d = distances[0]
    idx = indices[0]

    # Exclude self if present at position 0.
    if int(idx[0]) == int(row_index) and len(idx) > 1:
        d = d[1:]
        idx = idx[1:]

    if len(idx) == 0:
        return CKODraw(donor_index=row_index, donor_distance=0.0)

    # Softmax-like weights to bias toward closer donors.
    tau = float(np.quantile(d, 0.9)) if len(d) else 1.0
    tau = max(tau / np.log(10.0), 1e-9)
    w = np.exp(-d / tau)
    w = w / w.sum() if w.sum() > 0 else np.ones_like(w) / len(w)

    chosen = int(rng.choice(idx, p=w))
    pos = int(np.where(idx == chosen)[0][0]) if np.any(idx == chosen) else 0
    return CKODraw(donor_index=chosen, donor_distance=float(d[pos]))


def apply_block_replacement(
    *,
    base_attributes: dict,
    donor_attributes: dict,
    block_cols: list[str],
) -> dict:
    out = dict(base_attributes)
    for c in block_cols:
        if c in donor_attributes:
            out[c] = donor_attributes[c]
    return out
