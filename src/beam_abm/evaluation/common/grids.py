"""Perturbation grid construction and on-manifold item realization."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors


@dataclass(frozen=True)
class StratumChoice:
    level: str
    key: str


def choose_primary_stratum(
    *,
    df: pd.DataFrame,
    primary_col: str,
    country_col: str,
    primary_value: Any,
    country: Any,
    min_n: int,
    primary_label: str = "anchor",
) -> tuple[StratumChoice, pd.DataFrame]:
    """Primary-stratum fallback: primary×country → country → pooled."""

    both_mask = (df[primary_col] == primary_value) & (df[country_col] == country)
    both = df[both_mask]
    if len(both) >= min_n:
        return StratumChoice(level=f"{primary_label}_country", key=f"{primary_value}|{country}"), both

    c_mask = df[country_col] == country
    c_df = df[c_mask]
    if len(c_df) >= min_n:
        return StratumChoice(level="country", key=str(country)), c_df

    return StratumChoice(level="pooled", key="pooled"), df


@dataclass(frozen=True)
class GridSpec:
    target: str
    grid_points: list[float]
    delta_x: float
    delta_z: float
    stratum_level: str
    stratum_key: str


@dataclass(frozen=True)
class PCAFit:
    pca: PCA
    mean_: np.ndarray
    scale_: np.ndarray
    pc_scores: np.ndarray


def fit_pca(
    *,
    df: pd.DataFrame,
    columns: list[str],
    n_components: int = 2,
) -> PCAFit:
    x = df[columns].to_numpy(dtype=float)
    mean_ = np.nanmean(x, axis=0)
    x = np.where(np.isnan(x), mean_, x)

    scale_ = np.nanstd(x, axis=0, ddof=1)
    scale_ = np.where(scale_ > 0, scale_, 1.0)
    xz = (x - mean_) / scale_

    pca = PCA(n_components=n_components, random_state=0)
    scores = pca.fit_transform(xz)
    return PCAFit(pca=pca, mean_=mean_, scale_=scale_, pc_scores=scores)


def three_point_iqr_grid(values: pd.Series) -> list[float]:
    q25, q50, q75 = np.nanquantile(values.to_numpy(dtype=float), [0.25, 0.5, 0.75])
    return [float(q25), float(q50), float(q75)]


def two_point_binary_grid() -> list[float]:
    return [0.0, 1.0]


def build_grid_for_target(
    *,
    df: pd.DataFrame,
    target_col: str,
    primary_col: str,
    country_col: str,
    primary_value: Any,
    country: Any,
    min_n: int = 200,
    is_binary: bool = False,
    primary_label: str = "anchor",
) -> GridSpec:
    choice, sdf = choose_primary_stratum(
        df=df,
        primary_col=primary_col,
        country_col=country_col,
        primary_value=primary_value,
        country=country,
        min_n=min_n,
        primary_label=primary_label,
    )

    series = sdf[target_col].astype(float)
    if is_binary:
        grid = two_point_binary_grid()
    else:
        grid = three_point_iqr_grid(series)

    # Track realized shock size and standardized delta
    dx = float(grid[-1] - grid[0])
    sd = float(np.nanstd(series.to_numpy(dtype=float), ddof=1))
    dz = float(dx / sd) if sd > 0 else float("nan")

    return GridSpec(
        target=target_col,
        grid_points=grid,
        delta_x=dx,
        delta_z=dz,
        stratum_level=choice.level,
        stratum_key=choice.key,
    )


def _calibrate_tau(distances: np.ndarray) -> float:
    # Choose tau so that ~90% of weight lies within the inner 90th percentile distances.
    if distances.size == 0:
        return 1.0
    p90 = np.quantile(distances, 0.9)
    if p90 <= 0:
        return 1.0

    # Solve for tau in exp(-p90/tau) ~ 0.1 -> tau ~ p90 / ln(10)
    return float(p90 / math.log(10.0))


def knn_realize_scalar(
    *,
    df: pd.DataFrame,
    score_col: str,
    x_star: float,
    k: int,
    rng: np.random.Generator | None = None,
) -> pd.Series:
    """Score-conditional kNN resampling for scalar constructs."""

    if rng is None:
        rng = np.random.default_rng(0)

    scores = df[score_col].astype(float).to_numpy()
    distances = np.abs(scores - float(x_star))

    order = np.argsort(distances)
    k = max(1, min(int(k), len(order)))
    idx = order[:k]

    d = distances[idx]
    tau = _calibrate_tau(d)
    weights = np.exp(-d / max(tau, 1e-9))
    weights = weights / weights.sum() if weights.sum() > 0 else np.ones_like(weights) / len(weights)

    chosen = int(rng.choice(idx, p=weights))
    return df.iloc[int(chosen)]


def knn_realize_pc(
    *,
    df: pd.DataFrame,
    pc_scores: np.ndarray,
    target_point: np.ndarray,
    k: int,
    rng: np.random.Generator | None = None,
) -> tuple[pd.Series, float]:
    """kNN realization in PCA space.

    Returns (row, nearest_distance).
    """

    if rng is None:
        rng = np.random.default_rng(0)

    if pc_scores.shape[0] != len(df):
        raise ValueError("pc_scores must align with df")

    nbrs = NearestNeighbors(n_neighbors=min(k, len(df)), metric="euclidean")
    nbrs.fit(pc_scores)
    distances, indices = nbrs.kneighbors(target_point.reshape(1, -1), return_distance=True)

    d = distances[0]
    idx = indices[0]
    tau = _calibrate_tau(d)
    weights = np.exp(-d / max(tau, 1e-9))
    weights = weights / weights.sum() if weights.sum() > 0 else np.ones_like(weights) / len(weights)
    chosen = int(rng.choice(idx, p=weights))
    return df.iloc[chosen], float(d[0])


def estimate_within_stratum_nn_distances(
    *,
    points: np.ndarray,
    sample_n: int = 2000,
    rng_seed: int = 0,
) -> np.ndarray:
    """Estimate within-stratum nearest-neighbor distances.

    Used for out-of-manifold diagnostics. Uses a random subsample to keep runtime bounded.
    """

    points = np.asarray(points, dtype=float)
    if points.ndim == 1:
        points = points.reshape(-1, 1)

    n = points.shape[0]
    if n < 2:
        return np.array([], dtype=float)

    rng = np.random.default_rng(rng_seed)
    if n > sample_n:
        idx = rng.choice(n, size=sample_n, replace=False)
        pts = points[idx]
    else:
        pts = points

    nbrs = NearestNeighbors(n_neighbors=2, metric="euclidean")
    nbrs.fit(pts)
    distances, _ = nbrs.kneighbors(pts)
    # distances[:, 0] is self (0), distances[:, 1] is nearest other
    return distances[:, 1].astype(float)

