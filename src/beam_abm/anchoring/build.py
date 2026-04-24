"""Anchor-based heterogeneity representation and memberships."""

from __future__ import annotations

import json
import math
import os
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
from beartype import beartype
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.impute import KNNImputer
from sklearn.neighbors import NearestNeighbors

from beam_abm.common.logging import get_logger
from beam_abm.decision_function.io import build_type_buckets, ensure_outdir, load_column_types, parse_list_arg
from beam_abm.decision_function.taxonomy import DEFAULT_COLUMN_TYPES_PATH
from beam_abm.settings import CSV_FILE_CLEAN, SEED

logger = get_logger(__name__)

DEFAULT_OUTDIR = Path(os.getenv("ANCHORS_OUTPUT_DIR", "empirical/output/anchors/system"))

NUMERIC_TYPES = {"continuous", "ordinal", "binary"}

COUNTRY_COL = "country"
ROW_ID_COL = "row_id"

NORMALIZATION_WITHIN_COUNTRY_ZSCORE = "within_country_zscore"
NORMALIZATION_DEMEAN_COUNTRY_THEN_GLOBAL_SCALE = "demean_country_then_global_scale"
NORMALIZATION_GLOBAL_SCALE_ONLY = "global_scale_only"
NORMALIZATION_DEFAULT = NORMALIZATION_DEMEAN_COUNTRY_THEN_GLOBAL_SCALE

DEFAULT_MISSINGNESS_THRESHOLD = 0.35
DEFAULT_ANCHOR_K = 10
DEFAULT_NEIGHBOR_K = 200
DEFAULT_GLOBALITY_MIN_COUNTRIES = 3
DEFAULT_GLOBALITY_PI_THRESHOLD = 0.05

DEFAULT_TAU_KNN_K = 10
DEFAULT_TAU_QUANTILE = 0.5
DEFAULT_TAU_SAMPLE_N = 5000

DEFAULT_PCA_COMPONENTS = 0
DEFAULT_PCA_VARIANCE_TARGET = 0.97

KNN_IMPUTER_NEIGHBORS = 10
EPS = 1e-12


__all__ = [
    "AnchorConfig",
    "AnchorResults",
    "DEFAULT_OUTDIR",
    "NORMALIZATION_DEFAULT",
    "NORMALIZATION_DEMEAN_COUNTRY_THEN_GLOBAL_SCALE",
    "NORMALIZATION_GLOBAL_SCALE_ONLY",
    "NORMALIZATION_WITHIN_COUNTRY_ZSCORE",
    "build_anchor_system",
    "build_anchor_neighborhoods",
    "default_core_features",
]


@dataclass(slots=True, frozen=False)
class AnchorConfig:
    input_csv: Path = Path(CSV_FILE_CLEAN)
    outdir: Path = DEFAULT_OUTDIR
    column_types_path: Path = DEFAULT_COLUMN_TYPES_PATH
    features: tuple[str, ...] = ()
    country_col: str = COUNTRY_COL
    normalization: str = NORMALIZATION_DEFAULT
    missingness_threshold: float = DEFAULT_MISSINGNESS_THRESHOLD
    anchor_k: int = DEFAULT_ANCHOR_K
    neighbor_k: int = DEFAULT_NEIGHBOR_K
    globality_min_countries: int = DEFAULT_GLOBALITY_MIN_COUNTRIES
    globality_pi_threshold: float = DEFAULT_GLOBALITY_PI_THRESHOLD
    globality_pi_factor: float | None = None
    pca_components: int = DEFAULT_PCA_COMPONENTS
    pca_variance_target: float | None = DEFAULT_PCA_VARIANCE_TARGET
    tau: float | None = None
    tau_method: str = "knn_quantile"
    tau_keff_target: float | None = None
    tau_knn_k: int = DEFAULT_TAU_KNN_K
    tau_quantile: float = DEFAULT_TAU_QUANTILE
    tau_sample_n: int = DEFAULT_TAU_SAMPLE_N
    max_anchor_attempts: int = 5
    seed: int = SEED


@dataclass(slots=True, frozen=True)
class AnchorResults:
    df: pl.DataFrame
    features: tuple[str, ...]
    X: np.ndarray
    X_dist: np.ndarray
    anchor_indices: np.ndarray
    anchor_coords: np.ndarray
    distances: np.ndarray
    weights: np.ndarray
    entropy: np.ndarray
    k_eff: np.ndarray
    tau: float
    pca: PCA | None
    pca_metadata: dict[str, object]
    country_weights: pl.DataFrame


def _dedupe(items: Iterable[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    deduped: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        deduped.append(item)
    return tuple(deduped)


def _core_feature_set() -> list[str]:
    return [
        "institutional_trust_avg",
        "fivec_confidence",
        "fivec_collective_responsibility",
        "fivec_low_constraints",
        "fivec_low_complacency",
        "fivec_calculation",
        "covax_legitimacy_scepticism_idx",
        "infection_likelihood_avg",
        "severity_if_infected_avg",
        "disease_fear_avg",
        "vaccine_risk_avg",
        "vaccine_fear_avg",
        "att_vaccines_importance",
        "fivec_pro_vax_idx",
    ]


def _additional_block_features() -> list[str]:
    return [
        "trust_who_vax_info",
        "trust_national_health_institutions_vax_info",
        "trust_national_gov_vax_info",
        "trust_local_health_authorities_vax_info",
        "trust_gp_vax_info",
        "trust_religious_leaders_vax_info",
        "trust_influencers_vax_info",
        "progressive_attitudes_nonvax_idx",
        "family_progressive_norms_nonvax_idx",
        "friends_progressive_norms_nonvax_idx",
        "colleagues_progressive_norms_nonvax_idx",
        "family_vaccines_importance",
        "friends_vaccines_importance",
        "colleagues_vaccines_importance",
        "mfq_relevance_individualizing_idx",
        "mfq_relevance_binding_idx",
        "mfq_agreement_individualizing_idx",
        "mfq_agreement_binding_idx",
    ]


def _npi_driver_feature_set() -> list[str]:
    return [
        "disease_fear_avg",
        "severity_if_infected_avg",
        "infection_likelihood_avg",
        "institutional_trust_avg",
        "fivec_confidence",
        "fivec_low_constraints",
        "fivec_low_complacency",
        "fivec_calculation",
        "mfq_relevance_individualizing_idx",
        "mfq_agreement_individualizing_idx",
        "mfq_relevance_binding_idx",
        "mfq_agreement_binding_idx",
        "progressive_attitudes_nonvax_idx",
    ]


def default_core_features() -> tuple[str, ...]:
    """Return the union of driver features used for anchor selection."""

    return _dedupe(_core_feature_set() + _additional_block_features() + _npi_driver_feature_set())


def _filter_features(
    features: Sequence[str],
    *,
    df: pl.DataFrame,
    column_types_path: Path,
) -> tuple[str, ...]:
    if not features:
        return ()
    column_types = load_column_types(column_types_path)
    type_buckets, _, _, _ = build_type_buckets(df, column_types, overrides=())
    numeric_cols: set[str] = set()
    for typ in NUMERIC_TYPES:
        numeric_cols.update(type_buckets.get(typ, []))

    available = set(df.columns)
    kept: list[str] = []
    for feature in features:
        if feature not in available:
            continue
        if feature not in numeric_cols:
            continue
        kept.append(feature)
    return tuple(kept)


@beartype
def _prepare_matrix(
    df: pl.DataFrame,
    features: Sequence[str],
    *,
    country_col: str,
    missingness_threshold: float,
    normalization: str,
) -> tuple[np.ndarray, tuple[str, ...], pl.DataFrame]:
    if country_col not in df.columns:
        raise ValueError(f"Missing required country column: {country_col}")

    n_rows = df.height
    if n_rows == 0:
        raise ValueError("Input dataset is empty")

    missingness_rows: list[dict[str, object]] = []
    kept: list[str] = []

    for feature in features:
        missing_count = int(df.select(pl.col(feature).is_null().sum()).item())
        missing_frac = float(missing_count / n_rows) if n_rows else 0.0
        dropped = missing_frac > missingness_threshold
        missingness_rows.append(
            {
                "feature": feature,
                "n_total": int(n_rows),
                "n_missing": missing_count,
                "missing_frac": missing_frac,
                "dropped": bool(dropped),
            }
        )
        if dropped:
            continue
        kept.append(feature)

    missingness_df = pl.DataFrame(missingness_rows).sort(["dropped", "missing_frac"], descending=[False, True])

    if not kept:
        raise ValueError(f"All anchor features exceeded missingness threshold ({missingness_threshold:.2f}).")

    exprs: list[pl.Expr] = []
    norm = str(normalization)
    for feature in kept:
        value = pl.col(feature).cast(pl.Float64, strict=False)
        if norm == NORMALIZATION_WITHIN_COUNTRY_ZSCORE:
            mean = value.mean().over(country_col)
            std = value.std().over(country_col)
            safe_std = pl.when(std.is_null() | (std == 0)).then(1.0).otherwise(std)
            exprs.append(((value - mean) / safe_std).alias(feature))
        elif norm == NORMALIZATION_DEMEAN_COUNTRY_THEN_GLOBAL_SCALE:
            demeaned = value - value.mean().over(country_col)
            std = demeaned.std()
            safe_std = pl.when(std.is_null() | (std == 0)).then(1.0).otherwise(std)
            exprs.append((demeaned / safe_std).alias(feature))
        elif norm == NORMALIZATION_GLOBAL_SCALE_ONLY:
            mean = value.mean()
            std = value.std()
            safe_std = pl.when(std.is_null() | (std == 0)).then(1.0).otherwise(std)
            exprs.append(((value - mean) / safe_std).alias(feature))
        else:
            raise ValueError(f"Unknown normalization: {normalization}")

    feature_df = df.select(exprs)
    X_z = feature_df.to_numpy().astype(float)
    return X_z, tuple(kept), missingness_df


@beartype
def _impute_within_country_knn(
    df: pl.DataFrame,
    *,
    X_z: np.ndarray,
    country_col: str,
    seed: int,
) -> np.ndarray:
    if X_z.shape[0] != df.height:
        raise ValueError("Row mismatch between standardized matrix and dataframe")

    countries = df[country_col].to_list()
    X_out = np.array(X_z, copy=True)

    index_by_country: dict[str, list[int]] = {}
    for i, c in enumerate(countries):
        key = "__MISSING__" if c is None else str(c)
        index_by_country.setdefault(key, []).append(i)

    rng = np.random.default_rng(seed)
    for _, idxs in index_by_country.items():
        block = X_out[idxs]
        if np.isnan(block).sum() == 0:
            continue

        rng.shuffle(idxs)
        k_neighbors = min(int(KNN_IMPUTER_NEIGHBORS), max(1, len(idxs) - 1))
        imputer = KNNImputer(n_neighbors=k_neighbors, weights="distance")
        X_out[idxs] = imputer.fit_transform(block)

    return np.nan_to_num(X_out, nan=0.0)


def _max_pca_components(X: np.ndarray) -> int:
    return int(max(1, min(X.shape[0], X.shape[1])))


def _safe_pca_components(X: np.ndarray, n_pca: int) -> int:
    max_comp = _max_pca_components(X)
    if n_pca > max_comp:
        logger.warning("Requested n_components exceeds max; clamping.")
        return max_comp
    return n_pca


def _fit_pca(
    X: np.ndarray,
    *,
    n_components: int,
    seed: int,
) -> tuple[np.ndarray, PCA, dict[str, object]]:
    n_components = _safe_pca_components(X, n_components)
    pca = PCA(n_components=n_components, random_state=seed)
    X_pca = pca.fit_transform(X)
    variance = pca.explained_variance_ratio_
    metadata = {
        "method": "components",
        "n_components": int(n_components),
        "variance_kept": float(np.sum(variance)),
        "variance_ratio": variance.tolist(),
        "variance_target": None,
    }
    return X_pca, pca, metadata


def _fit_pca_by_variance(
    X: np.ndarray,
    *,
    variance_target: float,
    seed: int,
) -> tuple[np.ndarray, PCA, dict[str, object]]:
    if not (0.0 < variance_target <= 1.0):
        raise ValueError("pca_variance_target must be in (0, 1].")
    pca = PCA(n_components=variance_target, svd_solver="full", random_state=seed)
    X_pca = pca.fit_transform(X)
    variance = pca.explained_variance_ratio_
    metadata = {
        "method": "variance_target",
        "n_components": int(pca.n_components_),
        "variance_kept": float(np.sum(variance)),
        "variance_ratio": variance.tolist(),
        "variance_target": float(variance_target),
    }
    return X_pca, pca, metadata


def _fit_kmedoids(X: np.ndarray, *, k: int, seed: int, max_iter: int = 200) -> tuple[np.ndarray, np.ndarray]:
    if X.size == 0:
        raise ValueError("Cannot fit k-medoids on empty matrix")

    n = X.shape[0]
    k = int(min(max(1, k), n))
    rng = np.random.default_rng(seed)
    medoids = rng.choice(n, size=k, replace=False)

    for _ in range(max_iter):
        dist_to_medoids = cdist(X, X[medoids], metric="euclidean")
        labels = np.argmin(dist_to_medoids, axis=1).astype(int)

        new_medoids = np.array(medoids, copy=True)
        for cluster in range(k):
            idx = np.where(labels == cluster)[0]
            if idx.size == 0:
                new_medoids[cluster] = int(rng.integers(0, n))
                continue
            sub = X[idx]
            dist = cdist(sub, sub, metric="euclidean")
            sums = np.sum(dist, axis=1)
            new_medoids[cluster] = int(idx[int(np.argmin(sums))])

        if np.array_equal(new_medoids, medoids):
            break
        medoids = new_medoids

    dist_to_medoids = cdist(X, X[medoids], metric="euclidean")
    labels = np.argmin(dist_to_medoids, axis=1).astype(int)
    return labels, medoids.astype(int)


def _compute_tau(
    X: np.ndarray,
    *,
    k: int,
    quantile: float,
    sample_n: int,
    seed: int,
) -> float:
    n = X.shape[0]
    if n <= 1:
        return 1.0

    rng = np.random.default_rng(seed)
    if n > sample_n:
        idx = rng.choice(n, size=sample_n, replace=False)
        X_use = X[idx]
    else:
        X_use = X

    k_eff = min(max(1, int(k)), max(1, X_use.shape[0] - 1))
    nbrs = NearestNeighbors(n_neighbors=k_eff + 1, metric="euclidean")
    nbrs.fit(X_use)
    distances, _ = nbrs.kneighbors(X_use, return_distance=True)
    kth = distances[:, k_eff]
    tau = float(np.quantile(kth, quantile))
    return max(tau, EPS)


def _find_tau_for_target_keff(
    distances: np.ndarray,
    *,
    target_keff: float,
    seed: int,
) -> float:
    rng = np.random.default_rng(seed)
    sample_n = min(5000, distances.shape[0])
    if distances.shape[0] > sample_n:
        idx = rng.choice(distances.shape[0], size=sample_n, replace=False)
        sample = distances[idx]
    else:
        sample = distances

    low = float(np.percentile(sample, 1))
    high = float(np.percentile(sample, 99))
    low = max(low, EPS)
    high = max(high, low * 10.0)

    def _median_keff(tau: float) -> float:
        w = _compute_memberships(sample, tau=tau)
        return float(np.nanmedian(_effective_anchors(w)))

    keff_low = _median_keff(low)
    keff_high = _median_keff(high)

    for _ in range(20):
        if keff_low <= target_keff <= keff_high:
            break
        if keff_low > target_keff:
            low = max(low * 0.2, EPS)
            keff_low = _median_keff(low)
        if keff_high < target_keff:
            high = high * 2.0
            keff_high = _median_keff(high)

    if not (keff_low <= target_keff <= keff_high):
        if abs(keff_low - target_keff) < abs(keff_high - target_keff):
            return float(low)
        return float(high)

    for _ in range(40):
        mid = math.sqrt(low * high)
        keff = _median_keff(mid)
        if keff < target_keff:
            low = mid
        else:
            high = mid
    return float(high)


def _compute_memberships(distances: np.ndarray, *, tau: float) -> np.ndarray:
    weights = np.exp(-distances / max(float(tau), EPS))
    row_sums = weights.sum(axis=1, keepdims=True)
    weights = np.where(row_sums > 0, weights / row_sums, 1.0 / float(weights.shape[1]))
    return weights


def _membership_entropy(weights: np.ndarray) -> np.ndarray:
    w = np.clip(weights, EPS, 1.0)
    return -np.sum(w * np.log(w), axis=1)


def _effective_anchors(weights: np.ndarray) -> np.ndarray:
    denom = np.sum(weights**2, axis=1)
    denom = np.where(denom > 0, denom, np.nan)
    return 1.0 / denom


def _country_weights(
    weights: np.ndarray,
    *,
    countries: Sequence[object],
) -> pl.DataFrame:
    data: dict[str, object] = {"country": list(countries)}
    for k in range(weights.shape[1]):
        data[f"anchor_{k}"] = weights[:, k].tolist()
    df = pl.DataFrame(data)
    grouped = df.group_by("country").mean()
    return grouped


def _anchor_country_share(
    weights: np.ndarray,
    *,
    countries: Sequence[object],
) -> pl.DataFrame:
    data: dict[str, object] = {"country": list(countries)}
    for k in range(weights.shape[1]):
        data[f"anchor_{k}"] = weights[:, k].tolist()
    df = pl.DataFrame(data)
    anchors = [col for col in df.columns if col.startswith("anchor_")]
    long = (
        df.melt(id_vars=["country"], value_vars=anchors, variable_name="anchor", value_name="weight")
        .with_columns(pl.col("anchor").str.replace("anchor_", "").cast(pl.Int64).alias("anchor_id"))
        .drop("anchor")
    )
    by_country = long.group_by(["anchor_id", "country"]).agg(pl.col("weight").sum().alias("weight_sum"))
    totals = by_country.group_by("anchor_id").agg(pl.col("weight_sum").sum().alias("total_weight"))
    return (
        by_country.join(totals, on="anchor_id", how="left")
        .with_columns(
            pl.when(pl.col("total_weight") > 0)
            .then(pl.col("weight_sum") / pl.col("total_weight"))
            .otherwise(0.0)
            .alias("share")
        )
        .sort(["anchor_id", "country"])
    )


def _long_country_weights(
    country_weights: pl.DataFrame,
) -> pl.DataFrame:
    anchors = [col for col in country_weights.columns if col.startswith("anchor_")]
    melted = country_weights.melt(id_vars=["country"], value_vars=anchors, variable_name="anchor", value_name="pi")
    return (
        melted.with_columns(pl.col("anchor").str.replace("anchor_", "").cast(pl.Int64).alias("anchor_id"))
        .drop("anchor")
        .sort(["country", "anchor_id"])
    )


def _globality_counts(
    country_weights_long: pl.DataFrame,
    *,
    pi_threshold: float,
) -> dict[int, int]:
    counts: dict[int, int] = {}
    if country_weights_long.height == 0:
        return counts
    for row in country_weights_long.iter_rows(named=True):
        anchor_id = int(row["anchor_id"])
        pi = float(row["pi"])
        if pi >= float(pi_threshold):
            counts[anchor_id] = counts.get(anchor_id, 0) + 1
    return counts


def _select_anchors(
    X: np.ndarray,
    *,
    k: int,
    seed: int,
    attempts: int,
    countries: Sequence[object],
    tau: float,
    globality_min_countries: int,
    globality_pi_threshold: float,
) -> np.ndarray:
    best_medoids: np.ndarray | None = None
    best_penalty = math.inf
    best_coverage = math.inf

    for attempt in range(max(1, attempts)):
        _, medoids = _fit_kmedoids(X, k=k, seed=seed + attempt)
        distances = cdist(X, X[medoids], metric="euclidean")
        weights = _compute_memberships(distances, tau=tau)
        cw_long = _long_country_weights(_country_weights(weights, countries=countries))
        counts = _globality_counts(cw_long, pi_threshold=globality_pi_threshold)
        penalty = sum(1 for anchor_id in range(len(medoids)) if counts.get(anchor_id, 0) < int(globality_min_countries))
        coverage = float(np.mean(np.min(distances, axis=1)))
        if penalty < best_penalty or (penalty == best_penalty and coverage < best_coverage):
            best_penalty = penalty
            best_coverage = coverage
            best_medoids = medoids

    if best_medoids is None:
        raise RuntimeError("Failed to select anchors")

    if best_penalty > 0:
        logger.warning(
            "Anchor selection did not satisfy globality for {n} anchors.",
            n=int(best_penalty),
        )
    return best_medoids


def build_anchor_neighborhoods(
    X_dist: np.ndarray,
    *,
    anchors: np.ndarray,
    countries: Sequence[object],
    row_ids: Sequence[int],
    neighbor_k: int,
) -> pl.DataFrame:
    if neighbor_k <= 0:
        return pl.DataFrame({"anchor_id": [], "country": [], "row_id": [], "rank": [], "dist": []})

    countries_arr = np.asarray(countries, dtype=object)
    row_ids_arr = np.asarray(row_ids, dtype=int)
    out_rows: list[dict[str, object]] = []

    for anchor_id, anchor_idx in enumerate(anchors):
        anchor_vec = X_dist[int(anchor_idx)]
        for country in np.unique(countries_arr):
            mask = countries_arr == country
            if not mask.any():
                continue
            idxs = np.where(mask)[0]
            block = X_dist[idxs]
            dist = np.linalg.norm(block - anchor_vec, axis=1)
            order = np.argsort(dist)
            take = min(int(neighbor_k), order.size)
            for rank, j in enumerate(order[:take], start=1):
                out_rows.append(
                    {
                        "anchor_id": int(anchor_id),
                        "country": country,
                        "row_id": int(row_ids_arr[idxs[int(j)]]),
                        "rank": int(rank),
                        "dist": float(dist[int(j)]),
                    }
                )
    return pl.DataFrame(out_rows)


@beartype
def _select_unique_eval_ids(
    neighborhoods: pl.DataFrame,
    *,
    per_anchor_n: int,
    candidate_k: int,
) -> pl.DataFrame:
    """Select a unique per-(anchor,country) eval sample.

    Guarantees no `row_id` repeats across anchors *within a country* by solving a min-cost flow
    assignment with costs based on `dist` (and deterministic tie-breakers).
    """

    required = {"anchor_id", "country", "row_id", "rank", "dist"}
    missing = required - set(neighborhoods.columns)
    if missing:
        raise ValueError(f"Neighborhoods missing required columns: {sorted(missing)}")

    if per_anchor_n <= 0:
        return neighborhoods.head(0)

    if candidate_k <= 0:
        raise ValueError("candidate_k must be > 0 to select eval profiles.")

    candidates = neighborhoods.filter(pl.col("rank") <= int(candidate_k))
    if candidates.is_empty():
        raise ValueError("No candidate neighborhoods available (rank filter removed all rows).")

    anchor_ids = sorted(int(x) for x in candidates.get_column("anchor_id").unique().to_list())
    countries_raw = candidates.get_column("country").unique().to_list()
    countries = sorted(countries_raw, key=lambda x: str(x))
    if not anchor_ids:
        raise ValueError("No anchors available in candidate neighborhoods.")
    if not countries:
        raise ValueError("No countries available in candidate neighborhoods.")

    try:
        import networkx as nx
    except ImportError as e:  # pragma: no cover
        raise ImportError("Unique eval sampling requires networkx (declared dependency).") from e

    selected_rows: list[dict[str, object]] = []

    for country in countries:
        sub = candidates.filter(pl.col("country") == country)
        counts = sub.group_by("anchor_id").len().rename({"len": "n"})
        counts_map = {int(r["anchor_id"]): int(r["n"]) for r in counts.iter_rows(named=True)}
        insufficient = sorted(a for a in anchor_ids if counts_map.get(int(a), 0) < int(per_anchor_n))
        if insufficient:
            raise ValueError(
                f"Insufficient candidate pool for country={country}: "
                f"anchors missing >= {int(per_anchor_n)} candidates: {insufficient}. "
                "Consider increasing neighbor_k (candidate pool) or reducing eval_per_group."
            )

        total_need = int(per_anchor_n) * len(anchor_ids)

        G = nx.DiGraph()
        source = "s"
        sink = "t"
        G.add_node(source, demand=-total_need)
        G.add_node(sink, demand=total_need)

        for anchor_id in anchor_ids:
            anchor_node = f"a{anchor_id}"
            G.add_node(anchor_node, demand=0)
            G.add_edge(source, anchor_node, capacity=int(per_anchor_n), weight=0)

        sub_sorted = sub.sort(["anchor_id", "rank", "row_id"])
        for row in sub_sorted.iter_rows(named=True):
            anchor_id = int(row["anchor_id"])
            row_id = int(row["row_id"])
            dist = float(row["dist"])
            rank = int(row["rank"])

            anchor_node = f"a{anchor_id}"
            row_node = f"r{row_id}"
            if row_node not in G:
                G.add_node(row_node, demand=0)
                G.add_edge(row_node, sink, capacity=1, weight=0)

            # Costs must be integers for network simplex; scale distances and add tie-break terms.
            weight = int(math.floor(dist * 1_000_000.0)) + rank * 1_000 + (row_id % 1_000)
            G.add_edge(anchor_node, row_node, capacity=1, weight=weight)

        try:
            flow = nx.min_cost_flow(G)
        except Exception as e:
            raise ValueError(
                f"Failed to compute unique eval assignment for country={country} "
                f"(anchors={len(anchor_ids)}, per_anchor_n={int(per_anchor_n)}, candidate_k={int(candidate_k)})."
            ) from e

        used_row_ids: set[int] = set()
        for anchor_id in anchor_ids:
            anchor_node = f"a{anchor_id}"
            assigned = [int(node[1:]) for node, amt in flow[anchor_node].items() if amt > 0]
            if len(assigned) != int(per_anchor_n):
                raise ValueError(
                    f"Solver returned {len(assigned)} assignments for anchor_id={anchor_id} country={country}, "
                    f"expected {int(per_anchor_n)}."
                )
            overlap = used_row_ids.intersection(assigned)
            if overlap:
                raise ValueError(
                    f"Solver produced duplicate row_ids across anchors for country={country}: {sorted(overlap)[:10]}"
                )
            used_row_ids.update(assigned)
            for rid in assigned:
                selected_rows.append({"anchor_id": int(anchor_id), "country": country, "row_id": int(rid)})

    selected = pl.DataFrame(selected_rows)
    return selected.join(
        candidates.select(["anchor_id", "country", "row_id", "rank", "dist"]),
        on=["anchor_id", "country", "row_id"],
        how="left",
    )


@beartype
def build_anchor_system(
    config: AnchorConfig,
    *,
    feature_list: str | None = None,
) -> AnchorResults:
    """Build anchors, memberships, and neighbor pools.

    Parameters
    ----------
    config:
        AnchorConfig defining inputs and hyperparameters.
    feature_list:
        Optional comma-separated list of features to override defaults.

    Returns
    -------
    AnchorResults
        In-memory results for downstream diagnostics or sampling.
    """

    ensure_outdir(config.outdir)
    df = pl.read_csv(config.input_csv, infer_schema_length=10_000).with_row_index(name=ROW_ID_COL)

    if feature_list:
        raw_features = parse_list_arg(feature_list)
    elif config.features:
        raw_features = config.features
    else:
        raw_features = default_core_features()

    features = _filter_features(raw_features, df=df, column_types_path=config.column_types_path)
    if not features:
        raise ValueError("No usable anchor features found after filtering.")

    X_z, kept, missingness_df = _prepare_matrix(
        df,
        features,
        country_col=config.country_col,
        missingness_threshold=config.missingness_threshold,
        normalization=config.normalization,
    )
    missingness_df.write_csv(config.outdir / "anchor_missingness.csv")

    X = _impute_within_country_knn(df, X_z=X_z, country_col=config.country_col, seed=config.seed)

    pca = None
    pca_metadata: dict[str, object] = {
        "method": "none",
        "n_components": 0,
        "variance_kept": None,
        "variance_ratio": None,
        "variance_target": None,
    }
    X_dist = X
    if config.pca_components and config.pca_components > 0:
        X_dist, pca, pca_metadata = _fit_pca(X, n_components=config.pca_components, seed=config.seed)
    elif config.pca_variance_target:
        X_dist, pca, pca_metadata = _fit_pca_by_variance(
            X, variance_target=float(config.pca_variance_target), seed=config.seed
        )
    if pca is not None:
        np.savez(
            config.outdir / "anchor_pca.npz",
            components=pca.components_,
            mean=pca.mean_,
            explained_variance_ratio=pca.explained_variance_ratio_,
        )

    countries = df[config.country_col].to_list()
    tau_init = (
        float(config.tau)
        if config.tau is not None
        else _compute_tau(
            X_dist,
            k=config.tau_knn_k,
            quantile=config.tau_quantile,
            sample_n=config.tau_sample_n,
            seed=config.seed,
        )
    )

    globality_pi_threshold = float(config.globality_pi_threshold)
    if config.globality_pi_factor is not None:
        globality_pi_threshold = float(config.globality_pi_factor) / max(1, int(config.anchor_k))

    anchors = _select_anchors(
        X_dist,
        k=config.anchor_k,
        seed=config.seed,
        attempts=config.max_anchor_attempts,
        countries=countries,
        tau=tau_init,
        globality_min_countries=config.globality_min_countries,
        globality_pi_threshold=globality_pi_threshold,
    )

    distances = cdist(X_dist, X_dist[anchors], metric="euclidean")
    if config.tau is not None:
        tau = tau_init
        tau_method = "fixed"
    elif config.tau_method.lower() == "target_keff":
        if config.tau_keff_target is None:
            raise ValueError("tau_keff_target required when tau_method=target_keff")
        tau = _find_tau_for_target_keff(distances, target_keff=float(config.tau_keff_target), seed=config.seed)
        tau_method = "target_keff"
    else:
        tau = tau_init
        tau_method = "knn_quantile"

    weights = _compute_memberships(distances, tau=tau)
    entropy = _membership_entropy(weights)
    k_eff = _effective_anchors(weights)

    country_weights = _country_weights(weights, countries=countries)
    country_weights_long = _long_country_weights(country_weights)
    country_weights_long.write_csv(config.outdir / "anchor_country_weights.csv")
    _anchor_country_share(weights, countries=countries).write_csv(config.outdir / "anchor_country_share.csv")

    anchor_coords = X_dist[anchors]
    coords_df = pl.DataFrame(anchor_coords, schema=[f"coord_{i + 1}" for i in range(anchor_coords.shape[1])])
    anchor_row_ids = df[ROW_ID_COL].to_numpy()[anchors].astype(int).tolist()
    anchor_countries = np.asarray(df[config.country_col].to_list(), dtype=object)[anchors].tolist()
    coords_df = coords_df.with_columns(
        pl.Series("anchor_id", list(range(anchor_coords.shape[0]))),
        pl.Series("row_id", anchor_row_ids),
        pl.Series("country", anchor_countries),
    )
    coords_df.select(
        ["anchor_id", "row_id", "country"] + [c for c in coords_df.columns if c.startswith("coord_")]
    ).write_csv(config.outdir / "anchor_coords.csv")

    row_ids = df[ROW_ID_COL].to_numpy().astype(int)
    membership_rows: list[dict[str, object]] = []
    max_weights = np.max(weights, axis=1)
    for i in range(weights.shape[0]):
        for k in range(weights.shape[1]):
            membership_rows.append(
                {
                    "row_id": int(row_ids[i]),
                    "anchor_id": int(k),
                    "weight": float(weights[i, k]),
                    "entropy": float(entropy[i]),
                    "k_eff": float(k_eff[i]),
                    "max_weight": float(max_weights[i]),
                }
            )
    pl.DataFrame(membership_rows).write_parquet(config.outdir / "anchor_memberships.parquet")

    neighborhoods = build_anchor_neighborhoods(
        X_dist,
        anchors=anchors,
        countries=countries,
        row_ids=row_ids,
        neighbor_k=config.neighbor_k,
    )
    neighborhoods.write_parquet(config.outdir / "anchor_neighborhoods.parquet")

    # Export a stable default evaluation sample for microvalidation.
    # This is derived deterministically from the neighborhood pool (top-K by rank).
    eval_root = config.outdir.parent / "eval"
    ensure_outdir(eval_root)
    eval_per_group = min(50, int(config.neighbor_k)) if int(config.neighbor_k) > 0 else 0
    if eval_per_group > 0:
        eval_ids = _select_unique_eval_ids(
            neighborhoods,
            per_anchor_n=int(eval_per_group),
            candidate_k=int(config.neighbor_k),
        )
        df_eval = df
        if config.country_col != "country":
            df_eval = df_eval.rename({config.country_col: "country"})
        if "country" in df_eval.columns:
            df_eval = df_eval.drop("country")

        eval_profiles = eval_ids.join(df_eval, on=ROW_ID_COL, how="left").with_columns(
            pl.format(
                "{}::{}::{}",
                pl.col("anchor_id"),
                pl.col("country"),
                pl.col(ROW_ID_COL),
            ).alias("eval_id")
        )
        eval_profiles.write_parquet(eval_root / "eval_profiles.parquet")

    metadata = {
        "input_csv": str(config.input_csv),
        "country_col": str(config.country_col),
        "row_id_col": ROW_ID_COL,
        "normalization": str(config.normalization),
        "missingness_threshold": float(config.missingness_threshold),
        "features": list(kept),
        "anchor_k": int(config.anchor_k),
        "neighbor_k": int(config.neighbor_k),
        "distance": "euclidean",
        "pca": pca_metadata,
        "pca_path": "anchor_pca.npz" if pca is not None else None,
        "tau": float(tau),
        "tau_method": str(tau_method),
        "tau_init": float(tau_init),
        "tau_keff_target": float(config.tau_keff_target) if config.tau_keff_target is not None else None,
        "tau_knn_k": int(config.tau_knn_k),
        "tau_quantile": float(config.tau_quantile),
        "tau_sample_n": int(config.tau_sample_n),
        "seed": int(config.seed),
        "globality": {
            "min_countries": int(config.globality_min_countries),
            "pi_threshold": float(config.globality_pi_threshold),
            "pi_threshold_effective": float(globality_pi_threshold),
            "pi_factor": float(config.globality_pi_factor) if config.globality_pi_factor is not None else None,
        },
    }
    (config.outdir / "anchor_metadata.json").write_text(json.dumps(metadata, indent=2))

    return AnchorResults(
        df=df,
        features=kept,
        X=X,
        X_dist=X_dist,
        anchor_indices=anchors,
        anchor_coords=anchor_coords,
        distances=distances,
        weights=weights,
        entropy=entropy,
        k_eff=k_eff,
        tau=tau,
        pca=pca,
        pca_metadata=pca_metadata,
        country_weights=country_weights_long,
    )
