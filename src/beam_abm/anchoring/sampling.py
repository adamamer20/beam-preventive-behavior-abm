"""Sampling utilities and perturbation effects for anchor systems."""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
from beartype import beartype
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import NearestNeighbors
from xgboost import XGBRegressor

from beam_abm.anchoring.build import _impute_within_country_knn, _prepare_matrix, build_anchor_neighborhoods
from beam_abm.common.logging import get_logger
from beam_abm.empirical.io import ensure_outdir
from beam_abm.empirical.missingness import (
    DEFAULT_MISSINGNESS_THRESHOLD,
    apply_missingness_plan_pl,
    build_missingness_plan_pl,
    drop_nonfinite_rows_pl,
    missingness_indicator_name,
)
from beam_abm.settings import CSV_FILE_CLEAN, SEED

logger = get_logger(__name__)


@dataclass(slots=True, frozen=False)
class PerturbationConfig:
    input_csv: Path = Path(CSV_FILE_CLEAN)
    anchors_dir: Path = Path("empirical/output/anchors/system")
    outdir: Path | None = None
    country_col: str = "country"
    target_col: str = "protective_behaviour_avg"
    driver_cols: tuple[str, ...] = ()
    levers: tuple[str, ...] = ()
    neighborhood_features: tuple[str, ...] = ()
    neighbor_k: int = 200
    min_n: int = 200
    oom_percentile: float = 0.95
    propensity_slices_path: Path | None = None
    propensity_outcome: str | None = None
    propensity_slice_col: str = "slice"
    predictor_missingness_threshold: float = DEFAULT_MISSINGNESS_THRESHOLD
    seed: int = SEED
    fallback_min_n: int = 300
    fallback_min_delta_z: float = 0.25
    fallback_quantiles: tuple[float, float] = (0.10, 0.90)
    fallback_quantiles_wide: tuple[float, float] = (0.05, 0.95)
    fallback_nonzero_quantile: float = 0.50
    ref_models: tuple[str, ...] = ("linear",)
    eval_profiles_path: Path | None = None
    compute_eval_off_manifold: bool = False
    off_manifold_method: str = "distance_nn"
    conditional_tail_alpha: float = 0.05
    conditional_tail_knn_k: int = 200
    xgb_n_estimators: int = 300
    xgb_max_depth: int = 4
    xgb_learning_rate: float = 0.05
    xgb_subsample: float = 0.8
    xgb_colsample_bytree: float = 0.8


@dataclass(frozen=True)
class _ShockStats:
    q25: float
    q50: float
    q75: float
    sigma_x: float
    x_min: float
    x_max: float
    delta_x: float
    delta_z: float


@dataclass(frozen=True)
class _ConditionalTailModel:
    feat_idx: int
    y_train: np.ndarray
    nn_model: NearestNeighbors | None


def _fit_conditional_tail_model(
    X_core: np.ndarray,
    *,
    stratum_indices: np.ndarray,
    feat_idx: int,
    knn_k: int,
) -> _ConditionalTailModel | None:
    if stratum_indices.size == 0:
        return None
    y = X_core[stratum_indices, feat_idx].astype(float)
    finite = np.isfinite(y)
    if not finite.any():
        return None

    y_train = y[finite]
    z_train = np.delete(X_core[stratum_indices[finite]], feat_idx, axis=1)
    if z_train.shape[1] == 0:
        return _ConditionalTailModel(feat_idx=feat_idx, y_train=y_train, nn_model=None)

    k_eff = max(1, min(int(knn_k), int(z_train.shape[0])))
    nn = NearestNeighbors(n_neighbors=k_eff, metric="euclidean")
    nn.fit(z_train)
    return _ConditionalTailModel(feat_idx=feat_idx, y_train=y_train, nn_model=nn)


def _conditional_tail_stats(
    model: _ConditionalTailModel,
    *,
    X_core: np.ndarray,
    query_indices: np.ndarray,
    x_star: float,
    tail_alpha: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if query_indices.size == 0:
        empty = np.array([], dtype=float)
        return empty, empty, np.array([], dtype=bool)

    if model.nn_model is None:
        p = float(np.mean(model.y_train <= x_star))
        percentile = np.full(query_indices.size, p, dtype=float)
    else:
        z_query = np.delete(X_core[query_indices], model.feat_idx, axis=1)
        _, nn_idx = model.nn_model.kneighbors(z_query, return_distance=True)
        neigh_vals = model.y_train[nn_idx]
        percentile = np.mean(neigh_vals <= x_star, axis=1).astype(float)

    percentile = np.clip(percentile, 0.0, 1.0)
    tail = 2.0 * np.minimum(percentile, 1.0 - percentile)
    is_off = tail < float(tail_alpha)
    return percentile, tail, is_off


def _load_anchor_metadata(anchors_dir: Path) -> dict[str, object]:
    meta_path = anchors_dir / "anchor_metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing anchor metadata: {meta_path}")
    return json.loads(meta_path.read_text(encoding="utf-8"))


def _prepare_model_matrix(
    df: pl.DataFrame, *, target_col: str, driver_cols: Sequence[str]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    df = drop_nonfinite_rows_pl(df, [*driver_cols, target_col], context=f"anchor_pe:{target_col}")
    if df.is_empty():
        raise ValueError(f"No finite rows left after dropping missing values for {target_col}.")
    X = df.select(driver_cols).to_numpy().astype(float)
    y = df.get_column(target_col).to_numpy().astype(float)
    means = np.nanmean(X, axis=0)
    X = np.where(np.isnan(X), means, X)
    return X, y, means


def _fit_xgb_model(
    X: np.ndarray,
    y: np.ndarray,
    *,
    seed: int,
    n_estimators: int,
    max_depth: int,
    learning_rate: float,
    subsample: float,
    colsample_bytree: float,
) -> XGBRegressor:
    model = XGBRegressor(
        n_estimators=int(n_estimators),
        max_depth=int(max_depth),
        learning_rate=float(learning_rate),
        subsample=float(subsample),
        colsample_bytree=float(colsample_bytree),
        objective="reg:squarederror",
        random_state=int(seed),
        n_jobs=-1,
    )
    model.fit(X, y)
    return model


def _sanitize_dummy_name(value: str) -> str:
    cleaned = re.sub(r"[^0-9a-zA-Z]+", "_", str(value).strip())
    cleaned = cleaned.strip("_").lower()
    return cleaned or "missing"


def _country_dummy_specs(
    df: pl.DataFrame,
    *,
    country_col: str,
) -> tuple[tuple[str, str], ...]:
    if country_col not in df.columns:
        return ()
    raw = df.get_column(country_col).unique().to_list()
    cats = sorted({"__MISSING__" if c is None else str(c) for c in raw})
    if len(cats) <= 1:
        return ()
    # Drop a baseline category to avoid perfect multicollinearity with the intercept.
    cats = cats[1:]
    return tuple((cat, f"{country_col}__{_sanitize_dummy_name(cat)}") for cat in cats)


def _with_country_dummies(
    df: pl.DataFrame,
    *,
    country_col: str,
    specs: Sequence[tuple[str, str]],
) -> pl.DataFrame:
    if not specs or country_col not in df.columns:
        return df
    c = pl.col(country_col).cast(pl.Utf8, strict=False).fill_null("__MISSING__")
    exprs = [(c == cat).cast(pl.Float64).alias(name) for cat, name in specs]
    return df.with_columns(exprs)


def _estimate_within_stratum_nn_distances(points: np.ndarray, *, sample_n: int, seed: int) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    if points.ndim == 1:
        points = points.reshape(-1, 1)
    n = points.shape[0]
    if n < 2:
        return np.array([], dtype=float)

    rng = np.random.default_rng(seed)
    if n > sample_n:
        idx = rng.choice(n, size=sample_n, replace=False)
        pts = points[idx]
    else:
        pts = points

    nbrs = NearestNeighbors(n_neighbors=2, metric="euclidean")
    nbrs.fit(pts)
    distances, _ = nbrs.kneighbors(pts)
    return distances[:, 1].astype(float)


def _read_table(path: Path) -> pl.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pl.read_parquet(path)
    return pl.read_csv(path, infer_schema_length=10_000)


def _load_propensity_slices(
    path: Path,
    *,
    outcome: str | None,
    slice_col: str,
) -> pl.DataFrame:
    df = _read_table(path)
    expected = {"row_id", slice_col}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Propensity scores missing required columns: {sorted(missing)}")
    if outcome and "outcome" in df.columns:
        df = df.filter(pl.col("outcome") == outcome)
    return df.select(["row_id", slice_col])


def _resolve_stratum_id(
    *,
    country: object,
    slice_label: object | None,
    min_n: int,
    counts: dict[tuple[object | None, object | None], int],
) -> tuple[object | None, object | None]:
    if slice_label is not None:
        stratum_id = (country, slice_label)
        if counts.get(stratum_id, 0) >= min_n:
            return stratum_id
    country_id = (country, None)
    if counts.get(country_id, 0) >= min_n:
        return country_id
    return (None, None)


def _stratum_level(stratum_id: tuple[object | None, object | None]) -> str:
    country, slice_label = stratum_id
    if country is None:
        return "pooled"
    if slice_label is None:
        return "country"
    return "country_slice"


def _stratum_key(stratum_id: tuple[object | None, object | None]) -> str:
    country, slice_label = stratum_id
    if country is None:
        return "pooled"
    if slice_label is None:
        return str(country)
    return f"{country}|{slice_label}"


def _compute_shock_stats(
    df: pl.DataFrame,
    *,
    levers: Sequence[str],
    country_col: str,
    slice_col: str | None,
    fallback_min_n: int,
    fallback_min_delta_z: float,
    fallback_quantiles: tuple[float, float],
    fallback_quantiles_wide: tuple[float, float],
    fallback_nonzero_quantile: float,
) -> tuple[
    dict[tuple[object | None, object | None], dict[str, _ShockStats]],
    dict[tuple[object | None, object | None], int],
]:
    stats: dict[tuple[object | None, object | None], dict[str, _ShockStats]] = {}
    counts: dict[tuple[object | None, object | None], int] = {}

    def _build_stats(sub: pl.DataFrame, key: tuple[object | None, object | None]) -> None:
        counts[key] = int(sub.height)
        lever_stats: dict[str, _ShockStats] = {}
        for lever in levers:
            values = sub.get_column(lever).to_numpy().astype(float)
            if values.size == 0 or not np.isfinite(values).any():
                lever_stats[lever] = _ShockStats(
                    float("nan"),
                    float("nan"),
                    float("nan"),
                    float("nan"),
                    float("nan"),
                    float("nan"),
                    float("nan"),
                    float("nan"),
                )
                continue
            finite = values[np.isfinite(values)]
            q25, q50, q75 = np.nanquantile(finite, [0.25, 0.5, 0.75])
            x_min = float(np.nanmin(finite))
            x_max = float(np.nanmax(finite))
            delta_x = float(q75 - q25)
            sd = float(np.nanstd(finite, ddof=1))
            delta_z = float(delta_x / sd) if sd > 0 else float("nan")

            n_unique = int(np.unique(finite).size)
            n_obs = finite.size

            def _quantile_pair(q_lo: float, q_hi: float, finite=finite) -> tuple[float, float]:
                lo, hi = np.nanquantile(finite, [q_lo, q_hi])
                return float(lo), float(hi)

            def _apply_fallback(finite=finite) -> tuple[float, float]:
                if (finite == 0).any():
                    nonzero = finite[finite != 0]
                    if nonzero.size:
                        hi = float(np.nanquantile(nonzero, fallback_nonzero_quantile))
                        if np.isfinite(hi) and hi != 0:
                            return 0.0, hi
                lo, hi = _quantile_pair(*fallback_quantiles)
                if lo == hi:
                    lo, hi = _quantile_pair(*fallback_quantiles_wide)
                return lo, hi

            if not np.isfinite(delta_x) or delta_x == 0.0:
                q25, q75 = _apply_fallback()
            elif n_unique <= 3 and delta_x == 0.0:
                q25, q75 = _apply_fallback()
            elif n_obs < fallback_min_n and np.isfinite(delta_z) and delta_z < fallback_min_delta_z:
                q25, q75 = _apply_fallback()

            delta_x = float(q75 - q25)
            delta_z = float(delta_x / sd) if sd > 0 else float("nan")
            lever_stats[lever] = _ShockStats(
                float(q25),
                float(q50),
                float(q75),
                sd,
                x_min,
                x_max,
                delta_x,
                delta_z,
            )
        stats[key] = lever_stats

    _build_stats(df, (None, None))

    for country in df.get_column(country_col).unique().to_list():
        sub = df.filter(pl.col(country_col) == country)
        _build_stats(sub, (country, None))

    if slice_col and slice_col in df.columns:
        sliced = df.filter(pl.col(slice_col).is_not_null())
        grouped = sliced.group_by([country_col, slice_col]).agg(pl.len().alias("n"))
        for row in grouped.iter_rows(named=True):
            country = row[country_col]
            slice_label = row[slice_col]
            sub = df.filter((pl.col(country_col) == country) & (pl.col(slice_col) == slice_label))
            _build_stats(sub, (country, slice_label))

    return stats, counts


def _compute_oom_thresholds(
    X_dist: np.ndarray,
    *,
    indices_by_stratum: dict[tuple[object | None, object | None], np.ndarray],
    seed: int,
    percentile: float,
) -> dict[tuple[object | None, object | None], float]:
    thresholds: dict[tuple[object | None, object | None], float] = {}
    for stratum_id, indices in indices_by_stratum.items():
        if indices.size == 0:
            thresholds[stratum_id] = float("inf")
            continue
        distances = _estimate_within_stratum_nn_distances(
            X_dist[indices],
            sample_n=2000,
            seed=seed,
        )
        thresholds[stratum_id] = float(np.quantile(distances, percentile)) if distances.size else float("inf")
    return thresholds


def _ensure_eval_ids(profiles: pl.DataFrame, *, country_col: str) -> pl.DataFrame:
    if "eval_id" in profiles.columns:
        return profiles
    if "row_id" in profiles.columns and country_col in profiles.columns:
        return profiles.with_columns(
            pl.concat_str(
                [
                    pl.lit("eval::"),
                    pl.col(country_col).cast(pl.Utf8, strict=False).fill_null("unknown"),
                    pl.lit("::"),
                    pl.col("row_id").cast(pl.Utf8),
                ]
            ).alias("eval_id")
        )
    if "row_id" in profiles.columns:
        return profiles.with_columns(pl.concat_str([pl.lit("eval::"), pl.col("row_id").cast(pl.Utf8)]).alias("eval_id"))
    return (
        profiles.with_row_index("__eval_idx")
        .with_columns(pl.concat_str([pl.lit("eval::"), pl.col("__eval_idx").cast(pl.Utf8)]).alias("eval_id"))
        .drop("__eval_idx")
    )


@beartype
def run_anchor_pe(config: PerturbationConfig) -> None:
    """Compute perturbation effects with anchor-based stratification.

    Parameters
    ----------
    config:
        PerturbationConfig defining inputs, levers, and thresholds.

    Returns
    -------
    None
        Writes PE summaries to disk.
    """

    anchors_dir = config.anchors_dir
    outdir = config.outdir or anchors_dir
    ensure_outdir(outdir)
    if config.off_manifold_method not in {"distance_nn", "conditional_tail"}:
        raise ValueError(
            "off_manifold_method must be one of {'distance_nn', 'conditional_tail'}; "
            f"got: {config.off_manifold_method}"
        )

    meta = _load_anchor_metadata(anchors_dir)
    features = tuple(meta.get("features", []))
    if not features:
        raise ValueError("anchor_metadata.json missing features list")

    df_raw = pl.read_csv(config.input_csv, infer_schema_length=10_000).with_row_index(name="row_id")
    df_stats = df_raw

    if not config.driver_cols:
        driver_cols = features
    else:
        driver_cols = config.driver_cols

    if not config.levers:
        levers = driver_cols
    else:
        levers = config.levers

    if config.target_col not in df_raw.columns:
        raise ValueError(f"Target column not in dataset: {config.target_col}")

    for lever in levers:
        if lever not in df_raw.columns:
            raise ValueError(f"Lever column not in dataset: {lever}")
        if lever not in driver_cols:
            raise ValueError(f"Lever column not in driver cols: {lever}")
    for col in driver_cols:
        if col not in df_raw.columns:
            raise ValueError(f"Driver column not in dataset: {col}")

    missingness_plan = build_missingness_plan_pl(
        df_raw,
        predictors=driver_cols,
        threshold=float(config.predictor_missingness_threshold),
        country_col=config.country_col,
    )
    df_model = apply_missingness_plan_pl(df_raw, missingness_plan, country_col=config.country_col)

    country_specs = _country_dummy_specs(df_raw, country_col=config.country_col)
    df_for_model = _with_country_dummies(df_model, country_col=config.country_col, specs=country_specs)
    indicator_cols = tuple(
        missingness_indicator_name(col)
        for col in missingness_plan.columns
        if missingness_indicator_name(col) in df_for_model.columns
    )
    model_cols = tuple(driver_cols) + indicator_cols + tuple(name for _, name in country_specs)
    X_model, y_model, means = _prepare_model_matrix(
        df_for_model,
        target_col=config.target_col,
        driver_cols=model_cols,
    )

    ref_models = tuple(m.strip().lower() for m in config.ref_models if str(m).strip())
    if not ref_models:
        ref_models = ("linear",)

    models: dict[str, object] = {}
    # Always fit the linear baseline for PE reference computations.
    linear_model = LinearRegression()
    linear_model.fit(X_model, y_model)
    models["linear"] = linear_model

    for ref_model in ref_models:
        if ref_model in {"linear", "lin"}:
            continue
        if ref_model in {"xgb", "xgboost"}:
            models["xgb"] = _fit_xgb_model(
                X_model,
                y_model,
                seed=config.seed,
                n_estimators=config.xgb_n_estimators,
                max_depth=config.xgb_max_depth,
                learning_rate=config.xgb_learning_rate,
                subsample=config.xgb_subsample,
                colsample_bytree=config.xgb_colsample_bytree,
            )
            continue
        raise ValueError(f"Unsupported ref model: {ref_model}")

    X_z, _, _missingness = _prepare_matrix(
        df_raw,
        features,
        country_col=config.country_col,
        missingness_threshold=float(meta.get("missingness_threshold", 0.35)),
        normalization=str(meta.get("normalization", "demean_country_then_global_scale")),
    )
    X_core = _impute_within_country_knn(df_raw, X_z=X_z, country_col=config.country_col, seed=config.seed)
    X_dist = X_core
    pca_path = meta.get("pca_path")
    pca_components = None
    pca_mean = None
    if pca_path:
        pca_file = anchors_dir / str(pca_path)
        if pca_file.exists():
            pca_data = np.load(pca_file)
            pca_components = pca_data["components"]
            pca_mean = pca_data["mean"]
            X_dist = (X_core - pca_mean) @ pca_components.T
        else:
            logger.warning("PCA file not found at {path}; using full space.", path=pca_file)

    row_ids = df_raw.get_column("row_id").to_numpy().astype(int)
    row_index = {int(rid): idx for idx, rid in enumerate(row_ids)}

    if config.neighborhood_features:
        neigh_features = config.neighborhood_features
        X_neigh_z, _, _ = _prepare_matrix(
            df_raw,
            neigh_features,
            country_col=config.country_col,
            missingness_threshold=float(meta.get("missingness_threshold", 0.35)),
            normalization=str(meta.get("normalization", "demean_country_then_global_scale")),
        )
        X_neigh = _impute_within_country_knn(df_raw, X_z=X_neigh_z, country_col=config.country_col, seed=config.seed)
        coords = pl.read_csv(anchors_dir / "anchor_coords.csv")
        anchor_row_ids = coords.get_column("row_id").to_numpy().astype(int)
        anchor_indices = np.asarray([row_index[int(rid)] for rid in anchor_row_ids if int(rid) in row_index], dtype=int)
        neighborhoods = build_anchor_neighborhoods(
            X_neigh,
            anchors=anchor_indices,
            countries=df_raw[config.country_col].to_list(),
            row_ids=row_ids,
            neighbor_k=config.neighbor_k,
        )
    else:
        neighborhoods = pl.read_parquet(anchors_dir / "anchor_neighborhoods.parquet")

    slice_col: str | None = None
    if config.propensity_slices_path:
        slices_df = _load_propensity_slices(
            config.propensity_slices_path,
            outcome=config.propensity_outcome,
            slice_col=config.propensity_slice_col,
        )
        neighborhoods = neighborhoods.join(slices_df, on="row_id", how="left")
        df_stats = df_stats.join(slices_df, on="row_id", how="left")
        slice_col = config.propensity_slice_col

    anchor_ids = neighborhoods.get_column("anchor_id").unique().to_list()
    countries = neighborhoods.get_column("country").unique().to_list()

    shock_stats, stratum_counts = _compute_shock_stats(
        df_stats,
        levers=levers,
        country_col=config.country_col,
        slice_col=slice_col,
        fallback_min_n=config.fallback_min_n,
        fallback_min_delta_z=config.fallback_min_delta_z,
        fallback_quantiles=config.fallback_quantiles,
        fallback_quantiles_wide=config.fallback_quantiles_wide,
        fallback_nonzero_quantile=config.fallback_nonzero_quantile,
    )

    filtered_levers: list[str] = []
    dropped_levers: list[str] = []
    countries_list = df_stats.get_column(config.country_col).unique().to_list()
    for lever in levers:
        deltas: list[float] = []
        for country in countries_list:
            stratum_id = _resolve_stratum_id(
                country=country,
                slice_label=None,
                min_n=config.min_n,
                counts=stratum_counts,
            )
            stats = shock_stats.get(stratum_id, {}).get(lever)
            if stats is None:
                continue
            if np.isfinite(stats.delta_x):
                deltas.append(float(stats.delta_x))
        if not deltas or all(abs(dx) <= 0 for dx in deltas):
            dropped_levers.append(lever)
        else:
            filtered_levers.append(lever)
    if dropped_levers:
        logger.warning(
            "Dropping levers with zero shock size across countries: {levers}",
            levers=", ".join(dropped_levers),
        )
    levers = tuple(filtered_levers)

    indices_by_stratum: dict[tuple[object | None, object | None], np.ndarray] = {
        (None, None): np.arange(len(row_ids), dtype=int)
    }
    for country in df_stats.get_column(config.country_col).unique().to_list():
        idx = np.where(df_stats.get_column(config.country_col).to_numpy() == country)[0]
        indices_by_stratum[(country, None)] = idx.astype(int)
    if slice_col:
        sliced = df_stats.filter(pl.col(slice_col).is_not_null())
        grouped = sliced.group_by([config.country_col, slice_col]).agg(pl.col("row_id").alias("row_ids"))
        for row in grouped.iter_rows(named=True):
            country = row[config.country_col]
            slice_label = row[slice_col]
            idx = np.asarray([row_index[int(rid)] for rid in row["row_ids"] if int(rid) in row_index], dtype=int)
            indices_by_stratum[(country, slice_label)] = idx

    oom_thresholds = _compute_oom_thresholds(
        X_dist,
        indices_by_stratum=indices_by_stratum,
        seed=config.seed,
        percentile=config.oom_percentile,
    )

    nn_full = NearestNeighbors(n_neighbors=1, metric="euclidean")
    nn_full.fit(X_dist)

    anchor_feature_idx = {name: i for i, name in enumerate(features)}
    driver_idx = {name: i for i, name in enumerate(driver_cols)}
    conditional_tail_cache: dict[tuple[tuple[object | None, object | None], str], _ConditionalTailModel | None] = {}

    def _get_conditional_tail_model(
        stratum_id: tuple[object | None, object | None],
        lever: str,
    ) -> _ConditionalTailModel | None:
        key = (stratum_id, lever)
        if key in conditional_tail_cache:
            return conditional_tail_cache[key]
        feat_idx = anchor_feature_idx.get(lever)
        if feat_idx is None:
            conditional_tail_cache[key] = None
            return None
        stratum_indices = indices_by_stratum.get(stratum_id, np.array([], dtype=int))
        model = _fit_conditional_tail_model(
            X_core,
            stratum_indices=stratum_indices,
            feat_idx=feat_idx,
            knn_k=config.conditional_tail_knn_k,
        )
        conditional_tail_cache[key] = model
        return model

    oom_rows: list[dict[str, object]] = []

    beta_raw = np.asarray(linear_model.coef_, dtype=float).ravel()
    beta_by_lever = {name: float(beta_raw[idx]) for name, idx in driver_idx.items()}

    shock_table_rows: list[dict[str, object]] = []
    pe_ref_rows: list[dict[str, object]] = []

    countries_all = df_stats.get_column(config.country_col).unique().to_list()
    for country in countries_all:
        resolved = _resolve_stratum_id(
            country=country,
            slice_label=None,
            min_n=config.min_n,
            counts=stratum_counts,
        )
        level = _stratum_level(resolved)
        key = _stratum_key(resolved)
        stats = shock_stats.get(resolved, {})
        for lever in levers:
            lever_stats = stats.get(lever)
            if lever_stats is None:
                continue
            beta = beta_by_lever.get(lever)
            if beta is None:
                continue
            shock_table_rows.append(
                {
                    "country": country,
                    "slice": None,
                    "lever": lever,
                    "q25": lever_stats.q25,
                    "q50": lever_stats.q50,
                    "q75": lever_stats.q75,
                    "sigma_x": lever_stats.sigma_x,
                    "x_min": lever_stats.x_min,
                    "x_max": lever_stats.x_max,
                    "delta_x": lever_stats.delta_x,
                    "delta_z": lever_stats.delta_z,
                    "stratum_level": level,
                    "stratum_key": key,
                    "n": int(stratum_counts.get(resolved, 0)),
                }
            )
            pe_ref = float(beta * lever_stats.delta_x) if np.isfinite(lever_stats.delta_x) else float("nan")
            pe_sens_x = float(beta)
            pe_sens_z = (
                float(beta * lever_stats.delta_x / lever_stats.delta_z)
                if lever_stats.delta_z and np.isfinite(lever_stats.delta_z)
                else float("nan")
            )
            pe_ref_rows.append(
                {
                    "country": country,
                    "slice": None,
                    "lever": lever,
                    "pe_ref": pe_ref,
                    "pe_sens_x": pe_sens_x,
                    "pe_sens_z": pe_sens_z,
                    "delta_x": lever_stats.delta_x,
                    "delta_z": lever_stats.delta_z,
                    "stratum_level": level,
                    "stratum_key": key,
                    "n": int(stratum_counts.get(resolved, 0)),
                }
            )

    if slice_col:
        sliced = df_stats.filter(pl.col(slice_col).is_not_null())
        grouped = sliced.group_by([config.country_col, slice_col]).agg(pl.len().alias("n"))
        for row in grouped.iter_rows(named=True):
            country = row[config.country_col]
            slice_label = row[slice_col]
            resolved = _resolve_stratum_id(
                country=country,
                slice_label=slice_label,
                min_n=config.min_n,
                counts=stratum_counts,
            )
            level = _stratum_level(resolved)
            key = _stratum_key(resolved)
            stats = shock_stats.get(resolved, {})
            for lever in levers:
                lever_stats = stats.get(lever)
                if lever_stats is None:
                    continue
                beta = beta_by_lever.get(lever)
                if beta is None:
                    continue
                shock_table_rows.append(
                    {
                        "country": country,
                        "slice": slice_label,
                        "lever": lever,
                        "q25": lever_stats.q25,
                        "q50": lever_stats.q50,
                        "q75": lever_stats.q75,
                        "sigma_x": lever_stats.sigma_x,
                        "x_min": lever_stats.x_min,
                        "x_max": lever_stats.x_max,
                        "delta_x": lever_stats.delta_x,
                        "delta_z": lever_stats.delta_z,
                        "stratum_level": level,
                        "stratum_key": key,
                        "n": int(stratum_counts.get(resolved, 0)),
                    }
                )
                pe_ref = float(beta * lever_stats.delta_x) if np.isfinite(lever_stats.delta_x) else float("nan")
                pe_sens_x = float(beta)
                pe_sens_z = (
                    float(beta * lever_stats.delta_x / lever_stats.delta_z)
                    if lever_stats.delta_z and np.isfinite(lever_stats.delta_z)
                    else float("nan")
                )
                pe_ref_rows.append(
                    {
                        "country": country,
                        "slice": slice_label,
                        "lever": lever,
                        "pe_ref": pe_ref,
                        "pe_sens_x": pe_sens_x,
                        "pe_sens_z": pe_sens_z,
                        "delta_x": lever_stats.delta_x,
                        "delta_z": lever_stats.delta_z,
                        "stratum_level": level,
                        "stratum_key": key,
                        "n": int(stratum_counts.get(resolved, 0)),
                    }
                )

    for anchor_id in anchor_ids:
        for country in countries:
            neigh = neighborhoods.filter((pl.col("anchor_id") == anchor_id) & (pl.col("country") == country))
            if neigh.height == 0:
                continue

            if slice_col and slice_col in neigh.columns:
                grouped = neigh.group_by(slice_col).agg(pl.col("row_id").alias("row_ids"))
                groups = grouped.iter_rows(named=True)
                use_slice = True
            else:
                groups = [{"slice_label": None, "row_ids": neigh.get_column("row_id").to_list()}]
                use_slice = False

            total_rows = neigh.height
            for row in groups:
                slice_label = row[slice_col] if use_slice else row["slice_label"]
                row_list = row["row_ids"]
                row_indices = np.asarray([row_index[int(rid)] for rid in row_list if int(rid) in row_index], dtype=int)
                if row_indices.size == 0:
                    continue

                sample_weight = float(row_indices.size / total_rows) if total_rows else 0.0
                stratum_id = _resolve_stratum_id(
                    country=country,
                    slice_label=slice_label,
                    min_n=config.min_n,
                    counts=stratum_counts,
                )
                level = _stratum_level(stratum_id)
                key = _stratum_key(stratum_id)

                for lever in levers:
                    lever_idx = driver_idx.get(lever)
                    if lever_idx is None:
                        continue
                    lever_stats = shock_stats.get(stratum_id, {}).get(lever)
                    if lever_stats is None:
                        continue

                    q25 = lever_stats.q25
                    q75 = lever_stats.q75

                    if not np.isfinite(q25) or not np.isfinite(q75):
                        continue

                    if lever in anchor_feature_idx:
                        feat_idx = anchor_feature_idx[lever]
                        stratum_indices = indices_by_stratum.get(stratum_id, np.array([], dtype=int))
                        if stratum_indices.size == 0:
                            continue

                        X_stratum = X_core[stratum_indices]
                        if X_stratum.size == 0:
                            continue

                        q25_norm = float(np.nanquantile(X_stratum[:, feat_idx], 0.25))
                        q75_norm = float(np.nanquantile(X_stratum[:, feat_idx], 0.75))

                        X_low_core = X_core[row_indices].copy()
                        X_high_core = X_core[row_indices].copy()
                        X_low_core[:, feat_idx] = q25_norm
                        X_high_core[:, feat_idx] = q75_norm

                        if config.off_manifold_method == "distance_nn":
                            if pca_components is not None and pca_mean is not None:
                                X_low_dist = (X_low_core - pca_mean) @ pca_components.T
                                X_high_dist = (X_high_core - pca_mean) @ pca_components.T
                            else:
                                X_low_dist = X_low_core
                                X_high_dist = X_high_core

                            thr = float(oom_thresholds.get(stratum_id, float("inf")))
                            d_low, _ = nn_full.kneighbors(X_low_dist, return_distance=True)
                            d_high, _ = nn_full.kneighbors(X_high_dist, return_distance=True)
                            d = np.concatenate([d_low[:, 0], d_high[:, 0]])
                            oom_rate = float(np.mean(d > thr)) if d.size else float("nan")
                            tail_mean = float("nan")
                        else:
                            model = _get_conditional_tail_model(stratum_id, lever)
                            if model is None:
                                continue
                            _, tail_low, off_low = _conditional_tail_stats(
                                model,
                                X_core=X_core,
                                query_indices=row_indices,
                                x_star=q25_norm,
                                tail_alpha=config.conditional_tail_alpha,
                            )
                            _, tail_high, off_high = _conditional_tail_stats(
                                model,
                                X_core=X_core,
                                query_indices=row_indices,
                                x_star=q75_norm,
                                tail_alpha=config.conditional_tail_alpha,
                            )
                            off = np.concatenate([off_low.astype(float), off_high.astype(float)])
                            tail = np.concatenate([tail_low, tail_high])
                            oom_rate = float(np.mean(off)) if off.size else float("nan")
                            tail_mean = float(np.mean(tail)) if tail.size else float("nan")
                            thr = float(config.conditional_tail_alpha)

                        oom_rows.append(
                            {
                                "anchor_id": int(anchor_id),
                                "country": country,
                                "slice": slice_label,
                                "lever": lever,
                                "off_manifold_rate": oom_rate,
                                "threshold": thr,
                                "tail_probability_mean": tail_mean,
                                "off_manifold_method": config.off_manifold_method,
                                "stratum_level": level,
                                "stratum_key": key,
                                "sample_weight": sample_weight,
                            }
                        )

    if oom_rows:
        pl.DataFrame(oom_rows).write_csv(outdir / "pe_off_manifold.csv")
    if shock_table_rows:
        shock_df = pl.DataFrame(shock_table_rows)
        shock_df.write_csv(outdir / "shock_table.csv")
    else:
        shock_df = pl.DataFrame(
            {
                c: []
                for c in [
                    "country",
                    "slice",
                    "lever",
                    "q25",
                    "q50",
                    "q75",
                    "sigma_x",
                    "x_min",
                    "x_max",
                    "delta_x",
                    "delta_z",
                    "stratum_level",
                    "stratum_key",
                    "n",
                ]
            }
        )
    if pe_ref_rows:
        pl.DataFrame(pe_ref_rows).write_csv(outdir / "pe_ref.csv")

    # Microvalidation handoff artifacts (default):
    # - design.parquet: base profiles × (lever, low/high)
    # - ice_ref.csv: reference ICE levels computed on the same design
    anchors_root = anchors_dir.parent
    eval_profiles_path = config.eval_profiles_path or (anchors_root / "eval" / "eval_profiles.parquet")
    if not eval_profiles_path.exists():
        raise FileNotFoundError(
            f"Missing eval profiles at {eval_profiles_path}. "
            "Run anchor build first (it exports eval profiles by default)."
        )

    profiles = _read_table(eval_profiles_path)
    # Prevent label leakage into LLM prompts: the design table should not include the
    # outcome/target column that the reference model is predicting.
    if config.target_col in profiles.columns:
        profiles = profiles.drop(config.target_col)
    if slice_col:
        if slice_col not in profiles.columns and config.propensity_slices_path:
            slices_df = _load_propensity_slices(
                config.propensity_slices_path,
                outcome=config.propensity_outcome,
                slice_col=config.propensity_slice_col,
            )
        if slice_col in df_stats.columns:
            # Join slice labels onto the base profiles (does not change which profiles are selected).
            profiles = profiles.join(df_stats.select(["row_id", slice_col]).unique(), on="row_id", how="left")
    else:
        profiles = profiles.with_columns(pl.lit(None).alias("slice"))
        slice_col = "slice"

    profiles = apply_missingness_plan_pl(profiles, missingness_plan, country_col=config.country_col)
    profiles = _ensure_eval_ids(profiles, country_col=config.country_col)

    if config.compute_eval_off_manifold:
        if "row_id" not in profiles.columns:
            logger.warning("Eval profiles missing row_id; skipping eval off-manifold diagnostics.")
        else:
            eval_oom_rows: list[dict[str, object]] = []
            base_group_cols = [config.country_col]
            if slice_col and slice_col in profiles.columns:
                base_group_cols.append(slice_col)
            grouped = (
                profiles.select(["eval_id", "row_id", *base_group_cols])
                .group_by(base_group_cols)
                .agg(
                    pl.col("eval_id").alias("eval_ids"),
                    pl.col("row_id").alias("row_ids"),
                )
            )
            for row in grouped.iter_rows(named=True):
                country = row[config.country_col]
                group_slice = row.get(slice_col) if slice_col else None
                stratum_id = _resolve_stratum_id(
                    country=country,
                    slice_label=group_slice,
                    min_n=config.min_n,
                    counts=stratum_counts,
                )
                level = _stratum_level(stratum_id)
                key = _stratum_key(stratum_id)
                stratum_indices = indices_by_stratum.get(stratum_id, np.array([], dtype=int))
                if stratum_indices.size == 0:
                    continue

                eval_ids_raw = [str(v) for v in (row.get("eval_ids") or [])]
                row_ids_raw = [int(v) for v in (row.get("row_ids") or [])]
                aligned_eval_ids: list[str] = []
                aligned_idx: list[int] = []
                for eval_id, rid in zip(eval_ids_raw, row_ids_raw, strict=False):
                    idx = row_index.get(int(rid))
                    if idx is None:
                        continue
                    aligned_eval_ids.append(eval_id)
                    aligned_idx.append(int(idx))
                if not aligned_idx:
                    continue

                row_indices = np.asarray(aligned_idx, dtype=int)
                for lever in levers:
                    if lever not in anchor_feature_idx:
                        continue
                    feat_idx = anchor_feature_idx[lever]
                    X_stratum = X_core[stratum_indices]
                    if X_stratum.size == 0:
                        continue
                    q25_norm = float(np.nanquantile(X_stratum[:, feat_idx], 0.25))
                    q75_norm = float(np.nanquantile(X_stratum[:, feat_idx], 0.75))

                    X_low_core = X_core[row_indices].copy()
                    X_high_core = X_core[row_indices].copy()
                    X_low_core[:, feat_idx] = q25_norm
                    X_high_core[:, feat_idx] = q75_norm

                    if config.off_manifold_method == "distance_nn":
                        if pca_components is not None and pca_mean is not None:
                            X_low_dist = (X_low_core - pca_mean) @ pca_components.T
                            X_high_dist = (X_high_core - pca_mean) @ pca_components.T
                        else:
                            X_low_dist = X_low_core
                            X_high_dist = X_high_core

                        thr = float(oom_thresholds.get(stratum_id, float("inf")))
                        d_low, _ = nn_full.kneighbors(X_low_dist, return_distance=True)
                        d_high, _ = nn_full.kneighbors(X_high_dist, return_distance=True)
                        for eval_id, dist in zip(aligned_eval_ids, d_low[:, 0], strict=False):
                            dist_f = float(dist)
                            eval_oom_rows.append(
                                {
                                    "eval_id": eval_id,
                                    "country": country,
                                    "slice": group_slice,
                                    "lever": lever,
                                    "grid_point": "low",
                                    "distance_to_manifold": dist_f,
                                    "percentile_conditional": float("nan"),
                                    "tail_probability": float("nan"),
                                    "threshold": thr,
                                    "is_off_manifold": bool(dist_f > thr),
                                    "off_manifold_method": config.off_manifold_method,
                                    "stratum_level": level,
                                    "stratum_key": key,
                                }
                            )
                        for eval_id, dist in zip(aligned_eval_ids, d_high[:, 0], strict=False):
                            dist_f = float(dist)
                            eval_oom_rows.append(
                                {
                                    "eval_id": eval_id,
                                    "country": country,
                                    "slice": group_slice,
                                    "lever": lever,
                                    "grid_point": "high",
                                    "distance_to_manifold": dist_f,
                                    "percentile_conditional": float("nan"),
                                    "tail_probability": float("nan"),
                                    "threshold": thr,
                                    "is_off_manifold": bool(dist_f > thr),
                                    "off_manifold_method": config.off_manifold_method,
                                    "stratum_level": level,
                                    "stratum_key": key,
                                }
                            )
                    else:
                        model = _get_conditional_tail_model(stratum_id, lever)
                        if model is None:
                            continue
                        p_low, tail_low, off_low = _conditional_tail_stats(
                            model,
                            X_core=X_core,
                            query_indices=row_indices,
                            x_star=q25_norm,
                            tail_alpha=config.conditional_tail_alpha,
                        )
                        p_high, tail_high, off_high = _conditional_tail_stats(
                            model,
                            X_core=X_core,
                            query_indices=row_indices,
                            x_star=q75_norm,
                            tail_alpha=config.conditional_tail_alpha,
                        )
                        thr = float(config.conditional_tail_alpha)
                        for eval_id, p, tail, off in zip(aligned_eval_ids, p_low, tail_low, off_low, strict=False):
                            eval_oom_rows.append(
                                {
                                    "eval_id": eval_id,
                                    "country": country,
                                    "slice": group_slice,
                                    "lever": lever,
                                    "grid_point": "low",
                                    "distance_to_manifold": float("nan"),
                                    "percentile_conditional": float(p),
                                    "tail_probability": float(tail),
                                    "threshold": thr,
                                    "is_off_manifold": bool(off),
                                    "off_manifold_method": config.off_manifold_method,
                                    "stratum_level": level,
                                    "stratum_key": key,
                                }
                            )
                        for eval_id, p, tail, off in zip(aligned_eval_ids, p_high, tail_high, off_high, strict=False):
                            eval_oom_rows.append(
                                {
                                    "eval_id": eval_id,
                                    "country": country,
                                    "slice": group_slice,
                                    "lever": lever,
                                    "grid_point": "high",
                                    "distance_to_manifold": float("nan"),
                                    "percentile_conditional": float(p),
                                    "tail_probability": float(tail),
                                    "threshold": thr,
                                    "is_off_manifold": bool(off),
                                    "off_manifold_method": config.off_manifold_method,
                                    "stratum_level": level,
                                    "stratum_key": key,
                                }
                            )
            if eval_oom_rows:
                eval_oom_df = pl.DataFrame(eval_oom_rows)
                eval_oom_df.write_csv(outdir / "pe_off_manifold_eval.csv")
                (
                    eval_oom_df.group_by(
                        ["country", "slice", "lever", "off_manifold_method", "stratum_level", "stratum_key"],
                    )
                    .agg(
                        pl.len().alias("n_total"),
                        pl.col("is_off_manifold").sum().alias("n_off_manifold"),
                        pl.col("is_off_manifold").mean().alias("off_manifold_rate"),
                        pl.col("tail_probability").mean().alias("tail_probability_mean"),
                        pl.col("percentile_conditional").mean().alias("percentile_conditional_mean"),
                        pl.col("threshold").first().alias("threshold"),
                    )
                    .sort(["country", "slice", "lever"])
                    .write_csv(outdir / "pe_off_manifold_eval_summary.csv")
                )

    stats_cols = [
        "q25",
        "q50",
        "q75",
        "sigma_x",
        "x_min",
        "x_max",
        "delta_x",
        "delta_z",
        "stratum_level",
        "stratum_key",
        "n",
    ]

    design_parts: list[pl.DataFrame] = []
    shock_design_parts: list[pl.DataFrame] = []
    for lever in levers:
        if lever not in profiles.columns:
            raise ValueError(f"Eval profiles missing lever column: {lever}")

        stats = shock_df.filter(pl.col("lever") == lever).select(["country", "slice", *stats_cols])
        slice_stats = stats.filter(pl.col("slice").is_not_null()).rename({c: f"{c}_slice" for c in stats_cols})
        country_stats = (
            stats.filter(pl.col("slice").is_null()).drop("slice").rename({c: f"{c}_country" for c in stats_cols})
        )

        base = profiles
        if slice_col:
            base = base.join(
                slice_stats,
                left_on=["country", slice_col],
                right_on=["country", "slice"],
                how="left",
            )
        else:
            for c in stats_cols:
                base = base.with_columns(pl.lit(None).alias(f"{c}_slice"))

        base = base.join(country_stats, on="country", how="left")
        base = base.with_columns(
            pl.coalesce([pl.col("q25_slice"), pl.col("q25_country")]).alias("q25"),
            pl.coalesce([pl.col("q50_slice"), pl.col("q50_country")]).alias("q50"),
            pl.coalesce([pl.col("q75_slice"), pl.col("q75_country")]).alias("q75"),
            pl.coalesce([pl.col("sigma_x_slice"), pl.col("sigma_x_country")]).alias("sigma_x"),
            pl.coalesce([pl.col("x_min_slice"), pl.col("x_min_country")]).alias("x_min"),
            pl.coalesce([pl.col("x_max_slice"), pl.col("x_max_country")]).alias("x_max"),
            pl.coalesce([pl.col("delta_x_slice"), pl.col("delta_x_country")]).alias("delta_x"),
            pl.coalesce([pl.col("delta_z_slice"), pl.col("delta_z_country")]).alias("delta_z"),
            pl.coalesce([pl.col("stratum_level_slice"), pl.col("stratum_level_country")]).alias("stratum_level"),
            pl.coalesce([pl.col("stratum_key_slice"), pl.col("stratum_key_country")]).alias("stratum_key"),
            pl.coalesce([pl.col("n_slice"), pl.col("n_country")]).alias("n"),
        ).drop([f"{c}_slice" for c in stats_cols] + [f"{c}_country" for c in stats_cols])

        base = base.with_columns(
            pl.col(lever).cast(pl.Float64).alias("x_base"),
            pl.when(
                pl.col("x_min").is_finite()
                & pl.col("x_max").is_finite()
                & pl.col("sigma_x").is_finite()
                & (pl.col("sigma_x") > 0)
                & pl.col(lever).cast(pl.Float64).is_finite()
            )
            .then(
                pl.min_horizontal(
                    pl.max_horizontal(pl.col(lever).cast(pl.Float64) + pl.col("sigma_x"), pl.col("x_min")),
                    pl.col("x_max"),
                )
            )
            .otherwise(None)
            .alias("x_plus_1sd"),
            pl.when(
                pl.col("x_min").is_finite()
                & pl.col("x_max").is_finite()
                & pl.col("sigma_x").is_finite()
                & (pl.col("sigma_x") > 0)
                & pl.col(lever).cast(pl.Float64).is_finite()
            )
            .then(
                pl.min_horizontal(
                    pl.max_horizontal(pl.col(lever).cast(pl.Float64) - pl.col("sigma_x"), pl.col("x_min")),
                    pl.col("x_max"),
                )
            )
            .otherwise(None)
            .alias("x_minus_1sd"),
        ).with_columns(
            pl.when(
                pl.col("sigma_x").is_finite()
                & (pl.col("sigma_x") > 0)
                & pl.col("x_plus_1sd").is_finite()
                & pl.col("x_base").is_finite()
            )
            .then((pl.col("x_plus_1sd") - pl.col("x_base")) / pl.col("sigma_x"))
            .otherwise(None)
            .alias("realized_k_plus"),
            pl.when(
                pl.col("sigma_x").is_finite()
                & (pl.col("sigma_x") > 0)
                & pl.col("x_minus_1sd").is_finite()
                & pl.col("x_base").is_finite()
            )
            .then((pl.col("x_base") - pl.col("x_minus_1sd")) / pl.col("sigma_x"))
            .otherwise(None)
            .alias("realized_k_minus"),
        )

        low = base.with_columns(
            pl.col("eval_id").alias("id"),
            pl.lit(config.target_col).alias("outcome"),
            pl.lit(lever).alias("target"),
            pl.lit("low").alias("grid_point"),
            pl.col("q25").cast(pl.Float64).alias("grid_value"),
            pl.col("q25").cast(pl.Float64).alias(lever),
        )
        mid = base.with_columns(
            pl.col("eval_id").alias("id"),
            pl.lit(config.target_col).alias("outcome"),
            pl.lit(lever).alias("target"),
            pl.lit("mid").alias("grid_point"),
            pl.col("q50").cast(pl.Float64).alias("grid_value"),
            pl.col("q50").cast(pl.Float64).alias(lever),
        )
        high = base.with_columns(
            pl.col("eval_id").alias("id"),
            pl.lit(config.target_col).alias("outcome"),
            pl.lit(lever).alias("target"),
            pl.lit("high").alias("grid_point"),
            pl.col("q75").cast(pl.Float64).alias("grid_value"),
            pl.col("q75").cast(pl.Float64).alias(lever),
        )

        base_shock = base.with_columns(
            pl.col("eval_id").alias("id"),
            pl.lit(config.target_col).alias("outcome"),
            pl.lit(lever).alias("target"),
            pl.lit("base_shock").alias("grid_point"),
            pl.col("x_base").cast(pl.Float64).alias("grid_value"),
            pl.col("x_base").cast(pl.Float64).alias(lever),
        )
        plus_shock = base.with_columns(
            pl.col("eval_id").alias("id"),
            pl.lit(config.target_col).alias("outcome"),
            pl.lit(lever).alias("target"),
            pl.lit("shock_plus_1sd").alias("grid_point"),
            pl.col("x_plus_1sd").cast(pl.Float64).alias("grid_value"),
            pl.col("x_plus_1sd").cast(pl.Float64).alias(lever),
        )
        minus_shock = base.with_columns(
            pl.col("eval_id").alias("id"),
            pl.lit(config.target_col).alias("outcome"),
            pl.lit(lever).alias("target"),
            pl.lit("shock_minus_1sd").alias("grid_point"),
            pl.col("x_minus_1sd").cast(pl.Float64).alias("grid_value"),
            pl.col("x_minus_1sd").cast(pl.Float64).alias(lever),
        )
        design_parts.extend([low, mid, high])
        shock_design_parts.extend([base_shock, plus_shock, minus_shock])

    if design_parts:
        design = pl.concat(design_parts, how="vertical_relaxed")
        design.write_parquet(outdir / "design.parquet")

        design_for_pred = _with_country_dummies(design, country_col=config.country_col, specs=country_specs)
        X = design_for_pred.select(model_cols).to_numpy().astype(float)
        X = np.where(np.isnan(X), means, X)

        shock_design = pl.concat(shock_design_parts, how="vertical_relaxed") if shock_design_parts else pl.DataFrame()
        if not shock_design.is_empty():
            shock_for_pred = _with_country_dummies(shock_design, country_col=config.country_col, specs=country_specs)
            X_shock = shock_for_pred.select(model_cols).to_numpy().astype(float)
            X_shock = np.where(np.isnan(X_shock), means, X_shock)
        else:
            X_shock = np.empty((0, len(model_cols)), dtype=float)

        keys = ["id", "target", "outcome"]
        for name, ref_model in models.items():
            y_hat = np.asarray(ref_model.predict(X), dtype=float)
            design_pred = design.with_columns(pl.Series("prediction", y_hat))

            if X_shock.shape[0] > 0:
                y_hat_shock = np.asarray(ref_model.predict(X_shock), dtype=float)
                shock_pred = shock_design.with_columns(pl.Series("prediction", y_hat_shock))
            else:
                shock_pred = pl.DataFrame()

            low_df = (
                design_pred.filter(pl.col("grid_point") == "low")
                .group_by(keys)
                .agg(
                    pl.mean("prediction").alias("ice_low"),
                    pl.first("delta_z").alias("delta_z"),
                )
            )
            mid_df = (
                design_pred.filter(pl.col("grid_point") == "mid")
                .group_by(keys)
                .agg(pl.mean("prediction").alias("ice_mid"))
            )
            high_df = (
                design_pred.filter(pl.col("grid_point") == "high")
                .group_by(keys)
                .agg(pl.mean("prediction").alias("ice_high"))
            )
            ice_ref = low_df.join(mid_df, on=keys, how="inner").join(high_df, on=keys, how="inner")

            if not shock_pred.is_empty():
                shock_keys = [
                    "id",
                    "target",
                    "outcome",
                    "x_base",
                    "x_plus_1sd",
                    "x_minus_1sd",
                    "sigma_x",
                    "realized_k_plus",
                    "realized_k_minus",
                ]
                shock_base_df = (
                    shock_pred.filter(pl.col("grid_point") == "base_shock")
                    .group_by(shock_keys[:3])
                    .agg(
                        pl.mean("prediction").alias("ice_base"),
                        pl.first("x_base").alias("x_base"),
                        pl.first("x_plus_1sd").alias("x_plus_1sd"),
                        pl.first("x_minus_1sd").alias("x_minus_1sd"),
                        pl.first("sigma_x").alias("sigma_x"),
                        pl.first("realized_k_plus").alias("realized_k_plus"),
                        pl.first("realized_k_minus").alias("realized_k_minus"),
                    )
                )
                shock_plus_df = (
                    shock_pred.filter(pl.col("grid_point") == "shock_plus_1sd")
                    .group_by(keys)
                    .agg(pl.mean("prediction").alias("ice_plus_1sd"))
                )
                shock_minus_df = (
                    shock_pred.filter(pl.col("grid_point") == "shock_minus_1sd")
                    .group_by(keys)
                    .agg(pl.mean("prediction").alias("ice_minus_1sd"))
                )
                shock_ref = (
                    shock_base_df.join(shock_plus_df, on=keys, how="inner")
                    .join(shock_minus_df, on=keys, how="inner")
                    .with_columns(
                        (pl.col("ice_plus_1sd") - pl.col("ice_base")).alias("se_1sd"),
                        (pl.col("ice_plus_1sd") - pl.col("ice_minus_1sd")).alias("se_pm1sd"),
                        pl.col("realized_k_plus").alias("realized_shock_sd_plus"),
                        pl.col("realized_k_minus").alias("realized_shock_sd_minus"),
                    )
                    .with_columns(
                        pl.col("se_1sd").alias("shock_effect_1sd"),
                        pl.col("se_pm1sd").alias("shock_effect_pm1sd"),
                    )
                    .select(
                        [
                            *keys,
                            "ice_base",
                            "ice_plus_1sd",
                            "ice_minus_1sd",
                            "se_1sd",
                            "se_pm1sd",
                            "shock_effect_1sd",
                            "shock_effect_pm1sd",
                            "x_base",
                            "x_plus_1sd",
                            "x_minus_1sd",
                            "sigma_x",
                            "realized_k_plus",
                            "realized_k_minus",
                            "realized_shock_sd_plus",
                            "realized_shock_sd_minus",
                        ]
                    )
                )
                ice_ref = ice_ref.join(shock_ref, on=keys, how="left")

            ice_ref = ice_ref.sort(keys)
            if name == "linear":
                ice_path = outdir / "ice_ref.csv"
            else:
                ice_path = outdir / f"ice_ref__{name}.csv"
            ice_ref.write_csv(ice_path)

            if name != "linear":
                agg_keys = ["country", "target", "outcome"]
                slice_key = slice_col if slice_col else "slice"
                if slice_key in design_pred.columns:
                    agg_keys.insert(1, slice_key)

                low_agg = (
                    design_pred.filter(pl.col("grid_point") == "low")
                    .group_by(agg_keys)
                    .agg(pl.mean("prediction").alias("ice_low"))
                )
                high_agg = (
                    design_pred.filter(pl.col("grid_point") == "high")
                    .group_by(agg_keys)
                    .agg(pl.mean("prediction").alias("ice_high"))
                )
                pe_df = low_agg.join(high_agg, on=agg_keys, how="inner").with_columns(
                    (pl.col("ice_high") - pl.col("ice_low")).alias("pe_ref"),
                    pl.lit(float("nan")).alias("pe_sens_x"),
                    pl.lit(float("nan")).alias("pe_sens_z"),
                )

                if slice_key in pe_df.columns and slice_key != "slice":
                    pe_df = pe_df.rename({slice_key: "slice"})

                pe_df = pe_df.rename({"target": "lever"})
                if "slice" not in pe_df.columns:
                    pe_df = pe_df.with_columns(pl.lit(None).alias("slice"))

                pe_df = pe_df.join(
                    shock_df,
                    left_on=["country", "slice", "lever"],
                    right_on=["country", "slice", "lever"],
                    how="left",
                )

                pe_path = outdir / f"pe_ref__{name}.csv"
                pe_df.write_csv(pe_path)

            group_cols = [
                "country",
                "slice",
                "target",
                "outcome",
                "stratum_level",
                "stratum_key",
                "n",
                "delta_x",
                "delta_z",
            ]
            low_pe = (
                design_pred.filter(pl.col("grid_point") == "low")
                .group_by(group_cols)
                .agg(pl.mean("prediction").alias("ice_low"))
            )
            high_pe = (
                design_pred.filter(pl.col("grid_point") == "high")
                .group_by(group_cols)
                .agg(pl.mean("prediction").alias("ice_high"))
            )
            pe_ref = (
                low_pe.join(high_pe, on=group_cols, how="inner")
                .with_columns(
                    (pl.col("ice_high") - pl.col("ice_low")).alias("pe_ref"),
                    pl.lit(float("nan")).alias("pe_sens_x"),
                    pl.lit(float("nan")).alias("pe_sens_z"),
                    pl.col("target").alias("lever"),
                )
                .select(
                    [
                        "country",
                        "slice",
                        "lever",
                        "pe_ref",
                        "pe_sens_x",
                        "pe_sens_z",
                        "delta_x",
                        "delta_z",
                        "stratum_level",
                        "stratum_key",
                        "n",
                    ]
                )
            )

            if not shock_pred.is_empty():
                shock_group_cols = [
                    "country",
                    "slice",
                    "target",
                    "outcome",
                    "stratum_level",
                    "stratum_key",
                    "n",
                    "sigma_x",
                ]
                shock_base_pe = (
                    shock_pred.filter(pl.col("grid_point") == "base_shock")
                    .group_by(shock_group_cols)
                    .agg(pl.mean("prediction").alias("ice_base"))
                )
                shock_plus_pe = (
                    shock_pred.filter(pl.col("grid_point") == "shock_plus_1sd")
                    .group_by(shock_group_cols)
                    .agg(pl.mean("prediction").alias("ice_plus_1sd"))
                )
                shock_minus_pe = (
                    shock_pred.filter(pl.col("grid_point") == "shock_minus_1sd")
                    .group_by(shock_group_cols)
                    .agg(pl.mean("prediction").alias("ice_minus_1sd"))
                )
                shock_pe = (
                    shock_base_pe.join(shock_plus_pe, on=shock_group_cols, how="inner")
                    .join(shock_minus_pe, on=shock_group_cols, how="inner")
                    .with_columns(
                        (pl.col("ice_plus_1sd") - pl.col("ice_base")).alias("se_1sd"),
                        (pl.col("ice_plus_1sd") - pl.col("ice_minus_1sd")).alias("se_pm1sd"),
                    )
                    .with_columns(
                        pl.col("se_1sd").alias("shock_effect_1sd"),
                        pl.col("se_pm1sd").alias("shock_effect_pm1sd"),
                    )
                    .select(
                        [
                            "country",
                            "slice",
                            "target",
                            "outcome",
                            "stratum_level",
                            "stratum_key",
                            "n",
                            "sigma_x",
                            "ice_base",
                            "ice_plus_1sd",
                            "ice_minus_1sd",
                            "se_1sd",
                            "se_pm1sd",
                            "shock_effect_1sd",
                            "shock_effect_pm1sd",
                        ]
                    )
                )
                pe_ref = pe_ref.join(
                    shock_pe,
                    on=["country", "slice", "target", "outcome", "stratum_level", "stratum_key", "n"],
                    how="left",
                )
            if name == "linear":
                pe_path = outdir / "pe_ref__linear.csv"
            else:
                pe_path = outdir / f"pe_ref__{name}.csv"
            pe_ref.write_csv(pe_path)
