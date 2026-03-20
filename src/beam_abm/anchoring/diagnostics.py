"""Diagnostics for the anchor-based heterogeneity system."""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
from beartype import beartype
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from beam_abm.anchoring.build import (
    _find_tau_for_target_keff,
    _fit_kmedoids,
    _impute_within_country_knn,
    _prepare_matrix,
    build_anchor_neighborhoods,
)
from beam_abm.common.logging import get_logger
from beam_abm.decision_function.io import ensure_outdir
from beam_abm.settings import CSV_FILE_CLEAN, SEED

logger = get_logger(__name__)


@dataclass(slots=True, frozen=False)
class DiagnosticsConfig:
    input_csv: Path = Path(CSV_FILE_CLEAN)
    anchors_dir: Path = Path("empirical/output/anchors")
    outdir: Path | None = None
    country_col: str = "country"
    missingness_threshold: float = 0.35
    normalization: str = "demean_country_then_global_scale"
    pca_components: int = 0
    coverage_k_grid: tuple[int, ...] = (5, 10, 15, 20, 25)
    stability_k_grid: tuple[int, ...] | None = None
    bootstrap_runs: int = 20
    bootstrap_subsample_frac: float = 0.8
    globality_pi_threshold: float = 0.05
    seed: int = SEED


def _load_anchor_metadata(anchors_dir: Path) -> dict[str, object]:
    meta_path = anchors_dir / "anchor_metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing anchor metadata: {meta_path}")
    return json.loads(meta_path.read_text(encoding="utf-8"))


def _coverage_curve(
    X: np.ndarray,
    *,
    countries: Sequence[object],
    k_grid: Sequence[int],
    tau_method: str | None,
    tau_keff_target: float | None,
    tau_fixed: float | None,
    seed: int,
) -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    countries_arr = np.asarray(countries, dtype=object)
    method = (tau_method or "").strip().lower()
    tau_fixed_value: float | None
    try:
        tau_fixed_value = float(tau_fixed) if tau_fixed is not None and math.isfinite(float(tau_fixed)) else None
    except (TypeError, ValueError):
        tau_fixed_value = None

    target_keff: float | None
    try:
        target_keff = float(tau_keff_target) if tau_keff_target is not None else None
    except (TypeError, ValueError):
        target_keff = None
    for k in k_grid:
        _, medoids = _fit_kmedoids(X, k=int(k), seed=int(seed))
        distances = cdist(X, X[medoids], metric="euclidean")
        min_dist = np.min(distances, axis=1)

        if method == "fixed":
            tau_k = tau_fixed_value
        elif method == "target_keff" and target_keff is not None:
            tau_k = float(_find_tau_for_target_keff(distances, target_keff=target_keff, seed=int(seed)))
        else:
            tau_k = tau_fixed_value

        rows.append(
            {
                "k": int(k),
                "scope": "pooled",
                "country": None,
                "n": int(min_dist.size),
                "mean_dist": float(np.mean(min_dist)),
                "p90_dist": float(np.quantile(min_dist, 0.9)),
                "tau": tau_k,
                "cov_tau": float(np.mean(min_dist <= tau_k)) if tau_k is not None else None,
            }
        )
        for country in np.unique(countries_arr):
            mask = countries_arr == country
            if not mask.any():
                continue
            vals = min_dist[mask]
            rows.append(
                {
                    "k": int(k),
                    "scope": "country",
                    "country": country,
                    "n": int(vals.size),
                    "mean_dist": float(np.mean(vals)),
                    "p90_dist": float(np.quantile(vals, 0.9)),
                    "tau": tau_k,
                    "cov_tau": float(np.mean(vals <= tau_k)) if tau_k is not None else None,
                }
            )
    return pl.DataFrame(rows)


def _anchor_stability(
    X: np.ndarray,
    *,
    k: int,
    runs: int,
    subsample_frac: float,
    seed: int,
) -> tuple[pl.DataFrame, dict[str, object]]:
    _, ref_medoids = _fit_kmedoids(X, k=int(k), seed=int(seed))
    ref_coords = X[ref_medoids]

    rng = np.random.default_rng(seed)
    n = X.shape[0]
    take = max(2, int(math.floor(n * subsample_frac)))
    rows: list[dict[str, object]] = []

    for run in range(int(runs)):
        idx = rng.choice(n, size=take, replace=False)
        _, sub_medoids = _fit_kmedoids(X[idx], k=int(k), seed=int(seed + run + 1))
        sub_coords = X[idx][sub_medoids]
        dist = cdist(ref_coords, sub_coords, metric="euclidean")
        row_ind, col_ind = linear_sum_assignment(dist)
        matched = dist[row_ind, col_ind]
        rows.append(
            {
                "run_id": int(run),
                "mean_matched_dist": float(np.mean(matched)),
                "max_matched_dist": float(np.max(matched)),
                "min_matched_dist": float(np.min(matched)),
            }
        )

    df = pl.DataFrame(rows)
    summary = {
        "runs": int(runs),
        "mean_matched_dist": float(df["mean_matched_dist"].mean()),
        "median_matched_dist": float(df["mean_matched_dist"].median()),
        "max_matched_dist": float(df["max_matched_dist"].max()),
    }
    return df, summary


def _anchor_stability_curve(
    X: np.ndarray,
    *,
    k_grid: Sequence[int],
    runs: int,
    subsample_frac: float,
    seed: int,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Compute anchor stability diagnostics across multiple values of k.

    Stability is measured by fitting k-medoids on the full sample to produce a
    reference set of medoids, then refitting on bootstrap-style subsamples and
    matching subsample medoids to the reference medoids using a minimum-cost
    assignment (Hungarian algorithm). The primary metric is the mean matched
    Euclidean distance between reference and subsample medoids.

    Returns
    -------
    (runs_df, summary_df)
        runs_df contains per-(k, run_id) metrics.
        summary_df aggregates per-k quantiles and maxima of those metrics.
    """

    per_run_frames: list[pl.DataFrame] = []
    summary_rows: list[dict[str, object]] = []
    for k in k_grid:
        df_runs, summary = _anchor_stability(
            X,
            k=int(k),
            runs=int(runs),
            subsample_frac=float(subsample_frac),
            seed=int(seed),
        )
        if df_runs.height:
            per_run_frames.append(df_runs.with_columns(pl.lit(int(k)).alias("k")))
        summary_rows.append(
            {
                "k": int(k),
                **summary,
            }
        )

    runs_df = pl.concat(per_run_frames, how="vertical") if per_run_frames else pl.DataFrame([])
    summary_df = pl.DataFrame(summary_rows)
    return runs_df, summary_df


def _anchor_leakage(
    country_weights: pl.DataFrame,
    *,
    pi_threshold: float,
) -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    if country_weights.height == 0:
        return pl.DataFrame(rows)

    anchors = country_weights.get_column("anchor_id").unique().to_list()
    for anchor_id in anchors:
        sub = country_weights.filter(pl.col("anchor_id") == anchor_id)
        pis = sub.get_column("pi").to_numpy().astype(float)
        if pis.sum() <= 0:
            rows.append(
                {
                    "anchor_id": int(anchor_id),
                    "entropy": float("nan"),
                    "max_share": float("nan"),
                    "n_countries": int(sub.height),
                    "n_countries_above_threshold": 0,
                }
            )
            continue
        p = pis / float(np.sum(pis))
        entropy = float(-np.sum(np.clip(p, 1e-12, 1.0) * np.log(np.clip(p, 1e-12, 1.0))))
        max_share = float(np.max(p))
        above = int(np.sum(pis >= float(pi_threshold)))
        rows.append(
            {
                "anchor_id": int(anchor_id),
                "entropy": entropy,
                "max_share": max_share,
                "n_countries": int(sub.height),
                "n_countries_above_threshold": above,
            }
        )
    return pl.DataFrame(rows)


def _membership_entropy_summary(memberships: pl.DataFrame) -> pl.DataFrame:
    if memberships.height == 0:
        return pl.DataFrame({"metric": [], "value": []})

    per_row = memberships.group_by("row_id").agg(
        pl.first("entropy").alias("entropy"),
        pl.first("k_eff").alias("k_eff"),
    )
    rows: list[dict[str, object]] = []
    for metric in ("entropy", "k_eff"):
        values = per_row.get_column(metric).to_numpy().astype(float)
        for q in (0.1, 0.25, 0.5, 0.75, 0.9):
            rows.append(
                {
                    "metric": metric,
                    "quantile": float(q),
                    "value": float(np.quantile(values, q)),
                }
            )
    return pl.DataFrame(rows)


def _sample_synthetic(
    df: pl.DataFrame,
    *,
    neighborhoods: pl.DataFrame,
    country_weights: pl.DataFrame,
    country_col: str,
    seed: int,
) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    if df.height == 0:
        return df

    by_country = df.group_by(country_col, maintain_order=True)
    countries = by_country.agg(pl.len().alias("n"))
    rows: list[dict[str, object]] = []

    row_lookup = df.select(["row_id"]).to_series(0).to_list()
    row_lookup_map = {int(rid): idx for idx, rid in enumerate(row_lookup)}
    df_rows = df.to_dicts()

    for record in countries.iter_rows(named=True):
        country = record[country_col]
        n_c = int(record["n"])
        weights_c = country_weights.filter(pl.col("country") == country)
        if weights_c.height == 0:
            # fallback to pooled if no country weights
            choices = rng.choice(df.height, size=n_c, replace=True)
            rows.extend(df_rows[int(i)] for i in choices)
            continue

        anchors = weights_c.get_column("anchor_id").to_numpy().astype(int)
        pis = weights_c.get_column("pi").to_numpy().astype(float)
        pis = pis / pis.sum() if pis.sum() > 0 else np.ones_like(pis) / len(pis)

        for _ in range(n_c):
            anchor_id = int(rng.choice(anchors, p=pis))
            neigh = neighborhoods.filter((pl.col("anchor_id") == anchor_id) & (pl.col("country") == country))
            if neigh.height == 0:
                # fallback to pooled if no neighborhood rows
                idx = int(rng.integers(0, df.height))
                rows.append(df_rows[idx])
                continue
            dist = neigh.get_column("dist").to_numpy().astype(float)
            if dist.size == 0:
                idx = int(rng.integers(0, df.height))
                rows.append(df_rows[idx])
                continue
            scale = float(np.median(dist)) if np.isfinite(dist).any() else 1.0
            scale = max(scale, 1e-6)
            weights = np.exp(-dist / scale)
            weights = weights / weights.sum() if weights.sum() > 0 else np.ones_like(weights) / len(weights)
            choice = int(rng.choice(neigh.get_column("row_id").to_numpy().astype(int), p=weights))
            idx = row_lookup_map.get(choice)
            if idx is None:
                idx = int(rng.integers(0, df.height))
            rows.append(df_rows[int(idx)])

    if not rows:
        return df
    return pl.DataFrame(rows, schema=df.schema)


def _mean_abs_corr_diff(real: np.ndarray, synth: np.ndarray) -> float:
    def _corr(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        means = np.nanmean(x, axis=0)
        x = np.where(np.isnan(x), means, x)
        return np.corrcoef(x, rowvar=False)

    corr_real = _corr(real)
    corr_synth = _corr(synth)
    if corr_real.shape != corr_synth.shape:
        return float("nan")
    diff = np.abs(corr_real - corr_synth)
    mask = ~np.eye(diff.shape[0], dtype=bool)
    vals = diff[mask]
    return float(np.mean(vals)) if vals.size else float("nan")


@beartype
def run_anchor_diagnostics(config: DiagnosticsConfig) -> None:
    """Compute anchor diagnostics and write outputs.

    Parameters
    ----------
    config:
        DiagnosticsConfig specifying inputs and thresholds.

    Returns
    -------
    None
        Writes diagnostics tables to disk.
    """

    anchors_dir = config.anchors_dir
    outdir = config.outdir or anchors_dir
    ensure_outdir(outdir)

    meta = _load_anchor_metadata(anchors_dir)
    features = tuple(meta.get("features", []))
    if not features:
        raise ValueError("anchor_metadata.json missing features list")

    df = pl.read_csv(config.input_csv, infer_schema_length=10_000).with_row_index(name="row_id")
    norm = str(meta.get("normalization", config.normalization))
    miss = float(meta.get("missingness_threshold", config.missingness_threshold))
    X_z, _, _missingness = _prepare_matrix(
        df,
        features,
        country_col=config.country_col,
        missingness_threshold=miss,
        normalization=norm,
    )
    X = _impute_within_country_knn(df, X_z=X_z, country_col=config.country_col, seed=config.seed)
    X_dist = X
    pca_path = meta.get("pca_path")
    if pca_path:
        pca_file = anchors_dir / str(pca_path)
        if pca_file.exists():
            pca_data = np.load(pca_file)
            components = pca_data["components"]
            mean = pca_data["mean"]
            X_dist = (X - mean) @ components.T
        else:
            logger.warning("PCA file not found at {path}; using full space.", path=pca_file)

    countries = df[config.country_col].to_list()

    tau_method = str(meta.get("tau_method", "") or "")
    tau_keff_target = meta.get("tau_keff_target")
    tau_fixed = meta.get("tau")
    coverage = _coverage_curve(
        X_dist,
        countries=countries,
        k_grid=config.coverage_k_grid,
        tau_method=tau_method,
        tau_keff_target=float(tau_keff_target) if tau_keff_target is not None else None,
        tau_fixed=float(tau_fixed) if tau_fixed is not None else None,
        seed=config.seed,
    )
    coverage.write_csv(outdir / "anchor_coverage.csv")

    stability_k_grid = config.stability_k_grid or config.coverage_k_grid
    stability_sweep_df, stability_sweep_summary = _anchor_stability_curve(
        X_dist,
        k_grid=stability_k_grid,
        runs=config.bootstrap_runs,
        subsample_frac=config.bootstrap_subsample_frac,
        seed=config.seed,
    )
    if stability_sweep_df.height:
        stability_sweep_df.write_csv(outdir / "anchor_stability_k_sweep.csv")
    stability_sweep_summary.write_csv(outdir / "anchor_stability_k_sweep_summary.csv")

    anchor_k = int(meta.get("anchor_k", 0) or 0)
    if anchor_k > 0:
        stability_df, stability_summary = _anchor_stability(
            X_dist,
            k=anchor_k,
            runs=config.bootstrap_runs,
            subsample_frac=config.bootstrap_subsample_frac,
            seed=config.seed,
        )
        stability_df.write_csv(outdir / "anchor_stability.csv")
        (outdir / "anchor_stability_summary.json").write_text(json.dumps(stability_summary, indent=2))

    country_weights = pl.read_csv(anchors_dir / "anchor_country_weights.csv")
    leakage = _anchor_leakage(country_weights, pi_threshold=config.globality_pi_threshold)
    leakage.write_csv(outdir / "anchor_leakage.csv")

    memberships = pl.read_parquet(anchors_dir / "anchor_memberships.parquet")
    entropy_summary = _membership_entropy_summary(memberships)
    entropy_summary.write_csv(outdir / "anchor_membership_entropy_summary.csv")

    neighborhoods_path = anchors_dir / "anchor_neighborhoods.parquet"
    if neighborhoods_path.exists():
        neighborhoods = pl.read_parquet(neighborhoods_path)
    else:
        coords = pl.read_csv(anchors_dir / "anchor_coords.csv")
        anchor_row_ids = coords.get_column("row_id").to_numpy().astype(int)
        row_ids = df.get_column("row_id").to_numpy().astype(int)
        row_index = {int(rid): idx for idx, rid in enumerate(row_ids)}
        anchor_indices = np.asarray([row_index[int(rid)] for rid in anchor_row_ids if int(rid) in row_index], dtype=int)
        neighborhoods = build_anchor_neighborhoods(
            X_dist,
            anchors=anchor_indices,
            countries=countries,
            row_ids=row_ids,
            neighbor_k=int(meta.get("neighbor_k", 0) or 0),
        )

    synth = _sample_synthetic(
        df,
        neighborhoods=neighborhoods,
        country_weights=country_weights,
        country_col=config.country_col,
        seed=config.seed,
    )

    comparison_rows: list[dict[str, object]] = []
    for feature in features:
        real = df.get_column(feature).to_numpy()
        syn = synth.get_column(feature).to_numpy()
        real = real[np.isfinite(real)]
        syn = syn[np.isfinite(syn)]
        if real.size == 0 or syn.size == 0:
            continue
        comparison_rows.append(
            {
                "feature": feature,
                "mean_real": float(np.mean(real)),
                "mean_synth": float(np.mean(syn)),
                "mean_diff": float(np.mean(syn) - np.mean(real)),
                "q25_real": float(np.quantile(real, 0.25)),
                "q25_synth": float(np.quantile(syn, 0.25)),
                "q75_real": float(np.quantile(real, 0.75)),
                "q75_synth": float(np.quantile(syn, 0.75)),
            }
        )

    if comparison_rows:
        real_mat = df.select(features).to_numpy().astype(float)
        synth_mat = synth.select(features).to_numpy().astype(float)
        comparison_rows.append(
            {
                "feature": "__corr__",
                "mean_real": float("nan"),
                "mean_synth": float("nan"),
                "mean_diff": float("nan"),
                "q25_real": float("nan"),
                "q25_synth": float("nan"),
                "q75_real": float("nan"),
                "q75_synth": float("nan"),
                "corr_mean_abs_diff": _mean_abs_corr_diff(real_mat, synth_mat),
            }
        )

    pl.DataFrame(comparison_rows).write_csv(outdir / "synthetic_real_comparison.csv")
