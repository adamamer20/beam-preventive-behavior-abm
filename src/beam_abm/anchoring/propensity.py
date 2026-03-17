"""Cross-fitted propensity score utilities."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
from beartype import beartype
from sklearn.linear_model import LinearRegression

from beam_abm.common.logging import get_logger
from beam_abm.empirical.io import ensure_outdir, parse_list_arg
from beam_abm.settings import CSV_FILE_CLEAN, SEED

logger = get_logger(__name__)


@dataclass(slots=True, frozen=False)
class PropensityConfig:
    input_csv: Path = Path(CSV_FILE_CLEAN)
    outdir: Path = Path("empirical/output/anchors")
    country_col: str = "country"
    target_cols: tuple[str, ...] = ()
    driver_cols: tuple[str, ...] = ()
    folds: int = 5
    slice_quantiles: tuple[float, float] = (0.33, 0.67)
    slice_min_n: int = 200
    seed: int = SEED


def _fit_model(X: np.ndarray, y: np.ndarray) -> tuple[LinearRegression, np.ndarray]:
    means = np.nanmean(X, axis=0)
    X = np.where(np.isnan(X), means, X)
    model = LinearRegression()
    model.fit(X, y)
    return model, means


def _predict(model: LinearRegression, means: np.ndarray, X: np.ndarray) -> np.ndarray:
    X = np.where(np.isnan(X), means, X)
    return model.predict(X)


@beartype
def compute_propensity_scores(
    config: PropensityConfig,
    *,
    driver_list: str | None = None,
    target_list: str | None = None,
) -> pl.DataFrame:
    """Compute cross-fitted propensity scores and optional slices.

    Parameters
    ----------
    config:
        PropensityConfig defining inputs and model columns.
    driver_list:
        Optional comma-separated override for driver columns.
    target_list:
        Optional comma-separated override for target columns.

    Returns
    -------
    pl.DataFrame
        Long-form table of out-of-sample propensity scores.
    """

    ensure_outdir(config.outdir)
    df = pl.read_csv(config.input_csv, infer_schema_length=10_000).with_row_index(name="row_id")

    driver_cols = config.driver_cols or tuple(parse_list_arg(driver_list))
    if not driver_cols:
        meta_path = Path(config.outdir) / "anchor_metadata.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            driver_cols = tuple(meta.get("features", []))
    if not driver_cols:
        raise ValueError("Driver columns required to compute propensity scores.")

    target_cols = config.target_cols or tuple(parse_list_arg(target_list))
    if not target_cols:
        raise ValueError("Target columns required to compute propensity scores.")

    q1, q2 = config.slice_quantiles
    if not (0.0 < q1 < q2 < 1.0):
        raise ValueError("slice_quantiles must be two values with 0 < q1 < q2 < 1")

    for col in driver_cols:
        if col not in df.columns:
            raise ValueError(f"Driver column not in dataset: {col}")

    results: list[dict[str, object]] = []
    cutpoints_rows: list[dict[str, object]] = []

    for target_col in target_cols:
        if target_col not in df.columns:
            raise ValueError(f"Target column not in dataset: {target_col}")

        valid = df.filter(pl.col(target_col).is_not_null())
        if valid.height == 0:
            logger.warning("Skipping propensity target with no data: {target}", target=target_col)
            continue

        X = valid.select(driver_cols).to_numpy().astype(float)
        y = valid.get_column(target_col).to_numpy().astype(float)
        row_ids = valid.get_column("row_id").to_numpy().astype(int)
        countries = valid.get_column(config.country_col).to_list()

        rng = np.random.default_rng(config.seed)
        indices = rng.permutation(len(row_ids))
        folds = np.array_split(indices, int(config.folds))
        oos_scores = np.full(len(row_ids), np.nan, dtype=float)

        for _fold_idx, test_idx in enumerate(folds):
            train_idx = np.setdiff1d(indices, test_idx)
            if train_idx.size == 0:
                continue
            model, means = _fit_model(X[train_idx], y[train_idx])
            preds = _predict(model, means, X[test_idx])
            oos_scores[test_idx] = preds

        for rid, country, score in zip(row_ids, countries, oos_scores, strict=True):
            results.append(
                {
                    "row_id": int(rid),
                    "country": country,
                    "outcome": target_col,
                    "propensity_score": float(score),
                }
            )

        pooled_scores = oos_scores[np.isfinite(oos_scores)]
        if pooled_scores.size == 0:
            continue
        pooled_cuts = np.quantile(pooled_scores, [q1, q2])

        for country in valid.get_column(config.country_col).unique().to_list():
            mask = np.array([c == country for c in countries])
            country_scores = oos_scores[mask]
            finite_scores = country_scores[np.isfinite(country_scores)]
            if finite_scores.size < config.slice_min_n:
                cut_low, cut_high = pooled_cuts
                cut_source = "pooled"
            else:
                cut_low, cut_high = np.quantile(finite_scores, [q1, q2])
                cut_source = "country"
            cutpoints_rows.append(
                {
                    "country": country,
                    "outcome": target_col,
                    "cut_low": float(cut_low),
                    "cut_high": float(cut_high),
                    "cut_source": cut_source,
                    "n": int(finite_scores.size),
                }
            )

    out = pl.DataFrame(results)
    out.write_parquet(config.outdir / "propensity_scores.parquet")

    if out.height > 0:
        cutpoints_df = pl.DataFrame(cutpoints_rows)
        cutpoints_df.write_csv(config.outdir / "propensity_cutpoints.csv")

        cuts_by_key = {(row["country"], row["outcome"]): (row["cut_low"], row["cut_high"]) for row in cutpoints_rows}

        slice_rows: list[dict[str, object]] = []
        for row in out.iter_rows(named=True):
            score = float(row["propensity_score"])
            if not np.isfinite(score):
                continue
            cut_low, cut_high = cuts_by_key.get((row["country"], row["outcome"]))
            if score <= cut_low:
                label = "low"
            elif score <= cut_high:
                label = "mid"
            else:
                label = "high"
            slice_rows.append({**row, "slice": label})

        pl.DataFrame(slice_rows).write_csv(config.outdir / "propensity_slices.csv")

    (config.outdir / "propensity_metadata.json").write_text(
        json.dumps(
            {
                "target_cols": list(target_cols),
                "driver_cols": list(driver_cols),
                "folds": int(config.folds),
                "slice_quantiles": list(config.slice_quantiles),
                "slice_min_n": int(config.slice_min_n),
            },
            indent=2,
        )
    )

    return out
