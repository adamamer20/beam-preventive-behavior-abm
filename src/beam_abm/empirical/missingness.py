"""Utilities for handling missing/non-finite values consistently."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
import polars as pl

from beam_abm.common.logging import get_logger

logger = get_logger(__name__)

STRUCTURAL_MISSINGNESS_COLUMNS: tuple[str, ...] = (
    "colleagues_immigration_tolerance",
    "colleagues_lgbt_adoption_rights",
    "colleagues_climate_harm",
    "colleagues_vaccines_importance",
    "colleagues_progressive_norms_nonvax_idx",
    "trust_traditional",
    "trust_onlineinfo",
    "trust_social",
)

DEFAULT_MISSINGNESS_THRESHOLD = 0.35
MEDIAN_FILL_COLUMNS: tuple[str, ...] = ("income_ppp_norm",)
COUNTRY_MEDIAN_FILL_COLUMNS: tuple[str, ...] = ("income_ppp_norm",)
SPECIAL_STRUCTURAL_PREFIXES: tuple[str, ...] = ("colleagues_",)

__all__ = [
    "DEFAULT_MISSINGNESS_THRESHOLD",
    "COUNTRY_MEDIAN_FILL_COLUMNS",
    "MEDIAN_FILL_COLUMNS",
    "MissingnessPlan",
    "SPECIAL_STRUCTURAL_PREFIXES",
    "STRUCTURAL_MISSINGNESS_COLUMNS",
    "apply_missingness_plan_pd",
    "apply_missingness_plan_pl",
    "build_missingness_plan_pd",
    "build_missingness_plan_pl",
    "missingness_indicator_name",
    "init_structural_missingness_indicators_pd",
    "finalize_structural_missingness_values_pd",
    "apply_structural_missingness_pl",
    "drop_nonfinite_rows_pd",
    "drop_nonfinite_rows_pl",
]


def missingness_indicator_name(column: str) -> str:
    return f"{column}_missing"


@dataclass(frozen=True)
class MissingnessPlan:
    columns: tuple[str, ...]
    threshold: float
    fill_strategy: dict[str, str]
    fill_values: dict[str, float]
    country_fill_values: dict[str, dict[object, float]]
    standardize_stats: dict[str, tuple[float, float]]


_NUMERIC_TYPES = {"continuous", "ordinal", "binary", "boolean"}


def _is_numeric_col_pd(name: str, series: pd.Series, column_types: Mapping[str, str]) -> bool:
    col_type = column_types.get(name, "").lower()
    if col_type in _NUMERIC_TYPES:
        return True
    if col_type and col_type not in _NUMERIC_TYPES:
        return False
    return pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series)


def _is_binary_col_pd(name: str, series: pd.Series, column_types: Mapping[str, str]) -> bool:
    col_type = column_types.get(name, "").lower()
    if col_type in {"binary", "boolean"}:
        return True
    if pd.api.types.is_bool_dtype(series):
        return True
    numeric = pd.to_numeric(series, errors="coerce")
    finite = numeric[np.isfinite(numeric.to_numpy())]
    unique = pd.unique(finite)
    if len(unique) == 0:
        return False
    return set(unique.tolist()) <= {0.0, 1.0}


def _missing_mask_pd(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric.isna() | ~np.isfinite(numeric.to_numpy())


def _fill_stats_pd(series: pd.Series, *, strategy: str, country: pd.Series | None) -> tuple[float, dict[object, float]]:
    numeric = pd.to_numeric(series, errors="coerce")
    finite = numeric[np.isfinite(numeric.to_numpy())]
    if strategy == "binary":
        return 0.0, {}
    if strategy == "country_mean" and country is not None:
        grouped = (
            pd.DataFrame({"value": numeric, "country": country})
            .assign(value=lambda df: pd.to_numeric(df["value"], errors="coerce"))
            .dropna(subset=["value"])
            .groupby("country")["value"]
            .mean()
        )
        fill_values = grouped.to_dict()
        mean = float(finite.mean()) if not finite.empty else 0.0
        if not math.isfinite(mean):
            mean = 0.0
        return mean, fill_values
    if strategy == "country_median" and country is not None:
        grouped = (
            pd.DataFrame({"value": numeric, "country": country})
            .assign(value=lambda df: pd.to_numeric(df["value"], errors="coerce"))
            .dropna(subset=["value"])
            .groupby("country")["value"]
            .median()
        )
        fill_values = grouped.to_dict()
        median = float(finite.median()) if not finite.empty else 0.0
        if not math.isfinite(median):
            median = 0.0
        return median, fill_values
    if strategy == "median":
        median = float(finite.median()) if not finite.empty else 0.0
        return median if math.isfinite(median) else 0.0, {}
    mean = float(finite.mean()) if not finite.empty else 0.0
    return mean if math.isfinite(mean) else 0.0, {}


def build_missingness_plan_pd(
    df: pd.DataFrame,
    predictors: Sequence[str],
    *,
    threshold: float = DEFAULT_MISSINGNESS_THRESHOLD,
    column_types: Mapping[str, str],
    country_col: str | None = None,
) -> MissingnessPlan:
    columns: list[str] = []
    fill_strategy: dict[str, str] = {}
    fill_values: dict[str, float] = {}
    country_fill_values: dict[str, dict[object, float]] = {}
    standardize_stats: dict[str, tuple[float, float]] = {}

    for col in predictors:
        if col not in df.columns:
            continue
        series = df[col]
        if not _is_numeric_col_pd(col, series, column_types):
            continue
        missing_frac = float(_missing_mask_pd(series).mean())
        is_special_structural = col.startswith(SPECIAL_STRUCTURAL_PREFIXES)
        if missing_frac <= 0.0:
            continue
        if (
            missing_frac > threshold
            and not is_special_structural
            and col not in COUNTRY_MEDIAN_FILL_COLUMNS
            and col not in MEDIAN_FILL_COLUMNS
        ):
            continue

        strategy = "mean"
        if is_special_structural and country_col and country_col in df.columns:
            strategy = "country_mean"
        elif col in COUNTRY_MEDIAN_FILL_COLUMNS and country_col and country_col in df.columns:
            strategy = "country_median"
        elif _is_binary_col_pd(col, series, column_types):
            strategy = "binary"
        elif col in MEDIAN_FILL_COLUMNS:
            strategy = "median"

        country_series = df[country_col] if country_col and country_col in df.columns else None
        fill_value, country_values = _fill_stats_pd(series, strategy=strategy, country=country_series)
        numeric = pd.to_numeric(series, errors="coerce")
        finite = numeric[np.isfinite(numeric.to_numpy())]
        mean = float(finite.mean()) if not finite.empty else 0.0
        std = float(finite.std(ddof=0)) if not finite.empty else 0.0
        standardize_stats[col] = (mean if math.isfinite(mean) else 0.0, std if math.isfinite(std) else 0.0)

        columns.append(col)
        fill_strategy[col] = strategy
        fill_values[col] = fill_value
        if country_values:
            country_fill_values[col] = country_values

    return MissingnessPlan(
        columns=tuple(columns),
        threshold=float(threshold),
        fill_strategy=fill_strategy,
        fill_values=fill_values,
        country_fill_values=country_fill_values,
        standardize_stats=standardize_stats,
    )


def apply_missingness_plan_pd(
    df: pd.DataFrame,
    plan: MissingnessPlan,
    *,
    standardize: bool,
    country_col: str | None = None,
) -> pd.DataFrame:
    if not plan.columns:
        return df

    out = df.copy()
    for col in plan.columns:
        if col not in out.columns:
            continue
        missing_mask = _missing_mask_pd(out[col])
        indicator = missingness_indicator_name(col)
        if indicator not in out.columns:
            out[indicator] = missing_mask.astype(float)

        fill_value: float | pd.Series
        if (
            plan.fill_strategy.get(col) in {"country_median", "country_mean"}
            and country_col
            and country_col in out.columns
        ):
            mapping = plan.country_fill_values.get(col, {})
            fill_value = out[country_col].map(mapping).astype(float)
            fill_value = fill_value.fillna(plan.fill_values.get(col, 0.0))
        else:
            fill_value = float(plan.fill_values.get(col, 0.0))

        if standardize:
            mean, std = plan.standardize_stats.get(col, (0.0, 0.0))
            if std and not math.isclose(std, 0.0):
                if isinstance(fill_value, pd.Series):
                    fill_value = (fill_value - mean) / std
                else:
                    fill_value = (float(fill_value) - mean) / std
            else:
                fill_value = pd.Series(0.0, index=out.index) if isinstance(fill_value, pd.Series) else 0.0

        numeric = pd.to_numeric(out[col], errors="coerce")
        if isinstance(fill_value, pd.Series):
            fill_value = fill_value.reindex(out.index)
            out[col] = numeric.where(np.isfinite(numeric.to_numpy()), fill_value)
        else:
            out[col] = numeric.where(np.isfinite(numeric.to_numpy()), fill_value)

    return out


def _missing_fraction_pl(df: pl.DataFrame, col: str) -> float:
    dtype = df.schema.get(col)
    if dtype is None:
        return 1.0
    base = pl.col(col).is_null()
    if dtype.is_numeric() and dtype != pl.Boolean:
        col_float = pl.col(col).cast(pl.Float64, strict=False)
        base = base | col_float.is_nan() | ~col_float.is_finite()
    elif dtype == pl.String:
        col_float = pl.col(col).cast(pl.Float64, strict=False)
        base = base | col_float.is_null() | col_float.is_nan() | ~col_float.is_finite()
    return float(df.select(base.mean()).item())


def _is_binary_col_pl(df: pl.DataFrame, col: str) -> bool:
    dtype = df.schema.get(col)
    if dtype is None:
        return False
    if dtype == pl.Boolean:
        return True
    if not dtype.is_numeric():
        return False
    unique_vals = (
        df.select(
            pl.when(pl.col(col).cast(pl.Float64, strict=False).is_finite())
            .then(pl.col(col).cast(pl.Float64, strict=False))
            .otherwise(None)
            .unique()
        )
        .to_series()
        .drop_nulls()
        .to_list()
    )
    if not unique_vals:
        return False
    return set(unique_vals) <= {0.0, 1.0}


def build_missingness_plan_pl(
    df: pl.DataFrame,
    predictors: Sequence[str],
    *,
    threshold: float = DEFAULT_MISSINGNESS_THRESHOLD,
    country_col: str | None = None,
) -> MissingnessPlan:
    columns: list[str] = []
    fill_strategy: dict[str, str] = {}
    fill_values: dict[str, float] = {}
    country_fill_values: dict[str, dict[object, float]] = {}
    standardize_stats: dict[str, tuple[float, float]] = {}

    for col in predictors:
        if col not in df.columns:
            continue
        dtype = df.schema.get(col)
        is_numeric_like = bool(
            dtype is not None
            and (
                dtype.is_numeric()
                or dtype == pl.Boolean
                or (dtype == pl.String and (col in COUNTRY_MEDIAN_FILL_COLUMNS or col in MEDIAN_FILL_COLUMNS))
            )
        )
        if not is_numeric_like:
            continue
        missing_frac = _missing_fraction_pl(df, col)
        is_special_structural = col.startswith(SPECIAL_STRUCTURAL_PREFIXES)
        if missing_frac <= 0.0:
            continue
        if (
            missing_frac > threshold
            and not is_special_structural
            and col not in COUNTRY_MEDIAN_FILL_COLUMNS
            and col not in MEDIAN_FILL_COLUMNS
        ):
            continue

        strategy = "mean"
        if is_special_structural and country_col and country_col in df.columns:
            strategy = "country_mean"
        elif col in COUNTRY_MEDIAN_FILL_COLUMNS and country_col and country_col in df.columns:
            strategy = "country_median"
        elif _is_binary_col_pl(df, col):
            strategy = "binary"
        elif col in MEDIAN_FILL_COLUMNS:
            strategy = "median"

        values = df.select(
            pl.when(pl.col(col).cast(pl.Float64, strict=False).is_finite())
            .then(pl.col(col).cast(pl.Float64, strict=False))
            .otherwise(None)
        ).to_series()
        finite = values.drop_nulls()
        if strategy == "binary":
            fill_values[col] = 0.0
        elif strategy == "median":
            med = float(finite.median()) if finite.len() else 0.0
            fill_values[col] = med if math.isfinite(med) else 0.0
        elif strategy == "country_mean" and country_col and country_col in df.columns:
            grouped = (
                df.group_by(country_col)
                .agg(
                    pl.when(pl.col(col).cast(pl.Float64, strict=False).is_finite())
                    .then(pl.col(col).cast(pl.Float64, strict=False))
                    .otherwise(None)
                    .mean()
                    .alias("fill")
                )
                .drop_nulls()
            )
            fill_values[col] = float(finite.mean()) if finite.len() else 0.0
            country_fill_values[col] = dict(
                zip(grouped[country_col].to_list(), grouped["fill"].to_list(), strict=False)
            )
        elif strategy == "country_median" and country_col and country_col in df.columns:
            grouped = (
                df.group_by(country_col)
                .agg(
                    pl.when(pl.col(col).cast(pl.Float64, strict=False).is_finite())
                    .then(pl.col(col).cast(pl.Float64, strict=False))
                    .otherwise(None)
                    .median()
                    .alias("fill")
                )
                .drop_nulls()
            )
            fill_values[col] = float(finite.median()) if finite.len() else 0.0
            country_fill_values[col] = dict(
                zip(grouped[country_col].to_list(), grouped["fill"].to_list(), strict=False)
            )
        else:
            mean_val = float(finite.mean()) if finite.len() else 0.0
            fill_values[col] = mean_val if math.isfinite(mean_val) else 0.0

        mean = float(finite.mean()) if finite.len() else 0.0
        std = float(finite.std(ddof=0)) if finite.len() else 0.0
        standardize_stats[col] = (mean if math.isfinite(mean) else 0.0, std if math.isfinite(std) else 0.0)

        columns.append(col)
        fill_strategy[col] = strategy

    return MissingnessPlan(
        columns=tuple(columns),
        threshold=float(threshold),
        fill_strategy=fill_strategy,
        fill_values=fill_values,
        country_fill_values=country_fill_values,
        standardize_stats=standardize_stats,
    )


def apply_missingness_plan_pl(
    df: pl.DataFrame,
    plan: MissingnessPlan,
    *,
    country_col: str | None = None,
) -> pl.DataFrame:
    if df.is_empty() or not plan.columns:
        return df

    out = df
    for col in plan.columns:
        if col not in out.columns:
            continue
        dtype = out.schema.get(col)
        is_numeric_like = bool(
            dtype is not None
            and (
                dtype.is_numeric()
                or dtype == pl.Boolean
                or (dtype == pl.String and (col in COUNTRY_MEDIAN_FILL_COLUMNS or col in MEDIAN_FILL_COLUMNS))
            )
        )
        if not is_numeric_like:
            continue
        indicator = missingness_indicator_name(col)
        col_float = pl.col(col).cast(pl.Float64, strict=False)
        missing_expr = pl.col(col).is_null() | col_float.is_null() | col_float.is_nan() | ~col_float.is_finite()
        if indicator not in out.columns:
            out = out.with_columns(missing_expr.cast(pl.Float64).alias(indicator))

        fill_value = plan.fill_values.get(col, 0.0)
        if (
            plan.fill_strategy.get(col) in {"country_median", "country_mean"}
            and country_col
            and country_col in out.columns
        ):
            mapping = plan.country_fill_values.get(col, {})
            if mapping:
                key_dtype = out.schema.get(country_col) or pl.Utf8
                fill_df = pl.DataFrame(
                    {
                        country_col: list(mapping.keys()),
                        f"{col}__fill": list(mapping.values()),
                    },
                    schema={country_col: key_dtype, f"{col}__fill": pl.Float64},
                )
                out = out.join(fill_df, on=country_col, how="left")
                fill_expr = pl.coalesce([pl.col(f"{col}__fill"), pl.lit(fill_value)])
                out = out.with_columns(pl.when(missing_expr).then(fill_expr).otherwise(col_float).alias(col))
                out = out.drop(f"{col}__fill")
            else:
                out = out.with_columns(pl.when(missing_expr).then(pl.lit(fill_value)).otherwise(col_float).alias(col))
        else:
            out = out.with_columns(pl.when(missing_expr).then(pl.lit(fill_value)).otherwise(col_float).alias(col))

    return out


def init_structural_missingness_indicators_pd(df: pd.DataFrame) -> dict[str, pd.Series]:
    originals: dict[str, pd.Series] = {}
    for column in STRUCTURAL_MISSINGNESS_COLUMNS:
        if column not in df.columns:
            continue
        originals[column] = df[column].copy()
        indicator = missingness_indicator_name(column)
        if indicator not in df.columns:
            df[indicator] = df[column].isna().astype(float)
    return originals


def finalize_structural_missingness_values_pd(
    df: pd.DataFrame,
    *,
    standardize: bool,
    originals: dict[str, pd.Series],
) -> None:
    if not originals:
        return
    for column, original in originals.items():
        if column not in df.columns:
            continue
        if standardize:
            df[column] = df[column].fillna(0.0)
        else:
            mean_val = float(original.mean(skipna=True)) if not original.dropna().empty else 0.0
            if not np.isfinite(mean_val):
                mean_val = 0.0
            df[column] = df[column].fillna(mean_val)


def apply_structural_missingness_pl(
    df: pl.DataFrame,
    *,
    standardize: bool = False,
    fill_means: dict[str, float] | None = None,
) -> pl.DataFrame:
    if df.is_empty():
        return df

    updates: list[pl.Expr] = []
    for column in STRUCTURAL_MISSINGNESS_COLUMNS:
        if column not in df.columns:
            continue
        indicator = missingness_indicator_name(column)
        dtype = df.schema.get(column)
        if dtype is None:
            continue
        if dtype is not None and dtype.is_numeric() and dtype != pl.Boolean:
            col_float = pl.col(column).cast(pl.Float64, strict=False)
            is_missing = pl.col(column).is_null() | col_float.is_nan()
        else:
            is_missing = pl.col(column).is_null()
        if indicator not in df.columns:
            updates.append(is_missing.cast(pl.Float64).alias(indicator))

        if dtype.is_numeric() and dtype != pl.Boolean:
            if standardize:
                filled = pl.col(column).fill_null(0.0).fill_nan(0.0)
            else:
                mean_val = None
                if fill_means is not None and column in fill_means:
                    mean_val = float(fill_means[column])
                else:
                    mean_val = df.select(
                        pl.when(pl.col(column).cast(pl.Float64, strict=False).is_finite())
                        .then(pl.col(column).cast(pl.Float64, strict=False))
                        .otherwise(None)
                        .mean()
                    ).item()
                if mean_val is None or not np.isfinite(mean_val):
                    mean_val = 0.0
                filled = pl.col(column).fill_null(mean_val).fill_nan(mean_val)
            updates.append(filled.alias(column))
    if not updates:
        return df
    return df.with_columns(updates)


def drop_nonfinite_rows_pd(df: pd.DataFrame, cols: Sequence[str], *, context: str) -> pd.DataFrame:
    """Drop rows with null/NaN/inf values in specified columns (pandas)."""
    if cols is None or len(cols) == 0 or df.empty:
        return df

    mask = np.ones(len(df), dtype=bool)
    for col in cols:
        series = df[col]
        if pd.api.types.is_numeric_dtype(series):
            mask &= np.isfinite(series.to_numpy())
        else:
            mask &= series.notna().to_numpy()

    dropped = int(len(df) - np.count_nonzero(mask))
    if dropped:
        logger.warning("Dropping {count} rows with non-finite values in {context}.", count=dropped, context=context)

    return df.loc[mask].copy()


def drop_nonfinite_rows_pl(df: pl.DataFrame, cols: Sequence[str], *, context: str) -> pl.DataFrame:
    """Drop rows with null/NaN/inf values in specified columns (polars)."""
    if not cols or df.is_empty():
        return df

    exprs = []
    for col in cols:
        dtype = df.schema.get(col)
        base = pl.col(col).is_not_null()
        if dtype is None:
            exprs.append(base)
            continue
        if dtype.is_numeric() and dtype != pl.Boolean:
            exprs.append(base & pl.col(col).is_finite())
        elif dtype == pl.Boolean:
            exprs.append(base)
        elif dtype == pl.String:
            col_float = pl.col(col).cast(pl.Float64, strict=False)
            exprs.append(base & col_float.is_not_null() & col_float.is_finite())
        else:
            exprs.append(base)
    cleaned = df.filter(pl.all_horizontal(exprs))
    dropped = df.height - cleaned.height
    if dropped:
        logger.warning("Dropping {count} rows with non-finite values in {context}.", count=dropped, context=context)

    return cleaned
