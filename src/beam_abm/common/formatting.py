"""Shared helpers for rounding reporting outputs."""

from __future__ import annotations

import math
from collections.abc import Iterable

import pandas as pd
import polars as pl


def round_sigfigs(value: float | int | None, sigfigs: int) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        value = int(value)
    if not isinstance(value, int | float):
        return None
    if not math.isfinite(float(value)):
        return None
    if value == 0:
        return 0.0
    if sigfigs <= 0:
        return float(value)
    decimals = sigfigs - 1 - int(math.floor(math.log10(abs(float(value)))))
    return float(round(float(value), decimals))


def round_sigfigs_polars(df: pl.DataFrame, sigfigs: int) -> pl.DataFrame:
    if df.height == 0:
        return df
    float_types = {pl.Float32, pl.Float64}
    float_cols = [col for col, dtype in zip(df.columns, df.dtypes, strict=True) if dtype in float_types]
    if not float_cols or sigfigs <= 0:
        return df
    rounded_exprs = []
    for col in float_cols:
        expr = pl.col(col)
        abs_expr = expr.abs()
        mask = (expr.is_finite() & (abs_expr > 0)).fill_null(False)
        safe_abs = pl.when(mask).then(abs_expr).otherwise(pl.lit(1.0))
        digits = (
            pl.when(mask).then((pl.lit(sigfigs) - 1 - safe_abs.log10().floor()).cast(pl.Int32)).otherwise(pl.lit(0))
        )
        scale = pl.lit(10.0) ** digits
        rounded = pl.when(mask).then((expr * scale).round(0) / scale).otherwise(expr).alias(col)
        rounded_exprs.append(rounded)
    return df.with_columns(rounded_exprs)


def round_sigfigs_pandas(df: pd.DataFrame, sigfigs: int) -> pd.DataFrame:
    if df.empty or sigfigs <= 0:
        return df
    float_cols: Iterable[str] = df.select_dtypes(include=["floating"]).columns
    if not float_cols:
        return df
    copy = df.copy()
    for col in float_cols:
        series = copy[col]

        def _round(v: float | int | None) -> float | None:
            if v is None:
                return None
            try:
                return round_sigfigs(float(v), sigfigs)
            except Exception:
                return None

        copy[col] = series.apply(_round)
    return copy
