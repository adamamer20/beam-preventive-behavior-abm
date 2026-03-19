from __future__ import annotations

import json
import os
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
from beartype import beartype

from beam_abm.common.logging import get_logger
from beam_abm.empirical.io import (
    build_type_buckets,
    ensure_outdir,
    load_column_types,
    parse_list_arg,
    write_grouped_views,
)
from beam_abm.empirical.taxonomy import DEFAULT_COLUMN_TYPES_PATH
from beam_abm.settings import CSV_FILE_CLEAN

logger = get_logger(__name__)

# Write analysis outputs under the dataset analysis folder
DEFAULT_OUTDIR = Path("empirical/output/descriptives")

COVERAGE_ONLY_INDICES = {
    "cohabitant_age",
    "cohabitant_covax_num",
    "cohabitant_covid_num",
    "duration_infomedia",
    "duration_socialmedia",
    "noncohabitant_age",
    "noncohabitant_covax_num",
    "noncohabitant_covid_num",
}


@dataclass(slots=True)
class DescriptivesConfig:
    input_csv: Path
    outdir: Path
    column_types_path: Path
    spec_path: Path | None
    compute_alpha: bool
    categorical_overrides: tuple[str, ...] = ()
    # Allow callers to force columns into the categorical bucket


# NOTE: removed backward-compatible alias map — callers should request
# canonical column names that exist in the dataset. Legacy aliasing was
# removed to simplify behavior and avoid silent remapping.


@dataclass(slots=True)
class DescriptivesResult:
    continuous: pl.DataFrame
    ordinal_summary: pl.DataFrame
    ordinal_distribution: pl.DataFrame
    binary_summary: pl.DataFrame
    binary_distribution: pl.DataFrame
    categorical_distribution: pl.DataFrame
    reliability: pl.DataFrame


def _safe_round(value: float | int | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        value = int(value)
    if isinstance(value, int | float) and np.isfinite(value):
        return float(value)
    return None


def _stable_sort_values(values: Sequence[object]) -> list[object]:
    """Return values sorted in a deterministic way across types."""

    return sorted(values, key=lambda v: (v.__class__.__name__, str(v)))


@beartype
def _continuous_summary(df: pl.DataFrame, columns: Sequence[str]) -> pl.DataFrame:
    if not columns:
        return pl.DataFrame({})

    rows: list[dict[str, object]] = []
    for col in columns:
        if col not in df.columns:
            logger.warning("Continuous column not found in dataframe: {col}", col=col)
            continue

        series = df.get_column(col).cast(pl.Float64, strict=False)
        total = series.len()
        n_missing = series.null_count()
        non_null = series.drop_nulls()
        n = non_null.len()
        row: dict[str, object] = {
            "variable": col,
            "n": int(n),
            "mean": None,
            "std": None,
            "median": None,
            "iqr": None,
            "p05": None,
            "p95": None,
            "skew": None,
            "kurtosis": None,
            "missing_pct": _safe_round((n_missing / total * 100.0) if total else 0.0),
        }
        if n:
            mean = non_null.mean()
            std = non_null.std()
            median = non_null.median()
            q1 = non_null.quantile(0.25)
            q3 = non_null.quantile(0.75)
            iqr = (q3 - q1) if (q3 is not None and q1 is not None) else None
            p05 = non_null.quantile(0.05)
            p95 = non_null.quantile(0.95)
            skew = non_null.skew()
            kurt = non_null.kurtosis()

            row.update(
                {
                    "mean": _safe_round(mean),
                    "std": _safe_round(std),
                    "median": _safe_round(median),
                    "iqr": _safe_round(iqr),
                    "p05": _safe_round(p05),
                    "p95": _safe_round(p95),
                    "skew": _safe_round(skew),
                    "kurtosis": _safe_round(kurt),
                }
            )
        rows.append(row)

    if not rows:
        return pl.DataFrame({})

    col_order = [
        "variable",
        "n",
        "mean",
        "std",
        "median",
        "iqr",
        "p05",
        "p95",
        "skew",
        "kurtosis",
        "missing_pct",
    ]
    return pl.DataFrame(rows)[col_order]


@beartype
def _ordinal_summary(df: pl.DataFrame, columns: Sequence[str]) -> tuple[pl.DataFrame, pl.DataFrame]:
    if not columns:
        return pl.DataFrame({}), pl.DataFrame({})

    summary_rows: list[dict[str, object]] = []
    dist_rows: list[dict[str, object]] = []

    for col in columns:
        if col not in df.columns:
            logger.warning("Ordinal column not found in dataframe: {col}", col=col)
            continue

        series = df.get_column(col)
        total = series.len()
        n_missing = series.null_count()
        non_null = series.drop_nulls()
        n = non_null.len()

        numeric_non_null = non_null.cast(pl.Float64, strict=False).drop_nulls()
        median = numeric_non_null.median() if numeric_non_null.len() else None
        q1 = numeric_non_null.quantile(0.25) if numeric_non_null.len() else None
        q3 = numeric_non_null.quantile(0.75) if numeric_non_null.len() else None
        iqr = (q3 - q1) if (q3 is not None and q1 is not None) else None

        summary_rows.append(
            {
                "variable": col,
                "n": int(n),
                "median": _safe_round(median),
                "iqr": _safe_round(iqr),
                "missing_pct": _safe_round((n_missing / total * 100.0) if total else 0.0),
            }
        )

        if n == 0:
            continue

        value_counts = non_null.to_frame(name=col).group_by(col).len().rename({"len": "count"}).sort(col)
        values = value_counts[col].to_list()
        counts = value_counts["count"].to_list()
        for value, count in zip(values, counts, strict=False):
            pct = count / n * 100.0 if n else 0.0
            dist_rows.append(
                {
                    "variable": col,
                    "value": str(value),
                    "count": int(count),
                    "pct": _safe_round(pct),
                }
            )

    return pl.DataFrame(summary_rows), pl.DataFrame(dist_rows)


def _choose_positive_label(unique_values: Sequence[object]) -> object | None:
    if not unique_values:
        return None

    unique_values = _stable_sort_values(unique_values)

    for val in unique_values:
        if isinstance(val, bool) and val:
            return val

    for val in unique_values:
        if isinstance(val, int | float) and val == 1:
            return val

    normalised = {val: str(val).strip().lower() for val in unique_values}
    preferred = [
        "yes",
        "true",
        "female",
        "y",
        "positive",
        "present",
        "active",
        "vaccinated",
        "employed",
        "1",
    ]
    for candidate in preferred:
        for val, norm in normalised.items():
            if norm == candidate:
                return val

    if len(unique_values) == 2:
        # Fall back to the value that is not obviously false-like.
        false_like = {0, "0", "false", "no"}
        for val in unique_values:
            if str(val).strip().lower() not in false_like:
                return val
        return sorted(unique_values, key=lambda x: str(x))[-1]

    return None


@beartype
def _binary_summary(df: pl.DataFrame, columns: Sequence[str]) -> tuple[pl.DataFrame, pl.DataFrame]:
    if not columns:
        return pl.DataFrame({}), pl.DataFrame({})

    summary_rows: list[dict[str, object]] = []
    dist_rows: list[dict[str, object]] = []

    for col in columns:
        if col not in df.columns:
            logger.warning("Binary column not found in dataframe: {col}", col=col)
            continue

        series = df.get_column(col)
        total = series.len()
        n_missing = series.null_count()
        non_null = series.drop_nulls()
        n = non_null.len()

        unique_values = non_null.unique().to_list()
        positive_label = _choose_positive_label(unique_values)

        positive_count = None
        positive_pct = None
        if positive_label is not None:
            positive_count = non_null.to_frame(name=col).select(pl.col(col) == positive_label).to_series(0).sum()
            if isinstance(positive_count, int | float):
                positive_count = int(positive_count)
            else:
                positive_count = int(non_null.filter(non_null == positive_label).len())
            positive_pct = _safe_round((positive_count / n * 100.0) if n else None)
        else:
            if n:
                logger.warning(
                    "Could not infer positive label for binary column {col}; skipping prevalence",
                    col=col,
                )

        summary_rows.append(
            {
                "variable": col,
                "n": int(n),
                "positive_label": str(positive_label) if positive_label is not None else None,
                "positive_count": positive_count,
                "positive_pct": positive_pct,
                "missing_pct": _safe_round((n_missing / total * 100.0) if total else 0.0),
            }
        )

        if n == 0:
            continue

        value_counts = non_null.to_frame(name=col).group_by(col).len().rename({"len": "count"}).sort(col)
        values = value_counts[col].to_list()
        counts = value_counts["count"].to_list()
        for value, count in zip(values, counts, strict=False):
            pct = count / n * 100.0 if n else 0.0
            dist_rows.append(
                {
                    "variable": col,
                    "value": str(value),
                    "count": int(count),
                    "pct": _safe_round(pct),
                }
            )

    return pl.DataFrame(summary_rows), pl.DataFrame(dist_rows)


@beartype
def _categorical_distribution(df: pl.DataFrame, columns: Sequence[str]) -> pl.DataFrame:
    if not columns:
        return pl.DataFrame({})

    out_rows: list[pl.DataFrame] = []

    for col in columns:
        if col not in df.columns:
            logger.warning("Categorical column not found: {col}", col=col)
            continue

        non_missing_total = df.select(pl.col(col).is_not_null().sum().alias("n_non_missing")).item()
        missing_total = df.select(pl.col(col).is_null().sum().alias("n_missing")).item()
        total = non_missing_total + missing_total
        missing_pct = (missing_total / total * 100.0) if total else 0.0

        if non_missing_total == 0:
            out_rows.append(
                pl.DataFrame(
                    {
                        "variable": [col],
                        "category": ["Missing"],
                        "count": [None],
                        "pct": [None],
                        "missing_pct": [_safe_round(missing_pct)],
                    }
                )
            )
            continue

        counts = (
            df.select(pl.col(col))
            .drop_nulls()
            .group_by(col)
            .len()
            .rename({"len": "count"})
            .with_columns((pl.col("count") / non_missing_total * 100.0).alias("pct"))
            .rename({col: "category"})
            .with_columns(pl.col("category").cast(pl.String))
            .with_columns(pl.lit(col).alias("variable"))
            .with_columns(pl.lit(_safe_round(missing_pct)).alias("missing_pct"))
            .select("variable", "category", "count", "pct", "missing_pct")
            .sort("category")
        )

        out_rows.append(counts)
        out_rows.append(
            pl.DataFrame(
                {
                    "variable": [col],
                    "category": ["Missing"],
                    "count": [None],
                    "pct": [None],
                    "missing_pct": [_safe_round(missing_pct)],
                }
            )
        )

    if not out_rows:
        return pl.DataFrame({})
    combined = pl.concat(out_rows)
    return combined.sort(["variable", "category"], nulls_last=True)


@beartype
def _load_index_groups(spec_path: Path | None) -> dict[str, list[str]]:
    if spec_path is None or not spec_path.exists():
        return {}

    try:
        with spec_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Failed to read spec for index grouping: {e}", e=str(exc))
        return {}

    new_columns = data.get("new_columns") or []
    rename_map: dict[str, str] = {}
    for entry in new_columns:
        col_id = entry.get("id")
        if not col_id:
            continue
        for trans in entry.get("transformations") or []:
            if trans.get("type") == "renamed" and trans.get("rename_from"):
                rename_map[str(trans["rename_from"])] = col_id

    groups: dict[str, list[str]] = {}
    for entry in new_columns:
        col_id = entry.get("id")
        if not col_id:
            continue
        transformations = entry.get("transformations") or []
        for trans in transformations:
            if trans.get("type") != "merged":
                continue
            merged_from = trans.get("merged_from") or []
            merging_type = trans.get("merging_type")
            if len(merged_from) < 2:
                continue
            if merging_type not in {"mean", "sum"}:
                continue
            resolved = [rename_map.get(str(src), str(src)) for src in merged_from]
            groups[col_id] = resolved
            break
    return groups


@beartype
def _cronbach_alpha(arr: np.ndarray) -> float | None:
    if arr.ndim != 2:
        return None
    # Drop items with fewer than two valid responses
    valid_per_item = np.sum(np.isfinite(arr), axis=0)
    keep_mask = valid_per_item >= 2
    if not keep_mask.any():
        return None
    arr = arr[:, keep_mask]
    n_obs, n_items = arr.shape
    if n_items < 2:
        return None
    # Keep rows with at least two answered items
    row_non_nulls = np.sum(np.isfinite(arr), axis=1)
    arr = arr[row_non_nulls >= 2]
    n_obs = arr.shape[0]
    if n_obs < 2:
        return None

    item_vars = np.nanvar(arr, axis=0, ddof=1)
    total_scores = np.nansum(arr, axis=1)
    total_var = np.nanvar(total_scores, ddof=1)

    if not np.isfinite(total_var) or np.isclose(total_var, 0.0):
        return None

    alpha = (n_items / (n_items - 1.0)) * (1.0 - item_vars.sum() / total_var)
    if not np.isfinite(alpha):
        return None
    return float(np.clip(alpha, -1.0, 1.0))


@beartype
def _compute_index_reliability(
    df: pl.DataFrame, groups: dict[str, list[str]], skip_alpha_for: Iterable[str] = ()
) -> pl.DataFrame:
    if not groups:
        return pl.DataFrame({})

    rows: list[dict[str, object]] = []
    total_rows = df.height
    skipped: list[tuple[str, list[str]]] = []
    skip_alpha = set(skip_alpha_for)

    for index_name, items in sorted(groups.items()):
        missing = [col for col in items if col not in df.columns]
        if missing:
            # If the aggregate column itself exists and alpha is not desired, still report coverage.
            if index_name in skip_alpha and index_name in df.columns:
                coverage = df.select(pl.col(index_name).is_not_null().sum()).item()
                coverage_pct = (coverage / total_rows * 100.0) if total_rows else 0.0
                rows.append(
                    {
                        "index": index_name,
                        "items": index_name,
                        "n_items": 1,
                        "n_complete": int(coverage),
                        "coverage_pct": coverage_pct,
                        "cronbach_alpha": None,
                    }
                )
                continue

            skipped.append((index_name, missing))
            continue

        numeric_subset = df.select([pl.col(col).cast(pl.Float64, strict=False).alias(col) for col in items])
        non_null_count = pl.sum_horizontal(*[pl.col(col).is_not_null().cast(pl.Int64) for col in items]).alias(
            "non_nulls"
        )
        with_counts = numeric_subset.with_columns(non_null_count)
        coverage_rows = with_counts.filter(pl.col("non_nulls") >= 1)
        alpha_rows = with_counts.filter(pl.col("non_nulls") >= 2)
        n_complete = alpha_rows.height
        coverage_pct = (coverage_rows.height / total_rows * 100.0) if total_rows else 0.0
        alpha = None
        if index_name not in skip_alpha and n_complete >= 2 and len(items) >= 2:
            arr = alpha_rows.drop("non_nulls").to_numpy()
            alpha = _cronbach_alpha(arr)

        rows.append(
            {
                "index": index_name,
                "items": ", ".join(items),
                "n_items": len(items),
                "n_complete": n_complete,
                "coverage_pct": coverage_pct,
                "cronbach_alpha": alpha,
            }
        )

    if skipped:
        sample = ", ".join(name for name, _ in skipped[:5])
        logger.info(
            "Skipped {n} index(es) without full component coverage (e.g., {sample})",
            n=len(skipped),
            sample=sample,
        )

    return pl.DataFrame(rows)


@beartype
def run_descriptives(cfg: DescriptivesConfig) -> DescriptivesResult:
    logger.info("Reading CSV: {p}", p=str(cfg.input_csv))
    df = pl.read_csv(cfg.input_csv)

    column_types = load_column_types(cfg.column_types_path, include_is_index=True)
    buckets, missing_in_data, unknown_types, untyped_cols = build_type_buckets(
        df, column_types, cfg.categorical_overrides
    )

    if unknown_types:
        logger.warning(
            "Ignoring columns with unknown types: {cols}",
            cols=", ".join(unknown_types[:10]),
        )
    if missing_in_data:
        logger.info(
            "Columns listed in spec but missing from dataset: {cols}",
            cols=", ".join(missing_in_data[:15]),
        )
    if untyped_cols:
        logger.info(
            "Dataset columns without type assignments: {cols}",
            cols=", ".join(untyped_cols[:15]),
        )

    continuous = _continuous_summary(df, buckets["continuous"])
    ordinal_summary, ordinal_distribution = _ordinal_summary(df, buckets["ordinal"])
    binary_summary, binary_distribution = _binary_summary(df, buckets["binary"])
    categorical_distribution = _categorical_distribution(df, buckets["categorical"])

    index_groups = _load_index_groups(cfg.spec_path) if cfg.compute_alpha else {}
    reliability = (
        _compute_index_reliability(df, index_groups, COVERAGE_ONLY_INDICES)
        if cfg.compute_alpha and index_groups
        else pl.DataFrame({})
    )

    def _round_floats_2(d: pl.DataFrame) -> pl.DataFrame:
        return d

    continuous = _round_floats_2(continuous)
    ordinal_summary = _round_floats_2(ordinal_summary)
    ordinal_distribution = _round_floats_2(ordinal_distribution)
    binary_summary = _round_floats_2(binary_summary)
    binary_distribution = _round_floats_2(binary_distribution)
    categorical_distribution = _round_floats_2(categorical_distribution)
    reliability = _round_floats_2(reliability)

    ensure_outdir(cfg.outdir)
    raw_dir = cfg.outdir / "raw"
    ensure_outdir(raw_dir)

    outputs: list[tuple[str, pl.DataFrame]] = [
        ("continuous_summary.csv", continuous),
        ("ordinal_summary.csv", ordinal_summary),
        ("ordinal_distribution.csv", ordinal_distribution),
        ("binary_summary.csv", binary_summary),
        ("binary_distribution.csv", binary_distribution),
        ("categorical_distribution.csv", categorical_distribution),
    ]

    for filename, data in outputs:
        if data.height == 0:
            logger.info("Skipping empty output: {name}", name=filename)
            continue
        raw_path = raw_dir / filename
        data.write_csv(raw_path)
        logger.info("Wrote raw/{name} -> {p}", name=filename, p=str(raw_path))

    # Optional: also write per-section tables under views/by_section/ if section annotations exist
    try:
        if "section" in column_types.columns:
            var_map = column_types.select(
                pl.col("column").alias("variable"),
                pl.coalesce([pl.col("section"), pl.lit("Unspecified")]).alias("section"),
            )
            write_grouped_views(
                outputs,
                var_map,
                join_col="variable",
                group_col="section",
                base_dir=cfg.outdir / "views" / "by_section",
            )
            logger.info(
                "Wrote section tables under: {p}",
                p=str((cfg.outdir / "views" / "by_section").resolve()),
            )
    except Exception as e:
        logger.warning("Failed to write section tables: {e}", e=str(e))

    # Optional: write per-block tables matching inferential's output layout
    try:
        if "block" in column_types.columns:
            block_map = column_types.select(
                pl.col("column").alias("variable"),
                pl.coalesce([pl.col("block"), pl.lit("Unspecified")]).alias("block"),
            )
            write_grouped_views(
                outputs,
                block_map,
                join_col="variable",
                group_col="block",
                base_dir=cfg.outdir / "views" / "by_block",
            )
            logger.info(
                "Wrote block tables under: {p}",
                p=str((cfg.outdir / "views" / "by_block").resolve()),
            )
    except Exception as e:
        logger.warning("Failed to write block tables: {e}", e=str(e))

    if reliability.height > 0:
        rel_raw_path = raw_dir / "index_reliability.csv"
        reliability.write_csv(rel_raw_path)
        logger.info("Index reliability -> {p}", p=str(rel_raw_path))

        # Also write reliability by section when available (map index -> section)
        try:
            if "section" in column_types.columns:
                idx_map = column_types.select(
                    pl.col("column").alias("index"),
                    pl.coalesce([pl.col("section"), pl.lit("Unspecified")]).alias("section"),
                )
                write_grouped_views(
                    [("index_reliability.csv", reliability)],
                    idx_map,
                    join_col="index",
                    group_col="section",
                    base_dir=cfg.outdir / "views" / "by_section",
                )
        except Exception as e:
            logger.warning("Failed to write sectioned reliability: {e}", e=str(e))

        # Also write reliability by block when available (map index -> block)
        try:
            if "block" in column_types.columns:
                idx_map = column_types.select(
                    pl.col("column").alias("index"),
                    pl.coalesce([pl.col("block"), pl.lit("Unspecified")]).alias("block"),
                )
                write_grouped_views(
                    [("index_reliability.csv", reliability)],
                    idx_map,
                    join_col="index",
                    group_col="block",
                    base_dir=cfg.outdir / "views" / "by_block",
                )
        except Exception as e:
            logger.warning("Failed to write block reliability: {e}", e=str(e))
    elif cfg.compute_alpha:
        logger.info("No index reliability outputs generated (no qualifying groups)")

    return DescriptivesResult(
        continuous=continuous,
        ordinal_summary=ordinal_summary,
        ordinal_distribution=ordinal_distribution,
        binary_summary=binary_summary,
        binary_distribution=binary_distribution,
        categorical_distribution=categorical_distribution,
        reliability=reliability,
    )


@beartype
def _find_prefixed_boolean_columns(df: pl.DataFrame, prefixes: Sequence[str]) -> dict[str, list[str]]:
    found: dict[str, list[str]] = {}
    for pref in prefixes:
        cols = [
            c
            for c in df.columns
            if c.startswith(pref) and df.schema.get(c) in (pl.Boolean, pl.Int64, pl.Int32, pl.UInt8)
        ]
        if cols:
            found[pref] = cols
    return found


@beartype
def _summarize_boolean_groups(df: pl.DataFrame, groups: dict[str, list[str]]) -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    n_rows = df.height
    for pref, cols in groups.items():
        for c in cols:
            s = df.get_column(c)
            # Coerce to numeric 0/1; treat True/1 as 1
            non_null = s.drop_nulls()
            true_count = non_null.cast(pl.Int64, strict=False).sum() if non_null.len() > 0 else 0
            coverage = non_null.len() / n_rows * 100.0 if n_rows else 0.0
            pct_true = true_count / non_null.len() * 100.0 if non_null.len() else None
            rows.append(
                {
                    "group": pref.rstrip("_"),
                    "variable": c,
                    "true_count": int(true_count) if true_count is not None else None,
                    "pct_true": pct_true,
                    "coverage_pct": coverage,
                }
            )
    if not rows:
        return pl.DataFrame({})
    out = pl.DataFrame(rows)
    return out



def run_descriptives_workflow(
    *,
    input_path: Path = CSV_FILE_CLEAN,
    outdir: Path = DEFAULT_OUTDIR,
    column_types_path: Path = DEFAULT_COLUMN_TYPES_PATH,
    categorical_overrides: str = "",
    spec_path: Path = Path("preprocess/specs/5_clean_transformed_questions.json"),
    compute_alpha: bool = True,
    bool_groups: str = "covax_reason__,covax_no_child__",
    log_level: str | None = None,
) -> None:
    """Run descriptives workflow and write outputs."""

    effective_log_level = log_level or os.getenv("LOG_LEVEL", "INFO")
    logger.remove()
    logger.add(lambda msg: print(msg, end=""), level=effective_log_level)

    cfg = DescriptivesConfig(
        input_csv=input_path,
        outdir=outdir,
        column_types_path=column_types_path,
        spec_path=spec_path,
        compute_alpha=compute_alpha,
        categorical_overrides=parse_list_arg(categorical_overrides, dedupe=False),
    )
    run_descriptives(cfg)

    prefixes = parse_list_arg(bool_groups, dedupe=False)
    try:
        df = pl.read_csv(input_path)
        groups = _find_prefixed_boolean_columns(df, prefixes)
        if groups:
            logger.info("Summarizing boolean groups: {g}", g=", ".join(groups.keys()))
            summary = _summarize_boolean_groups(df, groups)
            if summary.height > 0:
                ensure_outdir(outdir)
                raw_bool_path = outdir / "raw" / "boolean_groups_summary.csv"
                ensure_outdir(raw_bool_path.parent)
                summary.write_csv(raw_bool_path)
                logger.info("Boolean groups summary -> {p}", p=str(raw_bool_path))
        else:
            logger.info("No boolean groups found for prefixes: {p}", p=", ".join(prefixes))
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Failed to compute boolean groups summary: {e}", e=str(exc))
