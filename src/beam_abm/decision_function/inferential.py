"""Inferential statistical helpers for exploratory analysis."""

from __future__ import annotations

import math
import numbers
import os
import warnings
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import TypeVar

import numpy as np
import polars as pl
from beartype import beartype
from scipy import stats
from scipy.stats import ConstantInputWarning, NearConstantInputWarning

from beam_abm.common.logging import get_logger

logger = get_logger(__name__)

__all__ = [
    "InferentialResult",
    "HighlightCriteria",
    "compute_spearman_correlations",
    "compute_categorical_associations",
    "compute_binary_numeric_tests",
    "compute_nominal_numeric_tests",
    "compute_inferential_suite",
    "compute_inferential_highlights",
    "split_inferential_by_section",
    "split_inferential_by_hypothesis",
    "to_canonical_raw_format",
    "generate_by_outcome_view",
    "generate_by_block_view",
    "generate_by_section_view",
    "generate_predictor_predictor_views",
]

_EFFECT_THRESHOLDS: dict[str, tuple[float, float, float]] = {
    "spearman": (0.1, 0.3, 0.5),
    "cramers_v": (0.05, 0.15, 0.3),
    "cohens_d": (0.2, 0.5, 0.8),
    "eta_squared": (0.01, 0.06, 0.14),
    "epsilon_squared": (0.01, 0.06, 0.14),
}

_EFFECT_LABEL_ORDER: dict[str, int] = {
    "negligible": 0,
    "small": 1,
    "medium": 2,
    "large": 3,
}

_SECTION_COLUMNS: dict[str, tuple[str, ...]] = {
    "spearman": ("var_x", "var_y"),
    "categorical": ("var_x", "var_y"),
    "binary_numeric": ("binary_var", "numeric_var"),
    "nominal_numeric": ("nominal_var", "numeric_var"),
}


@dataclass(slots=True)
class InferentialResult:
    """Container for inferential analysis tables."""

    spearman: pl.DataFrame
    categorical: pl.DataFrame
    binary_numeric: pl.DataFrame
    nominal_numeric: pl.DataFrame


@dataclass(slots=True)
class HighlightCriteria:
    """Configuration for filtering highlight-worthy associations."""

    alpha: float = 0.05
    min_effect_label: str = "medium"
    min_spearman_n: int = 50
    min_categorical_n: int = 50
    min_binary_group_n: int = 20
    min_nominal_group_n: int = 15

    def __post_init__(self) -> None:
        if not 0.0 <= float(self.alpha) <= 1.0:
            raise ValueError("alpha must be between 0 and 1")
        label = (self.min_effect_label or "").strip().lower()
        if label not in _EFFECT_LABEL_ORDER:
            allowed = ", ".join(_EFFECT_LABEL_ORDER.keys())
            raise ValueError(f"Unknown effect label '{self.min_effect_label}'. Expected one of: {allowed}")
        self.min_effect_label = label
        mins = [
            self.min_spearman_n,
            self.min_categorical_n,
            self.min_binary_group_n,
            self.min_nominal_group_n,
        ]
        if any(value < 0 for value in mins):
            raise ValueError("Minimum sample thresholds must be non-negative")


T = TypeVar("T")


def _run_tasks(
    tasks: Sequence[T],
    worker: Callable[[T], dict[str, object] | None],
    max_workers: int,
) -> list[dict[str, object]]:
    if not tasks:
        return []
    if max_workers <= 1 or len(tasks) == 1:
        return [result for task in tasks if (result := worker(task)) is not None]
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        return [result for result in executor.map(worker, tasks) if result is not None]


def _default_max_workers() -> int:
    """Return a conservative default for worker threads.

    Heuristic: use half of the available CPUs where available, otherwise 1.
    """
    try:
        cpus = os.cpu_count() or 1
    except (OSError, TypeError, ValueError):
        cpus = 1
    return max(1, cpus // 2)


def _stable_level_order(values: Sequence[object]) -> list[object]:
    """Sort heterogeneous values deterministically across runs."""

    return sorted(values, key=lambda v: (v.__class__.__name__, str(v)))


def _label_effect_size(value: float | None, metric: str) -> str | None:
    thresholds = _EFFECT_THRESHOLDS.get(metric)
    if value is None or not np.isfinite(value) or thresholds is None:
        return None
    magnitude = abs(float(value))
    small, medium, large = thresholds
    if magnitude < small:
        return "negligible"
    if magnitude < medium:
        return "small"
    if magnitude < large:
        return "medium"
    return "large"


@beartype
def _drop_missing_pairs(df: pl.DataFrame, col_x: str, col_y: str) -> pl.DataFrame:
    return df.select(col_x, col_y).drop_nulls()


@beartype
def compute_spearman_correlations(
    df: pl.DataFrame,
    columns: Sequence[str],
    *,
    skip: Callable[[str, str], bool] | None = None,
) -> pl.DataFrame:
    """Compute pairwise Spearman correlations with p-values and effect labels."""
    col_list = [col for col in dict.fromkeys(columns) if col in df.columns]
    if len(col_list) < 2:
        return pl.DataFrame({})

    rows: list[dict[str, object]] = []
    for i, col_x in enumerate(col_list[:-1]):
        for j in range(i + 1, len(col_list)):
            col_y = col_list[j]
            if skip is not None and skip(col_x, col_y):
                continue
            pair_df = _drop_missing_pairs(df, col_x, col_y)
            if pair_df.height < 3:
                continue
            x = pair_df.get_column(col_x).cast(pl.Float64, strict=False).to_numpy()
            y = pair_df.get_column(col_y).cast(pl.Float64, strict=False).to_numpy()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", category=ConstantInputWarning)
                warnings.simplefilter("always", category=NearConstantInputWarning)
                warnings.simplefilter("always", category=RuntimeWarning)
                result = stats.spearmanr(x, y, nan_policy="omit")
            for warn in caught:
                warn_category = warn.category.__name__ if hasattr(warn.category, "__name__") else str(warn.category)
                logger.warning(
                    "Spearman warning for {x} vs {y}: [{category}] {message}",
                    x=col_x,
                    y=col_y,
                    category=warn_category,
                    message=str(warn.message),
                )
            if isinstance(result, tuple):
                rho_val, p_val = result
            else:
                rho_attr = getattr(result, "correlation", getattr(result, "statistic", np.nan))
                p_attr = getattr(result, "pvalue", getattr(result, "p", np.nan))
                rho_val, p_val = rho_attr, p_attr
            rho_val = float(rho_val)
            if not np.isfinite(rho_val):
                continue
            p_val = float(p_val) if np.isfinite(p_val) else None
            effect_size = abs(rho_val)
            rows.append(
                {
                    "var_x": col_x,
                    "var_y": col_y,
                    "n": pair_df.height,
                    "spearman_rho": rho_val,
                    "p_value": p_val,
                    "effect_size": effect_size,
                    "effect_size_metric": "spearman",
                    "effect_size_label": _label_effect_size(effect_size, "spearman"),
                }
            )
    if not rows:
        return pl.DataFrame({})
    out = pl.DataFrame(rows)
    return out.sort(["var_x", "var_y"])


@beartype
def compute_categorical_associations(
    df: pl.DataFrame,
    columns: Sequence[str],
    *,
    max_workers: int | None = None,
    skip: Callable[[str, str], bool] | None = None,
) -> pl.DataFrame:
    """Compute chi-square and Cramer's V for categorical pairs."""
    col_list = [col for col in dict.fromkeys(columns) if col in df.columns]
    tasks = []
    for i in range(len(col_list)):
        for j in range(i + 1, len(col_list)):
            a, b = col_list[i], col_list[j]
            if skip is not None and skip(a, b):
                continue
            tasks.append((a, b))

    def worker(pair: tuple[str, str]) -> dict[str, object] | None:
        col_x, col_y = pair
        subset = _drop_missing_pairs(df, col_x, col_y)
        if subset.height < 3:
            return None
        counts = (
            subset.group_by(col_x, col_y)
            .len()
            .rename({"len": "count"})
            .pivot(index=col_x, columns=col_y, values="count")
            .fill_null(0)
        )
        if counts.height == 0:
            return None
        value_cols = [c for c in counts.columns if c != col_x]
        if not value_cols:
            return None
        table = np.asarray(counts.select(value_cols).to_numpy(), dtype=float)
        if table.size == 0:
            return None
        try:
            chi2, p_value, _, _ = stats.chi2_contingency(table)
        except ValueError as exc:
            logger.warning(
                "Chi-square failed for {x} vs {y}: {err}",
                x=col_x,
                y=col_y,
                err=str(exc),
            )
            return None
        n = float(table.sum())
        if n <= 0:
            return None
        min_dim = min(table.shape)
        if min_dim <= 1:
            return None
        phi2 = max(chi2 / n, 0.0)
        denom = min_dim - 1
        if denom <= 0:
            return None
        cramers_v = math.sqrt(phi2 / denom)
        if not np.isfinite(cramers_v):
            cramers_v = 0.0
        effect_size = abs(cramers_v)
        return {
            "var_x": col_x,
            "var_y": col_y,
            "n": subset.height,
            "chi2_statistic": float(chi2),
            "chi2_p_value": float(p_value),
            "cramers_v": float(cramers_v),
            "effect_size": effect_size,
            "effect_size_metric": "cramers_v",
            "effect_size_label": _label_effect_size(effect_size, "cramers_v"),
        }

    workers = max_workers if max_workers is not None else _default_max_workers()
    rows = _run_tasks(tasks, worker, workers)
    if not rows:
        return pl.DataFrame({})
    out = pl.DataFrame(rows)
    return out.sort(["var_x", "var_y"])


@beartype
def compute_binary_numeric_tests(
    df: pl.DataFrame,
    binary_cols: Sequence[str],
    numeric_cols: Sequence[str],
    *,
    max_workers: int | None = None,
) -> pl.DataFrame:
    """Conduct Mann–Whitney U and t-tests for binary versus numeric variables."""
    tasks: list[tuple[str, str, object, object]] = []
    for b_col in dict.fromkeys(binary_cols):
        if b_col not in df.columns:
            logger.warning("Missing binary column: {col}", col=b_col)
            continue
        series = df.get_column(b_col).drop_nulls()
        levels = _stable_level_order(series.unique().to_list())
        if len(levels) != 2:
            continue
        level_a, level_b = levels
        for n_col in dict.fromkeys(numeric_cols):
            if n_col not in df.columns:
                logger.warning("Missing numeric column: {col}", col=n_col)
                continue
            tasks.append((b_col, n_col, level_a, level_b))

    def _is_binary_01(level_a: object, level_b: object) -> bool:
        """Return True when the two category levels represent a 0/1-coded item."""

        levels = {level_a, level_b}

        # Bool is a subclass of int in Python, so check it first.
        if any(isinstance(v, bool) for v in levels):
            return {bool(level_a), bool(level_b)} == {False, True}

        if all(isinstance(v, numbers.Integral) and not isinstance(v, bool) for v in levels):
            return {int(level_a), int(level_b)} == {0, 1}

        if all(isinstance(v, numbers.Real) and not isinstance(v, bool) for v in levels):
            return {float(level_a), float(level_b)} == {0.0, 1.0}

        if all(isinstance(v, str) for v in levels):
            return {str(level_a), str(level_b)} == {"0", "1"}

        return False

    def worker(task: tuple[str, str, object, object]) -> dict[str, object] | None:
        b_col, n_col, level_a, level_b = task
        subset = df.select(b_col, n_col).drop_nulls()
        if subset.height < 4:
            return None
        group_a = subset.filter(pl.col(b_col) == level_a).get_column(n_col).cast(pl.Float64, strict=False).to_numpy()
        group_b = subset.filter(pl.col(b_col) == level_b).get_column(n_col).cast(pl.Float64, strict=False).to_numpy()
        len_a = len(group_a)
        len_b = len(group_b)
        if len_a < 2 or len_b < 2:
            return None
        total_n = len_a + len_b
        mann_whitney = stats.mannwhitneyu(group_a, group_b, alternative="two-sided")
        t_stat, t_p = stats.ttest_ind(group_a, group_b, equal_var=False)
        # Report the larger of the two Mann-Whitney U statistics so test shows separation
        try:
            u_val = float(mann_whitney.statistic)
            n1, n2 = float(len(group_a)), float(len(group_b))
            u = max(u_val, n1 * n2 - u_val)
        except (TypeError, ValueError):
            u = float(mann_whitney.statistic)
        var_a = np.var(group_a, ddof=1)
        var_b = np.var(group_b, ddof=1)
        denom = len(group_a) + len(group_b) - 2
        pooled_sd = None
        if denom > 0 and np.isfinite(var_a) and np.isfinite(var_b):
            pooled_sd = math.sqrt(((len(group_a) - 1) * var_a + (len(group_b) - 1) * var_b) / denom)
        d_signed = None
        if pooled_sd and pooled_sd > 0:
            d_raw = float((np.mean(group_a) - np.mean(group_b)) / pooled_sd)
            # For 0/1-coded items, enforce an intuitive direction:
            # d = mean(item=1) - mean(item=0)
            d_signed = float(-d_raw) if _is_binary_01(level_a, level_b) else float(d_raw)
        effect_size = abs(d_signed) if d_signed is not None else None
        t_stat_out = float(t_stat) if np.isfinite(t_stat) else None
        t_p_out = float(t_p) if np.isfinite(t_p) else None
        return {
            "binary_var": b_col,
            "numeric_var": n_col,
            "n_total": total_n,
            "group_a_n": len_a,
            "group_b_n": len_b,
            "mann_whitney_u": float(u),
            "mann_whitney_p": float(mann_whitney.pvalue),
            "t_statistic": t_stat_out,
            "t_p_value": t_p_out,
            "cohens_d": d_signed,
            "cohens_d_abs": effect_size,
            "cohens_d_signed": d_signed,
            "effect_size": effect_size,
            "effect_size_metric": "cohens_d",
            "effect_size_label": _label_effect_size(effect_size, "cohens_d"),
        }

    workers = max_workers if max_workers is not None else _default_max_workers()
    rows = _run_tasks(tasks, worker, workers)
    if not rows:
        return pl.DataFrame({})
    out = pl.DataFrame(rows)
    return out.sort(["binary_var", "numeric_var"])


@beartype
def compute_nominal_numeric_tests(
    df: pl.DataFrame,
    nominal_cols: Sequence[str],
    numeric_cols: Sequence[str],
    *,
    max_workers: int | None = None,
) -> pl.DataFrame:
    """Run Kruskal–Wallis and ANOVA tests for nominal versus numeric variables."""
    tasks: list[tuple[str, str, list[object]]] = []
    for n_col in dict.fromkeys(nominal_cols):
        if n_col not in df.columns:
            logger.warning("Missing nominal column: {col}", col=n_col)
            continue
        cats = _stable_level_order(df.get_column(n_col).drop_nulls().unique().to_list())
        if len(cats) < 3:
            continue
        for num_col in dict.fromkeys(numeric_cols):
            if num_col not in df.columns:
                logger.warning("Missing numeric column: {col}", col=num_col)
                continue
            tasks.append((n_col, num_col, cats))

    def worker(task: tuple[str, str, list[object]]) -> dict[str, object] | None:
        n_col, num_col, cats = task
        subset = df.select(n_col, num_col).drop_nulls()
        if subset.height < 6:
            return None
        groups = [
            subset.filter(pl.col(n_col) == level).get_column(num_col).cast(pl.Float64, strict=False).to_numpy()
            for level in cats
        ]
        group_counts = [len(g) for g in groups]
        if any(count < 2 for count in group_counts):
            return None
        total_n = sum(group_counts)
        kruskal = stats.kruskal(*groups)
        anova = stats.f_oneway(*groups)
        values = subset.get_column(num_col).cast(pl.Float64, strict=False).to_numpy()
        if len(values) == 0:
            return None
        overall_mean = float(np.mean(values))
        ss_between = float(sum(len(g) * (float(np.mean(g)) - overall_mean) ** 2 for g in groups))
        ss_total = float(np.sum((values - overall_mean) ** 2))
        eta_sq = float(ss_between / ss_total) if ss_total > 0 else None
        if eta_sq is not None and not np.isfinite(eta_sq):
            eta_sq = None
        df_num = len(cats) - 1
        denom = total_n - len(cats)
        epsilon_sq = None
        if denom > 0 and np.isfinite(kruskal.statistic):
            epsilon_sq = float(max((kruskal.statistic - df_num) / denom, 0.0))
            if not np.isfinite(epsilon_sq):
                epsilon_sq = None
        effect_value = eta_sq if eta_sq is not None else epsilon_sq
        effect_metric = "eta_squared" if eta_sq is not None else ("epsilon_squared" if epsilon_sq is not None else None)
        effect_size = abs(effect_value) if effect_value is not None else None
        kruskal_h = float(kruskal.statistic) if np.isfinite(kruskal.statistic) else None
        kruskal_p = float(kruskal.pvalue) if np.isfinite(kruskal.pvalue) else None
        anova_f = float(anova.statistic) if np.isfinite(anova.statistic) else None
        anova_p = float(anova.pvalue) if np.isfinite(anova.pvalue) else None
        return {
            "nominal_var": n_col,
            "numeric_var": num_col,
            "n_total": total_n,
            "min_group_n": min(group_counts) if group_counts else None,
            "num_groups": len(group_counts),
            "kruskal_h": kruskal_h,
            "kruskal_p": kruskal_p,
            "anova_f": anova_f,
            "anova_p": anova_p,
            "eta_squared": eta_sq,
            "epsilon_squared": epsilon_sq,
            "effect_size": effect_size,
            "effect_size_metric": effect_metric,
            "effect_size_label": _label_effect_size(effect_size, effect_metric or "epsilon_squared"),
        }

    workers = max_workers if max_workers is not None else _default_max_workers()
    rows = _run_tasks(tasks, worker, workers)
    if not rows:
        return pl.DataFrame({})
    out = pl.DataFrame(rows)
    return out.sort(["nominal_var", "numeric_var"])


@beartype
def split_inferential_by_section(
    result: InferentialResult,
    section_map: pl.DataFrame,
    *,
    unspecified_label: str = "Unspecified",
) -> dict[str, dict[str, pl.DataFrame]]:
    """Slice inferential tables by column sections.

    Parameters
    ----------
    result:
        Collection of inferential output tables.
    section_map:
        DataFrame mapping variables to sections. Must include columns ``variable``
        and ``section``.
    unspecified_label:
        Label assigned when a variable lacks a section name.

    Returns
    -------
    dict[str, dict[str, pl.DataFrame]]
        Mapping of section name to a mapping of table identifiers and their
        filtered DataFrames. Identifiers match the attributes on
        :class:`InferentialResult` (``spearman``, ``categorical``,
        ``binary_numeric``, ``nominal_numeric``).

    Raises
    ------
    ValueError
        If ``section_map`` is missing the required columns.
    """

    required_cols = {"variable", "section"}
    missing_cols = required_cols - set(section_map.columns)
    if missing_cols:
        cols = ", ".join(sorted(missing_cols))
        raise ValueError(f"section_map missing required columns: {cols}")

    normalised = (
        section_map.select(
            pl.col("variable").cast(pl.String),
            pl.col("section").cast(pl.String, strict=False),
        )
        .drop_nulls(subset=["variable"])
        .unique(subset=["variable"], keep="first")
    )

    mapping = {
        row["variable"]: row["section"] if row["section"] else unspecified_label
        for row in normalised.iter_rows(named=True)
    }

    tables: dict[str, pl.DataFrame] = {
        "spearman": result.spearman,
        "categorical": result.categorical,
        "binary_numeric": result.binary_numeric,
        "nominal_numeric": result.nominal_numeric,
    }

    section_tables: dict[str, dict[str, pl.DataFrame]] = {}

    def _lookup_section(value: object) -> str:
        if value is None:
            return unspecified_label
        key = str(value)
        return mapping.get(key, unspecified_label)

    for table_name, table in tables.items():
        variable_cols = _SECTION_COLUMNS.get(table_name, ())
        if table.height == 0 or not variable_cols:
            continue
        present = [col for col in variable_cols if col in table.columns]
        if not present:
            continue
        annotation_exprs = []
        annotation_cols: list[str] = []
        for col in present:
            section_col = f"__section_{col}"
            annotation_cols.append(section_col)
            expr = pl.col(col).map_elements(_lookup_section, return_dtype=pl.String).alias(section_col)
            annotation_exprs.append(expr)

        annotated = table.with_columns(annotation_exprs)
        section_names: set[str] = set()
        for sec_col in annotation_cols:
            sec_values = (
                annotated.select(pl.col(sec_col).fill_null(unspecified_label)).to_series(0).drop_nulls().to_list()
            )
            section_names.update(str(val) for val in sec_values)

        for section in sorted(section_names):
            mask: pl.Expr | None = None
            for sec_col in annotation_cols:
                expr = pl.col(sec_col) == section
                mask = expr if mask is None else mask | expr
            if mask is None:
                continue
            subset = annotated.filter(mask)
            if subset.height == 0:
                continue
            cleaned = subset.drop(annotation_cols)
            section_tables.setdefault(section, {})[table_name] = cleaned

    return section_tables


@beartype
def split_inferential_by_hypothesis(
    result: InferentialResult,
    hypothesis_pairs: pl.DataFrame,
) -> dict[str, dict[str, pl.DataFrame]]:
    """Slice inferential tables according to predefined hypothesis pairings.

    Parameters
    ----------
    result:
        Collection of inferential output tables.
    hypothesis_pairs:
        DataFrame mapping hypothesis identifiers to variable pairs. Must include
        columns ``H_ID``, ``var_x`` and ``var_y``.

    Returns
    -------
    dict[str, dict[str, pl.DataFrame]]
        Mapping of hypothesis identifiers to dictionaries of inferential tables
        filtered to the requested variable pairs. Keys match the attributes on
        :class:`InferentialResult`.

    Raises
    ------
    ValueError
        If ``hypothesis_pairs`` misses any required columns.
    """

    required_cols = {"H_ID", "var_x", "var_y"}
    missing_cols = required_cols - set(hypothesis_pairs.columns)
    if missing_cols:
        cols = ", ".join(sorted(missing_cols))
        raise ValueError(f"hypothesis_pairs missing required columns: {cols}")

    normalised = (
        hypothesis_pairs.select(
            pl.col("H_ID").cast(pl.String),
            pl.col("var_x").cast(pl.String, strict=False),
            pl.col("var_y").cast(pl.String, strict=False),
        )
        .drop_nulls(subset=["H_ID", "var_x", "var_y"])
        .unique(subset=["H_ID", "var_x", "var_y"])
    )
    if normalised.height == 0:
        return {}

    pair_map: dict[frozenset[str], set[str]] = {}
    for row in normalised.iter_rows(named=True):
        key = frozenset((row["var_x"], row["var_y"]))
        if len(key) != 2:
            continue
        pair_map.setdefault(key, set()).add(row["H_ID"])

    if not pair_map:
        return {}

    tables: dict[str, pl.DataFrame] = {
        "spearman": result.spearman,
        "categorical": result.categorical,
        "binary_numeric": result.binary_numeric,
        "nominal_numeric": result.nominal_numeric,
    }

    grouped_rows: dict[str, dict[str, list[dict[str, object]]]] = {}

    for table_name, table in tables.items():
        variable_cols = _SECTION_COLUMNS.get(table_name, ())
        if table.height == 0 or not variable_cols:
            continue
        if len(variable_cols) != 2:
            continue
        if any(col not in table.columns for col in variable_cols):
            continue
        col_a, col_b = variable_cols
        for row in table.iter_rows(named=True):
            a_val = row.get(col_a)
            b_val = row.get(col_b)
            if a_val is None or b_val is None:
                continue
            key = frozenset((str(a_val), str(b_val)))
            hypothesis_ids = pair_map.get(key)
            if not hypothesis_ids:
                continue
            for hypothesis_id in hypothesis_ids:
                grouped_rows.setdefault(hypothesis_id, {}).setdefault(table_name, []).append(row)

    grouped_tables: dict[str, dict[str, pl.DataFrame]] = {}
    for hypothesis_id, table_rows in grouped_rows.items():
        for table_name, rows in table_rows.items():
            if not rows:
                continue
            grouped_tables.setdefault(hypothesis_id, {})[table_name] = pl.DataFrame(rows)

    return grouped_tables


@beartype
def compute_inferential_suite(
    df: pl.DataFrame,
    numeric_columns: Sequence[str],
    categorical_columns: Sequence[str],
    binary_columns: Sequence[str],
    nominal_columns: Sequence[str],
    *,
    max_workers: int = 1,
    spearman_skip: Callable[[str, str], bool] | None = None,
    categorical_skip: Callable[[str, str], bool] | None = None,
) -> InferentialResult:
    numeric_list = [col for col in dict.fromkeys(numeric_columns) if col in df.columns]
    categorical_list = [col for col in dict.fromkeys(categorical_columns) if col in df.columns]
    binary_list = [col for col in dict.fromkeys(binary_columns) if col in df.columns]
    nominal_list = [col for col in dict.fromkeys(nominal_columns) if col in df.columns]

    spearman = compute_spearman_correlations(df, numeric_list, skip=spearman_skip)
    categorical = compute_categorical_associations(df, categorical_list, max_workers=max_workers, skip=categorical_skip)
    binary_numeric = compute_binary_numeric_tests(df, binary_list, numeric_list, max_workers=max_workers)
    nominal_numeric = compute_nominal_numeric_tests(df, nominal_list, numeric_list, max_workers=max_workers)

    return InferentialResult(
        spearman=spearman,
        categorical=categorical,
        binary_numeric=binary_numeric,
        nominal_numeric=nominal_numeric,
    )


def _benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    n = p_values.size
    if n == 0:
        return np.array([], dtype=float)
    order = np.argsort(p_values)
    sorted_p = p_values[order]
    adjusted = np.empty(n, dtype=float)
    cumulative = 1.0
    for idx in range(n - 1, -1, -1):
        rank = idx + 1
        value = sorted_p[idx] * n / rank
        cumulative = min(cumulative, value)
        adjusted[idx] = min(cumulative, 1.0)
    result = np.empty(n, dtype=float)
    result[order] = adjusted
    return result


def _add_q_values(df: pl.DataFrame, p_columns: Sequence[str], *, q_col: str = "q_value") -> pl.DataFrame:
    if df.height == 0:
        return df
    p_vals = np.full(df.height, np.nan, dtype=float)
    for column in p_columns:
        if column not in df.columns:
            continue
        series = df.get_column(column).cast(pl.Float64, strict=False).to_numpy()
        if series.size != p_vals.size:
            continue
        mask = np.isnan(p_vals) & np.isfinite(series)
        if not mask.any():
            continue
        p_vals[mask] = np.clip(series[mask], 0.0, 1.0)
    valid_mask = np.isfinite(p_vals)
    q_vals = np.full_like(p_vals, np.nan, dtype=float)
    if valid_mask.any():
        q_vals[valid_mask] = _benjamini_hochberg(p_vals[valid_mask])
    series = pl.Series(name=q_col, values=q_vals.tolist())
    return df.with_columns(series)


def _highlight_table(
    df: pl.DataFrame,
    *,
    p_columns: Sequence[str],
    criteria: HighlightCriteria,
    min_rank: int,
    extra_filter: pl.Expr | None = None,
) -> pl.DataFrame:
    if df.height == 0:
        return df
    table = _add_q_values(df, p_columns)
    if "q_value" not in table.columns:
        return pl.DataFrame({})
    label_expr = (
        pl.col("effect_size_label")
        .cast(pl.Utf8, strict=False)
        .str.to_lowercase()
        .replace(_EFFECT_LABEL_ORDER, default=-1)
    )
    filter_expr = (
        pl.col("effect_size").is_not_null()
        & pl.col("q_value").is_not_null()
        & (pl.col("q_value") <= criteria.alpha)
        & label_expr.ge(min_rank)
    )
    if extra_filter is not None:
        filter_expr = filter_expr & extra_filter
    filtered = table.filter(filter_expr)
    if filtered.height == 0:
        return filtered
    filtered = filtered.sort(
        by=[pl.col("effect_size").abs(), pl.col("q_value")],
        descending=[True, False],
        nulls_last=True,
    )
    return filtered


def compute_inferential_highlights(result: InferentialResult, criteria: HighlightCriteria) -> InferentialResult:
    """Return filtered tables containing only the strongest associations."""

    min_rank = _EFFECT_LABEL_ORDER[criteria.min_effect_label]

    spearman = _highlight_table(
        result.spearman,
        p_columns=("p_value",),
        criteria=criteria,
        min_rank=min_rank,
        extra_filter=pl.col("n") >= criteria.min_spearman_n,
    )

    categorical = _highlight_table(
        result.categorical,
        p_columns=("chi2_p_value",),
        criteria=criteria,
        min_rank=min_rank,
        extra_filter=pl.col("n") >= criteria.min_categorical_n,
    )

    binary_numeric = _highlight_table(
        result.binary_numeric,
        p_columns=("mann_whitney_p",),
        criteria=criteria,
        min_rank=min_rank,
        extra_filter=(
            (pl.col("group_a_n") >= criteria.min_binary_group_n) & (pl.col("group_b_n") >= criteria.min_binary_group_n)
        ),
    )

    nominal_numeric = _highlight_table(
        result.nominal_numeric,
        p_columns=("kruskal_p",),
        criteria=criteria,
        min_rank=min_rank,
        extra_filter=pl.col("min_group_n") >= criteria.min_nominal_group_n,
    )

    return InferentialResult(
        spearman=spearman,
        categorical=categorical,
        binary_numeric=binary_numeric,
        nominal_numeric=nominal_numeric,
    )


@beartype
def to_canonical_raw_format(
    result: InferentialResult,
    column_metadata: pl.DataFrame,
) -> pl.DataFrame:
    """Convert inferential results to a single canonical raw format.

    Parameters
    ----------
    result:
        Collection of inferential output tables.
    column_metadata:
        DataFrame with columns: column, section, block (at minimum).

    Returns
    -------
    pl.DataFrame
        Unified table with columns: var_x, var_y, section_x, section_y,
        block_x, block_y, test_family, effect_size, effect_metric,
        effect_label, p_value, q_value, n.
    """

    # Ensure metadata has the required columns
    meta_cols = set(column_metadata.columns)
    if not {"column", "section", "block"}.issubset(meta_cols):
        raise ValueError("column_metadata must have 'column', 'section', 'block' columns")

    meta = column_metadata.select(
        pl.col("column").alias("variable"),
        pl.col("section"),
        pl.col("block"),
    )

    all_rows: list[pl.DataFrame] = []

    # Process Spearman correlations
    if result.spearman.height > 0:
        df = result.spearman
        df = _add_q_values(df, ("p_value",))

        spearman_df = df.select(
            [
                pl.col("var_x"),
                pl.col("var_y"),
                pl.lit("numeric_numeric").alias("test_family"),
                pl.col("effect_size"),
                pl.col("effect_size_metric").alias("effect_metric"),
                pl.col("effect_size_label").alias("effect_label"),
                pl.col("p_value"),
                pl.col("q_value") if "q_value" in df.columns else pl.lit(None).alias("q_value"),
                pl.col("n"),
            ]
        )

        # Add metadata
        spearman_df = (
            spearman_df.join(meta, left_on="var_x", right_on="variable", how="left")
            .rename({"section": "section_x", "block": "block_x"})
            .join(meta, left_on="var_y", right_on="variable", how="left")
            .rename({"section": "section_y", "block": "block_y"})
        )

        all_rows.append(spearman_df)

    # Process categorical associations
    if result.categorical.height > 0:
        df = result.categorical
        df = _add_q_values(df, ("chi2_p_value",))

        cat_df = df.select(
            [
                pl.col("var_x"),
                pl.col("var_y"),
                pl.lit("categorical_categorical").alias("test_family"),
                pl.col("effect_size"),
                pl.col("effect_size_metric").alias("effect_metric"),
                pl.col("effect_size_label").alias("effect_label"),
                pl.col("chi2_p_value").alias("p_value"),
                pl.col("q_value") if "q_value" in df.columns else pl.lit(None).alias("q_value"),
                pl.col("n"),
            ]
        )

        cat_df = (
            cat_df.join(meta, left_on="var_x", right_on="variable", how="left")
            .rename({"section": "section_x", "block": "block_x"})
            .join(meta, left_on="var_y", right_on="variable", how="left")
            .rename({"section": "section_y", "block": "block_y"})
        )

        all_rows.append(cat_df)

    # Process binary-numeric tests
    if result.binary_numeric.height > 0:
        df = result.binary_numeric
        df = _add_q_values(df, ("mann_whitney_p",))

        bin_df = df.select(
            [
                pl.col("binary_var").alias("var_x"),
                pl.col("numeric_var").alias("var_y"),
                pl.lit("binary_numeric").alias("test_family"),
                pl.col("effect_size"),
                pl.col("effect_size_metric").alias("effect_metric"),
                pl.col("effect_size_label").alias("effect_label"),
                pl.col("mann_whitney_p").alias("p_value"),
                pl.col("q_value") if "q_value" in df.columns else pl.lit(None).alias("q_value"),
                pl.col("n_total").alias("n"),
            ]
        )

        bin_df = (
            bin_df.join(meta, left_on="var_x", right_on="variable", how="left")
            .rename({"section": "section_x", "block": "block_x"})
            .join(meta, left_on="var_y", right_on="variable", how="left")
            .rename({"section": "section_y", "block": "block_y"})
        )

        all_rows.append(bin_df)

    # Process nominal-numeric tests
    if result.nominal_numeric.height > 0:
        df = result.nominal_numeric
        df = _add_q_values(df, ("kruskal_p",))

        nom_df = df.select(
            [
                pl.col("nominal_var").alias("var_x"),
                pl.col("numeric_var").alias("var_y"),
                pl.lit("nominal_numeric").alias("test_family"),
                pl.col("effect_size"),
                pl.col("effect_size_metric").alias("effect_metric"),
                pl.col("effect_size_label").alias("effect_label"),
                pl.col("kruskal_p").alias("p_value"),
                pl.col("q_value") if "q_value" in df.columns else pl.lit(None).alias("q_value"),
                pl.col("n_total").alias("n"),
            ]
        )

        nom_df = (
            nom_df.join(meta, left_on="var_x", right_on="variable", how="left")
            .rename({"section": "section_x", "block": "block_x"})
            .join(meta, left_on="var_y", right_on="variable", how="left")
            .rename({"section": "section_y", "block": "block_y"})
        )

        all_rows.append(nom_df)

    if not all_rows:
        return pl.DataFrame({})

    # Concatenate all and ensure consistent schema
    combined = pl.concat(all_rows, how="diagonal")

    # Reorder columns
    column_order = [
        "var_x",
        "var_y",
        "section_x",
        "section_y",
        "block_x",
        "block_y",
        "test_family",
        "effect_size",
        "effect_metric",
        "effect_label",
        "p_value",
        "q_value",
        "n",
    ]

    return combined.select([col for col in column_order if col in combined.columns])


def _top_k_per_group(
    df: pl.DataFrame,
    *,
    group_col: str,
    k: int,
) -> pl.DataFrame:
    """Select Top-K rows per group deterministically.

    Ordering: higher |effect_size| first, then lower q_value (fallback p_value).
    """

    if df.height == 0:
        return df
    if k <= 0:
        return pl.DataFrame({})

    working = df.with_columns(
        pl.col("effect_size").abs().alias("_abs_effect"),
        pl.when(pl.col("q_value").is_not_null()).then(pl.col("q_value")).otherwise(pl.col("p_value")).alias("_sig"),
    )
    working = working.sort(
        by=[group_col, "_abs_effect", "_sig"],
        descending=[False, True, False],
        nulls_last=True,
    )
    # Polars' `cum_count` requires at least one column name. Here we want a
    # deterministic 0-based rank within each group after sorting.
    ranked = working.with_columns(pl.int_range(0, pl.len()).over(group_col).alias("_rank"))
    ranked = ranked.filter(pl.col("_rank") < k)
    return ranked.drop(["_abs_effect", "_sig", "_rank"])


def _medium_or_top_k_per_group(
    df: pl.DataFrame,
    *,
    group_col: str,
    k: int,
    min_effect_label: str = "medium",
) -> pl.DataFrame:
    """Select rows per group keeping all >= min_effect_label, else fall back to Top-K.

    For each group:
    - If there are any rows whose effect label rank is >= min_effect_label,
      return *only* those rows (even if fewer than K).
    - Otherwise, return the Top-K rows by |effect_size| then q_value/p_value.

    This matches the "show all medium+ relationships, otherwise show the top K"
    rule used for inferential view exports.
    """

    if df.height == 0:
        return df
    if k <= 0:
        return pl.DataFrame({})
    if group_col not in df.columns:
        return _top_k_per_group(df, group_col=group_col, k=k)
    if "effect_label" not in df.columns:
        return _top_k_per_group(df, group_col=group_col, k=k)

    label = (min_effect_label or "").strip().lower()
    if label not in _EFFECT_LABEL_ORDER:
        allowed = ", ".join(_EFFECT_LABEL_ORDER.keys())
        raise ValueError(f"Unknown effect label '{min_effect_label}'. Expected one of: {allowed}")
    min_rank = _EFFECT_LABEL_ORDER[label]

    prepared = df.with_columns(
        pl.col("effect_label")
        .cast(pl.Utf8)
        .str.to_lowercase()
        .replace(_EFFECT_LABEL_ORDER, default=-1)
        .alias("_effect_rank")
    )

    groups = prepared.select(group_col).unique().to_series().to_list()
    pieces: list[pl.DataFrame] = []
    for group_val in groups:
        subset = prepared.filter(pl.col(group_col) == group_val)
        if subset.height == 0:
            continue

        strong = subset.filter(pl.col("_effect_rank") >= min_rank).drop(["_effect_rank"])
        topk = _top_k_per_group(subset.drop(["_effect_rank"]), group_col=group_col, k=k)

        keep = strong if strong.height > 0 else topk

        # Sort deterministically for readability.
        keep = (
            keep.with_columns(
                pl.col("effect_size").abs().alias("_abs_effect"),
                pl.when(pl.col("q_value").is_not_null())
                .then(pl.col("q_value"))
                .otherwise(pl.col("p_value"))
                .alias("_sig"),
            )
            .sort(
                by=["_abs_effect", "_sig"],
                descending=[True, False],
                nulls_last=True,
            )
            .drop(["_abs_effect", "_sig"])
        )

        pieces.append(keep)

    if not pieces:
        return pl.DataFrame({})
    return pl.concat(pieces, how="vertical")


def _collapse_equal_columns(
    df: pl.DataFrame,
    *,
    left: str,
    right: str,
    merged: str,
) -> pl.DataFrame:
    if df.height == 0 or left not in df.columns or right not in df.columns:
        return df

    same = df.select((pl.col(left) == pl.col(right)).all().alias("same")).to_series(0).item()
    if not same:
        return df

    return df.with_columns(pl.col(left).alias(merged)).drop([left, right])


@beartype
def generate_by_outcome_view(
    raw: pl.DataFrame,
    outcome_vars: Sequence[str],
    highlights: pl.DataFrame | None = None,
    top_k: int = 10,
) -> dict[str, pl.DataFrame]:
    """Generate per-outcome views showing top predictors.

    Parameters
    ----------
    raw:
        Canonical raw format DataFrame.
    outcome_vars:
        List of outcome variable names.
    highlights:
        Optional pre-filtered highlights DataFrame. If provided and contains
        sufficient rows for an outcome, use those; otherwise fall back to top_k.
    top_k:
        Number of top effects to include per outcome.

    Returns
    -------
    dict[str, pl.DataFrame]
        Mapping from outcome variable name to top effects DataFrame.
    """

    views: dict[str, pl.DataFrame] = {}

    for outcome in outcome_vars:
        base = raw.filter(pl.col("var_y") == outcome)
        if base.height == 0:
            continue

        # View rule: keep all medium+ relationships; if fewer than Top-K,
        # pad with strongest remaining relationships to reach Top-K.
        # `highlights` is intentionally not used for selection here.
        views[outcome] = _medium_or_top_k_per_group(base, group_col="var_y", k=top_k, min_effect_label="medium")

    return views


@beartype
def generate_by_block_view(
    raw: pl.DataFrame,
    block_name: str,
    outcome_block: str = "outcome",
    top_k_per_outcome: int = 10,
) -> pl.DataFrame:
    """Generate view of block effects on outcomes.

    Parameters
    ----------
    raw:
        Canonical raw format DataFrame.
    block_name:
        The block to filter on (e.g., 'structural', 'psychological').
    outcome_block:
        The block label for outcomes (default: 'outcome').
    top_k_per_outcome:
        Number of top effects to keep per outcome variable.

    Returns
    -------
    pl.DataFrame
        Top effects from block_name to outcomes.
    """

    # Filter: block_x == block_name AND block_y == outcome_block
    filtered = raw.filter((pl.col("block_x") == block_name) & (pl.col("block_y") == outcome_block))

    if filtered.height == 0:
        return pl.DataFrame({})

    return _medium_or_top_k_per_group(
        filtered,
        group_col="var_y",
        k=top_k_per_outcome,
        min_effect_label="medium",
    )


@beartype
def generate_by_section_view(
    raw: pl.DataFrame,
    section_name: str,
    outcome_block: str = "outcome",
    top_k_per_outcome: int = 10,
) -> pl.DataFrame:
    """Generate view of section effects on outcomes.

    Parameters
    ----------
    raw:
        Canonical raw format DataFrame.
    section_name:
        The section to filter on.
    outcome_block:
        The block label for outcomes (default: 'outcome').
    top_k_per_outcome:
        Number of top effects to keep per outcome variable.

    Returns
    -------
    pl.DataFrame
        Top effects from section_name to outcomes.
    """

    # Filter: section_x == section_name AND block_y == outcome_block
    filtered = raw.filter((pl.col("section_x") == section_name) & (pl.col("block_y") == outcome_block))

    if filtered.height == 0:
        return pl.DataFrame({})

    return _medium_or_top_k_per_group(
        filtered,
        group_col="var_y",
        k=top_k_per_outcome,
        min_effect_label="medium",
    )


@beartype
def generate_predictor_predictor_views(
    raw: pl.DataFrame,
    max_within_block: int = 50,
    max_cross_block: int = 50,
) -> dict[str, pl.DataFrame]:
    """Generate curated predictor-predictor association views.

    Parameters
    ----------
    raw:
        Canonical raw format DataFrame.
    max_within_block:
        Maximum number of within-block associations to include.
    max_cross_block:
        Maximum number of cross-block associations to include.

    Returns
    -------
    dict[str, pl.DataFrame]
        Dictionary with keys 'within_block_redundancy' and 'cross_block_links'.
    """

    # Exclude outcome-* rows and keep only symmetric association families.
    predictor_only = raw.filter(
        (pl.col("block_x") != "outcome")
        & (pl.col("block_y") != "outcome")
        & pl.col("test_family").is_in(pl.lit(["numeric_numeric", "categorical_categorical"]))
    )

    if predictor_only.height == 0:
        return {
            "within_block_redundancy": pl.DataFrame({}),
            "cross_block_links": pl.DataFrame({}),
        }

    # Within-block: same block on both sides
    within = predictor_only.filter(pl.col("block_x") == pl.col("block_y"))
    within = _medium_or_top_k_per_group(
        within.with_columns(pl.lit("all").alias("_group")),
        group_col="_group",
        k=max_within_block,
        min_effect_label="medium",
    ).drop("_group")
    within = _collapse_equal_columns(within, left="block_x", right="block_y", merged="block")

    # Cross-block: different blocks
    cross = predictor_only.filter(pl.col("block_x") != pl.col("block_y"))
    cross = _medium_or_top_k_per_group(
        cross.with_columns(pl.lit("all").alias("_group")),
        group_col="_group",
        k=max_cross_block,
        min_effect_label="medium",
    ).drop("_group")

    return {
        "within_block_redundancy": within,
        "cross_block_links": cross,
    }
