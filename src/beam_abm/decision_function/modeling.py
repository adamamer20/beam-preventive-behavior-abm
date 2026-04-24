"""Reusable modeling utilities for the structural backbone analyses.

Implements common model specifications (linear, logistic, ordinal logit),
standardisation helpers, and a small result container so CLI scripts can
focus on orchestration and I/O.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
import patsy
import statsmodels.api as sm
import statsmodels.formula.api as smf
from beartype import beartype
from statsmodels.miscmodels.ordinal_model import OrderedModel

from beam_abm.common.logging import get_logger

logger = get_logger(__name__)

ModelType = Literal["linear", "logit", "ordinal"]
SampleStrategy = Literal["core", "conditional"]


@dataclass(slots=True)
class ModelBlock:
    """Logical block of predictors (e.g., demographics, trust)."""

    name: str
    predictors: tuple[str, ...]
    sample_strategy: SampleStrategy = "core"


@dataclass(slots=True)
class ModelSpec:
    """Full definition of a single model."""

    name: str
    outcome: str
    model_type: ModelType
    blocks: tuple[ModelBlock, ...]
    interactions: tuple[tuple[str, ...], ...] = ()
    add_country: bool = True
    description: str | None = None
    group: str | None = None

    def predictor_list(self) -> list[str]:
        items: list[str] = []
        for block in self.blocks:
            items.extend(block.predictors)
        for interaction in self.interactions:
            items.extend(interaction)
        if self.add_country:
            items.append("country")
        # Preserve order but drop duplicates
        seen: set[str] = set()
        deduped: list[str] = []
        for item in items:
            if item not in seen:
                seen.add(item)
                deduped.append(item)
        return deduped


@dataclass(slots=True)
class ModelResult:
    """Fitted model summary."""

    spec: ModelSpec
    formula: str
    coefficients: pd.DataFrame
    nobs: int
    pseudo_r2: float | None


_CATEGORICAL_TYPES = {"categorical", "binary"}
_NUMERIC_TYPES = {"continuous", "ordinal"}


def _is_numeric_col(name: str, column_types: Mapping[str, str]) -> bool:
    col_type = column_types.get(name, "")
    return col_type.lower() in _NUMERIC_TYPES


def _term_for_predictor(
    name: str,
    column_types: Mapping[str, str],
    df: pd.DataFrame,
    reference_levels: Mapping[str, object] | None,
) -> str:
    """Return a formula-ready term for the predictor."""
    if name.startswith("C(") and name.endswith(")"):
        return name
    col_type = column_types.get(name, "").lower()
    if col_type == "binary":
        # Keep binary predictors as numeric 0/1 to avoid redundant dummy columns
        # when the formula excludes an intercept (ordinal models use 0 + ...).
        return name
    if col_type in _CATEGORICAL_TYPES:
        if reference_levels is not None:
            ref = reference_levels.get(name)
            if ref is not None and name in df.columns:
                values = set(df[name].dropna().unique())
                if ref in values:
                    ref_repr = f"'{ref}'" if isinstance(ref, str) else repr(ref)
                    return f"C({name}, Treatment(reference={ref_repr}))"
                logger.debug(f"Reference {ref} for {name} not present in data; using default coding")
        return f"C({name})"
    return name


def _assign_block(term: str, spec: ModelSpec) -> str | None:
    """Map a coefficient term back to its logical block."""
    base_term = term.replace("C(", "").replace(")", "")
    base_term = base_term.split("[", 1)[0].split(":", 1)[0]
    for block in spec.blocks:
        for predictor in block.predictors:
            if predictor == base_term or predictor in term:
                return block.name
    if "country" in term:
        return "country"
    return None


@beartype
def standardize_columns(df: pd.DataFrame, columns: Iterable[str], column_types: Mapping[str, str]) -> pd.DataFrame:
    """Standardise selected numeric columns to mean 0 / sd 1."""
    copy = df.copy()
    for name in columns:
        if name not in copy.columns or not _is_numeric_col(name, column_types):
            continue
        series = copy[name]
        if not pd.api.types.is_numeric_dtype(series):
            continue
        mean = series.mean(skipna=True)
        std = series.std(skipna=True, ddof=0)
        if std and not math.isclose(std, 0.0):
            copy[name] = (series - mean) / std
        else:
            logger.debug(f"Skipping standardisation for {name} (zero variance)")
    return copy


def _prepare_formula(
    spec: ModelSpec,
    df: pd.DataFrame,
    column_types: Mapping[str, str],
    reference_levels: Mapping[str, object] | None,
) -> tuple[str, set[str]]:
    missing: set[str] = set()
    terms: list[str] = []
    for block in spec.blocks:
        for predictor in block.predictors:
            if predictor == spec.outcome:
                logger.debug(f"Skipping predictor {predictor} because it matches outcome {spec.outcome}")
                continue
            if predictor not in df.columns:
                missing.add(predictor)
                continue
            terms.append(_term_for_predictor(predictor, column_types, df, reference_levels))

    for interaction in spec.interactions:
        if not interaction:
            continue
        missing_vars = [var for var in interaction if var not in df.columns]
        if missing_vars:
            missing.update(missing_vars)
            continue
        formatted = [_term_for_predictor(var, column_types, df, reference_levels) for var in interaction]
        terms.append(":".join(formatted))

    if spec.add_country and "country" in df.columns:
        terms.append(_term_for_predictor("country", column_types, df, reference_levels))
    deduped_terms: list[str] = []
    seen: set[str] = set()
    for term in terms:
        if term not in seen:
            seen.add(term)
            deduped_terms.append(term)

    rhs = " + ".join(deduped_terms) if deduped_terms else "1"
    formula = f"{spec.outcome} ~ {rhs}"
    return formula, missing


def _extract_coefficient_table(
    result,
    spec: ModelSpec,
    *,
    drop_thresholds: bool = True,
) -> pd.DataFrame:
    def _classify_effect_strength(abs_z: float | None) -> str | None:
        if abs_z is None:
            return None
        if abs_z < 1.0:
            return "negligible"
        if abs_z < 2.0:
            return "weak"
        if abs_z < 3.0:
            return "moderate"
        return "strong"

    def _significance_from_p(p_value: float | None) -> str | None:
        if p_value is None or not np.isfinite(p_value):
            return None
        if p_value < 0.001:
            return "***"
        if p_value < 0.01:
            return "**"
        if p_value < 0.05:
            return "*"
        return ""

    params = result.params
    bse = result.bse
    ci = result.conf_int()
    if hasattr(ci, "columns") and list(ci.columns) == [0, 1]:
        ci.columns = ["ci_low", "ci_high"]
    table_rows: list[dict[str, object]] = []
    statistic = getattr(result, "zvalues", None) or getattr(result, "tvalues", None)
    pvalues = getattr(result, "pvalues", None)
    for term, estimate in params.items():
        term_lower = term.lower()
        if drop_thresholds and (term_lower.startswith("cut") or term_lower.startswith("threshold")):
            continue
        z_value = float(statistic[term]) if statistic is not None and term in statistic else None
        abs_z = abs(z_value) if z_value is not None else None
        p_value = float(pvalues[term]) if pvalues is not None and term in pvalues else None
        row = {
            "term": term,
            "estimate": float(estimate),
            "std_err": float(bse[term]) if term in bse else None,
            "z_value": z_value,
            "abs_z": abs_z,
            "effect_strength": _classify_effect_strength(abs_z),
            "significance": _significance_from_p(p_value),
            "ci_low": float(ci.loc[term, "ci_low"]) if term in ci.index else None,
            "ci_high": float(ci.loc[term, "ci_high"]) if term in ci.index else None,
            "block": _assign_block(term, spec),
        }
        if spec.model_type in {"logit", "ordinal"}:
            row["odds_ratio"] = float(np.exp(estimate))
            if row["ci_low"] is not None:
                row["odds_ratio_ci_low"] = float(np.exp(row["ci_low"]))
            if row["ci_high"] is not None:
                row["odds_ratio_ci_high"] = float(np.exp(row["ci_high"]))
        table_rows.append(row)
    df = pd.DataFrame(table_rows)
    df.insert(0, "model", spec.name)
    if spec.group:
        df.insert(1, "group", spec.group)
    return df


def _fit_linear(formula: str, data: pd.DataFrame, weights: pd.Series | None):
    if weights is not None:
        return smf.wls(formula=formula, data=data, weights=weights).fit()
    return smf.ols(formula=formula, data=data).fit()


def _fit_logit(formula: str, data: pd.DataFrame, weights: pd.Series | None):
    model = smf.glm(formula=formula, data=data, family=sm.families.Binomial(), freq_weights=weights)
    return model.fit()


def _fit_ordinal(formula: str, data: pd.DataFrame, weights: pd.Series | None, *, skip_hessian: bool = False):
    y, X = patsy.dmatrices(formula, data=data, return_type="dataframe")
    # OrderedModel handles cutpoints separately; drop the intercept if present.
    if "Intercept" in X.columns:
        X = X.drop(columns=["Intercept"])
    constant_cols = [name for name in X.columns if np.isclose(X[name].std(ddof=0), 0.0)]
    if constant_cols:
        logger.debug(f"Dropping constant predictors from ordinal design: {', '.join(constant_cols)}")
        X = X.drop(columns=constant_cols)
    model = OrderedModel(y.iloc[:, 0], X, distr="logit", weights=weights if weights is not None else None)
    return model.fit(method="bfgs", disp=False, skip_hessian=skip_hessian)


def _compute_pseudo_r2(result, model_type: ModelType) -> float | None:
    if model_type == "linear":
        return float(getattr(result, "rsquared", np.nan))
    if model_type == "logit":
        null_dev = getattr(result, "null_deviance", None)
        dev = getattr(result, "deviance", None)
        if null_dev is None or dev is None or not np.isfinite(null_dev):
            return None
        return float(1.0 - dev / null_dev) if null_dev else None
    if model_type == "ordinal":
        llf = getattr(result, "llf", None)
        llnull = getattr(result, "llnull", None)
        if llf is not None and llnull is not None and np.isfinite(llnull):
            return float((llnull - llf) / llnull)
        return None
    return None


def bic_consistent(result) -> float | None:
    """Return a log-likelihood based BIC that is consistent across model classes."""

    def _as_float(value) -> float | None:
        if value is None or not np.isfinite(value):
            return None
        return float(value)

    bic_llf = _as_float(getattr(result, "bic_llf", None))
    if bic_llf is not None:
        return bic_llf

    bic = _as_float(getattr(result, "bic", None))
    if bic is not None:
        return bic

    llf = _as_float(getattr(result, "llf", None))
    nobs = _as_float(getattr(result, "nobs", None))
    params = getattr(result, "params", None)
    if llf is None or nobs is None or nobs <= 0 or params is None:
        return None
    try:
        k = len(params)
    except TypeError:
        k = int(getattr(result, "df_model", 0)) + 1
    if k <= 0:
        return None
    return float(-2.0 * llf + math.log(nobs) * k)


@beartype
def fit_model(
    spec: ModelSpec,
    df: pd.DataFrame,
    column_types: Mapping[str, str],
    reference_levels: Mapping[str, object] | None = None,
    *,
    weight_column: str | None = None,
    outcome_overrides: Mapping[str, pd.Series] | None = None,
) -> ModelResult:
    """Fit a single model specification."""
    if spec.outcome not in df.columns:
        raise ValueError(f"Outcome column '{spec.outcome}' not found in dataframe")

    needed_cols = [term for term in spec.predictor_list() if term in df.columns]
    needed_cols.append(spec.outcome)
    if weight_column:
        needed_cols.append(weight_column)
    # Drop duplicates while preserving order to avoid duplicated column names downstream.
    seen_cols: set[str] = set()
    deduped: list[str] = []
    for col in needed_cols:
        if col in seen_cols:
            continue
        seen_cols.add(col)
        deduped.append(col)
    needed_cols = deduped

    working = df[needed_cols].copy()
    if outcome_overrides:
        original = outcome_overrides.get(spec.outcome)
        if original is not None:
            working[spec.outcome] = original.reindex(working.index)
    # Fail fast on unexpected dtypes instead of silently coercing.
    if not pd.api.types.is_numeric_dtype(working[spec.outcome]):
        raise TypeError(f"Outcome '{spec.outcome}' must be numeric; got {working[spec.outcome].dtype}")
    if weight_column and weight_column in working and not pd.api.types.is_numeric_dtype(working[weight_column]):
        raise TypeError(f"Weight column '{weight_column}' must be numeric; got {working[weight_column].dtype}")
    working = working.dropna()
    formula, missing = _prepare_formula(spec, working, column_types, reference_levels)
    if missing:
        logger.warning(f"Model {spec.name} missing predictors: {', '.join(sorted(missing))}")
    weights = working[weight_column] if weight_column and weight_column in working.columns else None
    logger.info(f"Fitting {spec.name} ({working.shape[0]} rows) with formula: {formula}")

    if spec.model_type == "linear":
        result = _fit_linear(formula, working, weights)
    elif spec.model_type == "logit":
        result = _fit_logit(formula, working, weights)
    elif spec.model_type == "ordinal":
        result = _fit_ordinal(formula, working, weights)
    else:
        raise ValueError(f"Unknown model type: {spec.model_type}")

    coef_table = _extract_coefficient_table(result, spec, drop_thresholds=spec.model_type == "ordinal")
    pseudo_r2 = _compute_pseudo_r2(result, spec.model_type)
    return ModelResult(
        spec=spec,
        formula=formula,
        coefficients=coef_table,
        nobs=int(getattr(result, "nobs", working.shape[0])),
        pseudo_r2=pseudo_r2 if pseudo_r2 is None or np.isfinite(pseudo_r2) else None,
    )


@beartype
def run_model_suite(
    specs: Sequence[ModelSpec],
    df: pd.DataFrame,
    column_types: Mapping[str, str],
    reference_levels: Mapping[str, object] | None = None,
    *,
    weight_column: str | None = None,
    max_workers: int | None = None,
    outcome_overrides: Mapping[str, pd.Series] | None = None,
) -> list[ModelResult]:
    """Fit a list of ModelSpecs against the provided dataframe."""

    def _worker(spec: ModelSpec) -> ModelResult | None:
        try:
            return fit_model(
                spec,
                df,
                column_types,
                reference_levels,
                weight_column=weight_column,
                outcome_overrides=outcome_overrides,
            )
        except Exception as exc:  # pragma: no cover - safety net for CLI use
            logger.exception(f"Failed to fit model {spec.name}: {exc}")
            return None

    if max_workers is None or max_workers <= 1 or len(specs) <= 1:
        return [result for spec in specs if (result := _worker(spec)) is not None]

    results: list[ModelResult] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for result in executor.map(_worker, specs):
            if result is not None:
                results.append(result)
    return results
