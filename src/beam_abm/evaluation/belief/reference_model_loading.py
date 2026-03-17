"""Reference reduced P-model loading and prediction helpers."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True, slots=True)
class RefStateModel:
    """Reduced P-model used for PE_ref_P generation."""

    model_name: str
    terms: tuple[str, ...]
    coefficients: np.ndarray
    intercept: float
    thresholds: np.ndarray | None
    term_means: np.ndarray | None = None
    term_stds: np.ndarray | None = None


_C_DUMMY_RE = re.compile(r"^C\((?P<var>[^,\)]+)(?:,.*)?\)\[T\.(?P<level>.*)\]$")
_SIMPLE_DUMMY_RE = re.compile(r"^(?P<var>[^\[]+)\[T\.(?P<level>.*)\]$")


def default_ref_state_models() -> dict[str, str]:
    """Default mutable-state to reduced P-model mapping."""

    return {
        "fivec_low_constraints": "P_5C_fivec_low_constraints_rev_REF",
        "institutional_trust_avg": "P_TRUST_institutional_trust_REF",
        "group_trust_vax": "P_TRUST_group_trust_vax_REF",
        "covax_legitimacy_scepticism_idx": "P_LEGIT_covax_legitimacy_REF",
        "vaccine_risk_avg": "P_RISK_vaccine_risk_REF",
        "social_norms_vax_avg": "P_NORMS_social_norms_vax_avg_REF",
        "covid_perceived_danger_T12": "P_STAKES_covid_perceived_danger_T12_REF",
    }


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return np.where(x >= 0.0, 1.0 / (1.0 + np.exp(-x)), np.exp(x) / (1.0 + np.exp(x)))


def _safe_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(out) or math.isinf(out):
        return None
    return out


def _as_string_level(value: object) -> str:
    if value is None:
        return "__MISSING__"
    if isinstance(value, float) and math.isnan(value):
        return "__MISSING__"
    if isinstance(value, bool):
        return "True" if value else "False"
    return str(value).strip()


def _match_level(value: object, level: str) -> bool:
    lvl = str(level).strip()
    if not lvl:
        return False

    if lvl in {"True", "False"}:
        target = lvl == "True"
        if isinstance(value, bool):
            return bool(value) == target
        val_f = _safe_float(value)
        if val_f is not None:
            if abs(val_f - 1.0) <= 1e-9:
                return target
            if abs(val_f) <= 1e-9:
                return not target
        sval = _as_string_level(value).strip().lower()
        if sval in {"true", "false"}:
            return (sval == "true") == target
        if sval in {"1", "1.0"}:
            return target
        if sval in {"0", "0.0"}:
            return not target
        return False

    val_f = _safe_float(value)
    lvl_f = _safe_float(lvl)
    if val_f is not None and lvl_f is not None:
        return abs(val_f - lvl_f) <= 1e-9

    return _as_string_level(value) == lvl


def _strip_q_wrapper(var_expr: str) -> str:
    expr = var_expr.strip()
    if expr.startswith("Q(") and expr.endswith(")"):
        inner = expr[2:-1].strip()
        if (inner.startswith('"') and inner.endswith('"')) or (inner.startswith("'") and inner.endswith("'")):
            return inner[1:-1]
        return inner
    return expr


def _term_series(df: pd.DataFrame, term: str) -> np.ndarray:
    """Evaluate a coefficients term against a profile dataframe."""

    if term in df.columns:
        return pd.to_numeric(df[term], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)

    if term.endswith("_missing"):
        base = term[: -len("_missing")]
        if base in df.columns:
            return pd.isna(df[base]).astype(float).to_numpy(dtype=np.float64)
        return np.zeros(len(df), dtype=np.float64)

    m_c = _C_DUMMY_RE.match(term)
    if m_c:
        var = _strip_q_wrapper(m_c.group("var"))
        level = m_c.group("level")
        if var not in df.columns:
            return np.zeros(len(df), dtype=np.float64)
        return np.asarray([1.0 if _match_level(v, level) else 0.0 for v in df[var].tolist()], dtype=np.float64)

    m_s = _SIMPLE_DUMMY_RE.match(term)
    if m_s:
        var = _strip_q_wrapper(m_s.group("var"))
        level = m_s.group("level")
        if var not in df.columns:
            return np.zeros(len(df), dtype=np.float64)
        return np.asarray([1.0 if _match_level(v, level) else 0.0 for v in df[var].tolist()], dtype=np.float64)

    return np.zeros(len(df), dtype=np.float64)


def _find_coefficients_path(modeling_dir: Path, model_name: str) -> Path | None:
    for group_dir in sorted(p for p in modeling_dir.iterdir() if p.is_dir()):
        candidate = group_dir / model_name / "coefficients.csv"
        if candidate.is_file():
            return candidate
    return None


def _parse_thresholds(threshold_terms: list[tuple[str, float]]) -> np.ndarray:
    parsed: list[tuple[float, float]] = []
    for term, estimate in threshold_terms:
        left = term.split("/", 1)[0].strip()
        left_f = _safe_float(left)
        if left_f is None:
            continue
        parsed.append((left_f, float(estimate)))
    if not parsed:
        return np.asarray([], dtype=np.float64)
    parsed.sort(key=lambda x: x[0])
    raw = np.asarray([x[1] for x in parsed], dtype=np.float64)
    thresholds = np.empty_like(raw)
    thresholds[0] = raw[0]
    if raw.size > 1:
        thresholds[1:] = raw[0] + np.cumsum(np.exp(raw[1:]))
    return thresholds


def _load_ref_model(modeling_dir: Path, model_name: str) -> RefStateModel | None:
    coeff_path = _find_coefficients_path(modeling_dir=modeling_dir, model_name=model_name)
    if coeff_path is None:
        return None

    df = pd.read_csv(coeff_path)
    if "term" not in df.columns or "estimate" not in df.columns:
        return None

    intercept = 0.0
    terms: list[str] = []
    coefs: list[float] = []
    threshold_terms: list[tuple[str, float]] = []

    for _, row in df.iterrows():
        term = str(row.get("term", "")).strip()
        est = _safe_float(row.get("estimate"))
        if not term or est is None:
            continue
        if term == "Intercept":
            intercept = float(est)
            continue
        if "/" in term:
            threshold_terms.append((term, float(est)))
            continue
        terms.append(term)
        coefs.append(float(est))

    thresholds = _parse_thresholds(threshold_terms) if threshold_terms else None
    if thresholds is not None and int(thresholds.size) == 0:
        thresholds = None

    return RefStateModel(
        model_name=model_name,
        terms=tuple(terms),
        coefficients=np.asarray(coefs, dtype=np.float64),
        intercept=float(intercept),
        thresholds=thresholds,
    )


def _predict_model(model: RefStateModel, df: pd.DataFrame) -> np.ndarray:
    if not model.terms:
        eta = np.full(len(df), model.intercept, dtype=np.float64)
    else:
        X = np.column_stack([_term_series(df, t) for t in model.terms]).astype(np.float64)
        if model.term_means is not None and model.term_stds is not None:
            X = (X - model.term_means[np.newaxis, :]) / model.term_stds[np.newaxis, :]
        eta = model.intercept + X @ model.coefficients

    if model.thresholds is None:
        return eta

    # Ordinal expected value on 1..J scale.
    thresholds = model.thresholds.astype(np.float64)
    n = eta.shape[0]
    n_cat = int(len(thresholds) + 1)

    cum = _sigmoid(thresholds[np.newaxis, :] - eta[:, np.newaxis])
    probs = np.empty((n, n_cat), dtype=np.float64)
    probs[:, 0] = cum[:, 0]
    for j in range(1, n_cat - 1):
        probs[:, j] = cum[:, j] - cum[:, j - 1]
    probs[:, n_cat - 1] = 1.0 - cum[:, n_cat - 2]

    probs = np.clip(probs, 0.0, 1.0)
    row_sum = probs.sum(axis=1, keepdims=True)
    row_sum = np.where(row_sum > 0.0, row_sum, 1.0)
    probs /= row_sum

    categories = np.arange(1.0, float(n_cat) + 1.0, dtype=np.float64)
    return probs @ categories


def _compute_ref_term_scaler_stats(
    terms: tuple[str, ...],
    source_df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """Match fit-time standardization for REF-model numeric design terms."""

    means = np.zeros(len(terms), dtype=np.float64)
    stds = np.ones(len(terms), dtype=np.float64)
    for i, term in enumerate(terms):
        if term.endswith("_missing") or "[T." in term:
            continue
        if term not in source_df.columns:
            continue
        series = pd.to_numeric(source_df[term], errors="coerce")
        finite = series[np.isfinite(series)]
        if finite.empty:
            continue
        unique = np.sort(finite.unique())
        if unique.size <= 2 and np.isin(unique, [0.0, 1.0]).all():
            continue
        mean = float(finite.mean())
        std = float(finite.std(ddof=0))
        if not np.isfinite(std) or np.isclose(std, 0.0):
            continue
        means[i] = mean
        stds[i] = std
    return means, stds
