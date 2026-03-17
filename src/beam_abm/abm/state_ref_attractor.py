"""Survey-anchored state attractors built from reduced P-models.

Implements per-tick mutable-state targets:

``S*(t) = f_k(X_t)``

and persistent offsets:

``u_i = S_i(0) - f_k(X_i(0))``.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from loguru import logger

from beam_abm.abm.data_contracts import BackboneModel
from beam_abm.abm.state_schema import CONSTRUCT_BOUNDS, MUTABLE_COLUMNS

__all__ = [
    "StateRefAttractor",
    "build_state_ref_attractor",
    "default_ref_state_model_aliases",
    "default_ref_state_models",
]


_C_DUMMY_RE = re.compile(r"^C\((?P<var>[^,\)]+)(?:,.*)?\)\[T\.(?P<level>.*)\]$")
_SIMPLE_DUMMY_RE = re.compile(r"^(?P<var>[^\[]+)\[T\.(?P<level>.*)\]$")
_THRESHOLD_TERM_RE = re.compile(r"^\s*[-+]?\d+(?:\.\d+)?\s*/\s*[-+]?\d+(?:\.\d+)?\s*$")


@dataclass(frozen=True, slots=True)
class _StateBinding:
    state_var: str
    runtime_columns: tuple[str, ...]
    model_ref: str
    model: BackboneModel
    term_indices: np.ndarray
    coefficients_effective: np.ndarray
    intercept_effective: float


@dataclass(frozen=True, slots=True)
class _CompiledTerm:
    term: str
    kind: str
    column: str
    level: str | None = None


@dataclass(slots=True)
class StateRefAttractor:
    """Runtime state-attractor models and persistent offsets."""

    bindings: tuple[_StateBinding, ...]
    shared_terms: tuple[_CompiledTerm, ...]
    mutable_term_indices: np.ndarray
    immutable_terms_template: np.ndarray
    persistent_offsets: dict[str, np.ndarray]
    model_status: dict[str, Any]
    strict: bool = False

    def predict_targets(self, agents: pl.DataFrame) -> dict[str, np.ndarray]:
        """Predict per-construct attractor targets for one tick."""
        targets: dict[str, np.ndarray] = {}
        raw_cache: dict[str, np.ndarray] = {}
        numeric_cache: dict[str, np.ndarray] = {}
        missing_cache: dict[str, np.ndarray] = {}
        shared_x = _assemble_compiled_terms_matrix(
            agents=agents,
            compiled_terms=self.shared_terms,
            strict=self.strict,
            raw_cache=raw_cache,
            numeric_cache=numeric_cache,
            missing_cache=missing_cache,
        )
        for binding in self.bindings:
            predicted = _predict_expected_from_shared(
                model=binding.model,
                shared_x=shared_x,
                term_indices=binding.term_indices,
                coefficients_effective=binding.coefficients_effective,
                intercept_effective=binding.intercept_effective,
                n_agents=agents.height,
            )
            for col in binding.runtime_columns:
                if col in targets:
                    continue
                lo, hi = CONSTRUCT_BOUNDS.get(col, (-np.inf, np.inf))
                targets[col] = np.clip(predicted, float(lo), float(hi))
        return targets

    def merged_offsets(
        self,
        dynamic_offsets: dict[str, np.ndarray] | None,
    ) -> dict[str, np.ndarray] | None:
        """Merge persistent offsets with transient dynamic offsets."""
        if not self.persistent_offsets and not dynamic_offsets:
            return None
        out: dict[str, np.ndarray] = {key: values.copy() for key, values in self.persistent_offsets.items()}
        if dynamic_offsets:
            for key, values in dynamic_offsets.items():
                if key in out:
                    out[key] = out[key] + values
                else:
                    out[key] = values.copy()
        return out


def default_ref_state_models() -> dict[str, str]:
    """Default mutable-state to reduced P-model mapping."""
    return {
        "fivec_low_constraints": "P_CONSTRAINTS_fivec_low_constraints_rev_REF",
        "institutional_trust_avg": "P_TRUST_institutional_trust_REF",
        "group_trust_vax": "P_TRUST_group_trust_vax_REF",
        "covax_legitimacy_scepticism_idx": "P_LEGIT_covax_legitimacy_REF",
        "vaccine_risk_avg": "P_RISK_vaccine_risk_REF",
        "social_norms_vax_avg": "P_NORMS_social_norms_vax_avg_REF",
        "covid_perceived_danger_T12": "P_STAKES_covid_perceived_danger_T12_REF",
    }


def default_ref_state_model_aliases() -> dict[str, tuple[str, ...]]:
    """Fallback aliases for model names that vary across runs."""
    return {
        "fivec_low_constraints": (
            "P_CONSTRAINTS_fivec_low_constraints_rev_REF",
            "P_5C_fivec_low_constraints_rev",
            "P_5C_fivec_low_constraints_rev_PN_ITEMS",
        ),
        "institutional_trust_avg": ("P_TRUST_institutional_trust_REF",),
        "group_trust_vax": ("P_TRUST_group_trust_vax_REF",),
        "covax_legitimacy_scepticism_idx": ("P_LEGIT_covax_legitimacy_REF",),
        "vaccine_risk_avg": ("P_RISK_vaccine_risk_REF",),
        "social_norms_vax_avg": ("P_NORMS_social_norms_vax_avg_REF",),
        "covid_perceived_danger_T12": (
            "P_STAKES_covid_perceived_danger_T12_REF",
            "P_STAKES_covid_perceived_danger_Y_REF",
        ),
    }


def _canonical_state_var(state_var: str) -> str:
    if state_var == "covid_perceived_danger_Y":
        return "covid_perceived_danger_T12"
    return state_var


def build_state_ref_attractor(
    *,
    agents_t0: pl.DataFrame,
    modeling_dir: Path,
    spec_path: Path | None,
    strict: bool,
) -> StateRefAttractor | None:
    """Build survey-anchored attractor models from reduced P-model refs."""
    spec_payload: dict[str, Any] = {}
    if spec_path is not None and spec_path.exists():
        spec_payload = json.loads(spec_path.read_text(encoding="utf-8"))

    ref_map_raw = spec_payload.get("ref_state_models")
    ref_map = (
        {str(k): str(v) for k, v in ref_map_raw.items()}
        if isinstance(ref_map_raw, dict)
        else default_ref_state_models()
    )
    mutable_raw = spec_payload.get("mutable_state")
    mutable_state = (
        [str(x) for x in mutable_raw if str(x).strip()] if isinstance(mutable_raw, list) else list(ref_map.keys())
    )

    aliases_raw = spec_payload.get("ref_state_model_aliases")
    aliases: dict[str, tuple[str, ...]] = {}
    if isinstance(aliases_raw, dict):
        for key, values in aliases_raw.items():
            if isinstance(values, list):
                aliases[str(key)] = tuple(str(v) for v in values if str(v).strip())
    if not aliases:
        aliases = default_ref_state_model_aliases()

    raw_bindings: list[tuple[str, tuple[str, ...], str, BackboneModel]] = []
    missing_models: dict[str, str] = {}
    alias_resolved: dict[str, str] = {}

    for state_var in mutable_state:
        canonical_state_var = _canonical_state_var(state_var)
        requested = ref_map.get(state_var) or ref_map.get(canonical_state_var)
        if not requested:
            continue
        runtime_cols = _resolve_runtime_columns(canonical_state_var, agents_t0)
        if not runtime_cols:
            continue

        candidates = [str(requested), *list(aliases.get(canonical_state_var, ())), *list(aliases.get(state_var, ()))]
        seen: set[str] = set()
        selected_ref: str | None = None
        selected_model: BackboneModel | None = None
        for candidate in candidates:
            model_ref = candidate.strip()
            if not model_ref or model_ref in seen:
                continue
            seen.add(model_ref)
            selected_model = _load_ref_model(modeling_dir=modeling_dir, model_ref=model_ref)
            if selected_model is not None:
                selected_ref = model_ref
                break
        if selected_model is None or selected_ref is None:
            missing_models[canonical_state_var] = str(requested)
            continue
        if selected_ref != str(requested):
            alias_resolved[canonical_state_var] = f"{requested} -> {selected_ref}"
        means, stds = _compute_ref_term_scaler_stats(selected_model.terms, agents_t0)
        selected_model = BackboneModel(
            name=selected_model.name,
            model_type=selected_model.model_type,
            terms=selected_model.terms,
            coefficients=selected_model.coefficients,
            thresholds=selected_model.thresholds,
            intercept=selected_model.intercept,
            blocks=selected_model.blocks,
            term_means=means,
            term_stds=stds,
        )
        raw_bindings.append((canonical_state_var, runtime_cols, selected_ref, selected_model))

    if not raw_bindings:
        logger.warning("No state reference models were resolved for attractor updates.")
        return None

    shared_terms: list[_CompiledTerm] = []
    shared_term_index: dict[str, int] = {}
    bindings: list[_StateBinding] = []
    for state_var, runtime_cols, model_ref, model in raw_bindings:
        term_indices = np.empty(len(model.terms), dtype=np.int64)
        for i, term in enumerate(model.terms):
            idx = shared_term_index.get(term)
            if idx is None:
                idx = len(shared_terms)
                shared_term_index[term] = idx
                shared_terms.append(_compile_term(term))
            term_indices[i] = idx

        coefficients_effective = model.coefficients.astype(np.float64, copy=True)
        intercept_effective = float(model.intercept or 0.0)
        if model.term_means is not None and model.term_stds is not None and coefficients_effective.size > 0:
            coefficients_effective = coefficients_effective / model.term_stds
            intercept_effective -= float(np.dot(model.term_means, coefficients_effective))

        bindings.append(
            _StateBinding(
                state_var=state_var,
                runtime_columns=runtime_cols,
                model_ref=model_ref,
                model=model,
                term_indices=term_indices,
                coefficients_effective=coefficients_effective,
                intercept_effective=intercept_effective,
            )
        )

    shared_terms_tuple = tuple(shared_terms)
    mutable_term_indices = np.fromiter(
        (i for i, term in enumerate(shared_terms_tuple) if term.column in MUTABLE_COLUMNS),
        dtype=np.int64,
    )
    template_raw_cache: dict[str, np.ndarray] = {}
    template_numeric_cache: dict[str, np.ndarray] = {}
    template_missing_cache: dict[str, np.ndarray] = {}
    immutable_terms_template = _assemble_compiled_terms_matrix(
        agents=agents_t0,
        compiled_terms=shared_terms_tuple,
        strict=bool(strict),
        raw_cache=template_raw_cache,
        numeric_cache=template_numeric_cache,
        missing_cache=template_missing_cache,
    )
    if mutable_term_indices.size > 0:
        immutable_terms_template[:, mutable_term_indices] = 0.0

    attractor = StateRefAttractor(
        bindings=tuple(bindings),
        shared_terms=shared_terms_tuple,
        mutable_term_indices=mutable_term_indices,
        immutable_terms_template=immutable_terms_template,
        persistent_offsets={},
        model_status={
            "loaded": {b.state_var: b.model_ref for b in bindings},
            "missing": missing_models,
            "alias_resolved": alias_resolved,
        },
        strict=bool(strict),
    )

    try:
        targets_t0 = attractor.predict_targets(agents_t0)
    except ValueError as exc:
        if strict:
            raise
        logger.warning("Disabling state attractor: {msg}", msg=str(exc))
        return None

    offsets: dict[str, np.ndarray] = {}
    for col, target in targets_t0.items():
        observed = agents_t0[col].cast(pl.Float64).to_numpy().astype(np.float64)
        offsets[col] = observed - target
    attractor.persistent_offsets = offsets
    logger.info(
        "Loaded state attractor with {n} bindings for columns: {cols}",
        n=len(bindings),
        cols=sorted(offsets.keys()),
    )
    return attractor


def _find_coefficients_path(modeling_dir: Path, model_ref: str) -> Path | None:
    for group_dir in sorted(modeling_dir.iterdir()):
        if not group_dir.is_dir():
            continue
        candidate = group_dir / model_ref / "coefficients.csv"
        if candidate.is_file():
            return candidate
    return None


def _load_ref_model(modeling_dir: Path, model_ref: str) -> BackboneModel | None:
    coeff_path = _find_coefficients_path(modeling_dir=modeling_dir, model_ref=model_ref)
    if coeff_path is None:
        return None

    df = pl.read_csv(coeff_path)
    if not {"term", "estimate"}.issubset(set(df.columns)):
        return None

    intercept: float | None = None
    terms: list[str] = []
    coefficients: list[float] = []
    threshold_terms: list[tuple[float, float]] = []

    for row in df.iter_rows(named=True):
        term = str(row.get("term", "")).strip()
        if not term:
            continue
        try:
            estimate = float(row.get("estimate"))
        except (TypeError, ValueError):
            continue
        if not np.isfinite(estimate):
            continue
        if term == "Intercept":
            intercept = float(estimate)
            continue
        if _THRESHOLD_TERM_RE.match(term):
            left = term.split("/", 1)[0].strip()
            try:
                lower = float(left)
            except ValueError:
                continue
            threshold_terms.append((lower, float(estimate)))
            continue
        terms.append(term)
        coefficients.append(float(estimate))

    if threshold_terms:
        threshold_terms.sort(key=lambda item: item[0])
        raw = np.asarray([estimate for _, estimate in threshold_terms], dtype=np.float64)
        thresholds = np.empty_like(raw)
        thresholds[0] = raw[0]
        if raw.size > 1:
            thresholds[1:] = raw[0] + np.cumsum(np.exp(raw[1:]))
        model_type = "ordinal"
    else:
        thresholds = None
        model_type = "binary"

    return BackboneModel(
        name=model_ref,
        model_type=model_type,
        terms=tuple(terms),
        coefficients=np.asarray(coefficients, dtype=np.float64),
        thresholds=thresholds,
        intercept=intercept,
        blocks={},
    )


def _resolve_runtime_columns(state_var: str, agents: pl.DataFrame) -> tuple[str, ...]:
    state_var = _canonical_state_var(state_var)
    cols = set(agents.columns)
    if state_var in cols:
        if state_var == "social_norms_vax_avg":
            desc = "social_norms_descriptive_vax"
            if desc in cols:
                return (desc,)
        return (state_var,)
    if state_var == "covid_perceived_danger_T12" and "covid_perceived_danger_Y" in cols:
        return ("covid_perceived_danger_Y",)
    return ()


def _compile_term(term: str) -> _CompiledTerm:
    if term.endswith("_missing"):
        return _CompiledTerm(term=term, kind="missing", column=term[: -len("_missing")])

    m_c = _C_DUMMY_RE.match(term)
    if m_c is not None:
        return _CompiledTerm(
            term=term,
            kind="categorical_dummy",
            column=_strip_q_wrapper(m_c.group("var")),
            level=m_c.group("level"),
        )

    m_s = _SIMPLE_DUMMY_RE.match(term)
    if m_s is not None:
        return _CompiledTerm(
            term=term,
            kind="categorical_dummy",
            column=_strip_q_wrapper(m_s.group("var")),
            level=m_s.group("level"),
        )

    return _CompiledTerm(term=term, kind="direct", column=term)


def _assemble_compiled_terms_matrix(
    *,
    agents: pl.DataFrame,
    compiled_terms: tuple[_CompiledTerm, ...],
    strict: bool,
    raw_cache: dict[str, np.ndarray],
    numeric_cache: dict[str, np.ndarray],
    missing_cache: dict[str, np.ndarray],
) -> np.ndarray:
    n = agents.height
    if not compiled_terms:
        return np.empty((n, 0), dtype=np.float64)

    x = np.empty((n, len(compiled_terms)), dtype=np.float64)
    for i, compiled in enumerate(compiled_terms):
        x[:, i] = _compiled_term_array(
            agents=agents,
            compiled=compiled,
            strict=strict,
            raw_cache=raw_cache,
            numeric_cache=numeric_cache,
            missing_cache=missing_cache,
        )
    return x


def _predict_expected_from_shared(
    *,
    model: BackboneModel,
    shared_x: np.ndarray,
    term_indices: np.ndarray,
    coefficients_effective: np.ndarray,
    intercept_effective: float,
    n_agents: int,
) -> np.ndarray:
    if term_indices.size == 0:
        eta = np.full(n_agents, intercept_effective, dtype=np.float64)
    else:
        eta = intercept_effective + shared_x[:, term_indices] @ coefficients_effective

    if model.thresholds is None:
        # REF P-models without thresholds are linear regressions. Applying a
        # logistic link here would squash variation into [0, 1] and collapse
        # attractor dynamics after downstream construct clipping.
        if model.name.startswith("P_"):
            return eta
        return _sigmoid(eta)
    thresholds = model.thresholds
    if thresholds.dtype != np.float64:
        thresholds = thresholds.astype(np.float64)
    return _ordinal_expected(eta, thresholds)


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return np.where(values >= 0.0, 1.0 / (1.0 + np.exp(-values)), np.exp(values) / (1.0 + np.exp(values)))


def _ordinal_expected(eta: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    n = eta.shape[0]
    n_cat = int(len(thresholds) + 1)
    cumulative = _sigmoid(thresholds[np.newaxis, :] - eta[:, np.newaxis])
    probs = np.empty((n, n_cat), dtype=np.float64)
    probs[:, 0] = cumulative[:, 0]
    for j in range(1, n_cat - 1):
        probs[:, j] = cumulative[:, j] - cumulative[:, j - 1]
    probs[:, n_cat - 1] = 1.0 - cumulative[:, n_cat - 2]
    probs = np.clip(probs, 0.0, 1.0)
    row_sum = probs.sum(axis=1, keepdims=True)
    row_sum = np.where(row_sum > 0.0, row_sum, 1.0)
    probs = probs / row_sum
    categories = np.arange(1.0, float(n_cat) + 1.0, dtype=np.float64)
    return probs @ categories


def _compiled_term_array(
    *,
    agents: pl.DataFrame,
    compiled: _CompiledTerm,
    strict: bool,
    raw_cache: dict[str, np.ndarray],
    numeric_cache: dict[str, np.ndarray],
    missing_cache: dict[str, np.ndarray],
) -> np.ndarray:
    if compiled.kind == "direct":
        if compiled.column in agents.columns:
            return _numeric_column(agents=agents, column=compiled.column, cache=numeric_cache)
        if strict:
            msg = f"Unsupported term '{compiled.term}' in state reference model."
            raise ValueError(msg)
        return np.zeros(agents.height, dtype=np.float64)

    if compiled.kind == "missing":
        if compiled.column not in agents.columns:
            if strict:
                msg = f"Required missing-indicator base column '{compiled.column}' is unavailable."
                raise ValueError(msg)
            return np.zeros(agents.height, dtype=np.float64)
        return _missing_indicator(agents=agents, column=compiled.column, cache=missing_cache)

    if compiled.kind == "categorical_dummy":
        if compiled.column not in agents.columns:
            if strict:
                msg = f"Required categorical predictor '{compiled.column}' is unavailable."
                raise ValueError(msg)
            return np.zeros(agents.height, dtype=np.float64)
        raw = _raw_column(agents=agents, column=compiled.column, cache=raw_cache)
        level = "" if compiled.level is None else compiled.level
        out = np.asarray([1.0 if _match_level(value, level) else 0.0 for value in raw], dtype=np.float64)
        return out

    if strict:
        msg = f"Unsupported term '{compiled.term}' in state reference model."
        raise ValueError(msg)
    return np.zeros(agents.height, dtype=np.float64)


def _compute_ref_term_scaler_stats(
    terms: tuple[str, ...],
    agents: pl.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-term scaling stats for REF-model design columns.

    Matches choice-model scaling behavior:
    - dummy/missing terms stay unscaled,
    - binary 0/1 predictors stay unscaled,
    - numeric continuous predictors are standardized.
    """
    means = np.zeros(len(terms), dtype=np.float64)
    stds = np.ones(len(terms), dtype=np.float64)
    for i, term in enumerate(terms):
        if term.endswith("_missing"):
            continue
        if "[T." in term:
            continue
        if term not in agents.columns:
            continue
        series = agents[term]
        if not series.dtype.is_numeric():
            continue
        s = series.cast(pl.Float64).drop_nulls().drop_nans()
        if s.len() == 0:
            continue
        unique_vals = s.unique().sort()
        if unique_vals.len() <= 2:
            vals = unique_vals.to_numpy()
            if np.isin(vals, [0.0, 1.0]).all():
                continue
        mean = float(s.mean())
        std = float(s.std(ddof=0))
        if not np.isfinite(std) or np.isclose(std, 0.0):
            continue
        means[i] = mean
        stds[i] = std
    return means, stds


def _raw_column(
    *,
    agents: pl.DataFrame,
    column: str,
    cache: dict[str, np.ndarray],
) -> np.ndarray:
    cached = cache.get(column)
    if cached is not None:
        return cached
    values = agents[column].to_numpy()
    cache[column] = values
    return values


def _numeric_column(
    *,
    agents: pl.DataFrame,
    column: str,
    cache: dict[str, np.ndarray],
) -> np.ndarray:
    cached = cache.get(column)
    if cached is not None:
        return cached
    series = agents[column]
    if series.dtype.is_numeric():
        arr = series.cast(pl.Float64).to_numpy()
    else:
        try:
            arr = series.cast(pl.Float64).to_numpy()
        except Exception:
            arr = np.zeros(agents.height, dtype=np.float64)
    arr = np.nan_to_num(arr.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    cache[column] = arr
    return arr


def _missing_indicator(
    *,
    agents: pl.DataFrame,
    column: str,
    cache: dict[str, np.ndarray],
) -> np.ndarray:
    cached = cache.get(column)
    if cached is not None:
        return cached
    series = agents[column]
    missing = series.is_null().to_numpy()
    if series.dtype.is_numeric():
        values = series.cast(pl.Float64).to_numpy()
        missing = np.logical_or(missing, ~np.isfinite(values))
    out = missing.astype(np.float64)
    cache[column] = out
    return out


def _strip_q_wrapper(var_expr: str) -> str:
    expr = var_expr.strip()
    if expr.startswith("Q(") and expr.endswith(")"):
        inner = expr[2:-1].strip()
        if (inner.startswith('"') and inner.endswith('"')) or (inner.startswith("'") and inner.endswith("'")):
            return inner[1:-1]
        return inner
    return expr


def _match_level(value: object, level: str) -> bool:
    level_token = str(level).strip()
    if not level_token:
        return False
    if level_token in {"True", "False"}:
        if isinstance(value, bool):
            return value is (level_token == "True")
        return str(value).strip() == level_token
    try:
        level_float = float(level_token)
        value_float = float(value)  # type: ignore[arg-type]
        if np.isfinite(level_float) and np.isfinite(value_float):
            return abs(value_float - level_float) <= 1e-9
    except (TypeError, ValueError):
        pass
    return str(value).strip() == level_token
