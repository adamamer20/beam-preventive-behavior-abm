"""Fixed choice module — immutable prediction functions for ABM outcomes.

Implements the survey-estimated choice rules as deterministic or
stochastic mappings from agent state to behavioural outcomes.  The
module never re-fits models; it only evaluates pre-estimated
coefficients.

Model types
-----------
Binary logistic
    ``P(Y=1) = sigmoid(intercept + X @ beta)``
Ordinal logistic (cumulative link)
    ``P(Y <= j) = sigmoid(threshold_j - X @ beta)``

Interface rule: ``e_Y(t)`` affects behaviour only through this module;
``e_S(t)`` must not be consumed here.

Examples
--------
>>> import numpy as np
>>> from beam_abm.abm.outcomes import linear_predictor
>>> eta = linear_predictor(
...     np.array([[0.5, -0.3]]),
...     np.array([1.0, 2.0]),
...     intercept=0.0,
... )
>>> eta.shape
(1,)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import polars as pl
from loguru import logger

from beam_abm.abm.data_contracts import BackboneModel, LeversConfig
from beam_abm.abm.message_contracts import NormChannelWeights
from beam_abm.abm.state_update import SignalSpec, apply_signals_to_frame

if TYPE_CHECKING:
    pass

__all__ = [
    "linear_predictor",
    "predict_binary",
    "predict_ordinal",
    "predict_outcome",
    "predict_all_outcomes",
    "assemble_design_matrix",
]

FIVEC_COMPONENT_COLUMNS: tuple[str, ...] = (
    "fivec_confidence",
    "fivec_low_complacency",
    "fivec_low_constraints",
    "fivec_collective_responsibility",
)


def _extract_term_column(
    agents: pl.DataFrame,
    term: str,
    *,
    country_arr: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Extract one design-term column from agents.

    Returns column values and potentially populated ``country_arr`` cache.
    """
    n = agents.height
    if term.startswith("C(country"):
        code = _extract_country_code(term)
        if country_arr is None:
            country_arr = agents["country"].to_numpy()
        return (country_arr == code).astype(np.float64), country_arr
    if term in agents.columns:
        col_series = agents[term].cast(pl.Float64)
        col_vals = col_series.to_numpy()
        if not np.isfinite(col_vals).all():
            msg = f"Design-matrix term '{term}' contains null/NaN/inf values."
            raise ValueError(msg)
        return col_vals, country_arr
    if term.endswith("_missing"):
        return np.zeros(n, dtype=np.float64), country_arr
    msg = f"Required term '{term}' not found in agent DataFrame."
    raise ValueError(msg)


def _assemble_terms_matrix(
    agents: pl.DataFrame,
    terms: tuple[str, ...],
) -> np.ndarray:
    """Build matrix for an arbitrary ordered term list."""
    n = agents.height
    x = np.zeros((n, len(terms)), dtype=np.float64)
    country_arr: np.ndarray | None = None
    for i, term in enumerate(terms):
        col_vals, country_arr = _extract_term_column(
            agents,
            term,
            country_arr=country_arr,
        )
        x[:, i] = col_vals
    return x


def _ordered_unique_terms(models: tuple[BackboneModel, ...]) -> tuple[str, ...]:
    """Return deduplicated model terms preserving first-seen order."""
    seen: set[str] = set()
    ordered: list[str] = []
    for model in models:
        for term in model.terms:
            if term in seen:
                continue
            seen.add(term)
            ordered.append(term)
    return tuple(ordered)


def _with_norm_composite(agents: pl.DataFrame, weights: NormChannelWeights | None = None) -> pl.DataFrame:
    """Build runtime social_norms_vax_avg from descriptive norms only."""
    if "social_norms_descriptive_vax" not in agents.columns:
        msg = "Missing required norm channel: 'social_norms_descriptive_vax'."
        raise ValueError(msg)
    _ = weights  # kept for API compatibility; composite is descriptive-only.
    desc = agents["social_norms_descriptive_vax"].cast(pl.Float64)
    desc_np = desc.to_numpy()
    if not np.isfinite(desc_np).all():
        msg = "Norm channels contain null/NaN/inf values; strict ABM mode disallows fallback imputation."
        raise ValueError(msg)
    return agents.with_columns(desc.alias("social_norms_vax_avg"))


def _with_fivec_composite(agents: pl.DataFrame) -> pl.DataFrame:
    """Build runtime fivec_pro_vax_idx from the four 5C component channels."""
    missing = [col for col in FIVEC_COMPONENT_COLUMNS if col not in agents.columns]
    if missing:
        msg = f"Missing required 5C components for fivec_pro_vax_idx: {missing}."
        raise ValueError(msg)

    components = []
    for col in FIVEC_COMPONENT_COLUMNS:
        arr = agents[col].cast(pl.Float64).to_numpy()
        if not np.isfinite(arr).all():
            msg = f"5C component '{col}' contains null/NaN/inf values; strict ABM mode disallows fallback imputation."
            raise ValueError(msg)
        components.append(arr)

    composite = np.mean(np.vstack(components), axis=0)
    return agents.with_columns(pl.Series("fivec_pro_vax_idx", composite))


# =========================================================================
# 1.  Low-level vectorised functions
# =========================================================================


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable logistic sigmoid.

    Parameters
    ----------
    x : np.ndarray
        Input array of any shape.

    Returns
    -------
    np.ndarray
        Element-wise ``1 / (1 + exp(-x))``.
    """
    return np.where(
        x >= 0,
        1.0 / (1.0 + np.exp(-x)),
        np.exp(x) / (1.0 + np.exp(x)),
    )


def linear_predictor(
    X: np.ndarray,
    beta: np.ndarray,
    intercept: float = 0.0,
) -> np.ndarray:
    """Compute the linear predictor ``eta = intercept + X @ beta``.

    Parameters
    ----------
    X : np.ndarray
        Design matrix of shape ``(n_agents, n_terms)``.
    beta : np.ndarray
        Coefficient vector of shape ``(n_terms,)``.
    intercept : float
        Scalar intercept (0 for ordinal models).

    Returns
    -------
    np.ndarray
        Linear predictor of shape ``(n_agents,)``.
    """
    return intercept + X @ beta


def predict_binary(
    eta: np.ndarray,
    *,
    deterministic: bool = False,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Generate binary outcomes from ``Bernoulli(sigma(eta))``.

    Parameters
    ----------
    eta : np.ndarray
        Linear predictor of shape ``(n,)``.
    deterministic : bool
        If *True*, return expected probabilities instead of sampling.
    rng : np.random.Generator or None
        Random generator for stochastic mode.

    Returns
    -------
    np.ndarray
        ``float64`` array: probabilities if *deterministic*,
        binary 0/1 otherwise.  Shape ``(n,)``.
    """
    p = sigmoid(eta)
    if deterministic:
        return p
    if rng is None:
        rng = np.random.default_rng()
    return rng.binomial(1, p).astype(np.float64)


def predict_ordinal(
    eta: np.ndarray,
    thresholds: np.ndarray,
    *,
    deterministic: bool = False,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Generate ordinal outcomes from cumulative-logit model.

    Uses the cumulative link (proportional odds) formulation::

        P(Y <= j) = sigmoid(threshold_j - eta)

    Category probabilities are obtained by differencing cumulative
    probabilities.

    Parameters
    ----------
    eta : np.ndarray
        Linear predictor of shape ``(n,)``.
    thresholds : np.ndarray
        Sorted cut-points of shape ``(J-1,)`` where ``J`` is the
        number of response categories.
    deterministic : bool
        If *True*, return the expected (mean) ordinal value instead
        of sampling.
    rng : np.random.Generator or None
        Random generator for stochastic mode.

    Returns
    -------
    np.ndarray
        Sampled integer categories (1-indexed) or expected means.
        Shape ``(n,)``.
    """
    n = eta.shape[0]
    J = len(thresholds) + 1  # number of categories

    # Cumulative probabilities: P(Y <= j) for j = 1..J-1
    # Shape: (n, J-1)
    cum_probs = sigmoid(thresholds[np.newaxis, :] - eta[:, np.newaxis])

    # Category probabilities by differencing
    # p_j = P(Y <= j) - P(Y <= j-1)
    # with P(Y <= 0) = 0 and P(Y <= J) = 1
    cat_probs = np.empty((n, J), dtype=np.float64)
    cat_probs[:, 0] = cum_probs[:, 0]
    for j in range(1, J - 1):
        cat_probs[:, j] = cum_probs[:, j] - cum_probs[:, j - 1]
    cat_probs[:, J - 1] = 1.0 - cum_probs[:, J - 2]

    # Clip numerical artefacts
    cat_probs = np.clip(cat_probs, 0.0, 1.0)
    # Re-normalise rows
    row_sums = cat_probs.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums > 0, row_sums, 1.0)
    cat_probs /= row_sums

    if deterministic:
        # Expected value: sum_j (j * p_j),  j = 1..J
        categories = np.arange(1, J + 1, dtype=np.float64)
        return cat_probs @ categories

    # Stochastic: multinomial sampling
    if rng is None:
        rng = np.random.default_rng()

    # Vectorised multinomial via cumulative sum + uniform draw
    u = rng.uniform(size=n)
    cum = np.cumsum(cat_probs, axis=1)
    # Find first category where cum_prob >= u
    sampled = np.argmax(cum >= u[:, np.newaxis], axis=1) + 1
    return sampled.astype(np.float64)


# =========================================================================
# 2.  Design-matrix assembly
# =========================================================================


def assemble_design_matrix(
    agents: pl.DataFrame,
    model: BackboneModel,
) -> np.ndarray:
    """Build the ``(n_agents, n_terms)`` design matrix for *model*.

    Extracts columns from the agent DataFrame in the exact order
    required by the model's coefficient vector.  Country dummy
    variables (patsy-style ``C(country, …)[T.XX]``) are built
    on the fly from the ``country`` column.

    Parameters
    ----------
    agents : pl.DataFrame
        Current agent state (all required columns present).
    model : BackboneModel
        The backbone model whose terms determine column order.

    Returns
    -------
    np.ndarray
        Design matrix of shape ``(n_agents, len(model.terms))``.

    Raises
    ------
    ValueError
        If a required column is missing and cannot be derived.
    """
    X = _assemble_terms_matrix(agents, model.terms)

    if model.term_means is not None and model.term_stds is not None:
        X = (X - model.term_means[np.newaxis, :]) / model.term_stds[np.newaxis, :]
    return X


def _extract_country_code(term: str) -> str:
    """Parse country code from a patsy-style dummy term.

    Parameters
    ----------
    term : str
        E.g. ``"C(country, Treatment(reference='IT'))[T.UK]"``.

    Returns
    -------
    str
        Two-letter country code, e.g. ``"UK"``.
    """
    # Extract the part after "[T." and before "]"
    start = term.index("[T.") + 3
    end = term.index("]", start)
    return term[start:end]


# =========================================================================
# 3.  Outcome-level prediction
# =========================================================================


def predict_outcome(
    agents: pl.DataFrame,
    model: BackboneModel,
    *,
    deterministic: bool = False,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Predict a single outcome for all agents.

    Parameters
    ----------
    agents : pl.DataFrame
        Agent state DataFrame.
    model : BackboneModel
        The fitted backbone model for this outcome.
    deterministic : bool
        If *True*, return expected values (probabilities or means).
    rng : np.random.Generator or None
        Random generator for stochastic prediction.

    Returns
    -------
    np.ndarray
        Predicted values of shape ``(n_agents,)``.
    """
    X = assemble_design_matrix(agents, model)
    eta = linear_predictor(X, model.coefficients, intercept=model.intercept or 0.0)

    if model.model_type == "binary":
        return predict_binary(eta, deterministic=deterministic, rng=rng)
    else:
        assert model.thresholds is not None
        return predict_ordinal(
            eta,
            model.thresholds,
            deterministic=deterministic,
            rng=rng,
        )


def predict_all_outcomes(
    agents: pl.DataFrame,
    models: dict[str, BackboneModel],
    levers_cfg: LeversConfig,
    *,
    deterministic: bool = False,
    rng: np.random.Generator | None = None,
    tick: int = 0,
    choice_signals: list[SignalSpec] | None = None,
    target_mask: np.ndarray | None = None,
    norm_channel_weights: NormChannelWeights | None = None,
) -> dict[str, np.ndarray]:
    """Predict all ABM outcomes for the current agent state.

    Parameters
    ----------
    agents : pl.DataFrame
        Agent state DataFrame.
    models : dict[str, BackboneModel]
        Mapping from outcome name to its backbone model.
    levers_cfg : LeversConfig
        Lever configuration (used for wiring validation).
    deterministic : bool
        If *True*, return expected values.
    rng : np.random.Generator or None
        Random generator for stochastic prediction.

    Returns
    -------
    dict[str, np.ndarray]
        Mapping from outcome name to prediction array.

    Raises
    ------
    KeyError
        If an outcome in *levers_cfg* has no loaded model.
    """
    if choice_signals:
        agents = apply_signals_to_frame(
            agents,
            choice_signals,
            tick,
            target_mask=target_mask,
        )

    agents = _with_norm_composite(agents, norm_channel_weights)
    agents = _with_fivec_composite(agents)

    ordered_pairs: list[tuple[str, BackboneModel]] = []
    for outcome_name, spec in levers_cfg.outcomes.items():
        if outcome_name not in models:
            msg = f"No model loaded for outcome '{outcome_name}' (expected model_ref='{spec.model_ref}')."
            raise KeyError(msg)
        ordered_pairs.append((outcome_name, models[outcome_name]))

    ordered_models = tuple(model for _, model in ordered_pairs)
    shared_terms = _ordered_unique_terms(ordered_models)
    shared_x = _assemble_terms_matrix(agents, shared_terms)
    term_idx = {term: idx for idx, term in enumerate(shared_terms)}

    results: dict[str, np.ndarray] = {}
    for outcome_name, model in ordered_pairs:
        model_term_idx = np.fromiter((term_idx[t] for t in model.terms), dtype=np.int64, count=len(model.terms))
        x_model = shared_x[:, model_term_idx] if model_term_idx.size > 0 else np.empty((agents.height, 0), np.float64)
        if model.term_means is not None and model.term_stds is not None:
            x_model = (x_model - model.term_means[np.newaxis, :]) / model.term_stds[np.newaxis, :]
        eta = linear_predictor(x_model, model.coefficients, intercept=model.intercept or 0.0)

        if model.model_type == "binary":
            results[outcome_name] = predict_binary(eta, deterministic=deterministic, rng=rng)
        else:
            assert model.thresholds is not None
            results[outcome_name] = predict_ordinal(
                eta,
                model.thresholds,
                deterministic=deterministic,
                rng=rng,
            )
        logger.debug(
            "Predicted '{outcome}': mean={mean:.4f}",
            outcome=outcome_name,
            mean=float(results[outcome_name].mean()),
        )
    return results
