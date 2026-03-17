"""Bounded state-transition kernels for mutable constructs.

Each mutable engine (trust, constraints, risk, stakes, norms, etc.)
gets a dedicated update kernel that transforms the current state based
on scenario-driven external signals ``e_S(t)`` and (optionally)
social-network aggregated peer influence.

Interface rule: update kernels consume ``(s_i(t), e_S(t), social_i(t))``;
``e_Y(t)`` must **not** be consumed in state updates.

All updates are clipped to declared construct bounds from
:mod:`beam_abm.abm.state_schema`.

Examples
--------
>>> from beam_abm.abm.state_update import SignalSpec
>>> sig = SignalSpec(target_column="vaccine_risk_avg", intensity=1.0, signal_type="pulse")
>>> round(sig.target_offset(0), 3)
1.0
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import polars as pl
from loguru import logger

from beam_abm.abm.belief_update_kernel import BeliefUpdateKernel
from beam_abm.abm.belief_update_surrogate import BeliefUpdateSurrogate
from beam_abm.abm.message_contracts import (
    NormChannelWeights,
    norm_channel_columns,
    risk_channel_columns,
    trust_channel_columns,
)
from beam_abm.abm.state_schema import CONSTRUCT_BOUNDS, MUTABLE_COLUMNS

__all__ = [
    "compute_observed_norm_exposure",
    "compute_observed_norm_fixed_point_share",
    "SignalSpec",
    "apply_social_norm_update",
    "apply_signals_to_frame",
    "apply_all_state_updates",
    "clip_to_bounds",
    "compute_target_mask",
]

FIVEC_COMPONENT_COLUMNS: tuple[str, ...] = (
    "fivec_confidence",
    "fivec_low_complacency",
    "fivec_low_constraints",
    "fivec_collective_responsibility",
)

SLOW_RELAXATION_COLUMNS: frozenset[str] = frozenset(
    {
        # Trust and norm channels.
        "social_norms_descriptive_vax",
        "social_norms_descriptive_npi",
        "group_trust_vax",
        "institutional_trust_avg",
        "covax_legitimacy_scepticism_idx",
        # 5C and risk-attitude channels with slower persistence.
        "fivec_confidence",
        "fivec_collective_responsibility",
        "vaccine_risk_avg",
    }
)

NORM_SHARE_HAT_COLUMN_VAX = "norm_share_hat_vax"
NORM_SHARE_HAT_COLUMN_NPI = "norm_share_hat_npi"
NOISE_Z_CLIP_DEFAULT = 3.0


def _draw_bounded_zero_mean_noise(
    *,
    shape: tuple[int, ...],
    noise_std: float,
    rng: np.random.Generator,
    noise_z_clip: float = NOISE_Z_CLIP_DEFAULT,
) -> np.ndarray:
    """Draw additive noise with bounded tails and zero cross-agent mean."""
    sigma = float(np.clip(noise_std, 0.0, np.inf))
    if sigma <= 0.0:
        return np.zeros(shape, dtype=np.float64)

    clip_z = float(noise_z_clip)
    if not np.isfinite(clip_z) or clip_z <= 0.0:
        clip_z = NOISE_Z_CLIP_DEFAULT
    cap = clip_z * sigma

    noise = rng.normal(0.0, sigma, size=shape).astype(np.float64, copy=False)
    noise = np.clip(noise, -cap, cap)
    if noise.size > 1:
        noise = noise - float(np.mean(noise))
    return noise


# =========================================================================
# 1.  Signal specification
# =========================================================================


@dataclass(frozen=True, slots=True)
class SignalSpec:
    """Specification for an external signal applied to a runtime construct.

    Parameters
    ----------
    target_column : str
        Runtime column to perturb. This can be a mutable construct or an
        exogenous precursor column used by state-reference attractors.
    intensity : float
        Magnitude of the shock in standardised units.
    signal_type : str
        One of ``"pulse"``, ``"impulse"``, ``"campaign"``,
        ``"sustained"``, ``"regime_shift"``.
    start_tick : int
        Tick at which the signal activates.
    end_tick : int or None
        Tick at which the signal deactivates (``None`` = indefinite).
    signal_id : str or None
        Optional LLM belief-update signal identifier.
    """

    target_column: str
    intensity: float
    signal_type: str = "pulse"
    start_tick: int = 0
    end_tick: int | None = None
    signal_id: str | None = None

    def __post_init__(self) -> None:
        """Validate target column."""
        if not self.target_column or not self.target_column.strip():
            msg = "Signal target column must be a non-empty string."
            raise ValueError(msg)
        allowed_types = {"pulse", "impulse", "campaign", "sustained", "regime_shift"}
        if self.signal_type not in allowed_types:
            msg = f"Unsupported signal_type '{self.signal_type}'. Expected one of {sorted(allowed_types)}."
            raise ValueError(msg)
        if self.start_tick < 0:
            msg = "start_tick must be >= 0."
            raise ValueError(msg)
        if self.end_tick is not None and self.end_tick <= self.start_tick:
            msg = "end_tick must be greater than start_tick (exclusive window end)."
            raise ValueError(msg)

    def is_active(self, tick: int) -> bool:
        """Check whether this signal is active at *tick*.

        Parameters
        ----------
        tick : int
            Current simulation tick.

        Returns
        -------
        bool
        """
        if tick < self.start_tick:
            return False
        # End tick is exclusive: active for start_tick <= tick < end_tick.
        if self.end_tick is not None and tick >= self.end_tick:
            return False
        return True

    def target_offset(self, tick: int) -> float:
        """Compute the target-shift offset at *tick*.

        Parameters
        ----------
        tick : int
            Current simulation tick.

        Returns
        -------
        float
            Target offset at tick. Zero if inactive.
        """
        if not self.is_active(tick):
            return 0.0

        elapsed = tick - self.start_tick
        if self.signal_type == "impulse":
            return self.intensity if elapsed == 0 else 0.0
        if self.signal_type == "pulse":
            # Pulse shocks are one-shot perturbations; persistence is endogenous.
            return self.intensity if elapsed == 0 else 0.0
        if self.signal_type in {"sustained", "campaign", "regime_shift"}:
            return self.intensity
        return 0.0


# =========================================================================
# 2.  Low-level update functions (vectorised)
# =========================================================================


def clip_to_bounds(values: np.ndarray, column: str) -> np.ndarray:
    """Clip *values* to the declared bounds for *column*.

    Parameters
    ----------
    values : np.ndarray
        Array to clip (modified in place and returned).
    column : str
        Construct column name.

    Returns
    -------
    np.ndarray
        Clipped array.
    """
    lo, hi = CONSTRUCT_BOUNDS.get(column, (-np.inf, np.inf))
    return np.clip(values, lo, hi)


def compute_target_mask(agents: pl.DataFrame, targeting: Literal["uniform", "targeted"]) -> np.ndarray | None:
    """Compute a boolean mask for targeted scenarios.

    Targeting is localised by default:
    1. score agents by media-exposure/trust composite,
    2. pick one seed country with highest mean score (if available),
    3. target the top exposure quartile inside that country.

    Returns ``None`` when targeting is uniform or when no targeting
    columns exist.
    """
    if targeting == "uniform":
        return None

    candidate_cols = [
        "duration_socialmedia",
        "trust_social",
        "trust_onlineinfo",
        "duration_infomedia",
    ]
    present = [c for c in candidate_cols if c in agents.columns]
    if not present:
        logger.warning("Targeting requested but no ecosystem columns are available.")
        return None

    values = agents.select(present).to_numpy().astype(np.float64)
    means = np.nanmean(values, axis=0)
    stds = np.nanstd(values, axis=0)
    stds = np.where(stds > 0, stds, 1.0)
    z = (values - means) / stds
    composite = np.nanmean(z, axis=1)
    local_mask = np.ones(agents.height, dtype=bool)
    if "country" in agents.columns:
        countries = agents["country"].to_numpy().astype(str)
        unique_countries = np.unique(countries)
        country_means: dict[str, float] = {}
        for country in unique_countries:
            idx = countries == country
            if not np.any(idx):
                continue
            country_means[country] = float(np.nanmean(composite[idx]))
        if country_means:
            seed_country = max(country_means.items(), key=lambda item: item[1])[0]
            local_mask = countries == seed_country

    local_scores = composite[local_mask]
    if local_scores.size == 0:
        return None
    threshold = np.nanpercentile(local_scores, 75)
    target_mask = np.zeros(agents.height, dtype=bool)
    target_mask[local_mask] = local_scores >= threshold
    return target_mask


def apply_social_norm_update(
    current: np.ndarray,
    peer_mean: np.ndarray,
    gain: float,
    noise_std: float = 0.01,
    noise_z_clip: float = NOISE_Z_CLIP_DEFAULT,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Update norms via social aggregation with bounded noise.

    ``s(t+1) = s(t) + gain * (peer_mean - s(t)) + noise``

    Parameters
    ----------
    current : np.ndarray
        Current norm values of shape ``(n,)``.
    peer_mean : np.ndarray
        Network-aggregated peer norm values of shape ``(n,)``.
    gain : float
        Step size for norm convergence (lambda in methodology).
    noise_std : float
        Standard deviation of additive noise.
    rng : np.random.Generator or None
        Random generator.

    Returns
    -------
    np.ndarray
        Updated norm values.
    """
    if rng is None:
        rng = np.random.default_rng()
    bounded_gain = float(np.clip(gain, 0.0, 1.0))
    noise = _draw_bounded_zero_mean_noise(
        shape=current.shape,
        noise_std=noise_std,
        rng=rng,
        noise_z_clip=noise_z_clip,
    )
    return current + bounded_gain * (peer_mean - current) + noise


def apply_observed_norm_update(
    current: np.ndarray,
    observed_share: np.ndarray,
    susceptibility: np.ndarray,
    threshold: np.ndarray,
    column: str,
    slope: float,
    noise_std: float,
    noise_z_clip: float = NOISE_Z_CLIP_DEFAULT,
    baseline_descriptive_norm: np.ndarray | None = None,
    baseline_exposure: np.ndarray | None = None,
    baseline_share_anchor: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Update norm perception from observed neighbor behavior.

    Complex contagion is approximated with a logistic response to
    observed peer adoption share.
    """
    if rng is None:
        rng = np.random.default_rng()
    lo, hi = CONSTRUCT_BOUNDS.get(column, (1.0, 7.0))
    span = hi - lo
    anchored_share_mode = baseline_descriptive_norm is not None and baseline_share_anchor is not None
    anchored_exposure_mode = baseline_descriptive_norm is not None and baseline_exposure is not None

    if anchored_share_mode:
        baseline_norm = np.asarray(baseline_descriptive_norm, dtype=np.float64)
        baseline_share = np.asarray(baseline_share_anchor, dtype=np.float64)
        if baseline_norm.shape != current.shape:
            msg = (
                "baseline_descriptive_norm must have same shape as current: "
                f"got {baseline_norm.shape} vs {current.shape}."
            )
            raise ValueError(msg)
        if baseline_share.shape != current.shape:
            msg = (
                f"baseline_share_anchor must have same shape as current: got {baseline_share.shape} vs {current.shape}."
            )
            raise ValueError(msg)
        if not np.isfinite(baseline_norm).all():
            msg = "baseline_descriptive_norm contains null/NaN/inf values."
            raise ValueError(msg)
        if not np.isfinite(baseline_share).all():
            msg = "baseline_share_anchor contains null/NaN/inf values."
            raise ValueError(msg)
        baseline_share = np.clip(baseline_share, 0.0, 1.0)
        obs = np.clip(np.asarray(observed_share, dtype=np.float64), 0.0, 1.0)
        target = np.clip(baseline_norm + span * (obs - baseline_share), lo, hi)
    elif anchored_exposure_mode:
        exposure = compute_observed_norm_exposure(
            observed_share=observed_share,
            threshold=threshold,
            slope=slope,
        )
        baseline_norm = np.asarray(baseline_descriptive_norm, dtype=np.float64)
        baseline_exp = np.asarray(baseline_exposure, dtype=np.float64)
        if baseline_norm.shape != current.shape:
            msg = (
                "baseline_descriptive_norm must have same shape as current: "
                f"got {baseline_norm.shape} vs {current.shape}."
            )
            raise ValueError(msg)
        if baseline_exp.shape != current.shape:
            msg = f"baseline_exposure must have same shape as current: got {baseline_exp.shape} vs {current.shape}."
            raise ValueError(msg)
        if not np.isfinite(baseline_norm).all():
            msg = "baseline_descriptive_norm contains null/NaN/inf values."
            raise ValueError(msg)
        if not np.isfinite(baseline_exp).all():
            msg = "baseline_exposure contains null/NaN/inf values."
            raise ValueError(msg)
        target = np.clip(baseline_norm + span * (exposure - baseline_exp), lo, hi)
    else:
        exposure = compute_observed_norm_exposure(
            observed_share=observed_share,
            threshold=threshold,
            slope=slope,
        )
        target = lo + span * exposure

    bounded_susceptibility = np.clip(np.asarray(susceptibility, dtype=np.float64), 0.0, 1.0)
    noise = _draw_bounded_zero_mean_noise(
        shape=current.shape,
        noise_std=noise_std,
        rng=rng,
        noise_z_clip=noise_z_clip,
    )
    updated = current + bounded_susceptibility * (target - current) + noise
    return np.clip(updated, lo, hi)


def compute_observed_norm_exposure(
    observed_share: np.ndarray,
    threshold: np.ndarray,
    *,
    slope: float,
    eps: float = 1e-6,
) -> np.ndarray:
    """Map observed adoption share to descriptive-norm exposure intensity.

    Exposure is the logistic response used by observed descriptive norms:
    ``sigmoid(slope * (logit(share) - logit(theta)))``.
    """
    eps_val = float(np.clip(eps, 1e-12, 1e-3))
    slope_val = float(slope)
    if not np.isfinite(slope_val) or slope_val <= 0.0:
        msg = f"slope must be finite and > 0. Got {slope_val}."
        raise ValueError(msg)

    share = np.clip(np.asarray(observed_share, dtype=np.float64), eps_val, 1.0 - eps_val)
    theta = np.clip(np.asarray(threshold, dtype=np.float64), eps_val, 1.0 - eps_val)
    if share.shape != theta.shape:
        msg = f"observed_share and threshold must have same shape: got {share.shape} vs {theta.shape}."
        raise ValueError(msg)
    if not np.isfinite(share).all() or not np.isfinite(theta).all():
        msg = "observed_share and threshold must be finite arrays."
        raise ValueError(msg)

    share_logit = np.log(share / (1.0 - share))
    theta_logit = np.log(theta / (1.0 - theta))
    return 1.0 / (1.0 + np.exp(-slope_val * (share_logit - theta_logit)))


def compute_observed_norm_fixed_point_share(
    descriptive_norm: np.ndarray,
    threshold: np.ndarray,
    *,
    column: str = "social_norms_descriptive_vax",
    slope: float,
    eps: float = 1e-6,
) -> np.ndarray:
    """Infer observed-share values that keep descriptive norms at fixed point.

    Solves the inverse of the logistic target map used by
    :func:`apply_observed_norm_update` so that, in a noise-free update,
    ``target == descriptive_norm`` for each agent.
    """
    if descriptive_norm.shape != threshold.shape:
        msg = (
            "descriptive_norm and threshold must have the same shape: "
            f"got {descriptive_norm.shape} vs {threshold.shape}."
        )
        raise ValueError(msg)

    slope_val = float(slope)
    if not np.isfinite(slope_val) or slope_val <= 0.0:
        msg = f"slope must be finite and > 0. Got {slope_val}."
        raise ValueError(msg)

    eps_val = float(np.clip(eps, 1e-12, 1e-3))
    lo, hi = CONSTRUCT_BOUNDS.get(column, (1.0, 7.0))
    span = float(hi - lo)
    if not np.isfinite(span) or span <= 0.0:
        msg = f"Invalid bounds for '{column}': ({lo}, {hi})."
        raise ValueError(msg)

    descriptive = descriptive_norm.astype(np.float64, copy=False)
    theta = np.clip(threshold.astype(np.float64, copy=False), eps_val, 1.0 - eps_val)
    exposure = np.clip((descriptive - float(lo)) / span, eps_val, 1.0 - eps_val)
    exposure_logit = np.log(exposure / (1.0 - exposure))
    theta_logit = np.log(theta / (1.0 - theta))
    observed_share_logit = theta_logit + (exposure_logit / slope_val)
    observed_share = 1.0 / (1.0 + np.exp(-observed_share_logit))
    return np.clip(observed_share, 0.0, 1.0)


# =========================================================================
# 3.  Generic construct updater
# =========================================================================


def update_generic_construct(
    current: np.ndarray,
    baseline: np.ndarray,
    column: str,
    signals: list[SignalSpec],
    tick: int,
    relaxation_rate: float,
    target_state: np.ndarray | None = None,
    peer_mean: np.ndarray | None = None,
    social_gain: float = 0.0,
    endogenous_offset: np.ndarray | None = None,
    signal_offset: np.ndarray | None = None,
    signal_mask: np.ndarray | None = None,
    social_noise_std: float = 0.01,
    social_noise_z_clip: float = NOISE_Z_CLIP_DEFAULT,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Apply signals and optional social influence to any mutable construct.

    Parameters
    ----------
    current : np.ndarray
        Current values for *column*.
    column : str
        Mutable construct name.
    signals : list[SignalSpec]
        All active signals (filtered internally by target).
    tick : int
        Current simulation tick.
    peer_mean : np.ndarray or None
        Network-aggregated peer values.
    norm_gain : float
        Social norm update gain.
    rng : np.random.Generator or None
        Random generator.

    Returns
    -------
    np.ndarray
        Updated values, clipped.
    """
    updated = current.copy()
    target = target_state.copy() if target_state is not None else baseline.copy()
    if target.shape != updated.shape:
        msg = f"target_state shape mismatch for '{column}': got {target.shape}, expected {updated.shape}."
        raise ValueError(msg)

    if endogenous_offset is not None:
        target = target + endogenous_offset

    if signal_offset is not None:
        if signal_mask is None:
            target = target + signal_offset
        else:
            target[signal_mask] = target[signal_mask] + signal_offset[signal_mask]

    for sig in signals:
        if sig.target_column != column:
            continue
        eff = sig.target_offset(tick)
        if eff == 0.0:
            continue
        if signal_mask is None:
            target = target + eff
        else:
            target[signal_mask] = target[signal_mask] + eff

    bounded_relaxation = float(np.clip(relaxation_rate, 0.0, 1.0))
    updated = updated + bounded_relaxation * (target - updated)

    if peer_mean is not None and social_gain > 0:
        updated = apply_social_norm_update(
            updated,
            peer_mean,
            social_gain,
            noise_std=social_noise_std,
            noise_z_clip=social_noise_z_clip,
            rng=rng,
        )

    return clip_to_bounds(updated, column)


def apply_signals_to_frame(
    agents: pl.DataFrame,
    signals: list[SignalSpec],
    tick: int,
    *,
    target_mask: np.ndarray | None = None,
) -> pl.DataFrame:
    """Apply signals to a copy of the agent frame (choice-context use)."""
    if not signals:
        return agents

    active_columns: set[str] = set()
    for sig in signals:
        if sig.is_active(tick):
            active_columns.add(sig.target_column)

    if not active_columns:
        return agents

    update_dict: dict[str, np.ndarray] = {}
    for col in active_columns:
        if col not in agents.columns:
            continue
        current = agents[col].to_numpy().copy()
        if target_mask is None:
            updated = current.copy()
            for sig in signals:
                if sig.target_column != col:
                    continue
                eff = sig.target_offset(tick)
                if eff != 0.0:
                    updated = updated + eff
            updated = clip_to_bounds(updated, col)
        else:
            updated = current.copy()
            for sig in signals:
                if sig.target_column != col:
                    continue
                eff = sig.target_offset(tick)
                if eff != 0.0:
                    updated[target_mask] = updated[target_mask] + eff
            updated = clip_to_bounds(updated, col)
        update_dict[col] = updated

    if not update_dict:
        return agents

    return agents.with_columns([pl.Series(name=col, values=vals) for col, vals in update_dict.items()])


# =========================================================================
# 4.  Master update dispatcher
# =========================================================================


def apply_all_state_updates(
    agents: pl.DataFrame,
    signals: list[SignalSpec],
    tick: int,
    attractor_targets: dict[str, np.ndarray] | None = None,
    peer_means: dict[str, np.ndarray] | None = None,
    k_social: float = 0.0,
    social_weight_norm: float = 0.34,
    social_weight_trust: float = 0.33,
    social_weight_risk: float = 0.33,
    rng: np.random.Generator | None = None,
    target_mask: np.ndarray | None = None,
    belief_kernel: BeliefUpdateKernel | BeliefUpdateSurrogate | None = None,
    baseline_agents: pl.DataFrame | None = None,
    observed_norm_share: np.ndarray | None = None,
    observed_norm_share_npi: np.ndarray | None = None,
    social_susceptibility: np.ndarray | None = None,
    norm_threshold: np.ndarray | None = None,
    norm_slope: float = 8.0,
    baseline_descriptive_norm: np.ndarray | None = None,
    baseline_norm_exposure: np.ndarray | None = None,
    baseline_share_hat: np.ndarray | None = None,
    baseline_observed_share: np.ndarray | None = None,
    baseline_descriptive_norm_npi: np.ndarray | None = None,
    baseline_norm_exposure_npi: np.ndarray | None = None,
    baseline_share_hat_npi: np.ndarray | None = None,
    baseline_observed_share_npi: np.ndarray | None = None,
    endogenous_offsets: dict[str, np.ndarray] | None = None,
    signal_overrides: dict[str, np.ndarray] | None = None,
    signal_absolute_targets: dict[str, np.ndarray] | None = None,
    norm_channel_weights: NormChannelWeights | None = None,
    social_noise_std: float = 0.01,
    social_noise_z_clip: float = NOISE_Z_CLIP_DEFAULT,
    norm_threshold_min: float = 0.05,
    norm_threshold_max: float = 0.95,
    state_relaxation_fast: float = 0.2,
    state_relaxation_slow: float = 0.12,
    frozen_columns: set[str] | frozenset[str] | None = None,
) -> pl.DataFrame:
    """Apply one tick of state updates to all mutable constructs.

    This is the master entry-point called by the simulator each step.
    It iterates over all mutable columns, applies the matching
    signals, and returns an updated DataFrame.

    Parameters
    ----------
    agents : pl.DataFrame
        Current agent state.
    signals : list[SignalSpec]
        All active signals for this tick.
    tick : int
        Current simulation tick.
    attractor_targets : dict[str, np.ndarray] or None
        Optional per-construct survey-anchored targets ``S* = f(X_t)``.
    peer_means : dict[str, np.ndarray] or None
        Network-aggregated peer values keyed by construct name.
    norm_gain : float
        Social norm update gain (lambda).
    rng : np.random.Generator or None
        Random generator.

    Returns
    -------
    pl.DataFrame
        Agent state with mutable columns updated.
    """
    # Identify which columns need updates (active or recovering from prior shifts)
    effective_k_social = float(np.clip(k_social, 0.0, 1.0))
    frozen_set = set(frozen_columns or ())
    frozen_set = {col for col in frozen_set if col in MUTABLE_COLUMNS and col in agents.columns}

    active_columns: set[str] = {sig.target_column for sig in signals}

    if attractor_targets:
        active_columns.update(attractor_targets.keys())

    # Also add columns with peer influence
    if peer_means:
        active_columns.update(peer_means.keys())

    if signal_overrides:
        active_columns.update(signal_overrides.keys())

    if signal_absolute_targets:
        active_columns.update(signal_absolute_targets.keys())

    # Endogenous feedback offsets (e.g., incidence->risk) must activate their
    # target columns even when no external signal/peer/attractor is present.
    if endogenous_offsets:
        active_columns.update(endogenous_offsets.keys())

    if observed_norm_share is not None and "social_norms_descriptive_vax" in agents.columns:
        active_columns.add("social_norms_descriptive_vax")
    if observed_norm_share_npi is not None and "social_norms_descriptive_npi" in agents.columns:
        active_columns.add("social_norms_descriptive_npi")

    # Descriptive norms are channel-specific states and must not be moved by
    # generic attractor/relaxation updates. They are updated only via the
    # observed-norm channel below (plus explicit direct signals).
    if "social_norms_descriptive_vax" in agents.columns and observed_norm_share is None:
        active_columns.discard("social_norms_descriptive_vax")
    if "social_norms_descriptive_npi" in agents.columns and observed_norm_share_npi is None:
        active_columns.discard("social_norms_descriptive_npi")

    # Composite norm is a deterministic derived construct from
    # descriptive channels and should never be updated as an
    # independent mutable state.
    active_columns.discard("social_norms_vax_avg")
    # 5C pro-vax is a deterministic composite of four mutable components.
    active_columns.discard("fivec_pro_vax_idx")
    if frozen_set:
        active_columns.difference_update(frozen_set)

    has_norm_channel_vax = "social_norms_descriptive_vax" in agents.columns
    has_fivec_components = all(col in agents.columns for col in FIVEC_COMPONENT_COLUMNS)

    if (
        not active_columns
        and not peer_means
        and observed_norm_share is None
        and observed_norm_share_npi is None
        and not endogenous_offsets
        and not has_norm_channel_vax
        and not has_fivec_components
    ):
        return agents

    # Build update expressions
    update_dict: dict[str, np.ndarray] = {}
    norm_threshold_arr: np.ndarray | None = None
    perceived_norm_shares: dict[str, np.ndarray] = {}

    observed_inputs: dict[str, np.ndarray | None] = {
        "social_norms_descriptive_vax": observed_norm_share,
        "social_norms_descriptive_npi": observed_norm_share_npi,
    }
    share_hat_columns = {
        "social_norms_descriptive_vax": NORM_SHARE_HAT_COLUMN_VAX,
        "social_norms_descriptive_npi": NORM_SHARE_HAT_COLUMN_NPI,
    }
    baseline_share_hats = {
        "social_norms_descriptive_vax": baseline_share_hat,
        "social_norms_descriptive_npi": baseline_share_hat_npi,
    }
    baseline_observed_shares = {
        "social_norms_descriptive_vax": baseline_observed_share,
        "social_norms_descriptive_npi": baseline_observed_share_npi,
    }

    for desc_col, observed_input in observed_inputs.items():
        if observed_input is None or desc_col not in agents.columns:
            continue

        observed = np.asarray(observed_input, dtype=np.float64)
        if observed.shape != (agents.height,):
            msg = f"{desc_col} observed share must have shape (n_agents,). Got {observed.shape} for n_agents={agents.height}."
            raise ValueError(msg)
        if not np.isfinite(observed).all():
            msg = f"{desc_col} observed share contains null/NaN/inf values."
            raise ValueError(msg)

        if norm_threshold is None:
            msg = "Missing required norm_threshold array for observed norm updates."
            raise ValueError(msg)
        if norm_threshold_arr is None:
            norm_threshold_arr = np.asarray(norm_threshold, dtype=np.float64)
            if norm_threshold_arr.shape != observed.shape:
                msg = (
                    "norm_threshold must have the same shape as observed norm share: "
                    f"got {norm_threshold_arr.shape} vs {observed.shape}."
                )
                raise ValueError(msg)
            if not np.isfinite(norm_threshold_arr).all():
                msg = "norm_threshold contains null/NaN/inf values."
                raise ValueError(msg)
            norm_threshold_arr = np.clip(norm_threshold_arr, norm_threshold_min, norm_threshold_max)

        hat_column = share_hat_columns[desc_col]
        if hat_column in agents.columns:
            perceived_norm_share = agents[hat_column].cast(pl.Float64).to_numpy().astype(np.float64)
            if perceived_norm_share.shape != observed.shape:
                msg = f"{hat_column} must have shape {observed.shape}; got {perceived_norm_share.shape}."
                raise ValueError(msg)
            if not np.isfinite(perceived_norm_share).all():
                msg = f"{hat_column} contains null/NaN/inf values."
                raise ValueError(msg)
            perceived_norm_share = np.clip(perceived_norm_share, 0.0, 1.0)
        else:
            descriptive = agents[desc_col].cast(pl.Float64).to_numpy().astype(np.float64)
            descriptive = np.nan_to_num(descriptive, nan=4.0, posinf=7.0, neginf=1.0)
            perceived_norm_share = compute_observed_norm_fixed_point_share(
                descriptive_norm=descriptive,
                threshold=norm_threshold_arr,
                column=desc_col,
                slope=float(norm_slope),
            )

        if hat_column not in frozen_set:
            learning_rate = float(np.clip(state_relaxation_slow, 0.0, 1.0))
            baseline_hat = baseline_share_hats[desc_col]
            baseline_obs = baseline_observed_shares[desc_col]
            if baseline_hat is not None and baseline_obs is not None:
                obs_ref = np.asarray(baseline_obs, dtype=np.float64)
                hat_0 = np.asarray(baseline_hat, dtype=np.float64)
                delta_obs = observed - obs_ref
                delta_hat = perceived_norm_share - hat_0
                delta_hat_new = delta_hat + learning_rate * (delta_obs - delta_hat)
                perceived_norm_share = np.clip(hat_0 + delta_hat_new, 0.0, 1.0)
            else:
                perceived_norm_share = np.clip(
                    perceived_norm_share + learning_rate * (observed - perceived_norm_share),
                    0.0,
                    1.0,
                )
            update_dict[hat_column] = perceived_norm_share
        perceived_norm_shares[desc_col] = perceived_norm_share

    # Apply LLM belief-update kernel first, so parametric updates can skip
    if belief_kernel is not None:
        for sig in signals:
            if sig.signal_id is None or not sig.is_active(tick):
                continue
            intensity = sig.target_offset(tick)
            llm_updates = belief_kernel.apply(
                agents,
                signal_id=sig.signal_id,
                intensity=intensity,
                target_mask=target_mask,
            )
            update_dict.update(llm_updates)
    if frozen_set:
        for col in frozen_set:
            update_dict.pop(col, None)
    for col in active_columns:
        if col not in MUTABLE_COLUMNS:
            continue
        if col not in agents.columns:
            logger.warning("Mutable column '{col}' not found in agents.", col=col)
            continue

        if col in update_dict:
            continue

        current = agents[col].to_numpy().copy()
        baseline = (
            baseline_agents[col].to_numpy().copy()
            if baseline_agents is not None and col in baseline_agents.columns
            else current.copy()
        )
        peer = peer_means.get(col) if peer_means else None
        base_channel_gain = 0.0
        if col in norm_channel_columns():
            base_channel_gain = float(max(0.0, social_weight_norm))
        elif col in trust_channel_columns():
            base_channel_gain = float(max(0.0, social_weight_trust))
        elif col in risk_channel_columns():
            base_channel_gain = float(max(0.0, social_weight_risk))
        social_gain = float(np.clip(effective_k_social * max(0.0, base_channel_gain), 0.0, 1.0))
        endogenous_offset = None if endogenous_offsets is None else endogenous_offsets.get(col)
        signal_offset = None if signal_overrides is None else signal_overrides.get(col)
        target_state = None if attractor_targets is None else attractor_targets.get(col)

        relaxation_rate = float(
            np.clip(state_relaxation_slow if col in SLOW_RELAXATION_COLUMNS else state_relaxation_fast, 0.0, 1.0)
        )

        if col in {"social_norms_descriptive_vax", "social_norms_descriptive_npi"} and col in perceived_norm_shares:
            if social_susceptibility is None:
                msg = "Missing required social_susceptibility array for observed norm updates."
                raise ValueError(msg)
            if norm_threshold_arr is None:
                msg = "Missing required norm_threshold array for observed norm updates."
                raise ValueError(msg)
            perceived_norm_share = perceived_norm_shares[col]
            susceptibility_arr = np.asarray(social_susceptibility, dtype=np.float64)
            if susceptibility_arr.shape != perceived_norm_share.shape:
                msg = (
                    "social_susceptibility must have the same shape as observed norm share: "
                    f"got {susceptibility_arr.shape} vs {perceived_norm_share.shape}."
                )
                raise ValueError(msg)
            if not np.isfinite(susceptibility_arr).all():
                msg = "social_susceptibility contains null/NaN/inf values."
                raise ValueError(msg)
            baseline_descriptive = (
                baseline_descriptive_norm if col == "social_norms_descriptive_vax" else baseline_descriptive_norm_npi
            )
            baseline_exposure = (
                baseline_norm_exposure if col == "social_norms_descriptive_vax" else baseline_norm_exposure_npi
            )
            baseline_share_anchor = baseline_share_hats.get(col)
            if baseline_share_anchor is None:
                baseline_share_anchor = baseline_observed_shares.get(col)

            updated = apply_observed_norm_update(
                current=current,
                observed_share=perceived_norm_share,
                susceptibility=np.clip(
                    susceptibility_arr * float(np.clip(effective_k_social, 0.0, 1.0)),
                    0.0,
                    1.0,
                ),
                threshold=norm_threshold_arr,
                column=col,
                slope=float(norm_slope),
                noise_std=social_noise_std,
                noise_z_clip=social_noise_z_clip,
                baseline_descriptive_norm=baseline_descriptive,
                baseline_exposure=baseline_exposure,
                baseline_share_anchor=baseline_share_anchor,
                rng=rng,
            )
            if signal_offset is not None:
                if target_mask is None:
                    updated = updated + signal_offset
                else:
                    updated[target_mask] = updated[target_mask] + signal_offset[target_mask]
            for sig in signals:
                if sig.target_column != col:
                    continue
                eff = sig.target_offset(tick)
                if eff == 0.0:
                    continue
                if target_mask is None:
                    updated = updated + eff
                else:
                    updated[target_mask] = updated[target_mask] + eff
            updated = clip_to_bounds(updated, col)
            update_dict[col] = updated
            continue

        if target_mask is None:
            updated = update_generic_construct(
                current=current,
                baseline=baseline,
                column=col,
                signals=signals,
                tick=tick,
                relaxation_rate=relaxation_rate,
                target_state=target_state,
                peer_mean=peer,
                social_gain=social_gain,
                endogenous_offset=endogenous_offset,
                signal_offset=signal_offset,
                social_noise_std=social_noise_std,
                social_noise_z_clip=social_noise_z_clip,
                rng=rng,
            )
        else:
            updated = update_generic_construct(
                current=current,
                baseline=baseline,
                column=col,
                signals=signals,
                tick=tick,
                relaxation_rate=relaxation_rate,
                target_state=target_state,
                peer_mean=peer,
                social_gain=social_gain,
                endogenous_offset=endogenous_offset,
                signal_offset=signal_offset,
                signal_mask=target_mask,
                social_noise_std=social_noise_std,
                social_noise_z_clip=social_noise_z_clip,
                rng=rng,
            )

        update_dict[col] = updated

    if signal_absolute_targets:
        for col, absolute_target in signal_absolute_targets.items():
            if col in frozen_set:
                continue
            if col not in MUTABLE_COLUMNS or col not in agents.columns:
                continue
            target_arr = np.asarray(absolute_target, dtype=np.float64)
            if target_arr.shape != (agents.height,):
                msg = (
                    f"signal_absolute_targets['{col}'] must have shape (n_agents,), "
                    f"got {target_arr.shape} for n_agents={agents.height}."
                )
                raise ValueError(msg)
            if not np.isfinite(target_arr).all():
                msg = f"signal_absolute_targets['{col}'] contains null/NaN/inf values."
                raise ValueError(msg)
            target_arr = clip_to_bounds(target_arr, col)
            current = update_dict.get(col)
            if current is None:
                current = agents[col].cast(pl.Float64).to_numpy().astype(np.float64)
            if target_mask is None:
                current = target_arr
            else:
                current = np.asarray(current, dtype=np.float64).copy()
                current[target_mask] = target_arr[target_mask]
            update_dict[col] = clip_to_bounds(current, col)

    if has_norm_channel_vax:
        desc = update_dict.get("social_norms_descriptive_vax", agents["social_norms_descriptive_vax"].to_numpy())
        update_dict["social_norms_vax_avg"] = clip_to_bounds(
            desc,
            "social_norms_vax_avg",
        )

    if has_fivec_components:
        fivec_components = [update_dict.get(col, agents[col].to_numpy()) for col in FIVEC_COMPONENT_COLUMNS]
        fivec_composite = np.mean(np.vstack(fivec_components), axis=0)
        update_dict["fivec_pro_vax_idx"] = clip_to_bounds(
            fivec_composite,
            "fivec_pro_vax_idx",
        )

    if not update_dict:
        return agents

    # Apply updates via with_columns for efficiency
    return agents.with_columns([pl.Series(name=col, values=vals) for col, vals in update_dict.items()])
