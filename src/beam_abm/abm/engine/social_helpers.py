"""Social substep helper functions for the ABM engine."""

from __future__ import annotations

import math

import numpy as np
import polars as pl


def _estimate_contact_rate_base(agents: pl.DataFrame) -> float:
    """Estimate weekly informative exposure base from survey interaction items.

    Anchors the fast social-diffusion layer to weekly-recall discussion
    variables when available, with legacy contact-group fallback.
    """
    n_agents = agents.height
    if n_agents == 0:
        return 1.0

    def _col(name: str, default: float) -> np.ndarray:
        if name not in agents.columns:
            return np.full(n_agents, default, dtype=np.float64)
        return np.nan_to_num(agents[name].cast(pl.Float64).to_numpy(), nan=default)

    discussion_exposure = (
        _col("discussion_group_cohabitant", 0.0)
        + _col("discussion_group_noncohabitant", 0.0)
        + _col("discussion_group_friends", 0.0)
        + _col("discussion_group_colleagues", 0.0)
        + _col("discussion_group_others", 0.0)
    )
    if float(np.nanmean(discussion_exposure)) <= 0.0:
        discussion_exposure = (
            _col("contact_group_cohabitant", 0.0)
            + _col("contact_group_noncohabitant", 0.0)
            + _col("contact_group_friends", 0.0)
            + _col("contact_group_colleagues", 0.0)
            + _col("contact_group_others", 0.0)
        )

    target_mean = float(np.mean(discussion_exposure)) if discussion_exposure.size > 0 else 1.0
    if target_mean <= 0.0:
        target_mean = 1.0

    activity = _col("activity_score", 0.0)
    propensity = np.clip(_col("contact_propensity", 1.0), 0.1, 5.0)
    sharing = np.clip(_col("social_sharing_phi", 1.0), 0.0, np.inf)
    sharing_gate = sharing / (1.0 + sharing)
    scale = propensity * (0.65 + activity + 0.35 * sharing_gate)
    mean_scale = float(np.mean(scale)) if scale.size > 0 else 1.0
    if mean_scale <= 1e-6:
        return 1.0
    return float(np.clip(target_mean / mean_scale, 0.1, 12.0))


def _rescale_monthly_gain_to_substep(monthly_gain: float, substeps_per_month: int) -> float:
    """Convert a monthly assimilation/relaxation gain to substep gain."""
    if substeps_per_month <= 1:
        return float(np.clip(monthly_gain, 0.0, 1.0))

    gain = float(np.clip(monthly_gain, 0.0, 1.0))
    if gain <= 0.0:
        return 0.0
    if gain >= 1.0:
        return 1.0
    return float(1.0 - math.pow(1.0 - gain, 1.0 / float(substeps_per_month)))


def _agent_norm_observation_targets_per_month(
    agents: pl.DataFrame,
    *,
    min_per_month: int,
    max_per_month: int,
) -> np.ndarray:
    """Map conversation-frequency precursor to per-agent monthly observation count."""
    n_agents = agents.height
    if n_agents == 0:
        return np.array([], dtype=np.int64)

    lo = int(min_per_month)
    hi = int(max_per_month)
    if hi < lo:
        lo, hi = hi, lo
    if hi == lo:
        return np.full(n_agents, lo, dtype=np.int64)

    column = "vax_convo_freq_cohabitants"
    if column not in agents.columns:
        midpoint = int(round((lo + hi) / 2.0))
        return np.full(n_agents, midpoint, dtype=np.int64)

    raw = agents[column].cast(pl.Float64).to_numpy().astype(np.float64)
    finite = raw[np.isfinite(raw)]
    if finite.size == 0:
        midpoint = int(round((lo + hi) / 2.0))
        return np.full(n_agents, midpoint, dtype=np.int64)

    center = float(np.median(finite))
    safe = np.where(np.isfinite(raw), raw, center)
    p05, p95 = np.percentile(finite, [5.0, 95.0])
    if not np.isfinite(p05) or not np.isfinite(p95) or p95 <= p05:
        norm = np.full(n_agents, 0.5, dtype=np.float64)
    else:
        norm = np.clip((safe - p05) / (p95 - p05), 0.0, 1.0)

    targets = lo + np.rint((hi - lo) * norm).astype(np.int64)
    return np.clip(targets, lo, hi).astype(np.int64)


def _agent_peer_pull_rates_per_substep(
    agents: pl.DataFrame,
    *,
    base_rate: float,
) -> np.ndarray:
    """Build per-agent peer-pull rates anchored on vaccine conversation counts.

    The returned vector is interpreted as expected interactions per weekly
    social substep and is mean-preserving around ``base_rate``.
    """
    n_agents = agents.height
    if n_agents == 0:
        return np.array([], dtype=np.float64)

    baseline = float(np.clip(base_rate, 0.05, 8.0))
    required = (
        "vax_convo_count_friends",
        "vax_convo_count_cohabitants",
        "vax_convo_count_others",
    )
    if not all(col in agents.columns for col in required):
        return np.full(n_agents, baseline, dtype=np.float64)

    total = np.zeros(n_agents, dtype=np.float64)
    for col in required:
        total += np.nan_to_num(agents[col].cast(pl.Float64).to_numpy(), nan=0.0, posinf=0.0, neginf=0.0)

    finite = total[np.isfinite(total)]
    if finite.size == 0:
        return np.full(n_agents, baseline, dtype=np.float64)

    p10, p90 = np.percentile(finite, [10.0, 90.0])
    if (not np.isfinite(p10)) or (not np.isfinite(p90)) or p90 <= p10:
        return np.full(n_agents, baseline, dtype=np.float64)

    norm = np.clip((total - p10) / (p90 - p10), 0.0, 1.0)
    multiplier = 0.4 + 1.6 * norm
    mean_multiplier = float(np.mean(multiplier))
    if np.isfinite(mean_multiplier) and mean_multiplier > 1e-8:
        multiplier = multiplier / mean_multiplier

    rates = baseline * multiplier
    return np.clip(rates, 0.05, 8.0).astype(np.float64)
