"""Instrumentation utilities for credibility diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl

from beam_abm.abm.engine import VaccineBehaviourModel, _rescale_monthly_gain_to_substep
from beam_abm.abm.scenario_defs import ScenarioConfig
from beam_abm.abm.state_schema import CONSTRUCT_BOUNDS
from beam_abm.abm.state_update import (
    NORM_SHARE_HAT_COLUMN_VAX as NORM_SHARE_HAT_COLUMN,
)
from beam_abm.abm.state_update import (
    clip_to_bounds,
    compute_observed_norm_exposure,
)

from .credibility_shared import TRACKED_CONSTRUCTS, SharedData, build_model


@dataclass
class SubstepRecord:
    """Per-substep record of contributions for one construct."""

    tick: int
    substep: int
    construct: str
    # Means across agents
    pre_value: float = 0.0
    post_value: float = 0.0
    delta_peer_pull: float = 0.0
    delta_observed_norm: float = 0.0
    delta_attractor: float = 0.0
    delta_noise: float = 0.0
    delta_clip: float = 0.0
    delta_total: float = 0.0
    # Perceived-share pipeline (D2) — only for descriptive norms
    mean_observed_share: float = float("nan")
    mean_share_hat: float = float("nan")
    mean_share_hat0: float = float("nan")
    mean_logit_share_hat: float = float("nan")
    mean_logit_share_hat0: float = float("nan")
    mean_delta_e: float = float("nan")
    frac_obs_below_hat0: float = float("nan")
    frac_share_hat_saturating: float = float("nan")
    # Sampling bias (D3) — only for descriptive norms
    pop_mean_behavior: float = float("nan")
    mean_obs_share: float = float("nan")


def _safe_logit(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    c = np.clip(x, eps, 1.0 - eps)
    return np.log(c / (1.0 - c))


def run_instrumented(
    shared: SharedData,
    scenario: ScenarioConfig,
    *,
    attractor_on: bool = True,
    label: str = "instrumented",
) -> list[SubstepRecord]:
    """Run model with full per-substep instrumentation via monkeypatching."""
    model = build_model(shared, scenario, attractor_on=attractor_on)
    records: list[SubstepRecord] = []

    # We need to intercept the substep loop inside SurveyAgents.step().
    # Approach: run tick-by-tick, and for each tick monkeypatch the step
    # to capture pre/post values around each mechanism.

    # (no per-month substep calculation required here – instrumentation is
    # handled inside the helper)

    for tick in range(int(shared.cfg.n_steps)):
        # Snapshot before the tick's state updates
        pre_tick_values = {}
        for col in TRACKED_CONSTRUCTS:
            if col in model.agents.df.columns:
                pre_tick_values[col] = model.agents.df[col].cast(pl.Float64).to_numpy().astype(np.float64).copy()

        pre_share_hat = None
        if NORM_SHARE_HAT_COLUMN in model.agents.df.columns:
            pre_share_hat = model.agents.df[NORM_SHARE_HAT_COLUMN].cast(pl.Float64).to_numpy().astype(np.float64).copy()

        # We'll capture per-substep contributions by wrapping the step method.
        # Instead of monkeypatching individual functions (fragile), we'll run
        # the model one tick and compute contributions from pre/post snapshots
        # with intermediate snapshots at each mechanism boundary.
        #
        # However, the production step() runs substeps in a tight loop without
        # hooks. So we'll replicate the substep logic here with instrumentation.

        _run_instrumented_tick(model, tick, records, pre_tick_values, pre_share_hat)

    return records


def _run_instrumented_tick(
    model: VaccineBehaviourModel,
    tick: int,
    records: list[SubstepRecord],
    pre_tick_values: dict[str, np.ndarray],
    pre_share_hat: np.ndarray | None,
) -> None:
    """Execute one tick with full instrumentation."""

    m = model
    # This helper only records aggregate per-tick drift; the detailed
    # substep gain calculations are unused here, so they are omitted to
    # satisfy the linter.

    # Run the production step for outcome prediction and incidence
    # so that latest_outcomes and community_incidence are populated
    m.step()

    # Now we know post-tick state. Compute per-construct drift.
    for col in TRACKED_CONSTRUCTS:
        if col not in m.agents.df.columns or col not in pre_tick_values:
            continue
        post = m.agents.df[col].cast(pl.Float64).to_numpy().astype(np.float64)
        pre = pre_tick_values[col]
        delta_total = float(np.mean(post - pre))

        rec = SubstepRecord(
            tick=tick,
            substep=-1,  # aggregated over all substeps
            construct=col,
            pre_value=float(np.mean(pre)),
            post_value=float(np.mean(post)),
            delta_total=delta_total,
        )
        records.append(rec)

    # Perceived-share diagnostics
    if NORM_SHARE_HAT_COLUMN in m.agents.df.columns and pre_share_hat is not None:
        post_share_hat = m.agents.df[NORM_SHARE_HAT_COLUMN].cast(pl.Float64).to_numpy().astype(np.float64)
        rec = records[-len(TRACKED_CONSTRUCTS)]  # the descriptive norm record
        if rec.construct == "social_norms_descriptive_vax":
            rec.mean_share_hat = float(np.mean(post_share_hat))
            rec.mean_share_hat0 = float(np.mean(pre_share_hat)) if tick == 0 else rec.mean_share_hat0
            rec.mean_logit_share_hat = float(np.mean(_safe_logit(post_share_hat)))
            rec.mean_logit_share_hat0 = (
                float(np.mean(_safe_logit(pre_share_hat))) if tick == 0 else rec.mean_logit_share_hat0
            )


def run_instrumented_detailed(
    shared: SharedData,
    scenario: ScenarioConfig,
    *,
    attractor_on: bool = True,
    label: str = "detailed",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Run with detailed per-substep decomposition.

    Instead of monkeypatching the production loop (which is fragile),
    we replicate the substep logic with instrumentation hooks to capture
    the additive contribution of each mechanism.

    Returns (substep_records, tick_summary_records).
    """
    model = build_model(shared, scenario, attractor_on=attractor_on)
    from beam_abm.abm.message_contracts import norm_channel_columns, risk_channel_columns, trust_channel_columns
    from beam_abm.abm.network import compute_pairwise_symmetric_peer_pull, compute_sampled_peer_adoption
    from beam_abm.abm.outcome_scales import normalize_outcome_to_unit_interval
    from beam_abm.abm.outcomes import predict_all_outcomes
    from beam_abm.abm.state_update import (
        apply_all_state_updates,
    )

    m = model
    substeps_per_month = max(1, int(m.cfg.constants.social_substeps_per_month))
    k_social_substep = _rescale_monthly_gain_to_substep(m.cfg.knobs.k_social, substeps_per_month)
    relax_fast_substep = _rescale_monthly_gain_to_substep(m.cfg.constants.state_relaxation_fast, substeps_per_month)
    relax_slow_substep = _rescale_monthly_gain_to_substep(m.cfg.constants.state_relaxation_slow, substeps_per_month)

    def _ema_perceived_share(pre_hat: np.ndarray, observed: np.ndarray) -> np.ndarray:
        """Mirror production share-hat EMA used by apply_all_state_updates."""
        lr = float(np.clip(relax_slow_substep, 0.0, 1.0))
        if m.baseline_share_hat is not None and m.baseline_expected_obs_share is not None:
            hat_0 = np.asarray(m.baseline_share_hat, dtype=np.float64)
            obs_ref = np.asarray(m.baseline_expected_obs_share, dtype=np.float64)
            delta_obs = observed - obs_ref
            delta_hat = pre_hat - hat_0
            delta_hat_new = delta_hat + lr * (delta_obs - delta_hat)
            return np.clip(hat_0 + delta_hat_new, 0.0, 1.0)
        return np.clip(pre_hat + lr * (observed - pre_hat), 0.0, 1.0)

    substep_records: list[dict[str, Any]] = []
    tick_records: list[dict[str, Any]] = []

    # Initial share_hat0 snapshot
    share_hat0 = None
    if NORM_SHARE_HAT_COLUMN in m.agents.df.columns:
        share_hat0 = m.agents.df[NORM_SHARE_HAT_COLUMN].cast(pl.Float64).to_numpy().astype(np.float64).copy()

    for tick in range(int(m.cfg.n_steps)):
        # 1. Run outcome prediction first (like production code)
        m._current_tick = tick
        if tick == 0 or m._seed_mask is None:
            m._seed_mask = m._compute_target_mask()
            m._target_mask = m._seed_mask
            m._graph_distance_from_seed = m._compute_graph_distance_from_seed(m._seed_mask)
        else:
            m._target_mask = m._seed_mask

        outcomes = predict_all_outcomes(
            m.agents.df,
            m.backbone_models,
            m.levers_cfg,
            deterministic=m.cfg.deterministic,
            rng=m.random,
            tick=tick,
            choice_signals=m._choice_signals,
            target_mask=m._target_mask,
            norm_channel_weights=m._norm_channel_weights,
        )
        m._latest_outcomes = outcomes
        m._update_community_incidence(outcomes)

        # Lazy-init baseline expected obs share from first tick's outcomes
        if m._baseline_expected_obs_share is None:
            m._baseline_expected_obs_share = m._compute_baseline_expected_obs_share(
                outcomes,
                outcome_names={"vax_willingness_T12"},
                fallback_to_all=False,
            )
            if m._baseline_expected_obs_share is not None and m._baseline_expected_obs_share.size > 0:
                m._sync_observed_norm_anchor_bundle(
                    channel="vax",
                    anchor_share=m._baseline_expected_obs_share,
                )
        if m._baseline_expected_obs_share_npi is None:
            m._baseline_expected_obs_share_npi = m._compute_baseline_expected_obs_share(
                outcomes,
                outcome_names={
                    "mask_when_symptomatic_crowded",
                    "stay_home_when_symptomatic",
                    "mask_when_pressure_high",
                },
                fallback_to_all=False,
            )
            if m._baseline_expected_obs_share_npi is not None and m._baseline_expected_obs_share_npi.size > 0:
                m._sync_observed_norm_anchor_bundle(
                    channel="npi",
                    anchor_share=m._baseline_expected_obs_share_npi,
                )

        # Write outcomes back
        for oname, ovals in outcomes.items():
            m.agents.df = m.agents.df.with_columns(pl.Series(name=oname, values=ovals))

        # Pre-tick snapshot
        pre_tick = {}
        for col in TRACKED_CONSTRUCTS:
            if col in m.agents.df.columns:
                pre_tick[col] = m.agents.df[col].cast(pl.Float64).to_numpy().astype(np.float64).copy()

        # per-tick snapshot of share_hat is not used by the current
        # diagnostics; keep the code simple by omitting it.

        # 2. Substep loop (replicated from SurveyAgents.step with instrumentation)
        for substep in range(substeps_per_month):
            global_substep = tick * substeps_per_month + substep

            # Snapshot pre-substep for each tracked construct
            pre_sub = {}
            for col in TRACKED_CONSTRUCTS:
                if col in m.agents.df.columns:
                    pre_sub[col] = m.agents.df[col].cast(pl.Float64).to_numpy().astype(np.float64).copy()

            pre_share_hat = None
            if NORM_SHARE_HAT_COLUMN in m.agents.df.columns:
                pre_share_hat = m.agents.df[NORM_SHARE_HAT_COLUMN].cast(pl.Float64).to_numpy().astype(np.float64).copy()

            working_df = m.apply_state_signals_to_precursors(m.agents.df, tick=tick, target_mask=m._target_mask)

            # --- MECHANISM 1: Pairwise peer pull ---
            post_peer_pull = {}
            if m.scenario.communication_enabled and m.scenario.peer_pull_enabled and m.network is not None:
                social_columns = tuple(
                    col for col in (norm_channel_columns() + trust_channel_columns() + risk_channel_columns())
                )
                gains_by_column: dict[str, float] = {}
                for col in social_columns:
                    if col in norm_channel_columns():
                        bcg = float(max(0.0, m.cfg.constants.social_weight_norm))
                    elif col in trust_channel_columns():
                        bcg = float(max(0.0, m.cfg.constants.social_weight_trust))
                    elif col in risk_channel_columns():
                        bcg = float(max(0.0, m.cfg.constants.social_weight_risk))
                    else:
                        bcg = 0.0
                    gain = float(np.clip(k_social_substep * bcg, 0.0, 1.0))
                    if gain > 0.0 and col in working_df.columns:
                        gains_by_column[col] = gain

                from beam_abm.abm.engine import _agent_peer_pull_rates_per_substep

                pairwise_updates, _ = compute_pairwise_symmetric_peer_pull(
                    working_df,
                    m.network,
                    columns=social_columns,
                    gains_by_column=gains_by_column,
                    homophily_gamma=m.cfg.constants.homophily_gamma,
                    homophily_w_trust=0.5,
                    homophily_w_moral=0.5,
                    rng=m.random,
                    interactions_per_agent=float(m.cfg.constants.peer_pull_interactions_per_agent),
                    per_agent_interactions=_agent_peer_pull_rates_per_substep(
                        working_df,
                        base_rate=float(m.cfg.constants.peer_pull_interactions_per_agent),
                    ),
                )
                if pairwise_updates:
                    working_df = working_df.with_columns(
                        [pl.Series(name=c, values=v) for c, v in pairwise_updates.items()]
                    )
                for col in TRACKED_CONSTRUCTS:
                    if col in pairwise_updates:
                        post_peer_pull[col] = float(np.mean(pairwise_updates[col] - pre_sub[col]))
                    else:
                        post_peer_pull[col] = 0.0
            else:
                for col in TRACKED_CONSTRUCTS:
                    post_peer_pull[col] = 0.0

            # Snapshot after peer pull, before observed norms
            after_peer = {}
            for col in TRACKED_CONSTRUCTS:
                if col in working_df.columns:
                    after_peer[col] = working_df[col].cast(pl.Float64).to_numpy().astype(np.float64).copy()

            # --- MECHANISM 2: Observed norm sampling + share_hat update ---
            observed_norm_share = None
            pop_mean_behavior = float("nan")
            mean_obs_share = float("nan")
            if (
                m.scenario.communication_enabled
                and m.scenario.observed_norms_enabled
                and m.network is not None
                and m._latest_outcomes is not None
                and "social_norms_descriptive_vax" in working_df.columns
            ):
                from beam_abm.abm.engine import _agent_norm_observation_targets_per_month

                obs_targets_month = _agent_norm_observation_targets_per_month(
                    working_df,
                    min_per_month=m.cfg.constants.norm_observation_min_per_month,
                    max_per_month=m.cfg.constants.norm_observation_max_per_month,
                )
                obs_targets_substep = np.maximum(
                    1,
                    np.ceil(obs_targets_month.astype(np.float64) / float(substeps_per_month)).astype(np.int64),
                )
                obs_targets_substep = np.minimum(obs_targets_substep, int(m.cfg.norm_observation_sample_size))

                social_attention = working_df["social_attention_alpha"].cast(pl.Float64).to_numpy().astype(np.float64)
                social_sharing = working_df["social_sharing_phi"].cast(pl.Float64).to_numpy().astype(np.float64)

                observed_norm_share, _ = compute_sampled_peer_adoption(
                    m.network,
                    m._latest_outcomes,
                    sample_size=m.cfg.norm_observation_sample_size,
                    rng=m.random,
                    attention=social_attention,
                    sharing=social_sharing,
                    contact_rate_base=m.cfg.constants.contact_rate_base,
                    contact_rate_multiplier=m.cfg.knobs.contact_rate_multiplier,
                    activity_score=working_df["activity_score"].cast(pl.Float64).to_numpy().astype(np.float64)
                    if "activity_score" in working_df.columns
                    else None,
                    contact_propensity=working_df["contact_propensity"].cast(pl.Float64).to_numpy().astype(np.float64)
                    if "contact_propensity" in working_df.columns
                    else None,
                    per_agent_sample_size=obs_targets_substep,
                )

                # D3: Sampling bias diagnostics
                outcome_arrays = []
                for oname, ovals in m._latest_outcomes.items():
                    outcome_arrays.append(normalize_outcome_to_unit_interval(ovals, oname))
                if outcome_arrays:
                    protection_index = np.mean(np.vstack(outcome_arrays), axis=0)
                    pop_mean_behavior = float(np.mean(protection_index))
                mean_obs_share = float(np.mean(observed_norm_share))

            # Now run apply_all_state_updates to get the full update
            social_susceptibility = (
                working_df["social_susceptibility_lambda"].cast(pl.Float64).to_numpy().astype(np.float64)
            )
            norm_threshold = working_df["norm_threshold_theta"].cast(pl.Float64).to_numpy().astype(np.float64)

            attractor_targets = None
            dynamic_offsets: dict[str, np.ndarray] = {}
            if m.scenario.enable_endogenous_loop and "locality_id" in working_df.columns:
                communities = working_df["locality_id"].to_numpy()
                incidence = np.array(
                    [
                        m.community_incidence.get(str(c), m.cfg.constants.community_incidence_baseline)
                        for c in communities
                    ],
                    dtype=np.float64,
                )
                stakes_shift = m.cfg.knobs.k_feedback_stakes * (
                    incidence - m.cfg.constants.community_incidence_baseline
                )
                dynamic_offsets = {
                    "covid_perceived_danger_T12": stakes_shift,
                    "disease_fear_avg": m.cfg.constants.stakes_to_fear_multiplier * stakes_shift,
                    "infection_likelihood_avg": m.cfg.constants.stakes_to_infection_multiplier * stakes_shift,
                }

            endogenous_offsets = dynamic_offsets or None
            if m.state_ref_attractor is not None:
                attractor_targets = m.state_ref_attractor.predict_targets(working_df)
                endogenous_offsets = m.state_ref_attractor.merged_offsets(dynamic_offsets or None)

            updated_df = apply_all_state_updates(
                agents=working_df,
                signals=m.state_signals,
                tick=tick,
                attractor_targets=attractor_targets,
                peer_means=None,
                k_social=k_social_substep,
                social_weight_norm=m.cfg.constants.social_weight_norm,
                social_weight_trust=m.cfg.constants.social_weight_trust,
                social_weight_risk=m.cfg.constants.social_weight_risk,
                rng=m.random,
                target_mask=m._target_mask,
                belief_kernel=m.belief_kernel,
                baseline_agents=m.population_t0,
                observed_norm_share=observed_norm_share,
                social_susceptibility=social_susceptibility,
                norm_threshold=norm_threshold,
                norm_slope=m.cfg.constants.norm_response_slope,
                baseline_descriptive_norm=m.baseline_descriptive_norm,
                baseline_norm_exposure=m.baseline_norm_exposure,
                baseline_share_hat=m.baseline_share_hat,
                baseline_observed_share=m.baseline_expected_obs_share,
                endogenous_offsets=endogenous_offsets,
                signal_overrides=None,
                norm_channel_weights=m._norm_channel_weights,
                social_noise_std=m.cfg.constants.state_update_noise_std,
                norm_threshold_min=m.cfg.constants.norm_threshold_min,
                norm_threshold_max=m.cfg.constants.norm_threshold_max,
                state_relaxation_fast=relax_fast_substep,
                state_relaxation_slow=relax_slow_substep,
                frozen_columns=set(m.scenario.frozen_state_columns),
            )

            # Save uid column
            uid = m.agents.df["unique_id"]
            m.agents.df = updated_df.with_columns(uid)

            # Post-full-update snapshots
            post_full = {}
            for col in TRACKED_CONSTRUCTS:
                if col in m.agents.df.columns:
                    post_full[col] = m.agents.df[col].cast(pl.Float64).to_numpy().astype(np.float64).copy()

            post_share_hat = None
            if NORM_SHARE_HAT_COLUMN in m.agents.df.columns:
                post_share_hat = (
                    m.agents.df[NORM_SHARE_HAT_COLUMN].cast(pl.Float64).to_numpy().astype(np.float64).copy()
                )

            # --- Decompose contributions ---
            # For descriptive norms specifically, we can decompose more precisely.
            # For generic constructs, we decompose as:
            #   delta_peer_pull = after_peer - pre_sub
            #   delta_update_total = post_full - after_peer
            # The update_total includes: observed_norm + attractor + noise + clip
            #
            # To separate attractor from observed norm, we'd need to re-run
            # apply_all_state_updates with attractor=None. Instead, for efficiency,
            # we compute the attractor contribution analytically where possible.

            for col in TRACKED_CONSTRUCTS:
                if col not in pre_sub or col not in post_full:
                    continue

                pre_mean = float(np.mean(pre_sub[col]))
                post_mean = float(np.mean(post_full[col]))

                delta_peer = post_peer_pull.get(col, 0.0)
                delta_total = post_mean - pre_mean

                # For attractor contribution estimation:
                # In production code, descriptive norm channels are updated
                # only via observed-norm pipeline (plus noise + clipping).
                # They are explicitly excluded from attractor updates.
                # Generic constructs may include attractor pull.
                delta_attractor = 0.0
                delta_obs_norm = 0.0
                delta_clip = 0.0
                delta_noise = 0.0

                after_peer_vals = after_peer.get(col, pre_sub[col])

                if col == "social_norms_descriptive_vax" and observed_norm_share is not None:
                    # Compute what observed norm update would produce (without attractor)
                    curr = after_peer_vals.copy()
                    susc = np.clip(
                        social_susceptibility * float(np.clip(k_social_substep, 0.0, 1.0)),
                        0.0,
                        1.0,
                    )
                    norm_threshold_arr = np.clip(
                        norm_threshold, m.cfg.constants.norm_threshold_min, m.cfg.constants.norm_threshold_max
                    )
                    if pre_share_hat is not None:
                        perceived_after_ema = _ema_perceived_share(pre_share_hat, observed_norm_share)
                    else:
                        perceived_after_ema = observed_norm_share

                    lo, hi = CONSTRUCT_BOUNDS.get(col, (1.0, 7.0))
                    span = hi - lo
                    baseline_norm = m.baseline_descriptive_norm
                    baseline_share_anchor = m.baseline_share_hat
                    baseline_exp = m.baseline_norm_exposure
                    if baseline_norm is not None and baseline_share_anchor is not None:
                        target_obs = np.clip(
                            baseline_norm + span * (perceived_after_ema - baseline_share_anchor),
                            lo,
                            hi,
                        )
                    elif baseline_norm is not None and baseline_exp is not None:
                        exposure = compute_observed_norm_exposure(
                            observed_share=perceived_after_ema,
                            threshold=norm_threshold_arr,
                            slope=float(m.cfg.constants.norm_response_slope),
                        )
                        target_obs = np.clip(baseline_norm + span * (exposure - baseline_exp), lo, hi)
                    else:
                        exposure = compute_observed_norm_exposure(
                            observed_share=perceived_after_ema,
                            threshold=norm_threshold_arr,
                            slope=float(m.cfg.constants.norm_response_slope),
                        )
                        target_obs = lo + span * exposure

                    after_obs = curr + susc * (target_obs - curr)
                    delta_obs_norm = float(np.mean(after_obs - curr))

                    clipped = clip_to_bounds(after_obs, col)
                    delta_clip = float(np.mean(clipped - after_obs))
                    delta_noise = delta_total - delta_peer - delta_obs_norm - delta_clip
                else:
                    # Generic construct: delta_update = post - after_peer
                    delta_update = float(np.mean(post_full[col] - after_peer_vals))
                    # Attractor contribution estimate
                    if attractor_targets is not None and col in attractor_targets:
                        att_target = attractor_targets[col]
                        relax_r = (
                            relax_slow_substep
                            if col
                            in {
                                "social_norms_descriptive_vax",
                                "social_norms_injunctive_vax",
                                "group_trust_vax",
                                "institutional_trust_avg",
                                "covax_legitimacy_scepticism_idx",
                                "fivec_confidence",
                                "fivec_collective_responsibility",
                                "vaccine_risk_avg",
                                "flu_vaccine_risk",
                            }
                            else relax_fast_substep
                        )
                        delta_attractor = float(np.mean(relax_r * (att_target - after_peer_vals)))
                    delta_obs_norm = 0.0
                    # Clip estimate
                    lo, hi = CONSTRUCT_BOUNDS.get(col, (-np.inf, np.inf))
                    # unclipped estimate would be after_peer_vals + (post_full[col] - after_peer_vals)
                    # but we don't need the temporary value here; drop it to appease the linter.
                    # Actually: unclipped = after_peer + relax*(target - after_peer)
                    # We can't cleanly separate without running without clip.
                    # Attribute noise as residual
                    delta_noise = delta_update - delta_attractor - delta_obs_norm
                    delta_clip = 0.0  # can't easily separate for generic constructs

                rec = {
                    "tick": tick,
                    "substep": substep,
                    "global_substep": global_substep,
                    "construct": col,
                    "pre_value": float(np.mean(pre_sub[col])),
                    "post_value": float(np.mean(post_full[col])),
                    "delta_peer_pull": delta_peer,
                    "delta_observed_norm": delta_obs_norm,
                    "delta_attractor": delta_attractor,
                    "delta_noise": delta_noise,
                    "delta_clip": delta_clip,
                    "delta_total": delta_total,
                    "label": label,
                    "attractor_on": attractor_on,
                }

                # D2: Perceived-share pipeline diagnostics
                if col == "social_norms_descriptive_vax" and observed_norm_share is not None:
                    rec["mean_observed_share"] = float(np.mean(observed_norm_share))
                    if post_share_hat is not None:
                        rec["mean_share_hat"] = float(np.mean(post_share_hat))
                        rec["mean_logit_share_hat"] = float(np.mean(_safe_logit(post_share_hat)))
                    if share_hat0 is not None:
                        rec["mean_share_hat0"] = float(np.mean(share_hat0))
                        rec["mean_logit_share_hat0"] = float(np.mean(_safe_logit(share_hat0)))
                        rec["frac_obs_below_hat0"] = float(np.mean(observed_norm_share < share_hat0))
                    if pre_share_hat is not None and observed_norm_share is not None:
                        perceived_after = _ema_perceived_share(pre_share_hat, observed_norm_share)
                        baseline_exp_arr = m.baseline_norm_exposure
                        if baseline_exp_arr is not None:
                            exposure_current = compute_observed_norm_exposure(
                                observed_share=perceived_after,
                                threshold=np.clip(
                                    norm_threshold,
                                    m.cfg.constants.norm_threshold_min,
                                    m.cfg.constants.norm_threshold_max,
                                ),
                                slope=float(m.cfg.constants.norm_response_slope),
                            )
                            rec["mean_delta_e"] = float(np.mean(exposure_current - baseline_exp_arr))
                    # Saturation
                    if post_share_hat is not None:
                        rec["frac_share_hat_saturating"] = float(
                            np.mean((post_share_hat < 0.01) | (post_share_hat > 0.99))
                        )

                    # D3: Sampling bias
                    rec["pop_mean_behavior"] = pop_mean_behavior
                    rec["mean_obs_share"] = mean_obs_share

                substep_records.append(rec)

            # Update pre_share_hat for next substep
            if NORM_SHARE_HAT_COLUMN in m.agents.df.columns:
                pre_share_hat = m.agents.df[NORM_SHARE_HAT_COLUMN].cast(pl.Float64).to_numpy().astype(np.float64).copy()

        # Tick-level summary
        for col in TRACKED_CONSTRUCTS:
            if col in pre_tick and col in m.agents.df.columns:
                post = m.agents.df[col].cast(pl.Float64).to_numpy().astype(np.float64)
                tick_records.append(
                    {
                        "tick": tick,
                        "construct": col,
                        "pre_mean": float(np.mean(pre_tick[col])),
                        "post_mean": float(np.mean(post)),
                        "drift": float(np.mean(post - pre_tick[col])),
                        "std_post": float(np.std(post)),
                        "label": label,
                        "attractor_on": attractor_on,
                    }
                )

    return substep_records, tick_records
