"""Polars-backed survey agents for the vaccine behaviour ABM."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl
from mesa_frames import AgentSet

from beam_abm.abm.engine.social_helpers import (
    _agent_norm_observation_targets_per_month,
    _agent_peer_pull_rates_per_substep,
    _rescale_monthly_gain_to_substep,
)
from beam_abm.abm.message_contracts import peer_pull_risk_columns, trust_channel_columns
from beam_abm.abm.network import ContactDiagnostics, compute_pairwise_symmetric_peer_pull
from beam_abm.abm.state_update import apply_all_state_updates

if TYPE_CHECKING:
    from beam_abm.abm.engine.model import VaccineBehaviourModel


class SurveyAgents(AgentSet):
    """Polars-backed agent set storing all mutable+immutable columns.

    Each row is one synthetic agent initialised via anchor-country
    neighbourhood resampling from ``preprocess`` survey data.
    The ``step`` implementation applies exogenous signal updates and
    optional social-norm diffusion; outcome prediction is triggered
    from the model's ``step``.

    Parameters
    ----------
    model : VaccineBehaviourModel
        The owning model instance.
    agent_df : pl.DataFrame
        Initial agent-state DataFrame (from ``initialise_population``).
    """

    def __init__(
        self,
        model: VaccineBehaviourModel,
        agent_df: pl.DataFrame,
    ) -> None:
        super().__init__(model)
        # mesa-frames auto-generates unique_id; drop ours if present
        if "unique_id" in agent_df.columns:
            agent_df = agent_df.drop("unique_id")
        self.add(agent_df)

    # -- vectorised state update (called each tick) ----------------------

    def step(self) -> None:
        """Apply one tick of mutable-state updates.

        Reads signals, peer means, and norm gain from the parent
        model and delegates to ``apply_all_state_updates``.
        """
        m: VaccineBehaviourModel = self.model  # type: ignore[assignment]

        def _require_numeric_column(df: pl.DataFrame, col: str) -> np.ndarray:
            if col not in df.columns:
                msg = f"Missing required agent column '{col}'."
                raise ValueError(msg)
            vals = df[col].cast(pl.Float64).to_numpy()
            if not np.isfinite(vals).all():
                msg = f"Column '{col}' contains null/NaN/inf values; strict ABM mode disallows fallbacks."
                raise ValueError(msg)
            return vals

        if m.freeze_state_updates:
            return

        substeps_per_month = max(1, int(m.cfg.constants.social_substeps_per_month))
        k_social_substep = _rescale_monthly_gain_to_substep(m.cfg.knobs.k_social, substeps_per_month)
        relax_fast_substep = _rescale_monthly_gain_to_substep(
            m.cfg.constants.state_relaxation_fast,
            substeps_per_month,
        )
        relax_slow_substep = _rescale_monthly_gain_to_substep(
            m.cfg.constants.state_relaxation_slow,
            substeps_per_month,
        )

        similarity_samples: list[float] = []
        contact_diag_samples: list[ContactDiagnostics] = []

        _do_trace = getattr(m.cfg, "debug_trace", False)
        _trace_cols = tuple(
            col
            for col in (
                "social_norms_descriptive_vax",
                "institutional_trust_avg",
                "group_trust_vax",
                "covid_perceived_danger_T12",
                "disease_fear_avg",
                "infection_likelihood_avg",
                "vaccine_risk_avg",
                "fivec_confidence",
                "fivec_low_complacency",
                "norm_share_hat_vax",
            )
        )

        def _snapshot_cols(df: pl.DataFrame) -> dict[str, np.ndarray]:
            out: dict[str, np.ndarray] = {}
            for c in _trace_cols:
                if c in df.columns:
                    out[c] = df[c].cast(pl.Float64).to_numpy().copy()
            return out

        def _record_module_delta(
            module: str,
            snap_before: dict[str, np.ndarray],
            snap_after: dict[str, np.ndarray],
            substep: int,
        ) -> None:
            for c in _trace_cols:
                b = snap_before.get(c)
                a = snap_after.get(c)
                if b is None or a is None:
                    continue
                delta = a - b
                m._debug_trace_records.append(
                    {
                        "tick": m.current_tick,
                        "substep": substep,
                        "module": module,
                        "variable": c,
                        "delta_l2": float(np.linalg.norm(delta)),
                        "delta_mean": float(np.mean(delta)),
                        "mean_before": float(np.mean(b)),
                        "mean_after": float(np.mean(a)),
                        "var_before": float(np.var(b)),
                        "var_after": float(np.var(a)),
                    }
                )

        for _substep in range(substeps_per_month):
            working_df = m.apply_state_signals_to_precursors(
                self.df,
                tick=m.current_tick,
                target_mask=m.target_mask,
            )

            if _do_trace:
                _snap_pre_peer = _snapshot_cols(working_df)

            # Pairwise symmetric peer pull if communication is enabled.
            # Norm constructs are excluded: descriptive norms move through
            # observed behavior, while trust/risk narratives diffuse via talk.
            if m.scenario.communication_enabled and m.scenario.peer_pull_enabled and m.network is not None:
                t_social = time.perf_counter()
                social_columns = tuple(
                    col for col in ("group_trust_vax",) + trust_channel_columns() + peer_pull_risk_columns()
                )
                gains_by_column: dict[str, float] = {}
                for col in social_columns:
                    if col == "group_trust_vax" or col in trust_channel_columns():
                        base_channel_gain = float(max(0.0, m.cfg.constants.social_weight_trust))
                    elif col in peer_pull_risk_columns():
                        base_channel_gain = float(max(0.0, m.cfg.constants.social_weight_risk))
                    else:
                        base_channel_gain = 0.0
                    gain = float(np.clip(k_social_substep * base_channel_gain, 0.0, 1.0))
                    if gain > 0.0 and col in working_df.columns:
                        gains_by_column[col] = gain

                pairwise_updates, avg_similarity = compute_pairwise_symmetric_peer_pull(
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
                        [pl.Series(name=col, values=vals) for col, vals in pairwise_updates.items()]
                    )
                similarity_samples.append(float(avg_similarity))
                m.timing.social_total += time.perf_counter() - t_social

            if _do_trace:
                _snap_post_peer = _snapshot_cols(working_df)
                _record_module_delta("peer_pull", _snap_pre_peer, _snap_post_peer, _substep)

            observed_norm_share: np.ndarray | None = None
            observed_norm_share_npi: np.ndarray | None = None
            social_attention = _require_numeric_column(working_df, "social_attention_alpha")
            social_sharing = _require_numeric_column(working_df, "social_sharing_phi")
            if (
                m.scenario.communication_enabled
                and m.scenario.observed_norms_enabled
                and m.network is not None
                and m.latest_outcomes is not None
                and "social_norms_descriptive_vax" in working_df.columns
            ):
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

                from beam_abm.abm import engine as engine_module

                sampled_peer_adoption: Any = engine_module.compute_sampled_peer_adoption
                # Descriptive vax norms track vaccination behaviour only —
                # NPI outcomes (masking, staying home) are a separate domain
                # and should not dilute the vaccination norm signal.
                vax_only_outcomes = {k: v for k, v in m.latest_outcomes.items() if k == "vax_willingness_T12"}
                if vax_only_outcomes:
                    observed_norm_share, contact_diag = sampled_peer_adoption(
                        m.network,
                        vax_only_outcomes,
                        sample_size=m.cfg.norm_observation_sample_size,
                        rng=m.random,
                        attention=social_attention,
                        sharing=social_sharing,
                        contact_rate_base=m.cfg.constants.contact_rate_base,
                        contact_rate_multiplier=m.cfg.knobs.contact_rate_multiplier,
                        activity_score=_require_numeric_column(working_df, "activity_score"),
                        contact_propensity=_require_numeric_column(working_df, "contact_propensity"),
                        per_agent_sample_size=obs_targets_substep,
                    )
                    contact_diag_samples.append(contact_diag)
                # Descriptive NPI norms track visible protection behaviour only.
                if "social_norms_descriptive_npi" in working_df.columns:
                    npi_only_outcomes = {
                        k: v
                        for k, v in m.latest_outcomes.items()
                        if k
                        in {"mask_when_symptomatic_crowded", "stay_home_when_symptomatic", "mask_when_pressure_high"}
                    }
                    if npi_only_outcomes:
                        observed_norm_share_npi, _ = sampled_peer_adoption(
                            m.network,
                            npi_only_outcomes,
                            sample_size=m.cfg.norm_observation_sample_size,
                            rng=m.random,
                            attention=social_attention,
                            sharing=social_sharing,
                            contact_rate_base=m.cfg.constants.contact_rate_base,
                            contact_rate_multiplier=m.cfg.knobs.contact_rate_multiplier,
                            activity_score=_require_numeric_column(working_df, "activity_score"),
                            contact_propensity=_require_numeric_column(working_df, "contact_propensity"),
                            per_agent_sample_size=obs_targets_substep,
                        )

            burn_in_ticks = int(m.cfg.constants.burn_in_ticks)
            burn_in_active = burn_in_ticks > 0 and m.current_tick < burn_in_ticks
            if burn_in_active:
                if observed_norm_share is not None:
                    m.update_baseline_expected_obs_share_anchor(observed_norm_share, channel="vax")
                if observed_norm_share_npi is not None:
                    m.update_baseline_expected_obs_share_anchor(observed_norm_share_npi, channel="npi")

                burn_in_anchor_updates: list[pl.Series] = []
                if m._baseline_share_hat is not None and "norm_share_hat_vax" in working_df.columns:
                    burn_in_anchor_updates.append(
                        pl.Series("norm_share_hat_vax", m._baseline_share_hat.astype(np.float64, copy=True))
                    )
                if m._baseline_share_hat_npi is not None and "norm_share_hat_npi" in working_df.columns:
                    burn_in_anchor_updates.append(
                        pl.Series("norm_share_hat_npi", m._baseline_share_hat_npi.astype(np.float64, copy=True))
                    )
                if burn_in_anchor_updates:
                    working_df = working_df.with_columns(burn_in_anchor_updates)

            dynamic_offsets: dict[str, np.ndarray] = {}
            if m.scenario.enable_endogenous_loop and "locality_id" in working_df.columns:
                communities = working_df["locality_id"].to_numpy()
                incidence = np.array(
                    [
                        m.community_incidence.get(
                            str(c),
                            m.cfg.constants.community_incidence_baseline,
                        )
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

            social_susceptibility = _require_numeric_column(working_df, "social_susceptibility_lambda")
            norm_threshold = _require_numeric_column(working_df, "norm_threshold_theta")
            attractor_targets: dict[str, np.ndarray] | None = None
            endogenous_offsets: dict[str, np.ndarray] | None = dynamic_offsets or None
            if m.state_ref_attractor is not None:
                attractor_targets = m.state_ref_attractor.predict_targets(working_df)
                endogenous_offsets = m.state_ref_attractor.merged_offsets(dynamic_offsets or None)

            # Apply mutable-state transitions
            t_update = time.perf_counter()
            updated_df = apply_all_state_updates(
                agents=working_df,
                signals=m.state_signals,
                tick=m.current_tick,
                attractor_targets=attractor_targets,
                peer_means=None,
                k_social=k_social_substep,
                social_weight_norm=m.cfg.constants.social_weight_norm,
                social_weight_trust=m.cfg.constants.social_weight_trust,
                social_weight_risk=m.cfg.constants.social_weight_risk,
                rng=m.random,
                target_mask=m.target_mask,
                belief_kernel=m.belief_kernel,
                baseline_agents=m.population_t0,
                observed_norm_share=observed_norm_share,
                observed_norm_share_npi=observed_norm_share_npi,
                social_susceptibility=social_susceptibility,
                norm_threshold=norm_threshold,
                norm_slope=m.cfg.constants.norm_response_slope,
                baseline_descriptive_norm=m.baseline_descriptive_norm,
                baseline_norm_exposure=m.baseline_norm_exposure,
                baseline_share_hat=m.baseline_share_hat,
                baseline_observed_share=m.baseline_expected_obs_share,
                baseline_descriptive_norm_npi=m.baseline_descriptive_norm_npi,
                baseline_norm_exposure_npi=m.baseline_norm_exposure_npi,
                baseline_share_hat_npi=m.baseline_share_hat_npi,
                baseline_observed_share_npi=m.baseline_expected_obs_share_npi,
                endogenous_offsets=endogenous_offsets,
                signal_overrides=m.country_calibrated_signal_overrides(m.current_tick),
                signal_absolute_targets=m.signal_absolute_targets(working_df, m.current_tick),
                norm_channel_weights=m.norm_channel_weights,
                social_noise_std=m.cfg.constants.state_update_noise_std,
                norm_threshold_min=m.cfg.constants.norm_threshold_min,
                norm_threshold_max=m.cfg.constants.norm_threshold_max,
                state_relaxation_fast=relax_fast_substep,
                state_relaxation_slow=relax_slow_substep,
                frozen_columns=(
                    set(m.scenario.frozen_state_columns)
                    | ({"norm_share_hat_vax", "norm_share_hat_npi"} if burn_in_active else set())
                ),
            )
            # Write back — preserve unique_id column from mesa-frames

            if _do_trace:
                _snap_post_update = _snapshot_cols(updated_df)
                # post-peer snapshot is the input to apply_all_state_updates
                _snap_input = _snap_post_peer if ("_snap_post_peer" in dir()) else _snap_pre_peer
                _record_module_delta("state_update_total", _snap_input, _snap_post_update, _substep)
                # Record whether observed-norms and incidence-feedback channels had input
                m._debug_trace_records.append(
                    {
                        "tick": m.current_tick,
                        "substep": _substep,
                        "module": "module_flags",
                        "variable": "_flags_",
                        "delta_l2": 0.0,
                        "delta_mean": 0.0,
                        "mean_before": 0.0,
                        "mean_after": 0.0,
                        "var_before": 0.0,
                        "var_after": 0.0,
                        "observed_norm_share_provided": observed_norm_share is not None,
                        "observed_norm_share_npi_provided": observed_norm_share_npi is not None,
                        "endogenous_offsets_provided": endogenous_offsets is not None,
                        "attractor_targets_provided": attractor_targets is not None,
                        "peer_pull_active": (
                            m.scenario.communication_enabled and m.scenario.peer_pull_enabled and m.network is not None
                        ),
                    }
                )

            uid = self.df["unique_id"]
            self.df = updated_df.with_columns(uid)
            m.timing.update_total += time.perf_counter() - t_update

        if similarity_samples:
            m.set_latest_social_similarity_mean(float(np.mean(np.asarray(similarity_samples, dtype=np.float64))))

        if contact_diag_samples:
            diag_arr = np.array(
                [
                    [
                        d.mean_contacts,
                        d.recurring_share,
                        d.household_share,
                        d.one_shot_share,
                        d.cross_locality_share,
                    ]
                    for d in contact_diag_samples
                ],
                dtype=np.float64,
            )
            averaged = ContactDiagnostics(
                mean_contacts=float(np.mean(diag_arr[:, 0])),
                recurring_share=float(np.mean(diag_arr[:, 1])),
                household_share=float(np.mean(diag_arr[:, 2])),
                one_shot_share=float(np.mean(diag_arr[:, 3])),
                cross_locality_share=float(np.mean(diag_arr[:, 4])),
            )
            m.set_latest_contact_diagnostics(averaged)
