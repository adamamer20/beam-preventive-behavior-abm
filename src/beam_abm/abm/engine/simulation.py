"""Simulation-step and diagnostics methods for VaccineBehaviourModel."""

from __future__ import annotations

import time

import numpy as np
import polars as pl
from loguru import logger

from beam_abm.abm.engine.incidence_helpers import _resolve_vax_outcome_key
from beam_abm.abm.metrics import compute_group_summary_rows, compute_mediator_summary, compute_outcome_summary
from beam_abm.abm.network import ContactDiagnostics
from beam_abm.abm.outcome_scales import OUTCOME_BOUNDS, normalize_outcome_to_unit_interval
from beam_abm.abm.outcomes import predict_all_outcomes
from beam_abm.abm.state_schema import CONSTRUCT_BOUNDS, MUTABLE_COLUMNS
from beam_abm.abm.state_update import SignalSpec


class VaccineBehaviourModelSimulationMixin:
    def _is_signal_effectively_active(self, signal: SignalSpec, *, tick: int) -> bool:
        """Return effective activation, with persistent perturbation worlds.

        In paired perturbation runs, low/high worlds represent persistent
        post-burn anchor states, so once the signal start tick is reached,
        anchor-world effects remain active for the rest of the simulation.
        """
        world = self.cfg.perturbation_world
        if self.cfg.effect_regime == "perturbation" and world in {"low", "high"}:
            return int(tick) >= int(signal.start_tick)
        return bool(signal.is_active(tick))

    def step(self) -> None:
        """Execute one simulation tick.

        1. Record mediator summaries.
        2. Predict all outcomes via the choice module.
        3. Record outcome summaries.
        4. Delegate mutable-state updates to ``SurveyAgents.step()``.
        5. Write predicted outcomes back into agent DataFrame.
        6. Advance tick counter.
        """
        tick = self._current_tick

        # 1. Compute targeting mask for this tick
        if tick == 0 or self._seed_mask is None:
            self._seed_mask = self._compute_target_mask()
            self._target_mask = self._seed_mask
            self._graph_distance_from_seed = self._compute_graph_distance_from_seed(self._seed_mask)
        else:
            self._target_mask = self._seed_mask

        self._record_realised_signals(tick)

        # 2. Outcome prediction
        t_choice = time.perf_counter()
        outcomes = predict_all_outcomes(
            self.agents.df,
            self.backbone_models,
            self.levers_cfg,
            deterministic=self.cfg.deterministic,
            rng=self.random,
            tick=tick,
            choice_signals=self._choice_signals,
            target_mask=self._target_mask,
            norm_channel_weights=self._norm_channel_weights,
        )
        self._timing.choice_total += time.perf_counter() - t_choice

        # 3. Outcome summary
        if tick == 1 and not self._outcome_t0_means:
            for outcome_name, values in outcomes.items():
                self._outcome_t0_means[outcome_name] = float(np.nanmean(values.astype(np.float64)))
        outcome_row = compute_outcome_summary(
            outcomes,
            tick,
            reference_means=self._outcome_t0_means if self._outcome_t0_means else None,
        )
        self._outcome_records.append(outcome_row)

        # 4. Mediator snapshot
        mediator_row = compute_mediator_summary(self.agents.df, tick, outcomes=outcomes)
        self._mediator_records.append(mediator_row)

        # 5. Group trajectories
        self._record_group_trajectories(tick, outcomes)
        self._record_tick_diagnostics(tick, outcomes)

        # 6. Closed-loop environment update before state transitions
        self._latest_outcomes = outcomes
        self._update_community_incidence(outcomes)
        self._latest_social_similarity_mean = 0.0
        self.latest_contact_diagnostics = None

        # 6b. Lazy-init baseline expected observed share from first tick's outcomes
        if self._baseline_expected_obs_share is None:
            self._baseline_expected_obs_share = self._compute_baseline_expected_obs_share(
                outcomes,
                outcome_names={"vax_willingness_T12"},
                fallback_to_all=False,
            )
            if self._baseline_expected_obs_share is not None and self._baseline_expected_obs_share.size > 0:
                self._sync_observed_norm_anchor_bundle(
                    channel="vax",
                    anchor_share=self._baseline_expected_obs_share,
                )
        if self._baseline_expected_obs_share_npi is None:
            self._baseline_expected_obs_share_npi = self._compute_baseline_expected_obs_share(
                outcomes,
                outcome_names={
                    "mask_when_symptomatic_crowded",
                    "stay_home_when_symptomatic",
                    "mask_when_pressure_high",
                },
                fallback_to_all=False,
            )
            if self._baseline_expected_obs_share_npi is not None and self._baseline_expected_obs_share_npi.size > 0:
                self._sync_observed_norm_anchor_bundle(
                    channel="npi",
                    anchor_share=self._baseline_expected_obs_share_npi,
                )

        # 7. State updates (via AgentSet.step)
        self.sets.do("step")

        # 8. Write outcome columns back into agent state
        for outcome_name, values in outcomes.items():
            self.agents.df = self.agents.df.with_columns(pl.Series(name=outcome_name, values=values))

        self.datacollector.collect()

        # 9. Advance tick
        self._current_tick += 1

    def _update_community_incidence(self, outcomes: dict[str, np.ndarray]) -> None:
        if not self.scenario.enable_endogenous_loop:
            return
        if "locality_id" in self.agents.df.columns:
            community = self.agents.df["locality_id"].to_numpy()
        elif "country" in self.agents.df.columns:
            community = self.agents.df["country"].to_numpy()
        else:
            return

        vax_key = _resolve_vax_outcome_key(outcomes)
        if vax_key is not None:
            vax = normalize_outcome_to_unit_interval(outcomes[vax_key], vax_key)
        else:
            vax = np.zeros(community.shape[0], dtype=np.float64)

        npi_terms: list[np.ndarray] = []
        for name in ["mask_when_symptomatic_crowded", "stay_home_when_symptomatic", "mask_when_pressure_high"]:
            values = outcomes.get(name)
            if values is None:
                continue
            npi_terms.append(normalize_outcome_to_unit_interval(values, name))
        if npi_terms:
            npi_protection_index = np.mean(np.vstack(npi_terms), axis=0)
        else:
            npi_protection_index = np.zeros(community.shape[0], dtype=np.float64)

        self._prev_community_incidence = dict(self._community_incidence)
        tick_now = int(self._current_tick)
        multiplier_active = self._is_incidence_forcing_active(tick=tick_now, component="importation_multiplier")
        multiplier = float(self.scenario.importation_multiplier)

        # First pass: compute base next incidence without multiplier-calibration bump.
        community_keys = [str(c) for c in np.unique(community)]
        base_next_by_community: dict[str, float] = {}
        targeted_share_by_community: dict[str, float] = {}
        for key in community_keys:
            idx = community == key
            if not np.any(idx):
                continue
            prev = float(self._community_incidence.get(key, self.cfg.constants.community_incidence_baseline))
            beta = float(self.cfg.knobs.k_incidence)
            gamma = float(self.cfg.knobs.incidence_gamma)
            alpha_v = float(self.cfg.constants.incidence_alpha_vax)
            alpha_p = float(self.cfg.constants.incidence_alpha_protection)
            mean_v = float(np.mean(vax[idx]))
            mean_p = float(np.mean(npi_protection_index[idx]))
            targeted_share = float(np.mean(self._target_mask[idx])) if self._target_mask is not None else 0.0
            targeted_share_by_community[key] = targeted_share
            forcing = self._compute_incidence_forcing(tick=tick_now, targeted_share=targeted_share)
            mitigation_scale = float(np.clip(1.0 - alpha_v * mean_v - alpha_p * mean_p, 0.0, 1.0))
            growth = beta * prev * (1.0 - prev) * mitigation_scale
            base_next_by_community[key] = (1.0 - gamma) * prev + growth + forcing

        if (
            self.scenario.incidence_mode == "importation"
            and multiplier_active
            and multiplier > 1.0
            and base_next_by_community
        ):
            if self.scenario.where == "global":
                # Global outbreak calibration targets mean-incidence peak.
                if self._importation_peak_reference is None:
                    current_vals = np.array(list(self._community_incidence.values()), dtype=np.float64)
                    current_mean = (
                        float(np.nanmean(current_vals))
                        if current_vals.size > 0
                        else float(self.cfg.constants.community_incidence_baseline)
                    )
                    if not np.isfinite(current_mean) or current_mean <= 0.0:
                        current_mean = float(self.cfg.constants.community_incidence_baseline)
                    self._importation_peak_reference = max(current_mean, 1e-6)
                target_peak = multiplier * float(self._importation_peak_reference)
                base_mean = float(np.mean(list(base_next_by_community.values())))
                global_bump = max(0.0, target_peak - base_mean)
                for key, base_next in base_next_by_community.items():
                    self._community_incidence[key] = float(np.clip(base_next + global_bump, 0.0, 1.0))
            else:
                # Local outbreak calibration keeps per-community peak targets.
                if self._importation_peak_reference_by_community is None:
                    refs = {
                        k: float(self._community_incidence.get(k, self.cfg.constants.community_incidence_baseline))
                        for k in base_next_by_community
                    }
                    for k, v in refs.items():
                        if not np.isfinite(v) or v <= 0.0:
                            refs[k] = float(self.cfg.constants.community_incidence_baseline)
                    self._importation_peak_reference_by_community = refs
                for key, base_next in base_next_by_community.items():
                    ref = float(self._importation_peak_reference_by_community.get(key, base_next))
                    target_peak = multiplier * max(ref, 1e-6)
                    bump = max(0.0, target_peak - base_next)
                    if self.scenario.targeting == "targeted":
                        bump *= float(np.clip(targeted_share_by_community.get(key, 0.0), 0.0, 1.0))
                    self._community_incidence[key] = float(np.clip(base_next + bump, 0.0, 1.0))
        else:
            for key, base_next in base_next_by_community.items():
                self._community_incidence[key] = float(np.clip(base_next, 0.0, 1.0))

    def _is_incidence_forcing_active(self, *, tick: int, component: str, default_when_missing: bool = True) -> bool:
        schedule = self.scenario.incidence_schedule
        if not schedule:
            return default_when_missing
        component_rules = tuple(spec for spec in schedule if spec.matches(component))
        if not component_rules:
            return default_when_missing
        return any(spec.is_active(tick) for spec in component_rules)

    def _compute_low_legitimacy_mask(self) -> np.ndarray | None:
        n = self.agents.df.height
        if n == 0:
            return None

        columns = set(self.agents.df.columns)
        legitimacy_score: np.ndarray | None = None
        if "legitimacy_engine" in columns:
            legitimacy_score = self.agents.df["legitimacy_engine"].cast(pl.Float64).to_numpy()
        elif "covax_legitimacy_scepticism_idx" in columns:
            scepticism = self.agents.df["covax_legitimacy_scepticism_idx"].cast(pl.Float64).to_numpy()
            legitimacy_score = -scepticism
        if legitimacy_score is None:
            return None

        cutoff = np.nanpercentile(legitimacy_score, self.cfg.constants.low_legitimacy_target_percentile)
        if not np.isfinite(cutoff):
            return None
        return legitimacy_score <= cutoff

    def _compute_target_mask(self) -> np.ndarray | None:
        if self.scenario.targeting == "uniform":
            return None

        n = self.agents.df.height
        if n == 0:
            return None

        if self.scenario.seed_rule == "high_degree" and self.network is not None:
            degree = self.network.core_degree.astype(np.float64)
            cutoff = np.percentile(degree, self.cfg.constants.high_degree_target_percentile)
            return degree >= cutoff

        if self.scenario.seed_rule in {"high_exposure", "high_degree_high_exposure", "community_targeted"}:
            if "activity_score" in self.agents.df.columns:
                score = self.agents.df["activity_score"].cast(pl.Float64).to_numpy()
            else:
                score = np.zeros(n, dtype=np.float64)
            local = np.ones(n, dtype=bool)
            if self.scenario.seed_rule == "community_targeted" and "locality_id" in self.agents.df.columns:
                communities = self.agents.df["locality_id"].to_numpy().astype(str)
                unique = np.unique(communities)
                means_by_comm: dict[str, float] = {}
                for comm in unique:
                    idx = communities == comm
                    means_by_comm[comm] = float(np.nanmean(score[idx]))
                target_community = max(means_by_comm.items(), key=lambda item: item[1])[0]
                local = communities == target_community
            threshold = np.nanpercentile(score[local], self.cfg.constants.high_exposure_target_percentile)
            mask = np.zeros(n, dtype=bool)
            mask[local] = score[local] >= threshold
            return mask

        if self.scenario.seed_rule == "low_legitimacy":
            return self._compute_low_legitimacy_mask()

        if self.scenario.seed_rule == "low_legitimacy_high_degree":
            low_legitimacy_mask = self._compute_low_legitimacy_mask()
            if low_legitimacy_mask is None:
                return None
            if self.network is None:
                return low_legitimacy_mask
            degree = self.network.core_degree.astype(np.float64)
            degree_cutoff = np.percentile(degree, self.cfg.constants.high_degree_target_percentile)
            return low_legitimacy_mask & (degree >= degree_cutoff)

        return None

    def _compute_graph_distance_from_seed(self, seed_mask: np.ndarray | None) -> np.ndarray | None:
        if seed_mask is None or self.network is None:
            return None
        seed_indices = np.where(seed_mask)[0]
        if seed_indices.size == 0:
            return None

        n = self.agents.df.height
        distance = np.full(n, -1, dtype=np.int32)
        queue = list(seed_indices.astype(int))
        for idx in queue:
            distance[idx] = 0

        head = 0
        while head < len(queue):
            node = queue[head]
            head += 1
            d_next = distance[node] + 1
            core = self.network.core_neighbors[node]
            household = self.network.household_neighbors[node]
            if core.size > 0 and household.size > 0:
                neighbors = np.unique(np.concatenate([core, household]))
            elif core.size > 0:
                neighbors = core
            else:
                neighbors = household
            for nbr in neighbors:
                j = int(nbr)
                if distance[j] != -1:
                    continue
                distance[j] = d_next
                queue.append(j)
        return distance

    def _record_tick_diagnostics(self, tick: int, outcomes: dict[str, np.ndarray]) -> None:
        for col in self.agents.df.columns:
            if col not in {
                "institutional_trust_avg",
                "vaccine_risk_avg",
                "social_norms_descriptive_vax",
            }:
                continue
            values = self.agents.df[col].cast(pl.Float64).to_numpy()
            if not np.isfinite(values).all():
                msg = f"Column '{col}' contains null/NaN/inf values during diagnostics."
                raise ValueError(msg)
            lo, hi = CONSTRUCT_BOUNDS.get(col, (-4.0, 4.0))
            lower_share = float(np.mean(np.isclose(values, lo, atol=self.cfg.constants.saturation_atol)))
            upper_share = float(np.mean(np.isclose(values, hi, atol=self.cfg.constants.saturation_atol)))
            self._saturation_records.append(
                {
                    "tick": tick,
                    "metric": "construct_bound_share",
                    "variable": col,
                    "lower_share": lower_share,
                    "upper_share": upper_share,
                }
            )

        for name, arr in outcomes.items():
            values = arr.astype(np.float64)
            bounds = OUTCOME_BOUNDS.get(name)
            if bounds is None:
                continue
            lo, hi = bounds
            atol = float(self.cfg.constants.saturation_atol)
            extreme_share = float(np.mean((values <= lo + atol) | (values >= hi - atol)))
            self._saturation_records.append(
                {
                    "tick": tick,
                    "metric": "outcome_extreme_share",
                    "variable": name,
                    "extreme_share": extreme_share,
                }
            )

        incidence_values = np.array(list(self._community_incidence.values()), dtype=np.float64)
        contact_diag = self.latest_contact_diagnostics
        avg_similarity = self._latest_social_similarity_mean
        self._incidence_records.append(
            {
                "tick": tick,
                "mean_incidence": float(np.mean(incidence_values)) if incidence_values.size > 0 else 0.0,
                "max_incidence": float(np.max(incidence_values)) if incidence_values.size > 0 else 0.0,
                "mean_contacts": float(contact_diag.mean_contacts) if contact_diag is not None else 0.0,
                "share_household": float(contact_diag.household_share) if contact_diag is not None else 0.0,
                "share_core": float(contact_diag.recurring_share) if contact_diag is not None else 0.0,
                "share_one_shot": float(contact_diag.one_shot_share) if contact_diag is not None else 0.0,
                "share_cross_locality": float(contact_diag.cross_locality_share) if contact_diag is not None else 0.0,
                "avg_similarity_weight": avg_similarity,
            }
        )

        vax_key = _resolve_vax_outcome_key(outcomes)
        if self._graph_distance_from_seed is not None and vax_key is not None:
            distances = self._graph_distance_from_seed
            values = outcomes[vax_key].astype(np.float64)
            max_d = int(np.max(distances[distances >= 0])) if np.any(distances >= 0) else -1
            for d in range(0, min(max_d, self.cfg.constants.distance_tracking_max_hops) + 1):
                idx = distances == d
                if not np.any(idx):
                    continue
                self._distance_trajectory_records.append(
                    {
                        "tick": tick,
                        "distance": d,
                        "mean_vax_willingness": float(np.mean(values[idx])),
                    }
                )

    def _record_realised_signals(self, tick: int) -> None:
        target_mask = self._target_mask
        current_rows: list[dict[str, float | int | str]] = []
        country_overrides = self.country_calibrated_signal_overrides(tick)
        country_overrides_map: dict[str, np.ndarray] = country_overrides if country_overrides is not None else {}
        n_agents = int(self.agents.df.height)

        base_treated_count = 0
        base_treated_share = 0.0
        trust_delta_treated: float | None = None
        trust_delta_others: float | None = None
        if target_mask is not None:
            base_treated_count = int(np.count_nonzero(target_mask))
            if n_agents > 0:
                base_treated_share = float(base_treated_count / n_agents)
            if (
                "institutional_trust_avg" in self.agents.df.columns
                and "institutional_trust_avg" in self._population_t0.columns
            ):
                trust_current = self.agents.df["institutional_trust_avg"].cast(pl.Float64).to_numpy().astype(np.float64)
                trust_baseline = (
                    self._population_t0["institutional_trust_avg"].cast(pl.Float64).to_numpy().astype(np.float64)
                )
                trust_delta = trust_current - trust_baseline
                if base_treated_count > 0:
                    trust_delta_treated = float(np.mean(trust_delta[target_mask]))
                others_count = int(np.count_nonzero(~target_mask))
                if others_count > 0:
                    trust_delta_others = float(np.mean(trust_delta[~target_mask]))

        for signal, calibrated_signal in zip(self._state_signals, self._state_signals_calibrated, strict=True):
            realised_nominal = float(signal.target_offset(tick))
            realised_base = float(calibrated_signal.target_offset(tick))
            if signal.target_column in country_overrides_map:
                realised = realised_base + float(np.mean(country_overrides_map[signal.target_column]))
            else:
                realised = realised_base
            targeted_mean: float | None = None
            others_mean: float | None = None
            active = self._is_signal_effectively_active(signal, tick=tick)
            treated_share = 1.0 if active else 0.0
            treated_count = n_agents if active else 0
            if target_mask is not None:
                targeted_mean = realised if int(np.count_nonzero(target_mask)) > 0 else 0.0
                others_mean = 0.0 if int(np.count_nonzero(~target_mask)) > 0 else None
                treated_share = base_treated_share if active else 0.0
                treated_count = base_treated_count if active else 0

            row = {
                "tick": tick,
                "signal_id": signal.signal_id or f"{signal.target_column}_{signal.signal_type}",
                "target_column": signal.target_column,
                "signal_type": signal.signal_type,
                "effect_regime": self.cfg.effect_regime,
                "perturbation_world": self.cfg.perturbation_world,
                "intensity_nominal": float(signal.intensity),
                "intensity_calibrated": float(calibrated_signal.intensity),
                "intensity_realised_nominal_mean": realised_nominal,
                "intensity_realised_mean": realised,
                "intensity_realised_targeted_mean": targeted_mean,
                "intensity_realised_others_mean": others_mean,
                "treated_share": treated_share,
                "treated_count": treated_count,
                "mean_delta_institutional_trust_treated": trust_delta_treated,
                "mean_delta_institutional_trust_others": trust_delta_others,
            }
            self._signal_records.append(row)
            current_rows.append(row)
        self._latest_signals_realised = current_rows

    def country_calibrated_signal_overrides(self, tick: int) -> dict[str, np.ndarray] | None:
        if self.cfg.effect_regime != "shock":
            return None
        if "country" not in self.agents.df.columns:
            return None

        n_agents = self.agents.df.height
        countries = self.agents.df["country"].cast(pl.Utf8).to_numpy()
        overrides: dict[str, np.ndarray] = {}

        for signal, calibrated_signal in zip(self._state_signals, self._state_signals_calibrated, strict=True):
            nominal_offset = float(signal.target_offset(tick))
            if nominal_offset == 0.0:
                continue
            global_offset = float(calibrated_signal.target_offset(tick))
            arr = overrides.setdefault(signal.target_column, np.zeros(n_agents, dtype=np.float64))
            for country in np.unique(countries):
                idx = countries == country
                country_offset = self._intensity_calibrator.calibrate_intensity(
                    signal.target_column,
                    nominal_offset,
                    country=str(country),
                    effect_regime=self.cfg.effect_regime,
                )
                arr[idx] = arr[idx] + float(country_offset - global_offset)

        return overrides if overrides else None

    def signal_absolute_targets(self, agents: pl.DataFrame, tick: int) -> dict[str, np.ndarray] | None:
        """Resolve absolute mutable targets for active perturbation-world signals."""
        world = self.cfg.perturbation_world
        if self.cfg.effect_regime != "perturbation" or world not in {"low", "high"}:
            return None

        targets: dict[str, np.ndarray] = {}
        for signal in self._state_signals:
            if not self._is_signal_effectively_active(signal, tick=tick):
                continue
            column = signal.target_column
            if column not in MUTABLE_COLUMNS:
                continue
            target_values = self._anchor_world_targets_for_column(agents, column)
            if target_values is None:
                continue
            targets[column] = target_values
        return targets if targets else None

    def _record_group_trajectories(
        self,
        tick: int,
        outcomes: dict[str, np.ndarray],
    ) -> None:
        group_specs: list[tuple[str, str]] = []
        outcome_rows_current: list[dict[str, float | int | str]] = []
        mediator_rows_current: list[dict[str, float | int | str]] = []

        runtime_agents = self.agents.df
        if self._target_mask is not None:
            targeted_values = np.where(self._target_mask, "targeted", "other")
            runtime_agents = runtime_agents.with_columns(pl.Series("_targeted_flag_runtime", targeted_values))
            group_specs.append(("targeted_flag", "_targeted_flag_runtime"))

        if "anchor_profile" in runtime_agents.columns:
            group_specs.append(("anchor_profile", "anchor_profile"))
        elif "anchor_id" in runtime_agents.columns:
            group_specs.append(("anchor_profile", "anchor_id"))

        if "country" in runtime_agents.columns:
            group_specs.append(("country", "country"))

        for group_type, group_column in group_specs:
            outcome_rows = compute_group_summary_rows(
                runtime_agents,
                tick,
                group_type=group_type,
                group_column=group_column,
                values=outcomes,
            )
            mediator_rows = compute_group_summary_rows(
                runtime_agents,
                tick,
                group_type=group_type,
                group_column=group_column,
                values=None,
            )

            outcome_rows_current.extend(outcome_rows)
            mediator_rows_current.extend(mediator_rows)

            for row in outcome_rows:
                self._outcomes_by_group_records.append(row)

            for row in mediator_rows:
                self._mediators_by_group_records.append(row)

        self._latest_outcomes_by_group = outcome_rows_current
        self._latest_mediators_by_group = mediator_rows_current

    def run_model(self, n_steps: int | None = None) -> None:
        """Run the simulation for *n_steps* ticks.

        Parameters
        ----------
        n_steps : int or None
            Number of ticks to run.  Defaults to ``cfg.n_steps``.
        """
        if n_steps is None:
            n_steps = self.cfg.n_steps

        t0 = time.perf_counter()
        for _ in range(n_steps):
            self.step()

        total = time.perf_counter() - t0
        self._timing.total = total
        logger.info(
            "{n_steps} ticks in {t:.2f}s (choice={tc:.2f}s, update={tu:.2f}s, social={ts:.2f}s).",
            n_steps=n_steps,
            t=total,
            tc=self._timing.choice_total,
            tu=self._timing.update_total,
            ts=self._timing.social_total,
        )

    def set_freeze_state_updates(self, freeze: bool) -> None:
        """Enable or disable mutable-state updates (mechanical gate use)."""
        self._freeze_state_updates = freeze

    def set_latest_contact_diagnostics(self, diagnostics: ContactDiagnostics | None) -> None:
        """Set tick-level contact diagnostics from the contact sampler."""
        self.latest_contact_diagnostics = diagnostics

    def set_latest_social_similarity_mean(self, value: float) -> None:
        """Set tick-level mean homophily similarity used in diffusion."""
        self._latest_social_similarity_mean = float(value)
