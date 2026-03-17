"""Initialization and baseline-state methods for VaccineBehaviourModel."""

from __future__ import annotations

import json

import numpy as np
import polars as pl
from loguru import logger
from mesa_frames import DataCollector

from beam_abm.abm.belief_update_kernel import BeliefUpdateKernel
from beam_abm.abm.belief_update_surrogate import BeliefUpdateSurrogate
from beam_abm.abm.config import SimulationConfig
from beam_abm.abm.data_contracts import BackboneModel, LeversConfig
from beam_abm.abm.engine.incidence_helpers import _importation_pulse_at_tick
from beam_abm.abm.engine.social_helpers import _agent_norm_observation_targets_per_month, _estimate_contact_rate_base
from beam_abm.abm.engine.survey_agents import SurveyAgents
from beam_abm.abm.engine.types import TimingBreakdown
from beam_abm.abm.intensity_calibration import (
    IntensityCalibrator,
    build_anchoring_diagnostics,
    build_country_iqr_scales,
    build_country_std_scales,
    build_iqr_scales,
    build_std_scales,
    calibrate_signal,
    load_anchor_perturbation_levels,
    load_anchor_perturbation_scales,
)
from beam_abm.abm.message_contracts import NormChannelWeights
from beam_abm.abm.metrics import SummaryRow
from beam_abm.abm.network import NetworkState
from beam_abm.abm.outcome_scales import normalize_outcome_to_unit_interval
from beam_abm.abm.scenario_defs import ScenarioConfig
from beam_abm.abm.state_ref_attractor import build_state_ref_attractor
from beam_abm.abm.state_schema import CONSTRUCT_BOUNDS, MUTABLE_COLUMNS
from beam_abm.abm.state_update import compute_observed_norm_exposure, compute_observed_norm_fixed_point_share


class VaccineBehaviourModelInitializationMixin:
    def __init__(
        self,
        cfg: SimulationConfig,
        scenario: ScenarioConfig,
        models: dict[str, BackboneModel],
        levers_cfg: LeversConfig,
        agent_df: pl.DataFrame,
        network: NetworkState | None = None,
        belief_kernel: BeliefUpdateKernel | BeliefUpdateSurrogate | None = None,
        *,
        seed: int = 42,
    ) -> None:
        super().__init__(seed=seed)

        # Configuration
        self.cfg = cfg
        self.scenario = scenario
        self.backbone_models = models
        self.levers_cfg = levers_cfg

        # Network
        self._network = network

        # Signals
        self._state_signals = list(scenario.state_signals)
        self._state_signals_calibrated = list(scenario.state_signals)
        self._choice_signals = list(scenario.choice_signals)
        self._norm_channel_weights = self._build_norm_channel_weights(scenario)

        # Norm columns present in the agent DataFrame
        self._norm_columns = sorted(
            c
            for c in [
                "social_norms_vax_avg",
                "social_norms_descriptive_vax",
                "social_norms_descriptive_npi",
                "institutional_trust_avg",
                "group_trust_vax",
            ]
            if c in agent_df.columns
        )

        # Initialise agents via mesa-frames AgentSet
        self.agents = SurveyAgents(self, agent_df)
        self.agents.df = self._ensure_norm_share_hat_state(self.agents.df)
        self.sets += self.agents

        # Tick counter and timing
        self._current_tick = 0
        self._timing = TimingBreakdown()

        # Targeting and update controls
        self._target_mask = None
        self._freeze_state_updates = not self.scenario.state_updates_enabled
        self._belief_kernel = belief_kernel
        self._state_ref_attractor = None
        # Build initial attractor from survey snapshot.
        self._population_t0 = self.agents.df.clone()
        self._refresh_state_ref_attractor()
        self.cfg.constants.contact_rate_base = _estimate_contact_rate_base(self._population_t0)

        # Snapshot t=0
        self._baseline_observed_norm_share = self._compute_baseline_observed_norm_share(self._population_t0)
        self._baseline_observed_norm_share_npi = self._compute_baseline_observed_norm_share_npi(self._population_t0)
        self._baseline_share_hat = self._extract_baseline_share_hat(self._population_t0)
        self._baseline_share_hat_npi = self._extract_baseline_share_hat_npi(self._population_t0)
        self._baseline_expected_obs_share = None  # lazy-init from first tick's outcomes
        self._baseline_expected_obs_share_npi = None  # lazy-init from first tick's outcomes
        self._baseline_expected_obs_share_updates = 0
        self._baseline_expected_obs_share_npi_updates = 0
        self._baseline_descriptive_norm, self._baseline_norm_exposure = self._compute_baseline_norm_anchors(
            self._population_t0
        )
        self._baseline_descriptive_norm_npi, self._baseline_norm_exposure_npi = self._compute_baseline_norm_anchors_npi(
            self._population_t0
        )
        self._precursor_shock_specs = self._load_precursor_shock_specs()
        self._validate_state_signal_precursor_contract()
        anchor_global_scales, anchor_country_scales = load_anchor_perturbation_scales(self.cfg.perturbation_anchor_root)
        empirical_global_iqr_scales = build_iqr_scales(self._population_t0)
        empirical_country_iqr_scales = build_country_iqr_scales(self._population_t0)
        self._anchor_level_table = load_anchor_perturbation_levels(self.cfg.perturbation_anchor_root)
        perturbation_global_scales = dict(anchor_global_scales)
        perturbation_country_scales: dict[str, dict[str, float]] = {
            country: dict(scales) for country, scales in anchor_country_scales.items()
        }
        # Fallback rule: when PE anchor mappings are unavailable, use
        # country-stratified empirical IQR scales (and global IQR as fallback).
        for country, scales in empirical_country_iqr_scales.items():
            country_scale_map = perturbation_country_scales.setdefault(country, {})
            for lever, scale in scales.items():
                country_scale_map.setdefault(lever, float(scale))
        for lever, scale in empirical_global_iqr_scales.items():
            perturbation_global_scales.setdefault(lever, float(scale))
        if anchor_global_scales:
            logger.info(
                (
                    "Loaded PE-anchored perturbation scales for {n} lever(s) "
                    "from {root}; missing levers use empirical country/global IQR fallback."
                ),
                n=len(anchor_global_scales),
                root=self.cfg.perturbation_anchor_root,
            )
        else:
            logger.info(
                ("No PE-anchored perturbation scales found at {root}; using empirical country/global IQR scales."),
                root=self.cfg.perturbation_anchor_root,
            )
        country_std_scales = build_country_std_scales(self._population_t0)
        self._intensity_calibrator = IntensityCalibrator(
            global_scales=perturbation_global_scales,
            country_scales=perturbation_country_scales or None,
            global_std_scales=build_std_scales(self._population_t0),
            country_std_scales=country_std_scales or None,
            anchor_levels=self._anchor_level_table,
        )
        self._state_signals_calibrated = [
            calibrate_signal(
                signal,
                self._intensity_calibrator,
                effect_regime=self.cfg.effect_regime,
            )
            for signal in self._state_signals
        ]
        self._anchoring_diagnostics = build_anchoring_diagnostics(self._population_t0)
        self._precursor_column_stats_cache = {}
        self._outcome_t0_means = {}

        # Trajectory accumulators
        self._outcome_records: list[SummaryRow] = []
        self._mediator_records: list[SummaryRow] = []
        self._signal_records = []
        self._outcomes_by_group_records = []
        self._mediators_by_group_records = []
        self._latest_signals_realised = []
        self._debug_trace_records = []
        self._latest_outcomes_by_group = []
        self._latest_mediators_by_group = []
        self._latest_outcomes = None
        self._latest_social_similarity_mean = 0.0
        self.latest_contact_diagnostics = None
        self._seed_mask = None
        self._graph_distance_from_seed = None
        self._saturation_records = []
        self._incidence_records = []
        self._distance_trajectory_records = []

        if "locality_id" in self.agents.df.columns:
            units = [str(c) for c in np.unique(self.agents.df["locality_id"].to_numpy())]
        elif "country" in self.agents.df.columns:
            units = [str(c) for c in np.unique(self.agents.df["country"].to_numpy())]
        else:
            units = ["global"]
        self._community_incidence = dict.fromkeys(units, self.cfg.constants.community_incidence_baseline)
        self._prev_community_incidence = dict(self._community_incidence)
        self._importation_peak_reference = None
        self._importation_peak_reference_by_community = None

        self.datacollector = DataCollector(
            model=self,
            model_reporters={
                "tick": lambda model: model.current_tick,
            },
        )

    @staticmethod
    def _build_norm_channel_weights(scenario: ScenarioConfig) -> NormChannelWeights:
        """Resolve scenario-specific norm-composition weights."""
        if scenario.norm_descriptive_share_w_d is None:
            return NormChannelWeights()
        return NormChannelWeights.from_descriptive_share(scenario.norm_descriptive_share_w_d).normalized()

    def _refresh_state_ref_attractor(self) -> None:
        """Rebuild the survey-anchored state attractor from current t0 population."""
        self._state_ref_attractor = build_state_ref_attractor(
            agents_t0=self._population_t0,
            modeling_dir=self.cfg.modeling_dir,
            spec_path=self.cfg.state_ref_spec_path,
            strict=bool(self.cfg.state_ref_strict),
        )

    def _extract_baseline_share_hat(self, agents: pl.DataFrame) -> np.ndarray | None:
        """Extract t0 perceived-adoption share from the snapshot for deviation-based EMA."""
        if "norm_share_hat_vax" not in agents.columns:
            return None
        return agents["norm_share_hat_vax"].cast(pl.Float64).to_numpy().astype(np.float64)

    def _extract_baseline_share_hat_npi(self, agents: pl.DataFrame) -> np.ndarray | None:
        """Extract t0 perceived NPI-adoption share from the snapshot."""
        if "norm_share_hat_npi" not in agents.columns:
            return None
        return agents["norm_share_hat_npi"].cast(pl.Float64).to_numpy().astype(np.float64)

    def _compute_baseline_expected_obs_share(
        self,
        outcomes: dict[str, np.ndarray],
        agents: pl.DataFrame | None = None,
        *,
        n_draws: int = 8,
        outcome_names: set[str] | None = None,
        fallback_to_all: bool = True,
    ) -> np.ndarray | None:
        """Estimate per-agent expected observed share under baseline behavior.

        This provides the ``obs_ref`` anchor for deviation-based share-hat EMA.
        Crucially, the anchor is *agent-specific* (network/locality/contact
        heterogeneity), not a single global mean.
        """
        if not outcomes:
            return None

        if outcome_names is None:
            eff_outcomes = outcomes
        else:
            selected_outcomes = {k: v for k, v in outcomes.items() if k in outcome_names}
            if selected_outcomes:
                eff_outcomes = selected_outcomes
            elif fallback_to_all:
                eff_outcomes = outcomes
            else:
                return None

        unit_arrays: list[np.ndarray] = []
        for name, values in eff_outcomes.items():
            if not isinstance(values, np.ndarray):
                continue
            unit_arrays.append(normalize_outcome_to_unit_interval(values, name))
        if not unit_arrays:
            return None

        protection_index = np.mean(np.vstack(unit_arrays), axis=0)
        n_agents = protection_index.shape[0]
        if n_agents == 0:
            return np.array([], dtype=np.float64)

        # Fallback if network/contact layer is unavailable.
        if self.network is None:
            pop_mean = float(np.mean(protection_index))
            return np.full(n_agents, pop_mean, dtype=np.float64)

        agent_df = self.agents.df if agents is None else agents

        def _col(name: str, default: float) -> np.ndarray:
            if name not in agent_df.columns:
                return np.full(n_agents, default, dtype=np.float64)
            arr = agent_df[name].cast(pl.Float64).to_numpy().astype(np.float64)
            return np.nan_to_num(arr, nan=default)

        # Keep baseline expected-observation sampling aligned with runtime
        # observed-norm sampling columns; fall back to legacy aliases if needed.
        if "social_attention_alpha" in agent_df.columns:
            attention = np.clip(_col("social_attention_alpha", 1.0), 0.0, np.inf)
        else:
            attention = np.clip(_col("social_attention_pi", 1.0), 0.0, np.inf)
        if "social_sharing_phi" in agent_df.columns:
            sharing = np.clip(_col("social_sharing_phi", 1.0), 0.0, np.inf)
        else:
            sharing = np.clip(_col("social_sharing_tau", 1.0), 0.0, np.inf)
        activity = _col("activity_score", 0.0)
        propensity = np.clip(_col("contact_propensity", 1.0), 0.1, 5.0)

        obs_targets_month = _agent_norm_observation_targets_per_month(
            agent_df,
            min_per_month=self.cfg.constants.norm_observation_min_per_month,
            max_per_month=self.cfg.constants.norm_observation_max_per_month,
        )
        substeps_per_month = max(1, int(self.cfg.constants.social_substeps_per_month))
        obs_targets_substep = np.maximum(
            1,
            np.ceil(obs_targets_month.astype(np.float64) / float(substeps_per_month)).astype(np.int64),
        )
        obs_targets_substep = np.minimum(obs_targets_substep, int(self.cfg.norm_observation_sample_size))

        # Use a dedicated RNG so this anchor estimation does not perturb the
        # model's main stochastic stream.
        mc_rng = np.random.default_rng(int(self.cfg.seed) + 10007)
        from beam_abm.abm import engine as engine_module

        accum = np.zeros(n_agents, dtype=np.float64)
        draws = int(max(1, n_draws))
        for _ in range(draws):
            observed_share, _ = engine_module.compute_sampled_peer_adoption(
                self.network,
                eff_outcomes,
                sample_size=self.cfg.norm_observation_sample_size,
                rng=mc_rng,
                attention=attention,
                sharing=sharing,
                contact_rate_base=self.cfg.constants.contact_rate_base,
                contact_rate_multiplier=self.cfg.knobs.contact_rate_multiplier,
                activity_score=activity,
                contact_propensity=propensity,
                per_agent_sample_size=obs_targets_substep,
            )
            accum += np.asarray(observed_share, dtype=np.float64)

        return np.clip(accum / float(draws), 0.0, 1.0)

    def _compute_baseline_observed_norm_share(self, agents: pl.DataFrame) -> np.ndarray | None:
        """Build fixed-point observed-share state from baseline descriptive norms."""
        return self._compute_baseline_observed_norm_share_for_column(agents, "social_norms_descriptive_vax")

    def _compute_baseline_observed_norm_share_npi(self, agents: pl.DataFrame) -> np.ndarray | None:
        """Build fixed-point observed-share state for the NPI descriptive norm."""
        return self._compute_baseline_observed_norm_share_for_column(agents, "social_norms_descriptive_npi")

    def _compute_baseline_observed_norm_share_for_column(self, agents: pl.DataFrame, column: str) -> np.ndarray | None:
        if column not in agents.columns:
            return None

        n_agents = agents.height
        if n_agents == 0:
            return np.array([], dtype=np.float64)

        descriptive = agents[column].cast(pl.Float64).to_numpy().astype(np.float64)
        if "norm_threshold_theta" in agents.columns:
            threshold = agents["norm_threshold_theta"].cast(pl.Float64).to_numpy().astype(np.float64)
        else:
            threshold = np.full(n_agents, 0.5, dtype=np.float64)

        threshold = np.nan_to_num(threshold, nan=0.5, posinf=0.5, neginf=0.5)
        threshold = np.clip(
            threshold,
            float(self.cfg.constants.norm_threshold_min),
            float(self.cfg.constants.norm_threshold_max),
        )

        descriptive = np.nan_to_num(descriptive, nan=4.0, posinf=7.0, neginf=1.0)
        return compute_observed_norm_fixed_point_share(
            descriptive_norm=descriptive,
            threshold=threshold,
            column=column,
            slope=float(self.cfg.constants.norm_response_slope),
        )

    def _ensure_norm_share_hat_state(self, agents: pl.DataFrame) -> pl.DataFrame:
        """Ensure latent perceived adoption share is coherent with baseline norms."""
        if (
            "social_norms_descriptive_vax" not in agents.columns
            and "social_norms_descriptive_npi" not in agents.columns
        ):
            return agents

        n_agents = agents.height
        if n_agents == 0:
            return agents.with_columns(
                pl.Series("norm_share_hat_vax", np.array([], dtype=np.float64)),
                pl.Series("norm_share_hat_npi", np.array([], dtype=np.float64)),
            )

        if "norm_threshold_theta" in agents.columns:
            threshold = agents["norm_threshold_theta"].cast(pl.Float64).to_numpy().astype(np.float64)
        else:
            threshold = np.full(n_agents, 0.5, dtype=np.float64)
        threshold = np.nan_to_num(threshold, nan=0.5, posinf=0.5, neginf=0.5)
        threshold = np.clip(
            threshold,
            float(self.cfg.constants.norm_threshold_min),
            float(self.cfg.constants.norm_threshold_max),
        )
        updates: list[pl.Series] = []
        if "social_norms_descriptive_vax" in agents.columns:
            descriptive_vax = agents["social_norms_descriptive_vax"].cast(pl.Float64).to_numpy().astype(np.float64)
            descriptive_vax = np.clip(np.nan_to_num(descriptive_vax, nan=4.0, posinf=7.0, neginf=1.0), 1.0, 7.0)
            share_hat_vax = compute_observed_norm_fixed_point_share(
                descriptive_norm=descriptive_vax,
                threshold=threshold,
                column="social_norms_descriptive_vax",
                slope=float(self.cfg.constants.norm_response_slope),
            )
            updates.append(pl.Series("norm_share_hat_vax", np.clip(share_hat_vax, 0.0, 1.0).astype(np.float64)))
        if "social_norms_descriptive_npi" in agents.columns:
            descriptive_npi = agents["social_norms_descriptive_npi"].cast(pl.Float64).to_numpy().astype(np.float64)
            descriptive_npi = np.clip(np.nan_to_num(descriptive_npi, nan=4.0, posinf=7.0, neginf=1.0), 1.0, 7.0)
            share_hat_npi = compute_observed_norm_fixed_point_share(
                descriptive_norm=descriptive_npi,
                threshold=threshold,
                column="social_norms_descriptive_npi",
                slope=float(self.cfg.constants.norm_response_slope),
            )
            updates.append(pl.Series("norm_share_hat_npi", np.clip(share_hat_npi, 0.0, 1.0).astype(np.float64)))
        if not updates:
            return agents
        return agents.with_columns(updates)

    def _compute_baseline_norm_anchors(self, agents: pl.DataFrame) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Compute baseline descriptive norms and implied baseline exposure."""
        return self._compute_baseline_norm_anchors_for_column(agents, column="social_norms_descriptive_vax")

    def _compute_baseline_norm_anchors_npi(self, agents: pl.DataFrame) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Compute baseline NPI descriptive norms and implied baseline exposure."""
        return self._compute_baseline_norm_anchors_for_column(agents, column="social_norms_descriptive_npi")

    def _compute_baseline_norm_anchors_for_column(
        self,
        agents: pl.DataFrame,
        *,
        column: str,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        if column not in agents.columns:
            return None, None

        n_agents = agents.height
        if n_agents == 0:
            empty = np.array([], dtype=np.float64)
            return empty, empty

        if "norm_threshold_theta" in agents.columns:
            threshold = agents["norm_threshold_theta"].cast(pl.Float64).to_numpy().astype(np.float64)
        else:
            threshold = np.full(n_agents, 0.5, dtype=np.float64)
        threshold = np.nan_to_num(threshold, nan=0.5, posinf=0.5, neginf=0.5)
        threshold = np.clip(
            threshold,
            float(self.cfg.constants.norm_threshold_min),
            float(self.cfg.constants.norm_threshold_max),
        )

        lo, hi = CONSTRUCT_BOUNDS.get(column, (1.0, 7.0))
        descriptive = agents[column].cast(pl.Float64).to_numpy().astype(np.float64)
        descriptive = np.clip(np.nan_to_num(descriptive, nan=4.0, posinf=hi, neginf=lo), lo, hi)

        baseline_share = compute_observed_norm_fixed_point_share(
            descriptive_norm=descriptive,
            threshold=threshold,
            column=column,
            slope=float(self.cfg.constants.norm_response_slope),
        )
        baseline_exposure = compute_observed_norm_exposure(
            observed_share=baseline_share,
            threshold=threshold,
            slope=float(self.cfg.constants.norm_response_slope),
        )
        return descriptive.astype(np.float64), np.clip(baseline_exposure, 0.0, 1.0).astype(np.float64)

    def _compute_incidence_forcing(self, *, tick: int, targeted_share: float) -> float:
        burn_in = int(self.cfg.constants.burn_in_ticks)

        importation = 0.0
        if self.scenario.incidence_mode == "importation" and self._is_incidence_forcing_active(
            tick=tick,
            component="importation_floor",
        ):
            pulse = 0.0
            pulse_delta = 0.0
            if tick >= burn_in:
                # Shift pulse clock so the first pulse fires after burn-in.
                effective_tick = tick - burn_in
                pulse = _importation_pulse_at_tick(
                    tick=effective_tick,
                    amplitude=float(self.cfg.constants.importation_pulse_amplitude),
                    period_ticks=int(self.cfg.constants.importation_pulse_period_ticks),
                    width_ticks=int(self.cfg.constants.importation_pulse_width_ticks),
                    start_tick=int(self.cfg.constants.importation_pulse_start_tick),
                )
                pulse_delta = _importation_pulse_at_tick(
                    tick=effective_tick,
                    amplitude=float(self.scenario.importation_pulse_delta),
                    period_ticks=int(self.cfg.constants.importation_pulse_period_ticks),
                    width_ticks=int(self.cfg.constants.importation_pulse_width_ticks),
                    start_tick=int(self.cfg.constants.importation_pulse_start_tick),
                )
                if not self._is_incidence_forcing_active(
                    tick=tick,
                    component="importation_pulse_delta",
                ):
                    pulse_delta = 0.0
            importation = float(self.cfg.constants.importation_outbreak_floor) + pulse + pulse_delta

        local_bump = 0.0
        if (
            self.scenario.incidence_mode == "importation"
            and self.scenario.where == "local"
            and self.scenario.targeting == "targeted"
            and self._is_incidence_forcing_active(tick=tick, component="local_bump")
        ):
            local_bump = (
                float(self.cfg.constants.importation_targeted_local_bump)
                * float(targeted_share)
                * float(self.scenario.local_bump_multiplier)
            )

        return float(importation + local_bump)

    def _load_precursor_shock_specs(self) -> dict[str, dict[str, object]]:
        """Load precursor shock metadata keyed by signal id from state-ref spec."""
        path = self.cfg.state_ref_spec_path
        if path is None or not path.exists():
            return {}

        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}

        signals_raw = raw.get("signals") if isinstance(raw, dict) else None
        if not isinstance(signals_raw, list):
            return {}

        out: dict[str, dict[str, object]] = {}
        for entry in signals_raw:
            if not isinstance(entry, dict):
                continue
            signal_id_raw = entry.get("id")
            shock_raw = entry.get("shock")
            if not isinstance(signal_id_raw, str) or not isinstance(shock_raw, dict):
                continue
            var_raw = shock_raw.get("var")
            shock_type_raw = shock_raw.get("type")
            if not isinstance(var_raw, str) or not var_raw.strip():
                continue
            shock_type = str(shock_type_raw or "continuous").strip().lower()
            if shock_type not in {"binary", "continuous"}:
                continue
            out[signal_id_raw] = {
                "var": var_raw,
                "type": shock_type,
                "from": shock_raw.get("from"),
                "to": shock_raw.get("to"),
            }
        return out

    def _validate_state_signal_precursor_contract(self) -> None:
        """Fail fast when scenario state signals do not resolve to precursor X vars."""
        if not self._state_signals:
            return
        errors: list[str] = []
        spec_path = self.cfg.state_ref_spec_path
        for signal in self._state_signals:
            if signal.target_column in MUTABLE_COLUMNS:
                continue
            signal_id = signal.signal_id or ""
            shock_spec = self._precursor_shock_specs.get(signal_id)
            if shock_spec is None:
                errors.append(
                    f"signal '{signal_id or signal.target_column}' has no shock spec in "
                    f"{spec_path if spec_path is not None else 'state_ref_spec'}"
                )
                continue
            target_column = str(shock_spec["var"])
            if target_column in MUTABLE_COLUMNS:
                errors.append(
                    f"signal '{signal_id}' resolves to mutable column '{target_column}' "
                    "but non-mutable-targeted state signals must resolve to precursor X columns"
                )
            if target_column not in self._population_t0.columns:
                errors.append(f"signal '{signal_id}' precursor column '{target_column}' not found in population data")
        if errors:
            msg = "Invalid state signal precursor contract:\n- " + "\n- ".join(errors)
            raise ValueError(msg)

    def _precursor_column_stats(self, column: str) -> dict[str, float] | None:
        cached = self._precursor_column_stats_cache.get(column)
        if cached is not None:
            return cached
        if column not in self._population_t0.columns:
            return None
        try:
            values = self._population_t0[column].cast(pl.Float64).to_numpy().astype(np.float64)
        except (pl.exceptions.InvalidOperationError, TypeError, ValueError):
            return None
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            return None
        stats = {
            "min": float(np.min(finite)),
            "max": float(np.max(finite)),
            "q10": float(np.percentile(finite, 10.0)),
            "q25": float(np.percentile(finite, 25.0)),
            "q50": float(np.percentile(finite, 50.0)),
            "q75": float(np.percentile(finite, 75.0)),
            "q90": float(np.percentile(finite, 90.0)),
            "std": float(np.std(finite)),
            "iqr": float(np.percentile(finite, 75.0) - np.percentile(finite, 25.0)),
        }
        self._precursor_column_stats_cache[column] = stats
        return stats

    def _resolve_precursor_endpoint(self, column: str, raw_value: object) -> float | None:
        if raw_value is None:
            return None
        if isinstance(raw_value, int | float):
            value = float(raw_value)
            return value if np.isfinite(value) else None
        if not isinstance(raw_value, str):
            return None
        token = raw_value.strip().lower()
        if not token:
            return None

        stats = self._precursor_column_stats(column)
        if token in {"median", "q50"}:
            return None if stats is None else float(stats["q50"])
        if token in {"q10", "q25", "q75", "q90"}:
            return None if stats is None else float(stats[token])
        if token in {"min", "max"}:
            return None if stats is None else float(stats[token])

        try:
            value = float(token)
        except ValueError:
            return None
        return value if np.isfinite(value) else None

    def _precursor_shock_delta(
        self,
        column: str,
        shock_spec: dict[str, object],
        *,
        effect_regime: str,
    ) -> float | None:
        from_value = self._resolve_precursor_endpoint(column, shock_spec.get("from"))
        to_value = self._resolve_precursor_endpoint(column, shock_spec.get("to"))

        if effect_regime == "shock":
            stats = self._precursor_column_stats(column)
            if stats is None:
                return None
            iqr = float(stats.get("iqr", 0.0))
            if not np.isfinite(iqr) or iqr <= 0.0:
                return None

            direction = 0.0
            if from_value is not None and to_value is not None:
                direction = float(np.sign(to_value - from_value))
            if direction == 0.0:
                direction_token = str(shock_spec.get("direction") or "").strip().lower()
                if direction_token in {"down", "decrease", "negative"}:
                    direction = -1.0
                elif direction_token in {"up", "increase", "positive"}:
                    direction = 1.0
            if direction == 0.0 and to_value is not None:
                direction = float(np.sign(to_value))
            if direction == 0.0:
                return None
            return float(direction * iqr)

        if from_value is not None and to_value is not None:
            return float(to_value - from_value)
        if to_value is not None:
            return float(to_value)
        return None

    def _anchor_world_targets_for_column(self, agents: pl.DataFrame, column: str) -> np.ndarray | None:
        """Resolve per-agent absolute anchor targets for perturbation world."""
        world = self.cfg.perturbation_world
        if self.cfg.effect_regime != "perturbation" or world not in {"low", "high"}:
            return None
        if column not in agents.columns:
            return None
        if "country" not in agents.columns:
            value = self._intensity_calibrator.resolve_anchor_world_level(
                column,
                country=None,
                world=world,
            )
            if value is None:
                return None
            return np.full(agents.height, float(value), dtype=np.float64)

        countries = agents["country"].cast(pl.Utf8).to_numpy()
        targets = np.full(agents.height, np.nan, dtype=np.float64)
        for country in np.unique(countries):
            idx = countries == country
            value = self._intensity_calibrator.resolve_anchor_world_level(
                column,
                country=str(country),
                world=world,
            )
            if value is None:
                value = self._intensity_calibrator.resolve_anchor_world_level(
                    column,
                    country=None,
                    world=world,
                )
            if value is None:
                continue
            targets[idx] = float(value)
        if not np.isfinite(targets).any():
            return None
        if not np.isfinite(targets).all():
            current = agents[column].cast(pl.Float64).to_numpy().astype(np.float64)
            targets = np.where(np.isfinite(targets), targets, current)
        return targets

    def apply_state_signals_to_precursors(
        self,
        agents: pl.DataFrame,
        *,
        tick: int,
        target_mask: np.ndarray | None,
    ) -> pl.DataFrame:
        """Apply active state signals to precursor X-columns (not mutable state).

        Signals are evaluated against baseline precursor values ``X(0)``, so
        shocks are non-cumulative across ticks. This preserves event semantics:
        pulse/campaign timing determines the current offset, and values revert
        when signals deactivate.
        """
        if not self._state_signals:
            return agents

        alias_groups: tuple[tuple[str, ...], ...] = (
            ("health_poor", "health_good"),
            ("health_poor_missing", "health_good_missing"),
        )

        def _with_aliases(column: str) -> tuple[str, ...]:
            expanded: list[str] = [column]
            for group in alias_groups:
                if column not in group:
                    continue
                for candidate in group:
                    if candidate not in expanded:
                        expanded.append(candidate)
            return tuple(expanded)

        precursor_targets: set[str] = set()
        for signal in self._state_signals:
            if signal.target_column in MUTABLE_COLUMNS:
                continue
            signal_id = signal.signal_id or ""
            shock_spec = self._precursor_shock_specs.get(signal_id)
            target_column = str(shock_spec["var"]) if shock_spec is not None else signal.target_column
            if target_column in MUTABLE_COLUMNS:
                msg = (
                    f"State signal '{signal_id or signal.target_column}' resolved to mutable column "
                    f"'{target_column}'. Signals must target precursor X columns."
                )
                raise ValueError(msg)
            for candidate in _with_aliases(target_column):
                if candidate not in agents.columns or candidate not in self._population_t0.columns:
                    continue
                precursor_targets.add(candidate)

        if not precursor_targets:
            return agents

        updates: dict[str, np.ndarray] = {
            column: self._population_t0[column].cast(pl.Float64).to_numpy().astype(np.float64).copy()
            for column in sorted(precursor_targets)
        }

        for signal in self._state_signals:
            if not self._is_signal_effectively_active(signal, tick=tick):
                continue

            if signal.target_column in MUTABLE_COLUMNS:
                continue

            signal_id = signal.signal_id or ""
            shock_spec = self._precursor_shock_specs.get(signal_id)
            target_column = str(shock_spec["var"]) if shock_spec is not None else signal.target_column
            if target_column in MUTABLE_COLUMNS:
                msg = (
                    f"State signal '{signal_id or signal.target_column}' resolved to mutable column "
                    f"'{target_column}'. Signals must target precursor X columns."
                )
                raise ValueError(msg)
            eff = float(signal.target_offset(tick))
            target_columns = [
                candidate
                for candidate in _with_aliases(target_column)
                if candidate in updates and candidate in agents.columns and candidate in self._population_t0.columns
            ]
            if not target_columns:
                continue

            anchor_world_targets = self._anchor_world_targets_for_column(agents, target_column)
            if anchor_world_targets is not None:
                for column in target_columns:
                    values = updates[column]
                    if target_mask is None:
                        values[:] = anchor_world_targets
                    else:
                        values[target_mask] = anchor_world_targets[target_mask]
                continue

            if shock_spec is not None and str(shock_spec.get("type")) == "binary":
                if eff == 0.0:
                    continue
                set_raw = shock_spec.get("to") if eff > 0.0 else shock_spec.get("from")
                set_value = self._resolve_precursor_endpoint(target_column, set_raw)
                if set_value is None:
                    continue
                for column in target_columns:
                    values = updates[column]
                    if target_mask is None:
                        values[:] = set_value
                    else:
                        values[target_mask] = set_value
                continue

            delta = float(eff)
            if shock_spec is not None:
                spec_delta = self._precursor_shock_delta(
                    target_column,
                    shock_spec,
                    effect_regime=self.cfg.effect_regime,
                )
                if spec_delta is not None:
                    delta = eff * spec_delta
            if delta == 0.0:
                continue
            for column in target_columns:
                values = updates[column]
                if target_mask is None:
                    values[:] = values + delta
                else:
                    values[target_mask] = values[target_mask] + delta

        clipped_updates: list[pl.Series] = []
        for column, values in updates.items():
            if column in CONSTRUCT_BOUNDS:
                lo, hi = CONSTRUCT_BOUNDS[column]
            else:
                stats = self._precursor_column_stats(column)
                if stats is None:
                    lo, hi = (-np.inf, np.inf)
                else:
                    lo, hi = (stats["min"], stats["max"])
            clipped_updates.append(pl.Series(column, np.clip(values, lo, hi)))

        return agents.with_columns(clipped_updates)
