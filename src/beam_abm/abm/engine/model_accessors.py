"""Result accessors and constructors for VaccineBehaviourModel."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl

from beam_abm.abm.belief_update_kernel import BeliefUpdateKernel
from beam_abm.abm.belief_update_surrogate import BeliefUpdateSurrogate
from beam_abm.abm.config import SimulationConfig
from beam_abm.abm.data_contracts import (
    BackboneModel,
    LeversConfig,
    get_abm_levers_config,
    load_all_backbone_models,
    load_anchor_system,
    load_levers_config,
    log_non_abm_outcomes,
)
from beam_abm.abm.engine.types import TimingBreakdown
from beam_abm.abm.io import load_survey_data
from beam_abm.abm.message_contracts import NormChannelWeights
from beam_abm.abm.network import NetworkState, build_contact_network
from beam_abm.abm.population import initialise_population
from beam_abm.abm.scenario_defs import ScenarioConfig
from beam_abm.abm.state_ref_attractor import StateRefAttractor
from beam_abm.abm.state_update import SignalSpec

if TYPE_CHECKING:
    from beam_abm.abm.engine.model import VaccineBehaviourModel


class VaccineBehaviourModelAccessorsMixin:
    @property
    def outcome_trajectory(self) -> pl.DataFrame:
        """Per-tick outcome summary as a DataFrame."""
        if not self._outcome_records:
            return pl.DataFrame()
        return pl.DataFrame([row.to_dict() for row in self._outcome_records])

    @property
    def mediator_trajectory(self) -> pl.DataFrame:
        """Per-tick mediator summary as a DataFrame."""
        if not self._mediator_records:
            return pl.DataFrame()
        return pl.DataFrame([row.to_dict() for row in self._mediator_records])

    @property
    def population_t0(self) -> pl.DataFrame:
        """Snapshot of agent state at initialisation (tick 0)."""
        return self._population_t0

    @property
    def population_final(self) -> pl.DataFrame:
        """Current (latest) agent state."""
        return self.agents.df

    @property
    def signals_realised(self) -> pl.DataFrame:
        """Per-tick realised external signal intensities."""
        if not self._signal_records:
            return pl.DataFrame()
        return pl.DataFrame(self._signal_records)

    @property
    def outcomes_by_group(self) -> pl.DataFrame:
        """Per-tick grouped outcome trajectory summaries."""
        if not self._outcomes_by_group_records:
            return pl.DataFrame()
        return pl.DataFrame(self._outcomes_by_group_records)

    @property
    def mediators_by_group(self) -> pl.DataFrame:
        """Per-tick grouped mediator trajectory summaries."""
        if not self._mediators_by_group_records:
            return pl.DataFrame()
        return pl.DataFrame(self._mediators_by_group_records)

    @property
    def saturation_trajectory(self) -> pl.DataFrame:
        """Per-tick saturation diagnostics (bounds and extremes)."""
        if not self._saturation_records:
            return pl.DataFrame()
        return pl.DataFrame(self._saturation_records)

    @property
    def incidence_trajectory(self) -> pl.DataFrame:
        """Per-tick endogenous incidence summaries."""
        if not self._incidence_records:
            return pl.DataFrame()
        return pl.DataFrame(self._incidence_records)

    @property
    def distance_trajectory(self) -> pl.DataFrame:
        """Per-tick willingness by graph distance from seed set."""
        if not self._distance_trajectory_records:
            return pl.DataFrame()
        return pl.DataFrame(self._distance_trajectory_records)

    @property
    def timing(self) -> TimingBreakdown:
        """Wall-clock timing breakdown."""
        return self._timing

    @property
    def state_signals(self) -> list[SignalSpec]:
        """Calibrated state-context signals (e_S)."""
        return self._state_signals_calibrated

    @property
    def state_signals_nominal(self) -> list[SignalSpec]:
        """Nominal (uncalibrated) state-context signals."""
        return self._state_signals

    @property
    def choice_signals(self) -> list[SignalSpec]:
        """Choice-context signals (e_Y)."""
        return self._choice_signals

    @property
    def current_tick(self) -> int:
        """Current simulation tick."""
        return self._current_tick

    @property
    def target_mask(self) -> np.ndarray | None:
        """Targeting mask for the current tick."""
        return self._target_mask

    @property
    def latest_outcomes(self) -> dict[str, np.ndarray] | None:
        """Latest predicted outcomes (current tick)."""
        return self._latest_outcomes

    @property
    def anchoring_diagnostics(self) -> pl.DataFrame:
        """Baseline channel anchoring diagnostics by country."""
        return self._anchoring_diagnostics

    @property
    def country_incidence(self) -> dict[str, float]:
        """Country-level endogenous incidence/salience proxy."""
        return self._community_incidence

    @property
    def previous_country_incidence(self) -> dict[str, float]:
        """Country-level incidence values from previous tick."""
        return self._prev_community_incidence

    @property
    def community_incidence(self) -> dict[str, float]:
        """Locality-level endogenous incidence/salience proxy."""
        return self._community_incidence

    @property
    def previous_community_incidence(self) -> dict[str, float]:
        """Locality-level incidence values from previous tick."""
        return self._prev_community_incidence

    @property
    def belief_kernel(self) -> BeliefUpdateKernel | BeliefUpdateSurrogate | None:
        """Optional belief-update kernel."""
        return self._belief_kernel

    @property
    def state_ref_attractor(self) -> StateRefAttractor | None:
        """Optional survey-anchored mutable-state attractor."""
        return self._state_ref_attractor

    @property
    def network(self) -> NetworkState | None:
        """Contact network for peer influence."""
        return self._network

    @property
    def norm_channel_weights(self) -> NormChannelWeights:
        """Runtime weights used to compose social_norms_vax_avg."""
        return self._norm_channel_weights

    @property
    def norm_columns(self) -> list[str]:
        """Norm columns used for peer aggregation."""
        return self._norm_columns

    @property
    def baseline_observed_norm_share(self) -> np.ndarray | None:
        """Tick-0 fixed-point observed-adoption state for descriptive norm coherence."""
        if self._baseline_observed_norm_share is None:
            return None
        return self._baseline_observed_norm_share.astype(np.float64, copy=True)

    @property
    def baseline_observed_norm_share_npi(self) -> np.ndarray | None:
        """Tick-0 fixed-point observed NPI share for descriptive norm coherence."""
        if self._baseline_observed_norm_share_npi is None:
            return None
        return self._baseline_observed_norm_share_npi.astype(np.float64, copy=True)

    @property
    def baseline_share_hat(self) -> np.ndarray | None:
        """Tick-0 perceived-adoption share (share_hat) anchor for deviation-based EMA."""
        if self._baseline_share_hat is None:
            return None
        return self._baseline_share_hat.astype(np.float64, copy=True)

    @property
    def baseline_share_hat_npi(self) -> np.ndarray | None:
        """Tick-0 perceived NPI share anchor for deviation-based EMA."""
        if self._baseline_share_hat_npi is None:
            return None
        return self._baseline_share_hat_npi.astype(np.float64, copy=True)

    @property
    def baseline_expected_obs_share(self) -> np.ndarray | None:
        """Expected observed share from initial outcomes (obs_ref for deviation-based EMA).

        Lazy-initialised from first tick's outcomes.  Returns None before
        the first model step.
        """
        if self._baseline_expected_obs_share is None:
            return None
        return self._baseline_expected_obs_share.astype(np.float64, copy=True)

    @property
    def baseline_expected_obs_share_npi(self) -> np.ndarray | None:
        """Expected observed NPI share from initial outcomes (obs_ref for NPI EMA)."""
        if self._baseline_expected_obs_share_npi is None:
            return None
        return self._baseline_expected_obs_share_npi.astype(np.float64, copy=True)

    @property
    def baseline_descriptive_norm(self) -> np.ndarray | None:
        """Tick-0 descriptive norm anchor used for observed norm updates."""
        if self._baseline_descriptive_norm is None:
            return None
        return self._baseline_descriptive_norm.astype(np.float64, copy=True)

    @property
    def baseline_descriptive_norm_npi(self) -> np.ndarray | None:
        """Tick-0 NPI descriptive norm anchor used for observed norm updates."""
        if self._baseline_descriptive_norm_npi is None:
            return None
        return self._baseline_descriptive_norm_npi.astype(np.float64, copy=True)

    @property
    def baseline_norm_exposure(self) -> np.ndarray | None:
        """Tick-0 descriptive-norm exposure implied by baseline perceived share."""
        if self._baseline_norm_exposure is None:
            return None
        return self._baseline_norm_exposure.astype(np.float64, copy=True)

    @property
    def baseline_norm_exposure_npi(self) -> np.ndarray | None:
        """Tick-0 NPI descriptive-norm exposure implied by baseline perceived share."""
        if self._baseline_norm_exposure_npi is None:
            return None
        return self._baseline_norm_exposure_npi.astype(np.float64, copy=True)

    @property
    def freeze_state_updates(self) -> bool:
        """Whether state updates are frozen."""
        return self._freeze_state_updates

    @classmethod
    def from_config(
        cls,
        cfg: SimulationConfig,
        scenario: ScenarioConfig,
        belief_kernel: BeliefUpdateKernel | None = None,
        *,
        seed: int | None = None,
    ) -> VaccineBehaviourModel:
        """Build a model by loading all data contracts from disk.

        Parameters
        ----------
        cfg : SimulationConfig
            Simulation parameters (includes all paths).
        scenario : ScenarioConfig
            Scenario to simulate.
        seed : int or None
            Override seed (defaults to ``cfg.seed``).

        Returns
        -------
        VaccineBehaviourModel
            Fully initialised model, ready for ``run_model()``.
        """
        if seed is None:
            seed = cfg.seed

        cfg.log_summary()

        # Load data contracts
        levers_cfg = load_levers_config(cfg.levers_path)
        log_non_abm_outcomes(levers_cfg, context="VaccineBehaviourModel.from_config")
        levers_cfg = get_abm_levers_config(levers_cfg)
        anchor_system = load_anchor_system(cfg.anchors_dir)
        survey_df = load_survey_data(cfg.survey_csv, cfg.countries)
        models = load_all_backbone_models(cfg.modeling_dir, levers_cfg, survey_df=survey_df)

        # Initialise population
        agent_df = initialise_population(
            n_agents=cfg.n_agents,
            countries=cfg.countries,
            anchor_system=anchor_system,
            survey_df=survey_df,
            seed=seed,
            norm_threshold_beta_alpha=cfg.constants.norm_threshold_beta_alpha,
        )

        # Build network if needed
        network: NetworkState | None = None
        if scenario.communication_enabled:
            network = build_contact_network(
                agent_df,
                bridge_prob=cfg.constants.bridge_prob,
                household_enabled=cfg.household_enabled,
                seed=seed,
            )

        return cls(
            cfg=cfg,
            scenario=scenario,
            models=models,
            levers_cfg=levers_cfg,
            agent_df=agent_df,
            network=network,
            belief_kernel=belief_kernel,
            seed=seed,
        )

    @classmethod
    def from_preloaded(
        cls,
        cfg: SimulationConfig,
        scenario: ScenarioConfig,
        models: dict[str, BackboneModel],
        levers_cfg: LeversConfig,
        agent_df: pl.DataFrame,
        network: NetworkState | None = None,
        belief_kernel: BeliefUpdateKernel | None = None,
        *,
        seed: int | None = None,
    ) -> VaccineBehaviourModel:
        """Build a model from already-loaded data contracts.

        Avoids re-loading from disk when running multiple
        replications or paired counterfactuals.

        Parameters
        ----------
        cfg : SimulationConfig
            Simulation parameters.
        scenario : ScenarioConfig
            Scenario to simulate.
        models : dict[str, BackboneModel]
            Pre-loaded backbone models.
        levers_cfg : LeversConfig
            Lever configuration.
        agent_df : pl.DataFrame
            Pre-built agent population.
        network : NetworkState or None
            Pre-built contact network.
        seed : int or None
            Override seed (defaults to ``cfg.seed``).

        Returns
        -------
        VaccineBehaviourModel
        """
        if seed is None:
            seed = cfg.seed

        return cls(
            cfg=cfg,
            scenario=scenario,
            models=models,
            levers_cfg=levers_cfg,
            agent_df=agent_df,
            network=network,
            belief_kernel=belief_kernel,
            seed=seed,
        )

    @property
    def debug_trace_records(self) -> list[dict[str, Any]]:
        return self._debug_trace_records
