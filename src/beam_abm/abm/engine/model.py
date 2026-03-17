"""Mesa-frames model class for vaccine behaviour ABM."""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl
from mesa_frames import DataCollector, Model

from beam_abm.abm.belief_update_kernel import BeliefUpdateKernel
from beam_abm.abm.belief_update_surrogate import BeliefUpdateSurrogate
from beam_abm.abm.config import SimulationConfig
from beam_abm.abm.data_contracts import BackboneModel, LeversConfig
from beam_abm.abm.engine.model_accessors import VaccineBehaviourModelAccessorsMixin
from beam_abm.abm.engine.model_initialization import VaccineBehaviourModelInitializationMixin
from beam_abm.abm.engine.simulation import VaccineBehaviourModelSimulationMixin
from beam_abm.abm.engine.state_snapshot import VaccineBehaviourModelStateSnapshotMixin
from beam_abm.abm.engine.survey_agents import SurveyAgents
from beam_abm.abm.engine.types import TimingBreakdown
from beam_abm.abm.intensity_calibration import AnchorLevelTable, IntensityCalibrator
from beam_abm.abm.message_contracts import NormChannelWeights
from beam_abm.abm.metrics import SummaryRow
from beam_abm.abm.network import ContactDiagnostics, NetworkState
from beam_abm.abm.scenario_defs import ScenarioConfig
from beam_abm.abm.state_ref_attractor import StateRefAttractor
from beam_abm.abm.state_update import SignalSpec


class VaccineBehaviourModel(
    VaccineBehaviourModelInitializationMixin,
    VaccineBehaviourModelSimulationMixin,
    VaccineBehaviourModelStateSnapshotMixin,
    VaccineBehaviourModelAccessorsMixin,
    Model,
):
    """Mesa-frames Model wrapping the full ABM lifecycle."""

    cfg: SimulationConfig
    scenario: ScenarioConfig
    backbone_models: dict[str, BackboneModel]
    levers_cfg: LeversConfig
    agents: SurveyAgents
    _network: NetworkState | None
    _state_signals: list[SignalSpec]
    _state_signals_calibrated: list[SignalSpec]
    _choice_signals: list[SignalSpec]
    _norm_columns: list[str]
    _current_tick: int
    _timing: TimingBreakdown
    _outcome_records: list[SummaryRow]
    _mediator_records: list[SummaryRow]
    _population_t0: pl.DataFrame
    _outcome_t0_means: dict[str, float]
    _target_mask: np.ndarray | None
    _freeze_state_updates: bool
    _belief_kernel: BeliefUpdateKernel | BeliefUpdateSurrogate | None
    _state_ref_attractor: StateRefAttractor | None
    _signal_records: list[dict[str, float | int | str]]
    _outcomes_by_group_records: list[dict[str, float | int | str]]
    _mediators_by_group_records: list[dict[str, float | int | str]]
    _latest_signals_realised: list[dict[str, float | int | str]]
    _debug_trace_records: list[dict[str, Any]]
    _latest_outcomes_by_group: list[dict[str, float | int | str]]
    _latest_mediators_by_group: list[dict[str, float | int | str]]
    _latest_outcomes: dict[str, np.ndarray] | None
    _community_incidence: dict[str, float]
    _prev_community_incidence: dict[str, float]
    _importation_peak_reference: float | None
    _importation_peak_reference_by_community: dict[str, float] | None
    _latest_social_similarity_mean: float
    latest_contact_diagnostics: ContactDiagnostics | None
    _seed_mask: np.ndarray | None
    _graph_distance_from_seed: np.ndarray | None
    _saturation_records: list[dict[str, float | int | str]]
    _incidence_records: list[dict[str, float | int | str]]
    _distance_trajectory_records: list[dict[str, float | int | str]]
    _baseline_observed_norm_share: np.ndarray | None
    _baseline_observed_norm_share_npi: np.ndarray | None
    _baseline_share_hat: np.ndarray | None
    _baseline_share_hat_npi: np.ndarray | None
    _baseline_expected_obs_share: np.ndarray | None
    _baseline_expected_obs_share_npi: np.ndarray | None
    _baseline_expected_obs_share_updates: int
    _baseline_expected_obs_share_npi_updates: int
    _baseline_descriptive_norm: np.ndarray | None
    _baseline_descriptive_norm_npi: np.ndarray | None
    _baseline_norm_exposure: np.ndarray | None
    _baseline_norm_exposure_npi: np.ndarray | None
    _intensity_calibrator: IntensityCalibrator
    _anchor_level_table: AnchorLevelTable
    _anchoring_diagnostics: pl.DataFrame
    _precursor_shock_specs: dict[str, dict[str, object]]
    _precursor_column_stats_cache: dict[str, dict[str, float]]
    _norm_channel_weights: NormChannelWeights
    datacollector: DataCollector
