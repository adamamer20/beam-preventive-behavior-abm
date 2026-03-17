"""Shared constants and model construction for credibility diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from beam_abm.abm.config import SimulationConfig
from beam_abm.abm.data_contracts import BackboneModel, LeversConfig
from beam_abm.abm.diagnostics.runtime import DiagnosticsRuntime
from beam_abm.abm.engine import VaccineBehaviourModel
from beam_abm.abm.network import NetworkState, build_contact_network
from beam_abm.abm.scenario_defs import ScenarioConfig

DEFAULT_SEED = 42

LONG_HORIZON_STEPS = 52
DRIFT_ABS_THRESHOLD = 0.5
DRIFT_PER_MONTH_THRESHOLD = 0.03
PULSE_MILD_STD_THRESHOLD = 0.05
PULSE_MILD_RANGE_THRESHOLD = 0.20
PULSE_MILD_MAX_THRESHOLD = 0.25
POLICY_RANK_TIE_TOLERANCE = 0.002

POLICY_RANKING_SCENARIOS = (
    "baseline_importation",
    "baseline_importation_no_communication",
    "baseline_importation_no_observed_norms",
    "baseline_importation_no_peer_pull",
)

# Key constructs to track
TRACKED_CONSTRUCTS = [
    "social_norms_descriptive_vax",
    "institutional_trust_avg",
    "vaccine_risk_avg",
]

# ===========================================================================
# Shared model construction (load data once, build variants cheaply)
# ===========================================================================


@dataclass(slots=True)
class SharedData:
    """Pre-loaded data contracts shared across diagnostic runs."""

    cfg: SimulationConfig
    models: dict[str, BackboneModel]
    levers_cfg: LeversConfig
    agent_df: pl.DataFrame
    network: NetworkState | None


def load_shared_data(
    runtime: DiagnosticsRuntime,
    *,
    n_steps: int,
    seed: int = DEFAULT_SEED,
) -> SharedData:
    """Build shared data bundle from diagnostics runtime contracts."""
    cfg = runtime.cfg.model_copy(deep=True)
    cfg.n_steps = int(n_steps)
    cfg.n_reps = 1
    cfg.seed = int(seed)
    network = build_contact_network(
        runtime.base_population,
        bridge_prob=cfg.constants.bridge_prob,
        household_enabled=cfg.household_enabled,
        seed=seed,
    )
    return SharedData(
        cfg=cfg,
        models=runtime.models,
        levers_cfg=runtime.levers_cfg,
        agent_df=runtime.base_population,
        network=network,
    )


def build_model(
    shared: SharedData,
    scenario: ScenarioConfig,
    *,
    attractor_on: bool = True,
    seed: int = DEFAULT_SEED,
) -> VaccineBehaviourModel:
    """Build a model from shared data with optional attractor control."""
    local_cfg = shared.cfg.model_copy(deep=True)
    local_cfg.seed = int(seed)
    model = VaccineBehaviourModel.from_preloaded(
        cfg=local_cfg,
        scenario=scenario,
        models=shared.models,
        levers_cfg=shared.levers_cfg,
        agent_df=shared.agent_df.clone(),
        network=shared.network,
        seed=seed,
    )
    if not attractor_on:
        model._state_ref_attractor = None
    return model
