"""Shared runtime loading utilities for ABM diagnostics suites."""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from beam_abm.abm.config import SimulationConfig
from beam_abm.abm.data_contracts import (
    AnchorSystem,
    BackboneModel,
    LeversConfig,
    get_abm_levers_config,
    load_all_backbone_models,
    load_anchor_system,
    load_levers_config,
    log_non_abm_outcomes,
)
from beam_abm.abm.io import load_survey_data
from beam_abm.abm.network import NetworkState, build_contact_network
from beam_abm.abm.population import initialise_population
from beam_abm.abm.runtime import SimulationResult, run_single_replication
from beam_abm.abm.scenario_defs import ScenarioConfig

__all__ = [
    "DiagnosticsRuntime",
    "build_network_if_needed",
    "clone_cfg",
    "load_shared_runtime",
    "run_single",
]


@dataclass(slots=True)
class DiagnosticsRuntime:
    """Shared ABM runtime contracts loaded once per diagnostics run."""

    cfg: SimulationConfig
    models: dict[str, BackboneModel]
    levers_cfg: LeversConfig
    anchor_system: AnchorSystem
    survey_df: pl.DataFrame
    base_population: pl.DataFrame


def load_shared_runtime(
    *,
    n_agents: int,
    n_steps: int,
    seed: int,
    n_reps: int,
    deterministic: bool = True,
) -> DiagnosticsRuntime:
    """Load contracts and baseline population once for suite reuse."""
    cfg = SimulationConfig(
        n_agents=n_agents,
        n_steps=n_steps,
        n_reps=n_reps,
        seed=seed,
        deterministic=deterministic,
    )
    levers_cfg = load_levers_config(cfg.levers_path)
    log_non_abm_outcomes(levers_cfg, context="diagnostics.load_shared_runtime")
    levers_cfg = get_abm_levers_config(levers_cfg)
    anchor_system = load_anchor_system(cfg.anchors_dir)
    survey_df = load_survey_data(cfg.survey_csv, cfg.countries)
    models = load_all_backbone_models(cfg.modeling_dir, levers_cfg, survey_df=survey_df)
    base_population = initialise_population(
        n_agents=cfg.n_agents,
        countries=cfg.countries,
        anchor_system=anchor_system,
        survey_df=survey_df,
        seed=cfg.seed,
        norm_threshold_beta_alpha=cfg.constants.norm_threshold_beta_alpha,
    )
    return DiagnosticsRuntime(
        cfg=cfg,
        models=models,
        levers_cfg=levers_cfg,
        anchor_system=anchor_system,
        survey_df=survey_df,
        base_population=base_population,
    )


def clone_cfg(
    cfg: SimulationConfig,
    *,
    knob_updates: dict[str, float] | None = None,
    constant_updates: dict[str, float | int | bool] | None = None,
) -> SimulationConfig:
    """Clone a simulation config with optional knob/constant updates."""
    cloned = cfg.model_copy(deep=True)
    if knob_updates:
        cloned.knobs = cloned.knobs.model_copy(update=knob_updates)
    if constant_updates:
        cloned.constants = cloned.constants.model_copy(update=constant_updates)
    return cloned


def build_network_if_needed(
    agent_df: pl.DataFrame,
    *,
    cfg: SimulationConfig,
    scenario: ScenarioConfig,
    seed: int,
) -> NetworkState | None:
    """Build network only for communication-enabled scenarios that use social channels."""
    if scenario.communication_enabled and (scenario.peer_pull_enabled or scenario.observed_norms_enabled):
        return build_contact_network(
            agent_df,
            bridge_prob=cfg.constants.bridge_prob,
            household_enabled=cfg.household_enabled,
            seed=seed,
        )
    return None


def run_single(
    runtime: DiagnosticsRuntime,
    *,
    scenario: ScenarioConfig,
    cfg: SimulationConfig,
    seed: int,
    rep_id: int = 0,
) -> SimulationResult:
    """Run one replication using shared runtime data and optional config overrides."""
    network = build_network_if_needed(
        runtime.base_population,
        cfg=cfg,
        scenario=scenario,
        seed=seed,
    )
    return run_single_replication(
        cfg=cfg,
        scenario=scenario,
        models=runtime.models,
        levers_cfg=runtime.levers_cfg,
        agent_df=runtime.base_population,
        network=network,
        rep_id=rep_id,
        seed=seed,
    )
