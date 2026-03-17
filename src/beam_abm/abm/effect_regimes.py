"""Explicit effect-regime execution strategies for ABM runs."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal

import polars as pl
from loguru import logger

from beam_abm.abm.config import SimulationConfig
from beam_abm.abm.metrics import (
    compute_effects_summary,
    compute_local_shadow_target_diff_in_diff,
    compute_paired_dynamics_diagnostics,
)
from beam_abm.abm.population_specs import PopulationSpec, SocietySpec
from beam_abm.abm.runtime import SimulationResult, run_simulation
from beam_abm.abm.scenario_defs import ScenarioConfig

__all__ = [
    "RegimeArtifact",
    "run_perturbation_paired",
    "run_shock_single",
]


@dataclass(slots=True)
class RegimeArtifact:
    """Concrete run artifact to be persisted by CLI orchestration."""

    run_id: str
    scenario_key: str
    scenario: ScenarioConfig
    effect_regime: Literal["perturbation", "shock"]
    result_metadata: object | None
    outcome_trajectory: pl.DataFrame
    mediator_trajectory: pl.DataFrame
    signals_realised: pl.DataFrame
    outcomes_by_group: pl.DataFrame
    mediators_by_group: pl.DataFrame
    saturation_trajectory: pl.DataFrame
    incidence_trajectory: pl.DataFrame
    distance_trajectory: pl.DataFrame
    effects: pl.DataFrame
    population_t0: pl.DataFrame
    population_final: pl.DataFrame
    scenario_metrics: pl.DataFrame
    network_adjacency: pl.DataFrame
    dynamics_summary: pl.DataFrame | None = None
    dynamics_trajectory: pl.DataFrame | None = None
    metadata_extra: dict[str, object] = field(default_factory=dict)


def _paired_baseline_key_for_scenario(scenario: ScenarioConfig) -> str:
    if scenario.incidence_mode == "importation":
        return "baseline_importation"
    if not scenario.communication_enabled:
        return "baseline_no_communication"
    return "baseline"


def _run_id_for_perturbation_world(
    scenario_key: str,
    *,
    world: Literal["low", "high"],
    rep_id: int,
    seed: int,
) -> str:
    return f"{scenario_key}_perturbation_{world}_rep{rep_id}_seed{seed}"


def _run_id_for_perturbation_contrast(
    scenario_key: str,
    *,
    rep_id: int,
    seed: int,
) -> str:
    return f"{scenario_key}_perturbation_contrast_rep{rep_id}_seed{seed}"


def _run_id_for_shock(
    scenario_key: str,
    *,
    rep_id: int,
    seed: int,
) -> str:
    return f"{scenario_key}_shock_rep{rep_id}_seed{seed}"


def _artifact_from_result(
    *,
    run_id: str,
    scenario_key: str,
    scenario: ScenarioConfig,
    effect_regime: Literal["perturbation", "shock"],
    result: SimulationResult,
    effects: pl.DataFrame | None = None,
    dynamics_summary: pl.DataFrame | None = None,
    dynamics_trajectory: pl.DataFrame | None = None,
    metadata_extra: dict[str, object] | None = None,
) -> RegimeArtifact:
    return RegimeArtifact(
        run_id=run_id,
        scenario_key=scenario_key,
        scenario=scenario,
        effect_regime=effect_regime,
        result_metadata=result.metadata,
        outcome_trajectory=result.outcome_trajectory,
        mediator_trajectory=result.mediator_trajectory,
        signals_realised=result.signals_realised,
        outcomes_by_group=result.outcomes_by_group,
        mediators_by_group=result.mediators_by_group,
        saturation_trajectory=result.saturation_trajectory,
        incidence_trajectory=result.incidence_trajectory,
        distance_trajectory=result.distance_trajectory,
        effects=result.effects if effects is None else effects,
        population_t0=result.population_t0,
        population_final=result.population_final,
        scenario_metrics=result.scenario_metrics,
        network_adjacency=result.network_adjacency,
        dynamics_summary=dynamics_summary,
        dynamics_trajectory=dynamics_trajectory,
        metadata_extra=(metadata_extra or {}),
    )


def _tick_mean_delta(high: pl.DataFrame, low: pl.DataFrame) -> pl.DataFrame:
    if "tick" not in high.columns or "tick" not in low.columns:
        return high
    mean_cols = [c for c in high.columns if c.endswith("_mean") and c in low.columns and c != "tick"]
    if not mean_cols:
        return high
    joined = high.select(["tick", *mean_cols]).join(
        low.select(["tick", *[pl.col(c).alias(f"__low__{c}") for c in mean_cols]]),
        on="tick",
        how="inner",
    )
    return joined.select(
        "tick",
        *[(pl.col(c) - pl.col(f"__low__{c}")).alias(c) for c in mean_cols],
    )


def run_perturbation_paired(
    *,
    cfg: SimulationConfig,
    scenario_key: str,
    scenario: ScenarioConfig,
    population_spec: PopulationSpec | None,
    society_spec: SocietySpec | None,
    rep_workers: int,
    projection_weights: Mapping[str, Mapping[str, float]] | None,
) -> list[RegimeArtifact]:
    """Run explicit perturbation low/high worlds and emit contrast artifacts."""
    cfg_low = cfg.model_copy(deep=True)
    cfg_low.effect_regime = "perturbation"
    cfg_low.perturbation_world = "low"

    cfg_high = cfg.model_copy(deep=True)
    cfg_high.effect_regime = "perturbation"
    cfg_high.perturbation_world = "high"

    low_results = run_simulation(
        cfg_low,
        scenario,
        population_spec=population_spec,
        society_spec=society_spec,
        rep_workers=rep_workers,
    )
    high_results = run_simulation(
        cfg_high,
        scenario,
        population_spec=population_spec,
        society_spec=society_spec,
        rep_workers=rep_workers,
    )

    low_by_rep = {int(result.rep_id): result for result in low_results}
    high_by_rep = {int(result.rep_id): result for result in high_results}

    artifacts: list[RegimeArtifact] = []
    for rep_id in sorted(set(low_by_rep).intersection(high_by_rep)):
        low_result = low_by_rep[rep_id]
        high_result = high_by_rep[rep_id]

        low_run_id = _run_id_for_perturbation_world(
            scenario_key,
            world="low",
            rep_id=low_result.rep_id,
            seed=low_result.seed,
        )
        high_run_id = _run_id_for_perturbation_world(
            scenario_key,
            world="high",
            rep_id=high_result.rep_id,
            seed=high_result.seed,
        )

        artifacts.append(
            _artifact_from_result(
                run_id=low_run_id,
                scenario_key=scenario_key,
                scenario=scenario,
                effect_regime="perturbation",
                result=low_result,
                metadata_extra={"perturbation_world": "low"},
            )
        )
        artifacts.append(
            _artifact_from_result(
                run_id=high_run_id,
                scenario_key=scenario_key,
                scenario=scenario,
                effect_regime="perturbation",
                result=high_result,
                metadata_extra={"perturbation_world": "high"},
            )
        )

        contrast_effects = compute_effects_summary(
            baseline_traj=low_result.outcome_trajectory,
            intervention_traj=high_result.outcome_trajectory,
            baseline_mediator_traj=low_result.mediator_trajectory,
            intervention_mediator_traj=high_result.mediator_trajectory,
            projection_weights=projection_weights,
        )
        dynamics_summary, dynamics_trajectory = compute_paired_dynamics_diagnostics(
            baseline_outcome_traj=low_result.outcome_trajectory,
            intervention_outcome_traj=high_result.outcome_trajectory,
            baseline_mediator_traj=low_result.mediator_trajectory,
            intervention_mediator_traj=high_result.mediator_trajectory,
            signals_realised=high_result.signals_realised,
            baseline_incidence_traj=low_result.incidence_trajectory,
            intervention_incidence_traj=high_result.incidence_trajectory,
        )

        contrast_result = SimulationResult(
            scenario_name=high_result.scenario_name,
            rep_id=high_result.rep_id,
            seed=high_result.seed,
            outcome_trajectory=_tick_mean_delta(high_result.outcome_trajectory, low_result.outcome_trajectory),
            mediator_trajectory=_tick_mean_delta(high_result.mediator_trajectory, low_result.mediator_trajectory),
            signals_realised=high_result.signals_realised,
            outcomes_by_group=high_result.outcomes_by_group,
            mediators_by_group=high_result.mediators_by_group,
            saturation_trajectory=high_result.saturation_trajectory,
            incidence_trajectory=_tick_mean_delta(high_result.incidence_trajectory, low_result.incidence_trajectory),
            distance_trajectory=high_result.distance_trajectory,
            effects=contrast_effects,
            scenario_metrics=high_result.scenario_metrics,
            network_adjacency=high_result.network_adjacency,
            population_t0=high_result.population_t0,
            population_final=high_result.population_final,
            timing=high_result.timing,
            metadata=high_result.metadata,
        )
        artifacts.append(
            _artifact_from_result(
                run_id=_run_id_for_perturbation_contrast(
                    scenario_key,
                    rep_id=high_result.rep_id,
                    seed=high_result.seed,
                ),
                scenario_key=scenario_key,
                scenario=scenario,
                effect_regime="perturbation",
                result=contrast_result,
                effects=contrast_effects,
                dynamics_summary=dynamics_summary,
                dynamics_trajectory=dynamics_trajectory,
                metadata_extra={
                    "contrast_type": "high_minus_low",
                    "reference_scenario": "q25_to_q75_high_minus_low",
                    "low_run_id": low_run_id,
                    "high_run_id": high_run_id,
                },
            )
        )

    if not artifacts:
        logger.warning("No perturbation artifacts generated for scenario={scenario}", scenario=scenario_key)
    return artifacts


def run_shock_single(
    *,
    cfg: SimulationConfig,
    scenario_key: str,
    scenario: ScenarioConfig,
    scenario_library: dict[str, ScenarioConfig],
    baseline_results_cache: dict[str, list[SimulationResult]],
    population_spec: PopulationSpec | None,
    society_spec: SocietySpec | None,
    rep_workers: int,
    projection_weights: Mapping[str, Mapping[str, float]] | None,
) -> list[RegimeArtifact]:
    """Run single-world shock and compare against matched CRN baseline."""
    cfg_run = cfg.model_copy(deep=True)
    cfg_run.effect_regime = "shock"
    cfg_run.perturbation_world = None

    baseline_key = _paired_baseline_key_for_scenario(scenario)
    baseline_results = baseline_results_cache.get(baseline_key)
    if baseline_results is None:
        baseline_results = run_simulation(
            cfg_run,
            scenario_library[baseline_key],
            population_spec=population_spec,
            society_spec=society_spec,
            rep_workers=rep_workers,
        )
        baseline_results_cache[baseline_key] = baseline_results

    shock_results = run_simulation(
        cfg_run,
        scenario,
        population_spec=population_spec,
        society_spec=society_spec,
        rep_workers=rep_workers,
    )

    base_by_rep = {int(result.rep_id): result for result in baseline_results}
    artifacts: list[RegimeArtifact] = []
    for shock_result in shock_results:
        base_result = base_by_rep.get(int(shock_result.rep_id))
        if base_result is None:
            logger.warning(
                "Skipping shock artifact for scenario={scenario} rep={rep}: missing baseline pair.",
                scenario=scenario_key,
                rep=shock_result.rep_id,
            )
            continue

        paired_effects = compute_effects_summary(
            baseline_traj=base_result.outcome_trajectory,
            intervention_traj=shock_result.outcome_trajectory,
            baseline_mediator_traj=base_result.mediator_trajectory,
            intervention_mediator_traj=shock_result.mediator_trajectory,
            projection_weights=projection_weights,
        )
        if scenario.where == "local" and scenario.targeting == "targeted":
            local_diffusion_effects = compute_local_shadow_target_diff_in_diff(
                baseline_population_t0=base_result.population_t0,
                baseline_population_final=base_result.population_final,
                intervention_population_t0=shock_result.population_t0,
                intervention_population_final=shock_result.population_final,
                seed_rule=scenario.seed_rule,
                high_degree_target_percentile=float(cfg.constants.high_degree_target_percentile),
                high_exposure_target_percentile=float(cfg.constants.high_exposure_target_percentile),
                low_legitimacy_target_percentile=float(cfg.constants.low_legitimacy_target_percentile),
            )
            if local_diffusion_effects.height > 0:
                paired_effects = (
                    local_diffusion_effects
                    if paired_effects.height == 0
                    else pl.concat([paired_effects, local_diffusion_effects], how="diagonal_relaxed")
                )

        dynamics_summary, dynamics_trajectory = compute_paired_dynamics_diagnostics(
            baseline_outcome_traj=base_result.outcome_trajectory,
            intervention_outcome_traj=shock_result.outcome_trajectory,
            baseline_mediator_traj=base_result.mediator_trajectory,
            intervention_mediator_traj=shock_result.mediator_trajectory,
            signals_realised=shock_result.signals_realised,
            baseline_incidence_traj=base_result.incidence_trajectory,
            intervention_incidence_traj=shock_result.incidence_trajectory,
        )

        artifacts.append(
            _artifact_from_result(
                run_id=_run_id_for_shock(
                    scenario_key,
                    rep_id=shock_result.rep_id,
                    seed=shock_result.seed,
                ),
                scenario_key=scenario_key,
                scenario=scenario,
                effect_regime="shock",
                result=shock_result,
                effects=paired_effects,
                dynamics_summary=dynamics_summary,
                dynamics_trajectory=dynamics_trajectory,
                metadata_extra={"reference_scenario": baseline_key},
            )
        )

    return artifacts
