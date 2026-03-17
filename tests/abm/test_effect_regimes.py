"""Unit tests for explicit ABM effect-regime strategy functions."""

from __future__ import annotations

import polars as pl
import pytest

from beam_abm.abm.config import SimulationConfig
from beam_abm.abm.effect_regimes import run_perturbation_paired, run_shock_single
from beam_abm.abm.runtime import SimulationMetadata, SimulationResult
from beam_abm.abm.scenario_defs import ScenarioConfig


def _make_result(*, scenario_name: str, rep_id: int, seed: int, base_shift: float) -> SimulationResult:
    ticks = [0, 1, 2]
    outcome = pl.DataFrame(
        {
            "tick": ticks,
            "vax_willingness_T12_mean": [1.0 + base_shift, 1.1 + base_shift, 1.2 + base_shift],
        }
    )
    mediator = pl.DataFrame(
        {
            "tick": ticks,
            "institutional_trust_avg_mean": [0.2 + base_shift, 0.22 + base_shift, 0.24 + base_shift],
        }
    )
    incidence = pl.DataFrame(
        {
            "tick": ticks,
            "mean_incidence": [0.1 + base_shift, 0.12 + base_shift, 0.13 + base_shift],
        }
    )
    effects = pl.DataFrame({"outcome": ["vax_willingness_T12"], "delta_final": [base_shift]})
    return SimulationResult(
        scenario_name=scenario_name,
        rep_id=rep_id,
        seed=seed,
        outcome_trajectory=outcome,
        mediator_trajectory=mediator,
        signals_realised=pl.DataFrame(
            {
                "tick": ticks,
                "intensity_realised_mean": [0.0, 0.0, 0.0],
                "intensity_realised_nominal_mean": [0.0, 0.0, 0.0],
            }
        ),
        outcomes_by_group=pl.DataFrame(),
        mediators_by_group=pl.DataFrame(),
        saturation_trajectory=pl.DataFrame(),
        incidence_trajectory=incidence,
        distance_trajectory=pl.DataFrame(),
        effects=effects,
        scenario_metrics=pl.DataFrame(),
        network_adjacency=pl.DataFrame(),
        population_t0=pl.DataFrame({"row_id": [0]}),
        population_final=pl.DataFrame({"row_id": [0]}),
        metadata=SimulationMetadata(effect_regime="perturbation"),
    )


def test_run_perturbation_paired_emits_low_high_and_contrast(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = SimulationConfig(n_reps=1)
    scenario = ScenarioConfig(name="trust_repair_socialinfo", state_signals=())

    def _run_simulation_stub(
        cfg_local: SimulationConfig,
        scenario_local: ScenarioConfig,
        *,
        population_spec: object,
        society_spec: object,
        rep_workers: int,
    ) -> list[SimulationResult]:
        del population_spec, society_spec, rep_workers
        if cfg_local.perturbation_world == "low":
            return [_make_result(scenario_name=scenario_local.name, rep_id=0, seed=41, base_shift=0.0)]
        if cfg_local.perturbation_world == "high":
            return [_make_result(scenario_name=scenario_local.name, rep_id=0, seed=41, base_shift=0.6)]
        msg = "unexpected perturbation world"
        raise AssertionError(msg)

    monkeypatch.setattr("beam_abm.abm.effect_regimes.run_simulation", _run_simulation_stub)
    monkeypatch.setattr(
        "beam_abm.abm.effect_regimes.compute_effects_summary",
        lambda **_: pl.DataFrame(
            {
                "outcome": ["vax_willingness_T12"],
                "delta_final": [0.6],
            }
        ),
    )
    monkeypatch.setattr(
        "beam_abm.abm.effect_regimes.compute_paired_dynamics_diagnostics",
        lambda **_: (
            pl.DataFrame({"metric": ["delta_vax_auc_per_input"], "value": [1.0]}),
            pl.DataFrame({"tick": [0, 1], "delta_vax": [0.0, 0.6]}),
        ),
    )

    artifacts = run_perturbation_paired(
        cfg=cfg,
        scenario_key="trust_repair_socialinfo",
        scenario=scenario,
        population_spec=None,
        society_spec=None,
        rep_workers=1,
        projection_weights=None,
    )

    assert len(artifacts) == 3
    by_run_id = {artifact.run_id: artifact for artifact in artifacts}

    low_id = "trust_repair_socialinfo_perturbation_low_rep0_seed41"
    high_id = "trust_repair_socialinfo_perturbation_high_rep0_seed41"
    contrast_id = "trust_repair_socialinfo_perturbation_contrast_rep0_seed41"
    assert set(by_run_id) == {low_id, high_id, contrast_id}

    assert by_run_id[low_id].metadata_extra["perturbation_world"] == "low"
    assert by_run_id[high_id].metadata_extra["perturbation_world"] == "high"

    contrast = by_run_id[contrast_id]
    assert contrast.metadata_extra["contrast_type"] == "high_minus_low"
    assert contrast.metadata_extra["reference_scenario"] == "q25_to_q75_high_minus_low"
    assert contrast.metadata_extra["low_run_id"] == low_id
    assert contrast.metadata_extra["high_run_id"] == high_id
    assert contrast.effect_regime == "perturbation"


def test_run_shock_single_uses_matched_baseline_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = SimulationConfig(n_reps=1)
    scenario = ScenarioConfig(name="trust_scandal_onlineinfo", communication_enabled=False, incidence_mode="none")
    baseline = ScenarioConfig(name="baseline_no_communication")
    scenario_library = {
        "baseline_no_communication": baseline,
        "trust_scandal_onlineinfo": scenario,
    }

    calls: list[tuple[str, str | None]] = []

    def _run_simulation_stub(
        cfg_local: SimulationConfig,
        scenario_local: ScenarioConfig,
        *,
        population_spec: object,
        society_spec: object,
        rep_workers: int,
    ) -> list[SimulationResult]:
        del population_spec, society_spec, rep_workers
        calls.append((scenario_local.name, cfg_local.effect_regime))
        if scenario_local.name == "baseline_no_communication":
            return [_make_result(scenario_name=scenario_local.name, rep_id=2, seed=77, base_shift=0.0)]
        return [_make_result(scenario_name=scenario_local.name, rep_id=2, seed=77, base_shift=0.2)]

    monkeypatch.setattr("beam_abm.abm.effect_regimes.run_simulation", _run_simulation_stub)
    monkeypatch.setattr(
        "beam_abm.abm.effect_regimes.compute_effects_summary",
        lambda **_: pl.DataFrame({"outcome": ["vax_willingness_T12"], "delta_final": [0.2]}),
    )
    monkeypatch.setattr(
        "beam_abm.abm.effect_regimes.compute_paired_dynamics_diagnostics",
        lambda **_: (
            pl.DataFrame({"metric": ["delta_vax_auc_per_input"], "value": [0.7]}),
            pl.DataFrame({"tick": [0, 1], "delta_vax": [0.0, 0.2]}),
        ),
    )

    baseline_results_cache: dict[str, list[SimulationResult]] = {}
    first = run_shock_single(
        cfg=cfg,
        scenario_key="trust_scandal_onlineinfo",
        scenario=scenario,
        scenario_library=scenario_library,
        baseline_results_cache=baseline_results_cache,
        population_spec=None,
        society_spec=None,
        rep_workers=1,
        projection_weights=None,
    )
    second = run_shock_single(
        cfg=cfg,
        scenario_key="trust_scandal_onlineinfo",
        scenario=scenario,
        scenario_library=scenario_library,
        baseline_results_cache=baseline_results_cache,
        population_spec=None,
        society_spec=None,
        rep_workers=1,
        projection_weights=None,
    )

    assert len(first) == 1
    assert len(second) == 1

    artifact = first[0]
    assert artifact.run_id == "trust_scandal_onlineinfo_shock_rep2_seed77"
    assert artifact.effect_regime == "shock"
    assert artifact.metadata_extra["reference_scenario"] == "baseline_no_communication"

    baseline_calls = [name for name, _ in calls if name == "baseline_no_communication"]
    scenario_calls = [name for name, _ in calls if name == "trust_scandal_onlineinfo"]
    assert len(baseline_calls) == 1
    assert len(scenario_calls) == 2
