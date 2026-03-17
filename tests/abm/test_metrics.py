"""Tests for effect-metric helpers."""

from __future__ import annotations

import polars as pl
import pytest

from beam_abm.abm.metrics import (
    compute_local_shadow_target_diff_in_diff,
    compute_paired_dynamics_diagnostics,
)


def test_local_shadow_did_high_degree_seed_rule() -> None:
    base_t0 = pl.DataFrame(
        {
            "unique_id": [0, 1, 2, 3],
            "core_degree": [1, 2, 3, 4],
            "vax_willingness_T12": [0.0, 0.0, 0.0, 0.0],
        }
    )
    base_final = base_t0.with_columns(pl.Series("vax_willingness_T12", [0.1, 0.1, 0.1, 0.2]))
    scen_t0 = base_t0.clone()
    scen_final = scen_t0.with_columns(pl.Series("vax_willingness_T12", [0.2, 0.2, 0.2, 1.0]))

    effects = compute_local_shadow_target_diff_in_diff(
        baseline_population_t0=base_t0,
        baseline_population_final=base_final,
        intervention_population_t0=scen_t0,
        intervention_population_final=scen_final,
        seed_rule="high_degree",
        outcomes=("vax_willingness_T12",),
    )

    row = effects.filter(pl.col("outcome") == "local_shadow_did::vax_willingness_T12")
    assert row.height == 1
    assert float(row["delta_final"].item()) == pytest.approx(0.7)
    assert int(row["targeted_n"].item()) == 1
    assert int(row["other_n"].item()) == 3
    assert float(row["scenario_gap_change"].item()) == pytest.approx(0.8)
    assert float(row["baseline_shadow_gap_change"].item()) == pytest.approx(0.1)


def test_local_shadow_did_community_targeted_seed_rule() -> None:
    base_t0 = pl.DataFrame(
        {
            "unique_id": [0, 1, 2, 3, 4, 5],
            "locality_id": ["A", "A", "A", "B", "B", "B"],
            "activity_score": [0.9, 0.8, 0.1, 0.2, 0.2, 0.2],
            "vax_willingness_T12": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        }
    )
    base_final = base_t0.with_columns(pl.Series("vax_willingness_T12", [0.2, 0.1, 0.1, 0.1, 0.1, 0.1]))
    scen_t0 = base_t0.clone()
    scen_final = scen_t0.with_columns(pl.Series("vax_willingness_T12", [0.9, 0.2, 0.2, 0.2, 0.2, 0.2]))

    effects = compute_local_shadow_target_diff_in_diff(
        baseline_population_t0=base_t0,
        baseline_population_final=base_final,
        intervention_population_t0=scen_t0,
        intervention_population_final=scen_final,
        seed_rule="community_targeted",
        outcomes=("vax_willingness_T12",),
    )

    row = effects.filter(pl.col("outcome") == "local_shadow_did::vax_willingness_T12")
    assert row.height == 1
    assert float(row["delta_final"].item()) == pytest.approx(0.6)
    assert int(row["targeted_n"].item()) == 1
    assert int(row["other_n"].item()) == 5
    assert str(row["seed_rule"].item()) == "community_targeted"


def test_local_shadow_did_low_legitimacy_seed_rule() -> None:
    base_t0 = pl.DataFrame(
        {
            "unique_id": [0, 1, 2, 3],
            "covax_legitimacy_scepticism_idx": [-2.0, -1.0, 2.0, 3.0],
            "vax_willingness_T12": [0.0, 0.0, 0.0, 0.0],
        }
    )
    base_final = base_t0.with_columns(pl.Series("vax_willingness_T12", [0.1, 0.1, 0.1, 0.2]))
    scen_t0 = base_t0.clone()
    scen_final = scen_t0.with_columns(pl.Series("vax_willingness_T12", [0.2, 0.2, 0.2, 1.0]))

    effects = compute_local_shadow_target_diff_in_diff(
        baseline_population_t0=base_t0,
        baseline_population_final=base_final,
        intervention_population_t0=scen_t0,
        intervention_population_final=scen_final,
        seed_rule="low_legitimacy",
        low_legitimacy_target_percentile=25.0,
        outcomes=("vax_willingness_T12",),
    )

    row = effects.filter(pl.col("outcome") == "local_shadow_did::vax_willingness_T12")
    assert row.height == 1
    assert float(row["delta_final"].item()) == pytest.approx(0.7)
    assert int(row["targeted_n"].item()) == 1
    assert int(row["other_n"].item()) == 3
    assert str(row["seed_rule"].item()) == "low_legitimacy"


def test_local_shadow_did_low_legitimacy_high_degree_seed_rule() -> None:
    base_t0 = pl.DataFrame(
        {
            "unique_id": [0, 1, 2, 3, 4, 5],
            "core_degree": [1.0, 2.0, 10.0, 9.0, 8.0, 1.0],
            "covax_legitimacy_scepticism_idx": [0.0, 0.0, 3.0, 3.0, 1.0, 1.0],
            "vax_willingness_T12": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        }
    )
    base_final = base_t0.with_columns(pl.Series("vax_willingness_T12", [0.1, 0.1, 0.1, 0.1, 0.1, 0.1]))
    scen_t0 = base_t0.clone()
    scen_final = scen_t0.with_columns(pl.Series("vax_willingness_T12", [0.2, 0.2, 1.0, 1.0, 0.2, 0.2]))

    effects = compute_local_shadow_target_diff_in_diff(
        baseline_population_t0=base_t0,
        baseline_population_final=base_final,
        intervention_population_t0=scen_t0,
        intervention_population_final=scen_final,
        seed_rule="low_legitimacy_high_degree",
        high_degree_target_percentile=70.0,
        low_legitimacy_target_percentile=40.0,
        outcomes=("vax_willingness_T12",),
    )

    row = effects.filter(pl.col("outcome") == "local_shadow_did::vax_willingness_T12")
    assert row.height == 1
    assert float(row["delta_final"].item()) == pytest.approx(0.8)
    assert int(row["targeted_n"].item()) == 2
    assert int(row["other_n"].item()) == 4
    assert str(row["seed_rule"].item()) == "low_legitimacy_high_degree"


def test_compute_paired_dynamics_diagnostics_basic() -> None:
    baseline_outcomes = pl.DataFrame(
        {
            "tick": [0, 1, 2, 3],
            "vax_willingness_T12_mean": [1.00, 1.00, 1.00, 1.00],
        }
    )
    intervention_outcomes = pl.DataFrame(
        {
            "tick": [0, 1, 2, 3],
            "vax_willingness_T12_mean": [1.00, 1.10, 1.08, 1.03],
        }
    )
    baseline_mediators = pl.DataFrame(
        {
            "tick": [0, 1, 2, 3],
            "institutional_trust_avg_mean": [2.0, 2.0, 2.0, 2.0],
            "covax_legitimacy_scepticism_idx_mean": [0.5, 0.5, 0.5, 0.5],
        }
    )
    intervention_mediators = pl.DataFrame(
        {
            "tick": [0, 1, 2, 3],
            "institutional_trust_avg_mean": [2.0, 2.1, 2.08, 2.03],
            "covax_legitimacy_scepticism_idx_mean": [0.5, 0.45, 0.46, 0.49],
        }
    )
    signals_realised = pl.DataFrame(
        {
            "tick": [0, 1, 2, 3],
            "intensity_realised_mean": [0.0, 0.5, 0.0, 0.0],
            "intensity_realised_nominal_mean": [0.0, 1.0, 0.0, 0.0],
        }
    )

    summary, trajectory = compute_paired_dynamics_diagnostics(
        baseline_outcome_traj=baseline_outcomes,
        intervention_outcome_traj=intervention_outcomes,
        baseline_mediator_traj=baseline_mediators,
        intervention_mediator_traj=intervention_mediators,
        signals_realised=signals_realised,
    )

    assert summary.height == 1
    assert trajectory.height == 4
    assert "delta_vax" in trajectory.columns
    assert "delta_institutional_trust_avg_mean" in trajectory.columns
    assert int(summary["signal_start_tick"].item()) == 1
    assert int(summary["signal_ticks_active"].item()) == 1
    assert float(summary["vax_peak_abs"].item()) == pytest.approx(0.1)
    assert float(summary["gain_jump_per_abs_input"].item()) == pytest.approx(0.16)
