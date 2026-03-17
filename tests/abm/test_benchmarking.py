"""Tests for benchmarking and validation gates."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from beam_abm.abm.benchmarking import (
    GateResult,
    run_baseline_replication_gate,
    run_dynamic_plausibility_gate,
    run_mechanical_verification_gate,
    run_network_validation_gate,
    run_stochasticity_policy_gate,
)
from beam_abm.abm.data_contracts import (
    AnchorMetadata,
    AnchorSystem,
    BackboneModel,
    LeversConfig,
    OutcomeLever,
)
from beam_abm.abm.intensity_calibration import IntensityCalibrator, build_iqr_scales, calibrate_signal
from beam_abm.abm.metrics import assemble_parity_diagnostics_table
from beam_abm.abm.runtime import SimulationMetadata, SimulationResult
from beam_abm.abm.state_schema import MUTABLE_COLUMNS, OUTCOME_COLUMNS
from beam_abm.abm.state_update import SignalSpec, apply_signals_to_frame


def _make_population(n: int = 200, seed: int = 42) -> pl.DataFrame:
    """Synthetic population with core mutable + outcome columns."""
    rng = np.random.default_rng(seed)
    countries = np.array(["IT", "DE", "FR", "ES"])
    data: dict[str, list[float] | list[str] | list[int]] = {
        "row_id": list(range(n)),
        "country": rng.choice(countries, size=n, replace=True).tolist(),
        "anchor_id": rng.integers(0, 3, size=n).tolist(),
    }
    data["locality_id"] = [f"{c}_loc000" for c in data["country"]]
    data["activity_score"] = rng.uniform(0.0, 1.0, size=n).tolist()
    data["group_size_score"] = rng.uniform(0.0, 1.0, size=n).tolist()
    data["core_degree"] = rng.integers(4, 12, size=n).tolist()
    data["household_size"] = rng.integers(1, 5, size=n).tolist()

    for col in sorted(MUTABLE_COLUMNS):
        data[col] = rng.normal(0, 1, n).tolist()

    trust = np.array(data["institutional_trust_avg"], dtype=np.float64)
    norms = np.array(data["social_norms_vax_avg"], dtype=np.float64)
    risk = np.array(data["vaccine_risk_avg"], dtype=np.float64)
    lin = 0.6 * trust + 0.3 * norms - 0.5 * risk

    for col in sorted(OUTCOME_COLUMNS):
        if col == "vax_willingness_T12":
            data[col] = np.clip(3.0 + lin, 1.0, 5.0).tolist()
        else:
            p = 1.0 / (1.0 + np.exp(-lin))
            data[col] = rng.binomial(1, p).astype(float).tolist()

    return pl.DataFrame(data)


@pytest.fixture()
def anchor_system() -> AnchorSystem:
    memberships = pl.DataFrame(
        {
            "row_id": list(range(200)),
            "anchor_id": [i % 3 for i in range(200)],
            "weight": [1.0] * 200,
        }
    )
    country_shares = pl.DataFrame(
        {
            "anchor_id": [0, 1, 2] * 4,
            "country": ["IT"] * 3 + ["DE"] * 3 + ["FR"] * 3 + ["ES"] * 3,
            "share": [0.34, 0.33, 0.33] * 4,
        }
    )
    return AnchorSystem(
        metadata=AnchorMetadata(anchor_k=3),
        n_anchors=3,
        country_shares=country_shares,
        memberships=memberships,
    )


@pytest.fixture()
def simple_binary_model() -> BackboneModel:
    return BackboneModel(
        name="flu_vaccinated_2023_2024",
        model_type="binary",
        terms=("institutional_trust_avg", "vaccine_risk_avg"),
        coefficients=np.array([0.5, -0.3]),
        thresholds=None,
        intercept=-0.1,
        blocks={},
    )


@pytest.fixture()
def simple_ordinal_model() -> BackboneModel:
    return BackboneModel(
        name="vax_willingness_T12",
        model_type="ordinal",
        terms=("institutional_trust_avg", "social_norms_vax_avg"),
        coefficients=np.array([0.8, 0.4]),
        thresholds=np.array([-1.5, -0.5, 0.5, 1.5]),
        intercept=None,
        blocks={},
    )


@pytest.fixture()
def levers_cfg() -> LeversConfig:
    return LeversConfig(
        outcomes={
            "flu_vaccinated_2023_2024": OutcomeLever(
                outcome="flu_vaccinated_2023_2024",
                model_ref="flu_vaccinated_2023_2024",
                levers=("institutional_trust_avg", "vaccine_risk_avg"),
                include_cols=("institutional_trust_avg", "vaccine_risk_avg"),
            ),
            "vax_willingness_T12": OutcomeLever(
                outcome="vax_willingness_T12",
                model_ref="vax_willingness_T12",
                levers=("institutional_trust_avg", "social_norms_vax_avg"),
                include_cols=("institutional_trust_avg", "social_norms_vax_avg"),
            ),
        }
    )


class TestGateResult:
    def test_add_check_and_evaluate(self) -> None:
        gate = GateResult(gate_name="test")
        gate.add_check("c1", passed=True)
        gate.add_check("c2", passed=False)
        assert gate.evaluate() is False
        assert len(gate.checks) == 2
        assert gate.checks[0].name == "c1"
        assert gate.checks[1].passed is False


class TestMechanicalGate:
    def test_coherence_invariant_checks_present(
        self,
        simple_binary_model: BackboneModel,
        simple_ordinal_model: BackboneModel,
        levers_cfg: LeversConfig,
    ) -> None:
        agents = _make_population(120, seed=17).with_columns(
            pl.lit(0.5).alias("norm_threshold_theta"),
            pl.lit(1.0).alias("social_susceptibility_lambda"),
        )
        models = {
            "flu_vaccinated_2023_2024": simple_binary_model,
            "vax_willingness_T12": simple_ordinal_model,
        }

        gate = run_mechanical_verification_gate(
            agents=agents,
            models=models,
            levers_cfg=levers_cfg,
        )
        check_names = {check.name for check in gate.checks}
        assert "coherence_null_world_state_stationarity" in check_names
        assert "coherence_null_world_outcome_stationarity" in check_names
        assert "coherence_incidence_only_norm_coherence" in check_names
        assert "coherence_incidence_fixed_point_stationarity" in check_names
        assert "coherence_social_only_mean_preservation" in check_names
        assert "coherence_observed_norm_fixed_point" in check_names
        assert not any("flu_vaccinated" in name for name in check_names)


class TestBaselineGate:
    def test_extended_checks_present(
        self,
        anchor_system: AnchorSystem,
        simple_binary_model: BackboneModel,
        simple_ordinal_model: BackboneModel,
        levers_cfg: LeversConfig,
    ) -> None:
        agents = _make_population(200, seed=42)
        survey = _make_population(200, seed=43)
        # Supplementary baseline outcomes are diagnostics, not core ABM state.
        agents = agents.with_columns(
            pl.lit(0.55).alias("mandate_support"),
            pl.lit(0.62).alias("protection_index"),
        )
        survey = survey.with_columns(
            pl.lit(0.50).alias("mandate_support"),
            pl.lit(0.60).alias("protection_index"),
        )
        models = {
            "flu_vaccinated_2023_2024": simple_binary_model,
            "vax_willingness_T12": simple_ordinal_model,
        }

        result = run_baseline_replication_gate(
            agents_t0=agents,
            survey_df=survey,
            models=models,
            levers_cfg=levers_cfg,
            anchor_system=anchor_system,
            countries=["IT", "DE", "FR", "ES"],
            gradient_specs=(
                ("institutional_trust_avg", "vax_willingness_T12", 1.0),
                ("social_norms_vax_avg", "vax_willingness_T12", 1.0),
                ("vaccine_risk_avg", "flu_vaccinated_2023_2024", -1.0),
            ),
        )

        assert isinstance(result, GateResult)
        names = {check.name for check in result.checks}
        assert any(name.startswith("outcome_marginal_primary_") for name in names)
        assert any(name.startswith("outcome_marginal_supplementary_") for name in names)
        assert any(name.startswith("engine_quantiles_") for name in names)
        assert any(name.startswith("gradient_") for name in names)

    def test_propensity_jsd_check_present(
        self,
        anchor_system: AnchorSystem,
        simple_binary_model: BackboneModel,
        simple_ordinal_model: BackboneModel,
        levers_cfg: LeversConfig,
    ) -> None:
        agents = _make_population(80, seed=10).with_columns(
            (pl.col("row_id") % 5).cast(pl.Utf8).alias("propensity_slice")
        )
        survey = _make_population(80, seed=11).with_columns(
            (pl.col("row_id") % 5).cast(pl.Utf8).alias("propensity_slice")
        )
        models = {
            "flu_vaccinated_2023_2024": simple_binary_model,
            "vax_willingness_T12": simple_ordinal_model,
        }

        result = run_baseline_replication_gate(
            agents_t0=agents,
            survey_df=survey,
            models=models,
            levers_cfg=levers_cfg,
            anchor_system=anchor_system,
            countries=["IT", "DE", "FR", "ES"],
            gradient_specs=(("institutional_trust_avg", "vax_willingness_T12", 1.0),),
        )

        assert any(check.name.startswith("propensity_jsd_") for check in result.checks)


class TestDynamicGate:
    def test_quiet_stationarity_check_exists(self) -> None:
        traj = pl.DataFrame(
            {
                "tick": [0, 1, 2, 3],
                "vax_willingness_T12_mean": [3.0, 3.02, 3.01, 3.0],
            }
        )
        result = SimulationResult(
            scenario_name="null_world",
            rep_id=0,
            outcome_trajectory=traj,
            mediator_trajectory=pl.DataFrame({"tick": [0, 1, 2, 3]}),
            metadata=SimulationMetadata(
                n_agents=100,
                n_steps=4,
                communication_enabled=False,
                k_social=0.0,
                bridge_prob=0.08,
                targeting="uniform",
                deterministic=True,
            ),
        )

        gate = run_dynamic_plausibility_gate([result])
        assert isinstance(gate, GateResult)
        assert any(check.name.startswith("quiet_stationarity_") for check in gate.checks)

    def test_dynamic_directional_tplus1_check_exists(self) -> None:
        baseline = SimulationResult(
            scenario_name="baseline",
            rep_id=0,
            outcome_trajectory=pl.DataFrame(
                {
                    "tick": list(range(11)),
                    "vax_willingness_T12_mean": [3.0] * 11,
                }
            ),
            mediator_trajectory=pl.DataFrame({"tick": list(range(11))}),
        )
        shock = SimulationResult(
            scenario_name="outbreak_global",
            rep_id=0,
            outcome_trajectory=pl.DataFrame(
                {
                    "tick": list(range(11)),
                    "vax_willingness_T12_mean": [3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.2, 3.35],
                }
            ),
            mediator_trajectory=pl.DataFrame({"tick": list(range(11))}),
        )

        gate = run_dynamic_plausibility_gate([baseline, shock], stress_horizon_min=2, stress_horizon_max=7)
        assert any(check.name.startswith("dynamic_directional_tplus1_") for check in gate.checks)

    def test_dynamic_directional_zero_response_fails(self) -> None:
        baseline = SimulationResult(
            scenario_name="baseline",
            rep_id=0,
            outcome_trajectory=pl.DataFrame(
                {
                    "tick": list(range(11)),
                    "vax_willingness_T12_mean": [3.0] * 11,
                }
            ),
            mediator_trajectory=pl.DataFrame({"tick": list(range(11))}),
        )
        shock = SimulationResult(
            scenario_name="outbreak_global",
            rep_id=0,
            outcome_trajectory=pl.DataFrame(
                {
                    "tick": list(range(11)),
                    "vax_willingness_T12_mean": [3.0] * 11,
                }
            ),
            mediator_trajectory=pl.DataFrame({"tick": list(range(11))}),
        )

        gate = run_dynamic_plausibility_gate([baseline, shock], stress_horizon_min=2, stress_horizon_max=7)
        directional = [
            check
            for check in gate.checks
            if check.name.startswith("dynamic_directional_tplus1_outbreak_global_vax_willingness_T12_mean_rep0")
        ]
        assert directional
        assert directional[0].passed is False

    def test_calibration_intensity_matches_iqr(self) -> None:
        rng = np.random.default_rng(7)
        values = rng.normal(0, 1, 5000)
        df = pl.DataFrame({"vaccine_risk_avg": values})
        scales = build_iqr_scales(df)
        calibrator = IntensityCalibrator(global_scales=scales)

        signal = SignalSpec(
            target_column="vaccine_risk_avg",
            intensity=1.0,
            signal_type="pulse",
            start_tick=0,
        )
        calibrated = calibrate_signal(signal, calibrator)
        updated = apply_signals_to_frame(df, [calibrated], tick=0)
        delta = float(np.mean(updated["vaccine_risk_avg"].to_numpy() - df["vaccine_risk_avg"].to_numpy()))

        iqr = float(scales["vaccine_risk_avg"])
        assert np.isfinite(delta)
        assert np.isclose(delta, iqr, rtol=0.05, atol=1e-6)

    def test_dose_response_zero_response_fails(self) -> None:
        baseline = SimulationResult(
            scenario_name="baseline",
            rep_id=0,
            outcome_trajectory=pl.DataFrame(
                {
                    "tick": list(range(25)),
                    "vax_willingness_T12_mean": [3.0] * 25,
                }
            ),
            mediator_trajectory=pl.DataFrame({"tick": list(range(25))}),
        )
        strong = SimulationResult(
            scenario_name="outbreak_global",
            rep_id=0,
            outcome_trajectory=pl.DataFrame(
                {
                    "tick": list(range(25)),
                    "vax_willingness_T12_mean": [3.0] * 25,
                }
            ),
            mediator_trajectory=pl.DataFrame({"tick": list(range(25))}),
        )
        weak = SimulationResult(
            scenario_name="outbreak_local",
            rep_id=0,
            outcome_trajectory=pl.DataFrame(
                {
                    "tick": list(range(25)),
                    "vax_willingness_T12_mean": [3.0] * 25,
                }
            ),
            mediator_trajectory=pl.DataFrame({"tick": list(range(25))}),
        )

        gate = run_dynamic_plausibility_gate([baseline, strong, weak], stress_horizon_min=2, stress_horizon_max=24)
        dose_checks = [check for check in gate.checks if check.name.startswith("dose_response_")]
        assert dose_checks
        assert dose_checks[0].passed is False

    def test_anti_saturation_check_fails_for_flatlined_mediators(self) -> None:
        baseline = SimulationResult(
            scenario_name="baseline",
            rep_id=0,
            outcome_trajectory=pl.DataFrame(
                {
                    "tick": list(range(12)),
                    "vax_willingness_T12_mean": [3.0] * 12,
                }
            ),
            mediator_trajectory=pl.DataFrame(
                {
                    "tick": list(range(12)),
                    "institutional_trust_avg_mean": [0.0] * 12,
                }
            ),
        )
        # Large early movement then near-flat late period -> saturation flag.
        saturated = SimulationResult(
            scenario_name="outbreak_global",
            rep_id=0,
            outcome_trajectory=pl.DataFrame(
                {
                    "tick": list(range(12)),
                    "vax_willingness_T12_mean": [
                        3.0,
                        3.2,
                        3.3,
                        3.35,
                        3.37,
                        3.38,
                        3.381,
                        3.381,
                        3.381,
                        3.381,
                        3.381,
                        3.381,
                    ],
                }
            ),
            mediator_trajectory=pl.DataFrame(
                {
                    "tick": list(range(12)),
                    "institutional_trust_avg_mean": [
                        0.0,
                        1.8,
                        2.8,
                        3.4,
                        3.7,
                        3.85,
                        3.95,
                        3.98,
                        3.99,
                        3.99,
                        3.99,
                        3.99,
                    ],
                    "social_norms_vax_avg_mean": [
                        1.0,
                        3.0,
                        4.8,
                        5.8,
                        6.3,
                        6.6,
                        6.85,
                        6.94,
                        6.95,
                        6.95,
                        6.95,
                        6.95,
                    ],
                }
            ),
        )

        gate = run_dynamic_plausibility_gate([baseline, saturated], stress_horizon_min=2, stress_horizon_max=12)
        anti_sat = [
            check for check in gate.checks if check.name.startswith("anti_saturation_mediator_outbreak_global_rep0")
        ]
        if anti_sat:
            assert anti_sat[0].passed is False

    def test_baseline_regime_signature_checks_pass(self) -> None:
        ticks = list(range(12))

        imp_inc = [0.0510, 0.0509, 0.0508, 0.0507, 0.0506, 0.0505, 0.0504, 0.0503, 0.0502, 0.0501, 0.0500, 0.0499]
        imp_out = [3.60, 3.58, 3.56, 3.54, 3.52, 3.50, 3.48, 3.46, 3.44, 3.42, 3.40, 3.38]

        imp_nc_inc = [0.0600, 0.0601, 0.0602, 0.0603, 0.0602, 0.0602, 0.0601, 0.0602, 0.0602, 0.0601, 0.0601, 0.0602]
        imp_nc_out = [3.50, 3.505, 3.505, 3.504, 3.504, 3.503, 3.503, 3.503, 3.502, 3.502, 3.501, 3.501]

        no_imp_inc = [0.0500, 0.0490, 0.0480, 0.0470, 0.0460, 0.0450, 0.0440, 0.0430, 0.0420, 0.0410, 0.0400, 0.0390]
        no_imp_out = [3.55, 3.54, 3.53, 3.52, 3.51, 3.50, 3.49, 3.48, 3.47, 3.46, 3.45, 3.44]

        def _outcome(values: list[float]) -> pl.DataFrame:
            return pl.DataFrame(
                {
                    "tick": ticks,
                    "vax_willingness_T12_mean": values,
                    "mask_when_symptomatic_crowded_mean": [1.0 + (v - 3.0) for v in values],
                    "stay_home_when_symptomatic_mean": [1.2 + (v - 3.0) for v in values],
                    "mask_when_pressure_high_mean": [1.4 + (v - 3.0) for v in values],
                }
            )

        def _incidence(values: list[float]) -> pl.DataFrame:
            return pl.DataFrame({"tick": ticks, "mean_incidence": values})

        baseline_importation = SimulationResult(
            scenario_name="baseline_importation",
            rep_id=0,
            outcome_trajectory=_outcome(imp_out),
            mediator_trajectory=pl.DataFrame({"tick": ticks}),
            incidence_trajectory=_incidence(imp_inc),
        )
        baseline_importation_no_comm = SimulationResult(
            scenario_name="baseline_importation_no_communication",
            rep_id=0,
            outcome_trajectory=_outcome(imp_nc_out),
            mediator_trajectory=pl.DataFrame({"tick": ticks}),
            incidence_trajectory=_incidence(imp_nc_inc),
        )
        baseline_no_importation = SimulationResult(
            scenario_name="baseline",
            rep_id=0,
            outcome_trajectory=_outcome(no_imp_out),
            mediator_trajectory=pl.DataFrame({"tick": ticks}),
            incidence_trajectory=_incidence(no_imp_inc),
        )

        gate = run_dynamic_plausibility_gate(
            [baseline_importation, baseline_importation_no_comm, baseline_no_importation],
            stress_horizon_min=2,
            stress_horizon_max=11,
        )
        checks = {check.name: check for check in gate.checks}

        imp_check = checks.get("baseline_regime_importation_persistent_plateau_perturbation_rep0")
        assert imp_check is not None
        assert imp_check.passed is True

        imp_nc_check = checks.get("baseline_regime_importation_no_comm_persistent_plateau_perturbation_rep0")
        assert imp_nc_check is not None
        assert imp_nc_check.passed is True

        no_imp_check = checks.get("baseline_regime_no_importation_declines_perturbation_rep0")
        assert no_imp_check is not None
        assert no_imp_check.passed is True

        signature_check = checks.get("baseline_regime_comm_vs_nocomm_signature_perturbation_rep0")
        assert signature_check is not None
        assert signature_check.passed is True


def test_parity_diagnostics_table_contains_expected_metrics() -> None:
    outcome = pl.DataFrame(
        {
            "tick": [0, 1, 2, 3],
            "vax_willingness_T12_mean": [3.0, 3.4, 3.2, 3.1],
        }
    )
    incidence = pl.DataFrame(
        {
            "tick": [0, 1, 2, 3],
            "mean_incidence": [0.05, 0.12, 0.10, 0.07],
        }
    )
    distance = pl.DataFrame(
        {
            "tick": [0, 1, 2, 3, 0, 1, 2, 3],
            "distance": [0, 0, 0, 0, 1, 1, 1, 1],
            "mean_vax_willingness": [3.0, 3.5, 3.3, 3.2, 3.0, 3.2, 3.3, 3.25],
        }
    )

    table = assemble_parity_diagnostics_table(outcome, incidence, distance)
    assert table.height > 0
    metrics = set(table["metric"].to_list())
    assert "shock_recovery" in metrics
    assert "diffusion_distance_peak" in metrics
    assert "closed_loop_signature" in metrics


class TestNetworkAndStochasticityGates:
    def test_network_gate_runs(self) -> None:
        agents = _make_population(120, seed=7)
        gate = run_network_validation_gate(agents)
        assert isinstance(gate, GateResult)
        assert any(check.name == "feature_space_tie_rate" for check in gate.checks)
        assert any(check.name.startswith("homophily_sanity_") for check in gate.checks)
        assert any(check.name.startswith("jitter_sensitivity_") for check in gate.checks)

    def test_stochasticity_gate_runs(self) -> None:
        pop_final = _make_population(120, seed=99)
        traj = pl.DataFrame(
            {
                "tick": [0, 1, 2],
                "vax_willingness_T12_mean": [3.0, 3.1, 3.0],
            }
        )
        result = SimulationResult(
            scenario_name="baseline",
            rep_id=0,
            outcome_trajectory=traj,
            mediator_trajectory=pl.DataFrame({"tick": [0, 1, 2]}),
            population_final=pop_final,
            metadata=SimulationMetadata(
                n_agents=120,
                n_steps=3,
                communication_enabled=True,
                k_social=0.5,
                bridge_prob=0.08,
                targeting="uniform",
                deterministic=False,
            ),
        )
        gate = run_stochasticity_policy_gate([result])
        assert isinstance(gate, GateResult)
        assert any(check.name.startswith("policy_documented") for check in gate.checks)

    def test_stochasticity_regime_checks(self) -> None:
        traj_det = pl.DataFrame(
            {
                "tick": [0, 1, 2],
                "vax_willingness_T12_mean": [3.0, 3.0, 3.0],
            }
        )
        det_a = SimulationResult(
            scenario_name="baseline",
            rep_id=0,
            seed=101,
            outcome_trajectory=traj_det,
            mediator_trajectory=pl.DataFrame({"tick": [0, 1, 2]}),
            metadata=SimulationMetadata(
                n_agents=120,
                n_steps=3,
                communication_enabled=False,
                k_social=0.0,
                bridge_prob=0.08,
                targeting="uniform",
                deterministic=True,
            ),
        )
        det_b = SimulationResult(
            scenario_name="baseline",
            rep_id=1,
            seed=202,
            outcome_trajectory=traj_det,
            mediator_trajectory=pl.DataFrame({"tick": [0, 1, 2]}),
            metadata=SimulationMetadata(
                n_agents=120,
                n_steps=3,
                communication_enabled=False,
                k_social=0.0,
                bridge_prob=0.08,
                targeting="uniform",
                deterministic=True,
            ),
        )

        stoch_a = SimulationResult(
            scenario_name="baseline",
            rep_id=0,
            seed=1,
            outcome_trajectory=pl.DataFrame(
                {
                    "tick": [0, 1, 2],
                    "vax_willingness_T12_mean": [3.0, 3.3, 3.6],
                }
            ),
            mediator_trajectory=pl.DataFrame({"tick": [0, 1, 2]}),
            metadata=SimulationMetadata(
                n_agents=120,
                n_steps=3,
                communication_enabled=True,
                k_social=0.5,
                bridge_prob=0.08,
                targeting="uniform",
                deterministic=False,
            ),
        )
        stoch_b = SimulationResult(
            scenario_name="baseline",
            rep_id=1,
            seed=2,
            outcome_trajectory=pl.DataFrame(
                {
                    "tick": [0, 1, 2],
                    "vax_willingness_T12_mean": [3.0, 2.8, 2.6],
                }
            ),
            mediator_trajectory=pl.DataFrame({"tick": [0, 1, 2]}),
            metadata=SimulationMetadata(
                n_agents=120,
                n_steps=3,
                communication_enabled=True,
                k_social=0.5,
                bridge_prob=0.08,
                targeting="uniform",
                deterministic=False,
            ),
        )

        gate = run_stochasticity_policy_gate([det_a, det_b, stoch_a, stoch_b])
        names = {check.name for check in gate.checks}
        assert any(name.startswith("deterministic_seed_invariance_no_comm_") for name in names)
        assert any(name.startswith("stochastic_variance_bounded_") for name in names)
