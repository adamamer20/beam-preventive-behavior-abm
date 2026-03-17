"""Tests for the simulator and mesa-frames model integration.

Covers:
- Scenario library well-formedness.
- VaccineBehaviourModel single & multi-tick execution.
- run_single_replication via the simulator module.
- Manual mini-loop for component integration.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from beam_abm.abm.config import ABMImplementationConstants, ABMUserKnobs, SimulationConfig
from beam_abm.abm.data_contracts import (
    BackboneModel,
    LeversConfig,
    OutcomeLever,
)
from beam_abm.abm.engine import SurveyAgents, VaccineBehaviourModel
from beam_abm.abm.network import build_contact_network, compute_peer_means
from beam_abm.abm.outcome_scales import normalize_outcome_to_unit_interval
from beam_abm.abm.outcomes import predict_outcome
from beam_abm.abm.runtime import SimulationResult, run_single_replication
from beam_abm.abm.scenario_defs import SCENARIO_LIBRARY, IncidenceSpec, ScenarioConfig
from beam_abm.abm.state_schema import MUTABLE_COLUMNS
from beam_abm.abm.state_update import SignalSpec, apply_all_state_updates

# =========================================================================
# Helpers
# =========================================================================

_N = 50


def _make_tiny_population(n: int = _N, seed: int = 42) -> pl.DataFrame:
    """A tiny synthetic population with real column names."""
    rng = np.random.default_rng(seed)
    data: dict[str, list[float] | list[str]] = {
        "country": (["IT"] * 10 + ["DE"] * 10 + ["FR"] * 10 + ["ES"] * 10 + ["UK"] * 5 + ["HU"] * 5),
    }
    for col in sorted(MUTABLE_COLUMNS):
        data[col] = rng.normal(0, 1, n).tolist()
    # Precursor columns used by X-space scenario signals.
    data["booking"] = rng.uniform(1.0, 5.0, n).tolist()
    data["vax_hub"] = rng.uniform(1.0, 5.0, n).tolist()
    data["impediment"] = rng.choice([0.0, 1.0], n, p=[0.8, 0.2]).tolist()
    data["exper_illness"] = rng.choice([0.0, 1.0], n, p=[0.7, 0.3]).tolist()
    data["health_poor"] = rng.uniform(1.0, 5.0, n).tolist()
    data["side_effects_other"] = rng.choice([0.0, 1.0], n, p=[0.85, 0.15]).tolist()
    data["vax_convo_freq_cohabitants"] = rng.uniform(1.0, 5.0, n).tolist()
    data["trust_national_gov_vax_info"] = rng.uniform(1.0, 5.0, n).tolist()
    data["trust_gp_vax_info"] = rng.uniform(1.0, 5.0, n).tolist()
    data["trust_who_vax_info"] = rng.uniform(1.0, 5.0, n).tolist()
    # Required dynamic columns in strict ABM mode.
    data["social_susceptibility_lambda"] = rng.uniform(0.05, 0.95, n).tolist()
    data["social_attention_alpha"] = rng.uniform(0.1, 3.0, n).tolist()
    data["social_sharing_phi"] = rng.uniform(0.1, 4.0, n).tolist()
    data["norm_threshold_theta"] = rng.uniform(0.2, 0.8, n).tolist()
    countries = data["country"]
    data["locality_id"] = [f"{c}_loc000" for c in countries]
    data["activity_score"] = rng.uniform(0.0, 1.0, n).tolist()
    data["contact_propensity"] = rng.uniform(0.4, 2.2, n).tolist()
    data["recurring_share"] = rng.uniform(0.2, 0.7, n).tolist()
    data["household_contact_share"] = rng.uniform(0.05, 0.45, n).tolist()
    data["group_size_score"] = rng.uniform(0.0, 1.0, n).tolist()
    data["core_degree"] = rng.integers(4, 10, n).tolist()
    data["household_size"] = rng.integers(1, 5, n).tolist()
    # Add some demographic columns models might need
    data["sex_female[T.True]"] = rng.choice([0.0, 1.0], n).tolist()
    data["age_bracket"] = rng.normal(40, 10, n).tolist()
    return pl.DataFrame(data)


# =========================================================================
# Fixtures
# =========================================================================


@pytest.fixture()
def tiny_config() -> SimulationConfig:
    """Minimal simulation config for smoke testing."""
    return SimulationConfig(
        n_agents=_N,
        n_steps=5,
        n_reps=1,
        seed=42,
        knobs=ABMUserKnobs(),
        constants=ABMImplementationConstants(bridge_prob=0.08),
    )


@pytest.fixture()
def tiny_population() -> pl.DataFrame:
    return _make_tiny_population()


@pytest.fixture()
def npi_model() -> BackboneModel:
    """An ordinal NPI model for smoke testing."""
    return BackboneModel(
        name="mask_when_symptomatic_crowded",
        model_type="ordinal",
        terms=("institutional_trust_avg", "fivec_confidence"),
        coefficients=np.array([0.5, 0.3]),
        thresholds=np.array([-2.0, -1.0, 0.0, 0.5, 1.0, 1.5]),
        intercept=None,
        blocks={
            "trust": ["institutional_trust_avg"],
            "5c": ["fivec_confidence"],
        },
    )


@pytest.fixture()
def ordinal_model() -> BackboneModel:
    """An ordinal model for smoke testing."""
    return BackboneModel(
        name="vax_willingness_T12",
        model_type="ordinal",
        terms=(
            "institutional_trust_avg",
            "vaccine_risk_avg",
            "fivec_confidence",
        ),
        coefficients=np.array([0.8, 0.3, 0.4]),
        thresholds=np.array([-1.5, -0.5, 0.5, 1.5]),
        intercept=None,
        blocks={
            "trust": ["institutional_trust_avg"],
            "risk": ["vaccine_risk_avg"],
            "5c": ["fivec_confidence"],
        },
    )


@pytest.fixture()
def mock_levers(
    npi_model: BackboneModel,
    ordinal_model: BackboneModel,
) -> LeversConfig:
    """LeversConfig matching the test models."""
    return LeversConfig(
        outcomes={
            "mask_when_symptomatic_crowded": OutcomeLever(
                outcome="mask_when_symptomatic_crowded",
                model_ref="mask_when_symptomatic_crowded",
                levers=("institutional_trust_avg", "fivec_confidence"),
                include_cols=("institutional_trust_avg", "fivec_confidence"),
            ),
            "vax_willingness_T12": OutcomeLever(
                outcome="vax_willingness_T12",
                model_ref="vax_willingness_T12",
                levers=(
                    "institutional_trust_avg",
                    "vaccine_risk_avg",
                    "fivec_confidence",
                ),
                include_cols=(
                    "institutional_trust_avg",
                    "vaccine_risk_avg",
                    "fivec_confidence",
                ),
            ),
        }
    )


@pytest.fixture()
def mock_models(
    npi_model: BackboneModel,
    ordinal_model: BackboneModel,
) -> dict[str, BackboneModel]:
    return {
        "mask_when_symptomatic_crowded": npi_model,
        "vax_willingness_T12": ordinal_model,
    }


@pytest.fixture()
def baseline_scenario() -> ScenarioConfig:
    return ScenarioConfig(
        name="test_baseline",
        description="No shocks for testing.",
        family="baseline",
        communication_enabled=True,
    )


@pytest.fixture()
def signal_scenario() -> ScenarioConfig:
    return ScenarioConfig(
        name="test_signal",
        description="Campaign on precursor trust_national_gov_vax_info.",
        family="information",
        state_signals=(
            SignalSpec(
                target_column="trust_national_gov_vax_info",
                intensity=1.0,
                signal_id="trust_national_gov_vax_info_shift",
                signal_type="campaign",
                start_tick=2,
                end_tick=5,
            ),
        ),
        communication_enabled=False,
    )


# =========================================================================
# Tests: Scenario definitions
# =========================================================================


class TestScenarioLibrary:
    """Verify scenario definitions are well-formed."""

    def test_baseline_exists(self) -> None:
        assert "baseline" in SCENARIO_LIBRARY

    def test_all_scenarios_have_name(self) -> None:
        for name, scenario in SCENARIO_LIBRARY.items():
            assert isinstance(scenario, ScenarioConfig)
            assert scenario.name == name or scenario.name

    def test_scenario_count(self) -> None:
        assert len(SCENARIO_LIBRARY) >= 5


# =========================================================================
# Tests: VaccineBehaviourModel (mesa-frames)
# =========================================================================


class TestVaccineBehaviourModel:
    """Test the mesa-frames Model wrapper."""

    def test_init_creates_agent_set(
        self,
        tiny_config: SimulationConfig,
        baseline_scenario: ScenarioConfig,
        mock_models: dict[str, BackboneModel],
        mock_levers: LeversConfig,
        tiny_population: pl.DataFrame,
    ) -> None:
        """Model should create a SurveyAgents set with correct shape."""
        model = VaccineBehaviourModel(
            cfg=tiny_config,
            scenario=baseline_scenario,
            models=mock_models,
            levers_cfg=mock_levers,
            agent_df=tiny_population,
            seed=42,
        )
        assert isinstance(model.agents, SurveyAgents)
        assert "unique_id" in model.agents.df.columns
        # Should have at least as many rows as agents
        assert model.agents.df.height == _N

    def test_single_step(
        self,
        tiny_config: SimulationConfig,
        baseline_scenario: ScenarioConfig,
        mock_models: dict[str, BackboneModel],
        mock_levers: LeversConfig,
        tiny_population: pl.DataFrame,
    ) -> None:
        """A single step should produce outcome + mediator records."""
        model = VaccineBehaviourModel(
            cfg=tiny_config,
            scenario=baseline_scenario,
            models=mock_models,
            levers_cfg=mock_levers,
            agent_df=tiny_population,
            seed=42,
        )
        model.step()
        assert len(model._outcome_records) == 1
        assert len(model._mediator_records) == 1
        assert model._current_tick == 1

    def test_run_model(
        self,
        tiny_config: SimulationConfig,
        baseline_scenario: ScenarioConfig,
        mock_models: dict[str, BackboneModel],
        mock_levers: LeversConfig,
        tiny_population: pl.DataFrame,
    ) -> None:
        """run_model should execute all ticks and produce trajectories."""
        model = VaccineBehaviourModel(
            cfg=tiny_config,
            scenario=baseline_scenario,
            models=mock_models,
            levers_cfg=mock_levers,
            agent_df=tiny_population,
            seed=42,
        )
        model.run_model(n_steps=5)

        assert model._current_tick == 5
        assert model.outcome_trajectory.height == 5
        assert model.mediator_trajectory.height == 5
        assert model.population_t0.height == _N
        assert model.population_final.height == _N
        assert "total" in model.timing

    def test_signal_changes_state(
        self,
        tiny_config: SimulationConfig,
        signal_scenario: ScenarioConfig,
        mock_models: dict[str, BackboneModel],
        mock_levers: LeversConfig,
        tiny_population: pl.DataFrame,
    ) -> None:
        """A precursor signal should change precursor values between ticks."""
        model = VaccineBehaviourModel(
            cfg=tiny_config,
            scenario=signal_scenario,
            models=mock_models,
            levers_cfg=mock_levers,
            agent_df=tiny_population,
            seed=42,
        )
        before = model.agents.df["trust_national_gov_vax_info"].to_numpy().copy()
        for _ in range(14):
            model.step()
        after = model.agents.df["trust_national_gov_vax_info"].to_numpy()
        assert not np.allclose(before, after), "Campaign should change trust_national_gov_vax_info"

    def test_tick0_observed_norm_uses_latent_fixed_point_share(
        self,
        tiny_config: SimulationConfig,
        mock_models: dict[str, BackboneModel],
        mock_levers: LeversConfig,
        tiny_population: pl.DataFrame,
    ) -> None:
        pop = tiny_population.with_columns(
            pl.lit(6.0).alias("social_norms_descriptive_vax"),
            pl.lit(5.0).alias("social_norms_vax_avg"),
            pl.lit(0.5).alias("norm_threshold_theta"),
            pl.lit(1.0).alias("social_susceptibility_lambda"),
        )

        cfg = tiny_config.model_copy(deep=True)
        cfg.knobs = cfg.knobs.model_copy(update={"k_social": 0.8})
        cfg.constants = cfg.constants.model_copy(
            update={
                "social_weight_norm": 0.0,
                "social_weight_trust": 0.5,
                "social_weight_risk": 0.5,
                "state_update_noise_std": 0.0,
                "state_relaxation_slow": 0.0,
                "social_substeps_per_month": 1,
            }
        )
        cfg.state_ref_spec_path = None
        scenario = ScenarioConfig(
            name="test_observed_norm_fixed_point",
            communication_enabled=True,
            enable_endogenous_loop=False,
        )
        network = build_contact_network(
            pop,
            bridge_prob=cfg.constants.bridge_prob,
            household_enabled=cfg.household_enabled,
            seed=cfg.seed,
        )

        model = VaccineBehaviourModel(
            cfg=cfg,
            scenario=scenario,
            models=mock_models,
            levers_cfg=mock_levers,
            agent_df=pop,
            network=network,
            seed=cfg.seed,
        )
        model._state_ref_attractor = None

        baseline_share = model.baseline_observed_norm_share
        assert baseline_share is not None
        assert 0.5 < float(np.mean(baseline_share)) < 0.6
        perceived = model.agents.df["norm_share_hat_vax"].cast(pl.Float64).to_numpy().astype(np.float64)
        np.testing.assert_allclose(perceived, baseline_share, atol=1e-6)

        model._latest_outcomes = {"vax_willingness_T12": np.zeros(pop.height, dtype=np.float64)}
        before = model.agents.df["social_norms_descriptive_vax"].to_numpy().astype(np.float64)
        model.agents.step()
        after = model.agents.df["social_norms_descriptive_vax"].to_numpy().astype(np.float64)
        np.testing.assert_allclose(after, before, atol=1e-6)

    def test_local_targeting_adds_importation_bump_without_zeroing_others(
        self,
        tiny_config: SimulationConfig,
        mock_models: dict[str, BackboneModel],
        mock_levers: LeversConfig,
        tiny_population: pl.DataFrame,
    ) -> None:
        scenario = ScenarioConfig(
            name="test_local_incidence",
            family="information",
            where="local",
            targeting="targeted",
            enable_endogenous_loop=True,
            incidence_mode="importation",
        )
        cfg = tiny_config.model_copy(
            update={
                "knobs": tiny_config.knobs.model_copy(update={"k_incidence": 0.0, "k_feedback_stakes": 0.0}),
                "constants": tiny_config.constants.model_copy(
                    update={
                        "importation_outbreak_floor": 0.01,
                        "importation_targeted_local_bump": 0.01,
                    }
                ),
            }
        )
        pop = tiny_population.with_columns(pl.Series("locality_id", ["A"] * (_N // 2) + ["B"] * (_N - (_N // 2))))
        model = VaccineBehaviourModel(
            cfg=cfg,
            scenario=scenario,
            models=mock_models,
            levers_cfg=mock_levers,
            agent_df=pop,
            seed=42,
        )

        mask = np.zeros(pop.height, dtype=bool)
        mask[: (_N // 2)] = True
        model._target_mask = mask

        outcomes = {
            "vax_willingness_T12": np.full(pop.height, 3.0, dtype=np.float64),
            "mask_when_symptomatic_crowded": np.zeros(pop.height, dtype=np.float64),
            "stay_home_when_symptomatic": np.zeros(pop.height, dtype=np.float64),
            "mask_when_pressure_high": np.zeros(pop.height, dtype=np.float64),
        }
        model._update_community_incidence(outcomes)

        inc_a = model.community_incidence["A"]
        inc_b = model.community_incidence["B"]

        assert inc_b > 0.0
        assert inc_a > inc_b

    def test_local_importation_is_truly_local(
        self,
        tiny_config: SimulationConfig,
        mock_models: dict[str, BackboneModel],
        mock_levers: LeversConfig,
        tiny_population: pl.DataFrame,
    ) -> None:
        scenario = ScenarioConfig(
            name="test_local_only_importation",
            family="information",
            where="local",
            targeting="targeted",
            enable_endogenous_loop=True,
            incidence_mode="importation",
        )
        cfg = tiny_config.model_copy(
            update={
                "knobs": tiny_config.knobs.model_copy(update={"k_incidence": 0.0, "k_feedback_stakes": 0.0}),
                "constants": tiny_config.constants.model_copy(
                    update={
                        "community_incidence_baseline": 0.0,
                        "importation_outbreak_floor": 0.02,
                        "importation_targeted_local_bump": 0.03,
                        "importation_pulse_amplitude": 0.0,
                        "importation_pulse_period_ticks": 0,
                        "importation_pulse_width_ticks": 0,
                    }
                ),
            }
        )
        pop = tiny_population.with_columns(pl.Series("locality_id", ["A"] * (_N // 2) + ["B"] * (_N - (_N // 2))))
        model = VaccineBehaviourModel(
            cfg=cfg,
            scenario=scenario,
            models=mock_models,
            levers_cfg=mock_levers,
            agent_df=pop,
            seed=42,
        )

        mask = np.zeros(pop.height, dtype=bool)
        mask[: (_N // 2)] = True
        model._target_mask = mask
        model._community_incidence = {"A": 0.0, "B": 0.0}

        outcomes = {
            "vax_willingness_T12": np.full(pop.height, 3.0, dtype=np.float64),
            "mask_when_symptomatic_crowded": np.zeros(pop.height, dtype=np.float64),
            "stay_home_when_symptomatic": np.zeros(pop.height, dtype=np.float64),
            "mask_when_pressure_high": np.zeros(pop.height, dtype=np.float64),
        }
        model._update_community_incidence(outcomes)

        assert model.community_incidence["A"] > 0.0
        assert model.community_incidence["A"] > model.community_incidence["B"]
        assert model.community_incidence["B"] == pytest.approx(float(cfg.constants.importation_outbreak_floor))

    def test_local_importation_bump_respects_incidence_schedule_window(
        self,
        tiny_config: SimulationConfig,
        mock_models: dict[str, BackboneModel],
        mock_levers: LeversConfig,
        tiny_population: pl.DataFrame,
    ) -> None:
        scenario = ScenarioConfig(
            name="test_local_importation_window",
            family="information",
            where="local",
            targeting="targeted",
            enable_endogenous_loop=True,
            incidence_mode="importation",
            incidence_schedule=(IncidenceSpec(component="local_bump", start_tick=8, end_tick=18),),
        )
        cfg = tiny_config.model_copy(
            update={
                "knobs": tiny_config.knobs.model_copy(
                    update={"k_incidence": 0.0, "incidence_gamma": 0.0, "k_feedback_stakes": 0.0}
                ),
                "constants": tiny_config.constants.model_copy(
                    update={
                        "community_incidence_baseline": 0.0,
                        "importation_outbreak_floor": 0.02,
                        "importation_targeted_local_bump": 0.03,
                        "importation_pulse_amplitude": 0.0,
                        "importation_pulse_period_ticks": 0,
                        "importation_pulse_width_ticks": 0,
                    }
                ),
            }
        )
        pop = tiny_population.with_columns(pl.Series("locality_id", ["A"] * (_N // 2) + ["B"] * (_N - (_N // 2))))
        model = VaccineBehaviourModel(
            cfg=cfg,
            scenario=scenario,
            models=mock_models,
            levers_cfg=mock_levers,
            agent_df=pop,
            seed=42,
        )

        mask = np.zeros(pop.height, dtype=bool)
        mask[: (_N // 2)] = True
        model._target_mask = mask
        model._community_incidence = {"A": 0.0, "B": 0.0}
        model._current_tick = 0

        outcomes = {
            "vax_willingness_T12": np.full(pop.height, 3.0, dtype=np.float64),
            "mask_when_symptomatic_crowded": np.zeros(pop.height, dtype=np.float64),
            "stay_home_when_symptomatic": np.zeros(pop.height, dtype=np.float64),
            "mask_when_pressure_high": np.zeros(pop.height, dtype=np.float64),
        }
        model._update_community_incidence(outcomes)

        assert model.community_incidence["A"] == pytest.approx(float(cfg.constants.importation_outbreak_floor))
        assert model.community_incidence["B"] == pytest.approx(float(cfg.constants.importation_outbreak_floor))

    def test_importation_multiplier_calibrates_incidence_peak_without_explicit_schedule(
        self,
        tiny_config: SimulationConfig,
        mock_models: dict[str, BackboneModel],
        mock_levers: LeversConfig,
        tiny_population: pl.DataFrame,
    ) -> None:
        cfg = tiny_config.model_copy(
            update={
                "knobs": tiny_config.knobs.model_copy(
                    update={"k_incidence": 0.0, "incidence_gamma": 0.0, "k_feedback_stakes": 0.0}
                ),
                "constants": tiny_config.constants.model_copy(
                    update={
                        "community_incidence_baseline": 0.1,
                        "burn_in_ticks": 12,
                        "importation_outbreak_floor": 0.0,
                        "importation_pulse_amplitude": 0.0,
                        "importation_pulse_period_ticks": 0,
                        "importation_pulse_width_ticks": 0,
                    }
                ),
            }
        )
        baseline = ScenarioConfig(
            name="test_importation_multiplier_baseline",
            incidence_mode="importation",
            importation_multiplier=1.0,
        )
        strong = ScenarioConfig(
            name="test_importation_multiplier_strong",
            incidence_mode="importation",
            importation_multiplier=3.0,
        )

        model_baseline = VaccineBehaviourModel(
            cfg=cfg,
            scenario=baseline,
            models=mock_models,
            levers_cfg=mock_levers,
            agent_df=tiny_population,
            seed=42,
        )
        model_strong = VaccineBehaviourModel(
            cfg=cfg,
            scenario=strong,
            models=mock_models,
            levers_cfg=mock_levers,
            agent_df=tiny_population,
            seed=42,
        )

        model_baseline._community_incidence = dict.fromkeys(model_baseline._community_incidence, 0.1)
        model_strong._community_incidence = dict.fromkeys(model_strong._community_incidence, 0.1)
        model_baseline._current_tick = 12
        model_strong._current_tick = 12

        outcomes = {
            "vax_willingness_T12": np.full(tiny_population.height, 3.0, dtype=np.float64),
            "mask_when_symptomatic_crowded": np.zeros(tiny_population.height, dtype=np.float64),
            "stay_home_when_symptomatic": np.zeros(tiny_population.height, dtype=np.float64),
            "mask_when_pressure_high": np.zeros(tiny_population.height, dtype=np.float64),
        }
        model_baseline._update_community_incidence(outcomes)
        model_strong._update_community_incidence(outcomes)

        baseline_vals = np.array(list(model_baseline.community_incidence.values()), dtype=np.float64)
        strong_vals = np.array(list(model_strong.community_incidence.values()), dtype=np.float64)
        assert np.allclose(baseline_vals, 0.1)
        assert np.allclose(strong_vals, 0.3)
        assert np.allclose(strong_vals, 3.0 * baseline_vals)

    def test_outbreak_global_incidence_strong_targets_configured_peak_multiplier_vs_baseline_importation(
        self,
        tiny_config: SimulationConfig,
        mock_models: dict[str, BackboneModel],
        mock_levers: LeversConfig,
        tiny_population: pl.DataFrame,
    ) -> None:
        cfg = tiny_config.model_copy(
            update={
                "knobs": tiny_config.knobs.model_copy(
                    update={"k_incidence": 0.0, "incidence_gamma": 0.0, "k_feedback_stakes": 0.0}
                ),
                "constants": tiny_config.constants.model_copy(
                    update={
                        "community_incidence_baseline": 0.1,
                        "importation_outbreak_floor": 0.0,
                        "importation_pulse_amplitude": 0.0,
                        "importation_pulse_period_ticks": 0,
                        "importation_pulse_width_ticks": 0,
                    }
                ),
            }
        )
        baseline = ScenarioConfig(name="baseline_importation_test", incidence_mode="importation")
        strong = SCENARIO_LIBRARY["outbreak_global_incidence_strong"]

        model_baseline = VaccineBehaviourModel(
            cfg=cfg,
            scenario=baseline,
            models=mock_models,
            levers_cfg=mock_levers,
            agent_df=tiny_population,
            seed=42,
        )
        model_strong = VaccineBehaviourModel(
            cfg=cfg,
            scenario=strong,
            models=mock_models,
            levers_cfg=mock_levers,
            agent_df=tiny_population,
            seed=42,
        )

        model_baseline._community_incidence = dict.fromkeys(model_baseline._community_incidence, 0.1)
        model_strong._community_incidence = dict.fromkeys(model_strong._community_incidence, 0.1)
        model_baseline._current_tick = 13  # Active window for strong outbreak.
        model_strong._current_tick = 13

        outcomes = {
            "vax_willingness_T12": np.full(tiny_population.height, 3.0, dtype=np.float64),
            "mask_when_symptomatic_crowded": np.zeros(tiny_population.height, dtype=np.float64),
            "stay_home_when_symptomatic": np.zeros(tiny_population.height, dtype=np.float64),
            "mask_when_pressure_high": np.zeros(tiny_population.height, dtype=np.float64),
        }
        model_baseline._update_community_incidence(outcomes)
        model_strong._update_community_incidence(outcomes)

        baseline_vals = np.array(list(model_baseline.community_incidence.values()), dtype=np.float64)
        strong_vals = np.array(list(model_strong.community_incidence.values()), dtype=np.float64)
        assert np.allclose(baseline_vals, 0.1)
        assert np.allclose(strong_vals, strong.importation_multiplier * baseline_vals)

        # Outside the active window, strong scenario falls back to baseline dynamics.
        model_strong._community_incidence = dict.fromkeys(model_strong._community_incidence, 0.1)
        model_strong._importation_peak_reference = None
        model_strong._current_tick = 19
        model_strong._update_community_incidence(outcomes)
        assert np.allclose(np.array(list(model_strong.community_incidence.values()), dtype=np.float64), 0.1)

    def test_incidence_fixed_point_initialization_is_stationary(
        self,
        tiny_config: SimulationConfig,
        mock_models: dict[str, BackboneModel],
        mock_levers: LeversConfig,
        tiny_population: pl.DataFrame,
    ) -> None:
        scenario = ScenarioConfig(
            name="test_incidence_fixed_point_stationarity",
            communication_enabled=False,
            enable_endogenous_loop=True,
            incidence_mode="importation",
        )
        cfg = tiny_config.model_copy(
            update={
                "knobs": tiny_config.knobs.model_copy(update={"k_feedback_stakes": 0.0}),
                "constants": tiny_config.constants.model_copy(
                    update={
                        "importation_outbreak_floor": 0.0,
                        "importation_targeted_local_bump": 0.0,
                        "importation_pulse_amplitude": 0.0,
                        "importation_pulse_period_ticks": 0,
                        "importation_pulse_width_ticks": 0,
                    }
                ),
            }
        )
        pop = tiny_population.with_columns(pl.Series("locality_id", ["A"] * (_N // 2) + ["B"] * (_N - (_N // 2))))
        model = VaccineBehaviourModel(
            cfg=cfg,
            scenario=scenario,
            models=mock_models,
            levers_cfg=mock_levers,
            agent_df=pop,
            seed=42,
        )

        outcomes = {
            "vax_willingness_T12": np.full(pop.height, 4.0, dtype=np.float64),
            "mask_when_symptomatic_crowded": np.full(pop.height, 4.0, dtype=np.float64),
            "stay_home_when_symptomatic": np.full(pop.height, 4.0, dtype=np.float64),
            "mask_when_pressure_high": np.full(pop.height, 4.0, dtype=np.float64),
        }

        communities = model.agents.df["locality_id"].cast(pl.Utf8).to_numpy().astype(str)
        vax = normalize_outcome_to_unit_interval(outcomes["vax_willingness_T12"], "vax_willingness_T12")
        npi = np.mean(
            np.vstack(
                [
                    normalize_outcome_to_unit_interval(
                        outcomes["mask_when_symptomatic_crowded"],
                        "mask_when_symptomatic_crowded",
                    ),
                    normalize_outcome_to_unit_interval(
                        outcomes["stay_home_when_symptomatic"],
                        "stay_home_when_symptomatic",
                    ),
                    normalize_outcome_to_unit_interval(
                        outcomes["mask_when_pressure_high"],
                        "mask_when_pressure_high",
                    ),
                ]
            ),
            axis=0,
        )

        beta = float(cfg.knobs.k_incidence)
        gamma = float(cfg.knobs.incidence_gamma)
        alpha_v = float(cfg.constants.incidence_alpha_vax)
        alpha_p = float(cfg.constants.incidence_alpha_protection)
        for unit in np.unique(communities):
            idx = communities == unit
            mitigation = float(
                np.clip(1.0 - alpha_v * float(np.mean(vax[idx])) - alpha_p * float(np.mean(npi[idx])), 0.0, 1.0)
            )
            denom = beta * mitigation
            if denom <= 0.0:
                fixed_point = 0.0
            else:
                fixed_point = float(np.clip(1.0 - (gamma / denom), 0.0, 1.0))
            model._community_incidence[str(unit)] = fixed_point
        model._prev_community_incidence = dict(model._community_incidence)

        max_step_delta = 0.0
        for _ in range(6):
            before = np.array(list(model.community_incidence.values()), dtype=np.float64)
            model._update_community_incidence(outcomes)
            after = np.array(list(model.community_incidence.values()), dtype=np.float64)
            max_step_delta = max(max_step_delta, float(np.max(np.abs(after - before))))

        assert max_step_delta < 1e-9

    def test_from_preloaded(
        self,
        tiny_config: SimulationConfig,
        baseline_scenario: ScenarioConfig,
        mock_models: dict[str, BackboneModel],
        mock_levers: LeversConfig,
        tiny_population: pl.DataFrame,
    ) -> None:
        """from_preloaded should build a functional model."""
        model = VaccineBehaviourModel.from_preloaded(
            cfg=tiny_config,
            scenario=baseline_scenario,
            models=mock_models,
            levers_cfg=mock_levers,
            agent_df=tiny_population,
            seed=42,
        )
        model.run_model(n_steps=3)
        assert model._current_tick == 3
        assert model.outcome_trajectory.height == 3

    def test_localized_vs_uniform_diverge(
        self,
        tiny_config: SimulationConfig,
        mock_models: dict[str, BackboneModel],
        mock_levers: LeversConfig,
        tiny_population: pl.DataFrame,
    ) -> None:
        localized = ScenarioConfig(
            name="localized_test",
            family="combined",
            state_signals=(
                SignalSpec(
                    target_column="trust_national_gov_vax_info",
                    intensity=1.0,
                    signal_id="trust_national_gov_vax_info_shift",
                    signal_type="campaign",
                    start_tick=2,
                    end_tick=5,
                ),
            ),
            communication_enabled=True,
            targeting="targeted",
            seed_rule="high_exposure",
        )
        uniform = ScenarioConfig(
            name="uniform_test",
            family="combined",
            state_signals=localized.state_signals,
            communication_enabled=True,
            targeting="uniform",
        )
        network = build_contact_network(tiny_population, seed=42)

        model_local = VaccineBehaviourModel(
            cfg=tiny_config,
            scenario=localized,
            models=mock_models,
            levers_cfg=mock_levers,
            agent_df=tiny_population.clone(),
            network=network,
            seed=42,
        )
        model_uniform = VaccineBehaviourModel(
            cfg=tiny_config,
            scenario=uniform,
            models=mock_models,
            levers_cfg=mock_levers,
            agent_df=tiny_population.clone(),
            network=network,
            seed=42,
        )
        model_local.run_model(n_steps=14)
        model_uniform.run_model(n_steps=14)

        local_mean = float(model_local.population_final["trust_national_gov_vax_info"].mean())
        uniform_mean = float(model_uniform.population_final["trust_national_gov_vax_info"].mean())
        assert not np.isclose(local_mean, uniform_mean)

    def test_importation_drives_closed_loop_dynamics(
        self,
        tiny_config: SimulationConfig,
        mock_models: dict[str, BackboneModel],
        mock_levers: LeversConfig,
        tiny_population: pl.DataFrame,
    ) -> None:
        tiny_config.constants = tiny_config.constants.model_copy(update={"importation_targeted_local_bump": 0.12})
        scenario = ScenarioConfig(
            name="importation_test",
            family="combined",
            communication_enabled=True,
            targeting="targeted",
            seed_rule="community_targeted",
            enable_endogenous_loop=True,
        )

        model = VaccineBehaviourModel(
            cfg=tiny_config,
            scenario=scenario,
            models=mock_models,
            levers_cfg=mock_levers,
            agent_df=tiny_population.clone(),
            network=build_contact_network(tiny_population, seed=42),
            seed=42,
        )
        model.run_model(n_steps=4)
        incidence = model.incidence_trajectory["mean_incidence"].to_numpy().astype(np.float64)
        assert np.max(incidence) >= (tiny_config.constants.community_incidence_baseline - 1e-9)
        assert np.std(incidence) > 0.0


# =========================================================================
# Tests: run_single_replication via simulator
# =========================================================================


class TestRunSingleReplication:
    """Test the simulator's run_single_replication function."""

    def test_returns_simulation_result(
        self,
        tiny_config: SimulationConfig,
        baseline_scenario: ScenarioConfig,
        mock_models: dict[str, BackboneModel],
        mock_levers: LeversConfig,
        tiny_population: pl.DataFrame,
    ) -> None:
        """Should return a SimulationResult with expected fields."""
        result = run_single_replication(
            cfg=tiny_config,
            scenario=baseline_scenario,
            models=mock_models,
            levers_cfg=mock_levers,
            agent_df=tiny_population,
            rep_id=0,
            seed=42,
        )
        assert isinstance(result, SimulationResult)
        assert result.outcome_trajectory.height == 5
        assert result.mediator_trajectory.height == 5
        assert result.population_t0.height == _N
        assert result.population_final.height == _N
        assert result.seed == 42
        assert result.rep_id == 0

    def test_access_intervention_changes_willingness_auc(
        self,
        tiny_config: SimulationConfig,
        mock_models: dict[str, BackboneModel],
        mock_levers: LeversConfig,
        tiny_population: pl.DataFrame,
    ) -> None:
        cfg = tiny_config.model_copy(update={"n_steps": 12, "deterministic": True})
        baseline = SCENARIO_LIBRARY["baseline"]
        access = SCENARIO_LIBRARY["access_intervention"]
        network = build_contact_network(tiny_population, seed=42)

        base_result = run_single_replication(
            cfg=cfg,
            scenario=baseline,
            models=mock_models,
            levers_cfg=mock_levers,
            agent_df=tiny_population,
            network=network,
            rep_id=0,
            seed=42,
        )
        access_result = run_single_replication(
            cfg=cfg,
            scenario=access,
            models=mock_models,
            levers_cfg=mock_levers,
            agent_df=tiny_population,
            network=network,
            rep_id=0,
            seed=42,
        )

        base_auc = float(base_result.outcome_trajectory["vax_willingness_T12_mean"].sum())
        access_auc = float(access_result.outcome_trajectory["vax_willingness_T12_mean"].sum())
        assert np.isfinite(base_auc)
        assert np.isfinite(access_auc)

    def test_with_network(
        self,
        tiny_config: SimulationConfig,
        baseline_scenario: ScenarioConfig,
        mock_models: dict[str, BackboneModel],
        mock_levers: LeversConfig,
        tiny_population: pl.DataFrame,
    ) -> None:
        """Should work with a pre-built contact network."""
        net = build_contact_network(tiny_population, seed=42)
        result = run_single_replication(
            cfg=tiny_config,
            scenario=baseline_scenario,
            models=mock_models,
            levers_cfg=mock_levers,
            agent_df=tiny_population,
            network=net,
            rep_id=0,
            seed=42,
        )
        assert result.outcome_trajectory.height == 5

    def test_access_intervention_changes_vax_auc(
        self,
        tiny_config: SimulationConfig,
        mock_models: dict[str, BackboneModel],
        mock_levers: LeversConfig,
        tiny_population: pl.DataFrame,
    ) -> None:
        """Access intervention should lift vax willingness AUC vs baseline."""
        cfg = tiny_config.model_copy(update={"n_steps": 12, "deterministic": True})
        baseline = SCENARIO_LIBRARY["baseline"]
        access = SCENARIO_LIBRARY["access_intervention"]

        base = run_single_replication(
            cfg=cfg,
            scenario=baseline,
            models=mock_models,
            levers_cfg=mock_levers,
            agent_df=tiny_population,
            rep_id=0,
            seed=123,
        )
        interv = run_single_replication(
            cfg=cfg,
            scenario=access,
            models=mock_models,
            levers_cfg=mock_levers,
            agent_df=tiny_population,
            rep_id=0,
            seed=123,
        )

        base_vals = base.outcome_trajectory["vax_willingness_T12_mean"].to_numpy().astype(np.float64)
        interv_vals = interv.outcome_trajectory["vax_willingness_T12_mean"].to_numpy().astype(np.float64)
        base_auc = float(np.trapezoid(base_vals))
        interv_auc = float(np.trapezoid(interv_vals))
        assert np.isfinite(base_auc)
        assert np.isfinite(interv_auc)


# =========================================================================
# Tests: Mini simulation loop (manual — component integration)
# =========================================================================


class TestMiniSimulationLoop:
    """Run a miniature simulation loop to verify components integrate."""

    def test_single_tick(
        self,
        tiny_population: pl.DataFrame,
        npi_model: BackboneModel,
        ordinal_model: BackboneModel,
    ) -> None:
        """Run a single tick: predict → network → update → predict."""
        y_npi = predict_outcome(tiny_population, npi_model, deterministic=True)
        y_ord = predict_outcome(tiny_population, ordinal_model, deterministic=True)
        assert y_npi.shape == (_N,)
        assert y_ord.shape == (_N,)

        net = build_contact_network(tiny_population, seed=42)
        peer_means = compute_peer_means(tiny_population, net, columns=["social_norms_vax_avg"])

        sig = SignalSpec(
            target_column="vaccine_risk_avg",
            intensity=0.3,
            signal_type="pulse",
            start_tick=0,
        )
        updated = apply_all_state_updates(
            tiny_population,
            signals=[sig],
            tick=0,
            peer_means=peer_means,
            k_social=1.0,
            social_weight_norm=0.5,
            social_weight_trust=0.0,
            social_weight_risk=0.0,
        )
        assert updated.shape == tiny_population.shape

        y_npi_new = predict_outcome(updated, npi_model, deterministic=True)
        y_ord_new = predict_outcome(updated, ordinal_model, deterministic=True)
        assert not np.allclose(y_npi, y_npi_new) or not np.allclose(y_ord, y_ord_new)

    def test_multi_tick_loop(
        self,
        tiny_population: pl.DataFrame,
        npi_model: BackboneModel,
    ) -> None:
        """Run a 5-tick loop and verify trajectories accumulate."""
        state = tiny_population.clone()
        net = build_contact_network(state, seed=42)
        trajectory: list[dict[str, float]] = []

        for tick in range(5):
            y = predict_outcome(state, npi_model, deterministic=True)
            trajectory.append({"tick": tick, "mean_y": float(y.mean())})

            peer_means = compute_peer_means(state, net, columns=["social_norms_vax_avg"])
            state = apply_all_state_updates(
                state,
                signals=[],
                tick=tick,
                peer_means=peer_means,
                k_social=1.0,
                social_weight_norm=0.5,
                social_weight_trust=0.0,
                social_weight_risk=0.0,
            )

        assert len(trajectory) == 5
        assert all("mean_y" in t for t in trajectory)
