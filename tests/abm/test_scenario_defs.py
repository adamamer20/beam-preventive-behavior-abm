"""Tests for scenario definition lookup and alias handling."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from beam_abm.abm.data_contracts import BackboneModel, LeversConfig, OutcomeLever
from beam_abm.abm.scenario_defs import (
    MUTABLE_LATENT_ENGINE_SCENARIO_LIBRARY,
    SCENARIO_LIBRARY,
    UPSTREAM_SIGNAL_SCENARIO_LIBRARY,
    ScenarioConfig,
    get_scenario,
    load_scenarios_from_json,
    validate_scenario_effect_mapping,
    validate_scenario_uniqueness,
)
from beam_abm.abm.state_schema import MUTABLE_COLUMNS
from beam_abm.abm.state_update import SignalSpec


def test_canonical_baseline_exists() -> None:
    assert "baseline" in SCENARIO_LIBRARY


def test_upstream_and_mutable_scenario_libraries_are_disjoint() -> None:
    overlap = set(UPSTREAM_SIGNAL_SCENARIO_LIBRARY) & set(MUTABLE_LATENT_ENGINE_SCENARIO_LIBRARY)
    assert overlap == set()


def test_mutable_latent_engine_scenarios_exist() -> None:
    expected = {
        "pe_mutable_engine_institutional_trust",
        "pe_mutable_engine_legitimacy_scepticism",
        "pe_mutable_engine_vaccine_risk",
        "pe_mutable_engine_covid_danger",
        "pe_mutable_engine_social_norms",
        "pe_mutable_engine_fivec_constraints",
        "pe_mutable_engine_group_trust",
    }
    assert expected.issubset(set(MUTABLE_LATENT_ENGINE_SCENARIO_LIBRARY))


def test_get_scenario_supports_mutable_latent_engine_library() -> None:
    scenario = get_scenario("pe_mutable_engine_vaccine_risk")
    assert scenario.name == "pe_mutable_engine_vaccine_risk"
    targets = {sig.target_column for sig in scenario.state_signals}
    assert targets.issubset(MUTABLE_COLUMNS)


def test_get_scenario_rejects_comm_off() -> None:
    with pytest.raises(KeyError):
        get_scenario("comm_off")


def test_rejects_removed_legacy_field(tmp_path) -> None:  # type: ignore[no-untyped-def]
    payload = {
        "legacy": {
            "family": "custom",
            "state_signals": [],
            "norm_update_gain": 0.5,
        }
    }
    path = tmp_path / "scenarios.json"
    path.write_text(__import__("json").dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="removed field"):
        load_scenarios_from_json(path)


def test_access_intervention_targets_vax_relevant_levers() -> None:
    scenario = SCENARIO_LIBRARY["access_intervention"]
    targets = {sig.target_column for sig in scenario.state_signals}
    assert "booking" in targets
    assert "vax_hub" in targets
    assert "fivec_pro_vax_idx" not in targets


def test_new_mechanism_scenarios_exist() -> None:
    expected = {
        "null_world",
        "social_only",
        "baseline_no_communication",
        "baseline_importation_no_communication",
        "baseline_no_peer_pull",
        "baseline_no_observed_norms",
        "baseline_importation_no_peer_pull",
        "baseline_importation_no_observed_norms",
        "trust_scandal_institutional",
        "trust_repair_institutional",
        "trust_scandal_legitimacy",
        "trust_repair_legitimacy",
        "trust_scandal_group_vax",
        "trust_repair_group_vax",
        "trust_scandal_socialinfo",
        "trust_repair_socialinfo",
        "trust_scandal_onlineinfo",
        "trust_repair_onlineinfo",
        "trust_scandal_bundle",
        "trust_repair_bundle",
        "targeted_norm_seed",
        "outbreak_misinfo_combined",
    }
    assert expected.issubset(set(SCENARIO_LIBRARY))


def test_baseline_channel_ablations_have_expected_toggles() -> None:
    canonical = SCENARIO_LIBRARY["baseline_importation"]
    no_comm = SCENARIO_LIBRARY["baseline_importation_no_communication"]
    no_peer = SCENARIO_LIBRARY["baseline_importation_no_peer_pull"]
    no_obs = SCENARIO_LIBRARY["baseline_importation_no_observed_norms"]

    assert canonical.communication_enabled is True
    assert canonical.peer_pull_enabled is True
    assert canonical.observed_norms_enabled is True

    assert no_comm.communication_enabled is False
    assert no_comm.peer_pull_enabled is False
    assert no_comm.observed_norms_enabled is False

    assert no_peer.communication_enabled is True
    assert no_peer.peer_pull_enabled is False
    assert no_peer.observed_norms_enabled is True

    assert no_obs.communication_enabled is True
    assert no_obs.peer_pull_enabled is True
    assert no_obs.observed_norms_enabled is False


def test_trust_scenario_taxonomy_targets_expected_state_channels() -> None:
    assert {sig.target_column for sig in SCENARIO_LIBRARY["trust_scandal_institutional"].state_signals} == {
        "institutional_trust_avg"
    }
    assert {sig.target_column for sig in SCENARIO_LIBRARY["trust_repair_institutional"].state_signals} == {
        "institutional_trust_avg"
    }
    assert {sig.target_column for sig in SCENARIO_LIBRARY["trust_scandal_legitimacy"].state_signals} == {
        "covax_legitimacy_scepticism_idx"
    }
    assert {sig.target_column for sig in SCENARIO_LIBRARY["trust_repair_legitimacy"].state_signals} == {
        "covax_legitimacy_scepticism_idx"
    }
    assert {sig.target_column for sig in SCENARIO_LIBRARY["trust_scandal_group_vax"].state_signals} == {
        "group_trust_vax"
    }
    assert {sig.target_column for sig in SCENARIO_LIBRARY["trust_repair_group_vax"].state_signals} == {
        "group_trust_vax"
    }


def test_trust_scenario_taxonomy_targets_expected_info_channels() -> None:
    assert {sig.target_column for sig in SCENARIO_LIBRARY["trust_scandal_socialinfo"].state_signals} == {"trust_social"}
    assert {sig.target_column for sig in SCENARIO_LIBRARY["trust_repair_socialinfo"].state_signals} == {"trust_social"}
    assert {sig.target_column for sig in SCENARIO_LIBRARY["trust_scandal_onlineinfo"].state_signals} == {
        "trust_onlineinfo"
    }
    assert {sig.target_column for sig in SCENARIO_LIBRARY["trust_repair_onlineinfo"].state_signals} == {
        "trust_onlineinfo"
    }


def test_trust_bundle_scenarios_are_explicit_multi_channel_shocks() -> None:
    scandal_targets = {sig.target_column for sig in SCENARIO_LIBRARY["trust_scandal_bundle"].state_signals}
    repair_targets = {sig.target_column for sig in SCENARIO_LIBRARY["trust_repair_bundle"].state_signals}
    expected = {"institutional_trust_avg", "covax_legitimacy_scepticism_idx", "group_trust_vax"}
    assert scandal_targets == expected
    assert repair_targets == expected


def test_outbreak_local_incidence_uses_importation_multiplier_window() -> None:
    scenario = SCENARIO_LIBRARY["outbreak_local_incidence_moderate"]
    assert scenario.incidence_mode == "importation"
    assert scenario.importation_multiplier == pytest.approx(3.0)
    assert len(scenario.incidence_schedule) == 1
    window = scenario.incidence_schedule[0]
    assert window.component == "importation_multiplier"
    assert window.start_tick == 12
    assert window.end_tick == 18


def test_salience_scenarios_use_importation_baseline() -> None:
    assert SCENARIO_LIBRARY["outbreak_salience_global"].incidence_mode == "importation"
    assert SCENARIO_LIBRARY["outbreak_local_salience"].incidence_mode == "importation"


def test_misinfo_and_info_shift_semantics_are_distinct() -> None:
    misinfo_ids = {sig.signal_id for sig in SCENARIO_LIBRARY["misinfo_shock"].state_signals}
    online_ids = {sig.signal_id for sig in SCENARIO_LIBRARY["trust_repair_onlineinfo"].state_signals}
    social_ids = {sig.signal_id for sig in SCENARIO_LIBRARY["trust_repair_socialinfo"].state_signals}

    assert {
        "trust_traditional_shift_down",
        "trust_gp_vax_info_shift_down",
        "trust_who_vax_info_shift_down",
        "trust_national_gov_vax_info_shift_down",
    }.issubset(misinfo_ids)
    assert "trust_social_shift" not in misinfo_ids
    assert "trust_onlineinfo_shift" not in misinfo_ids

    assert "trust_onlineinfo_shift" in online_ids
    assert "trust_social_shift" in social_ids


def test_norm_campaign_does_not_target_derived_composite() -> None:
    scenario = SCENARIO_LIBRARY["norm_campaign"]
    targets = {sig.target_column for sig in scenario.state_signals}
    assert "social_norms_vax_avg" not in targets
    assert "social_norms_descriptive_vax" in targets


def test_targeted_norm_seed_uses_descriptive_norm_channel() -> None:
    scenario = SCENARIO_LIBRARY["targeted_norm_seed"]
    targets = {sig.target_column for sig in scenario.state_signals}
    assert "social_norms_vax_avg" not in targets
    assert "vax_convo_freq_cohabitants" in targets


def test_scenario_library_uses_unit_intensity_signal_specs() -> None:
    for scenario in SCENARIO_LIBRARY.values():
        for signal in (*scenario.state_signals, *scenario.choice_signals):
            assert abs(signal.intensity) == pytest.approx(1.0)


def test_policy_signals_start_after_initialization_transient() -> None:
    for scenario in SCENARIO_LIBRARY.values():
        for signal in (*scenario.state_signals, *scenario.choice_signals):
            assert signal.start_tick >= 2


def test_trust_scandal_and_repair_mutable_signals_use_expected_directions() -> None:
    scandal_inst = SCENARIO_LIBRARY["trust_scandal_institutional"].state_signals
    repair_inst = SCENARIO_LIBRARY["trust_repair_institutional"].state_signals
    scandal_group = SCENARIO_LIBRARY["trust_scandal_group_vax"].state_signals
    repair_group = SCENARIO_LIBRARY["trust_repair_group_vax"].state_signals
    scandal_legit = SCENARIO_LIBRARY["trust_scandal_legitimacy"].state_signals
    repair_legit = SCENARIO_LIBRARY["trust_repair_legitimacy"].state_signals
    assert all(sig.intensity < 0 for sig in scandal_inst)
    assert all(sig.intensity > 0 for sig in repair_inst)
    assert all(sig.intensity < 0 for sig in scandal_group)
    assert all(sig.intensity > 0 for sig in repair_group)
    assert all(sig.intensity > 0 for sig in scandal_legit)
    assert all(sig.intensity < 0 for sig in repair_legit)


def test_belief_update_spec_contains_symmetric_downshift_signals() -> None:
    spec = json.loads(Path("evaluation/specs/belief_update_targets.json").read_text(encoding="utf-8"))
    ids = {str(item.get("id")) for item in spec.get("signals", []) if isinstance(item, dict)}
    expected = {
        "trust_onlineinfo_shift_down",
        "trust_social_shift_down",
        "trust_traditional_shift_down",
        "trust_gp_vax_info_shift_down",
        "trust_who_vax_info_shift_down",
        "trust_national_gov_vax_info_shift_down",
        "booking_access_worsens",
        "vax_hub_access_worsens",
        "vax_conversation_freq_drop",
        "impediment_off",
    }
    assert expected.issubset(ids)


def test_allows_signals_without_primary_outcomes(tmp_path) -> None:  # type: ignore[no-untyped-def]
    payload = {
        "bad": {
            "family": "custom",
            "state_signals": [{"target_column": "vaccine_risk_avg", "intensity": 0.5, "signal_id": "x"}],
        }
    }
    path = tmp_path / "scenarios.json"
    path.write_text(__import__("json").dumps(payload), encoding="utf-8")

    scenarios = load_scenarios_from_json(path)
    assert len(scenarios) == 1
    assert scenarios[0].primary_outcomes == ()


def test_duplicate_scenarios_are_rejected() -> None:
    signal = SignalSpec(target_column="vaccine_risk_avg", intensity=0.5, signal_id="dup")
    first = ScenarioConfig(
        name="first",
        state_signals=(signal,),
        primary_outcomes=("vax_willingness_T12",),
        communication_enabled=True,
        enable_endogenous_loop=True,
    )
    second = ScenarioConfig(
        name="second",
        state_signals=(signal,),
        primary_outcomes=("vax_willingness_T12",),
        communication_enabled=True,
        enable_endogenous_loop=True,
    )
    with pytest.raises(ValueError, match="Duplicate scenarios detected"):
        validate_scenario_uniqueness([first, second])


def test_scenario_effect_mapping_requires_meaningful_lever() -> None:
    scenario = ScenarioConfig(
        name="test_mapping",
        state_signals=(SignalSpec(target_column="vaccine_risk_avg", intensity=0.5, signal_id="s"),),
        primary_outcomes=("vax_willingness_T12",),
    )
    levers_cfg = LeversConfig(
        outcomes={
            "vax_willingness_T12": OutcomeLever(
                outcome="vax_willingness_T12",
                model_ref="mock",
                levers=("vaccine_risk_avg",),
                include_cols=(),
            )
        }
    )
    model = BackboneModel(
        name="mock",
        model_type="binary",
        terms=("vaccine_risk_avg",),
        coefficients=np.array([0.01]),
        thresholds=None,
        intercept=0.0,
        blocks={},
    )
    with pytest.raises(ValueError, match="Scenario-to-lever mapping failed"):
        validate_scenario_effect_mapping(
            [scenario],
            levers_cfg=levers_cfg,
            models={"vax_willingness_T12": model},
            agents_for_pe_ref=None,
            coeff_abs_threshold=0.05,
            pe_ref_abs_threshold=0.01,
        )


def test_scenario_effect_mapping_auto_mode_is_non_blocking_for_unmapped_targets() -> None:
    scenario = ScenarioConfig(
        name="auto_unmapped",
        state_signals=(SignalSpec(target_column="side_effects_other", intensity=1.0, signal_id="s"),),
        primary_outcomes=(),
    )
    levers_cfg = LeversConfig(
        outcomes={
            "vax_willingness_T12": OutcomeLever(
                outcome="vax_willingness_T12",
                model_ref="mock",
                levers=("social_norms_vax_avg",),
                include_cols=(),
            )
        }
    )
    model = BackboneModel(
        name="mock",
        model_type="binary",
        terms=("social_norms_vax_avg",),
        coefficients=np.array([0.01]),
        thresholds=None,
        intercept=0.0,
        blocks={},
    )

    validate_scenario_effect_mapping(
        [scenario],
        levers_cfg=levers_cfg,
        models={"vax_willingness_T12": model},
        agents_for_pe_ref=None,
        coeff_abs_threshold=0.05,
        pe_ref_abs_threshold=0.01,
    )


def test_scenario_effect_mapping_accepts_norm_channel_alias() -> None:
    scenario = ScenarioConfig(
        name="norm_alias",
        state_signals=(
            SignalSpec(target_column="social_norms_descriptive_vax", intensity=0.5, signal_id="norm_alias"),
        ),
        primary_outcomes=("vax_willingness_T12",),
    )
    levers_cfg = LeversConfig(
        outcomes={
            "vax_willingness_T12": OutcomeLever(
                outcome="vax_willingness_T12",
                model_ref="mock",
                levers=("social_norms_vax_avg",),
                include_cols=(),
            )
        }
    )
    model = BackboneModel(
        name="mock",
        model_type="binary",
        terms=("social_norms_vax_avg",),
        coefficients=np.array([0.2]),
        thresholds=None,
        intercept=0.0,
        blocks={},
    )

    validate_scenario_effect_mapping(
        [scenario],
        levers_cfg=levers_cfg,
        models={"vax_willingness_T12": model},
        agents_for_pe_ref=None,
        coeff_abs_threshold=0.05,
        pe_ref_abs_threshold=0.01,
    )


def test_scenario_effect_mapping_ignores_unrelated_outcome_prereqs_in_pe_ref() -> None:
    scenario = ScenarioConfig(
        name="pe_ref_scope",
        state_signals=(SignalSpec(target_column="vaccine_risk_avg", intensity=0.5, signal_id="risk_shift"),),
        primary_outcomes=("mask_when_symptomatic_crowded",),
    )
    levers_cfg = LeversConfig(
        outcomes={
            "mask_when_symptomatic_crowded": OutcomeLever(
                outcome="mask_when_symptomatic_crowded",
                model_ref="mask",
                levers=("vaccine_risk_avg",),
                include_cols=(),
            ),
            "vax_willingness_T12": OutcomeLever(
                outcome="vax_willingness_T12",
                model_ref="vax",
                levers=("social_norms_vax_avg",),
                include_cols=(),
            ),
        }
    )
    mask_model = BackboneModel(
        name="mask",
        model_type="binary",
        terms=("vaccine_risk_avg",),
        coefficients=np.array([0.001]),
        thresholds=None,
        intercept=0.0,
        blocks={},
    )
    unrelated_vax_model = BackboneModel(
        name="vax",
        model_type="binary",
        terms=("social_norms_vax_avg",),
        coefficients=np.array([0.2]),
        thresholds=None,
        intercept=0.0,
        blocks={},
    )
    agents_for_pe_ref = pl.DataFrame({"vaccine_risk_avg": [2.0, 4.0, 6.0]})

    with pytest.raises(ValueError, match="Scenario-to-lever mapping failed"):
        validate_scenario_effect_mapping(
            [scenario],
            levers_cfg=levers_cfg,
            models={
                "mask_when_symptomatic_crowded": mask_model,
                "vax_willingness_T12": unrelated_vax_model,
            },
            agents_for_pe_ref=agents_for_pe_ref,
            coeff_abs_threshold=0.05,
            pe_ref_abs_threshold=0.01,
        )


def test_scenario_effect_mapping_allows_precursor_only_targets() -> None:
    scenario = ScenarioConfig(
        name="precursor_only",
        state_signals=(SignalSpec(target_column="booking", intensity=1.0, signal_id="booking_access_improves"),),
        primary_outcomes=("vax_willingness_T12",),
    )
    levers_cfg = LeversConfig(
        outcomes={
            "vax_willingness_T12": OutcomeLever(
                outcome="vax_willingness_T12",
                model_ref="mock",
                levers=("social_norms_vax_avg",),
                include_cols=(),
            )
        }
    )
    model = BackboneModel(
        name="mock",
        model_type="binary",
        terms=("social_norms_vax_avg",),
        coefficients=np.array([0.2]),
        thresholds=None,
        intercept=0.0,
        blocks={},
    )

    validate_scenario_effect_mapping(
        [scenario],
        levers_cfg=levers_cfg,
        models={"vax_willingness_T12": model},
        agents_for_pe_ref=None,
        coeff_abs_threshold=0.05,
        pe_ref_abs_threshold=0.01,
    )
