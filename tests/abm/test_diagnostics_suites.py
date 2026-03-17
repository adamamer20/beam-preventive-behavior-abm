"""Regression tests for diagnostics suite adapters and output artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import polars as pl
import pytest

from beam_abm.abm.benchmarking import GateResult, NonDegeneracyReport, PerformanceReport
from beam_abm.abm.diagnostics.core import run_core_suite
from beam_abm.abm.diagnostics.credibility import SharedData, run_credibility_suite
from beam_abm.abm.diagnostics.importation import run_importation_suite


def test_importation_outputs_include_key_files(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    runtime = SimpleNamespace(cfg=SimpleNamespace(n_agents=2000, n_steps=24))

    monkeypatch.setattr(
        "beam_abm.abm.diagnostics.importation.run_path_activation_checks",
        lambda *args, **kwargs: (
            pl.DataFrame({"seed": [42], "tick": [0], "series": ["x"], "delta": [0.1], "incidence_delta": [0.2]}),
            [{"seed": 42, "mediator": "disease_fear_avg", "auc_delta": 0.1, "peak_lag_ticks": 1, "peak_delta": 0.2}],
        ),
    )
    monkeypatch.setattr(
        "beam_abm.abm.diagnostics.importation.run_static_choice_sensitivity",
        lambda *args, **kwargs: [{"mediator": "m", "component": "c", "delta_mean": 0.1, "shift_sd": 0.2}],
    )
    monkeypatch.setattr(
        "beam_abm.abm.diagnostics.importation.run_toggle_checks",
        lambda *args, **kwargs: (
            pl.DataFrame({"seed": [42], "tick": [0], "variant": ["k_social_0"], "delta_protection": [0.1]}),
            [{"seed": 42, "variant": "k_social_0", "mean_delta_protection": 0.01, "peak_abs_delta_protection": 0.02}],
        ),
    )
    monkeypatch.setattr(
        "beam_abm.abm.diagnostics.importation.run_dose_response_checks",
        lambda *args, **kwargs: [
            {
                "amplitude": 0.016,
                "peak_delta_incidence": 0.1,
                "peak_abs_delta_protection": 0.1,
                "peak_tick_incidence_delta": 2,
                "peak_tick_protection_abs_delta": 3,
            }
        ],
    )
    monkeypatch.setattr(
        "beam_abm.abm.diagnostics.importation.run_timing_placebo_checks",
        lambda *args, **kwargs: {"timing_shift_observed": True, "placebo_small_effect": True},
    )
    monkeypatch.setattr(
        "beam_abm.abm.diagnostics.importation.run_coupling_diagnostics",
        lambda *args, **kwargs: {
            "per_seed": [
                {"seed": 42, "slope_delta_incidence_on_protection": -0.1, "slope_delta_protection_on_incidence": 0.1}
            ],
            "slope_delta_incidence_on_protection": {"mean": -0.1, "ci_low": -0.1, "ci_high": -0.1},
            "slope_delta_protection_on_incidence": {"mean": 0.1, "ci_low": 0.1, "ci_high": 0.1},
        },
    )

    result = run_importation_suite(runtime=runtime, suite_output_dir=tmp_path, seeds=[42, 43, 44, 45, 46])

    assert result.passed is True
    for name in (
        "path_activation_summary.csv",
        "static_choice_sensitivity.csv",
        "toggle_summary.csv",
        "dose_response_summary.csv",
        "artifact_falsification_summary.json",
        "report.json",
    ):
        assert (tmp_path / name).exists()


def test_credibility_outputs_include_key_files(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    fake_shared = SharedData(
        cfg=SimpleNamespace(n_agents=2000, n_steps=24),
        models={},
        levers_cfg=SimpleNamespace(),
        agent_df=pl.DataFrame({"x": [1]}),
        network=None,
    )

    monkeypatch.setattr("beam_abm.abm.diagnostics.credibility.load_shared_data", lambda *args, **kwargs: fake_shared)
    monkeypatch.setattr(
        "beam_abm.abm.diagnostics.credibility.run_instrumented_detailed",
        lambda *args, **kwargs: (
            [
                {
                    "tick": 0,
                    "construct": "social_norms_descriptive_vax",
                    "delta_observed_norm": 0.0,
                    "delta_attractor": 0.0,
                    "delta_peer_pull": 0.0,
                    "delta_clip": 0.0,
                }
            ],
            [
                {
                    "tick": 0,
                    "construct": "social_norms_descriptive_vax",
                    "drift": 0.0,
                    "std_post": 1.0,
                    "attractor_on": False,
                }
            ],
        ),
    )
    monkeypatch.setattr(
        "beam_abm.abm.diagnostics.credibility.run_2x2_social_ablation",
        lambda *args, **kwargs: pl.DataFrame(
            {
                "cell": ["both_off"],
                "attractor": ["att_off"],
                "construct": ["social_norms_descriptive_vax"],
                "drift": [0.0],
                "std_post": [1.0],
            }
        ),
    )
    monkeypatch.setattr(
        "beam_abm.abm.diagnostics.credibility.run_pulse_response",
        lambda *args, **kwargs: pl.DataFrame(
            {"label": ["baseline_import", "pulse_import"], "mean_incidence": [0.1, 0.11]}
        ),
    )
    monkeypatch.setattr(
        "beam_abm.abm.diagnostics.credibility.run_long_horizon_stationarity",
        lambda *args, **kwargs: pl.DataFrame(
            {"horizon_steps": [52], "attractor_on": [True], "drift_total": [0.0], "drift_per_month": [0.0]}
        ),
    )
    monkeypatch.setattr(
        "beam_abm.abm.diagnostics.credibility.run_policy_ranking_stability",
        lambda *args, **kwargs: {
            "top_scenario_stable": True,
            "full_ranking_stable": True,
            "pairwise_rank_agreement": 1.0,
            "horizons": [24, 52],
            "ranking_by_horizon": {},
        },
    )
    monkeypatch.setattr(
        "beam_abm.abm.diagnostics.credibility.analyze_decision_tree",
        lambda *args, **kwargs: {
            "buckets": [],
            "fix_recommendations": [],
            "invariant_checks": {"policy_top_rank_stable": True, "policy_full_rank_stable": True},
        },
    )
    monkeypatch.setattr("beam_abm.abm.diagnostics.credibility.generate_report", lambda *args, **kwargs: "report")

    runtime = SimpleNamespace(cfg=SimpleNamespace(seed=42))
    result = run_credibility_suite(runtime=runtime, suite_output_dir=tmp_path, seeds=[42, 43], n_steps=24)

    assert result.passed is True
    for name in (
        "substep_timeseries.csv",
        "tick_summaries.csv",
        "ablation_2x2.csv",
        "pulse_response.csv",
        "long_horizon_stationarity.csv",
        "policy_ranking.json",
        "findings.json",
        "report.md",
        "report.json",
    ):
        assert (tmp_path / name).exists()


def test_core_report_includes_gate_payloads(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    runtime = SimpleNamespace(
        survey_df=pl.DataFrame({"country": ["IT"]}),
        models={},
        levers_cfg=SimpleNamespace(),
        anchor_system=SimpleNamespace(),
    )

    monkeypatch.setattr("beam_abm.abm.diagnostics.core.build_population", lambda **kwargs: pl.DataFrame({"x": [1.0]}))
    monkeypatch.setattr("beam_abm.abm.diagnostics.core.get_scenario_library", lambda **kwargs: {"baseline": object()})
    monkeypatch.setattr(
        "beam_abm.abm.diagnostics.core.iter_scenario_regime_runs", lambda scenarios: (("baseline", "perturbation"),)
    )
    monkeypatch.setattr("beam_abm.abm.diagnostics.core.run_simulation", lambda *args, **kwargs: [SimpleNamespace()])

    gate = GateResult(gate_name="g")
    gate.add_check("c", True)
    gate.evaluate()

    class _GateSuite:
        def __init__(self) -> None:
            self.gates = (gate,)
            self.passed = True

    monkeypatch.setattr("beam_abm.abm.diagnostics.core.run_all_gates", lambda *args, **kwargs: _GateSuite())
    monkeypatch.setattr(
        "beam_abm.abm.diagnostics.core.run_performance_benchmarks", lambda *args, **kwargs: PerformanceReport()
    )
    monkeypatch.setattr(
        "beam_abm.abm.diagnostics.core.build_non_degeneracy_report",
        lambda *args, **kwargs: NonDegeneracyReport(
            status="passed", n_domains=0, n_failed_domains=0, n_incomplete_domains=0, domains=[]
        ),
    )

    result = run_core_suite(
        runtime=runtime,
        suite_output_dir=tmp_path,
        n_agents=100,
        n_steps=6,
        n_reps=1,
        seeds=[42],
    )

    assert result.passed is True
    payload = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert "gates" in payload
    assert isinstance(payload["gates"], list)
    assert payload["gates"][0]["gate_name"] == "g"
