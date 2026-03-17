"""Focused regression tests for credibility diagnostics helpers."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest

import beam_abm.abm.diagnostics.credibility as credibility
from beam_abm.abm.config import SimulationConfig
from beam_abm.abm.data_contracts import LeversConfig
from beam_abm.abm.diagnostics.credibility import (
    SharedData,
    build_model,
    run_policy_ranking_stability,
    summarize_pulse_band,
)
from beam_abm.abm.scenario_defs import ScenarioConfig


def test_build_model_uses_isolated_config_copy(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mutating model config must not mutate shared diagnostics config."""
    cfg = SimulationConfig()
    shared = SharedData(
        cfg=cfg,
        models={},
        levers_cfg=LeversConfig(outcomes={}),
        agent_df=pl.DataFrame({"x": [1.0]}),
        network=None,
    )
    scenario = ScenarioConfig(
        name="diagnostic",
        family="diagnostic",
        communication_enabled=False,
        peer_pull_enabled=False,
        observed_norms_enabled=False,
        enable_endogenous_loop=False,
        incidence_mode="none",
    )

    def _fake_from_preloaded(
        cls: type[object],
        *,
        cfg: SimulationConfig,
        scenario: ScenarioConfig,
        models: dict[str, object],
        levers_cfg: LeversConfig,
        agent_df: pl.DataFrame,
        network: object | None,
        belief_kernel: object | None = None,
        seed: int | None = None,
    ) -> object:
        _ = (cls, scenario, models, levers_cfg, agent_df, network, belief_kernel, seed)
        return SimpleNamespace(cfg=cfg, _state_ref_attractor=object())

    monkeypatch.setattr(
        credibility.VaccineBehaviourModel,
        "from_preloaded",
        classmethod(_fake_from_preloaded),
    )

    model = build_model(shared, scenario, attractor_on=True, seed=123)
    assert model.cfg is not shared.cfg
    assert int(model.cfg.seed) == 123

    model.cfg.constants.importation_pulse_amplitude = 0.123
    assert shared.cfg.constants.importation_pulse_amplitude != pytest.approx(0.123)


def test_summarize_pulse_band_uses_incremental_std() -> None:
    """Pulse mild-band std should be computed on pulse-baseline effect."""
    baseline = np.linspace(0.0, 0.18, 10)
    pulse = baseline + 0.02
    records: list[dict[str, float | int | str]] = []
    for tick, (b, p) in enumerate(zip(baseline, pulse, strict=True)):
        records.append({"tick": tick, "label": "baseline_import", "mean_incidence": float(b)})
        records.append({"tick": tick, "label": "pulse_import", "mean_incidence": float(p)})

    stats = summarize_pulse_band(pl.DataFrame(records))

    assert stats["pulse_incidence_std"] > 0.05
    assert stats["pulse_delta_std"] == pytest.approx(0.0, abs=1e-12)
    assert stats["pulse_mild_band_std"] is True


def test_policy_ranking_stability_treats_near_ties_as_stable(monkeypatch: pytest.MonkeyPatch) -> None:
    """Top-rank stability should pass when both horizons are top-tied."""
    scenarios = {
        "s1": ScenarioConfig(name="s1", family="diagnostic", incidence_mode="none"),
        "s2": ScenarioConfig(name="s2", family="diagnostic", incidence_mode="none"),
        "s3": ScenarioConfig(name="s3", family="diagnostic", incidence_mode="none"),
    }
    score_by_horizon = {
        24: {"s1": 1.0000, "s2": 0.9995, "s3": 0.8000},
        52: {"s1": 1.0008, "s2": 1.0012, "s3": 0.8000},
    }

    class _FakeModel:
        def __init__(self, scenario_name: str) -> None:
            self._scenario_name = scenario_name
            self._tick = 0
            self.latest_outcomes = {"vax_willingness_T12": np.array([0.0], dtype=np.float64)}

        def step(self) -> None:
            self._tick += 1
            score = score_by_horizon.get(self._tick, score_by_horizon[52])[self._scenario_name]
            self.latest_outcomes = {"vax_willingness_T12": np.array([score], dtype=np.float64)}

    monkeypatch.setattr(credibility, "SCENARIO_LIBRARY", scenarios)
    monkeypatch.setattr(
        credibility,
        "build_model",
        lambda _shared, scenario, **kwargs: _FakeModel(scenario.name),
    )

    shared = SimpleNamespace(cfg=SimpleNamespace(n_steps=24))
    result = run_policy_ranking_stability(
        shared,
        horizons=(24, 52),
        scenario_names=("s1", "s2", "s3"),
        tie_tolerance=0.002,
    )

    assert set(result["top_tied_short"]) == {"s1", "s2"}
    assert set(result["top_tied_long"]) == {"s1", "s2"}
    assert result["top_scenario_stable"] is True
    assert result["full_ranking_stable"] is True
