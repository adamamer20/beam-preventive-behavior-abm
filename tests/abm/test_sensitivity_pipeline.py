"""Unit tests for ABM sensitivity-pipeline utilities."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

import beam_abm.abm.sensitivity_pipeline as sensitivity_pipeline
from beam_abm.abm.sensitivity_pipeline import (
    SensitivityPipelineConfig,
    build_lhc_design,
    build_morris_design,
    compute_lhc_spearman_directionality,
    compute_morris_indices,
    compute_sobol_indices_from_surrogate,
    compute_surrogate_ice_slope_directionality,
    select_shortlist,
)


class _LinearModel:
    """Simple linear predictor for Sobol estimator tests."""

    def predict(self, x: np.ndarray) -> np.ndarray:
        return 2.0 * x[:, 0] + 0.1 * x[:, 1]


def test_build_morris_design_has_expected_shape() -> None:
    """Morris design should return trajectories with p+1 points each."""
    bounds = {
        "k_social": (0.0, 1.0),
        "k_incidence": (0.0, 0.5),
    }
    design = build_morris_design(
        parameter_bounds=bounds,
        n_trajectories=3,
        n_levels=6,
        seed=123,
    )

    assert design.height == 3 * (len(bounds) + 1)
    assert {"point_id", "trajectory_id", "step_index", "changed_param", "step_size"}.issubset(set(design.columns))


def test_compute_morris_indices_returns_per_scenario_and_aggregate() -> None:
    """Morris index computation should produce both granular and aggregate views."""
    design = pl.DataFrame(
        {
            "point_id": [0, 1, 2],
            "trajectory_id": [0, 0, 0],
            "step_index": [0, 1, 2],
            "prev_point_id": [None, 0, 1],
            "changed_param": [None, "k_social", "k_incidence"],
            "step_size": [None, 0.5, 0.2],
        }
    )

    delta = pl.DataFrame(
        {
            "point_id": [0, 1, 2, 0, 1, 2],
            "scenario": [
                "misinfo_shock",
                "misinfo_shock",
                "misinfo_shock",
                "norm_campaign",
                "norm_campaign",
                "norm_campaign",
            ],
            "output": ["vax_willingness_T12_mean"] * 6,
            "delta": [0.0, 0.2, 0.3, 0.0, 0.1, 0.12],
        }
    )

    per_scenario, aggregate = compute_morris_indices(design_df=design, delta_df=delta)

    assert per_scenario.height > 0
    assert aggregate.height == 2
    assert {"parameter", "mu_star", "sigma"}.issubset(set(per_scenario.columns))
    assert {"parameter", "mu_star_max", "sigma_max"}.issubset(set(aggregate.columns))


def test_select_shortlist_respects_target_and_sigma_flags() -> None:
    """Shortlist should include top importance and high-sigma parameters."""
    aggregate = pl.DataFrame(
        {
            "parameter": ["p1", "p2", "p3", "p4", "p5", "p6"],
            "mu_star_median": [0.9, 0.8, 0.6, 0.3, 0.2, 0.1],
            "mu_star_max": [1.0, 0.85, 0.62, 0.35, 0.22, 0.11],
            "sigma_median": [0.2, 0.9, 0.1, 0.05, 0.03, 0.02],
            "sigma_max": [0.3, 1.2, 0.2, 0.1, 0.09, 0.05],
            "n_effects_total": [20, 20, 20, 20, 20, 20],
        }
    )

    shortlist = select_shortlist(
        aggregate_indices=aggregate,
        target_size=3,
        max_size=4,
        sigma_quantile=0.8,
    )

    selected = shortlist.select("parameter").to_series().to_list()
    assert len(selected) <= 4
    assert "p1" in selected
    assert "p2" in selected


def test_compute_sobol_indices_from_surrogate_outputs_expected_columns() -> None:
    """Sobol estimator should output S_i and S_Ti for each parameter."""
    model = _LinearModel()
    bounds = {
        "x1": (0.0, 1.0),
        "x2": (0.0, 1.0),
    }

    sobol = compute_sobol_indices_from_surrogate(
        model=model,
        parameter_bounds=bounds,
        base_samples=256,
        seed=11,
    )

    assert sobol.height == 2
    assert {"parameter", "S_i", "S_Ti", "interaction_gap", "sobol_n"}.issubset(set(sobol.columns))
    assert float(sobol.select(pl.col("S_i").max()).item()) <= 1.0
    assert float(sobol.select(pl.col("S_Ti").max()).item()) <= 1.0


def test_build_morris_design_respects_incidence_net_growth_constraint() -> None:
    """Morris samples should satisfy k_incidence - incidence_gamma bounds."""
    bounds = {
        "k_incidence": (0.4, 3.8),
        "incidence_gamma": (0.25, 0.85),
        "k_social": (0.0, 1.0),
    }
    design = build_morris_design(
        parameter_bounds=bounds,
        n_trajectories=4,
        n_levels=6,
        seed=7,
    )

    net = design.select((pl.col("k_incidence") - pl.col("incidence_gamma")).alias("net_growth"))
    assert float(net.select(pl.col("net_growth").min()).item()) >= -0.8
    assert float(net.select(pl.col("net_growth").max()).item()) <= 3.3


def test_build_lhc_design_respects_incidence_net_growth_constraint() -> None:
    """LHC samples should satisfy k_incidence - incidence_gamma bounds."""
    bounds = {
        "k_incidence": (0.4, 3.8),
        "incidence_gamma": (0.25, 0.85),
        "k_social": (0.0, 1.0),
    }
    design = build_lhc_design(
        parameter_bounds=bounds,
        n_samples=200,
        seed=17,
    )

    net = design.select((pl.col("k_incidence") - pl.col("incidence_gamma")).alias("net_growth"))
    assert float(net.select(pl.col("net_growth").min()).item()) >= -0.8
    assert float(net.select(pl.col("net_growth").max()).item()) <= 3.3


def test_compute_lhc_spearman_directionality_signs() -> None:
    """Spearman diagnostics should recover monotonic direction signs."""
    lhc_delta = pl.DataFrame(
        {
            "scenario": ["s1"] * 6,
            "output": ["y1"] * 6,
            "delta": [1, 2, 3, 4, 5, 6],
            "p_up": [1, 2, 3, 4, 5, 6],
            "p_down": [6, 5, 4, 3, 2, 1],
            "p_const": [7] * 6,
        }
    )

    out = compute_lhc_spearman_directionality(
        lhc_delta=lhc_delta,
        parameter_names=("p_up", "p_down", "p_const"),
    )

    up = out.filter(pl.col("parameter") == "p_up").to_dicts()[0]
    down = out.filter(pl.col("parameter") == "p_down").to_dicts()[0]
    const = out.filter(pl.col("parameter") == "p_const").to_dicts()[0]

    assert up["direction_sign"] == "positive"
    assert down["direction_sign"] == "negative"
    assert const["n_unique_param"] == 1
    assert np.isnan(const["spearman_rho"])


def test_compute_surrogate_ice_slope_directionality_signs() -> None:
    """Surrogate ICE-slope diagnostics should capture mean slope direction."""

    class _SignedLinearModel:
        def predict(self, x: np.ndarray) -> np.ndarray:
            return 2.0 * x[:, 0] - 3.0 * x[:, 1]

    feature_df = pl.DataFrame(
        {
            "point_id": [0, 1, 2, 3],
            "x1": [0.1, 0.4, 0.6, 0.9],
            "x2": [0.2, 0.3, 0.5, 0.7],
        }
    )
    out = compute_surrogate_ice_slope_directionality(
        retained_models={("scenario_a", "outcome_b"): _SignedLinearModel()},
        feature_df=feature_df,
        parameter_bounds={"x1": (0.0, 1.0), "x2": (0.0, 1.0)},
    )

    x1 = out.filter(pl.col("parameter") == "x1").to_dicts()[0]
    x2 = out.filter(pl.col("parameter") == "x2").to_dicts()[0]
    assert x1["direction_sign"] == "positive"
    assert x2["direction_sign"] == "negative"
    assert x1["n_points"] == 4


def test_resolve_sensitivity_workers_auto_is_ram_aware(monkeypatch: pytest.MonkeyPatch) -> None:
    """Auto worker policy should cap by estimated available memory."""
    monkeypatch.setattr(sensitivity_pipeline.os, "cpu_count", lambda: 32)
    monkeypatch.setattr(sensitivity_pipeline, "_available_memory_gib", lambda: 10.0)
    monkeypatch.setattr(sensitivity_pipeline, "_current_process_rss_gib", lambda: 0.0)
    monkeypatch.setenv("BEAM_ABM_SA_WORKER_RESERVE_GIB", "6")
    monkeypatch.setenv("BEAM_ABM_SA_WORKER_MEM_GIB", "3")
    monkeypatch.setenv("BEAM_ABM_SA_MAX_AUTO_WORKERS", "64")

    workers = sensitivity_pipeline._resolve_sensitivity_workers(requested_workers=0, n_jobs=100)
    assert workers == 1


def test_resolve_sensitivity_workers_accounts_for_parent_rss(monkeypatch: pytest.MonkeyPatch) -> None:
    """Auto worker policy should get more conservative for large parent RSS."""
    monkeypatch.setattr(sensitivity_pipeline.os, "cpu_count", lambda: 32)
    monkeypatch.setattr(sensitivity_pipeline, "_available_memory_gib", lambda: 26.0)
    monkeypatch.setattr(sensitivity_pipeline, "_current_process_rss_gib", lambda: 8.0)
    monkeypatch.setenv("BEAM_ABM_SA_WORKER_RESERVE_GIB", "6")
    monkeypatch.setenv("BEAM_ABM_SA_WORKER_MEM_GIB", "3")
    monkeypatch.setenv("BEAM_ABM_SA_PARENT_RSS_PER_WORKER_FACTOR", "0.35")
    monkeypatch.setenv("BEAM_ABM_SA_PARENT_RSS_RESERVE_FACTOR", "0.25")
    monkeypatch.setenv("BEAM_ABM_SA_MAX_AUTO_WORKERS", "64")

    workers = sensitivity_pipeline._resolve_sensitivity_workers(requested_workers=0, n_jobs=100)
    assert workers == 3


def test_build_delta_table_handles_effect_regime_suffixes() -> None:
    """Delta construction should resolve suffixed baseline references."""
    cfg = SensitivityPipelineConfig(outputs=("vax_willingness_T12_mean",))
    scenarios = [
        "baseline__perturbation",
        "baseline_importation__perturbation",
        "access_intervention__perturbation",
    ]
    data: dict[str, list[float | int | str]] = {
        "point_id": [0, 0, 0],
        "scenario": scenarios,
        "vax_willingness_T12_mean": [1.0, 2.0, 1.5],
        "vax_willingness_T12_mean__auc": [10.0, 20.0, 14.0],
    }
    for name in sensitivity_pipeline.FULL_PARAMETER_SWEEP_VALUES:
        if name in sensitivity_pipeline.INTEGER_PARAMETERS:
            data[name] = [0, 0, 0]
        else:
            data[name] = [0.0, 0.0, 0.0]
    scenario_means = pl.DataFrame(data)
    delta_df = sensitivity_pipeline._build_delta_table(
        scenario_means=scenario_means,
        scenario_targets=("access_intervention__perturbation",),
        outputs=cfg.outputs,
        config=cfg,
    )
    assert delta_df.height == 1
    row = delta_df.to_dicts()[0]
    expected_reference = sensitivity_pipeline._reference_baseline("access_intervention__perturbation")
    assert row["reference_scenario"] == expected_reference
    assert row["delta"] == pytest.approx(-6.0)
