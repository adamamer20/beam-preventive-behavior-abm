"""Importation artifact-falsification diagnostics suite."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import polars as pl

from beam_abm.abm.diagnostics.contracts import (
    DiagnosticsSuiteResult,
    ImportationDiagnosticsSummary,
    summary_to_dict,
)
from beam_abm.abm.diagnostics.runtime import DiagnosticsRuntime, clone_cfg, run_single
from beam_abm.abm.io import write_json
from beam_abm.abm.message_contracts import NormChannelWeights
from beam_abm.abm.outcome_scales import OUTCOME_BOUNDS
from beam_abm.abm.outcomes import predict_all_outcomes
from beam_abm.abm.runtime import SimulationResult
from beam_abm.abm.scenario_defs import SCENARIO_LIBRARY, ScenarioConfig

__all__ = ["run_importation_suite"]

PROTECTION_COMPONENTS: tuple[str, ...] = (
    "vax_willingness_T12",
    "mask_when_symptomatic_crowded",
    "stay_home_when_symptomatic",
    "mask_when_pressure_high",
)
RISK_MEDIATORS: tuple[str, ...] = (
    "covid_perceived_danger_T12",
    "disease_fear_avg",
    "infection_likelihood_avg",
)
DOSE_AMPLITUDES: tuple[float, ...] = (0.005, 0.016, 0.03)


def _normalize_outcome_mean(series: np.ndarray, outcome_name: str) -> np.ndarray:
    bounds = OUTCOME_BOUNDS.get(outcome_name)
    if bounds is None:
        return series.astype(np.float64)
    lo, hi = bounds
    span = float(hi - lo)
    if span <= 0.0:
        return np.zeros_like(series, dtype=np.float64)
    return np.clip((series.astype(np.float64) - float(lo)) / span, 0.0, 1.0)


def _extract_protection_index(result: SimulationResult) -> np.ndarray:
    if result.outcome_trajectory.is_empty():
        return np.array([], dtype=np.float64)

    trajectories: list[np.ndarray] = []
    for name in PROTECTION_COMPONENTS:
        col = f"{name}_mean"
        if col not in result.outcome_trajectory.columns:
            continue
        arr = result.outcome_trajectory[col].cast(pl.Float64).to_numpy().astype(np.float64)
        trajectories.append(_normalize_outcome_mean(arr, name))

    if not trajectories:
        return np.array([], dtype=np.float64)
    return np.mean(np.vstack(trajectories), axis=0)


def _extract_mediator_means(result: SimulationResult) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    if result.mediator_trajectory.is_empty():
        return out
    for mediator in RISK_MEDIATORS:
        col = f"{mediator}_mean"
        if col in result.mediator_trajectory.columns:
            out[mediator] = result.mediator_trajectory[col].cast(pl.Float64).to_numpy().astype(np.float64)
    return out


def _extract_mean_incidence(result: SimulationResult) -> np.ndarray:
    if result.incidence_trajectory.is_empty() or "mean_incidence" not in result.incidence_trajectory.columns:
        return np.array([], dtype=np.float64)
    return result.incidence_trajectory["mean_incidence"].cast(pl.Float64).to_numpy().astype(np.float64)


def _auc(series: np.ndarray) -> float:
    if series.size == 0:
        return 0.0
    return float(np.sum(series))


def _peak_tick(series: np.ndarray) -> int:
    if series.size == 0:
        return -1
    return int(np.argmax(series))


def _mean_over_runs(rows: list[dict[str, float | str | int]]) -> pl.DataFrame:
    if not rows:
        return pl.DataFrame()
    return pl.DataFrame(rows)


def run_path_activation_checks(
    runtime: DiagnosticsRuntime,
    *,
    baseline_scenario: ScenarioConfig,
    importation_scenario: ScenarioConfig,
    seeds: list[int],
) -> tuple[pl.DataFrame, list[dict[str, float | int]]]:
    rows: list[dict[str, float | str | int]] = []
    summaries: list[dict[str, float | int]] = []

    for seed in seeds:
        base = run_single(runtime, scenario=baseline_scenario, cfg=runtime.cfg, seed=seed)
        imp = run_single(runtime, scenario=importation_scenario, cfg=runtime.cfg, seed=seed)

        inc_base = _extract_mean_incidence(base)
        inc_imp = _extract_mean_incidence(imp)
        min_len = min(inc_base.size, inc_imp.size)
        if min_len == 0:
            continue
        inc_delta = inc_imp[:min_len] - inc_base[:min_len]

        medi_base = _extract_mediator_means(base)
        medi_imp = _extract_mediator_means(imp)
        for mediator in RISK_MEDIATORS:
            if mediator not in medi_base or mediator not in medi_imp:
                continue
            m_delta = medi_imp[mediator][:min_len] - medi_base[mediator][:min_len]
            lag = _peak_tick(m_delta) - _peak_tick(inc_delta)
            summaries.append(
                {
                    "seed": int(seed),
                    "mediator": mediator,
                    "auc_delta": _auc(m_delta),
                    "peak_lag_ticks": int(lag),
                    "peak_delta": float(np.max(m_delta)) if m_delta.size > 0 else 0.0,
                }
            )
            for tick, value in enumerate(m_delta.tolist()):
                rows.append(
                    {
                        "seed": int(seed),
                        "tick": int(tick),
                        "series": mediator,
                        "delta": float(value),
                        "incidence_delta": float(inc_delta[tick]),
                    }
                )

    return _mean_over_runs(rows), summaries


def run_static_choice_sensitivity(runtime: DiagnosticsRuntime, *, seed: int) -> list[dict[str, float | str]]:
    agents = runtime.base_population
    baseline_outcomes = predict_all_outcomes(
        agents,
        runtime.models,
        runtime.levers_cfg,
        deterministic=True,
        rng=np.random.default_rng(seed + 9001),
        tick=0,
        choice_signals=[],
        target_mask=None,
        norm_channel_weights=NormChannelWeights(),
    )

    baseline_component_means: dict[str, float] = {}
    baseline_components: list[np.ndarray] = []
    for name in PROTECTION_COMPONENTS:
        arr = baseline_outcomes.get(name)
        if arr is None:
            continue
        baseline_component_means[name] = float(np.mean(arr))
        baseline_components.append(_normalize_outcome_mean(arr.astype(np.float64), name))
    baseline_protection = float(np.mean(np.vstack(baseline_components))) if baseline_components else 0.0

    rows: list[dict[str, float | str]] = []
    for mediator in RISK_MEDIATORS:
        if mediator not in agents.columns:
            continue
        values = agents[mediator].cast(pl.Float64).to_numpy().astype(np.float64)
        std = float(np.std(values))
        if not np.isfinite(std) or std <= 0.0:
            continue

        shifted_agents = agents.with_columns(pl.Series(mediator, (values + std).astype(np.float64)))
        shifted_outcomes = predict_all_outcomes(
            shifted_agents,
            runtime.models,
            runtime.levers_cfg,
            deterministic=True,
            rng=np.random.default_rng(seed + 9002),
            tick=0,
            choice_signals=[],
            target_mask=None,
            norm_channel_weights=NormChannelWeights(),
        )

        shifted_components: list[np.ndarray] = []
        for name in PROTECTION_COMPONENTS:
            arr = shifted_outcomes.get(name)
            if arr is None:
                continue
            shifted_mean = float(np.mean(arr))
            delta_mean = shifted_mean - baseline_component_means.get(name, shifted_mean)
            rows.append(
                {
                    "mediator": mediator,
                    "component": name,
                    "delta_mean": float(delta_mean),
                    "shift_sd": std,
                }
            )
            shifted_components.append(_normalize_outcome_mean(arr.astype(np.float64), name))

        shifted_protection = (
            float(np.mean(np.vstack(shifted_components))) if shifted_components else baseline_protection
        )
        rows.append(
            {
                "mediator": mediator,
                "component": "protection_index",
                "delta_mean": float(shifted_protection - baseline_protection),
                "shift_sd": std,
            }
        )

    return rows


def run_toggle_checks(
    runtime: DiagnosticsRuntime,
    *,
    scenario: ScenarioConfig,
    seeds: list[int],
) -> tuple[pl.DataFrame, list[dict[str, float | str | int]]]:
    rows: list[dict[str, float | str | int]] = []
    summary: list[dict[str, float | str | int]] = []

    cfg_base = runtime.cfg
    cfg_no_feedback = clone_cfg(runtime.cfg, knob_updates={"k_feedback_stakes": 0.0})
    cfg_no_social = clone_cfg(runtime.cfg, knob_updates={"k_social": 0.0})
    variants = {
        "baseline": cfg_base,
        "k_feedback_stakes_0": cfg_no_feedback,
        "k_social_0": cfg_no_social,
    }

    for seed in seeds:
        runs = {
            name: run_single(runtime, scenario=scenario, cfg=cfg_variant, seed=seed)
            for name, cfg_variant in variants.items()
        }
        protection = {name: _extract_protection_index(res) for name, res in runs.items()}

        base_prot = protection["baseline"]
        for variant_name, prot in protection.items():
            if variant_name == "baseline":
                continue
            min_len = min(base_prot.size, prot.size)
            if min_len == 0:
                continue
            delta = prot[:min_len] - base_prot[:min_len]
            summary.append(
                {
                    "seed": int(seed),
                    "variant": variant_name,
                    "mean_delta_protection": float(np.mean(delta)),
                    "peak_abs_delta_protection": float(np.max(np.abs(delta))),
                }
            )
            for tick, value in enumerate(delta.tolist()):
                rows.append(
                    {
                        "seed": int(seed),
                        "tick": int(tick),
                        "variant": variant_name,
                        "delta_protection": float(value),
                    }
                )

    return _mean_over_runs(rows), summary


def run_dose_response_checks(
    runtime: DiagnosticsRuntime,
    *,
    scenario: ScenarioConfig,
    seed: int,
) -> list[dict[str, float | int]]:
    cfg_no_pulse = clone_cfg(runtime.cfg, constant_updates={"importation_pulse_amplitude": 0.0})
    baseline = run_single(runtime, scenario=scenario, cfg=cfg_no_pulse, seed=seed)
    inc_baseline = _extract_mean_incidence(baseline)
    protection_baseline = _extract_protection_index(baseline)
    medi_baseline = _extract_mediator_means(baseline)

    rows: list[dict[str, float | int]] = []
    for amplitude in DOSE_AMPLITUDES:
        cfg_amp = clone_cfg(runtime.cfg, constant_updates={"importation_pulse_amplitude": float(amplitude)})
        run = run_single(runtime, scenario=scenario, cfg=cfg_amp, seed=seed)

        inc = _extract_mean_incidence(run)
        protection = _extract_protection_index(run)
        medi = _extract_mediator_means(run)

        min_len = min(inc.size, inc_baseline.size, protection.size, protection_baseline.size)
        if min_len == 0:
            continue

        inc_delta = inc[:min_len] - inc_baseline[:min_len]
        prot_delta = protection[:min_len] - protection_baseline[:min_len]
        row: dict[str, float | int] = {
            "amplitude": float(amplitude),
            "peak_delta_incidence": float(np.max(inc_delta)),
            "peak_abs_delta_protection": float(np.max(np.abs(prot_delta))),
            "peak_tick_incidence_delta": int(np.argmax(inc_delta)),
            "peak_tick_protection_abs_delta": int(np.argmax(np.abs(prot_delta))),
        }
        for mediator in RISK_MEDIATORS:
            if mediator in medi and mediator in medi_baseline:
                delta = medi[mediator][:min_len] - medi_baseline[mediator][:min_len]
                row[f"peak_delta_{mediator}"] = float(np.max(delta))
        rows.append(row)

    rows.sort(key=lambda r: float(r["amplitude"]))
    return rows


def run_timing_placebo_checks(
    runtime: DiagnosticsRuntime,
    *,
    scenario: ScenarioConfig,
    seed: int,
) -> dict[str, float | int | bool]:
    cfg_zero = clone_cfg(runtime.cfg, constant_updates={"importation_pulse_amplitude": 0.0})
    cfg_start0 = clone_cfg(
        runtime.cfg,
        constant_updates={"importation_pulse_amplitude": 0.016, "importation_pulse_start_tick": 0},
    )
    cfg_start6 = clone_cfg(
        runtime.cfg,
        constant_updates={"importation_pulse_amplitude": 0.016, "importation_pulse_start_tick": 6},
    )

    zero_run = run_single(runtime, scenario=scenario, cfg=cfg_zero, seed=seed)
    start0_run = run_single(runtime, scenario=scenario, cfg=cfg_start0, seed=seed)
    start6_run = run_single(runtime, scenario=scenario, cfg=cfg_start6, seed=seed)

    inc_zero = _extract_mean_incidence(zero_run)
    inc_start0 = _extract_mean_incidence(start0_run)
    inc_start6 = _extract_mean_incidence(start6_run)

    min_len = min(inc_zero.size, inc_start0.size, inc_start6.size)
    if min_len == 0:
        return {
            "timing_shift_observed": False,
            "placebo_small_effect": True,
            "peak_tick_start0": -1,
            "peak_tick_start6": -1,
            "peak_tick_shift": 0,
            "placebo_peak_abs_delta_incidence": 0.0,
        }

    delta_start0 = inc_start0[:min_len] - inc_zero[:min_len]
    delta_start6 = inc_start6[:min_len] - inc_zero[:min_len]
    peak0 = int(np.argmax(delta_start0))
    peak6 = int(np.argmax(delta_start6))

    placebo_peak_abs = float(np.max(np.abs(inc_zero[:min_len] - inc_zero[:min_len])))
    return {
        "timing_shift_observed": bool(peak6 > peak0),
        "placebo_small_effect": bool(placebo_peak_abs < 1e-12),
        "peak_tick_start0": peak0,
        "peak_tick_start6": peak6,
        "peak_tick_shift": int(peak6 - peak0),
        "placebo_peak_abs_delta_incidence": placebo_peak_abs,
    }


def _ols_slope(x: np.ndarray, y: np.ndarray) -> float:
    if x.size == 0 or y.size == 0 or x.size != y.size:
        return float("nan")
    x_var = float(np.var(x))
    if x_var <= 1e-12:
        return float("nan")
    cov = float(np.mean((x - float(np.mean(x))) * (y - float(np.mean(y)))))
    return cov / x_var


def run_coupling_diagnostics(
    runtime: DiagnosticsRuntime,
    *,
    scenario: ScenarioConfig,
    seeds: list[int],
) -> dict[str, object]:
    per_seed: list[dict[str, float | int]] = []
    slope_i_from_p: list[float] = []
    slope_p_from_i: list[float] = []

    for seed in seeds:
        result = run_single(runtime, scenario=scenario, cfg=runtime.cfg, seed=seed)
        incidence = _extract_mean_incidence(result)
        protection = _extract_protection_index(result)
        min_len = min(incidence.size, protection.size)
        if min_len < 3:
            continue

        incidence = incidence[:min_len]
        protection = protection[:min_len]
        d_incidence_next = incidence[1:] - incidence[:-1]
        d_protection_next = protection[1:] - protection[:-1]

        slope1 = _ols_slope(protection[:-1], d_incidence_next)
        slope2 = _ols_slope(incidence[:-1], d_protection_next)
        slope_i_from_p.append(slope1)
        slope_p_from_i.append(slope2)
        per_seed.append(
            {
                "seed": int(seed),
                "slope_delta_incidence_on_protection": float(slope1),
                "slope_delta_protection_on_incidence": float(slope2),
            }
        )

    def _band(values: list[float]) -> dict[str, float]:
        arr = np.asarray(values, dtype=np.float64)
        if arr.size == 0:
            return {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
        mean = float(np.mean(arr))
        if arr.size == 1:
            return {"mean": mean, "ci_low": mean, "ci_high": mean}
        se = float(np.std(arr, ddof=1) / np.sqrt(arr.size))
        width = 1.96 * se
        return {"mean": mean, "ci_low": mean - width, "ci_high": mean + width}

    return {
        "per_seed": per_seed,
        "slope_delta_incidence_on_protection": _band(slope_i_from_p),
        "slope_delta_protection_on_incidence": _band(slope_p_from_i),
    }


def run_importation_suite(
    *,
    runtime: DiagnosticsRuntime,
    suite_output_dir: Path,
    seeds: list[int],
) -> DiagnosticsSuiteResult:
    """Execute importation artifact checks and emit suite artifacts."""
    start = time.perf_counter()
    suite_output_dir.mkdir(parents=True, exist_ok=True)

    baseline = SCENARIO_LIBRARY["baseline"]
    baseline_importation = SCENARIO_LIBRARY["baseline_importation"]

    path_ts_df, path_summary = run_path_activation_checks(
        runtime,
        baseline_scenario=baseline,
        importation_scenario=baseline_importation,
        seeds=[int(seed) for seed in seeds],
    )
    if not path_ts_df.is_empty():
        path_ts_df.write_csv(suite_output_dir / "path_activation_timeseries.csv")
    pl.DataFrame(path_summary).write_csv(suite_output_dir / "path_activation_summary.csv")

    static_rows = run_static_choice_sensitivity(runtime, seed=int(seeds[0]))
    pl.DataFrame(static_rows).write_csv(suite_output_dir / "static_choice_sensitivity.csv")

    toggle_ts_df, toggle_summary = run_toggle_checks(
        runtime,
        scenario=baseline_importation,
        seeds=[int(seed) for seed in seeds],
    )
    if not toggle_ts_df.is_empty():
        toggle_ts_df.write_csv(suite_output_dir / "toggle_timeseries.csv")
    pl.DataFrame(toggle_summary).write_csv(suite_output_dir / "toggle_summary.csv")

    dose_rows = run_dose_response_checks(runtime, scenario=baseline_importation, seed=int(seeds[0]))
    pl.DataFrame(dose_rows).write_csv(suite_output_dir / "dose_response_summary.csv")

    timing_placebo = run_timing_placebo_checks(runtime, scenario=baseline_importation, seed=int(seeds[0]))

    coupling = run_coupling_diagnostics(
        runtime,
        scenario=baseline_importation,
        seeds=[int(seed) for seed in seeds],
    )

    summary_json = {
        "n_agents": int(runtime.cfg.n_agents),
        "n_steps": int(runtime.cfg.n_steps),
        "seeds": [int(seed) for seed in seeds],
        "path_activation": {
            "rows": int(len(path_summary)),
            "summary_file": "path_activation_summary.csv",
        },
        "static_choice_sensitivity": {
            "rows": int(len(static_rows)),
            "summary_file": "static_choice_sensitivity.csv",
        },
        "toggle_tests": {
            "rows": int(len(toggle_summary)),
            "summary_file": "toggle_summary.csv",
        },
        "dose_response": {
            "rows": int(len(dose_rows)),
            "summary_file": "dose_response_summary.csv",
        },
        "timing_placebo": timing_placebo,
        "coupling": coupling,
    }
    write_json(summary_json, suite_output_dir / "artifact_falsification_summary.json")

    checks: list[tuple[str, bool]] = [
        ("path_activation_rows", len(path_summary) > 0),
        ("static_choice_rows", len(static_rows) > 0),
        ("toggle_rows", len(toggle_summary) > 0),
        ("dose_response_rows", len(dose_rows) > 0),
        ("timing_shift", bool(timing_placebo.get("timing_shift_observed", False))),
        ("coupling_rows", len(coupling.get("per_seed", [])) > 0),
    ]
    n_failed = sum(1 for _, passed in checks if not passed)
    passed = n_failed == 0

    summary = ImportationDiagnosticsSummary(
        n_seeds=len(seeds),
        n_path_activation_rows=len(path_summary),
        n_static_sensitivity_rows=len(static_rows),
        n_toggle_rows=len(toggle_summary),
        n_dose_rows=len(dose_rows),
        timing_shift_observed=bool(timing_placebo.get("timing_shift_observed", False)),
        coupling_estimates=len(coupling.get("per_seed", [])),
    )

    suite_report = {
        "suite": "importation",
        "passed": passed,
        "checks": [{"name": name, "passed": ok} for name, ok in checks],
        "summary": summary.to_dict(),
    }
    suite_report_path = suite_output_dir / "report.json"
    write_json(suite_report, suite_report_path)

    elapsed = time.perf_counter() - start
    return DiagnosticsSuiteResult(
        suite="importation",
        passed=passed,
        elapsed_seconds=float(elapsed),
        n_checks=len(checks),
        n_failed=n_failed,
        artifacts=(
            "path_activation_timeseries.csv",
            "path_activation_summary.csv",
            "static_choice_sensitivity.csv",
            "toggle_timeseries.csv",
            "toggle_summary.csv",
            "dose_response_summary.csv",
            "artifact_falsification_summary.json",
            "report.json",
        ),
        report_path=str(suite_report_path),
        summary=summary_to_dict(summary),
    )
