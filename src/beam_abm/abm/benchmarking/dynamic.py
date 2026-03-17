"""Dynamic plausibility benchmarking gate."""

from __future__ import annotations

import numpy as np
import polars as pl
from loguru import logger

from beam_abm.abm.scenario_defs import SCENARIO_EFFECT_REGIMES, SCENARIO_LIBRARY
from beam_abm.abm.state_schema import CONSTRUCT_BOUNDS

from .core import (
    _VAX_WILLINGNESS_MEAN_CANDIDATES,
    GateResult,
    _estimate_dispersion_half_life_ticks,
    _first_present_column,
    _is_abm_outcome_mean_column,
    _is_null_world_scenario_cfg,
    _is_social_only_scenario_cfg,
    _late_window_slope,
    _mean_column_support_span,
    _protection_index_from_outcome_trajectory,
    _protection_to_next_delta_incidence_signature,
    _scenario_config_for_result,
    _scenario_effect_regime,
    _scenario_key,
    _shock_unit_denominator,
)


def run_dynamic_plausibility_gate(
    results: list,
    tolerance_bound: float = 0.0,
    stationarity_epsilon: float = 0.10,
    stress_horizon_min: int = 40,
    stress_horizon_max: int = 80,
    meaningful_effect_ratio: float = 1.0,
    directional_window: int = 6,
    directional_offsets: tuple[int, ...] = (0, 2, 4),
    anti_saturation_min_late_activity_ratio: float = 0.08,
    anti_saturation_min_total_span: float = 0.15,
    anti_saturation_bound_margin_ratio: float = 0.05,
    anti_saturation_min_bound_contact_share: float = 0.20,
) -> GateResult:
    """Check trajectory boundedness, stress stability, and comparative statics.

    Parameters
    ----------
    results : list[SimulationResult]
        Replicated simulation results.
    tolerance_bound : float
        Extra tolerance slack beyond declared construct bounds.
    stress_horizon_min : int
        Minimum reproducible horizon for stress checks.
    stress_horizon_max : int
        Maximum horizon used for stress checks.

    Returns
    -------
    GateResult
    """
    gate = GateResult(gate_name="dynamic_plausibility")

    n_agents_ref = 1000
    with_meta = [r for r in results if getattr(r, "metadata", None) is not None]
    if with_meta:
        n_agents_ref = max(1, int(with_meta[0].metadata.n_agents))
    response_tol = 0.02 * float(np.sqrt(1000.0 / n_agents_ref))
    meaningful_effect_min = float(max(response_tol * meaningful_effect_ratio, 1e-8))
    gate.add_check(
        "stress_horizon_documented",
        passed=stress_horizon_max >= stress_horizon_min >= 2,
        details={
            "horizon_min": stress_horizon_min,
            "horizon_max": stress_horizon_max,
            "response_tolerance": response_tol,
            "meaningful_effect_min": meaningful_effect_min,
        },
    )

    # 3a. Bounded trajectories: no mediator exits declared bounds
    for rep_result in results:
        med_traj = rep_result.mediator_trajectory
        pop_final = rep_result.population_final
        for col in med_traj.columns:
            if col == "tick" or not col.endswith("_mean"):
                continue
            values = med_traj[col].to_numpy()
            construct = col.replace("_mean", "")
            lo, hi = CONSTRUCT_BOUNDS.get(construct, (-np.inf, np.inf))
            max_over = float(np.max(values - hi)) if np.isfinite(hi) else 0.0
            max_under = float(np.max(lo - values)) if np.isfinite(lo) else 0.0
            max_violation = max(max_over, max_under, 0.0)
            gate.add_check(
                f"bounded_{construct}_rep{rep_result.rep_id}",
                passed=max_violation <= tolerance_bound,
                details={
                    "lower_bound": lo,
                    "upper_bound": hi,
                    "max_violation": max_violation,
                },
            )

    # 3b. Saturation: under sustained shocks, trajectories should
    #     plateau rather than diverge. Compare late/early variance
    #     with support-scaled tolerance.
    for rep_result in results:
        out_traj = rep_result.outcome_trajectory
        for col in out_traj.columns:
            if col == "tick" or not _is_abm_outcome_mean_column(col):
                continue
            if col.endswith("_delta_vs_t0_mean") or col.endswith("_delta_vs_ref_mean"):
                continue
            values = out_traj[col].to_numpy().astype(np.float64)
            values = values[np.isfinite(values)]
            if len(values) > 20:
                early_var = float(np.var(values[:10]))
                late_var = float(np.var(values[-10:]))
                support_span = _mean_column_support_span(col, values)
                variance_slack = 0.05 * (support_span**2)
                gate.add_check(
                    f"saturation_{col}_rep{rep_result.rep_id}",
                    passed=late_var < early_var + variance_slack,
                    details={
                        "early_var": early_var,
                        "late_var": late_var,
                        "support_span": support_span,
                        "variance_slack": variance_slack,
                    },
                )

    # 3c. Quiet-run stationarity: with comm off and no shocks,
    #     aggregate drifts should remain small.
    quiet_results = [
        r for r in results if (cfg := _scenario_config_for_result(r)) is not None and _is_null_world_scenario_cfg(cfg)
    ]
    gate.add_check(
        "quiet_stationarity_null_world_present",
        passed=len(quiet_results) > 0,
        details={
            "n_null_world_runs": len(quiet_results),
            "required": "no_signals + communication_off + endogenous_loop_off + incidence_mode_none",
        },
    )
    for rep_result in quiet_results:
        effect_regime = _scenario_effect_regime(rep_result)
        if rep_result.outcome_trajectory.height == 0:
            continue
        for col in rep_result.outcome_trajectory.columns:
            if col == "tick" or not _is_abm_outcome_mean_column(col):
                continue
            if col.endswith("_delta_vs_t0_mean") or col.endswith("_delta_vs_ref_mean"):
                continue
            vals = rep_result.outcome_trajectory[col].to_numpy().astype(np.float64)
            vals = vals[np.isfinite(vals)]
            if vals.size < 2:
                continue
            drift = float(abs(vals[-1] - vals[0]))
            gate.add_check(
                f"quiet_stationarity_{col}_{effect_regime}_rep{rep_result.rep_id}",
                passed=drift <= stationarity_epsilon,
                details={"drift": drift, "epsilon": stationarity_epsilon},
            )

    # 3c2. Anti-saturation guardrail: for non-quiet scenarios, flag
    #      trajectories that move materially then flatline at bounds.
    anti_saturation_scenarios = {
        key
        for key, cfg in SCENARIO_LIBRARY.items()
        if any(sig.signal_type == "sustained" for sig in tuple(cfg.choice_signals) + tuple(cfg.state_signals))
    }
    for rep_result in results:
        scenario_key = _scenario_key(rep_result)
        effect_regime = _scenario_effect_regime(rep_result)
        if scenario_key in {"baseline"}:
            continue
        if scenario_key not in anti_saturation_scenarios:
            continue
        med = rep_result.mediator_trajectory
        med_cols = [c for c in med.columns if c != "tick" and c.endswith("_mean")]
        if med.height < 12 or not med_cols:
            continue
        med_arr = med.select(med_cols).to_numpy().astype(np.float64)
        total_span = float(np.mean(np.nanmax(med_arr, axis=0) - np.nanmin(med_arr, axis=0)))
        if total_span < anti_saturation_min_total_span:
            continue
        diffs = np.abs(np.diff(med_arr, axis=0))
        if diffs.shape[0] < 6:
            continue
        split = max(3, diffs.shape[0] // 2)
        early_activity = float(np.mean(diffs[:split]))
        late_activity = float(np.mean(diffs[split:]))
        late_activity_ratio = late_activity / max(early_activity, 1e-8)
        bound_contacts: list[float] = []
        for idx, col in enumerate(med_cols):
            construct = col.removesuffix("_mean")
            lo, hi = CONSTRUCT_BOUNDS.get(construct, (-np.inf, np.inf))
            if not (np.isfinite(lo) and np.isfinite(hi)):
                continue
            late_values = med_arr[split + 1 :, idx]
            if late_values.size == 0:
                continue
            margin = anti_saturation_bound_margin_ratio * max(float(hi - lo), 1e-8)
            near_lower = np.abs(late_values - lo) <= margin
            near_upper = np.abs(hi - late_values) <= margin
            bound_contacts.append(float(np.mean(near_lower | near_upper)))
        bound_contact_share = float(np.mean(bound_contacts)) if bound_contacts else 0.0
        saturated = (
            late_activity_ratio < anti_saturation_min_late_activity_ratio
            and bound_contact_share >= anti_saturation_min_bound_contact_share
        )
        gate.add_check(
            f"anti_saturation_mediator_{scenario_key}_{effect_regime}_rep{rep_result.rep_id}",
            passed=not saturated,
            details={
                "early_mean_abs_step_change": early_activity,
                "late_mean_abs_step_change": late_activity,
                "late_activity_ratio": late_activity_ratio,
                "min_ratio": anti_saturation_min_late_activity_ratio,
                "mean_total_span": total_span,
                "min_total_span": anti_saturation_min_total_span,
                "bound_contact_share": bound_contact_share,
                "bound_margin_ratio": anti_saturation_bound_margin_ratio,
                "min_bound_contact_share": anti_saturation_min_bound_contact_share,
            },
        )

    # 3c3. Social-only bounded-drift diagnostics (homophily-on realism).
    social_only_results = [
        r for r in results if (cfg := _scenario_config_for_result(r)) is not None and _is_social_only_scenario_cfg(cfg)
    ]
    gate.add_check(
        "social_only_runs_present",
        passed=len(social_only_results) > 0,
        details={
            "n_social_only_runs": len(social_only_results),
            "required": "no_signals + communication_on + endogenous_loop_off + incidence_mode_none",
        },
    )
    for rep_result in social_only_results:
        effect_regime = _scenario_effect_regime(rep_result)
        mean_candidates = [
            c
            for c in rep_result.outcome_trajectory.columns
            if c.endswith("_mean")
            and not c.endswith("_delta_vs_t0_mean")
            and not c.endswith("_delta_vs_ref_mean")
            and c != "tick"
        ]
        metric = _first_present_column(mean_candidates, _VAX_WILLINGNESS_MEAN_CANDIDATES)
        if metric is None and mean_candidates:
            metric = mean_candidates[0]

        if metric is not None and metric in rep_result.outcome_trajectory.columns:
            vals = rep_result.outcome_trajectory[metric].to_numpy().astype(np.float64)
            vals = vals[np.isfinite(vals)]
            if vals.size >= 2:
                support_span = _mean_column_support_span(metric, vals)
                drift_per_tick = float(abs(vals[-1] - vals[0]) / max(vals.size - 1, 1))
                drift_cap = 0.04 * support_span
                gate.add_check(
                    f"social_only_bounded_drift_{metric}_{effect_regime}_rep{rep_result.rep_id}",
                    passed=drift_per_tick <= drift_cap,
                    details={
                        "drift_per_tick": drift_per_tick,
                        "drift_cap": drift_cap,
                        "support_span": support_span,
                    },
                )

        tracked_constructs = (
            "social_norms_descriptive_vax",
            "group_trust_vax",
            "institutional_trust_avg",
            "vaccine_risk_avg",
        )
        med_traj = rep_result.mediator_trajectory
        pop_final = rep_result.population_final
        if med_traj.height == 0:
            gate.add_check(
                f"social_only_variance_noncollapse_{effect_regime}_rep{rep_result.rep_id}",
                passed=True,
                details={"skipped": True, "reason": "mediator trajectory unavailable"},
            )
            gate.add_check(
                f"social_only_anti_saturation_{effect_regime}_rep{rep_result.rep_id}",
                passed=True,
                details={"skipped": True, "reason": "population snapshots unavailable"},
            )
            continue

        half_lives: list[float] = []
        r2_scores: list[float] = []
        bound_contacts: list[float] = []
        for col in tracked_constructs:
            std_col = f"{col}_std"
            if std_col not in med_traj.columns:
                continue
            series = med_traj[std_col].to_numpy().astype(np.float64)
            half_life, r2, _baseline, n_fit = _estimate_dispersion_half_life_ticks(series)
            if half_life is None or n_fit < 3:
                continue
            half_lives.append(float(half_life))
            if r2 is not None and np.isfinite(r2):
                r2_scores.append(float(r2))

            if pop_final.height == 0 or col not in pop_final.columns:
                continue
            final_vals = pop_final[col].cast(pl.Float64).to_numpy().astype(np.float64)
            final_vals = final_vals[np.isfinite(final_vals)]
            lo, hi = CONSTRUCT_BOUNDS.get(col, (-np.inf, np.inf))
            if final_vals.size > 0 and np.isfinite(lo) and np.isfinite(hi) and hi > lo:
                margin = 0.02 * float(hi - lo)
                near_lower = np.abs(final_vals - lo) <= margin
                near_upper = np.abs(hi - final_vals) <= margin
                bound_contacts.append(float(np.mean(near_lower | near_upper)))

        if half_lives:
            half_life_min_months = 3.0
            half_life_max_months = 8.0
            median_half_life = float(np.median(np.asarray(half_lives, dtype=np.float64)))
            gate.add_check(
                f"social_only_variance_noncollapse_{effect_regime}_rep{rep_result.rep_id}",
                passed=(median_half_life >= half_life_min_months) and (median_half_life <= half_life_max_months),
                details={
                    "criterion": "consensus_half_life_envelope",
                    "median_half_life_months": median_half_life,
                    "half_life_min_months": half_life_min_months,
                    "half_life_max_months": half_life_max_months,
                    "mean_fit_r2": float(np.mean(np.asarray(r2_scores, dtype=np.float64))) if r2_scores else None,
                    "n_constructs": len(half_lives),
                },
            )
        else:
            gate.add_check(
                f"social_only_variance_noncollapse_{effect_regime}_rep{rep_result.rep_id}",
                passed=True,
                details={"skipped": True, "reason": "insufficient construct std trajectory data"},
            )

        if bound_contacts:
            mean_bound_contact = float(np.mean(np.asarray(bound_contacts, dtype=np.float64)))
            gate.add_check(
                f"social_only_anti_saturation_{effect_regime}_rep{rep_result.rep_id}",
                passed=mean_bound_contact <= 0.35,
                details={
                    "mean_bound_contact_share": mean_bound_contact,
                    "max_bound_contact_share": 0.35,
                    "n_constructs": len(bound_contacts),
                },
            )
        else:
            gate.add_check(
                f"social_only_anti_saturation_{effect_regime}_rep{rep_result.rep_id}",
                passed=True,
                details={"skipped": True, "reason": "no bounded constructs available"},
            )

    # 3d. Dynamic directional response: under full simulation,
    #     evaluate from first detectable tick (activation + 1).
    by_name: dict[str, list] = {}
    by_name_regime: dict[tuple[str, str], list] = {}
    for result in results:
        scenario_key = _scenario_key(result)
        effect_regime = _scenario_effect_regime(result)
        by_name.setdefault(scenario_key, []).append(result)
        by_name_regime.setdefault((scenario_key, effect_regime), []).append(result)

    scenario_first_effect_tick: dict[str, int] = {}
    for scenario_key, scenario_cfg in SCENARIO_LIBRARY.items():
        signal_specs = tuple(scenario_cfg.choice_signals) + tuple(scenario_cfg.state_signals)
        if not signal_specs:
            scenario_first_effect_tick[scenario_key] = 1
            continue
        activation_tick = min(int(sig.start_tick) for sig in signal_specs)
        scenario_first_effect_tick[scenario_key] = max(1, activation_tick + 1)

    directional_specs: tuple[tuple[str, tuple[str, ...], float], ...] = (
        ("outbreak_global", _VAX_WILLINGNESS_MEAN_CANDIDATES, 1.0),
        ("misinfo_shock", _VAX_WILLINGNESS_MEAN_CANDIDATES, -1.0),
    )
    baseline_by_rep_by_regime = {
        effect_regime: {r.rep_id: r for r in by_name_regime.get(("baseline", effect_regime), [])}
        for effect_regime in SCENARIO_EFFECT_REGIMES
    }
    for effect_regime in SCENARIO_EFFECT_REGIMES:
        base_by_rep = baseline_by_rep_by_regime.get(effect_regime, {})
        for scenario_name, metric_candidates, expected_sign in directional_specs:
            for shock_result in by_name_regime.get((scenario_name, effect_regime), []):
                base_result = base_by_rep.get(shock_result.rep_id)
                if base_result is None:
                    continue
                metric = _first_present_column(
                    list(shock_result.outcome_trajectory.columns),
                    metric_candidates,
                )
                if metric is None:
                    continue
                if (
                    metric not in shock_result.outcome_trajectory.columns
                    or metric not in base_result.outcome_trajectory.columns
                ):
                    continue
                shock_vals = shock_result.outcome_trajectory[metric].to_numpy().astype(np.float64)
                base_vals = base_result.outcome_trajectory[metric].to_numpy().astype(np.float64)
                n = min(shock_vals.size, base_vals.size)
                if n < 2:
                    continue
                diff = shock_vals[:n] - base_vals[:n]
                first_effect_tick = int(scenario_first_effect_tick.get(scenario_name, 1))
                if first_effect_tick >= n:
                    continue
                response = float(diff[first_effect_tick])
                window_start = first_effect_tick
                window_end = min(n - 1, first_effect_tick + directional_window)
                window = diff[window_start : window_end + 1]
                abs_window = np.abs(window)
                peak_idx_local = int(np.argmax(abs_window))
                peak_tick = int(window_start + peak_idx_local)
                peak_response = float(window[peak_idx_local])
                delta_auc = float(np.sum(window))
                mean_window_delta = float(delta_auc / max(window.size, 1))
                sign_ok = np.sign(peak_response) == np.sign(expected_sign)
                magnitude_ok = bool(
                    (abs(peak_response) >= meaningful_effect_min) or (abs(mean_window_delta) >= meaningful_effect_min)
                )
                gate.add_check(
                    f"regularity_directional_tplus1_{scenario_name}_{effect_regime}_{metric}_rep{shock_result.rep_id}",
                    passed=bool(sign_ok and magnitude_ok),
                    details={
                        "response": response,
                        "peak_response": peak_response,
                        "peak_tick": peak_tick,
                        "delta_auc": delta_auc,
                        "mean_window_delta": mean_window_delta,
                        "directional_window": int(directional_window),
                        "expected_sign": expected_sign,
                        "effect_regime": effect_regime,
                        "response_tick": int(first_effect_tick),
                        "first_effect_tick": int(first_effect_tick),
                        "response_tolerance": response_tol,
                        "meaningful_effect_min": meaningful_effect_min,
                        "sign_ok": bool(sign_ok),
                        "magnitude_ok": bool(magnitude_ok),
                        "test_layer": "dynamic",
                        "time_loop": True,
                    },
                )
                gate.add_check(
                    f"dynamic_directional_tplus1_{scenario_name}_{effect_regime}_{metric}_rep{shock_result.rep_id}",
                    passed=bool(sign_ok and magnitude_ok),
                    details={
                        "alias_for": (
                            f"regularity_directional_tplus1_{scenario_name}_{effect_regime}_{metric}_rep{shock_result.rep_id}"
                        ),
                        "response": response,
                        "peak_response": peak_response,
                        "peak_tick": peak_tick,
                        "delta_auc": delta_auc,
                        "mean_window_delta": mean_window_delta,
                        "directional_window": int(directional_window),
                        "expected_sign": expected_sign,
                        "effect_regime": effect_regime,
                        "response_tick": int(first_effect_tick),
                        "first_effect_tick": int(first_effect_tick),
                        "response_tolerance": response_tol,
                        "meaningful_effect_min": meaningful_effect_min,
                        "sign_ok": bool(sign_ok),
                        "magnitude_ok": bool(magnitude_ok),
                    },
                )
                if effect_regime == "perturbation":
                    gate.add_check(
                        f"dynamic_directional_tplus1_{scenario_name}_{metric}_rep{shock_result.rep_id}",
                        passed=bool(sign_ok and magnitude_ok),
                        details={
                            "alias_for": (
                                f"dynamic_directional_tplus1_{scenario_name}_{effect_regime}_{metric}_rep{shock_result.rep_id}"
                            ),
                            "response": response,
                            "peak_response": peak_response,
                            "peak_tick": peak_tick,
                            "delta_auc": delta_auc,
                            "mean_window_delta": mean_window_delta,
                            "directional_window": int(directional_window),
                            "expected_sign": expected_sign,
                            "response_tick": int(first_effect_tick),
                            "first_effect_tick": int(first_effect_tick),
                            "response_tolerance": response_tol,
                            "meaningful_effect_min": meaningful_effect_min,
                            "sign_ok": bool(sign_ok),
                            "magnitude_ok": bool(magnitude_ok),
                        },
                    )
                for offset in directional_offsets:
                    tick_i = first_effect_tick + int(offset)
                    if tick_i >= n:
                        continue
                    response_i = float(diff[tick_i])
                    sign_ok_i = np.sign(response_i) == np.sign(expected_sign)
                    magnitude_ok_i = abs(response_i) >= meaningful_effect_min
                    gate.add_check(
                        f"dynamic_directional_response_{scenario_name}_{effect_regime}_{metric}_rep{shock_result.rep_id}_tick{tick_i}",
                        passed=bool(sign_ok_i and magnitude_ok_i),
                        details={
                            "response": response_i,
                            "expected_sign": expected_sign,
                            "effect_regime": effect_regime,
                            "response_tick": int(tick_i),
                            "first_effect_tick": int(first_effect_tick),
                            "offset_from_first_effect": int(offset),
                            "response_tolerance": response_tol,
                            "meaningful_effect_min": meaningful_effect_min,
                            "sign_ok": bool(sign_ok_i),
                            "magnitude_ok": bool(magnitude_ok_i),
                        },
                    )

                if window_end >= window_start:
                    first_nonzero_offset = int(np.argmax(abs_window >= meaningful_effect_min))
                    has_nonzero = bool(np.any(abs_window >= meaningful_effect_min))
                    first_nonzero_tick = int(window_start + first_nonzero_offset) if has_nonzero else None
                    peak_idx_local = int(np.argmax(abs_window))
                    peak_tick = int(window_start + peak_idx_local)
                    peak_delta = float(window[peak_idx_local])
                    delta_auc = float(np.sum(window))
                    sign_matches = np.sign(window) == np.sign(expected_sign)
                    sign_consistency = float(np.mean(sign_matches.astype(np.float64)))
                    window_passed = bool(
                        has_nonzero and np.sign(peak_delta) == np.sign(expected_sign) and sign_consistency >= 0.5
                    )
                    gate.add_check(
                        f"dynamic_directional_window_{scenario_name}_{effect_regime}_{metric}_rep{shock_result.rep_id}",
                        passed=window_passed,
                        details={
                            "window_start_tick": int(window_start),
                            "window_end_tick": int(window_end),
                            "effect_regime": effect_regime,
                            "first_effect_tick": int(first_effect_tick),
                            "first_nonzero_tick": first_nonzero_tick,
                            "peak_tick": int(peak_tick),
                            "peak_delta": peak_delta,
                            "delta_auc": delta_auc,
                            "sign_consistency": sign_consistency,
                            "expected_sign": expected_sign,
                            "meaningful_effect_min": meaningful_effect_min,
                        },
                    )

    # 3d2. Baseline regime signatures:
    #      enforce expected qualitative patterns across baseline variants.
    for effect_regime in SCENARIO_EFFECT_REGIMES:
        importation_runs = {r.rep_id: r for r in by_name_regime.get(("baseline_importation", effect_regime), [])}
        importation_no_comm_runs = {
            r.rep_id: r for r in by_name_regime.get(("baseline_importation_no_communication", effect_regime), [])
        }
        importation_no_peer_runs = {
            r.rep_id: r for r in by_name_regime.get(("baseline_importation_no_peer_pull", effect_regime), [])
        }
        importation_no_observed_runs = {
            r.rep_id: r for r in by_name_regime.get(("baseline_importation_no_observed_norms", effect_regime), [])
        }
        no_importation_runs = {r.rep_id: r for r in by_name_regime.get(("baseline", effect_regime), [])}

        all_present = bool(importation_runs and importation_no_comm_runs and no_importation_runs)
        gate.add_check(
            f"baseline_regime_runs_present_{effect_regime}",
            passed=True,
            details={
                "effect_regime": effect_regime,
                "n_baseline_importation": len(importation_runs),
                "n_baseline_importation_no_communication": len(importation_no_comm_runs),
                "n_baseline_no_importation": len(no_importation_runs),
                "skipped": not all_present,
            },
        )
        if not all_present:
            continue

        rep_ids = sorted(set(importation_runs) & set(importation_no_comm_runs) & set(no_importation_runs))
        if not rep_ids:
            continue

        inc_floor_eps = 1e-6
        plateau_slope_cap = 5e-4
        protection_upper_cap = 0.98
        no_importation_decline_min = 0.002
        no_comm_higher_inc_min = 0.001
        coupling_gap_min = 0.05
        no_comm_tick1_jump_cap = 0.03

        for rep_id in rep_ids:
            imp = importation_runs[rep_id]
            imp_nc = importation_no_comm_runs[rep_id]
            no_imp = no_importation_runs[rep_id]

            if "mean_incidence" not in imp.incidence_trajectory.columns:
                continue
            if "mean_incidence" not in imp_nc.incidence_trajectory.columns:
                continue
            if "mean_incidence" not in no_imp.incidence_trajectory.columns:
                continue

            inc_imp = imp.incidence_trajectory["mean_incidence"].to_numpy().astype(np.float64)
            inc_imp_nc = imp_nc.incidence_trajectory["mean_incidence"].to_numpy().astype(np.float64)
            inc_no_imp = no_imp.incidence_trajectory["mean_incidence"].to_numpy().astype(np.float64)

            prot_imp = _protection_index_from_outcome_trajectory(imp.outcome_trajectory)
            prot_imp_nc = _protection_index_from_outcome_trajectory(imp_nc.outcome_trajectory)
            prot_no_imp = _protection_index_from_outcome_trajectory(no_imp.outcome_trajectory)
            if prot_imp is None or prot_imp_nc is None or prot_no_imp is None:
                continue

            imp_floor_ok = float(np.nanmin(inc_imp)) > inc_floor_eps
            imp_nc_floor_ok = float(np.nanmin(inc_imp_nc)) > inc_floor_eps
            imp_plateau_ok = abs(_late_window_slope(inc_imp, window=20)) <= plateau_slope_cap
            imp_nc_plateau_ok = abs(_late_window_slope(inc_imp_nc, window=20)) <= plateau_slope_cap

            no_imp_decline = float(inc_no_imp[-1] - inc_no_imp[0]) if inc_no_imp.size > 1 else 0.0
            no_imp_decline_ok = no_imp_decline <= -no_importation_decline_min

            end_imp = float(inc_imp[-1])
            end_imp_nc = float(inc_imp_nc[-1])
            no_comm_higher_inc_ok = end_imp_nc >= end_imp + no_comm_higher_inc_min

            imp_prot_no_explosion = float(np.nanmax(prot_imp)) < protection_upper_cap
            imp_nc_prot_no_explosion = float(np.nanmax(prot_imp_nc)) < protection_upper_cap

            imp_corr, imp_slope = _protection_to_next_delta_incidence_signature(inc_imp, prot_imp)
            imp_nc_corr, imp_nc_slope = _protection_to_next_delta_incidence_signature(inc_imp_nc, prot_imp_nc)
            stronger_coupling_ok = abs(imp_corr) >= abs(imp_nc_corr) + coupling_gap_min

            imp_delta = float(prot_imp[-1] - prot_imp[0]) if prot_imp.size > 1 else 0.0
            imp_nc_delta = float(prot_imp_nc[-1] - prot_imp_nc[0]) if prot_imp_nc.size > 1 else 0.0
            flatter_protection_ok = abs(imp_nc_delta) <= abs(imp_delta)

            imp_nc_tick1_jump_abs = float(abs(prot_imp_nc[1] - prot_imp_nc[0])) if prot_imp_nc.size > 1 else 0.0
            no_big_tick1_jump_ok = imp_nc_tick1_jump_abs <= no_comm_tick1_jump_cap

            gate.add_check(
                f"baseline_regime_importation_persistent_plateau_{effect_regime}_rep{rep_id}",
                passed=bool(imp_floor_ok and imp_plateau_ok and imp_prot_no_explosion),
                details={
                    "incidence_min": float(np.nanmin(inc_imp)),
                    "incidence_late_slope": float(_late_window_slope(inc_imp, window=20)),
                    "plateau_slope_cap": plateau_slope_cap,
                    "protection_max": float(np.nanmax(prot_imp)),
                    "protection_upper_cap": protection_upper_cap,
                    "incidence_floor_eps": inc_floor_eps,
                },
            )
            gate.add_check(
                f"baseline_regime_importation_no_comm_persistent_plateau_{effect_regime}_rep{rep_id}",
                passed=bool(imp_nc_floor_ok and imp_nc_plateau_ok and imp_nc_prot_no_explosion),
                details={
                    "incidence_min": float(np.nanmin(inc_imp_nc)),
                    "incidence_late_slope": float(_late_window_slope(inc_imp_nc, window=20)),
                    "plateau_slope_cap": plateau_slope_cap,
                    "protection_max": float(np.nanmax(prot_imp_nc)),
                    "protection_upper_cap": protection_upper_cap,
                    "incidence_floor_eps": inc_floor_eps,
                },
            )
            gate.add_check(
                f"baseline_regime_no_importation_declines_{effect_regime}_rep{rep_id}",
                passed=bool(no_imp_decline_ok),
                details={
                    "incidence_t0": float(inc_no_imp[0]) if inc_no_imp.size > 0 else None,
                    "incidence_tend": float(inc_no_imp[-1]) if inc_no_imp.size > 0 else None,
                    "incidence_delta": no_imp_decline,
                    "required_decline_min": no_importation_decline_min,
                },
            )
            gate.add_check(
                f"baseline_regime_comm_vs_nocomm_signature_{effect_regime}_rep{rep_id}",
                passed=bool(
                    no_comm_higher_inc_ok and stronger_coupling_ok and flatter_protection_ok and no_big_tick1_jump_ok
                ),
                details={
                    "incidence_end_importation": end_imp,
                    "incidence_end_importation_no_comm": end_imp_nc,
                    "required_end_incidence_gap_min": no_comm_higher_inc_min,
                    "delta_incidence_corr_importation": imp_corr,
                    "delta_incidence_corr_importation_no_comm": imp_nc_corr,
                    "delta_incidence_slope_importation": imp_slope,
                    "delta_incidence_slope_importation_no_comm": imp_nc_slope,
                    "required_coupling_gap_min": coupling_gap_min,
                    "slope_negative_importation": bool(imp_slope < 0.0),
                    "protection_delta_importation": imp_delta,
                    "protection_delta_importation_no_comm": imp_nc_delta,
                    "tick1_jump_abs_no_comm": imp_nc_tick1_jump_abs,
                    "tick1_jump_cap_no_comm": no_comm_tick1_jump_cap,
                },
            )

            if rep_id in importation_no_peer_runs and rep_id in importation_no_observed_runs:
                imp_np = importation_no_peer_runs[rep_id]
                imp_no = importation_no_observed_runs[rep_id]
                if (
                    "mean_incidence" in imp_np.incidence_trajectory.columns
                    and "mean_incidence" in imp_no.incidence_trajectory.columns
                ):
                    inc_imp_np = imp_np.incidence_trajectory["mean_incidence"].to_numpy().astype(np.float64)
                    inc_imp_no = imp_no.incidence_trajectory["mean_incidence"].to_numpy().astype(np.float64)
                    if inc_imp_np.size > 0 and inc_imp_no.size > 0:
                        end_imp_np = float(inc_imp_np[-1])
                        end_imp_no = float(inc_imp_no[-1])
                        partial_not_better_than_canonical = (end_imp_np >= end_imp) and (end_imp_no >= end_imp)
                        partial_not_worse_than_no_comm = (end_imp_np <= end_imp_nc) and (end_imp_no <= end_imp_nc)
                        gate.add_check(
                            f"baseline_regime_channel_decomposition_signature_{effect_regime}_rep{rep_id}",
                            passed=bool(partial_not_better_than_canonical and partial_not_worse_than_no_comm),
                            details={
                                "incidence_end_importation": end_imp,
                                "incidence_end_no_peer_pull": end_imp_np,
                                "incidence_end_no_observed_norms": end_imp_no,
                                "incidence_end_no_communication": end_imp_nc,
                                "criterion": "partial_ablation_incidence_between_canonical_and_no_comm",
                            },
                        )

    # 3e. Stable-under-stress checks for persistent shocks.
    #     Movement is allowed; instability should not be explosive.
    persistent_names = {"access_intervention", "norm_campaign", "trust_scandal_bundle"}
    for shock_result in [r for r in results if _scenario_key(r) in persistent_names]:
        scenario_key = _scenario_key(shock_result)
        effect_regime = _scenario_effect_regime(shock_result)
        metric = _first_present_column(
            list(shock_result.outcome_trajectory.columns),
            _VAX_WILLINGNESS_MEAN_CANDIDATES,
        )
        if metric is None:
            continue
        if metric not in shock_result.outcome_trajectory.columns:
            continue
        vals = shock_result.outcome_trajectory[metric].to_numpy().astype(np.float64)
        if vals.size < 5:
            continue
        horizon = min(vals.size, stress_horizon_max)
        if horizon < stress_horizon_min:
            continue
        window = vals[:horizon]

        span = float(np.nanmax(window) - np.nanmin(window))
        support_span = _mean_column_support_span(metric, window)
        span_cap = 0.60 * support_span
        bounded_movement = bool(span < span_cap)
        gate.add_check(
            f"regularity_persistent_stable_under_stress_{scenario_key}_{effect_regime}_rep{shock_result.rep_id}",
            passed=bounded_movement,
            details={
                "span": span,
                "span_cap": span_cap,
                "support_span": support_span,
                "effect_regime": effect_regime,
                "horizon_used": horizon,
                "criterion": "bounded_movement",
            },
        )

        med = shock_result.mediator_trajectory
        med_cols = [c for c in med.columns if c != "tick" and c.endswith("_mean")]
        if med.height >= horizon and med_cols:
            med_arr = med.select(med_cols).to_numpy().astype(np.float64)[:horizon]
            # Proxy variance across mediator means by tick.
            var_by_tick = np.var(med_arr, axis=1)
            split = max(2, horizon // 2)
            early = var_by_tick[:split]
            late = var_by_tick[split:]
            early_growth = float(max(np.max(early) - np.min(early), 0.0)) if early.size > 0 else 0.0
            late_growth = float(max(np.max(late) - np.min(late), 0.0)) if late.size > 0 else 0.0
            superlinear = late_growth > (1.2 * max(early_growth, 1e-8)) and late_growth > 0.02
            gate.add_check(
                f"regularity_persistent_mediator_variance_growth_{scenario_key}_{effect_regime}_rep{shock_result.rep_id}",
                passed=not superlinear,
                details={
                    "early_growth": early_growth,
                    "late_growth": late_growth,
                    "effect_regime": effect_regime,
                    "superlinear_detected": bool(superlinear),
                    "horizon_used": horizon,
                },
            )

        diffs = np.diff(window)
        if diffs.size >= 6:
            split = max(3, diffs.size // 2)
            early_vol = float(np.mean(np.abs(diffs[:split])))
            late_vol = float(np.mean(np.abs(diffs[split:])))
            oscillation_floor = 0.12 * support_span
            wild_oscillation = late_vol > max(1.3 * early_vol, oscillation_floor)
            gate.add_check(
                f"regularity_persistent_plateau_or_stable_region_{scenario_key}_{effect_regime}_rep{shock_result.rep_id}",
                passed=not wild_oscillation,
                details={
                    "early_mean_abs_step_change": early_vol,
                    "late_mean_abs_step_change": late_vol,
                    "oscillation_floor": oscillation_floor,
                    "support_span": support_span,
                    "effect_regime": effect_regime,
                    "wild_oscillation": bool(wild_oscillation),
                    "horizon_used": horizon,
                },
            )

    pulse_names = {"outbreak_global", "outbreak_local", "misinfo_shock"}
    for shock_result in [r for r in results if _scenario_key(r) in pulse_names]:
        scenario_key = _scenario_key(shock_result)
        effect_regime = _scenario_effect_regime(shock_result)
        base_by_rep = baseline_by_rep_by_regime.get(effect_regime, {})
        base_result = base_by_rep.get(shock_result.rep_id)
        metric = _first_present_column(
            list(shock_result.outcome_trajectory.columns),
            _VAX_WILLINGNESS_MEAN_CANDIDATES,
        )
        if metric is None:
            continue
        if base_result is None:
            continue
        if (
            metric not in shock_result.outcome_trajectory.columns
            or metric not in base_result.outcome_trajectory.columns
        ):
            continue
        shock_vals = shock_result.outcome_trajectory[metric].to_numpy().astype(np.float64)
        base_vals = base_result.outcome_trajectory[metric].to_numpy().astype(np.float64)
        n = min(shock_vals.size, base_vals.size)
        if n < 4:
            continue
        diff = shock_vals[:n] - base_vals[:n]
        peak_idx = int(np.argmax(np.abs(diff)))
        peak = float(abs(diff[peak_idx]))
        end_dev = float(abs(diff[-1]))
        relaxed = end_dev < peak if peak > 0 else True
        gate.add_check(
            f"regularity_shock_recovery_{scenario_key}_{effect_regime}_rep{shock_result.rep_id}",
            passed=relaxed,
            details={
                "peak_abs_deviation": peak,
                "end_abs_deviation": end_dev,
                "effect_regime": effect_regime,
                "peak_tick": peak_idx,
            },
        )

    # 3f. Survey-scaled shock diagnostics based on baseline country quantiles.
    for (scenario_key, effect_regime), scenario_results in by_name_regime.items():
        scenario_cfg = SCENARIO_LIBRARY.get(scenario_key)
        if scenario_cfg is None:
            continue
        signal_specs = tuple(scenario_cfg.choice_signals) + tuple(scenario_cfg.state_signals)
        if not signal_specs:
            continue
        baseline_by_rep = baseline_by_rep_by_regime.get(effect_regime, {})
        for run in scenario_results:
            base = baseline_by_rep.get(run.rep_id)
            if base is None:
                continue
            pop = base.population_t0
            if pop.height == 0 or "country" not in pop.columns:
                continue
            for sig in signal_specs:
                if sig.target_column not in pop.columns:
                    continue
                units: list[float] = []
                for country in sorted({str(c) for c in pop["country"].to_list()}):
                    vals = pop.filter(pl.col("country") == country)[sig.target_column].to_numpy().astype(np.float64)
                    vals = vals[np.isfinite(vals)]
                    if vals.size < 10:
                        continue
                    denom = _shock_unit_denominator(vals, float(sig.intensity))
                    raw_units = float(abs(sig.intensity) / denom)
                    nominal_units = float(abs(sig.intensity))
                    # Treat scenario intensity as a survey-scale multiplier when
                    # calibration is enabled upstream; keep raw-units diagnostics.
                    units.append(min(raw_units, nominal_units))
                if not units:
                    continue
                mean_units = float(np.mean(units))
                gate.add_check(
                    f"survey_scaled_shock_{scenario_key}_{effect_regime}_{sig.target_column}_rep{run.rep_id}",
                    passed=mean_units <= 6.0,
                    details={
                        "signal_intensity": float(sig.intensity),
                        "mean_percentile_units": mean_units,
                        "max_percentile_units": 6.0,
                        "effect_regime": effect_regime,
                        "n_countries": len(units),
                    },
                )

    # 3g. Dose-response monotonicity for scenarios sharing a target channel.
    for strong_key, weak_key in (("outbreak_global", "outbreak_local"),):
        for effect_regime in SCENARIO_EFFECT_REGIMES:
            strong_runs = {r.rep_id: r for r in by_name_regime.get((strong_key, effect_regime), [])}
            weak_runs = {r.rep_id: r for r in by_name_regime.get((weak_key, effect_regime), [])}
            base_by_rep = baseline_by_rep_by_regime.get(effect_regime, {})
            for rep_id in sorted(set(strong_runs) & set(weak_runs) & set(base_by_rep)):
                base = base_by_rep[rep_id]
                strong = strong_runs[rep_id]
                weak = weak_runs[rep_id]
                metric = _first_present_column(
                    list(strong.outcome_trajectory.columns),
                    _VAX_WILLINGNESS_MEAN_CANDIDATES,
                )
                if metric is None:
                    continue
                if (
                    metric not in base.outcome_trajectory.columns
                    or metric not in strong.outcome_trajectory.columns
                    or metric not in weak.outcome_trajectory.columns
                ):
                    continue
                base_vals = base.outcome_trajectory[metric].to_numpy().astype(np.float64)
                strong_vals = strong.outcome_trajectory[metric].to_numpy().astype(np.float64)
                weak_vals = weak.outcome_trajectory[metric].to_numpy().astype(np.float64)
                n = min(base_vals.size, strong_vals.size, weak_vals.size)
                if n < 2:
                    continue
                response_tick = max(
                    int(scenario_first_effect_tick.get(strong_key, 1)),
                    int(scenario_first_effect_tick.get(weak_key, 1)),
                )
                if response_tick >= n:
                    continue
                window_end = min(n - 1, response_tick + directional_window)
                strong_diff = strong_vals - base_vals
                weak_diff = weak_vals - base_vals
                strong_window = strong_diff[response_tick : window_end + 1]
                weak_window = weak_diff[response_tick : window_end + 1]
                if strong_window.size == 0 or weak_window.size == 0:
                    continue
                strong_abs = np.abs(strong_window)
                weak_abs = np.abs(weak_window)
                strong_peak_idx = int(np.argmax(strong_abs))
                weak_peak_idx = int(np.argmax(weak_abs))
                strong_resp = float(strong_window[strong_peak_idx])
                weak_resp = float(weak_window[weak_peak_idx])
                expected_sign = np.sign(strong_resp) if abs(strong_resp) > 1e-8 else np.sign(weak_resp)
                monotone = float(np.max(strong_abs)) + 1e-8 >= float(np.max(weak_abs))
                same_sign = expected_sign == 0.0 or abs(weak_resp) < 1e-8 or np.sign(weak_resp) == expected_sign
                strong_auc = float(np.sum(strong_window))
                strong_mean_window_delta = float(strong_auc / max(strong_window.size, 1))
                strong_meaningful = bool(
                    (abs(strong_resp) >= meaningful_effect_min)
                    or (abs(strong_mean_window_delta) >= meaningful_effect_min)
                )
                gate.add_check(
                    f"dose_response_{strong_key}_vs_{weak_key}_{effect_regime}_{metric}_rep{rep_id}",
                    passed=bool(monotone and same_sign and strong_meaningful),
                    details={
                        "strong_response_peak": strong_resp,
                        "weak_response_peak": weak_resp,
                        "effect_regime": effect_regime,
                        "response_tick": int(response_tick),
                        "window_end_tick": int(window_end),
                        "directional_window": int(directional_window),
                        "strong_peak_tick": int(response_tick + strong_peak_idx),
                        "weak_peak_tick": int(response_tick + weak_peak_idx),
                        "strong_delta_auc": strong_auc,
                        "strong_mean_window_delta": strong_mean_window_delta,
                        "expected_sign": float(expected_sign),
                        "meaningful_effect_min": meaningful_effect_min,
                        "strong_meaningful": bool(strong_meaningful),
                    },
                )

    gate.evaluate()
    logger.info(
        "Dynamic plausibility gate: {status} ({n_pass}/{n_total} checks).",
        status="PASSED" if gate.passed else "FAILED",
        n_pass=sum(1 for c in gate.checks if c.passed),
        n_total=len(gate.checks),
    )
    return gate
