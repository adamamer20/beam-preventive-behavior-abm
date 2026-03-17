"""Mechanical verification benchmarking gate."""

from __future__ import annotations

import numpy as np
import polars as pl
from loguru import logger

from beam_abm.abm.config import SimulationConfig
from beam_abm.abm.data_contracts import BackboneModel, LeversConfig, get_abm_levers_config
from beam_abm.abm.outcome_scales import normalize_outcome_to_unit_interval
from beam_abm.abm.outcomes import predict_all_outcomes
from beam_abm.abm.state_schema import CONSTRUCT_BOUNDS, MUTABLE_COLUMNS
from beam_abm.abm.state_update import (
    SignalSpec,
    apply_all_state_updates,
    compute_observed_norm_fixed_point_share,
)

from .core import (
    GateResult,
    _chapter6_like_delta,
    _partial_effect,
)


def run_mechanical_verification_gate(
    agents: pl.DataFrame,
    models: dict[str, BackboneModel],
    levers_cfg: LeversConfig,
    cfg: SimulationConfig | None = None,
    tolerance: float = 0.01,
    pe_ref_meaningful_threshold: float = 0.01,
    pe_ref_responsiveness_ratio: float = 0.5,
    pe_ref_responsiveness_abs_floor: float = 0.005,
) -> GateResult:
    """One-step parity, perturbation, wiring, and context separation checks.

    Parameters
    ----------
    agents : pl.DataFrame
        Agent population (t=0 snapshot).
    models : dict[str, BackboneModel]
        Loaded backbone models.
    levers_cfg : LeversConfig
        Lever configuration.
    cfg : SimulationConfig or None
        Runtime configuration used for parameter-consistent coherence checks.
    tolerance : float
        Numerical tolerance for parity checks.

    Returns
    -------
    GateResult
    """
    gate = GateResult(gate_name="mechanical_verification")
    levers_cfg = get_abm_levers_config(levers_cfg)
    rng = np.random.default_rng(42)
    n_agents = max(1, agents.height)
    n_scale = float(np.sqrt(1000.0 / n_agents))

    # 2a. One-step parity: deterministic predictions should be reproducible
    preds_1 = predict_all_outcomes(agents, models, levers_cfg, deterministic=True, rng=rng)
    preds_2 = predict_all_outcomes(agents, models, levers_cfg, deterministic=True, rng=rng)
    for outcome_name in preds_1:
        diff = np.max(np.abs(preds_1[outcome_name] - preds_2[outcome_name]))
        gate.add_check(
            f"onestep_parity_{outcome_name}",
            passed=diff < tolerance,
            details={"max_diff": float(diff)},
        )

    # 2b. Outcome-to-model wiring: each outcome uses correct model_ref
    for outcome_name, spec in levers_cfg.outcomes.items():
        correct_model = spec.model_ref
        loaded_model = models.get(outcome_name)
        wired_correctly = loaded_model is not None and loaded_model.name == correct_model
        gate.add_check(
            f"wiring_{outcome_name}",
            passed=wired_correctly,
            details={
                "expected_ref": correct_model,
                "loaded_ref": loaded_model.name if loaded_model else None,
            },
        )

    # 2c. Partial-effect gradients and perturbation reference effects.
    for outcome_name, spec in levers_cfg.outcomes.items():
        model = models[outcome_name]
        candidates: list[tuple[str, float, float, float, float]] = []
        for lever in spec.levers:
            if lever not in model.terms:
                continue
            idx = list(model.terms).index(lever)
            coeff = float(model.coefficients[idx])
            if abs(coeff) < 0.05:
                continue
            vals = agents[lever].to_numpy().astype(np.float64)
            delta = _chapter6_like_delta(vals)
            pe_ref, pe_abm = _partial_effect(
                agents,
                models,
                levers_cfg,
                outcome_name=outcome_name,
                predictor=lever,
                delta=delta,
            )
            candidates.append((lever, coeff, delta, pe_ref, pe_abm))

        matched = False
        if candidates:
            lever, coeff, delta, pe_ref, pe_abm = max(candidates, key=lambda t: abs(t[3]))
            expected_sign = np.sign(coeff)
            partial_sign_ok = np.sign(pe_ref) == expected_sign or abs(pe_ref) < 1e-8
            gate.add_check(
                f"partial_gradient_{lever}_to_{outcome_name}",
                passed=bool(partial_sign_ok),
                details={
                    "coefficient": float(coeff),
                    "delta": float(delta),
                    "pe_ref": pe_ref,
                    "expected_sign": float(expected_sign),
                    "selection_rule": "max_abs_pe_ref",
                },
            )

            sign_ok = np.sign(pe_abm) == np.sign(pe_ref) or abs(pe_abm) < 1e-8 or abs(pe_ref) < 1e-8
            band = (0.02 + 0.5 * abs(pe_ref)) * n_scale
            mag_ok = abs(pe_abm - pe_ref) <= band
            meaningful_ref = abs(pe_ref) >= pe_ref_meaningful_threshold
            min_required = max(
                pe_ref_responsiveness_ratio * abs(pe_ref),
                pe_ref_responsiveness_abs_floor,
            )
            responsive_ok = (abs(pe_abm) >= min_required) if meaningful_ref else True
            gate.add_check(
                f"pe_ref_sign_{lever}_to_{outcome_name}",
                passed=bool(sign_ok),
                details={
                    "delta": float(delta),
                    "pe_ref": pe_ref,
                    "abm_one_step_delta": pe_abm,
                },
            )
            gate.add_check(
                f"pe_ref_closeness_{lever}_to_{outcome_name}",
                passed=bool(mag_ok),
                details={
                    "delta": float(delta),
                    "pe_ref": pe_ref,
                    "abm_one_step_delta": pe_abm,
                    "abs_diff": float(abs(pe_abm - pe_ref)),
                    "magnitude_band": float(band),
                },
            )
            gate.add_check(
                f"pe_ref_responsiveness_{lever}_to_{outcome_name}",
                passed=bool(responsive_ok),
                details={
                    "delta": float(delta),
                    "pe_ref": pe_ref,
                    "abm_one_step_delta": pe_abm,
                    "meaningful_ref": bool(meaningful_ref),
                    "pe_ref_meaningful_threshold": float(pe_ref_meaningful_threshold),
                    "min_required_abs_response": float(min_required),
                    "ratio_floor": float(pe_ref_responsiveness_ratio),
                    "abs_floor": float(pe_ref_responsiveness_abs_floor),
                },
            )
            gate.add_check(
                f"perturbation_ref_{lever}_to_{outcome_name}",
                passed=bool(sign_ok and mag_ok),
                details={
                    "delta": float(delta),
                    "pe_ref": pe_ref,
                    "abm_one_step_delta": pe_abm,
                    "abs_diff": float(abs(pe_abm - pe_ref)),
                    "magnitude_band": float(band),
                    "test_layer": "mechanical",
                    "time_loop": False,
                },
            )
            matched = True
        if not matched:
            gate.add_check(
                f"partial_gradient_available_{outcome_name}",
                passed=False,
                details={"reason": "no eligible lever with coefficient magnitude >= 0.05"},
            )

    # 2d. Context separation: choice-context signals should not mutate state
    choice_probe = SignalSpec(
        target_column="fivec_low_constraints",
        intensity=0.5,
        signal_type="sustained",
        start_tick=0,
    )
    before = agents.clone()
    probe_state = agents.clone()
    for tick in range(3):
        _ = predict_all_outcomes(
            probe_state,
            models,
            levers_cfg,
            deterministic=True,
            rng=np.random.default_rng(42 + tick),
            tick=tick,
            choice_signals=[choice_probe],
        )
    max_diff = 0.0
    for col in MUTABLE_COLUMNS:
        if col not in before.columns or col not in probe_state.columns:
            continue
        diff = float(np.max(np.abs(before[col].to_numpy() - probe_state[col].to_numpy())))
        max_diff = max(max_diff, diff)
    gate.add_check(
        "context_separation",
        passed=max_diff < tolerance,
        details={"max_state_diff": max_diff},
    )

    # 2e. Frozen-state protocol: no comm + no signals => no state change
    frozen = agents.clone()
    for tick in range(3):
        frozen = apply_all_state_updates(
            agents=frozen,
            signals=[],
            tick=tick,
            peer_means=None,
            k_social=0.0,
            rng=np.random.default_rng(142 + tick),
        )
    frozen_diff = 0.0
    for col in MUTABLE_COLUMNS:
        if col not in agents.columns or col not in frozen.columns:
            continue
        diff = float(np.max(np.abs(frozen[col].to_numpy() - agents[col].to_numpy())))
        frozen_diff = max(frozen_diff, diff)
    gate.add_check(
        "frozen_state_no_comm",
        passed=frozen_diff < tolerance,
        details={"max_state_diff": frozen_diff},
    )

    # 2f. Internal-coherence invariants under mechanism-specific ablations.
    coherent = apply_all_state_updates(
        agents=agents.clone(),
        signals=[],
        tick=0,
        peer_means=None,
        k_social=0.0,
        social_noise_std=0.0,
        state_relaxation_fast=0.0,
        state_relaxation_slow=0.0,
        rng=np.random.default_rng(223),
    )

    tracked_mutables = [
        col
        for col in MUTABLE_COLUMNS
        if col in coherent.columns and col not in {"social_norms_vax_avg", "fivec_pro_vax_idx"}
    ]

    null_world = coherent.clone()
    for tick in range(2):
        null_world = apply_all_state_updates(
            agents=null_world,
            signals=[],
            tick=tick,
            peer_means=None,
            k_social=0.0,
            social_noise_std=0.0,
            state_relaxation_fast=0.0,
            state_relaxation_slow=0.0,
            rng=np.random.default_rng(224 + tick),
        )

    null_state_drift = 0.0
    for col in tracked_mutables:
        before_vals = coherent[col].cast(pl.Float64).to_numpy().astype(np.float64)
        after_vals = null_world[col].cast(pl.Float64).to_numpy().astype(np.float64)
        if before_vals.shape != after_vals.shape:
            null_state_drift = np.inf
            break
        null_state_drift = max(null_state_drift, float(np.max(np.abs(after_vals - before_vals))))

    preds_null_before = predict_all_outcomes(
        coherent,
        models,
        levers_cfg,
        deterministic=True,
        rng=np.random.default_rng(3001),
        tick=0,
    )
    preds_null_after = predict_all_outcomes(
        null_world,
        models,
        levers_cfg,
        deterministic=True,
        rng=np.random.default_rng(3002),
        tick=0,
    )
    null_outcome_drift = 0.0
    for name in preds_null_before:
        if name not in preds_null_after:
            null_outcome_drift = np.inf
            break
        null_outcome_drift = max(
            null_outcome_drift,
            float(np.max(np.abs(preds_null_after[name] - preds_null_before[name]))),
        )

    gate.add_check(
        "coherence_null_world_state_stationarity",
        passed=null_state_drift < tolerance,
        details={"max_state_diff": null_state_drift, "tolerance": tolerance},
    )
    gate.add_check(
        "coherence_null_world_outcome_stationarity",
        passed=null_outcome_drift < tolerance,
        details={"max_outcome_diff": null_outcome_drift, "tolerance": tolerance},
    )

    incidence_targets = {
        "covid_perceived_danger_T12": 0.2,
        "disease_fear_avg": 0.14,
        "infection_likelihood_avg": 0.16,
    }
    offsets: dict[str, np.ndarray] = {
        col: np.full(coherent.height, shift, dtype=np.float64)
        for col, shift in incidence_targets.items()
        if col in coherent.columns
    }
    if offsets:
        incidence_only = apply_all_state_updates(
            agents=coherent,
            signals=[],
            tick=0,
            peer_means=None,
            k_social=0.0,
            endogenous_offsets=offsets,
            social_noise_std=0.0,
            state_relaxation_fast=0.2,
            state_relaxation_slow=0.2,
            rng=np.random.default_rng(331),
        )
        norm_cols = [
            col
            for col in ("social_norms_descriptive_vax",)
            if col in coherent.columns and col in incidence_only.columns
        ]
        norm_jump = 0.0
        for col in norm_cols:
            before_vals = coherent[col].cast(pl.Float64).to_numpy().astype(np.float64)
            after_vals = incidence_only[col].cast(pl.Float64).to_numpy().astype(np.float64)
            norm_jump = max(norm_jump, float(np.max(np.abs(after_vals - before_vals))))

        risk_move = 0.0
        for col in offsets:
            before_vals = coherent[col].cast(pl.Float64).to_numpy().astype(np.float64)
            after_vals = incidence_only[col].cast(pl.Float64).to_numpy().astype(np.float64)
            risk_move = max(risk_move, float(np.mean(np.abs(after_vals - before_vals))))

        gate.add_check(
            "coherence_incidence_only_norm_coherence",
            passed=(norm_jump < tolerance) and (risk_move > 0.0),
            details={
                "max_norm_diff": norm_jump,
                "mean_risk_move": risk_move,
                "tolerance": tolerance,
            },
        )
    else:
        gate.add_check(
            "coherence_incidence_only_norm_coherence",
            passed=True,
            details={"skipped": True, "reason": "required incidence-linked constructs missing"},
        )

    # Incidence-only fixed-point stationarity:
    # under constant protection and no importation, initializing incidence at
    # the recurrence fixed point should produce no drift across steps.
    beta = float(cfg.knobs.k_incidence) if cfg is not None else 1.05
    gamma = float(cfg.knobs.incidence_gamma) if cfg is not None else 0.65
    alpha_v = float(cfg.constants.incidence_alpha_vax) if cfg is not None else 0.35
    alpha_p = float(cfg.constants.incidence_alpha_protection) if cfg is not None else 0.30

    preds_for_incidence = predict_all_outcomes(
        coherent,
        models,
        levers_cfg,
        deterministic=True,
        rng=np.random.default_rng(333),
        tick=0,
    )
    if "vax_willingness_T12" in preds_for_incidence:
        vax_protection = normalize_outcome_to_unit_interval(
            preds_for_incidence["vax_willingness_T12"],
            "vax_willingness_T12",
        )
    else:
        vax_protection = np.zeros(coherent.height, dtype=np.float64)

    npi_arrays: list[np.ndarray] = []
    for outcome_name in (
        "mask_when_symptomatic_crowded",
        "stay_home_when_symptomatic",
        "mask_when_pressure_high",
    ):
        values = preds_for_incidence.get(outcome_name)
        if values is None:
            continue
        npi_arrays.append(normalize_outcome_to_unit_interval(values, outcome_name))
    if npi_arrays:
        npi_protection_index = np.mean(np.vstack(npi_arrays), axis=0)
    else:
        npi_protection_index = np.zeros(coherent.height, dtype=np.float64)

    if "locality_id" in coherent.columns:
        communities = coherent["locality_id"].cast(pl.Utf8).to_numpy().astype(str)
    elif "country" in coherent.columns:
        communities = coherent["country"].cast(pl.Utf8).to_numpy().astype(str)
    else:
        communities = np.array(["all"] * coherent.height, dtype=str)

    mitigation_scales: list[float] = []
    unit_means_vax: list[float] = []
    unit_means_npi: list[float] = []
    for unit in np.unique(communities):
        idx = communities == unit
        if not np.any(idx):
            continue
        mean_v = float(np.mean(vax_protection[idx]))
        mean_p = float(np.mean(npi_protection_index[idx]))
        unit_means_vax.append(mean_v)
        unit_means_npi.append(mean_p)
        mitigation_scales.append(float(np.clip(1.0 - alpha_v * mean_v - alpha_p * mean_p, 0.0, 1.0)))

    mitigation_scale_arr = np.asarray(mitigation_scales, dtype=np.float64)
    if mitigation_scale_arr.size == 0:
        mitigation_scale_arr = np.array([1.0], dtype=np.float64)

    denom = beta * mitigation_scale_arr
    fixed_points = np.zeros_like(mitigation_scale_arr)
    positive = denom > 0.0
    fixed_points[positive] = np.clip(1.0 - (gamma / denom[positive]), 0.0, 1.0)

    incidence = fixed_points.astype(np.float64, copy=True)
    max_stationarity_delta = 0.0
    for _ in range(4):
        next_inc = (1.0 - gamma) * incidence + beta * incidence * (1.0 - incidence) * mitigation_scale_arr
        next_inc = np.clip(next_inc, 0.0, 1.0)
        max_stationarity_delta = max(max_stationarity_delta, float(np.max(np.abs(next_inc - incidence))))
        incidence = next_inc

    gate.add_check(
        "coherence_incidence_fixed_point_stationarity",
        passed=max_stationarity_delta < tolerance,
        details={
            "max_incidence_step_diff": max_stationarity_delta,
            "tolerance": tolerance,
            "beta": beta,
            "gamma": gamma,
            "alpha_v": alpha_v,
            "alpha_p": alpha_p,
            "n_units": int(mitigation_scale_arr.size),
            "mean_vax_protection": float(np.mean(unit_means_vax)) if unit_means_vax else 0.0,
            "mean_npi_protection": float(np.mean(unit_means_npi)) if unit_means_npi else 0.0,
            "mean_mitigation_scale": float(np.mean(mitigation_scale_arr)),
            "mean_fixed_point": float(np.mean(fixed_points)),
        },
    )

    norm_col = "social_norms_descriptive_vax"
    if norm_col in coherent.columns and coherent.height >= 2:
        lo_norm, hi_norm = CONSTRUCT_BOUNDS.get(norm_col, (1.0, 7.0))
        safety_margin = 0.10 * max(float(hi_norm - lo_norm), 1e-8)
        desc_vals_raw = coherent[norm_col].cast(pl.Float64).to_numpy().astype(np.float64)
        desc_vals = np.clip(desc_vals_raw, lo_norm + safety_margin, hi_norm - safety_margin)
        coherent_kernel = coherent.with_columns(pl.Series(norm_col, desc_vals.astype(np.float64)))
        social_only = apply_all_state_updates(
            agents=coherent_kernel,
            signals=[],
            tick=0,
            peer_means={norm_col: np.roll(desc_vals, 1)},
            k_social=1.0,
            social_weight_norm=1.0,
            social_weight_trust=0.0,
            social_weight_risk=0.0,
            social_noise_std=0.0,
            state_relaxation_fast=0.0,
            state_relaxation_slow=0.0,
            rng=np.random.default_rng(440),
        )
        desc_after = social_only[norm_col].cast(pl.Float64).to_numpy().astype(np.float64)
        mean_drift = float(abs(np.mean(desc_after) - np.mean(desc_vals)))
        social_only_mean_tol = 5e-5
        gate.add_check(
            "coherence_social_only_mean_preservation",
            passed=mean_drift < social_only_mean_tol,
            details={
                "mean_drift": mean_drift,
                "tolerance": social_only_mean_tol,
                "controlled_kernel": True,
                "social_noise_std": 0.0,
                "state_relaxation_fast": 0.0,
                "state_relaxation_slow": 0.0,
                "homophily_gamma": 0.0,
                "clipping_avoided": True,
            },
        )
    else:
        gate.add_check(
            "coherence_social_only_mean_preservation",
            passed=True,
            details={"skipped": True, "reason": "descriptive norm channel missing or insufficient agents"},
        )

    if norm_col in coherent.columns and coherent.height >= 3:
        permutation_rng = np.random.default_rng(442)
        permutation = permutation_rng.permutation(coherent.height).astype(np.int64)
        desc_vals = coherent[norm_col].cast(pl.Float64).to_numpy().astype(np.float64)
        base_peer = np.roll(desc_vals, 1)

        base_updated = apply_all_state_updates(
            agents=coherent,
            signals=[],
            tick=0,
            peer_means={norm_col: base_peer},
            k_social=0.85,
            social_weight_norm=1.0,
            social_weight_trust=0.0,
            social_weight_risk=0.0,
            social_noise_std=0.0,
            state_relaxation_fast=0.0,
            state_relaxation_slow=0.0,
            rng=np.random.default_rng(443),
        )

        indexed = coherent.with_row_index("__row_id")
        perm_frame = pl.DataFrame(
            {
                "__perm_pos": np.arange(coherent.height, dtype=np.int64),
                "__row_id": permutation,
            }
        )
        permuted_agents = (
            perm_frame.join(indexed, on="__row_id", how="left").sort("__perm_pos").drop(["__perm_pos", "__row_id"])
        )

        permuted_updated = apply_all_state_updates(
            agents=permuted_agents,
            signals=[],
            tick=0,
            peer_means={norm_col: base_peer[permutation]},
            k_social=0.85,
            social_weight_norm=1.0,
            social_weight_trust=0.0,
            social_weight_risk=0.0,
            social_noise_std=0.0,
            state_relaxation_fast=0.0,
            state_relaxation_slow=0.0,
            rng=np.random.default_rng(443),
        )

        base_arr = base_updated[norm_col].cast(pl.Float64).to_numpy().astype(np.float64)
        perm_arr = permuted_updated[norm_col].cast(pl.Float64).to_numpy().astype(np.float64)
        unpermuted = np.empty_like(perm_arr)
        unpermuted[permutation] = perm_arr
        permutation_delta = float(np.max(np.abs(base_arr - unpermuted)))
        gate.add_check(
            "artifact_firewall_agent_permutation_invariance",
            passed=permutation_delta <= tolerance,
            details={"max_diff": permutation_delta, "tolerance": tolerance},
        )
    else:
        gate.add_check(
            "artifact_firewall_agent_permutation_invariance",
            passed=True,
            details={"skipped": True, "reason": "insufficient descriptive-norm support"},
        )

    zero_target = next((col for col in sorted(MUTABLE_COLUMNS) if col in coherent.columns), None)
    if zero_target is not None:
        zero_signal = SignalSpec(
            target_column=zero_target,
            intensity=0.0,
            signal_type="sustained",
            start_tick=0,
        )
        no_signal = apply_all_state_updates(
            agents=coherent,
            signals=[],
            tick=0,
            peer_means=None,
            k_social=0.0,
            social_noise_std=0.0,
            state_relaxation_fast=0.0,
            state_relaxation_slow=0.0,
            rng=np.random.default_rng(444),
        )
        zero_signal_state = apply_all_state_updates(
            agents=coherent,
            signals=[zero_signal],
            tick=0,
            peer_means=None,
            k_social=0.0,
            social_noise_std=0.0,
            state_relaxation_fast=0.0,
            state_relaxation_slow=0.0,
            rng=np.random.default_rng(444),
        )
        diff_max = 0.0
        for col in tracked_mutables:
            if col not in no_signal.columns or col not in zero_signal_state.columns:
                continue
            before_vals = no_signal[col].cast(pl.Float64).to_numpy().astype(np.float64)
            after_vals = zero_signal_state[col].cast(pl.Float64).to_numpy().astype(np.float64)
            diff_max = max(diff_max, float(np.max(np.abs(after_vals - before_vals))))
        gate.add_check(
            "artifact_firewall_zero_intensity_signal_identity",
            passed=diff_max <= tolerance,
            details={"max_state_diff": diff_max, "tolerance": tolerance, "target_column": zero_target},
        )
    else:
        gate.add_check(
            "artifact_firewall_zero_intensity_signal_identity",
            passed=True,
            details={"skipped": True, "reason": "no mutable columns available"},
        )

    if norm_col in coherent.columns:
        current_desc = coherent[norm_col].cast(pl.Float64).to_numpy().astype(np.float64)
        if "norm_threshold_theta" in coherent.columns:
            theta = coherent["norm_threshold_theta"].cast(pl.Float64).to_numpy().astype(np.float64)
            theta = np.nan_to_num(theta, nan=0.5, posinf=0.5, neginf=0.5)
        else:
            theta = np.full(coherent.height, 0.5, dtype=np.float64)
        theta = np.clip(theta, 0.05, 0.95)
        fixed_share = compute_observed_norm_fixed_point_share(
            descriptive_norm=current_desc,
            threshold=theta,
            column=norm_col,
            slope=8.0,
        )
        coherent_with_share = coherent.with_columns(pl.Series("norm_share_hat_vax", fixed_share.astype(np.float64)))
        fixed_point = apply_all_state_updates(
            agents=coherent_with_share,
            signals=[],
            tick=0,
            observed_norm_share=fixed_share,
            social_susceptibility=np.full(coherent.height, 1.0, dtype=np.float64),
            norm_threshold=theta,
            k_social=1.0,
            social_noise_std=0.0,
            state_relaxation_fast=0.0,
            state_relaxation_slow=0.0,
            rng=np.random.default_rng(441),
        )
        desc_after = fixed_point[norm_col].cast(pl.Float64).to_numpy().astype(np.float64)
        fixed_point_jump = float(np.max(np.abs(desc_after - current_desc)))
        fixed_point_tol = 1e-5
        gate.add_check(
            "coherence_observed_norm_fixed_point",
            passed=fixed_point_jump < fixed_point_tol,
            details={"max_norm_jump": fixed_point_jump, "tolerance": fixed_point_tol},
        )
    else:
        gate.add_check(
            "coherence_observed_norm_fixed_point",
            passed=True,
            details={"skipped": True, "reason": "descriptive norm channel missing"},
        )

    gate.evaluate()
    logger.info(
        "Mechanical verification gate: {status} ({n_pass}/{n_total} checks).",
        status="PASSED" if gate.passed else "FAILED",
        n_pass=sum(1 for c in gate.checks if c.passed),
        n_total=len(gate.checks),
    )
    return gate
