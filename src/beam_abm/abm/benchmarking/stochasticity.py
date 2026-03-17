"""Stochasticity policy benchmarking gate."""

from __future__ import annotations

import numpy as np
import polars as pl
from loguru import logger

from beam_abm.abm.state_schema import OUTCOME_COLUMNS

from .core import (
    GateResult,
    _abm_outcome_mean_columns,
    _is_abm_outcome_mean_column,
    _is_null_world_scenario_cfg,
    _scenario_config_for_result,
    _scenario_effect_regime,
)


def run_stochasticity_policy_gate(
    results: list,
    stationarity_epsilon: float = 0.10,
    stochastic_variance_cap: float = 0.25,
) -> GateResult:
    """Validate stochasticity policy and non-drift behavior."""
    gate = GateResult(gate_name="stochasticity_policy")

    if not results:
        gate.add_check("results_available", passed=False, details={"reason": "no simulation results"})
        gate.evaluate()
        return gate

    # Document current policy from metadata.
    deterministic_runs = [r for r in results if getattr(r, "metadata", None) is not None and r.metadata.deterministic]
    stochastic_runs = [r for r in results if getattr(r, "metadata", None) is not None and not r.metadata.deterministic]
    gate.add_check(
        "policy_documented_in_metadata",
        passed=any(getattr(r, "metadata", None) is not None for r in results),
        details={
            "n_deterministic": len(deterministic_runs),
            "n_stochastic": len(stochastic_runs),
        },
    )

    # Deterministic regime: same scenario/seed should produce identical
    # trajectories. For no-communication runs, trajectories should also
    # remain seed-invariant.
    deterministic_groups: dict[tuple[str, int], list] = {}
    for run in deterministic_runs:
        effect_regime = _scenario_effect_regime(run)
        deterministic_groups.setdefault((f"{run.scenario_name}__{effect_regime}", run.seed), []).append(run)

    for (scenario_name, seed), group in deterministic_groups.items():
        if len(group) < 2:
            continue
        cols = _abm_outcome_mean_columns(group[0].outcome_trajectory.columns)
        if not cols:
            continue
        base = group[0].outcome_trajectory.select(cols).to_numpy().astype(np.float64)
        max_diff = 0.0
        for other in group[1:]:
            arr = other.outcome_trajectory.select(cols).to_numpy().astype(np.float64)
            if arr.shape != base.shape:
                max_diff = np.inf
                break
            max_diff = max(max_diff, float(np.max(np.abs(arr - base))))
        gate.add_check(
            f"deterministic_same_seed_identical_{scenario_name}_seed{seed}",
            passed=max_diff <= 1e-12,
            details={"max_diff": max_diff},
        )

    deterministic_no_comm: dict[str, list] = {}
    for run in deterministic_runs:
        md = getattr(run, "metadata", None)
        if md is None or bool(md.communication_enabled):
            continue
        effect_regime = _scenario_effect_regime(run)
        deterministic_no_comm.setdefault(f"{run.scenario_name}__{effect_regime}", []).append(run)

    for scenario_name, group in deterministic_no_comm.items():
        if len(group) < 2:
            continue
        cols = _abm_outcome_mean_columns(group[0].outcome_trajectory.columns)
        if not cols:
            continue
        base = group[0].outcome_trajectory.select(cols).to_numpy().astype(np.float64)
        max_diff = 0.0
        for other in group[1:]:
            arr = other.outcome_trajectory.select(cols).to_numpy().astype(np.float64)
            if arr.shape != base.shape:
                max_diff = np.inf
                break
            max_diff = max(max_diff, float(np.max(np.abs(arr - base))))
        gate.add_check(
            f"deterministic_seed_invariance_no_comm_{scenario_name}",
            passed=max_diff <= 1e-12,
            details={"max_diff": max_diff, "n_runs": len(group)},
        )

    # Stochastic regime: variation across seeds should be non-zero but bounded.
    stochastic_groups: dict[str, list] = {}
    for run in stochastic_runs:
        effect_regime = _scenario_effect_regime(run)
        stochastic_groups.setdefault(f"{run.scenario_name}__{effect_regime}", []).append(run)

    if not stochastic_groups:
        gate.add_check(
            "stochastic_mode_skipped",
            passed=True,
            details={"reason": "no stochastic runs present; deterministic mode active"},
        )
    for scenario_name, group in stochastic_groups.items():
        if len(group) < 2:
            continue
        cols = _abm_outcome_mean_columns(group[0].outcome_trajectory.columns)
        if not cols:
            continue
        per_col_var: list[float] = []
        for col in cols:
            finals = [
                float(r.outcome_trajectory[col].to_numpy().astype(np.float64)[-1])
                for r in group
                if r.outcome_trajectory.height > 0
            ]
            if len(finals) < 2:
                continue
            per_col_var.append(float(np.var(finals)))
        if not per_col_var:
            continue
        mean_var = float(np.mean(per_col_var))
        gate.add_check(
            f"stochastic_variance_bounded_{scenario_name}",
            passed=(mean_var > 1e-10) and (mean_var <= stochastic_variance_cap),
            details={
                "mean_final_outcome_variance": mean_var,
                "variance_cap": stochastic_variance_cap,
                "n_runs": len(group),
            },
        )

    # Quiet scenarios should not drift materially at aggregate level.
    for result in results:
        scenario_cfg = _scenario_config_for_result(result)
        if scenario_cfg is None or not _is_null_world_scenario_cfg(scenario_cfg):
            continue
        effect_regime = _scenario_effect_regime(result)
        if result.outcome_trajectory.height == 0:
            continue
        for col in result.outcome_trajectory.columns:
            if col == "tick" or not _is_abm_outcome_mean_column(col):
                continue
            if col.endswith("_delta_vs_t0_mean") or col.endswith("_delta_vs_ref_mean"):
                continue
            vals = result.outcome_trajectory[col].to_numpy().astype(np.float64)
            if vals.size < 2:
                continue
            drift = float(abs(vals[-1] - vals[0]))
            gate.add_check(
                f"quiet_no_drift_{col}_{effect_regime}_rep{result.rep_id}",
                passed=drift <= stationarity_epsilon,
                details={"drift": drift, "epsilon": stationarity_epsilon},
            )

    # If duplicates exist, check they are not all perfectly lock-stepped.
    for result in results:
        md = getattr(result, "metadata", None)
        if md is not None and bool(md.deterministic):
            continue
        pop_final = result.population_final
        if "row_id" not in pop_final.columns:
            continue
        dup_counts = pop_final.group_by("row_id").agg(pl.len().alias("n"))
        duplicated = dup_counts.filter(pl.col("n") > 1)
        if duplicated.height == 0:
            continue

        tracked_cols = [c for c in sorted(OUTCOME_COLUMNS) if c in pop_final.columns]
        if "institutional_trust_avg" in pop_final.columns:
            tracked_cols.append("institutional_trust_avg")
        if not tracked_cols:
            continue

        merged = pop_final.join(duplicated.select("row_id"), on="row_id", how="inner")
        dispersion = (
            merged.group_by("row_id")
            .agg([pl.col(c).std().fill_null(0.0).alias(f"{c}_std") for c in tracked_cols])
            .select([pl.col(c).mean().alias(c) for c in [f"{x}_std" for x in tracked_cols]])
        )
        mean_std = float(np.mean(dispersion.to_numpy())) if dispersion.height > 0 else 0.0
        gate.add_check(
            f"duplicate_clone_dispersion_rep{result.rep_id}",
            passed=mean_std > 1e-6,
            details={"mean_clone_std": mean_std, "n_duplicate_groups": duplicated.height},
        )

    gate.evaluate()
    logger.info(
        "Stochasticity policy gate: {status} ({n_pass}/{n_total} checks).",
        status="PASSED" if gate.passed else "FAILED",
        n_pass=sum(1 for c in gate.checks if c.passed),
        n_total=len(gate.checks),
    )
    return gate
