"""Population-shift validation benchmarking gate."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
from loguru import logger
from scipy.spatial import KDTree

from beam_abm.abm.data_contracts import AnchorSystem, BackboneModel, LeversConfig
from beam_abm.abm.state_schema import MUTABLE_COLUMNS

if TYPE_CHECKING:
    from .core import GateResult


def run_population_shift_validation_gate(
    populations: dict[str, pl.DataFrame],
    survey_df: pl.DataFrame,
    models: dict[str, BackboneModel],
    levers_cfg: LeversConfig,
    anchor_system: AnchorSystem,
    countries: list[str],
    model_coefficients_path: Path,
    off_manifold_distance_threshold: float | None = None,
    off_manifold_share_threshold: float = 0.10,
) -> GateResult:
    """Validate synthetic/counterfactual populations with baseline suite."""
    from .core import GateResult, run_baseline_replication_gate

    gate = GateResult(gate_name="population_shift_validation")
    if not populations:
        gate.add_check("population_specs_present", passed=False, details={"reason": "no populations provided"})
        gate.evaluate()
        return gate

    baseline_pop = populations.get("baseline")
    if baseline_pop is None:
        first_key = sorted(populations)[0]
        baseline_pop = populations[first_key]

    for name, pop in populations.items():
        baseline_check = run_baseline_replication_gate(
            agents_t0=pop,
            survey_df=survey_df,
            models=models,
            levers_cfg=levers_cfg,
            anchor_system=anchor_system,
            countries=countries,
            model_coefficients_path=model_coefficients_path,
        )
        gate.add_check(
            f"extended_baseline_{name}",
            passed=baseline_check.passed,
            details={
                "n_checks": len(baseline_check.checks),
                "n_passed": sum(1 for c in baseline_check.checks if c.passed),
            },
        )

        if off_manifold_distance_threshold is not None:
            engine_cols = sorted(col for col in MUTABLE_COLUMNS if col in pop.columns and col in survey_df.columns)
            if engine_cols:
                pop_mat = np.nan_to_num(pop.select(engine_cols).to_numpy().astype(np.float64), nan=0.0)
                survey_mat = np.nan_to_num(survey_df.select(engine_cols).to_numpy().astype(np.float64), nan=0.0)
                if survey_mat.shape[0] > 0 and pop_mat.shape[0] > 0:
                    tree = KDTree(survey_mat)
                    nn_dist, _ = tree.query(pop_mat, k=1)
                    nn_dist = nn_dist.astype(np.float64)
                    exceed_share = float(np.mean(nn_dist > off_manifold_distance_threshold))
                    gate.add_check(
                        f"off_manifold_guardrail_{name}",
                        passed=exceed_share <= off_manifold_share_threshold,
                        details={
                            "distance_threshold": off_manifold_distance_threshold,
                            "share_threshold": off_manifold_share_threshold,
                            "share_exceeding_threshold": exceed_share,
                            "nn_distance_min": float(np.min(nn_dist)),
                            "nn_distance_median": float(np.median(nn_dist)),
                        },
                    )

        if name == "baseline" or "anchor_id" not in pop.columns or "anchor_id" not in baseline_pop.columns:
            continue

        shifted = (
            pop.group_by("anchor_id")
            .agg((pl.len() / pop.height).alias("share"))
            .join(
                baseline_pop.group_by("anchor_id").agg((pl.len() / baseline_pop.height).alias("share_baseline")),
                on="anchor_id",
                how="outer",
            )
        ).fill_null(0.0)
        max_shift = float((shifted["share"] - shifted["share_baseline"]).abs().max())
        gate.add_check(
            f"population_shift_detectable_{name}",
            passed=max_shift > 0.01,
            details={"max_anchor_share_shift": max_shift},
        )

    gate.evaluate()
    logger.info(
        "Population-shift validation gate: {status} ({n_pass}/{n_total} checks).",
        status="PASSED" if gate.passed else "FAILED",
        n_pass=sum(1 for c in gate.checks if c.passed),
        n_total=len(gate.checks),
    )
    return gate
