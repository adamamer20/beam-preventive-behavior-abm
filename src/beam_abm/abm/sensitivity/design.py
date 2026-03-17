"""Design-generation helpers for sensitivity analysis."""

from __future__ import annotations

import numpy as np
import polars as pl
from scipy.stats import qmc

from .core import INTEGER_PARAMETERS

INCIDENCE_NET_GROWTH_MIN: float = -0.8
INCIDENCE_NET_GROWTH_MAX: float = 3.3


def _incidence_constraint_active(param_names: list[str]) -> bool:
    return "k_incidence" in param_names and "incidence_gamma" in param_names


def _incidence_net_growth_valid(k_incidence: float, incidence_gamma: float) -> bool:
    net = k_incidence - incidence_gamma
    return INCIDENCE_NET_GROWTH_MIN <= net <= INCIDENCE_NET_GROWTH_MAX


def _relaxation_constraint_active(param_names: list[str]) -> bool:
    return "state_relaxation_fast" in param_names and "state_relaxation_slow" in param_names


def _relaxation_order_valid(state_relaxation_fast: float, state_relaxation_slow: float) -> bool:
    return state_relaxation_slow <= state_relaxation_fast


def _point_respects_design_constraints(
    *,
    normalized_point: np.ndarray,
    param_names: list[str],
    parameter_bounds: dict[str, tuple[float, float]],
) -> bool:
    """Check all active design constraints for a normalized design point."""
    if _incidence_constraint_active(param_names):
        k_idx = param_names.index("k_incidence")
        g_idx = param_names.index("incidence_gamma")
        k_lo, k_hi = parameter_bounds["k_incidence"]
        g_lo, g_hi = parameter_bounds["incidence_gamma"]

        k_val = k_lo + float(normalized_point[k_idx]) * (k_hi - k_lo)
        g_val = g_lo + float(normalized_point[g_idx]) * (g_hi - g_lo)
        if not _incidence_net_growth_valid(k_val, g_val):
            return False

    if _relaxation_constraint_active(param_names):
        fast_idx = param_names.index("state_relaxation_fast")
        slow_idx = param_names.index("state_relaxation_slow")
        fast_lo, fast_hi = parameter_bounds["state_relaxation_fast"]
        slow_lo, slow_hi = parameter_bounds["state_relaxation_slow"]
        fast_val = fast_lo + float(normalized_point[fast_idx]) * (fast_hi - fast_lo)
        slow_val = slow_lo + float(normalized_point[slow_idx]) * (slow_hi - slow_lo)
        if not _relaxation_order_valid(fast_val, slow_val):
            return False

    return True


def _trajectory_respects_design_constraints(
    *,
    normalized_points: list[np.ndarray],
    param_names: list[str],
    parameter_bounds: dict[str, tuple[float, float]],
) -> bool:
    """Check all active design constraints for trajectory points."""
    for point in normalized_points:
        if not _point_respects_design_constraints(
            normalized_point=point,
            param_names=param_names,
            parameter_bounds=parameter_bounds,
        ):
            return False
    return True


def _coerce_param_value(name: str, value: float) -> float | int:
    if name in INTEGER_PARAMETERS:
        return int(round(value))
    return float(value)


def build_morris_design(
    *,
    parameter_bounds: dict[str, tuple[float, float]],
    n_trajectories: int,
    n_levels: int,
    seed: int,
) -> pl.DataFrame:
    """Create a single Morris trajectory design over all parameters."""
    rng = np.random.default_rng(seed)
    param_names = list(parameter_bounds)
    n_params = len(param_names)
    if n_params == 0:
        raise ValueError("parameter_bounds must not be empty.")

    level_grid = np.linspace(0.0, 1.0, n_levels)
    step = 1.0 / float(n_levels - 1)
    valid_start = level_grid[level_grid <= (1.0 - step + 1e-12)]
    if valid_start.size == 0:
        raise ValueError("Invalid Morris setup: no feasible start points.")

    rows: list[dict[str, float | int | str | None]] = []
    point_id = 0
    for traj_idx in range(n_trajectories):
        normalized_points: list[np.ndarray] | None = None
        order: np.ndarray | None = None
        max_attempts = 2_000
        for _ in range(max_attempts):
            start = rng.choice(valid_start, size=n_params, replace=True).astype(float)
            candidate_order = rng.permutation(n_params)

            candidate_points: list[np.ndarray] = [start.copy()]
            current = start.copy()
            for param_index in candidate_order:
                current = current.copy()
                current[param_index] = min(1.0, current[param_index] + step)
                candidate_points.append(current)

            if _trajectory_respects_design_constraints(
                normalized_points=candidate_points,
                param_names=param_names,
                parameter_bounds=parameter_bounds,
            ):
                normalized_points = candidate_points
                order = candidate_order
                break
        if normalized_points is None or order is None:
            msg = "Unable to sample Morris trajectory satisfying incidence constraints."
            raise ValueError(msg)

        prev_point_id: int | None = None
        for step_index, normalized in enumerate(normalized_points):
            row: dict[str, float | int | str | None] = {
                "point_id": point_id,
                "trajectory_id": traj_idx,
                "step_index": step_index,
                "prev_point_id": prev_point_id,
                "changed_param": None,
                "step_size": None,
            }
            if step_index > 0:
                changed_index = int(order[step_index - 1])
                changed_name = param_names[changed_index]
                lower, upper = parameter_bounds[changed_name]
                actual_step = step * (upper - lower)
                if changed_name in INTEGER_PARAMETERS:
                    actual_step = float(max(1, int(round(actual_step))))
                row["changed_param"] = changed_name
                row["step_size"] = actual_step

            for idx, param_name in enumerate(param_names):
                lower, upper = parameter_bounds[param_name]
                value = lower + normalized[idx] * (upper - lower)
                row[param_name] = _coerce_param_value(param_name, value)

            rows.append(row)
            prev_point_id = point_id
            point_id += 1

    return pl.DataFrame(rows)


def build_lhc_design(
    *,
    parameter_bounds: dict[str, tuple[float, float]],
    n_samples: int,
    seed: int,
) -> pl.DataFrame:
    """Create an LHC design over selected parameters."""
    param_names = list(parameter_bounds)
    if not param_names:
        raise ValueError("parameter_bounds must not be empty for LHC.")

    engine = qmc.LatinHypercube(d=len(param_names), seed=seed)
    unit_samples = engine.random(n=n_samples)
    constrained_indices: set[int] = set()
    if _incidence_constraint_active(param_names):
        constrained_indices.update(
            {
                param_names.index("k_incidence"),
                param_names.index("incidence_gamma"),
            }
        )
    if _relaxation_constraint_active(param_names):
        constrained_indices.update(
            {
                param_names.index("state_relaxation_fast"),
                param_names.index("state_relaxation_slow"),
            }
        )

    if constrained_indices:
        rng = np.random.default_rng(seed + 13_579)
        max_attempts = 10_000
        attempts = 0
        while True:
            invalid_indices = np.array(
                [
                    idx
                    for idx in range(n_samples)
                    if not _point_respects_design_constraints(
                        normalized_point=unit_samples[idx, :],
                        param_names=param_names,
                        parameter_bounds=parameter_bounds,
                    )
                ],
                dtype=np.int64,
            )
            if invalid_indices.size == 0:
                break
            if attempts >= max_attempts:
                msg = "Unable to sample LHC points satisfying design constraints."
                raise ValueError(msg)
            for col_idx in sorted(constrained_indices):
                unit_samples[invalid_indices, col_idx] = rng.random(size=invalid_indices.size)
            attempts += 1

    rows: list[dict[str, float | int]] = []
    for point_id in range(n_samples):
        row: dict[str, float | int] = {"point_id": point_id}
        for index, param_name in enumerate(param_names):
            lower, upper = parameter_bounds[param_name]
            value = lower + float(unit_samples[point_id, index]) * (upper - lower)
            row[param_name] = _coerce_param_value(param_name, value)
        rows.append(row)
    return pl.DataFrame(rows)
