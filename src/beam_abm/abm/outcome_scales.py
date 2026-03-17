"""Shared outcome scale metadata and normalization helpers."""

from __future__ import annotations

import numpy as np

__all__ = [
    "OUTCOME_BOUNDS",
    "normalize_outcome_to_unit_interval",
]


OUTCOME_BOUNDS: dict[str, tuple[float, float]] = {
    "vax_willingness_T12": (1.0, 5.0),
    "mask_when_symptomatic_crowded": (1.0, 7.0),
    "stay_home_when_symptomatic": (1.0, 7.0),
    "mask_when_pressure_high": (1.0, 7.0),
}


def normalize_outcome_to_unit_interval(values: np.ndarray, outcome: str) -> np.ndarray:
    """Normalize an outcome array to ``[0, 1]`` using declared bounds."""
    bounds = OUTCOME_BOUNDS.get(outcome)
    if bounds is None:
        msg = f"No outcome bounds configured for '{outcome}'."
        raise ValueError(msg)
    lo, hi = bounds
    if hi <= lo:
        msg = f"Invalid outcome bounds for '{outcome}': ({lo}, {hi})"
        raise ValueError(msg)
    clipped = np.nan_to_num(values.astype(np.float64), nan=lo, posinf=hi, neginf=lo)
    return np.clip((clipped - lo) / (hi - lo), 0.0, 1.0)
