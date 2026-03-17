"""Incidence-related helper functions for the ABM engine."""

from __future__ import annotations

import numpy as np


def _importation_pulse_at_tick(
    *,
    tick: int,
    amplitude: float,
    period_ticks: int,
    width_ticks: int,
    start_tick: int,
) -> float:
    if amplitude <= 0.0 or period_ticks <= 0 or width_ticks <= 0:
        return 0.0
    if tick < start_tick:
        return 0.0
    phase = (tick - start_tick) % period_ticks
    if phase >= width_ticks:
        return 0.0
    return float(amplitude)


def _resolve_vax_outcome_key(outcomes: dict[str, np.ndarray]) -> str | None:
    if "vax_willingness_T12" in outcomes:
        return "vax_willingness_T12"
    return None
