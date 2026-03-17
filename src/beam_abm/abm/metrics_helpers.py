"""Shared helper utilities for ABM metrics computations."""

from __future__ import annotations

import numpy as np


def _first_present_name(candidates: tuple[str, ...], available: object) -> str | None:
    available_set = set(available)  # type: ignore[arg-type]
    for name in candidates:
        if name in available_set:
            return name
    return None


def _resolve_threshold(
    outcome_name: str,
    values: np.ndarray,
    thresholds: dict[str, float],
) -> float:
    if outcome_name in thresholds:
        return thresholds[outcome_name]
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0
    if float(np.nanmin(finite)) >= 0.0 and float(np.nanmax(finite)) <= 1.0:
        return 0.5
    return float(np.nanmedian(finite))


def _nan_corr(left: np.ndarray, right: np.ndarray) -> float:
    mask = np.isfinite(left) & np.isfinite(right)
    if np.count_nonzero(mask) < 3:
        return float("nan")
    left_valid = left[mask]
    right_valid = right[mask]
    if np.std(left_valid) == 0.0 or np.std(right_valid) == 0.0:
        return float("nan")
    return float(np.corrcoef(left_valid, right_valid)[0, 1])


def _safe_float(value: object) -> float:
    if value is None:
        return 0.0
    return float(value)
