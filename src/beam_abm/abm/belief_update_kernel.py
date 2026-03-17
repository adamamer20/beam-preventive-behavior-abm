"""LLM-backed belief-update kernel integration.

This module provides a lightweight adapter that maps validated
belief-update signals to bounded construct updates. It is designed
as an optional drop-in replacement for parametric state updates.
"""

from __future__ import annotations

import fnmatch
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import polars as pl
from loguru import logger

from beam_abm.abm.state_schema import CONSTRUCT_BOUNDS

__all__ = [
    "BeliefUpdateKernelConfig",
    "BeliefUpdateKernel",
]

Direction = Literal["up", "down", "none"]


@dataclass(slots=True)
class BeliefUpdateKernelConfig:
    """Configuration for the belief-update kernel.

    Parameters
    ----------
    targets_path : Path
        Path to belief-update targets JSON.
    intensity_scale : float
        Scale applied to signal intensity when updating constructs.
    """

    targets_path: Path
    intensity_scale: float = 0.3


@dataclass(slots=True)
class BeliefUpdateKernel:
    """Kernel mapping validated belief-update signals to construct deltas."""

    signal_effects: dict[str, dict[str, tuple[Direction, float]]]
    intensity_scale: float = 0.3

    @classmethod
    def from_targets_file(cls, path: Path, *, intensity_scale: float = 0.3) -> BeliefUpdateKernel:
        """Load a kernel mapping from a belief-update targets JSON file."""
        raw = json.loads(path.read_text(encoding="utf-8"))
        signals = _extract_signal_entries(raw)
        effects_map: dict[str, dict[str, tuple[Direction, float]]] = {}
        for entry in signals:
            signal_id = entry.get("id") or entry.get("signal_id")
            effects = entry.get("effects") or entry.get("intended_footprint") or {}
            if not signal_id or not isinstance(effects, dict):
                continue
            parsed: dict[str, tuple[Direction, float]] = {}
            for col, effect_spec in effects.items():
                direction_norm, magnitude = _parse_effect_spec(effect_spec)
                if direction_norm is None:
                    continue
                for resolved_col in _expand_columns(str(col)):
                    parsed[resolved_col] = (direction_norm, magnitude)
            if parsed:
                effects_map[str(signal_id)] = parsed
        logger.info("Loaded belief-update kernel: {n} signals from {p}.", n=len(effects_map), p=path)
        return cls(signal_effects=effects_map, intensity_scale=float(intensity_scale))

    def apply(
        self,
        agents: pl.DataFrame,
        *,
        signal_id: str,
        intensity: float,
        target_mask: np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        """Compute updated arrays for a belief-update signal.

        Parameters
        ----------
        agents : pl.DataFrame
            Agent state to update.
        signal_id : str
            Belief-update signal identifier.
        intensity : float
            Signal intensity at the current tick.
        target_mask : np.ndarray or None
            Optional boolean mask for targeted application.

        Returns
        -------
        dict[str, np.ndarray]
            Updated arrays for affected columns.
        """
        effects = self.signal_effects.get(signal_id)
        if not effects or intensity == 0.0:
            return {}

        updates: dict[str, np.ndarray] = {}
        for col, (direction, magnitude) in effects.items():
            if col not in agents.columns:
                continue
            delta_sign = 0.0
            if direction == "up":
                delta_sign = 1.0
            elif direction == "down":
                delta_sign = -1.0

            if delta_sign == 0.0:
                continue

            current = agents[col].to_numpy().astype(np.float64)
            updated = current.copy()
            delta = intensity * self.intensity_scale * float(magnitude) * delta_sign
            if target_mask is None:
                updated += delta
            else:
                updated[target_mask] = updated[target_mask] + delta

            lo, hi = CONSTRUCT_BOUNDS.get(col, (-np.inf, np.inf))
            updated = np.clip(updated, lo, hi)
            updates[col] = updated

        return updates


def _extract_signal_entries(raw: object) -> list[dict[str, object]]:
    if isinstance(raw, dict):
        direct_signals = raw.get("signals")
        if isinstance(direct_signals, list):
            return [entry for entry in direct_signals if isinstance(entry, dict)]

        entries: list[dict[str, object]] = []

        def _collect(block: object) -> None:
            if not isinstance(block, dict):
                return
            for signal_id, payload in block.items():
                if not isinstance(payload, dict):
                    continue
                if "effects" in payload or "intended_footprint" in payload:
                    entry = dict(payload)
                    entry.setdefault("id", str(signal_id))
                    entries.append(entry)

        for grouped_key in ("dual_mode_signals", "llm_only_signals"):
            _collect(raw.get(grouped_key))

        if entries:
            return entries

        _collect(raw)
        return entries
    return []


def _parse_direction(value: object) -> Direction | None:
    if not isinstance(value, str):
        return None
    lowered = value.strip().lower()
    if lowered.startswith("up"):
        return "up"
    if lowered.startswith("down"):
        return "down"
    if lowered in {"none", "neutral", "no change", "ambiguous"}:
        return "none"
    return None


def _parse_effect_spec(value: object) -> tuple[Direction | None, float]:
    if isinstance(value, (int, float)):
        numeric = float(value)
        if numeric > 0:
            return "up", abs(numeric)
        if numeric < 0:
            return "down", abs(numeric)
        return "none", 0.0

    if isinstance(value, dict):
        direction = _parse_direction(value.get("direction"))
        if direction is None:
            signed = value.get("signed_magnitude")
            if isinstance(signed, (int, float)):
                return _parse_effect_spec(float(signed))
            return None, 0.0
        magnitude_raw = value.get("magnitude", value.get("weight", 1.0))
        magnitude = float(magnitude_raw) if isinstance(magnitude_raw, (int, float)) else 1.0
        return direction, max(0.0, magnitude)

    if isinstance(value, str):
        direction = _parse_direction(value)
        return direction, 1.0

    return None, 0.0


def _expand_columns(pattern: str) -> tuple[str, ...]:
    if "*" not in pattern:
        return (pattern,)

    matched = tuple(col for col in CONSTRUCT_BOUNDS if fnmatch.fnmatch(col, pattern))
    if matched:
        return matched
    return ()
