"""Survey-relative intensity calibration utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import polars as pl

from beam_abm.abm.message_contracts import (
    norm_channel_columns,
    risk_channel_columns,
    trust_channel_columns,
)
from beam_abm.abm.state_update import SignalSpec

__all__ = [
    "AnchorLevelTable",
    "LeverAnchorLevels",
    "IntensityCalibrator",
    "build_iqr_scales",
    "build_country_iqr_scales",
    "build_std_scales",
    "build_country_std_scales",
    "load_anchor_perturbation_levels",
    "load_anchor_perturbation_scales",
    "build_anchoring_diagnostics",
    "calibrate_signal",
]

_LEVER_ALIASES: dict[str, tuple[str, ...]] = {
    # Descriptive-vax norms are mapped to the vax-avg anchor lever in PE tables.
    "social_norms_descriptive_vax": ("social_norms_vax_avg",),
}


def _candidate_lever_keys(lever: str) -> tuple[str, ...]:
    aliases = _LEVER_ALIASES.get(lever, ())
    return (lever, *aliases)


@dataclass(frozen=True, slots=True)
class LeverAnchorLevels:
    """Anchor-derived absolute levels and dispersion for one lever."""

    q25: float | None = None
    q50: float | None = None
    q75: float | None = None
    delta_x: float | None = None


@dataclass(slots=True)
class AnchorLevelTable:
    """Typed lookup table for anchor-derived lever levels."""

    global_levels: dict[str, LeverAnchorLevels]
    country_levels: dict[str, dict[str, LeverAnchorLevels]]

    def level_for(self, lever: str, country: str | None = None) -> LeverAnchorLevels | None:
        """Return country level if available, else global level."""
        keys = _candidate_lever_keys(lever)
        if country is not None:
            country_map = self.country_levels.get(country, {})
            for key in keys:
                candidate = country_map.get(key)
                if candidate is not None:
                    return candidate
        for key in keys:
            candidate = self.global_levels.get(key)
            if candidate is not None:
                return candidate
        return None


def build_iqr_scales(agents: pl.DataFrame) -> dict[str, float]:
    """Compute construct-level IQR scales from a baseline population."""
    scales: dict[str, float] = {}
    for column in agents.columns:
        try:
            arr = agents[column].cast(pl.Float64).to_numpy()
        except (pl.exceptions.InvalidOperationError, TypeError, ValueError):
            continue
        finite = arr[np.isfinite(arr)]
        if finite.size < 4:
            continue
        q25, q75 = np.percentile(finite, [25.0, 75.0])
        iqr = float(q75 - q25)
        scales[column] = iqr if iqr > 1e-8 else 0.0
    return scales


def build_std_scales(agents: pl.DataFrame) -> dict[str, float]:
    """Compute construct-level standard-deviation scales from baseline population."""
    scales: dict[str, float] = {}
    for column in agents.columns:
        try:
            arr = agents[column].cast(pl.Float64).to_numpy()
        except (pl.exceptions.InvalidOperationError, TypeError, ValueError):
            continue
        finite = arr[np.isfinite(arr)]
        if finite.size < 4:
            continue
        std = float(np.std(finite))
        scales[column] = std if std > 1e-8 else 0.0
    return scales


def _discover_shock_tables(root: Path) -> list[Path]:
    """Return mutable-engine shock-table paths under *root*."""
    candidates = [
        root,
        root / "mutable_engines" / "full",
        root / "mutable_engines" / "reduced",
        root / "full",
        root / "reduced",
    ]
    for candidate in candidates:
        if not candidate.exists() or not candidate.is_dir():
            continue
        tables = sorted(candidate.glob("*/shock_table.csv"))
        if tables:
            return tables
    return []


def _load_anchor_shock_rows(root: Path | None) -> pl.DataFrame:
    """Load normalized anchor shock rows with q25/q50/q75/delta_x."""
    if root is None:
        return pl.DataFrame()

    tables = _discover_shock_tables(root)
    if not tables:
        return pl.DataFrame()

    frames: list[pl.DataFrame] = []
    for table_path in tables:
        try:
            frame = pl.read_csv(table_path)
        except (OSError, pl.exceptions.PolarsError):
            continue
        required = {"lever", "q25", "q50", "q75", "delta_x"}
        if not required.issubset(set(frame.columns)):
            continue
        if "country" not in frame.columns:
            frame = frame.with_columns(pl.lit("ALL").alias("country"))
        frames.append(
            frame.select(
                pl.col("country").cast(pl.Utf8).alias("country"),
                pl.col("lever").cast(pl.Utf8).alias("lever"),
                pl.col("q25").cast(pl.Float64).alias("q25"),
                pl.col("q50").cast(pl.Float64).alias("q50"),
                pl.col("q75").cast(pl.Float64).alias("q75"),
                pl.col("delta_x").cast(pl.Float64).alias("delta_x"),
            )
        )

    if not frames:
        return pl.DataFrame()

    return pl.concat(frames, how="vertical").filter(
        pl.col("q25").is_finite()
        & pl.col("q50").is_finite()
        & pl.col("q75").is_finite()
        & pl.col("delta_x").is_finite()
    )


def load_anchor_perturbation_scales(
    root: Path | None,
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    """Load PE-anchored perturbation deltas from anchor shock tables.

    Parameters
    ----------
    root : Path | None
        Root under ``empirical/output/anchors/pe``.

    Returns
    -------
    tuple[dict[str, float], dict[str, dict[str, float]]]
        Global lever scales and country-conditional lever scales.
    """
    all_rows = _load_anchor_shock_rows(root)
    if all_rows.is_empty():
        return {}, {}

    per_country = (
        all_rows.group_by("country", "lever")
        .agg(pl.col("delta_x").median().alias("delta_x"))
        .with_columns(pl.col("delta_x").clip(lower_bound=0.0))
    )
    global_rows = (
        per_country.group_by("lever")
        .agg(pl.col("delta_x").median().alias("delta_x"))
        .with_columns(pl.col("delta_x").clip(lower_bound=0.0))
    )

    global_scales = {
        str(row["lever"]): float(row["delta_x"]) for row in global_rows.to_dicts() if np.isfinite(float(row["delta_x"]))
    }
    country_scales: dict[str, dict[str, float]] = {}
    for row in per_country.to_dicts():
        country = str(row["country"])
        lever = str(row["lever"])
        value = float(row["delta_x"])
        if not np.isfinite(value):
            continue
        country_scales.setdefault(country, {})[lever] = value
    return global_scales, country_scales


def load_anchor_perturbation_levels(root: Path | None) -> AnchorLevelTable:
    """Load anchor-derived absolute lever levels for perturbation worlds."""
    rows = _load_anchor_shock_rows(root)
    if rows.is_empty():
        return AnchorLevelTable(global_levels={}, country_levels={})

    per_country = (
        rows.group_by("country", "lever")
        .agg(
            [
                pl.col("q25").median().alias("q25"),
                pl.col("q50").median().alias("q50"),
                pl.col("q75").median().alias("q75"),
                pl.col("delta_x").median().alias("delta_x"),
            ]
        )
        .with_columns(pl.col("delta_x").clip(lower_bound=0.0))
    )
    global_rows = (
        per_country.group_by("lever")
        .agg(
            [
                pl.col("q25").median().alias("q25"),
                pl.col("q50").median().alias("q50"),
                pl.col("q75").median().alias("q75"),
                pl.col("delta_x").median().alias("delta_x"),
            ]
        )
        .with_columns(pl.col("delta_x").clip(lower_bound=0.0))
    )

    def _levels_from_row(row: dict[str, object]) -> LeverAnchorLevels:
        q25 = float(row["q25"]) if row.get("q25") is not None else None
        q50 = float(row["q50"]) if row.get("q50") is not None else None
        q75 = float(row["q75"]) if row.get("q75") is not None else None
        delta_x = float(row["delta_x"]) if row.get("delta_x") is not None else None
        if q25 is not None and not np.isfinite(q25):
            q25 = None
        if q50 is not None and not np.isfinite(q50):
            q50 = None
        if q75 is not None and not np.isfinite(q75):
            q75 = None
        if delta_x is not None and not np.isfinite(delta_x):
            delta_x = None
        return LeverAnchorLevels(q25=q25, q50=q50, q75=q75, delta_x=delta_x)

    global_levels = {str(row["lever"]): _levels_from_row(row) for row in global_rows.to_dicts()}
    country_levels: dict[str, dict[str, LeverAnchorLevels]] = {}
    for row in per_country.to_dicts():
        country = str(row["country"])
        lever = str(row["lever"])
        country_levels.setdefault(country, {})[lever] = _levels_from_row(row)

    return AnchorLevelTable(global_levels=global_levels, country_levels=country_levels)


def build_country_iqr_scales(
    agents: pl.DataFrame,
    *,
    country_column: str = "country",
) -> dict[str, dict[str, float]]:
    """Compute country-conditional construct-level IQR scales."""
    if country_column not in agents.columns:
        return {}

    country_scales: dict[str, dict[str, float]] = {}
    countries = agents[country_column].cast(pl.Utf8).to_numpy()
    for country in np.unique(countries):
        country_df = agents.filter(pl.col(country_column).cast(pl.Utf8) == str(country))
        country_scales[str(country)] = build_iqr_scales(country_df)
    return country_scales


def build_country_std_scales(
    agents: pl.DataFrame,
    *,
    country_column: str = "country",
) -> dict[str, dict[str, float]]:
    """Compute country-conditional construct-level standard deviations."""
    if country_column not in agents.columns:
        return {}

    country_scales: dict[str, dict[str, float]] = {}
    countries = agents[country_column].cast(pl.Utf8).to_numpy()
    for country in np.unique(countries):
        country_df = agents.filter(pl.col(country_column).cast(pl.Utf8) == str(country))
        country_scales[str(country)] = build_std_scales(country_df)
    return country_scales


def build_anchoring_diagnostics(
    agents: pl.DataFrame,
    *,
    country_column: str = "country",
) -> pl.DataFrame:
    """Build baseline parity diagnostics for norm/trust/risk channels by country."""
    channels = {
        "norm": norm_channel_columns(),
        "trust": trust_channel_columns(),
        "risk": risk_channel_columns(),
    }
    records: list[dict[str, float | str]] = []

    if country_column not in agents.columns:
        return pl.DataFrame()

    countries = agents[country_column].cast(pl.Utf8).to_numpy()
    for country in np.unique(countries):
        country_df = agents.filter(pl.col(country_column).cast(pl.Utf8) == str(country))
        for channel_name, cols in channels.items():
            for col in cols:
                if col not in country_df.columns:
                    continue
                values = country_df[col].cast(pl.Float64).to_numpy()
                finite = values[np.isfinite(values)]
                if finite.size == 0:
                    continue
                q25, q50, q75 = np.percentile(finite, [25.0, 50.0, 75.0])
                records.append(
                    {
                        "country": str(country),
                        "channel": channel_name,
                        "construct": col,
                        "mean": float(np.mean(finite)),
                        "q25": float(q25),
                        "q50": float(q50),
                        "q75": float(q75),
                        "iqr": float(q75 - q25),
                        "n": float(finite.size),
                    }
                )

    return pl.DataFrame(records) if records else pl.DataFrame()


@dataclass(slots=True)
class IntensityCalibrator:
    """Calibrate signal intensity into survey-relative units."""

    global_scales: dict[str, float]
    country_scales: dict[str, dict[str, float]] | None = None
    global_std_scales: dict[str, float] | None = None
    country_std_scales: dict[str, dict[str, float]] | None = None
    anchor_levels: AnchorLevelTable | None = None

    def resolve_anchor_world_level(
        self,
        target_column: str,
        *,
        country: str | None,
        world: Literal["low", "high"],
    ) -> float | None:
        """Resolve absolute anchor level for perturbation world."""
        if self.anchor_levels is None:
            return None
        levels = self.anchor_levels.level_for(target_column, country=country)
        if levels is None:
            return None
        value = levels.q25 if world == "low" else levels.q75
        if value is None or not np.isfinite(float(value)):
            return None
        return float(value)

    def calibrate_intensity(
        self,
        target_column: str,
        nominal_intensity: float,
        country: str | None = None,
        *,
        effect_regime: Literal["perturbation", "shock"] = "perturbation",
    ) -> float:
        if effect_regime == "shock":
            # Shock regime applies a one-sided 1-IQR move.
            global_scales = self.global_scales
            country_scales = self.country_scales
            default_scale = 0.0
        else:
            global_scales = self.global_scales
            country_scales = self.country_scales
            # Perturbation mode must not invent unit shocks when a scale is absent.
            default_scale = 0.0

        keys = _candidate_lever_keys(target_column)
        if country is not None and country_scales is not None:
            country_map = country_scales.get(country, {})
            for key in keys:
                scale = country_map.get(key)
                if scale is not None and np.isfinite(float(scale)):
                    return float(nominal_intensity) * float(max(float(scale), 0.0))
        scale = default_scale
        for key in keys:
            candidate = global_scales.get(key)
            if candidate is not None:
                scale = candidate
                break
        if not np.isfinite(float(scale)):
            scale = default_scale
        return float(nominal_intensity) * float(scale)


def calibrate_signal(
    signal: SignalSpec,
    calibrator: IntensityCalibrator,
    *,
    country: str | None = None,
    effect_regime: Literal["perturbation", "shock"] = "perturbation",
) -> SignalSpec:
    """Return a signal copy with calibrated intensity."""
    return SignalSpec(
        target_column=signal.target_column,
        intensity=calibrator.calibrate_intensity(
            signal.target_column,
            signal.intensity,
            country=country,
            effect_regime=effect_regime,
        ),
        signal_type=signal.signal_type,
        start_tick=signal.start_tick,
        end_tick=signal.end_tick,
        signal_id=signal.signal_id,
    )
