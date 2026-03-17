"""Scenario definitions with strict event-only schema."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Literal

import numpy as np
import polars as pl
from loguru import logger

from beam_abm.abm.data_contracts import BackboneModel, LeversConfig
from beam_abm.abm.outcomes import predict_outcome
from beam_abm.abm.scenarios import (
    build_mutable_latent_engine_scenario_library,
    build_upstream_signal_scenario_library,
)
from beam_abm.abm.state_schema import MUTABLE_COLUMNS
from beam_abm.abm.state_update import SignalSpec

__all__ = [
    "IncidenceSpec",
    "MUTABLE_LATENT_ENGINE_SCENARIO_LIBRARY",
    "SCENARIO_EFFECT_REGIMES",
    "ScenarioConfig",
    "SCENARIO_LIBRARY",
    "UPSTREAM_SIGNAL_SCENARIO_LIBRARY",
    "get_scenario_library",
    "get_scenario_label",
    "iter_scenario_regime_runs",
    "get_scenario",
    "scenario_to_dict",
    "load_scenarios_from_json",
    "validate_scenario_effect_mapping",
    "validate_scenario_uniqueness",
]

_SUPPORTED_KERNELS: tuple[str, ...] = ("parametric", "surrogate", "llm")
_MIN_POLICY_START_TICK = 2
_DEFAULT_BURN_IN_TICKS = 12
_STANDARD_CAMPAIGN_DURATION_MONTHS = 9
_STANDARD_PULSE_DURATION_TICKS = 1
_ALLOWED_SCENARIO_FIELDS: set[str] = {
    "name",
    "display_label",
    "family",
    "description",
    "state_signals",
    "choice_signals",
    "where",
    "who",
    "seed_rule",
    "targeting",
    "duration",
    "primary_outcomes",
    "communication_enabled",
    "peer_pull_enabled",
    "observed_norms_enabled",
    "enable_endogenous_loop",
    "incidence_mode",
    "incidence_schedule",
    "importation_pulse_delta",
    "importation_multiplier",
    "local_bump_multiplier",
    "norm_descriptive_share_w_d",
    "frozen_state_columns",
    "supported_kernels",
}
_REMOVED_FIELDS: set[str] = {
    "norm_update_gain",
    "norm_response_slope",
    "norm_observation_sample_size",
    "social_influence",
    "incidence_growth",
    "incidence_recovery",
    "incidence_vax_effect",
    "incidence_protect_effect",
    "stakes_feedback_gain",
    "trust_performance_gain",
    "importation_spec",
    "contact_sample_size",
    "construct_gains",
}
_PRIMARY_OUTCOME_ALIASES: dict[str, tuple[str, ...]] = {
    # Backward compatibility for legacy scenario JSON/config payloads.
    "vax_willingness_Y": ("vax_willingness_T12",),
}
_SIGNAL_TARGET_ALIASES_FOR_LEVERS: dict[str, tuple[str, ...]] = {
    "social_norms_descriptive_vax": ("social_norms_vax_avg",),
    "covid_perceived_danger_Y": ("covid_perceived_danger_T12",),
}

TICK_MONTHS: float = 1.0


def _months(months: float) -> int:
    return int(round(float(months) / TICK_MONTHS))


def _first_post_burn_in_tick() -> int:
    return int(_DEFAULT_BURN_IN_TICKS)


@dataclass(frozen=True, slots=True)
class IncidenceSpec:
    """Schedule entry controlling when incidence forcing components are active."""

    component: Literal[
        "all",
        "importation_floor",
        "local_bump",
        "importation_pulse_delta",
        "importation_multiplier",
    ] = "all"
    start_tick: int = 0
    end_tick: int | None = None

    def __post_init__(self) -> None:
        if self.component not in {
            "all",
            "importation_floor",
            "local_bump",
            "importation_pulse_delta",
            "importation_multiplier",
        }:
            msg = f"Unsupported incidence schedule component '{self.component}'."
            raise ValueError(msg)
        if self.start_tick < 0:
            msg = "incidence schedule start_tick must be >= 0."
            raise ValueError(msg)
        if self.end_tick is not None and self.end_tick <= self.start_tick:
            msg = "incidence schedule end_tick must be greater than start_tick (exclusive window end)."
            raise ValueError(msg)

    def is_active(self, tick: int) -> bool:
        if tick < self.start_tick:
            return False
        if self.end_tick is not None and tick >= self.end_tick:
            return False
        return True

    def matches(self, component: str) -> bool:
        return self.component == "all" or self.component == component


@dataclass(frozen=True, slots=True)
class ScenarioConfig:
    """Event-only scenario definition."""

    name: str
    display_label: str = ""
    family: str = "baseline"
    description: str = ""
    state_signals: tuple[SignalSpec, ...] = ()
    choice_signals: tuple[SignalSpec, ...] = ()
    where: str = "global"
    who: str = "all"
    seed_rule: str = "uniform"
    targeting: str = "uniform"
    duration: int = 0
    primary_outcomes: tuple[str, ...] = ()
    communication_enabled: bool = True
    peer_pull_enabled: bool = True
    observed_norms_enabled: bool = True
    enable_endogenous_loop: bool = True
    state_updates_enabled: bool = True
    incidence_mode: Literal["none", "importation"] = "importation"
    incidence_schedule: tuple[IncidenceSpec, ...] = ()
    importation_pulse_delta: float = 0.0
    importation_multiplier: float = 1.0
    local_bump_multiplier: float = 1.0
    norm_descriptive_share_w_d: float | None = None
    frozen_state_columns: tuple[str, ...] = ()
    supported_kernels: tuple[Literal["parametric", "surrogate", "llm"], ...] = ("parametric", "surrogate")

    def __post_init__(self) -> None:
        if self.targeting not in {"uniform", "targeted"}:
            msg = f"Unsupported targeting '{self.targeting}'."
            raise ValueError(msg)
        if self.duration < 0:
            msg = "duration must be >= 0."
            raise ValueError(msg)
        unsupported = sorted(set(self.supported_kernels) - set(_SUPPORTED_KERNELS))
        if unsupported:
            msg = f"Unsupported kernel(s) in supported_kernels: {unsupported}. Allowed: {_SUPPORTED_KERNELS}."
            raise ValueError(msg)
        if self.incidence_mode not in {"none", "importation"}:
            msg = f"Unsupported incidence_mode '{self.incidence_mode}'."
            raise ValueError(msg)
        if self.incidence_schedule and self.incidence_mode != "importation":
            msg = "incidence_schedule requires incidence_mode='importation'."
            raise ValueError(msg)
        if not np.isfinite(float(self.importation_pulse_delta)):
            msg = "importation_pulse_delta must be finite."
            raise ValueError(msg)
        if float(self.importation_pulse_delta) < 0.0:
            msg = "importation_pulse_delta must be >= 0."
            raise ValueError(msg)
        if self.importation_pulse_delta > 0.0 and self.incidence_mode != "importation":
            msg = "importation_pulse_delta requires incidence_mode='importation'."
            raise ValueError(msg)
        if not np.isfinite(float(self.importation_multiplier)):
            msg = "importation_multiplier must be finite."
            raise ValueError(msg)
        if float(self.importation_multiplier) < 0.0:
            msg = "importation_multiplier must be >= 0."
            raise ValueError(msg)
        if self.importation_multiplier != 1.0 and self.incidence_mode != "importation":
            msg = "importation_multiplier requires incidence_mode='importation'."
            raise ValueError(msg)
        if not np.isfinite(float(self.local_bump_multiplier)):
            msg = "local_bump_multiplier must be finite."
            raise ValueError(msg)
        if float(self.local_bump_multiplier) < 0.0:
            msg = "local_bump_multiplier must be >= 0."
            raise ValueError(msg)
        if self.local_bump_multiplier != 1.0 and self.incidence_mode != "importation":
            msg = "local_bump_multiplier requires incidence_mode='importation'."
            raise ValueError(msg)
        if self.norm_descriptive_share_w_d is not None:
            if not np.isfinite(float(self.norm_descriptive_share_w_d)):
                msg = "norm_descriptive_share_w_d must be finite when provided."
                raise ValueError(msg)
            if not (0.0 <= float(self.norm_descriptive_share_w_d) <= 1.0):
                msg = "norm_descriptive_share_w_d must be in [0, 1]."
                raise ValueError(msg)
        invalid_frozen = sorted(set(self.frozen_state_columns) - MUTABLE_COLUMNS)
        if invalid_frozen:
            msg = f"frozen_state_columns contains non-mutable columns: {invalid_frozen}."
            raise ValueError(msg)
        first_post_burn = _first_post_burn_in_tick()

        def _shift_signal(sig: SignalSpec) -> SignalSpec:
            start_tick = int(sig.start_tick)
            shifted_start = max(_MIN_POLICY_START_TICK, first_post_burn, start_tick)
            shifted_end = sig.end_tick
            if shifted_end is not None:
                duration = max(int(shifted_end) - start_tick, 1)
                shifted_end = shifted_start + duration
            normalized = replace(sig, start_tick=shifted_start, end_tick=shifted_end)
            if normalized.signal_type == "campaign":
                normalized = replace(
                    normalized,
                    end_tick=int(normalized.start_tick) + _months(_STANDARD_CAMPAIGN_DURATION_MONTHS),
                )
            if normalized.signal_type == "pulse":
                normalized = replace(
                    normalized,
                    end_tick=int(normalized.start_tick) + _STANDARD_PULSE_DURATION_TICKS,
                )
            return normalized

        def _shift_incidence(spec: IncidenceSpec) -> IncidenceSpec:
            start_tick = int(spec.start_tick)
            shifted_start = max(_MIN_POLICY_START_TICK, first_post_burn, start_tick)
            if shifted_start == start_tick:
                return spec
            shifted_end = spec.end_tick
            if shifted_end is not None:
                duration = max(int(shifted_end) - start_tick, 1)
                shifted_end = shifted_start + duration
            return replace(spec, start_tick=shifted_start, end_tick=shifted_end)

        state_signals = tuple(_shift_signal(sig) for sig in self.state_signals)
        choice_signals = tuple(_shift_signal(sig) for sig in self.choice_signals)
        incidence_schedule = tuple(_shift_incidence(spec) for spec in self.incidence_schedule)
        signal_types = {sig.signal_type for sig in (*state_signals, *choice_signals)}
        duration = int(self.duration)
        if "campaign" in signal_types:
            duration = _months(_STANDARD_CAMPAIGN_DURATION_MONTHS)
        elif "pulse" in signal_types:
            duration = _STANDARD_PULSE_DURATION_TICKS
        object.__setattr__(self, "state_signals", state_signals)
        object.__setattr__(self, "choice_signals", choice_signals)
        object.__setattr__(self, "incidence_schedule", incidence_schedule)
        object.__setattr__(self, "duration", duration)


def _build_scenario_library() -> dict[str, ScenarioConfig]:
    """Construct canonical scenario suite only."""
    return build_upstream_signal_scenario_library(
        ScenarioConfig=ScenarioConfig,
        SignalSpec=SignalSpec,
        IncidenceSpec=IncidenceSpec,
        months=_months,
    )


def _build_mutable_latent_engine_scenario_library() -> dict[str, ScenarioConfig]:
    """Construct PE-style scenarios that directly move mutable latent engines."""
    return build_mutable_latent_engine_scenario_library(
        ScenarioConfig=ScenarioConfig,
        SignalSpec=SignalSpec,
    )


def _signal_key(signal: SignalSpec) -> tuple[object, ...]:
    return (
        signal.target_column,
        float(signal.intensity),
        signal.signal_type,
        int(signal.start_tick),
        int(signal.end_tick) if signal.end_tick is not None else None,
        signal.signal_id,
    )


def _scenario_fingerprint(scenario: ScenarioConfig) -> tuple[object, ...]:
    return (
        tuple(sorted(_signal_key(sig) for sig in scenario.state_signals)),
        tuple(sorted(_signal_key(sig) for sig in scenario.choice_signals)),
        scenario.where,
        scenario.who,
        scenario.seed_rule,
        scenario.targeting,
        int(scenario.duration),
        bool(scenario.communication_enabled),
        bool(scenario.peer_pull_enabled),
        bool(scenario.observed_norms_enabled),
        bool(scenario.enable_endogenous_loop),
        scenario.incidence_mode,
        tuple(
            sorted(
                (
                    spec.component,
                    int(spec.start_tick),
                    int(spec.end_tick) if spec.end_tick is not None else None,
                )
                for spec in scenario.incidence_schedule
            )
        ),
        float(scenario.importation_pulse_delta),
        float(scenario.importation_multiplier),
        float(scenario.local_bump_multiplier),
        (float(scenario.norm_descriptive_share_w_d) if scenario.norm_descriptive_share_w_d is not None else None),
        tuple(sorted(scenario.frozen_state_columns)),
        tuple(sorted(scenario.supported_kernels)),
    )


def validate_scenario_uniqueness(scenarios: dict[str, ScenarioConfig] | list[ScenarioConfig]) -> None:
    """Fail if two scenarios are config-identical intervention specs."""
    keyed: list[tuple[str, ScenarioConfig]]
    if isinstance(scenarios, dict):
        keyed = list(scenarios.items())
    else:
        keyed = [(scenario.name, scenario) for scenario in scenarios]

    seen: dict[tuple[object, ...], str] = {}
    duplicates: list[tuple[str, str]] = []
    for key, scenario in keyed:
        fp = _scenario_fingerprint(scenario)
        first = seen.get(fp)
        if first is not None:
            duplicates.append((first, key))
        else:
            seen[fp] = key

    if duplicates:
        pairs = ", ".join(f"'{a}' == '{b}'" for a, b in duplicates)
        msg = f"Duplicate scenarios detected (identical config/signals/toggles): {pairs}"
        raise ValueError(msg)


def _chapter6_like_delta(values: np.ndarray) -> float:
    clean = values[np.isfinite(values)]
    if clean.size == 0:
        return 0.25
    q25, q75 = np.percentile(clean, [25.0, 75.0])
    delta = float(q75 - q25)
    if not np.isfinite(delta) or delta <= 0.0:
        q10, q90 = np.percentile(clean, [10.0, 90.0])
        delta = float(q90 - q10)
    if not np.isfinite(delta) or delta <= 0.0:
        delta = float(max(np.std(clean), 0.25) * 0.5)
    return float(max(delta, 0.05))


def _predict_mean_outcome(
    agents: pl.DataFrame,
    models: dict[str, BackboneModel],
    outcome: str,
) -> float:
    pred = predict_outcome(
        agents,
        models[outcome],
        deterministic=True,
        rng=np.random.default_rng(42),
    )
    return float(np.mean(pred))


def _lever_has_meaningful_effect(
    *,
    agents: pl.DataFrame | None,
    models: dict[str, BackboneModel],
    outcome: str,
    lever: str,
    coeff_abs_threshold: float,
    pe_ref_abs_threshold: float,
) -> bool:
    model = models[outcome]
    if lever not in model.terms:
        return False
    idx = list(model.terms).index(lever)
    coeff = float(model.coefficients[idx])
    if abs(coeff) >= coeff_abs_threshold:
        return True
    if agents is None or lever not in agents.columns:
        return False

    delta = _chapter6_like_delta(agents[lever].to_numpy().astype(np.float64))
    try:
        base_mean = _predict_mean_outcome(agents, models, outcome)
        perturbed = agents.with_columns((pl.col(lever) + delta).alias(lever))
        pert_mean = _predict_mean_outcome(perturbed, models, outcome)
    except (KeyError, ValueError) as exc:
        logger.debug(
            "Skipping PE_ref check for outcome='{outcome}' lever='{lever}': {error}",
            outcome=outcome,
            lever=lever,
            error=str(exc),
        )
        return False
    pe_ref = pert_mean - base_mean
    return abs(pe_ref) >= pe_ref_abs_threshold


def validate_scenario_effect_mapping(
    scenarios: dict[str, ScenarioConfig] | list[ScenarioConfig],
    *,
    levers_cfg: LeversConfig,
    models: dict[str, BackboneModel],
    agents_for_pe_ref: pl.DataFrame | None = None,
    coeff_abs_threshold: float = 0.05,
    pe_ref_abs_threshold: float = 0.01,
) -> None:
    """Fail if explicitly-outcome-scoped scenarios lack mapped meaningful levers.

    ``primary_outcomes`` is optional metadata. If provided, this validator is
    strict for those outcomes. If omitted, validation falls back to a best-effort
    auto mode and does not fail on empty/weak direct B-model lever overlap.
    """
    keyed: list[tuple[str, ScenarioConfig]]
    if isinstance(scenarios, dict):
        keyed = list(scenarios.items())
    else:
        keyed = [(scenario.name, scenario) for scenario in scenarios]

    def _resolve_primary_outcome_alias(outcome: str) -> str:
        if outcome in levers_cfg.outcomes:
            return outcome
        for alias in _PRIMARY_OUTCOME_ALIASES.get(outcome, ()):
            if alias in levers_cfg.outcomes:
                return alias
        return outcome

    def _expand_signal_targets_for_levers(signal_targets: set[str]) -> set[str]:
        expanded = set(signal_targets)
        for target in signal_targets:
            expanded.update(_SIGNAL_TARGET_ALIASES_FOR_LEVERS.get(target, ()))
        return expanded

    errors: list[str] = []
    for key, scenario in keyed:
        signal_targets = {sig.target_column for sig in (*scenario.state_signals, *scenario.choice_signals)}
        mapped_targets = _expand_signal_targets_for_levers(signal_targets)
        if not signal_targets:
            continue
        strict_outcome_scope = bool(scenario.primary_outcomes)
        scoped_outcomes = scenario.primary_outcomes
        if not strict_outcome_scope:
            scoped_outcomes = tuple(sorted(outcome for outcome in levers_cfg.outcomes if outcome in models))
            if not scoped_outcomes:
                logger.warning(
                    "Scenario '{name}' has signals but no loaded outcome models for auto validation; skipping.",
                    name=key,
                )
                continue

        for outcome in scoped_outcomes:
            resolved_outcome = _resolve_primary_outcome_alias(outcome)
            if strict_outcome_scope and outcome not in levers_cfg.outcomes:
                if resolved_outcome == outcome:
                    errors.append(f"{key}: primary outcome '{outcome}' not found in levers config")
                    continue
            if strict_outcome_scope and resolved_outcome not in levers_cfg.outcomes:
                errors.append(f"{key}: primary outcome '{outcome}' not found in levers config")
                continue
            if resolved_outcome not in levers_cfg.outcomes:
                continue
            if resolved_outcome not in models:
                if not strict_outcome_scope:
                    continue
                errors.append(f"{key}: primary outcome '{outcome}' model is not loaded")
                continue
            candidate_levers = [
                lever for lever in levers_cfg.outcomes[resolved_outcome].levers if lever in mapped_targets
            ]
            if not candidate_levers:
                if signal_targets and all(target not in MUTABLE_COLUMNS for target in signal_targets):
                    # Precursor-only scenarios are valid under the X->S*->Y
                    # contract and cannot be validated by direct lever
                    # intersection alone.
                    continue
                if strict_outcome_scope:
                    errors.append(
                        f"{key}: no signal target intersects levers for primary outcome '{outcome}' "
                        f"(signals={sorted(signal_targets)})"
                    )
                continue
            meaningful = any(
                _lever_has_meaningful_effect(
                    agents=agents_for_pe_ref,
                    models=models,
                    outcome=resolved_outcome,
                    lever=lever,
                    coeff_abs_threshold=coeff_abs_threshold,
                    pe_ref_abs_threshold=pe_ref_abs_threshold,
                )
                for lever in candidate_levers
            )
            if not meaningful:
                if strict_outcome_scope:
                    errors.append(
                        f"{key}: no mapped lever has |beta| >= {coeff_abs_threshold:.3f} "
                        f"or |PE_ref| >= {pe_ref_abs_threshold:.3f} for primary outcome '{outcome}'"
                    )

    if errors:
        msg = "Scenario-to-lever mapping failed:\n- " + "\n- ".join(errors)
        raise ValueError(msg)


UPSTREAM_SIGNAL_SCENARIO_LIBRARY: dict[str, ScenarioConfig] = _build_scenario_library()
MUTABLE_LATENT_ENGINE_SCENARIO_LIBRARY: dict[str, ScenarioConfig] = _build_mutable_latent_engine_scenario_library()
SCENARIO_EFFECT_REGIMES: tuple[Literal["perturbation", "shock"], ...] = ("perturbation", "shock")

_overlap = sorted(set(UPSTREAM_SIGNAL_SCENARIO_LIBRARY) & set(MUTABLE_LATENT_ENGINE_SCENARIO_LIBRARY))
if _overlap:
    overlap_msg = f"Scenario libraries contain duplicate keys: {_overlap}"
    raise ValueError(overlap_msg)

_ALL_SCENARIO_LIBRARY: dict[str, ScenarioConfig] = {
    **UPSTREAM_SIGNAL_SCENARIO_LIBRARY,
    **MUTABLE_LATENT_ENGINE_SCENARIO_LIBRARY,
}
SCENARIO_LIBRARY: dict[str, ScenarioConfig] = _ALL_SCENARIO_LIBRARY

validate_scenario_uniqueness(UPSTREAM_SIGNAL_SCENARIO_LIBRARY)
validate_scenario_uniqueness(MUTABLE_LATENT_ENGINE_SCENARIO_LIBRARY)
validate_scenario_uniqueness(_ALL_SCENARIO_LIBRARY)


def get_scenario_library(*, include_mutable: bool = True) -> dict[str, ScenarioConfig]:
    """Return scenario registry for runtime selection."""
    if include_mutable:
        return _ALL_SCENARIO_LIBRARY
    return UPSTREAM_SIGNAL_SCENARIO_LIBRARY


def get_scenario_label(name: str) -> str:
    """Return a human-readable label for a scenario key."""
    if name in _ALL_SCENARIO_LIBRARY:
        scenario = _ALL_SCENARIO_LIBRARY[name]
        if scenario.display_label.strip():
            return scenario.display_label
        return scenario.name.replace("_", " ").title()
    return name.replace("_", " ").title()


def iter_scenario_regime_runs(
    scenario_keys: tuple[str, ...] | list[str],
    *,
    effect_regimes: tuple[Literal["perturbation", "shock"], ...] = SCENARIO_EFFECT_REGIMES,
) -> tuple[tuple[str, Literal["perturbation", "shock"]], ...]:
    """Expand scenario keys into `(scenario_key, effect_regime)` run tuples."""
    runs: list[tuple[str, Literal["perturbation", "shock"]]] = []
    for scenario_key in scenario_keys:
        for effect_regime in effect_regimes:
            runs.append((scenario_key, effect_regime))
    return tuple(runs)


def get_scenario(name: str) -> ScenarioConfig:
    if name not in _ALL_SCENARIO_LIBRARY:
        msg = f"Scenario '{name}' not found. Available: {sorted(_ALL_SCENARIO_LIBRARY)}"
        raise KeyError(msg)
    return _ALL_SCENARIO_LIBRARY[name]


def _validate_spec_keys(name: str, spec: dict[str, object]) -> None:
    unknown = set(spec.keys()) - _ALLOWED_SCENARIO_FIELDS
    removed = sorted(k for k in unknown if k in _REMOVED_FIELDS)
    unsupported = sorted(k for k in unknown if k not in _REMOVED_FIELDS)
    if removed:
        msg = (
            f"Scenario '{name}' uses removed field(s): {removed}. "
            "This refactor is strict and does not support legacy schema."
        )
        raise ValueError(msg)
    if unsupported:
        msg = f"Scenario '{name}' has unknown field(s): {unsupported}. Allowed: {sorted(_ALLOWED_SCENARIO_FIELDS)}"
        raise ValueError(msg)


def load_scenarios_from_json(path: Path) -> list[ScenarioConfig]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        msg = "Scenario JSON must be a top-level object keyed by scenario name."
        raise ValueError(msg)

    parsed: list[ScenarioConfig] = []
    for name, maybe_spec in raw.items():
        if not isinstance(maybe_spec, dict):
            msg = f"Scenario '{name}' must map to an object."
            raise ValueError(msg)
        spec = dict(maybe_spec)
        _validate_spec_keys(name, spec)
        state_signals = tuple(
            SignalSpec(**{k: v for k, v in sig.items() if k != "decay_rate"}) for sig in spec.get("state_signals", [])
        )
        choice_signals = tuple(
            SignalSpec(**{k: v for k, v in sig.items() if k != "decay_rate"}) for sig in spec.get("choice_signals", [])
        )
        incidence_schedule = tuple(IncidenceSpec(**row) for row in spec.get("incidence_schedule", []))
        parsed.append(
            ScenarioConfig(
                name=str(spec.get("name", name)),
                display_label=str(
                    spec.get(
                        "display_label",
                        str(spec.get("name", name)).replace("_", " ").title(),
                    )
                ),
                family=str(spec.get("family", "custom")),
                description=str(spec.get("description", "")),
                state_signals=state_signals,
                choice_signals=choice_signals,
                where=str(spec.get("where", "global")),
                who=str(spec.get("who", "all")),
                seed_rule=str(spec.get("seed_rule", "uniform")),
                targeting=str(spec.get("targeting", "uniform")),
                duration=int(spec.get("duration", 0)),
                primary_outcomes=tuple(spec.get("primary_outcomes", ())),
                communication_enabled=bool(spec.get("communication_enabled", True)),
                peer_pull_enabled=bool(spec.get("peer_pull_enabled", True)),
                observed_norms_enabled=bool(spec.get("observed_norms_enabled", True)),
                enable_endogenous_loop=bool(spec.get("enable_endogenous_loop", True)),
                incidence_mode=str(spec.get("incidence_mode", "none")),
                incidence_schedule=incidence_schedule,
                importation_pulse_delta=float(spec.get("importation_pulse_delta", 0.0)),
                importation_multiplier=float(spec.get("importation_multiplier", 1.0)),
                local_bump_multiplier=float(spec.get("local_bump_multiplier", 1.0)),
                norm_descriptive_share_w_d=(
                    float(spec["norm_descriptive_share_w_d"])
                    if spec.get("norm_descriptive_share_w_d") is not None
                    else None
                ),
                frozen_state_columns=tuple(spec.get("frozen_state_columns", ())),
                supported_kernels=tuple(spec.get("supported_kernels", ("parametric", "surrogate"))),
            )
        )

    validate_scenario_uniqueness(parsed)
    logger.info("Loaded {n} scenario(s) from {p}.", n=len(parsed), p=path)
    return parsed


def _signal_to_dict(signal: SignalSpec) -> dict[str, object]:
    payload = asdict(signal)
    return {
        "target_column": payload["target_column"],
        "intensity": float(payload["intensity"]),
        "signal_type": payload["signal_type"],
        "start_tick": int(payload["start_tick"]),
        "end_tick": int(payload["end_tick"]) if payload["end_tick"] is not None else None,
        "signal_id": payload["signal_id"],
    }


def _incidence_to_dict(spec: IncidenceSpec) -> dict[str, object]:
    return {
        "component": spec.component,
        "start_tick": int(spec.start_tick),
        "end_tick": int(spec.end_tick) if spec.end_tick is not None else None,
    }


def scenario_to_dict(scenario: ScenarioConfig, *, scenario_key: str | None = None) -> dict[str, object]:
    key = scenario_key
    if key is None:
        for candidate_key, candidate in _ALL_SCENARIO_LIBRARY.items():
            if candidate is scenario:
                key = candidate_key
                break
    if key is None:
        key = "custom"

    return {
        "scenario_key": key,
        "name": scenario.name,
        "display_label": scenario.display_label,
        "family": scenario.family,
        "description": scenario.description,
        "state_signals": [_signal_to_dict(sig) for sig in scenario.state_signals],
        "choice_signals": [_signal_to_dict(sig) for sig in scenario.choice_signals],
        "where": scenario.where,
        "who": scenario.who,
        "seed_rule": scenario.seed_rule,
        "targeting": scenario.targeting,
        "duration": int(scenario.duration),
        "primary_outcomes": list(scenario.primary_outcomes),
        "communication_enabled": bool(scenario.communication_enabled),
        "peer_pull_enabled": bool(scenario.peer_pull_enabled),
        "observed_norms_enabled": bool(scenario.observed_norms_enabled),
        "enable_endogenous_loop": bool(scenario.enable_endogenous_loop),
        "incidence_mode": scenario.incidence_mode,
        "incidence_schedule": [_incidence_to_dict(spec) for spec in scenario.incidence_schedule],
        "importation_pulse_delta": float(scenario.importation_pulse_delta),
        "importation_multiplier": float(scenario.importation_multiplier),
        "local_bump_multiplier": float(scenario.local_bump_multiplier),
        "norm_descriptive_share_w_d": (
            float(scenario.norm_descriptive_share_w_d) if scenario.norm_descriptive_share_w_d is not None else None
        ),
        "frozen_state_columns": list(scenario.frozen_state_columns),
        "supported_kernels": list(scenario.supported_kernels),
    }
