"""Shared configuration and utilities for sensitivity pipeline modules."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from beam_abm.abm.config import ABMImplementationConstants, ABMUserKnobs, SimulationConfig
from beam_abm.abm.scenario_defs import SCENARIO_EFFECT_REGIMES, SCENARIO_LIBRARY

SENSITIVITY_LATENT_BOUNDS: dict[str, tuple[float, float]] = {
    "social_weight_delta_norm": (-1.5, 1.5),
    "social_weight_delta_trust": (-1.5, 1.5),
}

SENSITIVITY_EXCLUDED_PARAMETERS: frozenset[str] = frozenset(
    {
        "social_weight_norm",
        "social_weight_trust",
        "social_weight_risk",
        "social_substeps_per_month",
        "norm_observation_min_per_month",
        "norm_observation_max_per_month",
        "saturation_atol",
        "contact_rate_base",
    }
)


def _eligible_sa_parameter_names() -> list[str]:
    names: list[str] = []
    for source in (ABMUserKnobs.model_fields.keys(), ABMImplementationConstants.model_fields.keys()):
        for name in source:
            if name in SENSITIVITY_EXCLUDED_PARAMETERS:
                continue
            names.append(name)
    names.extend(SENSITIVITY_LATENT_BOUNDS.keys())
    return names


def _extract_bounds_from_schema(*, name: str, property_schema: dict[str, Any]) -> tuple[float, float]:
    recommended_range = property_schema.get("recommended_range")
    if (
        isinstance(recommended_range, list)
        and len(recommended_range) == 2
        and recommended_range[0] is not None
        and recommended_range[1] is not None
    ):
        lower = float(recommended_range[0])
        upper = float(recommended_range[1])
        if np.isfinite(lower) and np.isfinite(upper) and lower < upper:
            return (lower, upper)

    lower_value = property_schema.get("minimum")
    if lower_value is None and property_schema.get("exclusiveMinimum") is not None:
        lower_value = float(property_schema["exclusiveMinimum"]) + float(np.finfo(float).eps)

    upper_value = property_schema.get("maximum")
    if upper_value is None and property_schema.get("exclusiveMaximum") is not None:
        upper_value = float(property_schema["exclusiveMaximum"]) - float(np.finfo(float).eps)

    if lower_value is None or upper_value is None:
        msg = f"No finite sensitivity bounds found for parameter '{name}'."
        raise ValueError(msg)

    lower = float(lower_value)
    upper = float(upper_value)
    if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
        msg = f"Invalid sensitivity bounds for parameter '{name}': ({lower}, {upper})."
        raise ValueError(msg)
    return (lower, upper)


def _build_sa_bounds_and_types() -> tuple[dict[str, tuple[float, float]], frozenset[str]]:
    knob_schema = ABMUserKnobs.model_json_schema().get("properties", {})
    constant_schema = ABMImplementationConstants.model_json_schema().get("properties", {})

    bounds: dict[str, tuple[float, float]] = {}
    integer_parameters: set[str] = set()

    for name in _eligible_sa_parameter_names():
        if name in SENSITIVITY_LATENT_BOUNDS:
            bounds[name] = SENSITIVITY_LATENT_BOUNDS[name]
            continue

        property_schema = knob_schema.get(name)
        if property_schema is None:
            property_schema = constant_schema.get(name)
        if property_schema is None:
            msg = f"Parameter '{name}' not found in ABM schema properties."
            raise ValueError(msg)

        bounds[name] = _extract_bounds_from_schema(name=name, property_schema=property_schema)
        if property_schema.get("type") == "integer":
            integer_parameters.add(name)

    return bounds, frozenset(integer_parameters)


def _build_five_point_sweep_values(
    bounds: dict[str, tuple[float, float]],
    integer_parameters: frozenset[str],
) -> dict[str, tuple[float | int, ...]]:
    sweep_values: dict[str, tuple[float | int, ...]] = {}
    for name, (lower, upper) in bounds.items():
        values = np.linspace(lower, upper, num=5)
        if name in integer_parameters:
            rounded = sorted({int(round(v)) for v in values} | {int(round(lower)), int(round(upper))})
            sweep_values[name] = tuple(rounded)
        else:
            sweep_values[name] = tuple(float(v) for v in values)
    return sweep_values


DEFAULT_PARAMETER_BOUNDS, INTEGER_PARAMETERS = _build_sa_bounds_and_types()
FULL_PARAMETER_SWEEP_VALUES = _build_five_point_sweep_values(DEFAULT_PARAMETER_BOUNDS, INTEGER_PARAMETERS)

DEFAULT_OUTPUT_METRICS: tuple[str, ...] = (
    "vax_willingness_T12_mean",
    "mask_when_pressure_high_mean",
    "social_norms_descriptive_vax_mean",
)

DEFAULT_FLAGSHIP_SCENARIOS: tuple[str, ...] = (
    "outbreak_global_incidence_strong",
    "outbreak_salience_global",
    "misinfo_shock",
    "norm_campaign",
    "access_intervention",
    "trust_repair_legitimacy",
    "trust_scandal_legitimacy",
    "trust_repair_institutional",
)

BACKGROUND_SCENARIOS: tuple[str, ...] = (
    "baseline",
    "baseline_importation",
    "baseline_no_communication",
)


def _scenario_run_label(scenario_name: str, effect_regime: Literal["perturbation", "shock"]) -> str:
    return f"{scenario_name}__{effect_regime}"


def _parse_scenario_run_label(label: str) -> tuple[str, Literal["perturbation", "shock"]]:
    for effect_regime in SCENARIO_EFFECT_REGIMES:
        suffix = f"__{effect_regime}"
        if label.endswith(suffix):
            return label[: -len(suffix)], effect_regime
    return label, "perturbation"


PARAMETER_DEFAULTS: dict[str, float | int] = {
    **ABMUserKnobs().model_dump(),
    **ABMImplementationConstants().model_dump(),
    "social_weight_delta_norm": 0.0,
    "social_weight_delta_trust": 0.0,
}


def _social_weights_from_latent_deltas(delta_norm: float, delta_trust: float) -> tuple[float, float, float]:
    baseline = np.array(
        [
            float(PARAMETER_DEFAULTS["social_weight_norm"]),
            float(PARAMETER_DEFAULTS["social_weight_trust"]),
            float(PARAMETER_DEFAULTS["social_weight_risk"]),
        ],
        dtype=np.float64,
    )
    logits = np.log(np.clip(baseline, 1e-12, 1.0))
    logits[0] += float(delta_norm)
    logits[1] += float(delta_trust)
    logits -= float(np.max(logits))
    probs = np.exp(logits)
    probs /= float(np.sum(probs))
    return float(probs[0]), float(probs[1]), float(probs[2])


@dataclass(slots=True)
class SensitivityPipelineConfig:
    """Configuration for the full ABM sensitivity pipeline."""

    output_dir: Path = Path("abm/output/full_sensitivity_pipeline")
    n_agents: int = 2_000
    n_steps: int = 24
    seed_offsets: tuple[int, ...] = (0, 1, 2)
    base_seed: int = 42
    morris_trajectories: int = 8
    morris_levels: int = 6
    lhc_samples: int = 120
    sobol_base_samples: int = 4_096
    shortlist_target: int = 8
    shortlist_max: int = 10
    shortlist_sigma_quantile: float = 0.8
    surrogate_min_r2: float = 0.30
    outputs: tuple[str, ...] = DEFAULT_OUTPUT_METRICS
    flagship_scenarios: tuple[str, ...] = DEFAULT_FLAGSHIP_SCENARIOS
    effect_metric: Literal["final", "auc"] = "auc"
    persist_run_data: bool = True
    max_workers: int = 0
    resume_from_morris_raw: bool = False

    def __post_init__(self) -> None:
        self.output_dir = Path(self.output_dir)
        if self.n_agents <= 0:
            raise ValueError("n_agents must be positive.")
        if self.n_steps <= 0:
            raise ValueError("n_steps must be positive.")
        if self.morris_trajectories <= 0:
            raise ValueError("morris_trajectories must be positive.")
        if self.morris_levels < 3:
            raise ValueError("morris_levels must be >= 3.")
        if self.lhc_samples <= 0:
            raise ValueError("lhc_samples must be positive.")
        if self.sobol_base_samples < 32:
            raise ValueError("sobol_base_samples must be >= 32.")
        if not (0.0 < self.shortlist_sigma_quantile <= 1.0):
            raise ValueError("shortlist_sigma_quantile must be in (0, 1].")
        if self.shortlist_target <= 0:
            raise ValueError("shortlist_target must be positive.")
        if self.shortlist_max < self.shortlist_target:
            raise ValueError("shortlist_max must be >= shortlist_target.")
        if not self.outputs:
            raise ValueError("outputs must not be empty.")
        if self.effect_metric not in {"final", "auc"}:
            raise ValueError("effect_metric must be one of {'final', 'auc'}.")
        if self.max_workers < 0:
            raise ValueError("max_workers must be >= 0.")


def _ensure_output_dirs(root: Path) -> dict[str, Path]:
    """Create all pipeline output directories."""
    paths = {
        "morris": root / "morris",
        "shortlist": root / "shortlist",
        "lhc": root / "lhc",
        "surrogates": root / "surrogates",
        "sobol": root / "sobol",
        "reports": root / "reports",
        "figures": root / "figures",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def _base_config(config: SensitivityPipelineConfig) -> SimulationConfig:
    return SimulationConfig(
        n_agents=config.n_agents,
        n_steps=config.n_steps,
        deterministic=True,
    )


def _create_config_from_params(
    base_cfg: SimulationConfig,
    params: dict[str, float | int],
    *,
    seed: int,
) -> SimulationConfig:
    """Build a ``SimulationConfig`` from parameter values and seed."""
    knob_params = {key: params.get(key, PARAMETER_DEFAULTS[key]) for key in ABMUserKnobs.model_fields.keys()}
    constant_params = {
        key: params.get(key, PARAMETER_DEFAULTS[key]) for key in ABMImplementationConstants.model_fields.keys()
    }

    delta_norm = float(params.get("social_weight_delta_norm", PARAMETER_DEFAULTS["social_weight_delta_norm"]))
    delta_trust = float(params.get("social_weight_delta_trust", PARAMETER_DEFAULTS["social_weight_delta_trust"]))
    social_weight_norm, social_weight_trust, social_weight_risk = _social_weights_from_latent_deltas(
        delta_norm,
        delta_trust,
    )
    constant_params.update(
        {
            "social_weight_norm": social_weight_norm,
            "social_weight_trust": social_weight_trust,
            "social_weight_risk": social_weight_risk,
        }
    )

    knobs = ABMUserKnobs(**knob_params)
    constants = ABMImplementationConstants(**constant_params)
    return base_cfg.model_copy(
        update={
            "knobs": knobs,
            "constants": constants,
            "seed": seed,
        }
    )


def _selection_metric_column(*, output_name: str, effect_metric: Literal["final", "auc"]) -> str:
    if effect_metric == "auc":
        return f"{output_name}__auc"
    return output_name


def _reference_baseline(scenario_name: str) -> str:
    """Return the baseline scenario name for delta construction."""
    base_scenario, effect_regime = _parse_scenario_run_label(scenario_name)
    scenario = SCENARIO_LIBRARY[base_scenario]
    if scenario.incidence_mode == "importation":
        return _scenario_run_label("baseline_importation", effect_regime)
    return _scenario_run_label("baseline", effect_regime)


def _seed_for_point(base_seed: int, point_id: int, seed_offset: int) -> int:
    """Stable common-random-number seed mapping."""
    return int(base_seed + point_id * 10_000 + seed_offset)
