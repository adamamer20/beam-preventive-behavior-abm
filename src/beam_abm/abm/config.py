"""Typed simulation configuration and environment loading.

Centralises every tunable knob the ABM needs.  Values are read, in
priority order, from:

1. Explicit constructor arguments.
2. Environment variables prefixed with ``ABM_`` (e.g. ``ABM_N_AGENTS=5000``).
3. Hard-coded defaults that match the thesis design.

All filesystem paths are resolved relative to the auto-detected project
root so the config works identically inside containers and on bare metal.

Examples
--------
>>> cfg = SimulationConfig()
>>> cfg.n_agents
10000
>>> cfg.survey_csv.exists()
True
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, ClassVar, Literal

from loguru import logger
from pydantic import BaseModel, Field, model_validator

__all__ = [
    "ABMImplementationConstants",
    "ABMUserKnobs",
    "SimulationConfig",
    "get_project_root",
]

Provenance = Literal["empirical", "survey_estimate", "calibrated", "design"]


def _field_with_meta(
    *,
    default: Any,
    description: str,
    provenance: Provenance,
    reference: str,
    rec_min: float | int | None = None,
    rec_max: float | int | None = None,
    ge: float | int | None = None,
    le: float | int | None = None,
    gt: float | int | None = None,
    lt: float | int | None = None,
) -> Any:
    recommended_range: list[float | int] | None = None
    if rec_min is not None and rec_max is not None:
        recommended_range = [rec_min, rec_max]
    kwargs: dict[str, Any] = {
        "description": description,
        "json_schema_extra": {
            "recommended_range": recommended_range,
            "provenance": provenance,
            "reference": reference,
        },
    }
    if ge is not None:
        kwargs["ge"] = ge
    if le is not None:
        kwargs["le"] = le
    if gt is not None:
        kwargs["gt"] = gt
    if lt is not None:
        kwargs["lt"] = lt
    return Field(default=default, **kwargs)


class ABMUserKnobs(BaseModel):
    """User-facing behavioural knobs only.

    Keep this set small and interpretable; implementation constants live in
    :class:`ABMImplementationConstants`.
    """

    k_social: float = _field_with_meta(
        default=0.3,
        ge=0.0,
        le=2.0,
        rec_min=0.25,
        rec_max=0.8,
        provenance="survey_estimate",
        reference=(
            "Monthly convergence-timescale anchor from the model's DeGroot-style social update "
            "with small-effect social-norm evidence (Implementation Science 2021; Nature Human "
            "Behaviour 2025)."
        ),
        description=(
            "Collapsed social feedback magnitude (monthly tick; recommended policy envelope "
            "maps to ~3-8 month social half-life with near-equal channel weights)."
        ),
    )
    k_incidence: float = _field_with_meta(
        default=1.26,
        ge=0.0,
        le=4.0,
        rec_min=0.4,
        rec_max=3.8,
        provenance="calibrated",
        reference=(
            "Calibrated to reproduce realistic monthly wave growth under observed-survey state "
            "dynamics, informed by OWID case trajectory envelopes."
        ),
        description="Incidence growth intensity (monthly tick).",
    )
    incidence_gamma: float = _field_with_meta(
        default=0.58,
        ge=0.0,
        le=1.0,
        rec_min=0.25,
        rec_max=0.85,
        provenance="calibrated",
        reference=(
            "Calibrated monthly memory/decay term to match observed wave persistence without "
            "runaway incidence, using OWID dynamics as an external anchor."
        ),
        description="Incidence decay/memory parameter (monthly tick).",
    )
    k_feedback_stakes: float = _field_with_meta(
        default=1.35,
        ge=0.0,
        le=2.0,
        rec_min=0.2,
        rec_max=1.6,
        provenance="calibrated",
        reference=(
            "Calibrated effective incidence→stakes elasticity, informed by survey evidence linking "
            "incidence to perceived danger/fear and behaviour."
        ),
        description="Stakes/risk feedback strength from incidence level.",
    )
    contact_rate_multiplier: float = _field_with_meta(
        default=1.0,
        ge=0.0,
        rec_min=0.2,
        rec_max=1.5,
        provenance="empirical",
        reference=("POLYMOD pre-pandemic contact baseline with CoMix-era reductions and post-pandemic variation."),
        description="Multiplier on the survey-anchored contact-rate base (relative mixing).",
    )


class ABMImplementationConstants(BaseModel):
    """Internal implementation constants and diagnostics tolerances."""

    model_config: ClassVar[dict] = {"extra": "forbid", "frozen": False, "validate_default": True}  # type: ignore[misc]

    social_weight_norm: float = Field(default=0.34, ge=0.0, le=1.0)
    social_weight_trust: float = Field(default=0.33, ge=0.0, le=1.0)
    social_weight_risk: float = Field(default=0.33, ge=0.0, le=1.0)

    homophily_gamma: float = _field_with_meta(
        default=1.5,
        ge=0.0,
        le=10.0,
        rec_min=0.5,
        rec_max=3.0,
        provenance="calibrated",
        reference="Homophily as a robust social-network property (McPherson et al., 2001).",
        description="Homophily distance-decay strength (calibrated).",
    )
    bridge_prob: float = _field_with_meta(
        default=0.08,
        ge=0.0,
        le=1.0,
        rec_min=0.01,
        rec_max=0.15,
        provenance="calibrated",
        reference=(
            "Small-world rewiring literature (Watts-Strogatz class; small bridging probabilities "
            "produce short paths with clustered structure)."
        ),
        description="Probability of cross-locality bridging.",
    )
    contact_rate_base: float = Field(
        default=1.0,
        ge=0.0,
        description="Survey-anchored base informative exposure rate (expected weekly social exposures per substep).",
    )
    social_substeps_per_month: int = Field(
        default=4,
        ge=1,
        le=8,
        description="Fast social-learning substeps executed within each monthly tick.",
    )
    peer_pull_interactions_per_agent: float = Field(
        default=0.3,
        ge=0.05,
        le=8.0,
        description=(
            "Expected pairwise peer-pull interactions per agent per social substep for symmetric gossip updates."
        ),
    )
    norm_observation_min_per_month: int = Field(
        default=2,
        ge=1,
        le=64,
        description="Minimum per-agent monthly norm-observation opportunities.",
    )
    norm_observation_max_per_month: int = Field(
        default=10,
        ge=1,
        le=128,
        description="Maximum per-agent monthly norm-observation opportunities.",
    )
    # Monthly-tick rescale of wave rho=0.49 using T_wave=8 months:
    # rho_month = 1 - exp(-(-ln(0.51)/8)*1) ~= 0.081.
    state_relaxation_fast: float = Field(
        default=0.081,
        ge=0.0,
        le=1.0,
        description="Fast relaxation rate toward target state (monthly tick).",
    )
    state_relaxation_slow: float = Field(
        default=0.025,
        ge=0.0,
        le=1.0,
        description="Slow relaxation rate for persistent constructs.",
    )
    norm_response_slope: float = Field(
        default=8.0,
        gt=0.0,
        le=25.0,
        description="Observed descriptive-norm response slope.",
    )
    norm_threshold_beta_alpha: float = Field(
        default=2.0,
        gt=0.0,
        le=20.0,
        description="Symmetric Beta(alpha, alpha) shape for norm thresholds.",
    )

    norm_threshold_min: float = Field(default=0.05, ge=0.0, le=1.0)
    norm_threshold_max: float = Field(default=0.95, ge=0.0, le=1.0)
    state_update_noise_std: float = Field(default=0.006, ge=0.0, le=0.04)
    community_incidence_baseline: float = Field(default=0.05, ge=0.0, le=1.0)

    high_degree_target_percentile: float = Field(default=75.0, gt=0.0, le=100.0)
    high_exposure_target_percentile: float = Field(default=75.0, gt=0.0, le=100.0)
    low_legitimacy_target_percentile: float = Field(default=25.0, gt=0.0, le=100.0)
    saturation_atol: float = Field(default=1e-6, ge=0.0)
    distance_tracking_max_hops: int = Field(default=6, ge=0, le=10)

    incidence_alpha_vax: float = _field_with_meta(
        default=0.42,
        ge=0.0,
        le=1.0,
        rec_min=0.15,
        rec_max=0.65,
        provenance="empirical",
        reference="Meta-analytic/umbrella evidence for Omicron-era vaccine effectiveness against infection.",
        description="Infection-risk suppression multiplier from vaccination status.",
    )
    incidence_alpha_protection: float = _field_with_meta(
        default=0.52,
        ge=0.0,
        le=1.0,
        rec_min=0.10,
        rec_max=0.60,
        provenance="empirical",
        reference="Systematic evidence on incidence reduction from bundled NPIs (masking/distancing/hygiene).",
        description="Infection-risk suppression multiplier from protective behaviour.",
    )
    importation_outbreak_floor: float = Field(default=0.0005, ge=0.0, le=1.0)
    importation_targeted_local_bump: float = Field(default=0.0, ge=0.0, le=1.0)
    importation_pulse_amplitude: float = Field(default=0.006, ge=0.0, le=1.0)
    importation_pulse_period_ticks: int = _field_with_meta(
        default=12,
        ge=0,
        rec_min=6,
        rec_max=18,
        provenance="design",
        reference="OWID COVID-19 monthly incidence dynamics used to justify wave-like periodicity.",
        description="Importation pulse recurrence period in monthly ticks (when pulse mode is enabled).",
    )
    importation_pulse_width_ticks: int = _field_with_meta(
        default=2,
        ge=0,
        rec_min=1,
        rec_max=3,
        provenance="design",
        reference="OWID COVID-19 monthly incidence dynamics used to justify short importation shocks.",
        description="Importation pulse width in monthly ticks (when pulse mode is enabled).",
    )
    importation_pulse_start_tick: int = Field(default=0, ge=0, le=12)

    burn_in_ticks: int = Field(
        default=12,
        ge=0,
        le=24,
        description=(
            "Number of initial ticks during which importation forcing is "
            "suppressed. Peer-pull and observed-norms still run so agents "
            "can reach social quasi-equilibrium before external shocks; "
            "during this window, observed-share anchors are recalibrated "
            "from model-implied exposures."
        ),
    )

    stakes_to_fear_multiplier: float = _field_with_meta(
        default=1.15,
        ge=0.0,
        le=1.4,
        rec_min=0.4,
        rec_max=1.3,
        provenance="calibrated",
        reference=(
            "Calibrated effective mapping from stakes to fear construct to preserve positive "
            "incidence→fear elasticity while keeping closure-pulse responses sub-SD."
        ),
        description="Multiplier mapping stakes offset to perceived danger/fear update channels.",
    )
    stakes_to_infection_multiplier: float = _field_with_meta(
        default=0.25,
        ge=0.0,
        le=1.6,
        rec_min=0.1,
        rec_max=0.9,
        provenance="calibrated",
        reference=(
            "Calibrated effective mapping from stakes to perceived infection-likelihood channel to "
            "avoid over-counting risk updates across constructs."
        ),
        description="Multiplier mapping stakes offset to perceived infection-likelihood channel.",
    )

    @model_validator(mode="after")
    def _validate_social_weights(self) -> ABMImplementationConstants:
        total = self.social_weight_norm + self.social_weight_trust + self.social_weight_risk
        if total <= 0.0:
            msg = "social channel weights must sum to a positive value."
            raise ValueError(msg)
        if self.norm_observation_max_per_month < self.norm_observation_min_per_month:
            msg = "norm_observation_max_per_month must be >= norm_observation_min_per_month."
            raise ValueError(msg)
        self.social_weight_norm = float(self.social_weight_norm / total)
        self.social_weight_trust = float(self.social_weight_trust / total)
        self.social_weight_risk = float(self.social_weight_risk / total)
        return self


_DECLARED_USER_KNOBS: tuple[str, ...] = (
    "k_social",
    "k_incidence",
    "incidence_gamma",
    "k_feedback_stakes",
    "contact_rate_multiplier",
)

_DECLARED_IMPLEMENTATION_CONSTANTS: tuple[str, ...] = (
    "social_weight_norm",
    "social_weight_trust",
    "social_weight_risk",
    "homophily_gamma",
    "bridge_prob",
    "contact_rate_base",
    "social_substeps_per_month",
    "peer_pull_interactions_per_agent",
    "norm_observation_min_per_month",
    "norm_observation_max_per_month",
    "state_relaxation_fast",
    "state_relaxation_slow",
    "norm_response_slope",
    "norm_threshold_beta_alpha",
    "norm_threshold_min",
    "norm_threshold_max",
    "state_update_noise_std",
    "community_incidence_baseline",
    "high_degree_target_percentile",
    "high_exposure_target_percentile",
    "low_legitimacy_target_percentile",
    "saturation_atol",
    "distance_tracking_max_hops",
    "incidence_alpha_vax",
    "incidence_alpha_protection",
    "importation_outbreak_floor",
    "importation_targeted_local_bump",
    "importation_pulse_amplitude",
    "importation_pulse_period_ticks",
    "importation_pulse_width_ticks",
    "importation_pulse_start_tick",
    "burn_in_ticks",
    "stakes_to_fear_multiplier",
    "stakes_to_infection_multiplier",
)


def _validate_declared_parameters_are_used(project_root: Path) -> None:
    """Ensure every declared knob/constant is referenced by runtime code."""
    runtime_files: list[Path] = [
        project_root / "src" / "beam_abm" / "abm" / "engine.py",
        project_root / "src" / "beam_abm" / "abm" / "network.py",
        project_root / "src" / "beam_abm" / "abm" / "runtime.py",
        project_root / "src" / "beam_abm" / "abm" / "state_update.py",
        project_root / "src" / "beam_abm" / "abm" / "benchmarking.py",
    ]
    engine_pkg_dir = project_root / "src" / "beam_abm" / "abm" / "engine"
    if engine_pkg_dir.is_dir():
        runtime_files.extend(sorted(engine_pkg_dir.glob("*.py")))
    content = "\n".join(path.read_text(encoding="utf-8") for path in runtime_files if path.exists())

    missing_knobs = [
        name for name in _DECLARED_USER_KNOBS if re.search(rf"\b(?:cfg|m\.cfg)\.knobs\.{name}\b", content) is None
    ]
    missing_constants = [
        name
        for name in _DECLARED_IMPLEMENTATION_CONSTANTS
        if re.search(rf"\b(?:cfg|m\.cfg)\.constants\.{name}\b", content) is None
    ]
    if missing_knobs or missing_constants:
        msg = (
            "Declared ABM parameters are not used by runtime code; remove or wire them. "
            f"missing_knobs={missing_knobs}, missing_constants={missing_constants}"
        )
        raise ValueError(msg)


# ---------------------------------------------------------------------------
# Project root detection
# ---------------------------------------------------------------------------


def get_project_root() -> Path:
    """Walk up from this file until we find ``pyproject.toml``.

    Returns
    -------
    Path
        Absolute path to the repository root.

    Raises
    ------
    FileNotFoundError
        If no ``pyproject.toml`` is found in any ancestor directory.
    """
    current = Path(__file__).resolve().parent
    for parent in (current, *current.parents):
        if (parent / "pyproject.toml").is_file():
            return parent
    msg = "Could not locate project root (no pyproject.toml found)."
    raise FileNotFoundError(msg)


_PROJECT_ROOT: Path = get_project_root()


# ---------------------------------------------------------------------------
# Configuration model
# ---------------------------------------------------------------------------


class SimulationConfig(BaseModel):
    """Top-level configuration for one simulation batch.

    Parameters
    ----------
    n_agents : int
        Number of synthetic agents to initialise per replication.
    n_steps : int
        Simulation horizon in discrete ticks (monthly granularity).
    n_reps : int
        Independent replications for uncertainty bands.
    seed : int
        Base random seed; each replication offsets by its index.
    deterministic : bool
        If *True*, force single-threaded execution for exact
        reproducibility at the cost of speed.
    knobs : ABMUserKnobs
        User-facing behavioural controls.
    constants : ABMImplementationConstants
        Internal constants and safety tolerances.
    belief_update_kernel_path : Path or None
    debug_trace: bool
        Enable module-level delta tracing.
        Optional path to a belief-update targets JSON file.
    belief_update_surrogate_dir : Path or None
        Optional directory containing a trained surrogate bundle.
    belief_update_intensity_scale : float
        Scale applied when translating signals to belief updates.
    output_dir : Path
        Directory for simulation artefacts (created if absent).
    levers_path : Path
        JSON file mapping each outcome to its mutable predictors.
    modeling_dir : Path
        Directory containing fitted model coefficient CSVs.
    anchors_dir : Path
        Directory with per-country anchor-profile Parquet files.
    survey_csv : Path
        Clean survey micro-data used for population initialisation.
    countries : list[str]
        ISO-2 country codes included in the simulation.
    """

    model_config: ClassVar[dict] = {  # type: ignore[misc]
        "env_prefix": "ABM_",
        "env_nested_delimiter": "__",
        "extra": "forbid",
        "frozen": False,
        "validate_default": True,
    }
    _usage_validated: ClassVar[bool] = False

    # -- scale parameters ----------------------------------------------------
    n_agents: int = Field(
        default=10_000,
        gt=0,
        description="Number of agents per replication.",
    )
    n_steps: int = Field(
        default=52,
        gt=0,
        description="Simulation ticks (monthly).",
    )
    n_reps: int = Field(
        default=10,
        gt=0,
        description="Independent replications.",
    )
    seed: int = Field(
        default=42,
        description="Base RNG seed.",
    )
    deterministic: bool = Field(
        default=False,
        description="Force single-threaded execution.",
    )

    knobs: ABMUserKnobs = ABMUserKnobs()
    constants: ABMImplementationConstants = ABMImplementationConstants()
    debug_trace: bool = Field(default=False, description="Enable module-level delta tracing.")
    belief_update_kernel_path: Path | None = Field(
        default=None,
        description="Optional belief-update targets JSON for LLM kernel.",
    )
    belief_update_surrogate_dir: Path | None = Field(
        default=None,
        description="Optional directory for trained belief-update surrogate bundle.",
    )
    belief_update_intensity_scale: float = Field(
        default=0.3,
        ge=0.0,
        description="Signal intensity scale for belief-update kernel.",
    )
    state_ref_spec_path: Path | None = Field(
        default_factory=lambda: _PROJECT_ROOT / "evaluation" / "specs" / "belief_update_targets.json",
        description="Optional state-reference spec with reduced P-model mappings.",
    )
    state_ref_strict: bool = Field(
        default=False,
        description="Fail fast when state-reference models/predictors are missing.",
    )
    intensity_calibration_mode: Literal["global", "country"] = Field(
        default="global",
        description="Signal intensity calibration mode: global IQR or country-conditional IQR.",
    )
    effect_regime: Literal["perturbation", "shock"] = Field(
        default="perturbation",
        description=(
            "Signal effect regime for scenario execution: "
            "'perturbation' uses anchored low-to-high contrasts; "
            "'shock' uses signed 1-IQR shocks."
        ),
    )
    perturbation_world: Literal["low", "high"] | None = Field(
        default=None,
        description="Optional perturbation world selector used by paired perturbation runs.",
    )
    norm_observation_sample_size: int = Field(
        default=4,
        ge=1,
        description="Global per-substep cap for observed-contact samples in descriptive norms.",
    )
    household_enabled: bool = Field(
        default=True,
        description="Whether to generate synthetic household ties.",
    )

    # -- paths ---------------------------------------------------------------
    output_dir: Path = Field(
        default_factory=lambda: _PROJECT_ROOT / "abm" / "output",
        description="Artefact output directory.",
    )
    levers_path: Path = Field(
        default_factory=lambda: _PROJECT_ROOT / "config" / "levers.json",
        description="Lever-to-outcome mapping JSON.",
    )
    modeling_dir: Path = Field(
        default_factory=lambda: (_PROJECT_ROOT / "empirical" / "output" / "modeling"),
        description="Fitted model coefficient CSVs.",
    )
    anchors_dir: Path = Field(
        default_factory=lambda: (_PROJECT_ROOT / "empirical" / "output" / "anchors" / "system"),
        description="Per-country anchor-profile Parquets.",
    )
    survey_csv: Path = Field(
        default_factory=lambda: (_PROJECT_ROOT / "preprocess" / "output" / "clean_processed_survey.csv"),
        description="Clean survey micro-data CSV.",
    )
    perturbation_anchor_root: Path | None = Field(
        default_factory=lambda: (_PROJECT_ROOT / "empirical" / "output" / "anchors" / "pe"),
        description=(
            "Optional root for PE-anchored perturbation magnitudes (expects mutable-engine shock_table.csv artifacts)."
        ),
    )

    # -- country selection ---------------------------------------------------
    countries: list[str] = Field(
        default=["DE", "ES", "FR", "HU", "IT", "UK"],
        min_length=1,
        description="ISO-2 country codes to simulate.",
    )

    # -- validators ----------------------------------------------------------

    @model_validator(mode="after")
    def _resolve_and_validate_paths(self) -> SimulationConfig:
        """Resolve relative paths and create ``output_dir`` if needed."""
        if self.constants.state_relaxation_fast < self.constants.state_relaxation_slow:
            msg = "state_relaxation_fast must be >= state_relaxation_slow."
            raise ValueError(msg)
        if not type(self)._usage_validated:
            _validate_declared_parameters_are_used(_PROJECT_ROOT)
            type(self)._usage_validated = True

        self.output_dir = self.output_dir.resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        path_fields = {
            "levers_path": self.levers_path,
            "modeling_dir": self.modeling_dir,
            "anchors_dir": self.anchors_dir,
            "survey_csv": self.survey_csv,
        }
        for name, path in path_fields.items():
            resolved = path.resolve()
            if not resolved.exists():
                logger.warning(
                    "Config path {name}={path} does not exist.",
                    name=name,
                    path=resolved,
                )
            object.__setattr__(self, name, resolved)

        if self.belief_update_kernel_path is not None:
            kernel_path = Path(self.belief_update_kernel_path).resolve()
            if not kernel_path.exists():
                logger.warning("Belief-update kernel path does not exist: {p}", p=kernel_path)
            object.__setattr__(self, "belief_update_kernel_path", kernel_path)

        if self.belief_update_surrogate_dir is not None:
            surrogate_dir = Path(self.belief_update_surrogate_dir).resolve()
            if not surrogate_dir.exists():
                logger.warning("Belief-update surrogate dir does not exist: {p}", p=surrogate_dir)
            object.__setattr__(self, "belief_update_surrogate_dir", surrogate_dir)

        if self.state_ref_spec_path is not None:
            state_ref_path = Path(self.state_ref_spec_path).resolve()
            if not state_ref_path.exists():
                logger.warning("State-ref spec path does not exist: {p}", p=state_ref_path)
            object.__setattr__(self, "state_ref_spec_path", state_ref_path)

        if self.perturbation_anchor_root is not None:
            anchor_root = Path(self.perturbation_anchor_root).resolve()
            if not anchor_root.exists():
                logger.warning(
                    "Perturbation anchor root does not exist: {p}. Falling back to empirical scales.",
                    p=anchor_root,
                )
                object.__setattr__(self, "perturbation_anchor_root", None)
            else:
                object.__setattr__(self, "perturbation_anchor_root", anchor_root)
        return self

    # -- convenience ---------------------------------------------------------

    def log_summary(self) -> None:
        """Emit a compact summary of the active configuration."""
        logger.info(
            "SimulationConfig: {n_agents} agents × {n_steps} steps × {n_reps} reps  (seed={seed}, det={det})",
            n_agents=self.n_agents,
            n_steps=self.n_steps,
            n_reps=self.n_reps,
            seed=self.seed,
            det=self.deterministic,
        )
        logger.info("Countries: {c}", c=", ".join(self.countries))
        logger.info(
            "Mechanisms: k_social={ks:.3f}, homophily_gamma={hg:.3f}, norm_slope={ns:.3f}, "
            "theta_beta_alpha={ta:.3f}, "
            "k_incidence={ki:.3f}, incidence_gamma={ig:.3f}, k_feedback_stakes={kfs:.3f}, "
            "contact_rate_base={cb:.3f}, social_substeps={ss:d}, "
            "contact_multiplier={cmu:.3f}, "
            "calibration={cm}, effect_regime={em}",
            ks=self.knobs.k_social,
            hg=self.constants.homophily_gamma,
            ns=self.constants.norm_response_slope,
            ta=self.constants.norm_threshold_beta_alpha,
            ki=self.knobs.k_incidence,
            ig=self.knobs.incidence_gamma,
            kfs=self.knobs.k_feedback_stakes,
            cb=self.constants.contact_rate_base,
            ss=self.constants.social_substeps_per_month,
            cmu=self.knobs.contact_rate_multiplier,
            cm=self.intensity_calibration_mode,
            em=self.effect_regime,
        )
        logger.info(
            "Network/state: bridge_prob={bp:.3f}, relaxation_fast={rf:.3f}, relaxation_slow={rs:.3f}, "
            "n_obs_month=[{nmin},{nmax}], n_obs_substep_cap={ncap}",
            bp=self.constants.bridge_prob,
            rf=self.constants.state_relaxation_fast,
            rs=self.constants.state_relaxation_slow,
            nmin=self.constants.norm_observation_min_per_month,
            nmax=self.constants.norm_observation_max_per_month,
            ncap=self.norm_observation_sample_size,
        )
        logger.info(
            "State attractor: spec={spec}, strict={strict}",
            spec=str(self.state_ref_spec_path) if self.state_ref_spec_path is not None else "disabled",
            strict=self.state_ref_strict,
        )
        logger.info("Output dir: {d}", d=self.output_dir)

    @classmethod
    def from_env(cls) -> SimulationConfig:
        """Build a config purely from ``ABM_*`` environment variables.

        Returns
        -------
        SimulationConfig
            Configuration populated from the environment.

        Examples
        --------
        >>> import os
        >>> os.environ["ABM_N_AGENTS"] = "500"
        >>> cfg = SimulationConfig.from_env()
        >>> cfg.n_agents
        500
        """
        overrides: dict[str, str] = {}
        prefix = "ABM_"
        for key, value in os.environ.items():
            if key.startswith(prefix):
                field_name = key[len(prefix) :].lower()
                overrides[field_name] = value
        return cls(**overrides)  # type: ignore[arg-type]
