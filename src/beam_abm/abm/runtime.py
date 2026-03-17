"""Simulation loop orchestration using mesa-frames.

Runs the ABM simulation for a given configuration and scenario,
using :class:`~beam_abm.abm.engine.VaccineBehaviourModel` (a
``mesa_frames.Model`` subclass) to manage agent state.

- Batched seeds/replications.
- Common-random-number paired runs (baseline vs intervention).
- Per-step mediator and outcome logging.

Examples
--------
>>> from beam_abm.abm.runtime import run_simulation
>>> # results = run_simulation(cfg, scenario)
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from multiprocessing import get_context

import numpy as np
import polars as pl
from loguru import logger

from beam_abm.abm.belief_update_kernel import BeliefUpdateKernel
from beam_abm.abm.belief_update_surrogate import BeliefUpdateSurrogate
from beam_abm.abm.config import SimulationConfig
from beam_abm.abm.data_contracts import (
    BackboneModel,
    LeversConfig,
    get_abm_levers_config,
    load_all_backbone_models,
    load_anchor_system,
    load_levers_config,
    log_non_abm_outcomes,
)
from beam_abm.abm.engine import TimingBreakdown, VaccineBehaviourModel, _agent_peer_pull_rates_per_substep
from beam_abm.abm.io import load_survey_data
from beam_abm.abm.message_contracts import NormChannelWeights
from beam_abm.abm.metrics import assemble_parity_diagnostics_table
from beam_abm.abm.network import NetworkState, build_contact_network, export_network_adjacency
from beam_abm.abm.outcome_scales import normalize_outcome_to_unit_interval
from beam_abm.abm.population import initialise_population
from beam_abm.abm.population_specs import PopulationSpec, SocietySpec, build_population
from beam_abm.abm.scenario_defs import ScenarioConfig, validate_scenario_effect_mapping

__all__ = [
    "SimulationMetadata",
    "SimulationResult",
    "run_simulation",
    "run_single_replication",
    "run_paired_counterfactual",
    "run_scenario_batch",
]


# =========================================================================
# 1.  Result container
# =========================================================================


@dataclass(slots=True)
class SimulationMetadata:
    """Typed metadata for a simulation run."""

    n_agents: int = 0
    n_steps: int = 0
    communication_enabled: bool = True
    peer_pull_enabled: bool = True
    observed_norms_enabled: bool = True
    k_social: float = 0.5
    k_incidence: float = 1.05
    incidence_gamma: float = 0.65
    k_feedback_stakes: float = 0.5
    homophily_gamma: float = 1.5
    bridge_prob: float = 0.08
    contact_rate_base: float = 1.0
    contact_rate_multiplier: float = 1.0
    norm_response_slope: float = 8.0
    norm_threshold_beta_alpha: float = 2.0
    norm_descriptive_share_w_d: float = 0.65
    state_relaxation_fast: float = 0.2
    state_relaxation_slow: float = 0.12
    intensity_calibration_mode: str = "global"
    targeting: str = "uniform"
    where: str = "global"
    who: str = "all"
    channel: str = "institutional"
    seed_rule: str = "uniform"
    effect_regime: str = "perturbation"
    perturbation_world: str | None = None
    deterministic: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "n_agents": self.n_agents,
            "n_steps": self.n_steps,
            "communication_enabled": self.communication_enabled,
            "peer_pull_enabled": self.peer_pull_enabled,
            "observed_norms_enabled": self.observed_norms_enabled,
            "k_social": self.k_social,
            "k_incidence": self.k_incidence,
            "incidence_gamma": self.incidence_gamma,
            "k_feedback_stakes": self.k_feedback_stakes,
            "homophily_gamma": self.homophily_gamma,
            "bridge_prob": self.bridge_prob,
            "contact_rate_base": self.contact_rate_base,
            "contact_rate_multiplier": self.contact_rate_multiplier,
            "norm_response_slope": self.norm_response_slope,
            "norm_threshold_beta_alpha": self.norm_threshold_beta_alpha,
            "norm_descriptive_share_w_d": self.norm_descriptive_share_w_d,
            "state_relaxation_fast": self.state_relaxation_fast,
            "state_relaxation_slow": self.state_relaxation_slow,
            "intensity_calibration_mode": self.intensity_calibration_mode,
            "targeting": self.targeting,
            "where": self.where,
            "who": self.who,
            "channel": self.channel,
            "seed_rule": self.seed_rule,
            "effect_regime": self.effect_regime,
            "perturbation_world": self.perturbation_world,
            "deterministic": self.deterministic,
        }


@dataclass(slots=True)
class SimulationResult:
    """Container for a single replication's outputs.

    Parameters
    ----------
    scenario_name : str
        Key of the scenario that was run.
    rep_id : int
        Replication index.
    seed : int
        Random seed used.
    outcome_trajectory : pl.DataFrame
        Per-tick aggregate outcome metrics.
    mediator_trajectory : pl.DataFrame
        Per-tick aggregate mediator (mutable state) summaries.
    population_t0 : pl.DataFrame
        Snapshot of agent state at t=0.
    population_final : pl.DataFrame
        Snapshot of agent state at final tick.
    timing : TimingBreakdown
        Wall-clock timing breakdown.
    metadata : SimulationMetadata
        Extra metadata.
    """

    scenario_name: str = ""
    rep_id: int = 0
    seed: int = 42
    outcome_trajectory: pl.DataFrame = field(default_factory=pl.DataFrame)
    mediator_trajectory: pl.DataFrame = field(default_factory=pl.DataFrame)
    signals_realised: pl.DataFrame = field(default_factory=pl.DataFrame)
    outcomes_by_group: pl.DataFrame = field(default_factory=pl.DataFrame)
    mediators_by_group: pl.DataFrame = field(default_factory=pl.DataFrame)
    saturation_trajectory: pl.DataFrame = field(default_factory=pl.DataFrame)
    incidence_trajectory: pl.DataFrame = field(default_factory=pl.DataFrame)
    distance_trajectory: pl.DataFrame = field(default_factory=pl.DataFrame)
    anchoring_diagnostics: pl.DataFrame = field(default_factory=pl.DataFrame)
    effects: pl.DataFrame = field(default_factory=pl.DataFrame)
    scenario_metrics: pl.DataFrame = field(default_factory=pl.DataFrame)
    network_adjacency: pl.DataFrame = field(default_factory=pl.DataFrame)
    population_t0: pl.DataFrame = field(default_factory=pl.DataFrame)
    population_final: pl.DataFrame = field(default_factory=pl.DataFrame)
    timing: TimingBreakdown | None = None
    metadata: SimulationMetadata | None = None


# =========================================================================
# 2.  Single replication (via mesa-frames Model)
# =========================================================================


def _safe_corr(x: object, y: object) -> float:
    xa = np.asarray(x, dtype=np.float64)
    ya = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(xa) & np.isfinite(ya)
    if int(mask.sum()) < 3:
        return float("nan")
    xv = xa[mask]
    yv = ya[mask]
    if float(xv.std()) <= 0.0 or float(yv.std()) <= 0.0:
        return float("nan")
    return float(((xv - xv.mean()) * (yv - yv.mean())).mean() / (xv.std() * yv.std()))


def _compute_scenario_metrics(
    *,
    population_t0: pl.DataFrame,
    network: NetworkState | None,
    peer_pull_interactions_per_agent: float,
) -> pl.DataFrame:
    """Compute run-level diagnostics used by chapter-5 analyses."""
    rows: list[dict[str, float | str]] = []
    if network is None or population_t0.is_empty():
        return pl.DataFrame(rows)

    n_agents = int(population_t0.height)
    degree = network.core_degree.astype(float)
    peer_rate = _agent_peer_pull_rates_per_substep(
        population_t0,
        base_rate=float(peer_pull_interactions_per_agent),
    )
    rows.append({"metric": "corr_degree_peer_rate", "value": _safe_corr(degree, peer_rate)})

    if "institutional_trust_avg" in population_t0.columns:
        rows.append(
            {
                "metric": "corr_degree_trust",
                "value": _safe_corr(degree, population_t0["institutional_trust_avg"].cast(pl.Float64).to_numpy()),
            }
        )
    if "vaccine_risk_avg" in population_t0.columns:
        rows.append(
            {
                "metric": "corr_degree_risk",
                "value": _safe_corr(degree, population_t0["vaccine_risk_avg"].cast(pl.Float64).to_numpy()),
            }
        )

    if ("norm_share_hat_vax" in population_t0.columns) and (
        ("vax_willingness_T12" in population_t0.columns) or ("vax_willingness_Y" in population_t0.columns)
    ):
        y_col = "vax_willingness_T12" if "vax_willingness_T12" in population_t0.columns else "vax_willingness_Y"
        y = normalize_outcome_to_unit_interval(
            population_t0[y_col].cast(pl.Float64).to_numpy().astype(float),
            "vax_willingness_T12",
        )
        perceived = population_t0["norm_share_hat_vax"].cast(pl.Float64).to_numpy().astype(float)
        src = network.recurring_edge_src
        dst = network.recurring_edge_dst
        counts = np.bincount(src, minlength=n_agents).astype(float) if src.size > 0 else np.zeros(n_agents, dtype=float)
        sums = (
            np.bincount(src, weights=y[dst], minlength=n_agents).astype(float)
            if src.size > 0
            else np.zeros(n_agents, dtype=float)
        )
        true_share = y.copy()
        has_neighbors = counts > 0.0
        true_share[has_neighbors] = sums[has_neighbors] / counts[has_neighbors]
        true_share = np.clip(true_share, 0.0, 1.0)
        rows.extend(
            [
                {"metric": "corr_perceived_true_neighbor_share", "value": _safe_corr(perceived, true_share)},
                {"metric": "perceived_mean", "value": float(np.nanmean(perceived))},
                {"metric": "true_neighbor_share_mean", "value": float(np.nanmean(true_share))},
            ]
        )

    return pl.DataFrame(rows) if rows else pl.DataFrame()


def run_single_replication(
    cfg: SimulationConfig,
    scenario: ScenarioConfig,
    models: dict[str, BackboneModel],
    levers_cfg: LeversConfig,
    agent_df: pl.DataFrame,
    network: NetworkState | None = None,
    belief_kernel: BeliefUpdateKernel | BeliefUpdateSurrogate | None = None,
    rep_id: int = 0,
    seed: int = 42,
) -> SimulationResult:
    """Run one replication of the simulation loop.

    Creates a :class:`VaccineBehaviourModel`, runs it for
    ``cfg.n_steps`` ticks, and returns the result.

    Parameters
    ----------
    cfg : SimulationConfig
        Simulation parameters.
    scenario : ScenarioConfig
        Scenario to simulate.
    models : dict[str, BackboneModel]
        Loaded backbone models for each outcome.
    levers_cfg : LeversConfig
        Lever configuration.
    agent_df : pl.DataFrame
        Initial agent population (will be cloned internally).
    network : NetworkState or None
        Pre-built contact network.
    rep_id : int
        Replication index.
    seed : int
        Random seed.

    Returns
    -------
    SimulationResult
        Complete results for this replication.
    """
    if belief_kernel is None:
        runtime_kernel = "parametric"
    elif isinstance(belief_kernel, BeliefUpdateSurrogate):
        runtime_kernel = "surrogate"
    else:
        runtime_kernel = "llm"
    if runtime_kernel not in scenario.supported_kernels:
        msg = (
            f"Scenario '{scenario.name}' does not support runtime kernel '{runtime_kernel}'. "
            f"Supported: {scenario.supported_kernels}"
        )
        raise ValueError(msg)

    model = VaccineBehaviourModel.from_preloaded(
        cfg=cfg,
        scenario=scenario,
        models=models,
        levers_cfg=levers_cfg,
        agent_df=agent_df.clone(),
        network=network,
        belief_kernel=belief_kernel,
        seed=seed,
    )

    model.run_model()

    diagnostics = assemble_parity_diagnostics_table(
        outcome_trajectory=model.outcome_trajectory,
        incidence_trajectory=model.incidence_trajectory,
        distance_trajectory=model.distance_trajectory,
    )
    scenario_metrics = _compute_scenario_metrics(
        population_t0=model.population_t0,
        network=model.network,
        peer_pull_interactions_per_agent=float(cfg.constants.peer_pull_interactions_per_agent),
    )
    adjacency = export_network_adjacency(model.network) if model.network is not None else pl.DataFrame()

    logger.info(
        "Rep {rep}: {n_steps} ticks completed (seed={seed}).",
        rep=rep_id,
        n_steps=cfg.n_steps,
        seed=seed,
    )

    return SimulationResult(
        scenario_name=scenario.name,
        rep_id=rep_id,
        seed=seed,
        outcome_trajectory=model.outcome_trajectory,
        mediator_trajectory=model.mediator_trajectory,
        signals_realised=model.signals_realised,
        outcomes_by_group=model.outcomes_by_group,
        mediators_by_group=model.mediators_by_group,
        saturation_trajectory=model.saturation_trajectory,
        incidence_trajectory=model.incidence_trajectory,
        distance_trajectory=model.distance_trajectory,
        anchoring_diagnostics=model.anchoring_diagnostics,
        effects=diagnostics,
        scenario_metrics=scenario_metrics,
        network_adjacency=adjacency,
        population_t0=model.population_t0,
        population_final=model.population_final,
        timing=model.timing,
        metadata=SimulationMetadata(
            n_agents=cfg.n_agents,
            n_steps=cfg.n_steps,
            communication_enabled=scenario.communication_enabled,
            peer_pull_enabled=scenario.peer_pull_enabled,
            observed_norms_enabled=scenario.observed_norms_enabled,
            k_social=cfg.knobs.k_social,
            k_incidence=cfg.knobs.k_incidence,
            incidence_gamma=cfg.knobs.incidence_gamma,
            k_feedback_stakes=cfg.knobs.k_feedback_stakes,
            homophily_gamma=cfg.constants.homophily_gamma,
            bridge_prob=cfg.constants.bridge_prob,
            contact_rate_base=cfg.constants.contact_rate_base,
            contact_rate_multiplier=cfg.knobs.contact_rate_multiplier,
            norm_response_slope=cfg.constants.norm_response_slope,
            norm_threshold_beta_alpha=cfg.constants.norm_threshold_beta_alpha,
            norm_descriptive_share_w_d=(
                float(scenario.norm_descriptive_share_w_d)
                if scenario.norm_descriptive_share_w_d is not None
                else NormChannelWeights().descriptive
            ),
            state_relaxation_fast=cfg.constants.state_relaxation_fast,
            state_relaxation_slow=cfg.constants.state_relaxation_slow,
            intensity_calibration_mode=cfg.intensity_calibration_mode,
            targeting=scenario.targeting,
            where=scenario.where,
            who=scenario.who,
            channel="event_only",
            seed_rule=scenario.seed_rule,
            effect_regime=cfg.effect_regime,
            perturbation_world=cfg.perturbation_world,
            deterministic=cfg.deterministic,
        ),
    )


# =========================================================================
# 3.  Full simulation with replications
# =========================================================================

_AUTO_REP_WORKER_MEM_GIB = 2.5
_AUTO_REP_WORKER_RESERVE_GIB = 4.0
_WORKER_CONTEXT: dict[str, object] | None = None


def _available_memory_gib() -> float | None:
    """Return currently available memory in GiB when supported."""
    try:
        with open("/proc/meminfo", encoding="utf-8") as meminfo:
            for line in meminfo:
                if line.startswith("MemAvailable:"):
                    parts = line.split()
                    if len(parts) >= 2:
                        available_kib = int(parts[1])
                        return available_kib / float(1024**2)
    except (FileNotFoundError, OSError, ValueError):
        pass

    try:
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        available_pages = int(os.sysconf("SC_AVPHYS_PAGES"))
    except (AttributeError, OSError, ValueError):
        return None
    return (page_size * available_pages) / float(1024**3)


def _resolve_rep_workers(cfg: SimulationConfig, *, n_reps: int, requested_workers: int) -> int:
    """Resolve effective repetition workers with RAM-aware auto mode."""
    if cfg.deterministic:
        return 1

    cpu_cap = max(1, os.cpu_count() or 1)
    if requested_workers > 0:
        return max(1, min(requested_workers, n_reps, cpu_cap))

    available_gib = _available_memory_gib()
    if available_gib is None:
        return 1
    budget_gib = max(0.0, available_gib - _AUTO_REP_WORKER_RESERVE_GIB)
    memory_cap = int(budget_gib // _AUTO_REP_WORKER_MEM_GIB)
    return max(1, min(memory_cap, n_reps, cpu_cap))


def _init_replication_worker(
    cfg: SimulationConfig,
    scenario: ScenarioConfig,
    models: dict[str, BackboneModel],
    levers_cfg: LeversConfig,
    base_pop: pl.DataFrame,
    network: NetworkState | None,
    belief_kernel: BeliefUpdateKernel | BeliefUpdateSurrogate | None,
) -> None:
    """Initialise process-local replication context once per worker."""
    global _WORKER_CONTEXT
    _WORKER_CONTEXT = {
        "cfg": cfg,
        "scenario": scenario,
        "models": models,
        "levers_cfg": levers_cfg,
        "base_pop": base_pop,
        "network": network,
        "belief_kernel": belief_kernel,
    }


def _run_replication_in_worker(rep_and_seed: tuple[int, int]) -> SimulationResult:
    """Execute a single replication inside a process worker."""
    if _WORKER_CONTEXT is None:
        raise RuntimeError("Replication worker context is not initialised.")
    rep, rep_seed = rep_and_seed
    return run_single_replication(
        cfg=_WORKER_CONTEXT["cfg"],  # type: ignore[arg-type]
        scenario=_WORKER_CONTEXT["scenario"],  # type: ignore[arg-type]
        models=_WORKER_CONTEXT["models"],  # type: ignore[arg-type]
        levers_cfg=_WORKER_CONTEXT["levers_cfg"],  # type: ignore[arg-type]
        agent_df=_WORKER_CONTEXT["base_pop"],  # type: ignore[arg-type]
        network=_WORKER_CONTEXT["network"],  # type: ignore[arg-type]
        belief_kernel=_WORKER_CONTEXT["belief_kernel"],  # type: ignore[arg-type]
        rep_id=rep,
        seed=rep_seed,
    )


def run_simulation(
    cfg: SimulationConfig,
    scenario: ScenarioConfig,
    population_spec: PopulationSpec | None = None,
    society_spec: SocietySpec | None = None,
    rep_workers: int = 0,
) -> list[SimulationResult]:
    """Run a full simulation batch with multiple replications.

    Handles loading data, initialising population, building network,
    and running all replications.

    Parameters
    ----------
    cfg : SimulationConfig
        Simulation parameters.
    scenario : ScenarioConfig
        Scenario to simulate.
    population_spec : PopulationSpec or None
        Optional population construction spec.
    society_spec : SocietySpec or None
        Optional counterfactual society weighting.
    rep_workers : int
        Repetition workers. Use ``0`` for RAM-aware auto mode.

    Returns
    -------
    list[SimulationResult]
        Results for each replication.
    """
    cfg.log_summary()

    # Load data contracts
    levers_cfg = load_levers_config(cfg.levers_path)
    log_non_abm_outcomes(levers_cfg, context="run_simulation")
    levers_cfg = get_abm_levers_config(levers_cfg)
    anchor_system = load_anchor_system(cfg.anchors_dir)
    survey_df = load_survey_data(cfg.survey_csv, cfg.countries)
    models = load_all_backbone_models(cfg.modeling_dir, levers_cfg, survey_df=survey_df)
    validate_scenario_effect_mapping(
        [scenario],
        levers_cfg=levers_cfg,
        models=models,
        agents_for_pe_ref=survey_df,
    )

    belief_kernel = None
    if cfg.belief_update_surrogate_dir is not None:
        belief_kernel = BeliefUpdateSurrogate.from_bundle_dir(cfg.belief_update_surrogate_dir)
    elif cfg.belief_update_kernel_path is not None:
        belief_kernel = BeliefUpdateKernel.from_targets_file(
            cfg.belief_update_kernel_path,
            intensity_scale=cfg.belief_update_intensity_scale,
        )

    if population_spec is None:
        base_pop = initialise_population(
            n_agents=cfg.n_agents,
            countries=cfg.countries,
            anchor_system=anchor_system,
            survey_df=survey_df,
            seed=cfg.seed,
            norm_threshold_beta_alpha=cfg.constants.norm_threshold_beta_alpha,
        )
    else:
        base_pop = build_population(
            population=population_spec,
            society=society_spec,
            anchor_system=anchor_system,
            survey_df=survey_df,
            default_n_agents=cfg.n_agents,
            norm_threshold_beta_alpha=cfg.constants.norm_threshold_beta_alpha,
        )

    # Build network if needed
    network: NetworkState | None = None
    if scenario.communication_enabled and (scenario.peer_pull_enabled or scenario.observed_norms_enabled):
        network = build_contact_network(
            base_pop,
            bridge_prob=cfg.constants.bridge_prob,
            household_enabled=cfg.household_enabled,
            seed=cfg.seed,
        )

    workers = _resolve_rep_workers(cfg, n_reps=cfg.n_reps, requested_workers=rep_workers)
    if rep_workers == 0:
        logger.info(
            "Repetition workers auto-resolved to {workers} "
            "(available_mem_gib={mem:.2f}, reserve_gib={reserve}, per_worker_gib={per_worker}).",
            workers=workers,
            mem=(_available_memory_gib() or -1.0),
            reserve=_AUTO_REP_WORKER_RESERVE_GIB,
            per_worker=_AUTO_REP_WORKER_MEM_GIB,
        )
    else:
        logger.info("Using requested repetition workers: {workers}.", workers=workers)
    logger.info("Repetition backend: process.")

    def _run_rep(rep: int) -> SimulationResult:
        rep_seed = cfg.seed + rep
        return run_single_replication(
            cfg=cfg,
            scenario=scenario,
            models=models,
            levers_cfg=levers_cfg,
            agent_df=base_pop,
            network=network,
            belief_kernel=belief_kernel,
            rep_id=rep,
            seed=rep_seed,
        )

    # Run replications
    if workers == 1:
        return [_run_rep(rep) for rep in range(cfg.n_reps)]

    rep_seeds = [(rep, cfg.seed + rep) for rep in range(cfg.n_reps)]
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=get_context("spawn"),
        initializer=_init_replication_worker,
        initargs=(cfg, scenario, models, levers_cfg, base_pop, network, belief_kernel),
    ) as executor:
        results = list(executor.map(_run_replication_in_worker, rep_seeds))

    return results


# =========================================================================
# 4.  Paired counterfactual runner (CRN)
# =========================================================================


def run_paired_counterfactual(
    cfg: SimulationConfig,
    baseline_scenario: ScenarioConfig,
    intervention_scenario: ScenarioConfig,
    population_spec: PopulationSpec | None = None,
    society_spec: SocietySpec | None = None,
) -> tuple[list[SimulationResult], list[SimulationResult]]:
    """Run baseline and intervention with common random numbers.

    Uses identical seeds for population initialisation and RNG
    sequences, differing only in the applied scenario signals.

    Parameters
    ----------
    cfg : SimulationConfig
        Simulation parameters.
    baseline_scenario : ScenarioConfig
        The baseline (no-shock) scenario.
    intervention_scenario : ScenarioConfig
        The intervention scenario.
    population_spec : PopulationSpec or None
        Optional population construction spec.
    society_spec : SocietySpec or None
        Optional counterfactual society weighting.

    Returns
    -------
    tuple[list[SimulationResult], list[SimulationResult]]
        ``(baseline_results, intervention_results)`` with matched seeds.
    """
    logger.info(
        "Paired CRN run: baseline='{b}' vs intervention='{i}'.",
        b=baseline_scenario.name,
        i=intervention_scenario.name,
    )

    # Load shared artefacts
    levers_cfg = load_levers_config(cfg.levers_path)
    log_non_abm_outcomes(levers_cfg, context="run_paired_counterfactual")
    levers_cfg = get_abm_levers_config(levers_cfg)
    anchor_system = load_anchor_system(cfg.anchors_dir)
    survey_df = load_survey_data(cfg.survey_csv, cfg.countries)
    models = load_all_backbone_models(cfg.modeling_dir, levers_cfg, survey_df=survey_df)
    validate_scenario_effect_mapping(
        [baseline_scenario, intervention_scenario],
        levers_cfg=levers_cfg,
        models=models,
        agents_for_pe_ref=survey_df,
    )

    baseline_results: list[SimulationResult] = []
    intervention_results: list[SimulationResult] = []

    for rep in range(cfg.n_reps):
        rep_seed = cfg.seed + rep

        # Identical population per rep
        if population_spec is None:
            pop = initialise_population(
                n_agents=cfg.n_agents,
                countries=cfg.countries,
                anchor_system=anchor_system,
                survey_df=survey_df,
                seed=rep_seed,
                norm_threshold_beta_alpha=cfg.constants.norm_threshold_beta_alpha,
            )
        else:
            rep_population = population_spec.model_copy(update={"seed": rep_seed})
            pop = build_population(
                population=rep_population,
                society=society_spec,
                anchor_system=anchor_system,
                survey_df=survey_df,
                default_n_agents=cfg.n_agents,
                norm_threshold_beta_alpha=cfg.constants.norm_threshold_beta_alpha,
            )

        # Build networks
        base_net: NetworkState | None = None
        if baseline_scenario.communication_enabled and (
            baseline_scenario.peer_pull_enabled or baseline_scenario.observed_norms_enabled
        ):
            base_net = build_contact_network(
                pop,
                bridge_prob=cfg.constants.bridge_prob,
                household_enabled=cfg.household_enabled,
                seed=rep_seed,
            )

        interv_net: NetworkState | None = None
        if intervention_scenario.communication_enabled and (
            intervention_scenario.peer_pull_enabled or intervention_scenario.observed_norms_enabled
        ):
            interv_net = build_contact_network(
                pop,
                bridge_prob=cfg.constants.bridge_prob,
                household_enabled=cfg.household_enabled,
                seed=rep_seed,
            )

        # Run baseline
        base_result = run_single_replication(
            cfg=cfg,
            scenario=baseline_scenario,
            models=models,
            levers_cfg=levers_cfg,
            agent_df=pop,
            network=base_net,
            rep_id=rep,
            seed=rep_seed,
        )
        baseline_results.append(base_result)

        # Run intervention with same seed
        interv_result = run_single_replication(
            cfg=cfg,
            scenario=intervention_scenario,
            models=models,
            levers_cfg=levers_cfg,
            agent_df=pop,
            network=interv_net,
            rep_id=rep,
            seed=rep_seed,
        )
        intervention_results.append(interv_result)

    return baseline_results, intervention_results


# =========================================================================
# 5.  Batch scenario runner
# =========================================================================


def run_scenario_batch(
    cfg: SimulationConfig,
    scenarios: list[ScenarioConfig],
    baseline: ScenarioConfig | None = None,
    population_spec: PopulationSpec | None = None,
    society_spec: SocietySpec | None = None,
) -> dict[str, list[SimulationResult]]:
    """Run multiple scenarios, each paired with a common baseline.

    Parameters
    ----------
    cfg : SimulationConfig
        Simulation parameters.
    scenarios : list[ScenarioConfig]
        Scenarios to run.
    baseline : ScenarioConfig or None
        Baseline for paired comparison.  If ``None``, each scenario
        runs standalone.
    population_spec : PopulationSpec or None
        Optional population construction spec.
    society_spec : SocietySpec or None
        Optional counterfactual society weighting.

    Returns
    -------
    dict[str, list[SimulationResult]]
        Mapping from scenario name to its replication results.
        If *baseline* is given, also includes ``"baseline"`` key.
    """
    all_results: dict[str, list[SimulationResult]] = {}

    if baseline is not None:
        logger.info("Running baseline for paired comparisons.")
        all_results["baseline"] = run_simulation(
            cfg,
            baseline,
            population_spec=population_spec,
            society_spec=society_spec,
        )

    for scenario in scenarios:
        logger.info("Running scenario: {name}.", name=scenario.name)
        if baseline is not None:
            _, interv = run_paired_counterfactual(
                cfg,
                baseline,
                scenario,
                population_spec=population_spec,
                society_spec=society_spec,
            )
            all_results[scenario.name] = interv
        else:
            all_results[scenario.name] = run_simulation(
                cfg,
                scenario,
                population_spec=population_spec,
                society_spec=society_spec,
            )

    return all_results
