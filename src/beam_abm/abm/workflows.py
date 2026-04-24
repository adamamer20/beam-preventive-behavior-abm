"""ABM library workflows for script-facing orchestration."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Literal

import numpy as np
import polars as pl
from loguru import logger

from beam_abm.abm.config import SimulationConfig
from beam_abm.abm.sensitivity_pipeline import (
    DEFAULT_FLAGSHIP_SCENARIOS,
    DEFAULT_OUTPUT_METRICS,
    SensitivityPipelineConfig,
    run_full_sensitivity_pipeline,
)

__all__ = [
    "DEFAULT_FLAGSHIP_SCENARIOS",
    "DEFAULT_OUTPUT_METRICS",
    "run_build_population_workflow",
    "run_benchmark_workflow",
    "run_diagnostics_workflow",
    "run_scenarios_workflow",
    "run_report_workflow",
    "run_full_sensitivity_pipeline_workflow",
    "calibrate_homophily_gamma_workflow",
    "rebuild_heterogeneity_cdf_workflow",
]

DiagnosticsSuite = Literal["all", "core", "importation", "credibility"]
EffectRegime = Literal["perturbation", "shock", "both"]
PopulationMode = Literal["survey", "resample_on_manifold", "parquet"]
SensitivityEffectMetric = Literal["final", "auc"]


def run_build_population_workflow(
    *,
    n_agents: int = 10_000,
    seed: int = 42,
    output: Path | None = None,
) -> None:
    """Build baseline population artifacts and diagnostics."""
    from beam_abm.abm.benchmarking import (
        MutableSummaryRow,
        PopulationDiagnosticsReport,
        QuantileSummaryRow,
        run_baseline_replication_gate,
    )
    from beam_abm.abm.data_contracts import (
        get_abm_levers_config,
        load_all_backbone_models,
        load_anchor_system,
        load_levers_config,
        log_non_abm_outcomes,
    )
    from beam_abm.abm.io import load_survey_data, save_benchmark_report
    from beam_abm.abm.population import initialise_population
    from beam_abm.abm.state_schema import MUTABLE_COLUMNS

    cfg = SimulationConfig(n_agents=n_agents, seed=seed)
    anchor_system = load_anchor_system(cfg.anchors_dir)
    survey_df = load_survey_data(cfg.survey_csv, cfg.countries)

    agents = initialise_population(
        n_agents=cfg.n_agents,
        countries=cfg.countries,
        anchor_system=anchor_system,
        survey_df=survey_df,
        seed=cfg.seed,
        norm_threshold_beta_alpha=cfg.constants.norm_threshold_beta_alpha,
    )

    out_path = output or (cfg.output_dir / "populations" / "baseline.parquet")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    agents.write_parquet(out_path)
    logger.info("Population saved to {p}.", p=out_path)

    levers_cfg = load_levers_config(cfg.levers_path)
    log_non_abm_outcomes(levers_cfg, context="run_build_population_workflow")
    levers_cfg = get_abm_levers_config(levers_cfg)
    models = load_all_backbone_models(cfg.modeling_dir, levers_cfg, survey_df=survey_df)
    baseline_gate = run_baseline_replication_gate(
        agents_t0=agents,
        survey_df=survey_df,
        models=models,
        levers_cfg=levers_cfg,
        anchor_system=anchor_system,
        countries=cfg.countries,
        model_coefficients_path=cfg.modeling_dir / "model_coefficients_all.csv",
    )

    mutable_cols_present = [c for c in sorted(MUTABLE_COLUMNS) if c in agents.columns and c in survey_df.columns]
    mutable_summary_rows: list[MutableSummaryRow] = []
    quantile_summary_rows: list[QuantileSummaryRow] = []
    for col in mutable_cols_present:
        agent_mean = float(agents[col].mean())  # type: ignore[arg-type]
        survey_mean = float(survey_df[col].mean())  # type: ignore[arg-type]
        mutable_summary_rows.append(
            MutableSummaryRow(
                construct=col,
                agent_mean=agent_mean,
                survey_mean=survey_mean,
                abs_diff=abs(agent_mean - survey_mean),
            )
        )

        agent_q10 = float(agents[col].quantile(0.1, interpolation="nearest"))  # type: ignore[arg-type]
        agent_q50 = float(agents[col].quantile(0.5, interpolation="nearest"))  # type: ignore[arg-type]
        agent_q90 = float(agents[col].quantile(0.9, interpolation="nearest"))  # type: ignore[arg-type]
        survey_q10 = float(survey_df[col].quantile(0.1, interpolation="nearest"))  # type: ignore[arg-type]
        survey_q50 = float(survey_df[col].quantile(0.5, interpolation="nearest"))  # type: ignore[arg-type]
        survey_q90 = float(survey_df[col].quantile(0.9, interpolation="nearest"))  # type: ignore[arg-type]
        quantile_summary_rows.append(
            QuantileSummaryRow(
                construct=col,
                agent_q10=agent_q10,
                agent_q50=agent_q50,
                agent_q90=agent_q90,
                survey_q10=survey_q10,
                survey_q50=survey_q50,
                survey_q90=survey_q90,
            )
        )

    duplication_rate = 0.0
    if "row_id" in agents.columns:
        unique_rows = agents.get_column("row_id").n_unique()
        duplication_rate = 1.0 - (unique_rows / agents.height)

    anchor_coverage_rate = 0.0
    if "anchor_id" in agents.columns and anchor_system.n_anchors > 0:
        anchor_coverage_rate = agents.get_column("anchor_id").n_unique() / anchor_system.n_anchors

    report = PopulationDiagnosticsReport(
        gate=baseline_gate,
        mutable_summary=mutable_summary_rows,
        quantile_summary=quantile_summary_rows,
        duplication_rate=float(duplication_rate),
        anchor_coverage_rate=float(anchor_coverage_rate),
        n_agents=agents.height,
        countries=cfg.countries,
        source_population=str(out_path),
    )
    metrics_path = out_path.with_suffix(".metrics.json")
    save_benchmark_report(report, metrics_path)
    logger.info("Population diagnostics saved to {p}.", p=metrics_path)


def run_benchmark_workflow(
    *,
    n_agents: tuple[int, ...] = (1_000, 5_000, 10_000),
    n_steps: int = 52,
    n_reps: int = 3,
    output: Path | None = None,
) -> None:
    """Run ABM benchmark validation and performance workflows."""
    from beam_abm.abm.benchmarking import (
        BenchmarkReport,
        build_non_degeneracy_report,
        run_all_gates,
        run_performance_benchmarks,
    )
    from beam_abm.abm.data_contracts import (
        get_abm_levers_config,
        load_all_backbone_models,
        load_anchor_system,
        load_levers_config,
        log_non_abm_outcomes,
    )
    from beam_abm.abm.io import load_survey_data, save_benchmark_report
    from beam_abm.abm.population_specs import PopulationSpec, build_population
    from beam_abm.abm.runtime import run_simulation
    from beam_abm.abm.scenario_defs import get_scenario_library, iter_scenario_regime_runs

    cfg = SimulationConfig(
        n_agents=n_agents[0],
        n_steps=n_steps,
        n_reps=n_reps,
        deterministic=True,
    )

    levers_cfg = load_levers_config(cfg.levers_path)
    log_non_abm_outcomes(levers_cfg, context="run_benchmark_workflow")
    levers_cfg = get_abm_levers_config(levers_cfg)
    anchor_system = load_anchor_system(cfg.anchors_dir)
    survey_df = load_survey_data(cfg.survey_csv, cfg.countries)
    models = load_all_backbone_models(cfg.modeling_dir, levers_cfg, survey_df=survey_df)

    baseline_population_gate = PopulationSpec(
        mode="survey",
        n_agents=survey_df.height,
        seed=cfg.seed,
        stratify_by_country=True,
    )
    baseline_population_runtime = PopulationSpec(
        mode="resample_on_manifold",
        n_agents=cfg.n_agents,
        seed=cfg.seed,
        stratify_by_country=True,
    )
    agents_t0 = build_population(
        population=baseline_population_gate,
        society=None,
        anchor_system=anchor_system,
        survey_df=survey_df,
        default_n_agents=cfg.n_agents,
        norm_threshold_beta_alpha=cfg.constants.norm_threshold_beta_alpha,
    )

    scenario_library = get_scenario_library(include_mutable=True)
    benchmark_scenarios = tuple(sorted(scenario_library))
    results = []
    for scenario_key, effect_regime in iter_scenario_regime_runs(benchmark_scenarios):
        scenario = scenario_library[scenario_key]
        cfg_run = cfg.model_copy(deep=True)
        cfg_run.effect_regime = effect_regime
        logger.info("Benchmark run: scenario={name}, effect_regime={regime}", name=scenario_key, regime=effect_regime)
        results.extend(run_simulation(cfg_run, scenario, population_spec=baseline_population_runtime))

    gates = run_all_gates(
        agents_t0,
        survey_df,
        models,
        levers_cfg,
        anchor_system,
        results,
        cfg.countries,
        cfg.modeling_dir / "model_coefficients_all.csv",
        cfg=cfg,
    )

    perf = run_performance_benchmarks(cfg, list(n_agents))
    non_degeneracy = build_non_degeneracy_report(gates.gates)
    report = BenchmarkReport(
        gates=gates.gates,
        performance=perf,
        non_degeneracy=non_degeneracy,
    )

    out_path = output or (cfg.output_dir / "benchmarks" / "report.json")
    save_benchmark_report(report, out_path)
    logger.info(
        "Non-degeneracy initial checks: {status} (failed_domains={n_failed}, incomplete_domains={n_incomplete}).",
        status=non_degeneracy.status.upper(),
        n_failed=non_degeneracy.n_failed_domains,
        n_incomplete=non_degeneracy.n_incomplete_domains,
    )


def run_diagnostics_workflow(
    *,
    suite: DiagnosticsSuite = "all",
    n_agents: int = 2_000,
    n_steps: int = 24,
    n_reps: int = 3,
    seeds: tuple[int, ...] = (42, 43, 44, 45, 46),
    output_dir: Path = Path("abm/output/diagnostics"),
):
    """Run diagnostics suites and return the typed diagnostics report."""
    from beam_abm.abm.diagnostics import run_diagnostics

    report = run_diagnostics(
        suite=suite,
        n_agents=n_agents,
        n_steps=n_steps,
        n_reps=n_reps,
        seeds=list(seeds),
        output_dir=output_dir,
    )
    summary_path = output_dir / report.run_id / "summary.json"
    logger.info(
        "Diagnostics complete: run_id={run_id}, overall_passed={passed}, summary={summary}",
        run_id=report.run_id,
        passed=report.overall_passed,
        summary=summary_path,
    )
    return report


def _build_run_metadata_payload(
    *,
    artifact,
    simulation_cfg: SimulationConfig,
    scenario_to_dict,
) -> dict[str, object]:
    payload: dict[str, object] = {}
    result_metadata = artifact.result_metadata
    if result_metadata is not None and hasattr(result_metadata, "to_dict"):
        payload["run_metadata"] = result_metadata.to_dict()  # type: ignore[assignment]
    elif result_metadata is not None and hasattr(result_metadata, "model_dump"):
        payload["run_metadata"] = result_metadata.model_dump()  # type: ignore[assignment]

    payload["scenario_config"] = scenario_to_dict(artifact.scenario, scenario_key=artifact.scenario_key)  # type: ignore[arg-type]
    payload["simulation_config"] = simulation_cfg.model_dump(mode="json")
    payload["effect_regime"] = artifact.effect_regime
    payload.update(artifact.metadata_extra)
    return payload


def run_scenarios_workflow(
    *,
    scenario: tuple[str, ...] = ("baseline",),
    n_agents: int = 10_000,
    n_steps: int = 52,
    n_reps: int = 10,
    rep_workers: int = 0,
    seed: int = 42,
    society: str | None = None,
    pop_mode: PopulationMode | None = None,
    pop_n: int | None = None,
    pop_parquet: Path | None = None,
    deterministic: bool = False,
    belief_surrogate_dir: Path | None = None,
    effect_regime: EffectRegime = "perturbation",
) -> None:
    """Run ABM scenarios under explicit effect-regime orchestration."""
    from beam_abm.abm.data_contracts import (
        get_abm_levers_config,
        load_all_backbone_models,
        load_levers_config,
        log_non_abm_outcomes,
    )
    from beam_abm.abm.effect_regimes import run_perturbation_paired, run_shock_single
    from beam_abm.abm.io import RunPaths, load_survey_data, save_run_outputs
    from beam_abm.abm.metrics import build_behavioral_projection_weights
    from beam_abm.abm.population_specs import PopulationSpec, get_society
    from beam_abm.abm.scenario_defs import get_scenario, get_scenario_library, scenario_to_dict

    cfg = SimulationConfig(
        n_agents=n_agents,
        n_steps=n_steps,
        n_reps=n_reps,
        seed=seed,
        deterministic=deterministic,
        belief_update_surrogate_dir=belief_surrogate_dir,
    )
    cfg.log_summary()

    population_spec = None
    society_spec = None
    if society or pop_mode or pop_n is not None or pop_parquet is not None:
        if society:
            society_spec = get_society(society)
        pop_kwargs: dict[str, object] = {"seed": seed}
        if pop_mode:
            pop_kwargs["mode"] = pop_mode
        if pop_n is not None:
            pop_kwargs["n_agents"] = pop_n
        if pop_parquet:
            pop_kwargs["parquet_path"] = pop_parquet
        population_spec = PopulationSpec(**pop_kwargs)

    scenario_keys: list[str] = []
    seen_scenarios: set[str] = set()
    for scenario_key in scenario:
        if scenario_key in seen_scenarios:
            logger.info("Skipping duplicate scenario request: {scenario}.", scenario=scenario_key)
            continue
        seen_scenarios.add(scenario_key)
        scenario_keys.append(scenario_key)
    if not scenario_keys:
        raise ValueError("No scenarios selected for run workflow.")

    survey_df = load_survey_data(cfg.survey_csv, cfg.countries)
    levers_cfg = load_levers_config(cfg.levers_path)
    log_non_abm_outcomes(levers_cfg, context="run_scenarios_workflow")
    levers_cfg = get_abm_levers_config(levers_cfg)
    backbone_models = load_all_backbone_models(cfg.modeling_dir, levers_cfg, survey_df=survey_df)
    projection_weights = build_behavioral_projection_weights(backbone_models)

    scenario_library = get_scenario_library(include_mutable=True)
    baseline_results_cache: dict[str, list] = {}
    regimes_to_run: tuple[Literal["perturbation", "shock"], ...]
    if effect_regime == "both":
        regimes_to_run = ("perturbation", "shock")
    else:
        regimes_to_run = (effect_regime,)

    for scenario_key in scenario_keys:
        scenario_cfg = get_scenario(scenario_key)
        for regime in regimes_to_run:
            logger.info(
                "Running scenario={name} under effect_regime={regime}.",
                name=scenario_cfg.name,
                regime=regime,
            )
            cfg_run = cfg.model_copy(deep=True)
            cfg_run.effect_regime = regime
            cfg_run.perturbation_world = None

            if regime == "perturbation":
                artifacts = run_perturbation_paired(
                    cfg=cfg_run,
                    scenario_key=scenario_key,
                    scenario=scenario_cfg,
                    population_spec=population_spec,
                    society_spec=society_spec,
                    rep_workers=rep_workers,
                    projection_weights=projection_weights,
                )
            else:
                artifacts = run_shock_single(
                    cfg=cfg_run,
                    scenario_key=scenario_key,
                    scenario=scenario_cfg,
                    scenario_library=scenario_library,
                    baseline_results_cache=baseline_results_cache,
                    population_spec=population_spec,
                    society_spec=society_spec,
                    rep_workers=rep_workers,
                    projection_weights=projection_weights,
                )

            for artifact in artifacts:
                paths = RunPaths(cfg.output_dir, artifact.run_id)
                metadata_payload = _build_run_metadata_payload(
                    artifact=artifact,
                    simulation_cfg=cfg_run,
                    scenario_to_dict=scenario_to_dict,
                )
                save_run_outputs(
                    paths,
                    artifact.outcome_trajectory,
                    artifact.mediator_trajectory,
                    artifact.signals_realised,
                    artifact.outcomes_by_group,
                    artifact.mediators_by_group,
                    artifact.saturation_trajectory,
                    artifact.incidence_trajectory,
                    artifact.distance_trajectory,
                    artifact.effects,
                    artifact.population_t0,
                    artifact.population_final,
                    metadata_payload,
                    scenario_metrics=artifact.scenario_metrics,
                    adjacency=artifact.network_adjacency,
                    dynamics_summary=artifact.dynamics_summary,
                    dynamics_trajectory=artifact.dynamics_trajectory,
                )

    logger.info("All scenarios complete.")


def run_report_workflow(*, run_dir: Path | None = None) -> Path | None:
    """Aggregate per-run outcomes into report summary CSV."""
    cfg = SimulationConfig()
    selected_run_dir = run_dir or (cfg.output_dir / "runs")

    if not selected_run_dir.exists():
        logger.error("Run directory not found: {d}", d=selected_run_dir)
        return None

    all_trajectories: list[pl.DataFrame] = []
    for run_path in sorted(selected_run_dir.iterdir()):
        if not run_path.is_dir():
            continue
        traj_path = run_path / "trajectories" / "outcomes.parquet"
        if traj_path.is_file():
            traj = pl.read_parquet(traj_path).with_columns(pl.lit(run_path.name).alias("run_id"))
            all_trajectories.append(traj)

    if not all_trajectories:
        logger.warning("No trajectory files found.")
        return None

    combined = pl.concat(all_trajectories, how="diagonal_relaxed")
    report_path = cfg.output_dir / "report_summary.csv"
    combined.write_csv(report_path)
    logger.info(
        "Report: {n} runs aggregated to {p}.",
        n=len(all_trajectories),
        p=report_path,
    )
    return report_path


def _parse_csv_list(value: str) -> tuple[str, ...]:
    entries = tuple(item.strip() for item in value.split(",") if item.strip())
    if not entries:
        raise ValueError("Expected a non-empty comma-separated list.")
    return entries


def run_full_sensitivity_pipeline_workflow(
    *,
    output_dir: Path = Path("abm/output/full_sensitivity_pipeline"),
    n_agents: int = 2_000,
    n_steps: int = 24,
    seed: int = 42,
    seed_offsets: tuple[int, ...] = (0, 1, 2),
    morris_trajectories: int = 8,
    morris_levels: int = 6,
    lhc_samples: int = 120,
    sobol_base_samples: int = 4096,
    shortlist_target: int = 8,
    shortlist_max: int = 10,
    sigma_quantile: float = 0.8,
    surrogate_min_r2: float = 0.30,
    max_workers: int = 0,
    outputs: tuple[str, ...] = DEFAULT_OUTPUT_METRICS,
    flagship_scenarios: tuple[str, ...] = DEFAULT_FLAGSHIP_SCENARIOS,
    effect_metric: SensitivityEffectMetric = "auc",
    persist_run_data: bool = True,
    resume_from_morris_raw: bool = False,
) -> None:
    """Run canonical full sensitivity pipeline from typed workflow inputs."""
    config = SensitivityPipelineConfig(
        output_dir=output_dir,
        n_agents=n_agents,
        n_steps=n_steps,
        seed_offsets=seed_offsets,
        base_seed=seed,
        morris_trajectories=morris_trajectories,
        morris_levels=morris_levels,
        lhc_samples=lhc_samples,
        sobol_base_samples=sobol_base_samples,
        shortlist_target=shortlist_target,
        shortlist_max=shortlist_max,
        shortlist_sigma_quantile=sigma_quantile,
        surrogate_min_r2=surrogate_min_r2,
        outputs=outputs,
        flagship_scenarios=flagship_scenarios,
        effect_metric=effect_metric,
        persist_run_data=persist_run_data,
        max_workers=max_workers,
        resume_from_morris_raw=resume_from_morris_raw,
    )
    start = time.perf_counter()
    run_full_sensitivity_pipeline(config)
    elapsed = time.perf_counter() - start
    logger.info("Full sensitivity pipeline completed in {elapsed:.1f}s", elapsed=elapsed)


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return 0.0
    x_clean = x[mask]
    y_clean = y[mask]
    if np.std(x_clean) <= 1e-12 or np.std(y_clean) <= 1e-12:
        return 0.0
    return float(np.corrcoef(x_clean, y_clean)[0, 1])


def _available_target_pairs(survey_df: pl.DataFrame) -> list[tuple[str, str]]:
    candidates = [
        ("progressive_attitudes_nonvax_idx", "friends_progressive_norms_nonvax_idx"),
        ("progressive_attitudes_nonvax_idx", "family_progressive_norms_nonvax_idx"),
        ("progressive_attitudes_nonvax_idx", "colleagues_progressive_norms_nonvax_idx"),
        ("att_vaccines_importance", "friends_vaccines_importance"),
        ("att_vaccines_importance", "family_vaccines_importance"),
        ("att_vaccines_importance", "colleagues_vaccines_importance"),
    ]
    available = []
    cols = set(survey_df.columns)
    for self_col, peer_col in candidates:
        if self_col in cols and peer_col in cols:
            available.append((self_col, peer_col))
    return available


def _compute_target_corr(survey_df: pl.DataFrame) -> float:
    pairs = _available_target_pairs(survey_df)
    if not pairs:
        raise ValueError("No friend/self proxy columns found for homophily calibration.")

    corrs: list[float] = []
    for self_col, peer_col in pairs:
        x = survey_df[self_col].cast(pl.Float64).to_numpy()
        y = survey_df[peer_col].cast(pl.Float64).to_numpy()
        corr = _pearson(x, y)
        if np.isfinite(corr):
            corrs.append(corr)
    if not corrs:
        raise ValueError("Friend/self proxy columns are present but correlations are undefined.")
    return float(np.mean(corrs))


def _compute_peer_corr(
    *,
    agents: pl.DataFrame,
    network_seed: int,
    gamma: float,
    bridge_prob: float,
) -> float:
    from beam_abm.abm.network import build_contact_network, compute_homophily_weighted_peer_means

    network = build_contact_network(agents, bridge_prob=bridge_prob, seed=network_seed)
    means, _ = compute_homophily_weighted_peer_means(
        agents,
        network,
        columns=("institutional_trust_avg", "moral_progressive_orientation"),
        homophily_gamma=gamma,
        homophily_w_trust=0.5,
        homophily_w_moral=0.5,
    )

    corrs: list[float] = []
    for col, peer_vals in means.items():
        if col not in agents.columns:
            continue
        self_vals = agents[col].cast(pl.Float64).to_numpy()
        corrs.append(_pearson(self_vals, peer_vals))
    if not corrs:
        return 0.0
    return float(np.mean(corrs))


def _calibrate_homophily_gamma(
    *,
    agents: pl.DataFrame,
    target_corr: float,
    bridge_prob: float,
    seed: int,
    gamma_min: float,
    gamma_max: float,
    gamma_steps: int,
) -> dict[str, object]:
    grid = np.linspace(gamma_min, gamma_max, gamma_steps)
    best_gamma = None
    best_error = float("inf")
    rows: list[dict[str, float]] = []

    for gamma in grid:
        peer_corr = _compute_peer_corr(
            agents=agents,
            network_seed=seed,
            gamma=float(gamma),
            bridge_prob=bridge_prob,
        )
        error = abs(peer_corr - target_corr)
        rows.append({"gamma": float(gamma), "peer_corr": peer_corr, "abs_error": float(error)})
        if error < best_error:
            best_error = error
            best_gamma = float(gamma)

    if best_gamma is None:
        raise RuntimeError("Gamma calibration failed to produce a candidate.")

    return {
        "target_corr": float(target_corr),
        "best_gamma": float(best_gamma),
        "grid": rows,
    }


def calibrate_homophily_gamma_workflow(
    *,
    n_agents: int = 4000,
    seed: int = 42,
    gamma_min: float = 0.2,
    gamma_max: float = 4.0,
    gamma_steps: int = 25,
    output: Path | None = None,
) -> dict[str, object]:
    """Calibrate homophily gamma against survey friend/self similarity."""
    from beam_abm.abm.data_contracts import load_anchor_system
    from beam_abm.abm.io import load_survey_data
    from beam_abm.abm.population import initialise_population

    cfg = SimulationConfig(n_agents=n_agents, seed=seed)
    survey_df = load_survey_data(cfg.survey_csv, cfg.countries)
    target_corr = _compute_target_corr(survey_df)

    anchor_system = load_anchor_system(cfg.anchors_dir)
    agents = initialise_population(
        n_agents=cfg.n_agents,
        countries=cfg.countries,
        anchor_system=anchor_system,
        survey_df=survey_df,
        seed=cfg.seed,
    )

    report = _calibrate_homophily_gamma(
        agents=agents,
        target_corr=target_corr,
        bridge_prob=cfg.constants.bridge_prob,
        seed=cfg.seed,
        gamma_min=gamma_min,
        gamma_max=gamma_max,
        gamma_steps=gamma_steps,
    )

    logger.info("Target correlation: {t:.3f}", t=report["target_corr"])
    logger.info("Recommended homophily_gamma: {g:.3f}", g=report["best_gamma"])

    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        logger.info("Saved calibration report to {p}", p=output)
    return report


def rebuild_heterogeneity_cdf_workflow(*, output: Path | None = None) -> Path:
    """Rebuild heterogeneity CDF knots from the survey population."""
    from beam_abm.abm.io import load_survey_data
    from beam_abm.abm.population import build_heterogeneity_cdf_knots, load_heterogeneity_loadings

    cfg = SimulationConfig()
    survey_df = load_survey_data(cfg.survey_csv, cfg.countries)
    loadings = load_heterogeneity_loadings()
    cdf_knots = build_heterogeneity_cdf_knots(
        survey_df,
        loadings,
        quantiles=loadings.cdf_knots.quantiles,
        source="survey",
    )

    target_path = output
    if target_path is None:
        target_path = (
            Path(__file__).resolve().parents[2] / "beam_abm" / "abm" / "assets" / "heterogeneity_loadings.json"
        )

    raw = json.loads(target_path.read_text(encoding="utf-8"))
    raw.setdefault("cdf_knots", {})
    raw["cdf_knots"]["source"] = cdf_knots.source
    raw["cdf_knots"]["quantiles"] = list(cdf_knots.quantiles)
    raw["cdf_knots"]["scores"] = {key: list(values) for key, values in sorted(cdf_knots.score_knots.items())}
    target_path.write_text(json.dumps(raw, indent=2), encoding="utf-8")
    logger.info("Wrote heterogeneity CDF knots to {p}", p=target_path)
    return target_path


def parse_csv_tuple(value: str) -> tuple[str, ...]:
    """Public helper for script wrappers that parse CSV CLI lists."""
    return _parse_csv_list(value)
