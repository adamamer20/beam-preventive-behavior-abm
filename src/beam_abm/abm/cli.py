"""CLI entrypoints for ABM operations.

Exposes commands for population building, benchmarking, scenario
execution, and report aggregation.  All commands route through
``uv run`` in the Makefile.

Examples
--------
.. code-block:: bash

    uv run python -m beam_abm.abm.cli build-pop
    uv run python -m beam_abm.abm.cli diagnostics --suite all
    uv run python -m beam_abm.abm.cli run --scenario outbreak_salience
    uv run python -m beam_abm.abm.cli report
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Literal

from loguru import logger

from beam_abm.abm.config import SimulationConfig

__all__ = ["main"]


def _build_parser() -> argparse.ArgumentParser:
    """Construct the CLI argument parser.

    Returns
    -------
    argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser(
        prog="beam_abm.abm.cli",
        description="Agent-Based Model CLI for vaccine behaviour simulation.",
    )
    sub = parser.add_subparsers(dest="command", help="Available commands.")

    # -- build-pop --
    bp = sub.add_parser(
        "build-pop",
        help="Initialise baseline population artefacts.",
    )
    bp.add_argument(
        "--n-agents",
        type=int,
        default=10_000,
        help="Number of agents.",
    )
    bp.add_argument("--seed", type=int, default=42, help="Random seed.")
    bp.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output Parquet path (default: output/populations/baseline.parquet).",
    )

    # -- benchmark --
    bm = sub.add_parser(
        "benchmark",
        help="Run validation gates and performance benchmarks.",
    )
    bm.add_argument(
        "--n-agents",
        type=int,
        nargs="+",
        default=[1_000, 5_000, 10_000],
        help="Agent counts for benchmarking.",
    )
    bm.add_argument(
        "--n-steps",
        type=int,
        default=52,
        help="Simulation steps for validation gates.",
    )
    bm.add_argument(
        "--n-reps",
        type=int,
        default=3,
        help="Replications for validation gates.",
    )
    bm.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON path for benchmark report.",
    )

    # -- diagnostics --
    dg = sub.add_parser(
        "diagnostics",
        help="Run unified ABM diagnostics suites (core/importation/credibility).",
    )
    dg.add_argument(
        "--suite",
        type=str,
        choices=["all", "core", "importation", "credibility"],
        default="all",
        help="Diagnostics suite selection.",
    )
    dg.add_argument(
        "--n-agents",
        type=int,
        default=2_000,
        help="Number of agents used in diagnostics runtime.",
    )
    dg.add_argument(
        "--n-steps",
        type=int,
        default=24,
        help="Simulation steps for diagnostics runs.",
    )
    dg.add_argument(
        "--n-reps",
        type=int,
        default=3,
        help="Replications for diagnostics (used by core suite).",
    )
    dg.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42, 43, 44, 45, 46],
        help="Seed list for diagnostics suites.",
    )
    dg.add_argument(
        "--output-dir",
        type=str,
        default="abm/output/diagnostics",
        help="Diagnostics artifact root directory.",
    )

    # -- run --
    rn = sub.add_parser("run", help="Execute scenario(s).")
    rn.add_argument(
        "--scenario",
        type=str,
        nargs="+",
        default=["baseline"],
        help="Scenario key(s) from the library.",
    )
    rn.add_argument("--n-agents", type=int, default=10_000, help="Agents.")
    rn.add_argument("--n-steps", type=int, default=52, help="Simulation steps.")
    rn.add_argument("--n-reps", type=int, default=10, help="Replications.")
    rn.add_argument(
        "--rep-workers",
        type=int,
        default=0,
        help="Parallel workers across reps (0=auto, 1=sequential).",
    )
    rn.add_argument("--seed", type=int, default=42, help="Random seed.")
    rn.add_argument(
        "--society",
        type=str,
        default=None,
        help="Society key from SOCIETY_LIBRARY (optional).",
    )
    rn.add_argument(
        "--pop-mode",
        type=str,
        default=None,
        choices=["survey", "resample_on_manifold", "parquet"],
        help="Population construction mode (optional).",
    )
    rn.add_argument(
        "--pop-n",
        type=int,
        default=None,
        help="Override population size (optional).",
    )
    rn.add_argument(
        "--pop-parquet",
        type=str,
        default=None,
        help="Parquet path when --pop-mode=parquet (optional).",
    )
    rn.add_argument(
        "--deterministic",
        action="store_true",
        help="Deterministic mode (expected values).",
    )
    rn.add_argument(
        "--belief-surrogate-dir",
        type=str,
        default=None,
        help="Directory containing trained belief-update surrogate bundle (manifest.json + models/).",
    )
    rn.add_argument(
        "--effect-regime",
        type=str,
        choices=["perturbation", "shock", "both"],
        default="perturbation",
        help=(
            "Signal effect regime for scenario execution: 'perturbation' (paired Q25/Q75 worlds with high-low contrast), "
            "'shock' (signed 1-IQR vs matched baseline), or 'both'."
        ),
    )

    # -- report --
    rp = sub.add_parser("report", help="Aggregate trajectory/ATE summaries.")
    rp.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="Directory containing run outputs.",
    )

    return parser


def cmd_build_pop(args: argparse.Namespace) -> None:
    """Handle ``build-pop`` command.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments.
    """
    from beam_abm.abm.benchmarking import run_baseline_replication_gate
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

    cfg = SimulationConfig(n_agents=args.n_agents, seed=args.seed)
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

    out_path = Path(args.output or (cfg.output_dir / "populations" / "baseline.parquet"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    agents.write_parquet(out_path)
    logger.info("Population saved to {p}.", p=out_path)

    # Population diagnostics required by methodology:
    # baseline replication checks + summary distribution diagnostics.
    levers_cfg = load_levers_config(cfg.levers_path)
    log_non_abm_outcomes(levers_cfg, context="cmd_build_pop")
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
    from beam_abm.abm.benchmarking import MutableSummaryRow, PopulationDiagnosticsReport, QuantileSummaryRow

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


def cmd_benchmark(args: argparse.Namespace) -> None:
    """Handle ``benchmark`` command.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments.
    """
    from beam_abm.abm.benchmarking import (
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
        n_agents=args.n_agents[0],
        n_steps=args.n_steps,
        n_reps=args.n_reps,
        deterministic=True,
    )

    # Validation gates
    levers_cfg = load_levers_config(cfg.levers_path)
    log_non_abm_outcomes(levers_cfg, context="cmd_benchmark")
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

    # Performance benchmarks
    perf = run_performance_benchmarks(cfg, args.n_agents)

    # Compile report
    from beam_abm.abm.benchmarking import BenchmarkReport

    non_degeneracy = build_non_degeneracy_report(gates.gates)
    report = BenchmarkReport(
        gates=gates.gates,
        performance=perf,
        non_degeneracy=non_degeneracy,
    )

    out_path = Path(args.output or (cfg.output_dir / "benchmarks" / "report.json"))
    save_benchmark_report(report, out_path)
    logger.info(
        "Non-degeneracy initial checks: {status} (failed_domains={n_failed}, incomplete_domains={n_incomplete}).",
        status=non_degeneracy.status.upper(),
        n_failed=non_degeneracy.n_failed_domains,
        n_incomplete=non_degeneracy.n_incomplete_domains,
    )


def cmd_diagnostics(args: argparse.Namespace) -> None:
    """Handle ``diagnostics`` command."""
    from beam_abm.abm.diagnostics import run_diagnostics

    report = run_diagnostics(
        suite=args.suite,
        n_agents=int(args.n_agents),
        n_steps=int(args.n_steps),
        n_reps=int(args.n_reps),
        seeds=[int(seed) for seed in args.seeds],
        output_dir=Path(args.output_dir),
    )
    summary_path = Path(args.output_dir) / report.run_id / "summary.json"
    logger.info(
        "Diagnostics complete: run_id={run_id}, overall_passed={passed}, summary={summary}",
        run_id=report.run_id,
        passed=report.overall_passed,
        summary=summary_path,
    )
    if not report.overall_passed:
        sys.exit(1)


def cmd_run(args: argparse.Namespace) -> None:
    """Handle ``run`` command.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments.
    """
    from beam_abm.abm.data_contracts import (
        get_abm_levers_config,
        load_all_backbone_models,
        load_levers_config,
        log_non_abm_outcomes,
    )
    from beam_abm.abm.effect_regimes import RegimeArtifact, run_perturbation_paired, run_shock_single
    from beam_abm.abm.io import RunPaths, load_survey_data, save_run_outputs
    from beam_abm.abm.metrics import build_behavioral_projection_weights
    from beam_abm.abm.population_specs import PopulationSpec, get_society
    from beam_abm.abm.scenario_defs import (
        get_scenario,
        get_scenario_library,
        scenario_to_dict,
    )

    cfg = SimulationConfig(
        n_agents=args.n_agents,
        n_steps=args.n_steps,
        n_reps=args.n_reps,
        seed=args.seed,
        deterministic=args.deterministic,
        belief_update_surrogate_dir=(Path(args.belief_surrogate_dir) if args.belief_surrogate_dir else None),
    )
    cfg.log_summary()

    population_spec = None
    society_spec = None
    if args.society or args.pop_mode or args.pop_n or args.pop_parquet:
        if args.society:
            society_spec = get_society(args.society)
        pop_kwargs: dict[str, object] = {}
        if args.pop_mode:
            pop_kwargs["mode"] = args.pop_mode
        if args.pop_n is not None:
            pop_kwargs["n_agents"] = args.pop_n
        if args.pop_parquet:
            pop_kwargs["parquet_path"] = Path(args.pop_parquet)
        pop_kwargs["seed"] = args.seed
        population_spec = PopulationSpec(**pop_kwargs)

    scenario_keys: list[str] = []
    seen_scenarios: set[str] = set()
    for scenario_key in args.scenario:
        if scenario_key in seen_scenarios:
            logger.info("Skipping duplicate scenario request: {scenario}.", scenario=scenario_key)
            continue
        seen_scenarios.add(scenario_key)
        scenario_keys.append(scenario_key)

    survey_df = load_survey_data(cfg.survey_csv, cfg.countries)
    levers_cfg = load_levers_config(cfg.levers_path)
    log_non_abm_outcomes(levers_cfg, context="cmd_run")
    levers_cfg = get_abm_levers_config(levers_cfg)
    backbone_models = load_all_backbone_models(cfg.modeling_dir, levers_cfg, survey_df=survey_df)
    projection_weights = build_behavioral_projection_weights(backbone_models)

    def _build_run_metadata_payload(
        *,
        artifact: RegimeArtifact,
        simulation_cfg: SimulationConfig,
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

    scenario_library = get_scenario_library(include_mutable=True)
    baseline_results_cache: dict[str, list] = {}
    regimes_to_run: tuple[Literal["perturbation", "shock"], ...]
    if args.effect_regime == "both":
        regimes_to_run = ("perturbation", "shock")
    else:
        regimes_to_run = (args.effect_regime,)

    for scenario_key in scenario_keys:
        scenario = get_scenario(scenario_key)
        for regime in regimes_to_run:
            logger.info("Running scenario={name} under effect_regime={regime}.", name=scenario.name, regime=regime)
            cfg_run = cfg.model_copy(deep=True)
            cfg_run.effect_regime = regime
            cfg_run.perturbation_world = None

            if regime == "perturbation":
                artifacts = run_perturbation_paired(
                    cfg=cfg_run,
                    scenario_key=scenario_key,
                    scenario=scenario,
                    population_spec=population_spec,
                    society_spec=society_spec,
                    rep_workers=args.rep_workers,
                    projection_weights=projection_weights,
                )
            else:
                artifacts = run_shock_single(
                    cfg=cfg_run,
                    scenario_key=scenario_key,
                    scenario=scenario,
                    scenario_library=scenario_library,
                    baseline_results_cache=baseline_results_cache,
                    population_spec=population_spec,
                    society_spec=society_spec,
                    rep_workers=args.rep_workers,
                    projection_weights=projection_weights,
                )

            for artifact in artifacts:
                paths = RunPaths(cfg.output_dir, artifact.run_id)
                metadata_payload = _build_run_metadata_payload(
                    artifact=artifact,
                    simulation_cfg=cfg_run,
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


def cmd_report(args: argparse.Namespace) -> None:
    """Handle ``report`` command.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments.
    """
    import polars as pl

    cfg = SimulationConfig()
    run_dir = Path(args.run_dir or (cfg.output_dir / "runs"))

    if not run_dir.exists():
        logger.error("Run directory not found: {d}", d=run_dir)
        return

    # Aggregate trajectory summaries across runs
    all_trajectories: list[pl.DataFrame] = []
    for run_path in sorted(run_dir.iterdir()):
        if not run_path.is_dir():
            continue
        traj_path = run_path / "trajectories" / "outcomes.parquet"
        if traj_path.is_file():
            traj = pl.read_parquet(traj_path)
            traj = traj.with_columns(pl.lit(run_path.name).alias("run_id"))
            all_trajectories.append(traj)

    if not all_trajectories:
        logger.warning("No trajectory files found.")
        return

    combined = pl.concat(all_trajectories, how="diagonal_relaxed")
    report_path = cfg.output_dir / "report_summary.csv"
    combined.write_csv(report_path)
    logger.info(
        "Report: {n} runs aggregated to {p}.",
        n=len(all_trajectories),
        p=report_path,
    )


def main(argv: list[str] | None = None) -> None:
    """CLI entry point.

    Parameters
    ----------
    argv : list[str] or None
        Command-line arguments (defaults to ``sys.argv[1:]``).
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    dispatch = {
        "build-pop": cmd_build_pop,
        "benchmark": cmd_benchmark,
        "diagnostics": cmd_diagnostics,
        "run": cmd_run,
        "report": cmd_report,
    }

    handler = dispatch.get(args.command)
    if handler is None:
        logger.error("Unknown command: {cmd}", cmd=args.command)
        sys.exit(1)

    handler(args)


if __name__ == "__main__":
    main()
