"""Core diagnostics suite adapter over benchmarking gates."""

from __future__ import annotations

import time
from pathlib import Path

from beam_abm.abm.benchmarking import (
    BenchmarkReport,
    build_non_degeneracy_report,
    run_all_gates,
    run_performance_benchmarks,
)
from beam_abm.abm.config import SimulationConfig
from beam_abm.abm.diagnostics.contracts import (
    CoreDiagnosticsSummary,
    DiagnosticsSuiteResult,
    summary_to_dict,
)
from beam_abm.abm.io import save_benchmark_report, write_json
from beam_abm.abm.population_specs import PopulationSpec, build_population
from beam_abm.abm.runtime import run_simulation
from beam_abm.abm.scenario_defs import get_scenario_library, iter_scenario_regime_runs

from .runtime import DiagnosticsRuntime

__all__ = ["run_core_suite"]


def run_core_suite(
    *,
    runtime: DiagnosticsRuntime,
    suite_output_dir: Path,
    n_agents: int,
    n_steps: int,
    n_reps: int,
    seeds: list[int],
) -> DiagnosticsSuiteResult:
    """Run the core benchmarking and gate suite as diagnostics."""
    start = time.perf_counter()

    cfg = SimulationConfig(
        n_agents=n_agents,
        n_steps=n_steps,
        n_reps=n_reps,
        seed=int(seeds[0]),
        deterministic=True,
    )

    baseline_population_gate = PopulationSpec(
        mode="survey",
        n_agents=runtime.survey_df.height,
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
        anchor_system=runtime.anchor_system,
        survey_df=runtime.survey_df,
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
        results.extend(
            run_simulation(
                cfg_run,
                scenario,
                population_spec=baseline_population_runtime,
            )
        )

    gates = run_all_gates(
        agents_t0,
        runtime.survey_df,
        runtime.models,
        runtime.levers_cfg,
        runtime.anchor_system,
        results,
        cfg.countries,
        cfg.modeling_dir / "model_coefficients_all.csv",
        cfg=cfg,
    )
    perf = run_performance_benchmarks(cfg, [n_agents])
    non_degeneracy = build_non_degeneracy_report(gates.gates)
    report = BenchmarkReport(
        gates=gates.gates,
        performance=perf,
        non_degeneracy=non_degeneracy,
    )

    suite_output_dir.mkdir(parents=True, exist_ok=True)
    report_path = suite_output_dir / "report.json"
    save_benchmark_report(report, report_path)

    n_checks = int(sum(len(gate.checks) for gate in gates.gates))
    n_failed = int(sum(sum(1 for check in gate.checks if not check.passed) for gate in gates.gates))
    summary = CoreDiagnosticsSummary(
        n_scenarios=len(benchmark_scenarios),
        n_results=len(results),
        n_gates=len(gates.gates),
        n_failed_gates=sum(1 for gate in gates.gates if not gate.passed),
        n_performance_benchmarks=len(perf.benchmarks),
    )
    suite_report = {
        "suite": "core",
        "passed": bool(gates.passed),
        "summary": summary.to_dict(),
        "report": report.to_dict(),
    }
    suite_report_path = suite_output_dir / "suite_report.json"
    write_json(suite_report, suite_report_path)

    elapsed = time.perf_counter() - start
    return DiagnosticsSuiteResult(
        suite="core",
        passed=bool(gates.passed),
        elapsed_seconds=float(elapsed),
        n_checks=n_checks,
        n_failed=n_failed,
        artifacts=("report.json", "suite_report.json"),
        report_path=str(report_path),
        summary=summary_to_dict(summary),
    )
