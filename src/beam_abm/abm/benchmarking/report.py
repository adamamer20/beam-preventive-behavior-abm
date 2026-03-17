"""Benchmarking orchestration and performance reporting."""

from __future__ import annotations

import time
import tracemalloc
from pathlib import Path

import polars as pl
from loguru import logger

from beam_abm.abm.config import SimulationConfig
from beam_abm.abm.data_contracts import AnchorSystem, BackboneModel, LeversConfig, get_abm_levers_config
from beam_abm.abm.runtime import run_simulation
from beam_abm.abm.scenario_defs import SCENARIO_LIBRARY

from .core import (
    GateSuiteResult,
    PerformanceBenchmarkEntry,
    PerformanceReport,
    run_baseline_replication_gate,
)
from .dynamic import run_dynamic_plausibility_gate
from .mechanical import run_mechanical_verification_gate
from .network import run_network_validation_gate
from .stochasticity import run_stochasticity_policy_gate


def run_all_gates(
    agents_t0: pl.DataFrame,
    survey_df: pl.DataFrame,
    models: dict[str, BackboneModel],
    levers_cfg: LeversConfig,
    anchor_system: AnchorSystem,
    results: list,
    countries: list[str],
    model_coefficients_path: Path,
    cfg: SimulationConfig | None = None,
) -> GateSuiteResult:
    """Execute all three validation gate suites.

    Parameters
    ----------
    agents_t0 : pl.DataFrame
        Initial agent population.
    survey_df : pl.DataFrame
        Original survey data.
    models : dict[str, BackboneModel]
        Loaded backbone models.
    levers_cfg : LeversConfig
        Lever configuration.
    results : list[SimulationResult]
        Simulation results for plausibility checks.
    countries : list[str]
        Countries included.

    Returns
    -------
    GateSuiteResult
        Container holding all gate results.
    """
    levers_cfg = get_abm_levers_config(levers_cfg)
    gates = (
        run_baseline_replication_gate(
            agents_t0,
            survey_df,
            models,
            levers_cfg,
            anchor_system,
            countries,
            model_coefficients_path=model_coefficients_path,
        ),
        run_mechanical_verification_gate(agents_t0, models, levers_cfg, cfg=cfg),
        run_dynamic_plausibility_gate(results),
        run_network_validation_gate(agents_t0),
        run_stochasticity_policy_gate(results),
    )

    suite = GateSuiteResult(gates=gates)
    all_passed = suite.passed
    logger.info(
        "All gates: {status}.",
        status="PASSED" if all_passed else "SOME FAILED",
    )
    return suite


def run_performance_benchmarks(
    cfg: SimulationConfig,
    n_agents_list: list[int] | None = None,
) -> PerformanceReport:
    """Run performance benchmarks and report timing/memory.

    Parameters
    ----------
    cfg : SimulationConfig
        Base configuration.
    n_agents_list : list[int] or None
        Agent counts to benchmark.

    Returns
    -------
    PerformanceReport
        Timing and memory metrics.
    """

    if n_agents_list is None:
        n_agents_list = [1_000, 5_000, 10_000]

    results = PerformanceReport()
    baseline = SCENARIO_LIBRARY["baseline"]

    for n_agents in n_agents_list:
        bench_cfg = SimulationConfig(
            n_agents=n_agents,
            n_steps=52,
            n_reps=1,
            seed=cfg.seed,
            deterministic=True,
        )

        tracemalloc.start()
        t0 = time.perf_counter()

        run_simulation(bench_cfg, baseline)

        elapsed = time.perf_counter() - t0
        _, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        entry = PerformanceBenchmarkEntry(
            n_agents=n_agents,
            n_steps=52,
            elapsed_s=float(elapsed),
            peak_memory_mb=float(peak_mem / (1024 * 1024)),
            per_step_ms=float((elapsed / 52) * 1000),
        )
        results.benchmarks.append(entry)
        logger.info(
            "Benchmark {n} agents: {t:.2f}s, {mem:.1f} MB peak.",
            n=n_agents,
            t=elapsed,
            mem=entry.peak_memory_mb,
        )

    return results
