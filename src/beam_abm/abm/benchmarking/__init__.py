"""Benchmarking and validation gates."""

from __future__ import annotations

from .core import (
    BenchmarkReport,
    GateCheck,
    GateResult,
    GateSuiteResult,
    MutableSummaryRow,
    NonDegeneracyDomainSummary,
    NonDegeneracyReport,
    PerformanceBenchmarkEntry,
    PerformanceReport,
    PopulationDiagnosticsReport,
    QuantileSummaryRow,
    build_non_degeneracy_report,
    load_gradient_specs_from_modeling,
    run_baseline_replication_gate,
    run_population_shift_validation_gate,
)
from .dynamic import run_dynamic_plausibility_gate
from .mechanical import run_mechanical_verification_gate
from .network import run_network_validation_gate
from .report import run_all_gates, run_performance_benchmarks
from .stochasticity import run_stochasticity_policy_gate

__all__ = [
    "GateCheck",
    "GateResult",
    "GateSuiteResult",
    "NonDegeneracyDomainSummary",
    "NonDegeneracyReport",
    "BenchmarkReport",
    "PopulationDiagnosticsReport",
    "MutableSummaryRow",
    "QuantileSummaryRow",
    "PerformanceBenchmarkEntry",
    "PerformanceReport",
    "run_baseline_replication_gate",
    "run_mechanical_verification_gate",
    "run_dynamic_plausibility_gate",
    "run_network_validation_gate",
    "run_stochasticity_policy_gate",
    "run_population_shift_validation_gate",
    "build_non_degeneracy_report",
    "run_all_gates",
    "run_performance_benchmarks",
    "load_gradient_specs_from_modeling",
]
