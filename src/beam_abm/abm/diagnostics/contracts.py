"""Typed contracts for ABM diagnostics orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

SuiteName = Literal["core", "importation", "credibility"]
SuiteSelector = Literal["all", "core", "importation", "credibility"]


@dataclass(slots=True, frozen=True)
class CoreDiagnosticsSummary:
    """Summary payload for the core diagnostics suite."""

    n_scenarios: int
    n_results: int
    n_gates: int
    n_failed_gates: int
    n_performance_benchmarks: int

    def to_dict(self) -> dict[str, object]:
        return {
            "n_scenarios": self.n_scenarios,
            "n_results": self.n_results,
            "n_gates": self.n_gates,
            "n_failed_gates": self.n_failed_gates,
            "n_performance_benchmarks": self.n_performance_benchmarks,
        }


@dataclass(slots=True, frozen=True)
class ImportationDiagnosticsSummary:
    """Summary payload for the importation diagnostics suite."""

    n_seeds: int
    n_path_activation_rows: int
    n_static_sensitivity_rows: int
    n_toggle_rows: int
    n_dose_rows: int
    timing_shift_observed: bool
    coupling_estimates: int

    def to_dict(self) -> dict[str, object]:
        return {
            "n_seeds": self.n_seeds,
            "n_path_activation_rows": self.n_path_activation_rows,
            "n_static_sensitivity_rows": self.n_static_sensitivity_rows,
            "n_toggle_rows": self.n_toggle_rows,
            "n_dose_rows": self.n_dose_rows,
            "timing_shift_observed": self.timing_shift_observed,
            "coupling_estimates": self.coupling_estimates,
        }


@dataclass(slots=True, frozen=True)
class CredibilityDiagnosticsSummary:
    """Summary payload for the credibility diagnostics suite."""

    n_substep_rows: int
    n_tick_rows: int
    n_ablation_rows: int
    n_pulse_rows: int
    n_long_horizon_rows: int
    n_invariant_checks: int
    n_failed_invariants: int

    def to_dict(self) -> dict[str, object]:
        return {
            "n_substep_rows": self.n_substep_rows,
            "n_tick_rows": self.n_tick_rows,
            "n_ablation_rows": self.n_ablation_rows,
            "n_pulse_rows": self.n_pulse_rows,
            "n_long_horizon_rows": self.n_long_horizon_rows,
            "n_invariant_checks": self.n_invariant_checks,
            "n_failed_invariants": self.n_failed_invariants,
        }


@dataclass(slots=True, frozen=True)
class DiagnosticsSuiteResult:
    """Outcome contract for an individual diagnostics suite."""

    suite: SuiteName
    passed: bool
    elapsed_seconds: float
    n_checks: int
    n_failed: int
    artifacts: tuple[str, ...] = field(default_factory=tuple)
    report_path: str = ""
    error: str | None = None
    summary: dict[str, object] | None = None

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "suite": self.suite,
            "passed": self.passed,
            "elapsed_seconds": self.elapsed_seconds,
            "n_checks": self.n_checks,
            "n_failed": self.n_failed,
            "artifacts": list(self.artifacts),
            "report_path": self.report_path,
        }
        if self.error is not None:
            payload["error"] = self.error
        if self.summary is not None:
            payload["summary"] = self.summary
        return payload


@dataclass(slots=True, frozen=True)
class DiagnosticsConfig:
    """Run configuration persisted in diagnostics summary."""

    suite: SuiteSelector
    n_agents: int
    n_steps: int
    n_reps: int
    seeds: tuple[int, ...]
    output_dir: str

    def to_dict(self) -> dict[str, object]:
        return {
            "suite": self.suite,
            "n_agents": self.n_agents,
            "n_steps": self.n_steps,
            "n_reps": self.n_reps,
            "seeds": list(self.seeds),
            "output_dir": self.output_dir,
        }


@dataclass(slots=True, frozen=True)
class DiagnosticsReport:
    """Top-level diagnostics report contract written to ``summary.json``."""

    run_id: str
    timestamp_utc: str
    config: DiagnosticsConfig
    suite_order: tuple[SuiteName, ...]
    suites: tuple[DiagnosticsSuiteResult, ...]
    elapsed_seconds_total: float

    @property
    def overall_passed(self) -> bool:
        return all(suite.passed for suite in self.suites)

    def to_dict(self) -> dict[str, object]:
        suites_payload: list[dict[str, object]] = [suite.to_dict() for suite in self.suites]
        return {
            "run_id": self.run_id,
            "timestamp_utc": self.timestamp_utc,
            "config": self.config.to_dict(),
            "overall_passed": self.overall_passed,
            "suite_order": list(self.suite_order),
            "suites": suites_payload,
            "elapsed_seconds_total": self.elapsed_seconds_total,
        }


def summary_to_dict(summary: object | None) -> dict[str, object] | None:
    """Normalize a suite summary into a plain dictionary."""
    if summary is None:
        return None
    if hasattr(summary, "to_dict"):
        return summary.to_dict()  # type: ignore[no-any-return]
    if isinstance(summary, dict):
        return {str(k): v for k, v in summary.items()}
    return {"value": str(summary)}
