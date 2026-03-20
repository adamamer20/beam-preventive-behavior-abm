"""Tests for ABM diagnostics workflow semantics."""

from __future__ import annotations

from pathlib import Path

from beam_abm.abm.diagnostics.contracts import DiagnosticsConfig, DiagnosticsReport, DiagnosticsSuiteResult
from beam_abm.abm.workflows import run_diagnostics_workflow


def _report(passed: bool) -> DiagnosticsReport:
    suite = DiagnosticsSuiteResult(
        suite="core",
        passed=passed,
        elapsed_seconds=0.1,
        n_checks=1,
        n_failed=0 if passed else 1,
        artifacts=("core/report.json",),
        report_path="core/report.json",
        summary={"x": 1},
    )
    return DiagnosticsReport(
        run_id="run",
        timestamp_utc="2026-03-03T00:00:00Z",
        config=DiagnosticsConfig(
            suite="all",
            n_agents=2000,
            n_steps=24,
            n_reps=3,
            seeds=(42, 43, 44, 45, 46),
            output_dir="abm/output/diagnostics",
        ),
        suite_order=("core",),
        suites=(suite,),
        elapsed_seconds_total=0.1,
    )


def test_workflow_diagnostics_passes_args(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def _stub_run_diagnostics(**kwargs):
        captured.update(kwargs)
        return _report(True)

    monkeypatch.setattr("beam_abm.abm.diagnostics.run_diagnostics", _stub_run_diagnostics)
    report = run_diagnostics_workflow(
        suite="core",
        n_agents=123,
        n_steps=8,
        n_reps=2,
        seeds=(10, 11),
        output_dir=Path("tmp/diag"),
    )

    assert report.overall_passed is True
    assert captured["suite"] == "core"
    assert captured["n_agents"] == 123
    assert captured["n_steps"] == 8
    assert captured["n_reps"] == 2
    assert captured["seeds"] == [10, 11]
    assert captured["output_dir"] == Path("tmp/diag")


def test_workflow_diagnostics_returns_failure_report(monkeypatch) -> None:
    monkeypatch.setattr("beam_abm.abm.diagnostics.run_diagnostics", lambda **_: _report(False))
    report = run_diagnostics_workflow()
    assert report.overall_passed is False
