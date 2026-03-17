"""Tests for diagnostics contracts and serialization shape."""

from __future__ import annotations

from beam_abm.abm.diagnostics.contracts import (
    DiagnosticsConfig,
    DiagnosticsReport,
    DiagnosticsSuiteResult,
)


def test_diagnostics_report_serialization_has_required_top_level_keys() -> None:
    suite = DiagnosticsSuiteResult(
        suite="core",
        passed=True,
        elapsed_seconds=1.2,
        n_checks=5,
        n_failed=0,
        artifacts=("core/report.json",),
        report_path="abm/output/diagnostics/run/core/report.json",
        summary={"n_gates": 1},
    )
    report = DiagnosticsReport(
        run_id="20260303T120000Z_deadbeef",
        timestamp_utc="2026-03-03T12:00:00Z",
        config=DiagnosticsConfig(
            suite="all",
            n_agents=2000,
            n_steps=24,
            n_reps=3,
            seeds=(42, 43, 44, 45, 46),
            output_dir="abm/output/diagnostics",
        ),
        suite_order=("core", "importation", "credibility"),
        suites=(suite,),
        elapsed_seconds_total=3.5,
    )

    payload = report.to_dict()

    assert set(payload) == {
        "run_id",
        "timestamp_utc",
        "config",
        "overall_passed",
        "suite_order",
        "suites",
        "elapsed_seconds_total",
    }
    assert payload["overall_passed"] is True


def test_suite_result_to_dict_includes_required_fields() -> None:
    suite = DiagnosticsSuiteResult(
        suite="importation",
        passed=False,
        elapsed_seconds=2.0,
        n_checks=6,
        n_failed=2,
        artifacts=("importation/report.json",),
        report_path="abm/output/diagnostics/run/importation/report.json",
        error="missing artifact",
        summary={"n_seeds": 5},
    )

    payload = suite.to_dict()

    assert payload["suite"] == "importation"
    assert payload["passed"] is False
    assert payload["n_checks"] == 6
    assert payload["n_failed"] == 2
    assert payload["artifacts"] == ["importation/report.json"]
    assert payload["report_path"].endswith("report.json")
    assert payload["error"] == "missing artifact"
    assert payload["summary"] == {"n_seeds": 5}
