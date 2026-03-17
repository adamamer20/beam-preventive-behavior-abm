"""Tests for diagnostics CLI parsing and dispatch semantics."""

from __future__ import annotations

from pathlib import Path

import pytest

from beam_abm.abm.cli import _build_parser, cmd_diagnostics
from beam_abm.abm.diagnostics.contracts import DiagnosticsConfig, DiagnosticsReport, DiagnosticsSuiteResult


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


def test_parser_registers_diagnostics_command() -> None:
    parser = _build_parser()
    args = parser.parse_args(["diagnostics"])

    assert args.command == "diagnostics"


def test_parser_diagnostics_defaults() -> None:
    parser = _build_parser()
    args = parser.parse_args(["diagnostics"])

    assert args.suite == "all"
    assert args.n_agents == 2000
    assert args.n_steps == 24
    assert args.n_reps == 3
    assert args.seeds == [42, 43, 44, 45, 46]
    assert args.output_dir == "abm/output/diagnostics"


def test_parser_diagnostics_explicit_suite() -> None:
    parser = _build_parser()
    args = parser.parse_args(
        [
            "diagnostics",
            "--suite",
            "credibility",
            "--n-agents",
            "100",
            "--n-steps",
            "6",
            "--n-reps",
            "2",
            "--seeds",
            "1",
            "2",
            "--output-dir",
            str(Path("tmp/out")),
        ]
    )

    assert args.suite == "credibility"
    assert args.n_agents == 100
    assert args.n_steps == 6
    assert args.n_reps == 2
    assert args.seeds == [1, 2]


def test_cmd_diagnostics_exits_non_zero_on_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    parser = _build_parser()
    args = parser.parse_args(["diagnostics"])
    monkeypatch.setattr("beam_abm.abm.diagnostics.run_diagnostics", lambda **_: _report(False))

    with pytest.raises(SystemExit) as exc:
        cmd_diagnostics(args)

    assert exc.value.code == 1
