"""Tests for diagnostics orchestrator control flow."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from beam_abm.abm.diagnostics.contracts import DiagnosticsSuiteResult
from beam_abm.abm.diagnostics.orchestrator import run_diagnostics


class _RuntimeStub:
    pass


def _suite_result(suite: str, passed: bool = True) -> DiagnosticsSuiteResult:
    return DiagnosticsSuiteResult(
        suite=suite,  # type: ignore[arg-type]
        passed=passed,
        elapsed_seconds=0.1,
        n_checks=3,
        n_failed=0 if passed else 1,
        artifacts=(f"{suite}/report.json",),
        report_path=f"{suite}/report.json",
        summary={"suite": suite},
    )


def test_orchestrator_all_pass(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr("beam_abm.abm.diagnostics.orchestrator.load_shared_runtime", lambda **_: _RuntimeStub())
    monkeypatch.setattr("beam_abm.abm.diagnostics.orchestrator.run_core_suite", lambda **_: _suite_result("core", True))
    monkeypatch.setattr(
        "beam_abm.abm.diagnostics.orchestrator.run_importation_suite",
        lambda **_: _suite_result("importation", True),
    )
    monkeypatch.setattr(
        "beam_abm.abm.diagnostics.orchestrator.run_credibility_suite",
        lambda **_: _suite_result("credibility", True),
    )

    report = run_diagnostics(
        suite="all",
        n_agents=2000,
        n_steps=24,
        n_reps=3,
        seeds=[42, 43, 44, 45, 46],
        output_dir=tmp_path,
    )

    assert report.overall_passed is True
    assert report.suite_order == ("core", "importation", "credibility")
    summary_path = tmp_path / report.run_id / "summary.json"
    assert summary_path.exists()


def test_orchestrator_one_suite_fail(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr("beam_abm.abm.diagnostics.orchestrator.load_shared_runtime", lambda **_: _RuntimeStub())
    monkeypatch.setattr("beam_abm.abm.diagnostics.orchestrator.run_core_suite", lambda **_: _suite_result("core", True))
    monkeypatch.setattr(
        "beam_abm.abm.diagnostics.orchestrator.run_importation_suite",
        lambda **_: _suite_result("importation", False),
    )
    monkeypatch.setattr(
        "beam_abm.abm.diagnostics.orchestrator.run_credibility_suite",
        lambda **_: _suite_result("credibility", True),
    )

    report = run_diagnostics(
        suite="all",
        n_agents=2000,
        n_steps=24,
        n_reps=3,
        seeds=[42, 43, 44, 45, 46],
        output_dir=tmp_path,
    )

    assert report.overall_passed is False
    assert any(not suite.passed for suite in report.suites)


def test_orchestrator_suite_exception_continues(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr("beam_abm.abm.diagnostics.orchestrator.load_shared_runtime", lambda **_: _RuntimeStub())
    monkeypatch.setattr("beam_abm.abm.diagnostics.orchestrator.run_core_suite", lambda **_: _suite_result("core", True))

    def _boom(**_: object) -> DiagnosticsSuiteResult:
        raise RuntimeError("boom")

    monkeypatch.setattr("beam_abm.abm.diagnostics.orchestrator.run_importation_suite", _boom)
    monkeypatch.setattr(
        "beam_abm.abm.diagnostics.orchestrator.run_credibility_suite",
        lambda **_: _suite_result("credibility", True),
    )

    report = run_diagnostics(
        suite="all",
        n_agents=2000,
        n_steps=24,
        n_reps=3,
        seeds=[42, 43, 44, 45, 46],
        output_dir=tmp_path,
    )

    assert report.overall_passed is False
    importation = [suite for suite in report.suites if suite.suite == "importation"][0]
    assert importation.error is not None


def test_orchestrator_subset_suite_selection(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[str] = []
    monkeypatch.setattr("beam_abm.abm.diagnostics.orchestrator.load_shared_runtime", lambda **_: _RuntimeStub())

    def _core(**_: object) -> DiagnosticsSuiteResult:
        calls.append("core")
        return _suite_result("core", True)

    monkeypatch.setattr("beam_abm.abm.diagnostics.orchestrator.run_core_suite", _core)
    monkeypatch.setattr(
        "beam_abm.abm.diagnostics.orchestrator.run_importation_suite",
        lambda **_: _suite_result("importation", True),
    )
    monkeypatch.setattr(
        "beam_abm.abm.diagnostics.orchestrator.run_credibility_suite",
        lambda **_: _suite_result("credibility", True),
    )

    report = run_diagnostics(
        suite="core",
        n_agents=2000,
        n_steps=24,
        n_reps=3,
        seeds=[42, 43, 44, 45, 46],
        output_dir=tmp_path,
    )

    assert report.suite_order == ("core",)
    assert calls == ["core"]
    assert len(report.suites) == 1


def test_orchestrator_artifact_path_shape(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr("beam_abm.abm.diagnostics.orchestrator.load_shared_runtime", lambda **_: _RuntimeStub())
    monkeypatch.setattr("beam_abm.abm.diagnostics.orchestrator.run_core_suite", lambda **_: _suite_result("core", True))

    report = run_diagnostics(
        suite="core",
        n_agents=2000,
        n_steps=24,
        n_reps=3,
        seeds=[42, 43, 44, 45, 46],
        output_dir=tmp_path,
    )

    run_dir = tmp_path / report.run_id
    assert (run_dir / "summary.json").exists()
    assert (run_dir / "core").exists()

    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["run_id"] == report.run_id
    assert summary["suite_order"] == ["core"]
