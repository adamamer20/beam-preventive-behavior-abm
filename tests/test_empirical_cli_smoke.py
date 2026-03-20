from __future__ import annotations

import runpy
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("script_rel", "argv", "workflow_module", "workflow_func"),
    [
        (
            "empirical/scripts/compute_descriptives.py",
            ["--input", "preprocess/output/clean_processed_survey.csv"],
            "beam_abm.empirical.workflows.descriptives",
            "run_descriptives_workflow",
        ),
        (
            "empirical/scripts/inferential_analysis.py",
            ["--input", "preprocess/output/clean_processed_survey.csv"],
            "beam_abm.empirical.workflows.inferential",
            "run_inferential_workflow",
        ),
        (
            "empirical/scripts/modeling.py",
            ["fit", "--input", "preprocess/output/clean_processed_survey.csv"],
            "beam_abm.empirical.workflows.modeling",
            "run_modeling_fit_workflow",
        ),
        (
            "empirical/scripts/modeling.py",
            ["block-importance", "--input", "preprocess/output/clean_processed_survey.csv"],
            "beam_abm.empirical.workflows.modeling",
            "run_modeling_block_importance_workflow",
        ),
        (
            "empirical/scripts/modeling.py",
            ["run", "--input", "preprocess/output/clean_processed_survey.csv"],
            "beam_abm.empirical.workflows.modeling",
            "run_modeling_workflow",
        ),
        (
            "empirical/scripts/anchors.py",
            ["build", "--input", "preprocess/output/clean_processed_survey.csv"],
            "beam_abm.empirical.workflows.anchors",
            "run_anchor_build_workflow",
        ),
        (
            "empirical/scripts/anchors.py",
            ["diagnostics", "--input", "preprocess/output/clean_processed_survey.csv"],
            "beam_abm.empirical.workflows.anchors",
            "run_anchor_diagnostics_workflow",
        ),
        (
            "empirical/scripts/anchors.py",
            [
                "pe",
                "--input",
                "preprocess/output/clean_processed_survey.csv",
                "--target-col",
                "protective_behaviour_avg",
                "--driver-cols",
                "age",
                "--levers",
                "age",
            ],
            "beam_abm.empirical.workflows.anchors",
            "run_anchor_pe_workflow",
        ),
        (
            "empirical/scripts/anchors.py",
            [
                "pe-full",
                "--input",
                "preprocess/output/clean_processed_survey.csv",
                "--outdir",
                "empirical/output/anchors/pe/test_full",
                "--target-col",
                "protective_behaviour_avg",
                "--driver-cols",
                "age",
                "--levers",
                "age",
            ],
            "beam_abm.empirical.workflows.anchors",
            "run_anchor_full_pe_workflow",
        ),
        (
            "empirical/scripts/export_thesis_artifacts.py",
            ["--repo-root", str(REPO_ROOT)],
            "beam_abm.empirical.export",
            "export_thesis_artifacts",
        ),
    ],
)
def test_empirical_cli_dispatches_to_workflow(
    monkeypatch: pytest.MonkeyPatch,
    script_rel: str,
    argv: list[str],
    workflow_module: str,
    workflow_func: str,
) -> None:
    calls: list[dict[str, Any]] = []

    def _stub(*args: Any, **kwargs: Any) -> None:
        calls.append({"args": args, "kwargs": kwargs})

    module = __import__(workflow_module, fromlist=[workflow_func])
    monkeypatch.setattr(module, workflow_func, _stub)

    script_path = REPO_ROOT / script_rel
    old_argv = sys.argv[:]
    try:
        sys.argv = [str(script_path), *argv]
        try:
            runpy.run_path(str(script_path), run_name="__main__")
        except SystemExit as exc:
            assert exc.code == 0
    finally:
        sys.argv = old_argv

    assert len(calls) == 1
