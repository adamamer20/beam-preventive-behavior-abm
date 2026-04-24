from __future__ import annotations

import runpy
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("script_rel", "argv", "module_name", "func_name"),
    [
        (
            "evaluation/scripts/run_behavioural_outcomes_baseline.py",
            [],
            "beam_abm.llm_microvalidation.workflows.behavioural_outcomes",
            "run_behavioural_outcomes_baseline_reconstruction",
        ),
        (
            "evaluation/scripts/run_behavioural_outcomes_perturbed.py",
            [],
            "beam_abm.llm_microvalidation.workflows.behavioural_outcomes",
            "run_behavioural_outcomes_perturbation_response",
        ),
        (
            "evaluation/scripts/run_psychological_profiles_baseline.py",
            [],
            "beam_abm.llm_microvalidation.workflows.psychological_profiles",
            "run_psychological_profiles_baseline_reconstruction",
        ),
        (
            "evaluation/scripts/run_psychological_profiles_perturbed.py",
            [],
            "beam_abm.llm_microvalidation.workflows.psychological_profiles",
            "run_psychological_profiles_perturbation_response",
        ),
        (
            "evaluation/scripts/run_psychological_profile_reference_effects.py",
            [],
            "beam_abm.llm_microvalidation.workflows.psychological_profiles",
            "run_psychological_profile_reference_effects",
        ),
        (
            "evaluation/scripts/export_thesis_artifacts.py",
            ["--repo-root", str(REPO_ROOT)],
            "beam_abm.llm_microvalidation.export",
            "export_thesis_artifacts",
        ),
    ],
)
def test_evaluation_cli_dispatches_to_library(
    monkeypatch: pytest.MonkeyPatch,
    script_rel: str,
    argv: list[str],
    module_name: str,
    func_name: str,
) -> None:
    calls: list[dict[str, Any]] = []

    def _stub(*args: Any, **kwargs: Any) -> None:
        calls.append({"args": args, "kwargs": kwargs})

    module = __import__(module_name, fromlist=[func_name])
    monkeypatch.setattr(module, func_name, _stub)

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
