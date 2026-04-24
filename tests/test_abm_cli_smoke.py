from __future__ import annotations

import runpy
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("argv", "module_name", "func_name", "assertion"),
    [
        (
            [
                "diagnostics",
                "--suite",
                "core",
                "--n-agents",
                "100",
                "--n-steps",
                "6",
                "--n-reps",
                "2",
                "--seeds",
                "1,2",
            ],
            "beam_abm.abm.workflows",
            "run_diagnostics_workflow",
            lambda kwargs: kwargs["suite"] == "core"
            and kwargs["n_agents"] == 100
            and kwargs["n_steps"] == 6
            and kwargs["n_reps"] == 2
            and kwargs["seeds"] == (1, 2),
        ),
        (
            ["run", "--scenario", "baseline_importation q25_to_q75_high", "--n-agents", "100"],
            "beam_abm.abm.workflows",
            "run_scenarios_workflow",
            lambda kwargs: kwargs["scenario"] == ("baseline_importation", "q25_to_q75_high")
            and kwargs["n_agents"] == 100,
        ),
        (
            ["report", "--run-dir", "abm/output/runs"],
            "beam_abm.abm.workflows",
            "run_report_workflow",
            lambda kwargs: str(kwargs["run_dir"]).endswith("abm/output/runs"),
        ),
        (
            ["build-pop", "--n-agents", "250", "--seed", "7"],
            "beam_abm.abm.workflows",
            "run_build_population_workflow",
            lambda kwargs: kwargs["n_agents"] == 250 and kwargs["seed"] == 7,
        ),
        (
            ["benchmark", "--n-agents", "100,200"],
            "beam_abm.abm.workflows",
            "run_benchmark_workflow",
            lambda kwargs: kwargs["n_agents"] == (100, 200),
        ),
    ],
)
def test_abm_cli_dispatches_to_workflows(
    monkeypatch: pytest.MonkeyPatch,
    argv: list[str],
    module_name: str,
    func_name: str,
    assertion,
) -> None:
    calls: list[dict[str, Any]] = []

    def _stub(*args: Any, **kwargs: Any):
        calls.append({"args": args, "kwargs": kwargs})
        if func_name == "run_diagnostics_workflow":

            class _Report:
                overall_passed = True

            return _Report()
        return None

    module = __import__(module_name, fromlist=[func_name])
    monkeypatch.setattr(module, func_name, _stub)

    script_path = REPO_ROOT / "abm" / "scripts" / "abm.py"
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
    assert assertion(calls[0]["kwargs"])
