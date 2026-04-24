"""Choice evaluation workflow entrypoints."""

from __future__ import annotations


def run_behavioural_outcomes_baseline_reconstruction(argv: list[str] | None = None) -> None:
    from beam_abm.llm_microvalidation.behavioural_outcomes.baseline_reconstruction_workflow import (
        run_behavioural_outcomes_baseline_reconstruction as _run,
    )

    _run(argv)


def run_behavioural_outcomes_perturbation_response(argv: list[str] | None = None) -> None:
    from beam_abm.llm_microvalidation.behavioural_outcomes.perturbation_response_workflow import (
        run_behavioural_outcomes_perturbation_response as _run,
    )

    _run(argv)


__all__ = ["run_behavioural_outcomes_baseline_reconstruction", "run_behavioural_outcomes_perturbation_response"]
