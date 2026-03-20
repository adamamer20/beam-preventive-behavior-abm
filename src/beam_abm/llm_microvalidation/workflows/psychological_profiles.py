"""Belief evaluation workflow entrypoints."""

from __future__ import annotations


def run_psychological_profiles_perturbation_response(argv: list[str] | None = None) -> None:
    from beam_abm.llm_microvalidation.psychological_profiles.perturbation_response_workflow import (
        run_psychological_profiles_perturbation_response as _run,
    )

    _run(argv)


def run_psychological_profiles_baseline_reconstruction(argv: list[str] | None = None) -> None:
    from beam_abm.llm_microvalidation.psychological_profiles.baseline_reconstruction_workflow import (
        run_psychological_profiles_baseline_reconstruction as _run,
    )

    _run(argv)


def run_psychological_profile_reference_effects(argv: list[str] | None = None) -> None:
    from beam_abm.llm_microvalidation.psychological_profiles.reference_effects import (
        run_psychological_profile_reference_effects as _run,
    )

    _run(argv)


__all__ = [
    "run_psychological_profiles_baseline_reconstruction",
    "run_psychological_profiles_perturbation_response",
    "run_psychological_profile_reference_effects",
]
