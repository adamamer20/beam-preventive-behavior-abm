"""Choice evaluation workflow entrypoints."""

from __future__ import annotations


def run_choice_unperturbed(argv: list[str] | None = None) -> None:
    from beam_abm.evaluation.choice.run_unperturbed_workflow import run_choice_unperturbed as _run

    _run(argv)


def run_choice_perturbed(argv: list[str] | None = None) -> None:
    from beam_abm.evaluation.choice.run_perturbed_workflow import run_choice_perturbed as _run

    _run(argv)


__all__ = ["run_choice_unperturbed", "run_choice_perturbed"]
