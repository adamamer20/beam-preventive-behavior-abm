"""Belief evaluation workflow entrypoints."""

from __future__ import annotations


def run_belief_perturbed(argv: list[str] | None = None) -> None:
    from beam_abm.evaluation.belief.run_all_workflow import run_belief_perturbed as _run

    _run(argv)


def run_belief_unperturbed(argv: list[str] | None = None) -> None:
    from beam_abm.evaluation.belief.run_unperturbed_workflow import run_belief_unperturbed as _run

    _run(argv)


def run_belief_pe_ref_p(argv: list[str] | None = None) -> None:
    from beam_abm.evaluation.belief.pe_ref_p import run_belief_pe_ref_p as _run

    _run(argv)

__all__ = ["run_belief_unperturbed", "run_belief_perturbed", "run_belief_pe_ref_p"]
