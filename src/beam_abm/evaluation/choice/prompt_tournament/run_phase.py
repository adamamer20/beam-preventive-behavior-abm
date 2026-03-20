"""Run phase 0 (prompt tournament) for choice validation."""

from __future__ import annotations

from beam_abm.cli import parser_compat as cli


def run_cli(argv: list[str] | None = None) -> None:
    parser = cli.ArgumentParser()
    parser.add_argument(
        "step",
        choices=[
            "build-paired-profiles",
            "generate-prompts",
            "sample",
            "ice",
            "align",
            "calibrate",
            "apply-calibration",
            "unperturbed",
        ],
        help="Phase step to run.",
    )
    args, rest = parser.parse_known_args(argv)

    if args.step == "generate-prompts":
        from beam_abm.evaluation.choice.prompt_tournament.generate_prompts import run_cli as step_run_cli

        step_run_cli(rest)
        return

    if args.step == "build-paired-profiles":
        from beam_abm.evaluation.choice.prompt_tournament.build_paired_profiles import run_cli as step_run_cli

        step_run_cli(rest)
        return

    if args.step == "sample":
        from beam_abm.evaluation.common.llm_sampling import run_cli as step_run_cli

        step_run_cli(rest)
        return

    if args.step == "ice":
        from beam_abm.evaluation.common.compute_ice_from_samples import run_cli as step_run_cli

        step_run_cli(rest)
        return

    if args.step == "align":
        from beam_abm.evaluation.common.compute_alignment import run_cli as step_run_cli

        step_run_cli(rest)
        return

    if args.step == "calibrate":
        from beam_abm.evaluation.common.calibrate_llm_predictions import run_cli as step_run_cli

        step_run_cli(rest)
        return

    if args.step == "apply-calibration":
        from beam_abm.evaluation.common.apply_calibration_to_ice import run_cli as step_run_cli

        step_run_cli(rest)
        return

    if args.step == "unperturbed":
        from beam_abm.evaluation.choice.support.unperturbed.run_phase import run_cli as step_run_cli

        step_run_cli(rest)
        return
