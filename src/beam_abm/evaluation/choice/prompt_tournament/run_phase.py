"""Run phase 0 (prompt tournament) for choice validation."""

from __future__ import annotations

import argparse
import sys


def main() -> None:
    parser = argparse.ArgumentParser()
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
    args, rest = parser.parse_known_args()

    if args.step == "generate-prompts":
        from beam_abm.evaluation.choice.prompt_tournament import generate_prompts as step

        sys.argv = [sys.argv[0], *rest]
        step.main()
        return

    if args.step == "build-paired-profiles":
        from beam_abm.evaluation.choice.prompt_tournament import build_paired_profiles as step

        sys.argv = [sys.argv[0], *rest]
        step.main()
        return

    if args.step == "sample":
        from beam_abm.evaluation.common.llm_sampling import main as step_main

        sys.argv = [sys.argv[0], *rest]
        step_main()
        return

    if args.step == "ice":
        from beam_abm.evaluation.common.compute_ice_from_samples import main as step_main

        sys.argv = [sys.argv[0], *rest]
        step_main()
        return

    if args.step == "align":
        from beam_abm.evaluation.common.compute_alignment import main as step_main

        sys.argv = [sys.argv[0], *rest]
        step_main()
        return

    if args.step == "calibrate":
        from beam_abm.evaluation.common.calibrate_llm_predictions import main as step_main

        sys.argv = [sys.argv[0], *rest]
        step_main()
        return

    if args.step == "apply-calibration":
        from beam_abm.evaluation.common.apply_calibration_to_ice import main as step_main

        sys.argv = [sys.argv[0], *rest]
        step_main()
        return

    if args.step == "unperturbed":
        from pathlib import Path

        support_dir = Path(__file__).resolve().parents[1] / "support" / "unperturbed"
        sys.path.insert(0, str(support_dir))
        import run_phase as step

        sys.argv = [sys.argv[0], *rest]
        step.main()
        return


if __name__ == "__main__":
    main()
