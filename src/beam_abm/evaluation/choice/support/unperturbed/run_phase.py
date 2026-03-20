"""Run unperturbed microvalidation steps (prompt generation, sampling, metrics)."""

from __future__ import annotations

from pathlib import Path

from beam_abm.cli import parser_compat as cli
from beam_abm.evaluation.choice.support.unperturbed.unperturbed_utils import parse_prompt_families


def _build_args(*pairs: tuple[str, object]) -> list[str]:
    args: list[str] = []
    for flag, value in pairs:
        if value is None:
            continue
        if isinstance(value, bool):
            if value:
                args.append(flag)
            continue
        args.extend([flag, str(value)])
    return args


def run_cli(argv: list[str] | None = None) -> None:
    effective_argv = list(argv) if argv is not None else None
    if effective_argv and len(effective_argv) > 0:
        first_arg = effective_argv[0]
        valid_steps = {"generate-prompts", "sample", "metrics", "calibrate", "apply-calibration", "all"}
        if not first_arg.startswith("-") and first_arg not in valid_steps:
            effective_argv = ["--outdir", *effective_argv]

    base = cli.ArgumentParser()
    base.add_argument(
        "step",
        nargs="?",
        default="all",
        choices=["generate-prompts", "sample", "metrics", "calibrate", "apply-calibration", "all"],
        help="Unperturbed step to run.",
    )
    args, rest = base.parse_known_args(effective_argv)

    if args.step == "calibrate":
        from beam_abm.evaluation.common.calibrate_llm_predictions import run_cli as step_run_cli

        step_run_cli(rest)
        return

    if args.step == "apply-calibration":
        from beam_abm.evaluation.common.apply_calibration_to_predictions import run_cli as step_run_cli

        step_run_cli(rest)
        return

    parser = cli.ArgumentParser()
    parser.add_argument("--input", default="preprocess/output/clean_processed_survey.csv")
    parser.add_argument("--outdir", default="evaluation/output/debug_unperturbed")
    parser.add_argument("--outcomes", default=None)
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--prompt-family", default=None)
    parser.add_argument("--prompt-families", default=None)
    parser.add_argument("--strata-col", default="country")

    parser.add_argument("--lever-config", default="config/levers.json")
    parser.add_argument("--model-plan", default="empirical/output/modeling/model_plan.json")
    parser.add_argument("--display-labels", default="thesis/specs/display_labels.json")
    parser.add_argument("--clean-spec", default="preprocess/specs/5_clean_transformed_questions.json")
    parser.add_argument("--column-types", default="empirical/specs/column_types.tsv")

    parser.add_argument("--prompts", default=None)
    parser.add_argument("--samples", default=None)
    parser.add_argument("--predictions", default=None)
    parser.add_argument("--metrics-only", action="store_true")
    parser.add_argument("--pred-col", default="y_pred")

    parser.add_argument("--backend", choices=["vllm", "azure_openai"], default="azure_openai")
    parser.add_argument("--models", default=None)
    parser.add_argument("--model-options", default=None)
    parser.add_argument("--model-options-file", default=None)
    parser.add_argument("--azure-deployments", default=None)
    parser.add_argument("--azure-endpoint", default=None)
    parser.add_argument("--azure-api-version", default=None)

    parser.add_argument("--k", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument(
        "--max-model-len",
        dest="max_model_len",
        type=int,
        default=4096,
        help="Maximum model context length forwarded to the sampler.",
    )
    # Backwards-compatible alias used in earlier iterations of this runner.
    parser.add_argument(
        "--max-new-tokens",
        dest="max_model_len",
        type=int,
        default=None,
        help=cli.SUPPRESS,
    )
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--max-concurrency", type=int, default=1024)
    parser.add_argument("--no-structured", action="store_true")
    parser.add_argument("--method", default="debug_unperturbed")
    parser.add_argument("--run-id", default=None, help="Optional run id for row-level outputs")

    parsed = parser.parse_args(rest)
    families = parse_prompt_families(parsed.prompt_family, parsed.prompt_families)

    from beam_abm.evaluation.choice.support.unperturbed.compute_metrics import run_cli as compute_metrics
    from beam_abm.evaluation.choice.support.unperturbed.generate_prompts import run_cli as generate_prompts
    from beam_abm.evaluation.choice.support.unperturbed.run_sampling import run_cli as run_sampling

    if args.step in {"generate-prompts", "all"}:
        generate_args = _build_args(
            ("--input", parsed.input),
            ("--outdir", parsed.outdir),
            ("--outcomes", parsed.outcomes),
            ("--n", parsed.n),
            ("--seed", parsed.seed),
            ("--prompt-family", parsed.prompt_family),
            ("--prompt-families", parsed.prompt_families),
            ("--strata-col", parsed.strata_col),
            ("--lever-config", parsed.lever_config),
            ("--model-plan", parsed.model_plan),
            ("--display-labels", parsed.display_labels),
            ("--clean-spec", parsed.clean_spec),
            ("--column-types", parsed.column_types),
        )
        generate_prompts(generate_args)
        if args.step == "generate-prompts":
            return

    if args.step in {"sample", "all"}:
        sample_args = _build_args(
            ("--prompts", parsed.prompts),
            ("--outdir", parsed.outdir),
            ("--prompt-family", parsed.prompt_family),
            ("--prompt-families", parsed.prompt_families),
            ("--backend", parsed.backend),
            ("--models", parsed.models),
            ("--model-options", parsed.model_options),
            ("--model-options-file", parsed.model_options_file),
            ("--azure-deployments", parsed.azure_deployments),
            ("--azure-endpoint", parsed.azure_endpoint),
            ("--azure-api-version", parsed.azure_api_version),
            ("--k", parsed.k),
            ("--temperature", parsed.temperature),
            ("--top-p", parsed.top_p),
            ("--max-model-len", parsed.max_model_len),
            ("--max-retries", parsed.max_retries),
            ("--max-concurrency", parsed.max_concurrency),
            ("--no-structured", parsed.no_structured),
            ("--method", parsed.method),
            ("--run-id", parsed.run_id),
        )
        run_sampling(sample_args)
        if args.step == "sample":
            return

    if args.step in {"metrics", "all"}:
        metrics_args = _build_args(
            ("--samples", parsed.samples),
            ("--predictions", parsed.predictions),
            ("--outdir", parsed.outdir),
            ("--prompt-family", parsed.prompt_family),
            ("--prompt-families", parsed.prompt_families),
            ("--pred-col", parsed.pred_col),
        )
        if parsed.metrics_only:
            metrics_args.append("--metrics-only")
        compute_metrics(metrics_args)
        if args.step == "metrics" or parsed.metrics_only:
            return

        if args.step == "all":
            from beam_abm.evaluation.common.calibrate_llm_predictions import run_cli as run_calibrate

            outdir_path = Path(parsed.outdir)
            run_tag = outdir_path.name
            for family in families:
                pred_path = outdir_path / f"predictions__{family}__all.csv"
                if not pred_path.exists():
                    continue
                calibrated_path = outdir_path / f"predictions__{family}__all_calibrated.csv"
                if run_tag:
                    calibration_name = f"calibration__{family}__{run_tag}.json"
                else:
                    calibration_name = f"calibration__{family}.json"
                calibration_path = outdir_path / calibration_name

                calibrate_args = [
                    "--in",
                    str(pred_path),
                    "--out",
                    str(calibrated_path),
                    "--calibration-out",
                    str(calibration_path),
                    "--country-col",
                    parsed.strata_col,
                ]

                run_calibrate(calibrate_args)
        return
