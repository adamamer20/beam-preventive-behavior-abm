"""Run choice-validation perturbed pipeline with canonical outputs.

This wrapper:
1) Runs the prompt-tournament runner into a staging root under
    evaluation/output/choice_validation/perturbed/
2) Ingests (copy-only) staged prompt-tournament outputs into
    evaluation/output/choice_validation/perturbed/_runs/
3) Rebuilds canonical leaves + summaries.

The goal is to standardize perturbed outputs to match the unperturbed philosophy.
"""

from __future__ import annotations

import os
import shlex
from pathlib import Path

from beam_abm.cli import parser_compat as cli
from beam_abm.llm_microvalidation.behavioural_outcomes.paired_contrast.run_all import run_cli as run_prompt_tournament
from beam_abm.llm_microvalidation.behavioural_outcomes.rebuild_perturbation_response_metrics import (
    run_cli as rebuild_perturbed_metrics,
)


def _ensure_flag_in_args_str(args_str: str | None, flag: str, value: str) -> str:
    """Ensure a CLI flag/value pair exists inside an args string.

    This is used to translate top-level convenience flags (e.g. --keep-targets)
    into the phase0 runner's *-args strings.
    """

    tokens = shlex.split(args_str or "")
    if flag in tokens:
        return args_str or ""
    tokens.extend([flag, value])
    return shlex.join(tokens)


def run_behavioural_outcomes_perturbation_response(argv: list[str] | None = None) -> None:
    parser = cli.ArgumentParser()
    parser.add_argument(
        "--output-root",
        default="evaluation/output/choice_validation/perturbed",
        help="Canonical perturbed output root.",
    )

    # Forwarded configuration knobs.
    parser.add_argument("--models", default=None)
    parser.add_argument(
        "--prompt-families",
        default="data_only,first_person,third_person,multi_expert,paired_profile",
    )
    parser.add_argument(
        "--keep-outcomes",
        default=None,
        help=(
            "Comma-separated outcomes to keep. Defaults to all outcomes with ref ICE tables under "
            "empirical/output/anchors/pe/mutable_engines/reduced/."
        ),
    )
    parser.add_argument(
        "--keep-targets",
        default=None,
        help=(
            "Comma-separated perturbation targets/levers to keep (e.g., institutional_trust_avg). "
            "This is forwarded to phase0 prompt generation as --keep-targets."
        ),
    )
    parser.add_argument("--ref-models", default="linear")
    parser.add_argument("--run-tag", default="exp-choice-validation-perturbed")
    parser.add_argument("--timestamp-output", action="store_true")
    parser.add_argument("--resume-strategies", action="store_true")

    # Phase0 passthrough args (top-level; forwarded directly).
    parser.add_argument("--build-paired-profiles-args", default=None)
    parser.add_argument("--generate-prompts-args", default=None)
    parser.add_argument("--sample-args", default=None)
    parser.add_argument("--ice-args", default=None)
    parser.add_argument("--align-args", default=None)
    parser.add_argument("--unperturbed-args", default=None)
    parser.add_argument("--calibrate-args", default=None)
    parser.add_argument("--apply-calibration-args", default=None)
    parser.add_argument("--align-calibrated-args", default=None)
    parser.add_argument("--apply-unperturbed-calibration-args", default=None)
    parser.add_argument("--calibrated-metrics-args", default=None)
    parser.add_argument("--pe-ice-metrics", action="store_true")
    parser.add_argument("--pe-ice-metrics-args", default=None)
    parser.add_argument("--migrate-existing", action="store_true")
    parser.add_argument("--skip-posthoc", action="store_true")
    parser.add_argument("--skip-pe-calibration", action="store_true")
    parser.add_argument("--posthoc-args", default=None)

    # Convenience sampling flags (optional). If provided and --sample-args is not,
    # we will build a phase0-compatible --sample-args string for you.
    parser.add_argument("--backend", choices=["vllm", "azure_openai"], default=None)
    parser.add_argument("--max-concurrency", type=int, default=None)
    parser.add_argument("--k", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--model-options", default=None, help="JSON model options payload.")
    parser.add_argument("--model-options-file", default=None, help="Path to JSON model options.")
    parser.add_argument(
        "--azure-deployment",
        default=None,
        help="Azure OpenAI deployment name (supports {model_slug} template).",
    )
    parser.add_argument("--azure-endpoint", default=None)
    parser.add_argument("--azure-api-version", default=None)
    parser.add_argument("--azure-model-hint", default=None)

    args = parser.parse_args(argv)

    # NOTE: We intentionally do not auto-add data_only when multi_expert is requested.
    # multi_expert can reuse an existing data_only baseline (prompt-ID compatible) from
    # prior runs, and forcing a new data_only sampling run can be expensive.

    output_root = Path(args.output_root)
    staging_root = output_root / "_staging" / "prompt_tournament"
    staging_root.mkdir(parents=True, exist_ok=True)

    keep_outcomes = str(args.keep_outcomes or "").strip()
    if not keep_outcomes:
        anchors_root = Path("empirical/output/anchors/pe/mutable_engines/reduced")
        if anchors_root.exists():
            outcomes = sorted(
                p.name
                for p in anchors_root.iterdir()
                if p.is_dir() and not p.name.startswith("_") and (p / "design.parquet").exists()
            )
            keep_outcomes = ",".join(outcomes)
        else:
            keep_outcomes = "vax_willingness_Y"

    # If the user asks for multi_expert without data_only, we should *reuse* an existing
    # data_only baseline from canonical outputs. That only works when prompt IDs match.
    # With multiple outcomes in a single phase0 invocation, prompt IDs span outcomes and
    # there is typically no single prior data_only samples.jsonl containing all IDs.
    # Canonical perturbed outputs are per-outcome, so split the run per outcome.
    prompt_families = [f.strip() for f in str(args.prompt_families or "").split(",") if f.strip()]
    wants_multi_expert_only = ("multi_expert" in prompt_families) and ("data_only" not in prompt_families)
    keep_outcome_list = [o.strip() for o in keep_outcomes.split(",") if o.strip()]
    split_per_outcome = wants_multi_expert_only and len(keep_outcome_list) > 1
    outcome_runs: list[str] = keep_outcome_list if split_per_outcome else ["__all__"]

    keep_targets = str(args.keep_targets or "").strip()
    if keep_targets:
        args.generate_prompts_args = _ensure_flag_in_args_str(
            args.generate_prompts_args,
            "--keep-targets",
            keep_targets,
        )

    # If user provided convenience sampling flags but no explicit --sample-args,
    # assemble a phase0-compatible args string.
    if args.sample_args is None:
        sample_tokens: list[str] = []
        if args.backend:
            sample_tokens.extend(["--backend", str(args.backend)])
        if args.k is not None:
            sample_tokens.extend(["--k", str(int(args.k))])
        if args.max_concurrency is not None:
            sample_tokens.extend(["--max-concurrency", str(int(args.max_concurrency))])
        if args.temperature is not None:
            sample_tokens.extend(["--temperature", str(float(args.temperature))])
        if args.top_p is not None:
            sample_tokens.extend(["--top-p", str(float(args.top_p))])
        if args.model_options is not None:
            sample_tokens.extend(["--model-options", str(args.model_options)])
        if args.model_options_file is not None:
            sample_tokens.extend(["--model-options-file", str(args.model_options_file)])
        if args.azure_deployment is not None:
            sample_tokens.extend(["--azure-deployment", str(args.azure_deployment)])
        if args.azure_endpoint is not None:
            sample_tokens.extend(["--azure-endpoint", str(args.azure_endpoint)])
        if args.azure_api_version is not None:
            sample_tokens.extend(["--azure-api-version", str(args.azure_api_version)])
        if args.azure_model_hint is not None:
            sample_tokens.extend(["--azure-model-hint", str(args.azure_model_hint)])
        if sample_tokens:
            args.sample_args = shlex.join(sample_tokens)

    saved_env = dict(os.environ)
    try:
        os.environ["PROMPT_TOURNAMENT_OUTPUT_ROOT"] = str(staging_root)

        passthrough: list[tuple[str, object | None]] = [
            ("--build-paired-profiles-args", args.build_paired_profiles_args),
            ("--generate-prompts-args", args.generate_prompts_args),
            ("--sample-args", args.sample_args),
            ("--ice-args", args.ice_args),
            ("--align-args", args.align_args),
            ("--unperturbed-args", args.unperturbed_args),
            ("--calibrate-args", args.calibrate_args),
            ("--apply-calibration-args", args.apply_calibration_args),
            ("--align-calibrated-args", args.align_calibrated_args),
            ("--apply-unperturbed-calibration-args", args.apply_unperturbed_calibration_args),
            ("--calibrated-metrics-args", args.calibrated_metrics_args),
            ("--pe-ice-metrics-args", args.pe_ice_metrics_args),
            ("--posthoc-args", args.posthoc_args),
        ]

        for outcome in outcome_runs:
            prompt_tournament_argv: list[str] = []
            if args.models:
                prompt_tournament_argv.extend(["--models", str(args.models)])
            if args.prompt_families:
                prompt_tournament_argv.extend(["--prompt-families", str(args.prompt_families)])
            if split_per_outcome:
                prompt_tournament_argv.extend(["--keep-outcomes", str(outcome)])
            else:
                prompt_tournament_argv.extend(["--keep-outcomes", keep_outcomes])
            if args.ref_models:
                prompt_tournament_argv.extend(["--ref-models", str(args.ref_models)])

            # Ensure run directories don't collide when splitting across outcomes.
            run_tag = str(args.run_tag or "exp-choice-validation-perturbed").strip()
            if split_per_outcome:
                run_tag = f"{run_tag}--{outcome}"
            prompt_tournament_argv.extend(["--run-tag", run_tag])

            if args.timestamp_output:
                prompt_tournament_argv.append("--timestamp-output")
            if args.resume_strategies:
                prompt_tournament_argv.append("--resume-strategies")

            for flag, value in passthrough:
                if value is not None and str(value).strip():
                    prompt_tournament_argv.extend([flag, str(value)])
            if args.pe_ice_metrics:
                prompt_tournament_argv.append("--pe-ice-metrics")
            if args.migrate_existing:
                prompt_tournament_argv.append("--migrate-existing")
            if args.skip_posthoc:
                prompt_tournament_argv.append("--skip-posthoc")
            if args.skip_pe_calibration:
                prompt_tournament_argv.append("--skip-pe-calibration")

            run_prompt_tournament(prompt_tournament_argv)
    finally:
        os.environ.clear()
        os.environ.update(saved_env)

    # 2) Ingest into canonical `_runs/` and 3) rebuild canonical leaves.
    rebuild_perturbed_metrics(
        [
            "--output-root",
            str(output_root),
            "--ingest-prompt-tournament-root",
            str(staging_root),
            "--rename-prompt-tournament-model-dirs",
            "--migration-log",
            str(output_root / "_migration" / "ingest_prompt_tournament.jsonl"),
            "--anchors-root",
            "empirical/output/anchors/pe/mutable_engines/reduced",
            "--ref-models",
            str(args.ref_models),
            "--rerun-calibration",
        ]
    )
