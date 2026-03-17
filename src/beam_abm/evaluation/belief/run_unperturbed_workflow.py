"""Run belief-update unperturbed workflow end to end."""

from __future__ import annotations

import argparse
from pathlib import Path

from beam_abm.evaluation.artifacts import utc_timestamp
from beam_abm.evaluation.belief.run_all_workflow import _call_main, _load_local_main, _run_sampling


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        default="evaluation/output/belief_update_validation/unperturbed",
        help="Belief-update unperturbed output root.",
    )
    parser.add_argument("--run-id", default=None, help="Optional run id. Defaults to UTC timestamp.")

    parser.add_argument(
        "--data",
        default="empirical/output/anchors/eval/eval_profiles.parquet",
        help="Baseline rows for prompt generation.",
    )
    parser.add_argument(
        "--spec",
        default="evaluation/specs/belief_update_targets.json",
        help="Belief-update spec JSON.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--profile-col", default="anchor_id")
    parser.add_argument("--per-profile", type=int, default=100)
    parser.add_argument("--anchor-col", default="anchor_id")
    parser.add_argument("--keep-anchors", default=None)
    parser.add_argument("--n-per-anchor-country", type=int, default=0)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--prompt-families",
        default="data_only,first_person,third_person,multi_expert",
        help="Comma-separated prompt families to run.",
    )
    parser.add_argument("--country-col", default="country")
    parser.add_argument("--id-col", default="row_id")
    parser.add_argument("--model-plan", default="empirical/output/modeling/model_plan.json")
    parser.add_argument("--alpha-secondary", type=float, default=0.33)
    parser.add_argument("--epsilon-ref", type=float, default=0.03)
    parser.add_argument("--top-n-primary", type=int, default=1)

    parser.add_argument("--models", default="openai/gpt-oss-20b")
    parser.add_argument("--backend", choices=["vllm", "azure_openai"], default="vllm")
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--max-concurrency", type=int, default=10)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--model-options", default=None)
    parser.add_argument("--model-options-file", default=None)

    parser.add_argument("--skip-generate", action="store_true")
    parser.add_argument("--skip-sample", action="store_true")
    parser.add_argument("--skip-metrics", action="store_true")
    parser.add_argument(
        "--skip-existing-samples",
        action="store_true",
        help="Skip sampling for model outputs where _batch/<model>/samples.jsonl already has rows.",
    )

    args = parser.parse_args(argv)

    output_root = Path(args.output_root)
    run_id = args.run_id or utc_timestamp()
    run_root = output_root / "_runs" / run_id
    batch_root = run_root / "_batch"

    prompts_path = run_root / "prompts.jsonl"
    spec_src = Path(args.spec)
    spec_dst = run_root / "belief_update_targets.json"

    run_root.mkdir(parents=True, exist_ok=True)
    batch_root.mkdir(parents=True, exist_ok=True)

    if spec_src.exists() and not spec_dst.exists():
        spec_dst.write_text(spec_src.read_text(encoding="utf-8"), encoding="utf-8")

    gen_path = Path(__file__).resolve().parents[1] / "choice" / "prompt_tournament" / "generate_prompts.py"
    if not args.skip_generate:
        gen_main = _load_local_main(gen_path)
        gen_argv = [
            "--mv-task",
            "belief_update",
            "--run-type",
            "unperturbed",
            "--data",
            str(Path(args.data)),
            "--spec",
            str(spec_src),
            "--out",
            str(prompts_path),
            "--seed",
            str(int(args.seed)),
            "--profile-col",
            str(args.profile_col),
            "--per-profile",
            str(int(args.per_profile)),
            "--anchor-col",
            str(args.anchor_col),
            "--prompt-families",
            str(args.prompt_families),
            "--country-col",
            str(args.country_col),
            "--id-col",
            str(args.id_col),
            "--model-plan",
            str(args.model_plan),
            "--alpha-secondary",
            str(float(args.alpha_secondary)),
            "--epsilon-ref",
            str(float(args.epsilon_ref)),
            "--top-n-primary",
            str(int(args.top_n_primary)),
        ]
        if args.limit is not None:
            gen_argv.extend(["--limit", str(int(args.limit))])
        if args.shuffle:
            gen_argv.append("--shuffle")
        if args.keep_anchors:
            gen_argv.extend(["--keep-anchors", str(args.keep_anchors)])
        if int(args.n_per_anchor_country) > 0:
            gen_argv.extend(["--n-per-anchor-country", str(int(args.n_per_anchor_country))])
        _call_main(gen_main, gen_argv)

    if not args.skip_sample:
        _run_sampling(
            models_csv=str(args.models),
            batch_root=batch_root,
            run_id=run_id,
            prompts_path=prompts_path,
            spec_src=spec_src,
            prompt_families=str(args.prompt_families),
            backend=str(args.backend),
            k=int(args.k),
            max_concurrency=int(args.max_concurrency),
            max_model_len=int(args.max_model_len),
            model_options_file=args.model_options_file,
            model_options=args.model_options,
            run_type="unperturbed",
            skip_existing_samples=bool(args.skip_existing_samples),
        )

    canonical_main = _load_local_main(Path(__file__).resolve().parent / "rebuild_canonical_workflow.py")
    _call_main(canonical_main, ["--output-root", str(output_root)])

    if not args.skip_metrics:
        metrics_main = _load_local_main(Path(__file__).resolve().parent / "rebuild_unperturbed_metrics_workflow.py")
        _call_main(
            metrics_main,
            [
                "--output-root",
                str(output_root),
                "--spec",
                str(spec_src),
            ],
        )


if __name__ == "__main__":
    main()
