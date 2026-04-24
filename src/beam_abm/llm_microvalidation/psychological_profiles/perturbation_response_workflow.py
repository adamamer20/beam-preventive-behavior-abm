"""Run belief-update validation end-to-end with choice-validation-style tooling.

Pipeline:
1) Generate prompts for all requested prompt families.
2) Sample LLM outputs for each requested model.
3) Rebuild canonical samples from `_runs/`.
4) Recompute alignment metrics + write `summary_metrics/` artifacts.

Output contract:
evaluation/output/belief_update_validation/
  _runs/<run_id>/prompts.jsonl
  _runs/<run_id>/belief_update_targets.json
  _runs/<run_id>/_batch/<model>/samples.jsonl
  canonical/<model>/samples.jsonl
  canonical/<model>/... alignment outputs ...
  summary_metrics/<model>/summary_metrics.csv
  summary_metrics/all_models_strategies.csv
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from beam_abm.cli import parser_compat as cli
from beam_abm.llm_microvalidation.artifacts import utc_timestamp
from beam_abm.llm_microvalidation.behavioural_outcomes._canonicalize_ids import normalize_model_slug
from beam_abm.llm_microvalidation.behavioural_outcomes.paired_contrast.generate_prompts import (
    run_cli as generate_prompts,
)
from beam_abm.llm_microvalidation.psychological_profiles.rebuild_baseline_reconstruction_metrics_workflow import (
    run_cli as rebuild_unperturbed_metrics_workflow,
)
from beam_abm.llm_microvalidation.psychological_profiles.rebuild_canonical_workflow import (
    run_cli as rebuild_canonical_workflow,
)
from beam_abm.llm_microvalidation.psychological_profiles.rebuild_metrics_workflow import (
    run_cli as rebuild_metrics_workflow,
)
from beam_abm.llm_microvalidation.shared.sampling import run_cli as run_llm_sampling
from beam_abm.llm_microvalidation.utils.jsonl import read_jsonl as _read_jsonl
from beam_abm.llm_microvalidation.utils.jsonl import write_jsonl as _write_jsonl


def _parse_csv_list(value: str | None) -> list[str]:
    if not value:
        return []
    return [v.strip() for v in value.split(",") if v.strip()]


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _jsonl_has_rows(path: Path) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                return True
    return False


def _prompt_row_key(row: dict[str, Any]) -> tuple[str, str, str]:
    record_id = str(row.get("id") or "")
    strategy = str(row.get("strategy") or "data_only")
    human_prompt = str(row.get("human_prompt") or "")
    return record_id, strategy, human_prompt


def _sample_count(row: dict[str, Any]) -> int:
    samples = row.get("samples")
    if not isinstance(samples, list):
        return 0
    return len(samples)


def _run_sampling(
    *,
    models_csv: str,
    batch_root: Path,
    run_id: str,
    prompts_path: Path,
    spec_src: Path,
    prompt_families: str,
    backend: str,
    k: int,
    max_concurrency: int,
    max_model_len: int,
    model_options_file: str | None,
    model_options: str | None,
    run_type: str,
    skip_existing_samples: bool,
) -> None:
    prompts_rows = _read_jsonl(prompts_path)

    for model in _parse_csv_list(models_csv):
        model_norm = normalize_model_slug(model, reasoning_effort=None)
        model_out = batch_root / model_norm
        model_out.mkdir(parents=True, exist_ok=True)

        samples_path = model_out / "samples.jsonl"
        _write_json(
            model_out / "run_metadata.json",
            {
                "run_id": run_id,
                "run_type": run_type,
                "model": model,
                "model_norm": model_norm,
                "backend": backend,
                "k": int(k),
                "max_concurrency": int(max_concurrency),
                "max_model_len": int(max_model_len),
                "prompts": str(prompts_path),
                "spec": str(spec_src),
                "prompt_families": str(prompt_families),
            },
        )

        def _run_sampler(
            *,
            in_path: Path,
            out_path: Path,
            k_for_run: int,
            model=model,
            backend=backend,
            max_concurrency=max_concurrency,
        ) -> None:
            sampler_args: list[str] = [
                "--in",
                str(in_path),
                "--out",
                str(out_path),
                "--backend",
                str(backend),
                "--model",
                str(model),
                "--k",
                str(int(k_for_run)),
                "--two-pass",
                "--max-concurrency",
                str(int(max_concurrency)),
                "--response-model",
                "beam_abm.llm.schemas.predictions:BeliefUpdatePrediction",
            ]

            sampler_args.extend(
                [
                    "--max-model-len-by-strategy-file",
                    "evaluation/specs/prompt_tournament_max_model_len.json",
                ]
            )

            if model_options_file:
                sampler_args.extend(["--model-options-file", str(model_options_file)])
            elif model_options:
                sampler_args.extend(["--model-options", str(model_options)])
            else:
                sampler_args.extend(["--model-options", json.dumps({"max-model-len": int(max_model_len)})])

            run_llm_sampling(sampler_args)

        if not bool(skip_existing_samples) or not _jsonl_has_rows(samples_path):
            _run_sampler(in_path=prompts_path, out_path=samples_path, k_for_run=int(k))
            continue

        existing_rows = _read_jsonl(samples_path)
        existing_by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
        for row in existing_rows:
            existing_by_key[_prompt_row_key(row)] = dict(row)

        missing_by_count: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for prompt_row in prompts_rows:
            key = _prompt_row_key(prompt_row)
            existing_row = existing_by_key.get(key)
            if existing_row is None:
                existing_row = dict(prompt_row)
                existing_row["samples"] = []
                existing_by_key[key] = existing_row
            missing_n = max(0, int(k) - _sample_count(existing_row))
            if missing_n > 0:
                missing_by_count[missing_n].append(prompt_row)

        if not missing_by_count:
            print(f"[sampling] skip complete model={model_norm}: {samples_path}")
            continue

        total_missing_rows = sum(len(v) for v in missing_by_count.values())
        print(
            f"[sampling] resume model={model_norm}: {total_missing_rows} rows still below k={int(k)}; sampling only missing rows"
        )

        for missing_n in sorted(missing_by_count.keys()):
            prompts_missing = missing_by_count[missing_n]
            if not prompts_missing:
                continue
            tmp_in = model_out / f"_resume_missing_k{missing_n}.jsonl"
            tmp_out = model_out / f"_resume_samples_k{missing_n}.jsonl"
            _write_jsonl(tmp_in, prompts_missing)
            _run_sampler(in_path=tmp_in, out_path=tmp_out, k_for_run=int(missing_n))
            new_rows = _read_jsonl(tmp_out)
            for new_row in new_rows:
                key = _prompt_row_key(new_row)
                base = existing_by_key.get(key)
                if base is None:
                    base = dict(new_row)
                    base["samples"] = []
                base_samples = base.get("samples") if isinstance(base.get("samples"), list) else []
                new_samples = new_row.get("samples") if isinstance(new_row.get("samples"), list) else []
                base["samples"] = [*base_samples, *new_samples][: int(k)]
                existing_by_key[key] = base

        merged_rows: list[dict[str, Any]] = []
        for prompt_row in prompts_rows:
            key = _prompt_row_key(prompt_row)
            row = dict(existing_by_key.get(key) or prompt_row)
            samples = row.get("samples") if isinstance(row.get("samples"), list) else []
            row["samples"] = samples[: int(k)]
            merged_rows.append(row)
        _write_jsonl(samples_path, merged_rows)


def run_psychological_profiles_perturbation_response(argv: list[str] | None = None) -> None:
    parser = cli.ArgumentParser()
    parser.add_argument(
        "--output-root",
        default="evaluation/output/belief_update_validation",
        help="Belief-update output root.",
    )
    parser.add_argument("--run-id", default=None, help="Optional run id. Defaults to UTC timestamp.")
    parser.add_argument(
        "--with-unperturbed",
        action="store_true",
        help="Also run belief unperturbed pipeline under <output-root>/unperturbed.",
    )
    parser.add_argument(
        "--unperturbed-output-root",
        default=None,
        help="Optional output root for belief unperturbed pipeline. Defaults to <output-root>/unperturbed.",
    )
    parser.add_argument(
        "--unperturbed-prompt-families",
        default=None,
        help="Prompt families for unperturbed mode (defaults to --prompt-families).",
    )

    # Prompt generation.
    parser.add_argument(
        "--data",
        default="empirical/output/anchors/eval/eval_profiles.parquet",
        help="Baseline rows for prompt generation.",
    )
    parser.add_argument(
        "--pe-ref-p-table",
        default=None,
        help="Precomputed PE_ref_P table (parquet/csv/jsonl) keyed by (id, signal_id).",
    )
    parser.add_argument(
        "--spec",
        default="evaluation/specs/belief_update_targets.json",
        help="Belief-update spec JSON.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--profile-col",
        default="anchor_id",
        help="Column used for stratified prompt-row sampling.",
    )
    parser.add_argument(
        "--per-profile",
        type=int,
        default=100,
        help="Rows sampled per profile before prompt expansion.",
    )
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
    parser.add_argument("--epsilon-move-mult", type=float, default=3.0)
    parser.add_argument("--top-n-primary", type=int, default=1)
    parser.add_argument("--clean-spec", default="preprocess/specs/5_clean_transformed_questions.json")
    parser.add_argument("--eps", type=float, default=0.05)
    parser.add_argument("--eps-small", type=float, default=0.02)
    parser.add_argument("--eps-map", default=None)
    parser.add_argument("--moderation-quantile", type=float, default=0.25)
    parser.add_argument("--moderation-by-anchor", action="store_true")
    parser.add_argument("--denom", type=float, default=1.0)

    # Sampling.
    parser.add_argument("--models", default="openai/gpt-oss-20b")
    parser.add_argument("--backend", choices=["vllm", "azure_openai"], default="vllm")
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--max-concurrency", type=int, default=10)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument(
        "--model-options",
        default=None,
        help="JSON string model options (forwarded to sampler).",
    )
    parser.add_argument(
        "--model-options-file",
        default=None,
        help="Path to JSON model options (forwarded to sampler).",
    )

    # Control.
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

    if not args.pe_ref_p_table:
        raise ValueError("--pe-ref-p-table is required. Provide the precomputed PE_ref_P table from anchors.")
    pe_ref_p_table = Path(args.pe_ref_p_table)
    if not pe_ref_p_table.exists():
        raise FileNotFoundError(f"PE_ref_P table not found: {pe_ref_p_table}")

    if spec_src.exists() and not spec_dst.exists():
        spec_dst.write_text(spec_src.read_text(encoding="utf-8"), encoding="utf-8")

    # 1) Generate prompts.
    if not args.skip_generate:
        gen_argv = [
            "--mv-task",
            "belief_update",
            "--data",
            str(Path(args.data)),
            "--pe-ref-p",
            str(pe_ref_p_table),
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
        generate_prompts(gen_argv)

    # 2) Sample per model.
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
            run_type="perturbed",
            skip_existing_samples=bool(args.skip_existing_samples),
        )

    # 3) Rebuild canonical.
    rebuild_canonical_workflow(["--output-root", str(output_root)])

    # 4) Rebuild metrics.
    if not args.skip_metrics:
        metrics_argv = [
            "--output-root",
            str(output_root),
            "--spec",
            str(spec_src),
            "--clean-spec",
            str(Path(args.clean_spec)),
            "--eps",
            str(float(args.eps)),
            "--eps-small",
            str(float(args.eps_small)),
            "--moderation-quantile",
            str(float(args.moderation_quantile)),
            "--epsilon-ref",
            str(float(args.epsilon_ref)),
            "--epsilon-move-mult",
            str(float(args.epsilon_move_mult)),
            "--alpha-secondary",
            str(float(args.alpha_secondary)),
            "--top-n-primary",
            str(int(args.top_n_primary)),
            "--denom",
            str(float(args.denom)),
        ]
        if args.eps_map:
            metrics_argv.extend(["--eps-map", str(args.eps_map)])
        if args.moderation_by_anchor:
            metrics_argv.append("--moderation-by-anchor")
        rebuild_metrics_workflow(metrics_argv)

    # 5) Optional unperturbed belief pipeline.
    if args.with_unperturbed:
        unpert_output_root = (
            Path(args.unperturbed_output_root) if args.unperturbed_output_root else output_root / "unperturbed"
        )
        unpert_run_root = unpert_output_root / "_runs" / run_id
        unpert_batch_root = unpert_run_root / "_batch"
        unpert_prompts_path = unpert_run_root / "prompts.jsonl"
        unpert_spec_dst = unpert_run_root / "belief_update_targets.json"
        unpert_prompt_families = str(args.unperturbed_prompt_families or args.prompt_families)

        unpert_run_root.mkdir(parents=True, exist_ok=True)
        unpert_batch_root.mkdir(parents=True, exist_ok=True)
        if spec_src.exists() and not unpert_spec_dst.exists():
            unpert_spec_dst.write_text(spec_src.read_text(encoding="utf-8"), encoding="utf-8")

        if not args.skip_generate:
            unpert_gen_argv = [
                "--mv-task",
                "belief_update",
                "--run-type",
                "unperturbed",
                "--data",
                str(Path(args.data)),
                "--spec",
                str(spec_src),
                "--out",
                str(unpert_prompts_path),
                "--seed",
                str(int(args.seed)),
                "--profile-col",
                str(args.profile_col),
                "--per-profile",
                str(int(args.per_profile)),
                "--anchor-col",
                str(args.anchor_col),
                "--prompt-families",
                str(unpert_prompt_families),
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
                unpert_gen_argv.extend(["--limit", str(int(args.limit))])
            if args.shuffle:
                unpert_gen_argv.append("--shuffle")
            if args.keep_anchors:
                unpert_gen_argv.extend(["--keep-anchors", str(args.keep_anchors)])
            if int(args.n_per_anchor_country) > 0:
                unpert_gen_argv.extend(["--n-per-anchor-country", str(int(args.n_per_anchor_country))])
            generate_prompts(unpert_gen_argv)

        if not args.skip_sample:
            _run_sampling(
                models_csv=str(args.models),
                batch_root=unpert_batch_root,
                run_id=run_id,
                prompts_path=unpert_prompts_path,
                spec_src=spec_src,
                prompt_families=unpert_prompt_families,
                backend=str(args.backend),
                k=int(args.k),
                max_concurrency=int(args.max_concurrency),
                max_model_len=int(args.max_model_len),
                model_options_file=args.model_options_file,
                model_options=args.model_options,
                run_type="unperturbed",
                skip_existing_samples=bool(args.skip_existing_samples),
            )

        rebuild_canonical_workflow(["--output-root", str(unpert_output_root)])

        if not args.skip_metrics:
            rebuild_unperturbed_metrics_workflow(
                [
                    "--output-root",
                    str(unpert_output_root),
                    "--spec",
                    str(spec_src),
                ],
            )
