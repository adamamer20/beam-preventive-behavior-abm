"""Run LLM sampling for unperturbed prompt JSONL."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from beam_abm.cli import parser_compat as cli
from beam_abm.llm.schemas.predictions import extract_scalar_prediction
from beam_abm.llm_microvalidation.behavioural_outcomes.support.baseline_reconstruction.unperturbed_utils import (
    SUPPORTED_PROMPT_FAMILIES,
    parse_list_arg,
    parse_prompt_families,
)
from beam_abm.llm_microvalidation.utils.jsonl import numpy_json_default, read_jsonl, write_jsonl
from beam_abm.llm_microvalidation.utils.regex_fallback import regex_extract_scalar


def _safe_token(value: object) -> str:
    text = str(value) if value is not None else ""
    out = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in text).strip("_")
    return out or "unknown"


def _outcome(row: dict) -> str | None:
    outcome = row.get("outcome")
    if isinstance(outcome, str) and outcome:
        return outcome
    meta = _metadata(row)
    outcome = meta.get("outcome")
    if isinstance(outcome, str) and outcome:
        return outcome
    return None


def _metadata(row: dict) -> dict[str, Any]:
    meta = row.get("metadata")
    return meta if isinstance(meta, dict) else {}


def _pred_mean(samples: list[dict]) -> float | None:
    values: list[float] = []
    for sample in samples:
        if not isinstance(sample, dict):
            continue
        scalar = sample.get("prediction_scalar")
        if scalar is None:
            scalar = extract_scalar_prediction(sample.get("parsed"))
        if scalar is None:
            continue
        try:
            values.append(float(scalar))
        except (TypeError, ValueError):
            continue
    if not values:
        return None
    return float(sum(values) / len(values))


def _extract_scalar_from_sample(sample: dict[str, Any]) -> float | int | None:
    scalar = extract_scalar_prediction(sample.get("parsed"))
    if scalar is not None:
        return scalar

    raw_text = sample.get("raw_text")
    if isinstance(raw_text, str) and raw_text:
        try:
            payload = json.loads(raw_text)
        except json.JSONDecodeError:
            payload = None
        if payload is not None:
            scalar = extract_scalar_prediction(payload)
            if scalar is not None:
                return scalar
            if isinstance(payload, dict):
                scalar = payload.get("prediction_scalar")
                if scalar is not None:
                    return scalar
                scalar = payload.get("prediction")
                if scalar is not None:
                    return scalar
        scalar = regex_extract_scalar(raw_text)
        if scalar is not None:
            return scalar
    return None


def _ensure_baseline_predictions(
    *,
    outdir: Path,
    model_name: str,
    baseline_path: Path | None,
) -> Path:
    """Return a baseline predictions JSONL for multi_expert runs.

    Resolution order:
    1) If --baseline-predictions is provided and exists, use it.
    2) Otherwise, reuse the most recent baseline predictions file under the
       same `_runs/` tree (by timestamp directory name).
    3) Otherwise, build baseline predictions from the nearest available
       data_only samples (current outdir, sibling data_only dir, or latest run).
    """

    if baseline_path is not None and baseline_path.exists():
        return baseline_path

    model_token = _safe_token(model_name)

    def _infer_batch_model_token(outdir: Path) -> str | None:
        # Expected layout: .../_runs/<ts>/_batch/<model_token>/<family>/uncalibrated
        family_dir = outdir.parent
        if not family_dir.is_dir():
            return None
        model_dir = family_dir.parent
        if not model_dir.is_dir() or not model_dir.name:
            return None
        if model_dir.parent.name != "_batch":
            return None
        return str(model_dir.name)

    batch_model_token = _infer_batch_model_token(outdir)

    # Prefer the on-disk batch token, but fall back to the model-name token.
    token_candidates: list[str] = []
    for tok in (batch_model_token, model_token):
        if tok and tok not in token_candidates:
            token_candidates.append(tok)

    def _find_runs_root(start: Path) -> Path | None:
        for p in (start, *start.parents):
            if p.name == "_runs":
                return p
        return None

    def _iter_run_dirs_desc(runs_root: Path) -> list[Path]:
        # Run directories are ISO timestamps (YYYY-MM-DDTHH-MM-SS), so
        # lexicographic ordering matches chronological ordering.
        run_dirs = [p for p in runs_root.iterdir() if p.is_dir()]
        return sorted(run_dirs, key=lambda p: p.name, reverse=True)

    def _find_latest_baseline_in_runs(*, runs_root: Path, tokens: list[str]) -> Path | None:
        for run_dir in _iter_run_dirs_desc(runs_root):
            for folder_token in tokens:
                for file_token in tokens:
                    candidate = (
                        run_dir
                        / "_batch"
                        / folder_token
                        / "multi_expert"
                        / "uncalibrated"
                        / f"baseline_predictions__data_only__{file_token}.jsonl"
                    )
                    if candidate.exists():
                        return candidate
        return None

    def _find_latest_data_only_samples_in_runs(*, runs_root: Path, tokens: list[str]) -> Path | None:
        for run_dir in _iter_run_dirs_desc(runs_root):
            for folder_token in tokens:
                for file_token in tokens:
                    candidates = [
                        run_dir / "_batch" / folder_token / "data_only" / "uncalibrated" / "samples__data_only.jsonl",
                        run_dir
                        / "_batch"
                        / folder_token
                        / "data_only"
                        / "uncalibrated"
                        / f"samples__data_only__{file_token}.jsonl",
                    ]
                    found = next((p for p in candidates if p.exists()), None)
                    if found is not None:
                        return found
        return None

    runs_root = _find_runs_root(outdir)

    # Prefer reusing an existing baseline predictions file (fast + stable).
    if runs_root is not None:
        latest = _find_latest_baseline_in_runs(runs_root=runs_root, tokens=token_candidates)
        if latest is not None:
            return latest

    # Baseline data-only samples are normally written into the same outdir.
    # Under the choice-validation batch runner, however, we sample each family
    # into its own directory (e.g. .../_batch/<model>/<family>/uncalibrated).
    # In that case, multi_expert needs to reuse the sibling data_only samples.
    candidates = [
        outdir / "samples__data_only.jsonl",
        outdir / f"samples__data_only__{model_token}.jsonl",
        # .../_batch/<model>/multi_expert/uncalibrated -> .../_batch/<model>/data_only/uncalibrated
        outdir.parent.parent / "data_only" / "uncalibrated" / "samples__data_only.jsonl",
        outdir.parent.parent / "data_only" / "uncalibrated" / f"samples__data_only__{model_token}.jsonl",
    ]
    if batch_model_token and batch_model_token != model_token:
        candidates.extend(
            [
                outdir / f"samples__data_only__{batch_model_token}.jsonl",
                outdir.parent.parent / "data_only" / "uncalibrated" / f"samples__data_only__{batch_model_token}.jsonl",
            ]
        )
    samples_path = next((p for p in candidates if p.exists()), None)

    # If the current output directory doesn't have data_only samples, fall back
    # to the most recent run under the same `_runs/` tree.
    if samples_path is None and runs_root is not None:
        samples_path = _find_latest_data_only_samples_in_runs(runs_root=runs_root, tokens=token_candidates)

    if samples_path is None:
        raise FileNotFoundError(
            "Missing data_only samples for baseline. Looked in: " + ", ".join(str(p) for p in candidates)
        )

    rows = read_jsonl(samples_path, dicts_only=True)
    # Aggregated sample files include `model` per row; per-model files don't.
    has_model_field = any(isinstance(r, dict) and "model" in r for r in rows)

    out_rows: list[dict[str, Any]] = []
    for row in rows:
        if has_model_field and row.get("model") != model_name:
            continue
        row_id = row.get("id")
        if not isinstance(row_id, str) or not row_id:
            continue
        samples = row.get("samples")
        if not isinstance(samples, list):
            continue
        y_pred = _pred_mean(samples)
        if y_pred is None:
            continue
        join_id = row_id.split("__", 1)[1] if "__" in row_id else row_id
        out_rows.append(
            {
                "id": row_id,
                "join_id": join_id,
                "outcome": _outcome(row),
                "y_pred": y_pred,
            }
        )

    if not out_rows:
        raise ValueError("No baseline predictions could be built from data_only samples.")

    out_token = batch_model_token or model_token
    out_path = baseline_path or outdir / f"baseline_predictions__data_only__{out_token}.jsonl"
    write_jsonl(out_path, out_rows, json_default=numpy_json_default)
    return out_path


def run_cli(argv: list[str] | None = None) -> None:
    parser = cli.ArgumentParser()
    parser.add_argument("--prompts", default=None, help="Prompt JSONL path or directory")
    parser.add_argument("--outdir", default="evaluation/output/debug_unperturbed")
    parser.add_argument("--prompt-family", default=None)
    parser.add_argument(
        "--prompt-families",
        default=",".join(sorted(SUPPORTED_PROMPT_FAMILIES)),
        help="Comma-separated prompt families (data_only,first_person,multi_expert)",
    )

    parser.add_argument("--backend", choices=["vllm", "azure_openai"], default="azure_openai")
    parser.add_argument("--models", required=True, help="Comma-separated model names")
    parser.add_argument("--model-options", default=None, help="JSON object of model options")
    parser.add_argument("--model-options-file", default=None, help="Path to JSON model options")

    parser.add_argument(
        "--azure-deployments",
        default=None,
        help=(
            "Comma-separated Azure deployment names aligned with --models. "
            "Required to compare multiple Azure deployments in one run."
        ),
    )
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
    parser.add_argument(
        "--max-new-tokens",
        dest="max_model_len",
        type=int,
        default=None,
        help=cli.SUPPRESS,
    )
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--max-concurrency", type=int, default=1024)
    parser.add_argument("--no-structured", action="store_true", help="Disable structured outputs")
    parser.add_argument("--method", default="debug_unperturbed")
    parser.add_argument("--run-id", default=None, help="Optional run id for row-level outputs")
    parser.add_argument(
        "--baseline-predictions",
        default=None,
        help="Path to baseline data_only predictions for multi_expert runs (CSV/JSONL).",
    )
    parser.add_argument(
        "--baseline-join-key",
        default="id",
        help="Join key for baseline predictions (default: id).",
    )
    parser.add_argument(
        "--baseline-col",
        default="y_pred",
        help="Column in baseline predictions to use (default: y_pred).",
    )

    args = parser.parse_args(argv)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    families = parse_prompt_families(args.prompt_family, args.prompt_families)

    models = parse_list_arg(args.models)
    if not models:
        raise ValueError("No models parsed from --models")

    azure_deployments = parse_list_arg(args.azure_deployments)
    if args.backend == "azure_openai" and len(models) > 1 and not azure_deployments:
        raise ValueError(
            "backend=azure_openai with multiple models requires --azure-deployments (one deployment per model)."
        )
    if azure_deployments and len(azure_deployments) != len(models):
        raise ValueError("--azure-deployments must have the same length as --models")

    for family in families:
        prompts_path = Path(args.prompts) if args.prompts else outdir / f"prompts__{family}.jsonl"
        if prompts_path.is_dir():
            prompts_path = prompts_path / f"prompts__{family}.jsonl"
        if not prompts_path.exists():
            raise FileNotFoundError(f"Prompt JSONL not found: {prompts_path}")

        if family == "multi_expert":
            from beam_abm.llm_microvalidation.shared import sampling as run_llm_sampling

            out_rows: list[dict] = []
            for i, model_name in enumerate(models):
                deployment = azure_deployments[i] if azure_deployments else None
                # Do NOT strip the provider prefix - pass the full model name as deployment
                if args.backend == "azure_openai" and not deployment:
                    deployment = str(model_name)
                baseline_path = Path(args.baseline_predictions) if args.baseline_predictions else None
                baseline_path = _ensure_baseline_predictions(
                    outdir=outdir,
                    model_name=str(model_name),
                    baseline_path=baseline_path,
                )
                model_token = _safe_token(model_name)
                out_path = outdir / f"samples__multi_expert__{model_token}.jsonl"

                argv_local = [
                    "--in",
                    str(prompts_path),
                    "--out",
                    str(out_path),
                    "--backend",
                    str(args.backend),
                    "--model",
                    str(model_name),
                    "--strategy",
                    "multi_expert",
                    "--baseline-predictions",
                    str(baseline_path),
                    "--baseline-join-key",
                    "join_id",
                    "--baseline-col",
                    str(args.baseline_col),
                    "--k",
                    str(max(1, int(args.k))),
                    "--two-pass",
                    "--max-model-len",
                    str(int(args.max_model_len)),
                    "--max-retries",
                    str(int(args.max_retries)),
                    "--method",
                    str(args.method),
                ]
                if args.run_id:
                    argv_local.extend(["--run-id", str(args.run_id)])
                if args.temperature is not None:
                    argv_local.extend(["--temperature", str(args.temperature)])
                if args.top_p is not None:
                    argv_local.extend(["--top-p", str(args.top_p)])
                if args.max_concurrency is not None:
                    argv_local.extend(["--max-concurrency", str(int(args.max_concurrency))])
                if args.backend == "azure_openai":
                    if deployment:
                        argv_local.extend(["--azure-deployment", str(deployment)])
                    if args.azure_endpoint:
                        argv_local.extend(["--azure-endpoint", str(args.azure_endpoint)])
                    if args.azure_api_version:
                        argv_local.extend(["--azure-api-version", str(args.azure_api_version)])
                    argv_local.extend(["--azure-model-hint", str(model_name)])

                run_llm_sampling.run_cli(argv_local)

                for row in read_jsonl(out_path, dicts_only=True):
                    out_rows.append({**row, "model": model_name})

            write_jsonl(outdir / "samples__multi_expert.jsonl", out_rows, json_default=numpy_json_default)
            continue

        from beam_abm.llm_microvalidation.shared import sampling as run_llm_sampling

        out_rows: list[dict] = []
        for i, model_name in enumerate(models):
            deployment = azure_deployments[i] if azure_deployments else None
            # Do NOT strip the provider prefix - pass the full model name as deployment
            if args.backend == "azure_openai" and not deployment:
                deployment = str(model_name)
            model_token = _safe_token(model_name)
            out_path = outdir / f"samples__{family}__{model_token}.jsonl"
            argv_local = [
                "--in",
                str(prompts_path),
                "--out",
                str(out_path),
                "--backend",
                str(args.backend),
                "--model",
                str(model_name),
                "--strategy",
                "direct",
                "--k",
                str(max(1, int(args.k))),
                "--two-pass",
                "--max-model-len",
                str(int(args.max_model_len)),
                "--max-retries",
                str(int(args.max_retries)),
                "--method",
                f"{args.method}::{family}",
            ]
            if args.run_id:
                argv_local.extend(["--run-id", str(args.run_id)])
            if args.temperature is not None:
                argv_local.extend(["--temperature", str(args.temperature)])
            if args.top_p is not None:
                argv_local.extend(["--top-p", str(args.top_p)])
            if args.max_concurrency is not None:
                argv_local.extend(["--max-concurrency", str(int(args.max_concurrency))])
            if args.no_structured:
                argv_local.append("--no-structured")
            if args.model_options is not None:
                argv_local.extend(["--model-options", str(args.model_options)])
            if args.model_options_file is not None:
                argv_local.extend(["--model-options-file", str(args.model_options_file)])
            if args.backend == "azure_openai":
                if deployment:
                    argv_local.extend(["--azure-deployment", str(deployment)])
                if args.azure_endpoint:
                    argv_local.extend(["--azure-endpoint", str(args.azure_endpoint)])
                if args.azure_api_version:
                    argv_local.extend(["--azure-api-version", str(args.azure_api_version)])
                argv_local.extend(["--azure-model-hint", str(model_name)])

            run_llm_sampling.run_cli(argv_local)

            for row in read_jsonl(out_path, dicts_only=True):
                if isinstance(row, dict):
                    for sample in row.get("samples") or []:
                        if isinstance(sample, dict) and sample.get("prediction_scalar") is None:
                            sample["prediction_scalar"] = _extract_scalar_from_sample(sample)
                out_rows.append({**row, "model": model_name})

        write_jsonl(outdir / f"samples__{family}.jsonl", out_rows, json_default=numpy_json_default)
