"""Run a tiny temperature sweep for Phase 0 prompt tournament (data_only)."""

from __future__ import annotations

import json
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from beam_abm.cli import parser_compat as cli
from beam_abm.common import get_logger
from beam_abm.llm_microvalidation.artifacts import (
    PromptTournamentOutputPaths,
    build_run_metadata,
    utc_timestamp,
    utc_timestamp_iso,
    write_run_metadata,
)
from beam_abm.llm_microvalidation.utils import paired_contrast_utils as pt
from beam_abm.llm_microvalidation.utils.alignment_utils import summarize_alignment_rows

logger = get_logger(__name__)


@dataclass(frozen=True)
class SweepConfig:
    outcomes: list[str]
    targets: list[str]
    grid_points: list[str]
    temps: list[tuple[str, float]]
    k: int
    seed: int
    limit: int | None
    shuffle: bool
    n_ids: int | None
    ids_seed: int
    dedup_levers: bool
    n_ids_per_target: int | None
    strategy: str
    model: str
    model_slug: str
    backend: str
    max_model_len: int
    max_concurrency: int
    model_options: dict[str, object]
    anchors_root: Path
    output_root: Path
    sweep_id: str
    resume: bool


def _parse_list_arg(raw: str | None) -> list[str]:
    if raw is None:
        return []
    return [part.strip() for part in raw.split(",") if part.strip()]


def _parse_temps(raw: str) -> list[tuple[str, float]]:
    temps: list[tuple[str, float]] = []
    for part in _parse_list_arg(raw):
        temps.append((part, float(part)))
    if not temps:
        raise ValueError("Temperature list is empty.")
    return temps


def _safe_token(value: str | None) -> str:
    text = str(value or "").strip()
    out = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in text).strip("_")
    return out or "unknown"


def _temp_token(raw: str) -> str:
    token = raw.strip().replace("-", "neg")
    token = token.replace(".", "p")
    token = token.replace("+", "")
    return token or "0"


def _count_jsonl_rows(path: Path) -> int:
    if not path.exists():
        return 0
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                count += 1
    return count


def _sample_prompt_ids(
    *,
    source: Path,
    dest: Path,
    n_ids: int,
    seed: int,
) -> int:
    rows: list[dict] = []
    ids: list[str] = []
    with source.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            row_id = row.get("id")
            if not isinstance(row_id, str) or not row_id:
                continue
            rows.append(row)
            ids.append(row_id)

    if not rows:
        raise ValueError(f"No prompt rows found in {source}")

    unique_ids = sorted(set(ids))
    if n_ids >= len(unique_ids):
        shutil.copyfile(source, dest)
        return len(unique_ids)

    rng = pd.Series(unique_ids).sample(n=n_ids, random_state=int(seed), replace=False).tolist()
    keep_ids = {str(x) for x in rng}

    dest.parent.mkdir(parents=True, exist_ok=True)
    kept = 0
    with dest.open("w", encoding="utf-8") as handle:
        for row in rows:
            row_id = row.get("id")
            if row_id in keep_ids:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                kept += 1
    return len(keep_ids)


def _sample_paired_ids_per_target(
    *,
    source: Path,
    dest: Path,
    n_ids_per_target: int,
    seed: int,
    keep_grid_points: set[str],
) -> dict[str, int]:
    rows: list[dict] = []
    by_target_id: dict[tuple[str, str], set[str]] = {}

    with source.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            rid = row.get("id")
            target = row.get("target") or row.get("metadata", {}).get("target")
            grid_point = row.get("grid_point") or row.get("metadata", {}).get("grid_point")
            if not isinstance(rid, str) or not isinstance(target, str) or not isinstance(grid_point, str):
                continue
            if keep_grid_points and grid_point not in keep_grid_points:
                continue
            rows.append(row)
            key = (target, rid)
            by_target_id.setdefault(key, set()).add(grid_point)

    if not rows:
        raise ValueError(f"No prompt rows found in {source}")

    eligible_by_target: dict[str, list[str]] = {}
    for (target, rid), points in by_target_id.items():
        if keep_grid_points.issubset(points):
            eligible_by_target.setdefault(target, []).append(rid)

    selected_ids: dict[str, set[str]] = {}
    shortfalls: dict[str, int] = {}
    for target, ids in eligible_by_target.items():
        ids = sorted(set(ids))
        if not ids:
            continue
        if len(ids) < n_ids_per_target:
            shortfalls[target] = len(ids)
            continue
        rng = pd.Series(ids).sample(
            n=n_ids_per_target,
            random_state=int(seed) + abs(hash(target)) % 100_000,
            replace=False,
        )
        selected_ids[target] = {str(x) for x in rng.tolist()}

    if not selected_ids:
        raise ValueError("No targets with eligible paired low/high ids.")
    if shortfalls:
        msg = ", ".join([f"{t}={c}" for t, c in sorted(shortfalls.items())])
        raise ValueError(
            "Not enough paired ids per target for requested sample size. "
            f"Requested={n_ids_per_target}; available: {msg}"
        )

    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", encoding="utf-8") as handle:
        for row in rows:
            rid = row.get("id")
            target = row.get("target") or row.get("metadata", {}).get("target")
            grid_point = row.get("grid_point") or row.get("metadata", {}).get("grid_point")
            if not isinstance(rid, str) or not isinstance(target, str) or not isinstance(grid_point, str):
                continue
            if keep_grid_points and grid_point not in keep_grid_points:
                continue
            keep = rid in selected_ids.get(target, set())
            if keep:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    return {target: len(ids) for target, ids in selected_ids.items()}


def _load_determinism_summary(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if df.empty:
        return df
    mask = (df["target"] == "__all__") & (df["calibration_status"] == "uncalibrated")
    return df.loc[mask].copy()


def _resolve_outcome_paths(anchors_root: Path, outcomes: list[str]) -> list[Path]:
    paths: list[Path] = []
    missing: list[str] = []
    for outcome in outcomes:
        path = anchors_root / outcome / "design.parquet"
        if not path.exists():
            missing.append(str(path))
            continue
        paths.append(path)
    if missing:
        raise FileNotFoundError("Missing design.parquet paths:\n" + "\n".join(missing))
    return paths


def _build_generate_args(config: SweepConfig, prompts_path: Path) -> list[str]:
    args = [
        "--in-dir",
        str(config.anchors_root),
        "--out",
        str(prompts_path),
        "--strategy",
        config.strategy,
        "--keep-outcomes",
        ",".join(config.outcomes),
        "--keep-targets",
        ",".join(config.targets),
        "--keep-grid-points",
        ",".join(config.grid_points),
    ]
    args.append("--dedup-levers" if config.dedup_levers else "--no-dedup-levers")
    if config.shuffle:
        args.append("--shuffle")
        args.extend(["--seed", str(config.seed)])
    if config.limit is not None:
        args.extend(["--limit", str(config.limit)])
    return args


def _build_sample_args(
    *,
    config: SweepConfig,
    prompts_path: Path,
    samples_path: Path,
    run_id: str,
    temperature: float,
) -> list[str]:
    args = [
        "--in",
        str(prompts_path),
        "--out",
        str(samples_path),
        "--backend",
        config.backend,
        "--model",
        config.model,
        "--k",
        str(config.k),
        "--two-pass",
        "--model-options",
        json.dumps(config.model_options),
        "--max-model-len-by-strategy-file",
        "evaluation/specs/prompt_tournament_max_model_len.json",
        "--run-id",
        run_id,
        "--max-concurrency",
        str(config.max_concurrency),
        "--temperature",
        str(temperature),
    ]
    multi_outcome = config.dedup_levers and len(config.outcomes) > 1
    if not multi_outcome:
        args.extend(["--outcome-scale", "continuous"])
    return args


def _write_run_metadata(
    *,
    paths: PromptTournamentOutputPaths,
    config: SweepConfig,
    temperature: float,
    start_time: float,
    start_time_utc: str,
) -> None:
    end_time_utc = utc_timestamp_iso()
    run_meta = build_run_metadata(
        run_id=paths.run_id,
        run_type="perturbed",
        strategy=config.strategy,
        outcomes=config.outcomes,
        start_time_utc=start_time_utc,
        end_time_utc=end_time_utc,
        runtime_seconds=time.time() - start_time,
        config_snapshot={
            "backend": config.backend,
            "model": config.model,
            "model_slug": config.model_slug,
            "k": config.k,
            "temperature": temperature,
            "model_options": config.model_options,
            "prompt_families": [config.strategy],
            "keep_outcomes": config.outcomes,
            "keep_targets": config.targets,
            "keep_grid_points": config.grid_points,
            "limit": config.limit,
            "shuffle": config.shuffle,
            "seed": config.seed,
            "n_ids": config.n_ids,
            "ids_seed": config.ids_seed,
            "dedup_levers": config.dedup_levers,
            "n_ids_per_target": config.n_ids_per_target,
            "sweep_id": config.sweep_id,
        },
        dataset_inputs={"anchors_root": str(config.anchors_root)},
        anchors_inputs={"ref_ice": "empirical/output/anchors/pe/mutable_engines/reduced/<outcome>/ice_ref.csv"},
        perturbation_inputs={"lever_config": "config/levers.json"},
    )
    write_run_metadata(paths.run_metadata_path(), run_meta)


def _summarize_alignment(path: Path) -> dict[str, object]:
    df = pd.read_csv(path)
    if df.empty:
        return {}
    summary = summarize_alignment_rows(df)
    if summary.empty:
        return {}
    return summary.iloc[0].to_dict()


def run_cli(argv: list[str] | None = None) -> None:
    parser = cli.ArgumentParser()
    parser.add_argument(
        "--output-root",
        default=os.getenv("PROMPT_TOURNAMENT_OUTPUT_ROOT", "evaluation/output/prompt_tournament"),
        help="Root output directory for prompt tournament.",
    )
    parser.add_argument(
        "--anchors-root",
        default=os.getenv(
            "PROMPT_TOURNAMENT_ANCHORS_ROOT",
            "empirical/output/anchors/pe/mutable_engines/reduced",
        ),
        help="Root directory containing per-outcome design.parquet files.",
    )
    parser.add_argument("--run-tag", default="temp_sweep", help="Base tag for sweep outputs.")
    parser.add_argument(
        "--timestamp-output",
        action=cli.BooleanOptionalAction,
        default=True,
        help="Append UTC timestamp to run tag.",
    )
    parser.add_argument(
        "--temps",
        default="0,0.2,0.7,1.0",
        help="Comma-separated temperature list.",
    )
    parser.add_argument("--k", type=int, default=5, help="Number of samples per prompt.")
    parser.add_argument("--seed", type=int, default=0, help="Seed for prompt shuffling.")
    parser.add_argument("--limit", type=int, default=None, help="Limit prompts (rows) after filtering.")
    parser.add_argument("--no-limit", dest="limit", action="store_const", const=None, help="Disable prompt limit.")
    parser.add_argument(
        "--n-ids",
        type=int,
        default=100,
        help="Number of unique ids (people) to keep after prompt generation.",
    )
    parser.add_argument(
        "--no-n-ids",
        dest="n_ids",
        action="store_const",
        const=None,
        help="Disable unique-id sampling and keep all prompt rows.",
    )
    parser.add_argument("--ids-seed", type=int, default=0, help="Seed for unique-id sampling.")
    parser.add_argument("--shuffle", action=cli.BooleanOptionalAction, default=True, help="Shuffle prompts.")
    parser.add_argument(
        "--n-ids-per-target",
        type=int,
        default=50,
        help="Sample N ids per target with paired low/high rows (default: 50).",
    )
    parser.add_argument(
        "--no-n-ids-per-target",
        dest="n_ids_per_target",
        action="store_const",
        const=None,
        help="Disable per-target paired sampling.",
    )
    parser.add_argument(
        "--dedup-levers",
        action=cli.BooleanOptionalAction,
        default=True,
        help="Enable multi-outcome deduplication in prompt generation.",
    )
    parser.add_argument(
        "--keep-outcomes",
        default="vax_willingness_Y,mask_when_pressure_high,stay_home_when_symptomatic",
        help="Comma-separated outcomes to keep.",
    )
    parser.add_argument(
        "--keep-targets",
        default="vaccine_risk_avg,institutional_trust_avg,social_norms_vax_avg",
        help="Comma-separated targets to keep.",
    )
    parser.add_argument("--keep-grid-points", default="low,high", help="Comma-separated grid points to keep.")
    parser.add_argument("--model", default="openai/gpt-oss-20b")
    parser.add_argument("--backend", default=os.getenv("PROMPT_TOURNAMENT_BACKEND", "azure_openai"))
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=8192,
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=int(os.getenv("PROMPT_TOURNAMENT_MAX_CONCURRENCY", "50")),
    )
    parser.add_argument(
        "--model-options",
        default=None,
        help="Optional JSON for model options (overrides max-model-len).",
    )
    parser.add_argument("--resume", action="store_true", help="Skip steps if outputs exist.")
    args = parser.parse_args(argv)

    outcomes = _parse_list_arg(args.keep_outcomes)
    targets = _parse_list_arg(args.keep_targets)
    grid_points = _parse_list_arg(args.keep_grid_points)
    temps = _parse_temps(args.temps)

    anchors_root = Path(args.anchors_root)
    output_root = Path(args.output_root)

    _resolve_outcome_paths(anchors_root, outcomes)

    sweep_id = pt.resolve_run_tag(args.run_tag, timestamp_output=args.timestamp_output) or utc_timestamp()

    model_options = json.loads(args.model_options) if args.model_options else {"max-model-len": args.max_model_len}
    if args.n_ids_per_target is not None and args.limit is not None:
        logger.warning("Ignoring --limit because --n-ids-per-target is set for paired sampling.")
        args.limit = None

    config = SweepConfig(
        outcomes=outcomes,
        targets=targets,
        grid_points=grid_points,
        temps=temps,
        k=int(args.k),
        seed=int(args.seed),
        limit=args.limit if args.limit is None else int(args.limit),
        shuffle=bool(args.shuffle),
        n_ids=args.n_ids if args.n_ids is None else int(args.n_ids),
        ids_seed=int(args.ids_seed),
        dedup_levers=bool(args.dedup_levers),
        n_ids_per_target=args.n_ids_per_target if args.n_ids_per_target is None else int(args.n_ids_per_target),
        strategy="data_only",
        model=str(args.model),
        model_slug=_safe_token(str(args.model)),
        backend=str(args.backend),
        max_model_len=int(args.max_model_len),
        max_concurrency=int(args.max_concurrency),
        model_options=model_options,
        anchors_root=anchors_root,
        output_root=output_root,
        sweep_id=sweep_id,
        resume=bool(args.resume),
    )

    sweep_dir = output_root / "temp_sweep" / config.model_slug / sweep_id
    sweep_dir.mkdir(parents=True, exist_ok=True)
    prompts_full = sweep_dir / "prompts_full.jsonl"
    prompts_shared = sweep_dir / "prompts.jsonl"

    from beam_abm.llm_microvalidation.behavioural_outcomes.paired_contrast import posthoc_tests as posthoc_step
    from beam_abm.llm_microvalidation.behavioural_outcomes.paired_contrast.generate_prompts import (
        run_cli as generate_prompts_step,
    )
    from beam_abm.llm_microvalidation.shared.compute_alignment import run_cli as align_step
    from beam_abm.llm_microvalidation.shared.compute_ice_from_samples import run_cli as ice_step
    from beam_abm.llm_microvalidation.shared.sampling import run_cli as sample_step

    if not (config.resume and prompts_shared.exists()):
        logger.info("Generating prompts for sweep.")
        generate_args = _build_generate_args(config, prompts_full)
        pt.run_step(generate_prompts_step, generate_args)
        if config.n_ids_per_target is not None:
            kept_by_target = _sample_paired_ids_per_target(
                source=prompts_full,
                dest=prompts_shared,
                n_ids_per_target=int(config.n_ids_per_target),
                seed=int(config.ids_seed),
                keep_grid_points=set(config.grid_points),
            )
            logger.info(f"Kept paired ids per target: {kept_by_target}")
        elif config.n_ids is not None:
            kept_ids = _sample_prompt_ids(
                source=prompts_full,
                dest=prompts_shared,
                n_ids=int(config.n_ids),
                seed=int(config.ids_seed),
            )
            logger.info(f"Kept {kept_ids} unique ids in {prompts_shared}")
        else:
            shutil.copyfile(prompts_full, prompts_shared)

    prompt_count = _count_jsonl_rows(prompts_shared)
    logger.info(f"Prompt count: {prompt_count}")

    summary_rows: list[dict[str, object]] = []

    for raw_temp, temp in config.temps:
        temp_token = _temp_token(raw_temp)
        run_start = time.time()
        run_start_utc = utc_timestamp_iso()
        run_id = f"{sweep_id}__temp_{temp_token}"
        paths = PromptTournamentOutputPaths(
            run_type="perturbed",
            strategy=config.strategy,
            run_id=run_id,
            outcomes=config.outcomes,
            calibrated=False,
            model_slug=config.model_slug,
            root=config.output_root,
        )
        base_dir = paths.base_dir()
        base_dir.mkdir(parents=True, exist_ok=True)

        prompts_path = paths.prompts_path()
        if not (config.resume and prompts_path.exists()):
            shutil.copyfile(prompts_shared, prompts_path)

        samples_path = paths.samples_path()
        if not (config.resume and samples_path.exists()):
            logger.info(f"Sampling temp={raw_temp} -> {samples_path}")
            sample_args = _build_sample_args(
                config=config,
                prompts_path=prompts_path,
                samples_path=samples_path,
                run_id=run_id,
                temperature=temp,
            )
            pt.run_step(sample_step, sample_args)

        ice_path = paths.ice_path()
        if not (config.resume and ice_path.exists()):
            logger.info(f"Computing ICE for temp={raw_temp}")
            pt.run_step(ice_step, ["--in", str(samples_path), "--out", str(ice_path)])

        alignment_paths: list[Path] = []
        for outcome in config.outcomes:
            ref_ice = Path(f"empirical/output/anchors/pe/mutable_engines/reduced/{outcome}/ice_ref.csv")
            if not ref_ice.exists():
                logger.warning(f"Missing ref ICE for {outcome}; skipping alignment.")
                continue
            metrics_out = base_dir / f"_alignment_rows__{outcome}.csv"
            order_out = base_dir / f"_order_effects__{outcome}.csv"
            alignment_paths.append(metrics_out)
            if not (config.resume and metrics_out.exists() and order_out.exists()):
                align_args = [
                    "--ref",
                    str(ref_ice),
                    "--llm",
                    str(ice_path),
                    "--out",
                    str(metrics_out),
                    "--order-out",
                    str(order_out),
                ]
                pt.run_step(align_step, align_args)

        for outcome in config.outcomes:
            metrics_out = base_dir / f"_alignment_rows__{outcome}.csv"
            if not metrics_out.exists():
                continue
            align_summary = _summarize_alignment(metrics_out)
            if not align_summary:
                continue
            row = {
                "run_id": run_id,
                "sweep_id": sweep_id,
                "strategy": config.strategy,
                "outcome": outcome,
                "temperature": float(temp),
                "k": config.k,
                "prompt_rows": prompt_count,
            }
            row.update(align_summary)
            summary_rows.append(row)

        if alignment_paths:
            summary_frames = [pd.read_csv(path) for path in alignment_paths if path.exists()]
            if summary_frames:
                summary_rows_all = pd.concat(summary_frames, ignore_index=True)
                summarize_alignment_rows(summary_rows_all).to_csv(paths.summary_metrics_path(), index=False)

        if not (config.resume and (base_dir / "posthoc" / "determinism_summary.csv").exists()):
            logger.info(f"Running post-hoc determinism for temp={raw_temp}")
            posthoc_args = [
                "--output-root",
                str(config.output_root),
                "--run-id",
                run_id,
                "--model-slug",
                config.model_slug,
                "--strategies",
                config.strategy,
                "--skip-midpoint",
                "--skip-reliability",
            ]
            pt.run_step(posthoc_step.run_cli, posthoc_args)

        determinism_path = base_dir / "posthoc" / "determinism_summary.csv"
        det_df = _load_determinism_summary(determinism_path)
        if not det_df.empty:
            for _idx, row in enumerate(summary_rows):
                if row.get("run_id") != run_id:
                    continue
                outcome = row.get("outcome")
                match = det_df.loc[det_df["outcome"] == outcome]
                if match.empty:
                    continue
                first = match.iloc[0]
                row["mean_sd_low"] = float(first.get("mean_sd_low", float("nan")))
                row["mean_sd_high"] = float(first.get("mean_sd_high", float("nan")))
                row["mean_sd_delta"] = float(first.get("mean_sd_delta", float("nan")))
                row["median_sd_delta"] = float(first.get("median_sd_delta", float("nan")))

        _write_run_metadata(
            paths=paths,
            config=config,
            temperature=temp,
            start_time=run_start,
            start_time_utc=run_start_utc,
        )

        for path in alignment_paths:
            path.unlink(missing_ok=True)
        for outcome in config.outcomes:
            (base_dir / f"_order_effects__{outcome}.csv").unlink(missing_ok=True)

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_path = sweep_dir / "sweep_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        logger.info(f"Wrote sweep summary: {summary_path}")
