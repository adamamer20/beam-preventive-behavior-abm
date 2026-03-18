"""Per-outcome split and canonical rebuild orchestration for perturbed runs."""

from __future__ import annotations

import contextlib
import hashlib
import json
import re
import shutil
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from beam_abm.evaluation.choice.canonicalize import normalize_model_slug, write_jsonl
from beam_abm.evaluation.choice.perturbed_alignment import rebuild_alignment_summary
from beam_abm.evaluation.choice.perturbed_ingest import (
    _iter_jsonl_dicts,
    _row_outcome,
    _safe_token,
    normalize_model_dirname,
)
from beam_abm.evaluation.utils.id_utils import normalize_perturbed_dataset_id


def _row_merge_key(row: dict[str, Any], *, outcome: str | None = None) -> str:
    meta = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    target = meta.get("target") or row.get("target") or "unknown_target"
    grid_point = meta.get("grid_point") or row.get("grid_point") or "paired"
    paired_order = meta.get("paired_order") or row.get("paired_order") or ""

    raw_id = row.get("dataset_id") or row.get("id")
    base_id = normalize_perturbed_dataset_id(raw_id, outcome=outcome) or raw_id
    if not isinstance(base_id, str) or not base_id.strip():
        base_id = "unknown_id"

    return "::".join(
        [
            _safe_token(base_id),
            _safe_token(target),
            _safe_token(grid_point),
            _safe_token(paired_order),
        ]
    )


def _sample_dedupe_id_pert(*, row_key: str, sample: dict[str, Any], sample_idx: int) -> str:
    conv_id = sample.get("conversation_id")
    if isinstance(conv_id, str) and conv_id.strip():
        return conv_id.strip()
    raw_text = sample.get("raw_text_original") or sample.get("raw_text") or ""
    scalar = sample.get("prediction_scalar")
    parsed = sample.get("parsed")
    payload = f"{row_key}::{sample_idx}::{raw_text}::{scalar}::{json.dumps(parsed, sort_keys=True, ensure_ascii=False)}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _merge_samples_perturbed(samples_paths: list[Path], *, outcome: str | None = None) -> list[dict[str, Any]]:
    rows_by_key: dict[str, dict[str, Any]] = {}
    sample_ids_by_key: dict[str, set[str]] = {}

    def _is_missing(value: Any) -> bool:
        if value is None:
            return True
        if isinstance(value, float | np.floating):
            return not np.isfinite(float(value))
        if isinstance(value, str):
            return not value.strip()
        return False

    def _fill_missing_fields(dst: dict[str, Any], src: dict[str, Any], *, keys: tuple[str, ...]) -> None:
        for k in keys:
            if _is_missing(dst.get(k)) and not _is_missing(src.get(k)):
                dst[k] = src.get(k)

    def _fill_missing_metadata(dst: dict[str, Any], src: dict[str, Any]) -> None:
        dst_meta = dst.get("metadata")
        src_meta = src.get("metadata")
        if not isinstance(src_meta, dict) or not src_meta:
            return
        if not isinstance(dst_meta, dict):
            dst["metadata"] = dict(src_meta)
            return
        for k, v in src_meta.items():
            if _is_missing(dst_meta.get(k)) and not _is_missing(v):
                dst_meta[k] = v

    for path in samples_paths:
        for row in _iter_jsonl_dicts(path):
            key = _row_merge_key(row, outcome=outcome)
            base = rows_by_key.get(key)
            if base is None:
                base = dict(row)
                raw_id = base.get("dataset_id") or base.get("id")
                base_id = normalize_perturbed_dataset_id(raw_id, outcome=outcome) or raw_id
                if isinstance(base_id, str) and base_id.strip():
                    base["dataset_id"] = base_id
                    base["id"] = base_id
                base["samples"] = []
                rows_by_key[key] = base
                sample_ids_by_key[key] = set()
            else:
                _fill_missing_fields(
                    base,
                    row,
                    keys=(
                        "id",
                        "dataset_id",
                        "model",
                        "prompt_family",
                        "strategy",
                        "country",
                        "outcome",
                        "outcome_type",
                    ),
                )
                _fill_missing_metadata(base, row)

            samples = row.get("samples")
            if not isinstance(samples, list):
                samples = []

            keep = sample_ids_by_key[key]
            merged_samples: list[dict[str, Any]] = base.get("samples") if isinstance(base.get("samples"), list) else []
            for idx, sample in enumerate(samples):
                if not isinstance(sample, dict):
                    continue
                sid = _sample_dedupe_id_pert(row_key=key, sample=sample, sample_idx=int(idx))
                if sid in keep:
                    continue
                keep.add(sid)
                merged_samples.append(sample)

            base["samples"] = merged_samples

    return [rows_by_key[k] for k in sorted(rows_by_key.keys())]


def _write_predictions_from_row_level(
    *,
    row_level_path: Path,
    out_path: Path,
    model: str,
    strategy: str,
    run_id: str | None,
    calibration_status: str,
) -> None:
    if not row_level_path.exists():
        return

    df = pd.read_csv(row_level_path)
    if df.empty:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        return

    # Aggregate across samples (conversation_id) per dataset row.
    group_cols = [
        c
        for c in ["dataset_id", "outcome", "strategy", "perturbation_id", "perturbation_type", "country"]
        if c in df.columns
    ]
    if not group_cols:
        # Fallback: write minimal placeholder.
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        return

    df = df.copy()
    if "reference_value" in df.columns:
        df["reference_value"] = pd.to_numeric(df["reference_value"], errors="coerce")
    if "prediction_scalar" in df.columns:
        df["prediction_scalar"] = pd.to_numeric(df["prediction_scalar"], errors="coerce")

    agg = (
        df.groupby(group_cols, dropna=False)
        .agg(
            y_true=("reference_value", "first") if "reference_value" in df.columns else (group_cols[0], "size"),
            y_pred=("prediction_scalar", "mean") if "prediction_scalar" in df.columns else (group_cols[0], "size"),
            n_samples=("conversation_id", "nunique") if "conversation_id" in df.columns else (group_cols[0], "size"),
        )
        .reset_index()
    )

    # Normalize columns to match unperturbed-ish naming.
    if "reference_value" in df.columns:
        agg["y_true"] = pd.to_numeric(agg["y_true"], errors="coerce")
    if "prediction_scalar" in df.columns:
        agg["y_pred"] = pd.to_numeric(agg["y_pred"], errors="coerce")

    agg.insert(0, "model", model)
    if "strategy" not in agg.columns:
        agg["strategy"] = strategy
    if run_id is not None:
        agg["run_id"] = run_id
    agg["calibration_status"] = calibration_status

    out_path.parent.mkdir(parents=True, exist_ok=True)
    agg.to_csv(out_path, index=False)


def _write_row_level_from_samples(*, samples_path: Path, out_path: Path, strategy: str, run_id: str | None) -> None:
    """Best-effort row_level.csv synthesizer for legacy runs that only have samples.jsonl."""

    rows: list[dict[str, Any]] = []
    for r in _iter_jsonl_dicts(samples_path):
        outcome = _row_outcome(r)
        if not outcome:
            continue
        meta = r.get("metadata") if isinstance(r.get("metadata"), dict) else {}
        dataset_id = r.get("dataset_id") or r.get("id")
        y_true = meta.get("y_true") if isinstance(meta, dict) else None

        samples = r.get("samples") if isinstance(r.get("samples"), list) else []
        scalars: list[float] = []
        for s in samples:
            if not isinstance(s, dict):
                continue
            v = s.get("prediction_scalar")
            try:
                if v is not None:
                    scalars.append(float(v))
            except (TypeError, ValueError):
                continue

        pred = float(sum(scalars) / len(scalars)) if scalars else None
        rows.append(
            {
                "dataset_id": dataset_id,
                "outcome": outcome,
                "reference_value": y_true,
                "prediction_scalar": pred,
                "strategy": r.get("strategy") or strategy,
                "run_id": run_id,
            }
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)


def _rebuild_ice_from_samples(*, status_dir: Path) -> None:
    from beam_abm.evaluation.common.ice_from_samples import compute_ice_from_samples

    samples_path = status_dir / "samples.jsonl"
    if not samples_path.exists():
        return

    out_path = status_dir / "ice_llm.csv"
    summary_out = status_dir / "_ice_noncompliance.csv"

    compute_ice_from_samples(
        in_path=samples_path,
        out_path=out_path,
        summary_out=summary_out,
        paired_profile_swap_on_b_then_a=True,
    )


@contextlib.contextmanager
def _tmp_dir(parent: Path, *, prefix: str):
    parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=prefix, dir=str(parent)))
    try:
        yield tmp
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def rebuild_canonical_from_runs(
    *,
    output_root: Path,
    merge_runs: bool = True,
    anchors_root: Path | None = None,
    ref_models: list[str] | None = None,
    model_plan_path: Path | None = None,
    block_topk: int = 5,
    skip_block_alignment: bool = False,
    use_model_ref_for_linear: bool = False,
    ref_model_by_outcome: dict[str, str] | None = None,
) -> None:
    """Rebuild canonical perturbed leaves from `_runs/`.

    By default we merge across different run_ids when samples.jsonl is available,
    deduping rows + samples to form a stable canonical leaf. If no samples are
    available, we fall back to selecting a single run_id (latest).
    """

    runs_root = output_root / "_runs"
    locks_root = output_root / "_locks"
    locks_root.mkdir(parents=True, exist_ok=True)

    # Map (outcome, model, strategy, status) -> list[(run_id, leaf_dir)]
    buckets: dict[tuple[str, str, str, str], list[tuple[str, Path]]] = {}
    if not runs_root.exists():
        return

    for run_dir in sorted(p for p in runs_root.iterdir() if p.is_dir() and not p.name.startswith("_")):
        run_id = run_dir.name
        for outcome_dir in sorted(p for p in run_dir.iterdir() if p.is_dir() and not p.name.startswith("_")):
            outcome = outcome_dir.name
            for model_dir in sorted(p for p in outcome_dir.iterdir() if p.is_dir() and not p.name.startswith("_")):
                model = normalize_model_dirname(normalize_model_slug(model_dir.name, reasoning_effort=None))
                for strategy_dir in sorted(p for p in model_dir.iterdir() if p.is_dir() and not p.name.startswith("_")):
                    strategy = strategy_dir.name
                    for status_dir in sorted(
                        p for p in strategy_dir.iterdir() if p.is_dir() and p.name in {"uncalibrated", "calibrated"}
                    ):
                        status = status_dir.name
                        if not (status_dir / "row_level.csv").exists() and not (status_dir / "samples.jsonl").exists():
                            continue
                        key = (outcome, model, strategy, status)
                        buckets.setdefault(key, []).append((run_id, status_dir))

    ts_pattern = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}")

    def _run_id_sort_key(run_id: str) -> tuple[int, str, str]:
        # Prefer the latest timestamp embedded in the run_id when available.
        matches = ts_pattern.findall(run_id)
        if matches:
            # Use the last timestamp if multiple appear.
            return (1, matches[-1], run_id)
        return (0, "", run_id)

    for (outcome, model, strategy, status), candidates in sorted(buckets.items()):
        # Choose the latest run_id by embedded timestamp, fallback to lexicographic.
        chosen_run_id, chosen_dir = max(candidates, key=lambda x: _run_id_sort_key(x[0]))

        canonical_strategy_root = output_root / outcome / model / strategy
        canonical_status_root = canonical_strategy_root / status

        lock_path = locks_root / outcome / model / f"{strategy}.lock"
        lock_path.parent.mkdir(parents=True, exist_ok=True)

        from beam_abm.evaluation.choice.canonicalize import leaf_lock

        with leaf_lock(lock_path):
            canonical_strategy_root.mkdir(parents=True, exist_ok=True)

            samples_paths: list[Path] = []
            merged_run_ids: list[str] = []
            if merge_runs:
                for run_id, status_dir in candidates:
                    samples_path = status_dir / "samples.jsonl"
                    if samples_path.exists():
                        samples_paths.append(samples_path)
                        merged_run_ids.append(run_id)

            with _tmp_dir(canonical_strategy_root, prefix=f".tmp_{status}_") as tmp:
                tmp_status = tmp / status
                tmp_status.mkdir(parents=True, exist_ok=True)

                if samples_paths:
                    merged_rows = _merge_samples_perturbed(samples_paths, outcome=outcome)
                    write_jsonl(tmp_status / "samples.jsonl", merged_rows)
                    _write_row_level_from_samples(
                        samples_path=tmp_status / "samples.jsonl",
                        out_path=tmp_status / "row_level.csv",
                        strategy=strategy,
                        run_id="canonical",
                    )
                    _write_predictions_from_row_level(
                        row_level_path=tmp_status / "row_level.csv",
                        out_path=tmp_status / "predictions.csv",
                        model=model,
                        strategy=strategy,
                        run_id="canonical",
                        calibration_status=status,
                    )
                    _rebuild_ice_from_samples(status_dir=tmp_status)
                    if anchors_root is not None and ref_models:
                        rebuild_alignment_summary(
                            status_dir=tmp_status,
                            anchors_root=anchors_root,
                            ref_models=ref_models,
                            model_plan_path=model_plan_path,
                            block_topk=block_topk,
                            skip_block_alignment=skip_block_alignment,
                            use_model_ref_for_linear=use_model_ref_for_linear,
                            ref_model_by_outcome=ref_model_by_outcome,
                        )
                else:
                    # Fallback: copy through known artifacts from latest run.
                    for name in [
                        "prompts.jsonl",
                        "samples.jsonl",
                        "raw_conversations.jsonl",
                        "row_level.csv",
                        "ice_llm.csv",
                        "summary_metrics.csv",
                        "run_metadata.json",
                    ]:
                        src = chosen_dir / name
                        if src.exists():
                            shutil.copy2(src, tmp_status / name)

                    if not (tmp_status / "row_level.csv").exists() and (tmp_status / "samples.jsonl").exists():
                        _write_row_level_from_samples(
                            samples_path=tmp_status / "samples.jsonl",
                            out_path=tmp_status / "row_level.csv",
                            strategy=strategy,
                            run_id=chosen_run_id,
                        )

                    src_conv = chosen_dir / "conversations"
                    if src_conv.exists() and src_conv.is_dir():
                        shutil.copytree(src_conv, tmp_status / "conversations")

                    _write_predictions_from_row_level(
                        row_level_path=tmp_status / "row_level.csv",
                        out_path=tmp_status / "predictions.csv",
                        model=model,
                        strategy=strategy,
                        run_id=chosen_run_id,
                        calibration_status=status,
                    )

                if canonical_status_root.exists():
                    shutil.rmtree(canonical_status_root, ignore_errors=True)
                tmp_status.rename(canonical_status_root)

            if anchors_root is not None and ref_models:
                summary_path = canonical_status_root / "summary_metrics.csv"
                if not summary_path.exists():
                    rebuild_alignment_summary(
                        status_dir=canonical_status_root,
                        anchors_root=anchors_root,
                        ref_models=ref_models,
                        model_plan_path=model_plan_path,
                        block_topk=block_topk,
                        skip_block_alignment=skip_block_alignment,
                        use_model_ref_for_linear=use_model_ref_for_linear,
                        ref_model_by_outcome=ref_model_by_outcome,
                    )

            meta_path = canonical_strategy_root / "canonical_metadata.json"
            payload: dict[str, Any] = {
                "built_at_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "canonical_mode": "merge_runs" if samples_paths else "select_latest_run_id",
                "selected_run_id": chosen_run_id if not samples_paths else None,
                "merged_run_ids": sorted(set(merged_run_ids)) if samples_paths else [],
                "available_run_ids": [rid for rid, _ in sorted(candidates, key=lambda x: _run_id_sort_key(x[0]))],
                "status": status,
                "source_leaf": str(chosen_dir),
            }
            meta_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


__all__ = [
    "rebuild_canonical_from_runs",
]
