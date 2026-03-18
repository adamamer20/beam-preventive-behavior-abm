from __future__ import annotations

import contextlib
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from beam_abm.evaluation.choice._canonicalize_ids import _row_dataset_id, _sample_dedupe_id
from beam_abm.evaluation.choice._canonicalize_metrics import compute_metrics_df
from beam_abm.evaluation.choice._canonicalize_predictions import samples_to_predictions_df
from beam_abm.evaluation.choice._canonicalize_types import CanonicalizeStats
from beam_abm.evaluation.common.calibrate_predictions import calibrate_predictions
from beam_abm.evaluation.utils.jsonl import read_jsonl as _read_jsonl_shared
from beam_abm.evaluation.utils.jsonl import write_jsonl as _write_jsonl_shared


def find_samples_jsonl(dir_path: Path) -> Path | None:
    """Find the most relevant samples jsonl within an unperturbed leaf dir."""

    if not dir_path.exists() or not dir_path.is_dir():
        return None
    direct = dir_path / "samples.jsonl"
    if direct.exists():
        return direct
    candidates = sorted(dir_path.glob("samples*.jsonl"))
    if not candidates:
        return None
    # Prefer the largest file (typically the combined samples file).
    return max(candidates, key=lambda p: p.stat().st_size)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return _read_jsonl_shared(path, dicts_only=True)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    def _default(o: Any) -> Any:
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.bool_):
            return bool(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(f"Not JSON serializable: {type(o)}")
    _write_jsonl_shared(path, rows, json_default=_default)


def merge_samples(
    samples_paths: list[Path],
) -> tuple[list[dict[str, Any]], CanonicalizeStats]:
    rows_by_id: dict[str, dict[str, Any]] = {}
    sample_ids_by_row: dict[str, set[str]] = {}

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

    rows_in = 0
    samples_in = 0
    runs_seen = 0

    for path in samples_paths:
        runs_seen += 1
        run_rows = read_jsonl(path)
        for row in run_rows:
            rows_in += 1
            dataset_id = _row_dataset_id(row)
            samples = row.get("samples")
            if not isinstance(samples, list):
                samples = []
            samples_in += len(samples)

            base = rows_by_id.get(dataset_id)
            if base is None:
                base = dict(row)
                base["dataset_id"] = dataset_id
                # Ensure we always preserve id too (other code relies on it).
                base.setdefault("id", dataset_id)
                base["samples"] = []
                rows_by_id[dataset_id] = base
                sample_ids_by_row[dataset_id] = set()
            else:
                # Back-fill missing row-level fields from later runs.
                _fill_missing_fields(
                    base,
                    row,
                    keys=(
                        "id",
                        "model",
                        "prompt_family",
                        "strategy",
                        "country",
                        "outcome",
                        "outcome_type",
                        "y_true",
                    ),
                )
                _fill_missing_metadata(base, row)

            keep = sample_ids_by_row[dataset_id]
            merged_samples: list[dict[str, Any]] = base.get("samples") if isinstance(base.get("samples"), list) else []
            for idx, sample in enumerate(samples):
                if not isinstance(sample, dict):
                    continue
                sid = _sample_dedupe_id(row=row, sample=sample, sample_idx=int(idx))
                if sid in keep:
                    continue
                keep.add(sid)
                merged_samples.append(sample)

            base["samples"] = merged_samples

    merged_rows = [rows_by_id[k] for k in sorted(rows_by_id.keys())]
    rows_out = len(merged_rows)
    samples_out = sum(len(r.get("samples", [])) for r in merged_rows if isinstance(r.get("samples"), list))

    return merged_rows, CanonicalizeStats(
        runs_seen=runs_seen,
        rows_in=rows_in,
        rows_out=rows_out,
        samples_in=samples_in,
        samples_out=samples_out,
    )


@contextlib.contextmanager
def leaf_lock(lock_path: Path):
    """Advisory lock around canonical leaf updates (Linux-only)."""

    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as f:
        try:
            import fcntl

            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        except (ImportError, OSError):
            # Best-effort; on non-POSIX environments this becomes a no-op.
            pass
        try:
            yield
        finally:
            try:
                import fcntl

                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            except (ImportError, OSError):
                pass


def rebuild_canonical_leaf(
    *,
    canonical_leaf: Path,
    samples_paths: list[Path],
    strategy: str,
    locks_root: Path | None = None,
) -> CanonicalizeStats:
    """Rebuild canonical leaf from a set of run samples.jsonl paths."""

    canonical_leaf.mkdir(parents=True, exist_ok=True)
    lock_path = canonical_leaf / ".lock"
    if locks_root is not None:
        # canonical_leaf is expected to be <root>/<outcome>/<model>/<strategy>
        outcome = canonical_leaf.parent.parent.name
        model = canonical_leaf.parent.name
        strat = canonical_leaf.name
        lock_path = locks_root / outcome / model / f"{strat}.lock"

    with leaf_lock(lock_path):
        prior_run_ids: set[str] = set()
        meta_path = canonical_leaf / "canonical_metadata.json"
        if meta_path.exists():
            try:
                prior_obj = json.loads(meta_path.read_text(encoding="utf-8"))
                if isinstance(prior_obj, dict) and isinstance(prior_obj.get("merged_run_ids"), list):
                    prior_run_ids = {str(x) for x in prior_obj["merged_run_ids"] if str(x).strip()}
            except (OSError, json.JSONDecodeError):
                prior_run_ids = set()

        merged_rows, stats = merge_samples(samples_paths)

        expected_outcome = canonical_leaf.parent.parent.name

        def _row_outcome(row: dict[str, Any]) -> str | None:
            raw = row.get("outcome")
            if isinstance(raw, str) and raw.strip():
                return raw.strip()
            meta = row.get("metadata")
            if isinstance(meta, dict):
                m = meta.get("outcome")
                if isinstance(m, str) and m.strip():
                    return m.strip()
            rid = row.get("dataset_id") or row.get("id")
            if isinstance(rid, str) and rid.strip() and "__" in rid:
                return rid.split("__", 1)[0].strip() or None
            return None

        filtered_rows = [r for r in merged_rows if _row_outcome(r) == expected_outcome]
        filtered_stats = CanonicalizeStats(
            runs_seen=stats.runs_seen,
            rows_in=stats.rows_in,
            rows_out=len(filtered_rows),
            samples_in=stats.samples_in,
            samples_out=sum(len(r.get("samples", [])) for r in filtered_rows if isinstance(r.get("samples"), list)),
        )

        tmp_dir = Path(tempfile.mkdtemp(prefix=".tmp_leaf_", dir=str(canonical_leaf)))
        try:
            tmp_uncal = tmp_dir / "uncalibrated"
            tmp_cal = tmp_dir / "calibrated"
            tmp_uncal.mkdir(parents=True, exist_ok=True)
            tmp_cal.mkdir(parents=True, exist_ok=True)

            # Uncalibrated artifacts.
            write_jsonl(tmp_uncal / "samples.jsonl", filtered_rows)
            pred_df = samples_to_predictions_df(filtered_rows, strategy=strategy)
            pred_df.to_csv(tmp_uncal / "predictions.csv", index=False)
            metrics_uncal = compute_metrics_df(
                pred_df,
                pred_col="y_pred",
                run_id="canonical",
                strategy=strategy,
                status="uncalibrated",
            )
            metrics_uncal.to_csv(tmp_uncal / "summary_metrics.csv", index=False)
            metrics_uncal.to_csv(tmp_uncal / "metrics_summary.csv", index=False)

            # Calibrated artifacts (refit calibrator).
            calibration_out = tmp_cal / "calibration.json"
            calibrated_out = tmp_cal / "predictions_calibrated.csv"
            calibrate_predictions(
                in_path=tmp_uncal / "predictions.csv",
                out_path=calibrated_out,
                calibration_out=calibration_out,
                country_col="country",
            )

            cal_df = pd.read_csv(calibrated_out)
            metrics_cal = compute_metrics_df(
                cal_df,
                pred_col="y_pred_cal",
                run_id="canonical",
                strategy=strategy,
                status="calibrated",
            )
            metrics_cal.to_csv(tmp_cal / "summary_metrics.csv", index=False)
            metrics_cal.to_csv(tmp_cal / "metrics_summary.csv", index=False)

            # Ensure canonical leaf does not contain ICE artifacts.
            for unwanted in ["ice_llm.csv", "ice_llm_calibrated.csv"]:
                (tmp_uncal / unwanted).unlink(missing_ok=True)
                (tmp_cal / unwanted).unlink(missing_ok=True)

            # Atomic-ish per-leaf swap while keeping leaf directory stable.
            for sub in ["uncalibrated", "calibrated"]:
                dst = canonical_leaf / sub
                new = tmp_dir / sub
                backup = None
                if dst.exists():
                    backup = canonical_leaf / f".{sub}.old_{hashlib.sha256(os.urandom(16)).hexdigest()[:8]}"
                    dst.rename(backup)
                new.rename(dst)
                if backup is not None:
                    shutil.rmtree(backup, ignore_errors=True)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

        # Remove any legacy ICE in-place if it existed.
        (canonical_leaf / "uncalibrated" / "ice_llm.csv").unlink(missing_ok=True)
        (canonical_leaf / "calibrated" / "ice_llm_calibrated.csv").unlink(missing_ok=True)

        # Self-describing canonical metadata.
        run_ids: set[str] = set(prior_run_ids)
        for p in samples_paths:
            parts = p.resolve().parts
            if "_runs" in parts:
                idx = parts.index("_runs")
                if len(parts) > idx + 1:
                    rid = str(parts[idx + 1]).strip()
                    if rid:
                        run_ids.add(rid)

        payload: dict[str, Any] = {
            "built_at_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "code_version": _git_commit_hash(canonical_leaf),
            "merged_run_ids": sorted(run_ids),
            "runs_seen_in_merge": int(filtered_stats.runs_seen),
            "rows": {"in": int(filtered_stats.rows_in), "out": int(filtered_stats.rows_out)},
            "samples": {
                "in": int(filtered_stats.samples_in),
                "out": int(filtered_stats.samples_out),
                "dedupe_rate": (
                    float(filtered_stats.samples_in - filtered_stats.samples_out) / float(filtered_stats.samples_in)
                )
                if filtered_stats.samples_in
                else None,
            },
        }
        meta_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return filtered_stats


def _git_commit_hash(start: Path) -> str | None:
    git_root = _find_git_root(start)
    if git_root is None:
        return None
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(git_root),
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            return None
        return result.stdout.strip() or None
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


def _find_git_root(start: Path) -> Path | None:
    cur = start.resolve()
    for _ in range(20):
        if (cur / ".git").exists():
            return cur
        if cur.parent == cur:
            break
        cur = cur.parent
    return None
