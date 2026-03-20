"""Canonicalization + metrics helpers for belief-update validation.

This mirrors the choice-validation philosophy:
- Store raw runs under `_runs/<run_id>/...`
- Build deterministic canonical leaves under `canonical/<model>/...`
- Materialize scalar aggregates under `summary_metrics/...`

Belief-update prompts reuse a non-unique `id` (base_id) across signals/variants.
For canonicalization we instead key rows by `prompt_id`.
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import sqlite3
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from beam_abm.llm_microvalidation.behavioural_outcomes._canonicalize_ids import normalize_model_slug


@dataclass(frozen=True, slots=True)
class CanonicalizeStats:
    runs_seen: int
    rows_in: int
    rows_out: int
    samples_in: int
    samples_out: int


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                yield obj


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return True
    if isinstance(value, str) and not value.strip():
        return True
    return False


def _fill_missing_metadata(dst: dict[str, Any], src: dict[str, Any]) -> None:
    src_meta = src.get("metadata")
    if not isinstance(src_meta, dict) or not src_meta:
        return
    dst_meta = dst.get("metadata")
    if not isinstance(dst_meta, dict):
        dst["metadata"] = dict(src_meta)
        return
    for k, v in src_meta.items():
        if _is_missing(dst_meta.get(k)) and not _is_missing(v):
            dst_meta[k] = v


def _strategy_key(row: dict[str, Any]) -> str:
    strategy = row.get("strategy") or "data_only"
    return str(strategy).strip() or "data_only"


def _row_key(row: dict[str, Any]) -> str:
    strategy_key = _strategy_key(row)
    prompt_id = row.get("prompt_id")
    if isinstance(prompt_id, str) and prompt_id.strip():
        return f"{prompt_id.strip()}::{strategy_key}"
    # Fallback for repaired conversation-derived rows: use stable metadata tuple.
    metadata = row.get("metadata")
    if isinstance(metadata, dict):
        signal_id = metadata.get("signal_id")
        base_id = metadata.get("base_id", row.get("id"))
        row_index = metadata.get("row_index")
        country = metadata.get("country")
        run_type = metadata.get("run_type")
        if (
            isinstance(signal_id, str)
            and signal_id.strip()
            and isinstance(base_id, str | int | float)
            and str(base_id).strip()
        ):
            payload_obj = {
                "strategy": strategy_key,
                "signal_id": str(signal_id).strip(),
                "base_id": str(base_id).strip(),
                "row_index": row_index,
                "country": country,
                "run_type": run_type,
            }
            payload = json.dumps(payload_obj, sort_keys=True, ensure_ascii=False)
            h = hashlib.sha256(payload.encode("utf-8")).hexdigest()
            return f"meta_{h}"
    # Fallback for very old outputs: hash the prompt text.
    prompt_text = row.get("_prompt_text") or row.get("prompt_text") or ""
    if isinstance(prompt_text, str) and prompt_text.strip():
        payload = f"{prompt_text}::{strategy_key}"
        h = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        return f"prompt_{h}"
    raise ValueError("Unable to infer row key for belief-update samples row")


def _sample_dedupe_id(*, row_key: str, sample: dict[str, Any], sample_idx: int) -> str:
    raw_text = sample.get("raw_text_original") or sample.get("raw_text") or ""
    scalar = sample.get("prediction_scalar")
    parsed = sample.get("parsed")
    payload = f"{row_key}::{sample_idx}::{raw_text}::{scalar}::{json.dumps(parsed, sort_keys=True, ensure_ascii=False)}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _canonicalize_samples_sqlite_to_path(*, samples_paths: list[Path], out_path: Path) -> CanonicalizeStats:
    """Canonicalize potentially huge sample sets without loading everything into RAM.

    Uses a temporary sqlite database to:
    - merge rows by prompt_id
    - dedupe samples per prompt via stable sample_id
    - preserve first-seen sample ordering via `ordinal`
    """

    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows_in = 0
    samples_in = 0
    runs_seen = 0
    ordinal = 0

    with tempfile.TemporaryDirectory(prefix="belief_update_canonicalize_") as tmp:
        db_path = Path(tmp) / "canon.sqlite"
        conn = sqlite3.connect(str(db_path))
        try:
            cur = conn.cursor()
            cur.execute("PRAGMA journal_mode=WAL;")
            cur.execute("PRAGMA synchronous=OFF;")
            cur.execute("PRAGMA temp_store=MEMORY;")
            cur.execute("PRAGMA cache_size=-200000;")  # ~200MB cache (negative = KB)

            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS rows (
                    prompt_id TEXT PRIMARY KEY,
                    row_json  TEXT NOT NULL
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS samples (
                    prompt_id TEXT NOT NULL,
                    sample_id TEXT NOT NULL,
                    ordinal   INTEGER NOT NULL,
                    sample_json TEXT NOT NULL,
                    PRIMARY KEY (prompt_id, sample_id)
                )
                """
            )
            cur.execute("CREATE INDEX IF NOT EXISTS idx_samples_prompt ON samples(prompt_id);")
            conn.commit()

            # Ingest.
            for path in samples_paths:
                runs_seen += 1
                conn.execute("BEGIN;")
                for row in iter_jsonl(path):
                    rows_in += 1
                    key = _row_key(row)

                    samples = row.get("samples")
                    if not isinstance(samples, list):
                        samples = []
                    samples_in += len(samples)

                    base_row = dict(row)
                    base_row["prompt_id"] = key
                    base_row.pop("samples", None)

                    # Keep the first-seen base row for each prompt_id; it's typically complete
                    # (and this avoids a per-row SELECT/UPDATE that slows large rebuilds).
                    cur.execute(
                        "INSERT OR IGNORE INTO rows(prompt_id, row_json) VALUES (?, ?)",
                        (key, json.dumps(base_row, ensure_ascii=False)),
                    )

                    # Batch insert samples for this row.
                    inserts: list[tuple[str, str, int, str]] = []
                    for idx, sample in enumerate(samples):
                        if not isinstance(sample, dict):
                            continue
                        sid = _sample_dedupe_id(row_key=key, sample=sample, sample_idx=int(idx))
                        inserts.append((key, sid, ordinal, json.dumps(sample, ensure_ascii=False)))
                        ordinal += 1
                    if inserts:
                        cur.executemany(
                            "INSERT OR IGNORE INTO samples(prompt_id, sample_id, ordinal, sample_json) VALUES (?, ?, ?, ?)",
                            inserts,
                        )
                conn.commit()

            # Emit canonical JSONL deterministically.
            # NOTE: use separate cursors for nested queries; reusing one cursor
            # will clobber the outer iterator and truncate output.
            rows_out = 0
            samples_out = 0
            outer = conn.cursor()
            inner = conn.cursor()
            with out_path.open("w", encoding="utf-8") as f:
                for prompt_id, row_json in outer.execute("SELECT prompt_id, row_json FROM rows ORDER BY prompt_id ASC"):
                    try:
                        base = json.loads(row_json)
                    except json.JSONDecodeError:
                        base = {}
                    if not isinstance(base, dict):
                        base = {}
                    base["prompt_id"] = prompt_id

                    row_samples: list[dict[str, Any]] = []
                    for (sample_json,) in inner.execute(
                        "SELECT sample_json FROM samples WHERE prompt_id=? ORDER BY ordinal ASC", (prompt_id,)
                    ):
                        try:
                            s = json.loads(sample_json)
                        except json.JSONDecodeError:
                            continue
                        if isinstance(s, dict):
                            row_samples.append(s)

                    base["samples"] = row_samples
                    rows_out += 1
                    samples_out += len(row_samples)
                    f.write(json.dumps(base, ensure_ascii=False) + "\n")

            return CanonicalizeStats(
                runs_seen=runs_seen,
                rows_in=rows_in,
                rows_out=rows_out,
                samples_in=samples_in,
                samples_out=samples_out,
            )
        finally:
            conn.close()


def iter_run_sample_paths(*, output_root: Path) -> dict[str, list[Path]]:
    """Return model_norm -> list[samples.jsonl paths] across all runs."""

    runs_root = output_root / "_runs"
    out: dict[str, list[Path]] = {}
    if not runs_root.exists():
        return out

    for run_dir in sorted(p for p in runs_root.iterdir() if p.is_dir() and not p.name.startswith("_")):
        batch_root = run_dir / "_batch"
        if not batch_root.exists():
            continue
        for model_dir in sorted(p for p in batch_root.iterdir() if p.is_dir() and not p.name.startswith("_")):
            samples_path = model_dir / "samples.jsonl"
            if samples_path.exists():
                model_norm = normalize_model_slug(model_dir.name, reasoning_effort=None)
                out.setdefault(model_norm, []).append(samples_path)
    return out


def rebuild_canonical_for_model(*, output_root: Path, model_norm: str, clear: bool = False) -> CanonicalizeStats | None:
    canonical_root = output_root / "canonical" / model_norm
    if clear and canonical_root.exists():
        shutil.rmtree(canonical_root)

    model_to_paths = iter_run_sample_paths(output_root=output_root)
    samples_paths = model_to_paths.get(model_norm, [])
    if not samples_paths:
        return None

    canonical_root.mkdir(parents=True, exist_ok=True)
    stats = _canonicalize_samples_sqlite_to_path(samples_paths=samples_paths, out_path=canonical_root / "samples.jsonl")

    (canonical_root / "canonicalize_stats.json").write_text(
        json.dumps(
            {
                "model": model_norm,
                "runs_seen": stats.runs_seen,
                "rows_in": stats.rows_in,
                "rows_out": stats.rows_out,
                "samples_in": stats.samples_in,
                "samples_out": stats.samples_out,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    return stats


def rebuild_canonical_from_runs(*, output_root: Path, clear: bool = False) -> dict[str, CanonicalizeStats]:
    stats_by_model: dict[str, CanonicalizeStats] = {}
    model_to_paths = iter_run_sample_paths(output_root=output_root)
    for model_norm in sorted(model_to_paths.keys()):
        stats = rebuild_canonical_for_model(output_root=output_root, model_norm=model_norm, clear=clear)
        if stats is not None:
            stats_by_model[model_norm] = stats
    return stats_by_model


def infer_legacy_model_backend(*, legacy_run_dir: Path) -> tuple[str | None, str | None]:
    """Infer model/backend from raw_conversations.jsonl (legacy Phase 0 outputs)."""

    rc = legacy_run_dir / "raw_conversations.jsonl"
    if not rc.exists():
        return None, None
    try:
        with rc.open("r", encoding="utf-8") as f:
            first = f.readline().strip()
        obj = json.loads(first) if first else {}
    except (OSError, json.JSONDecodeError):
        return None, None

    model = obj.get("model")
    backend = obj.get("backend")
    model_str = model.strip() if isinstance(model, str) and model.strip() else None
    backend_str = backend.strip() if isinstance(backend, str) and backend.strip() else None
    return model_str, backend_str


def migrate_legacy_prompt_tournament_outputs(*, output_root: Path, dry_run: bool = False) -> list[dict[str, Any]]:
    """Migrate legacy timestamp dirs into the canonical `_runs/<run_id>/_batch/<model>/...` layout.

    Legacy layout lives directly under `output_root/<timestamp>/...`.

    Returns a list of migration log records.
    """

    migrations: list[dict[str, Any]] = []
    runs_root = output_root / "_runs"
    runs_root.mkdir(parents=True, exist_ok=True)

    for child in sorted(p for p in output_root.iterdir() if p.is_dir() and not p.name.startswith("_")):
        run_id = child.name
        # Skip if this is already a canonical run folder we created.
        if (runs_root / run_id).exists():
            continue

        model, backend = infer_legacy_model_backend(legacy_run_dir=child)
        model_norm = normalize_model_slug(model or "unknown", reasoning_effort=None)

        dest_run = runs_root / run_id
        dest_batch = dest_run / "_batch" / model_norm
        legacy_dest = dest_run / "_legacy_original"

        record: dict[str, Any] = {
            "run_id": run_id,
            "legacy_path": str(child),
            "model": model,
            "model_norm": model_norm,
            "backend": backend,
            "dest_run": str(dest_run),
        }
        migrations.append(record)

        if dry_run:
            continue

        dest_run.mkdir(parents=True, exist_ok=True)
        dest_batch.mkdir(parents=True, exist_ok=True)

        # Move the entire legacy folder under _runs so we preserve any extra artifacts.
        if not legacy_dest.exists():
            shutil.move(str(child), str(legacy_dest))

        # Promote key artifacts into the new expected layout (move within same filesystem).
        def _move_if_exists(src: Path, dst: Path) -> None:
            if not src.exists():
                return
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dst.exists():
                return
            shutil.move(str(src), str(dst))

        _move_if_exists(legacy_dest / "prompts.jsonl", dest_run / "prompts.jsonl")
        _move_if_exists(legacy_dest / "belief_update_targets.json", dest_run / "belief_update_targets.json")
        _move_if_exists(legacy_dest / "run_config.json", dest_run / "run_config.json")

        _move_if_exists(legacy_dest / "samples.jsonl", dest_batch / "samples.jsonl")
        _move_if_exists(legacy_dest / "raw_conversations.jsonl", dest_batch / "raw_conversations.jsonl")
        _move_if_exists(legacy_dest / "row_level.csv", dest_batch / "row_level.csv")

        (dest_batch / "run_metadata.json").write_text(
            json.dumps(
                {
                    "run_id": run_id,
                    "model": model,
                    "model_norm": model_norm,
                    "backend": backend,
                    "migrated_from": str(legacy_dest),
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )

    return migrations


def write_summary_metrics(*, rows: list[dict[str, Any]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)


def clear_summary_metrics(*, output_root: Path) -> None:
    summary_root = output_root / "summary_metrics"
    if summary_root.exists():
        shutil.rmtree(summary_root)
