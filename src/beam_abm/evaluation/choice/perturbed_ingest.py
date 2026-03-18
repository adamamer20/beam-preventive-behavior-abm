"""Ingest and row-collection utilities for perturbed choice evaluation."""

from __future__ import annotations

import json
import shutil
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from beam_abm.evaluation.choice._canonicalize_ids import normalize_model_slug

CalibrationStatus = Literal["uncalibrated", "calibrated"]


@dataclass(frozen=True, slots=True)
class PromptTournamentLeaf:
    model_slug: str
    run_type: str
    strategy: str
    run_id: str
    outcome_batch: str
    status: CalibrationStatus
    base_dir: Path


def _iter_prompt_tournament_leaves(*, prompt_tournament_root: Path, run_type: str) -> Iterable[PromptTournamentLeaf]:
    if not prompt_tournament_root.exists():
        return

    # prompt_tournament/<model_slug>/<run_type>/<strategy>/<run_id>/<outcome_batch>/<status>/...
    for model_dir in sorted(p for p in prompt_tournament_root.iterdir() if p.is_dir() and not p.name.startswith(".")):
        for run_type_dir in sorted(p for p in model_dir.iterdir() if p.is_dir() and p.name == run_type):
            for strategy_dir in sorted(p for p in run_type_dir.iterdir() if p.is_dir() and not p.name.startswith("_")):
                for run_dir in sorted(p for p in strategy_dir.iterdir() if p.is_dir() and not p.name.startswith("_")):
                    for batch_dir in sorted(p for p in run_dir.iterdir() if p.is_dir() and not p.name.startswith("_")):
                        for status_dir in sorted(
                            p for p in batch_dir.iterdir() if p.is_dir() and p.name in {"uncalibrated", "calibrated"}
                        ):
                            yield PromptTournamentLeaf(
                                model_slug=model_dir.name,
                                run_type=run_type_dir.name,
                                strategy=strategy_dir.name,
                                run_id=run_dir.name,
                                outcome_batch=batch_dir.name,
                                status=status_dir.name,  # type: ignore[arg-type]
                                base_dir=status_dir,
                            )


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


def _iter_jsonl_dicts(path: Path) -> Iterable[dict[str, Any]]:
    """Stream JSONL rows (dict-only)."""
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


def _write_migration_log_line(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _safe_token(value: object) -> str:
    text = str(value) if value is not None else ""
    out = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in text).strip("_")
    return out or "unknown"


def normalize_model_dirname(value: str) -> str:
    """Normalize model directory tokens for on-disk naming.

    Rules:
    - Drop trailing `_medium` (medium is the default).
    - Preserve explicit `_low`/`_high`.

    Note:
    - Do NOT infer `_high` based solely on a provider/model prefix. Effort
        should only be encoded when it is explicitly present in the directory
        name (or in metadata when creating new outputs).
    """

    raw = str(value or "").strip()
    if not raw:
        return "unknown_model"
    out = raw
    if out.endswith("_medium"):
        out = out[: -len("_medium")]
    if out.endswith("_low") or out.endswith("_high"):
        return out
    return out


def rename_prompt_tournament_model_dirs(*, prompt_tournament_root: Path, migration_log: Path | None = None) -> None:
    """Rename model directories under output/prompt_tournament to normalized names.

    This is in-place (moves folders). It is idempotent.
    """

    if not prompt_tournament_root.exists():
        return

    def _try_read_run_metadata(model_dir: Path) -> dict[str, Any] | None:
        # Prefer perturbed metadata (most relevant to this migration), but accept any.
        candidates = sorted(model_dir.glob("**/run_metadata.json"))
        for path in candidates:
            try:
                obj = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if isinstance(obj, dict):
                return obj
        return None

    def _extract_reasoning_effort_from_run_meta(run_meta: dict[str, Any]) -> str | None:
        snap = run_meta.get("config_snapshot")
        if not isinstance(snap, dict):
            return None
        effort = snap.get("reasoning_effort")
        if isinstance(effort, str) and effort.strip():
            return effort.strip().lower()
        reasoning = snap.get("reasoning")
        if isinstance(reasoning, dict):
            e2 = reasoning.get("effort")
            if isinstance(e2, str) and e2.strip():
                return e2.strip().lower()
        model_options = snap.get("model_options")
        if isinstance(model_options, str) and model_options.strip():
            try:
                model_options = json.loads(model_options)
            except json.JSONDecodeError:
                model_options = None
        if isinstance(model_options, dict):
            e3 = model_options.get("reasoning_effort")
            if isinstance(e3, str) and e3.strip():
                return e3.strip().lower()
            r2 = model_options.get("reasoning")
            if isinstance(r2, dict):
                e4 = r2.get("effort")
                if isinstance(e4, str) and e4.strip():
                    return e4.strip().lower()
        return None

    def _infer_normalized_dirname(model_dir: Path) -> str:
        # If the directory encodes effort but metadata is incomplete, preserve
        # the effort by inferring it from the directory name.
        inferred_effort: str | None = None
        lowered = model_dir.name.lower()
        if lowered.endswith("_high"):
            inferred_effort = "high"
        elif lowered.endswith("_low"):
            inferred_effort = "low"
        elif lowered.endswith("_medium"):
            inferred_effort = "medium"

        # Project policy: GPT-OSS 120B is treated as medium effort.
        if "gpt-oss-120b" in lowered and inferred_effort == "high":
            inferred_effort = "medium"

        run_meta = _try_read_run_metadata(model_dir)
        if run_meta is None:
            # normalize_model_slug handles provider collapsing and suffix parsing.
            return normalize_model_dirname(normalize_model_slug(model_dir.name, reasoning_effort=inferred_effort))
        snap = run_meta.get("config_snapshot")
        model_name = None
        if isinstance(snap, dict):
            raw_model = snap.get("model")
            if isinstance(raw_model, str) and raw_model.strip():
                model_name = raw_model.strip()
        effort = _extract_reasoning_effort_from_run_meta(run_meta) or inferred_effort
        if effort == "high" and isinstance(model_name, str) and "gpt-oss-120b" in model_name.strip().lower():
            effort = "medium"
        slug = normalize_model_slug(model_name or model_dir.name, reasoning_effort=effort)
        return normalize_model_dirname(slug)

    for model_dir in sorted(p for p in prompt_tournament_root.iterdir() if p.is_dir()):
        if model_dir.name in {"frozen_prompts"}:
            continue
        new_name = _infer_normalized_dirname(model_dir)
        if new_name == model_dir.name:
            continue
        dst = model_dir.parent / new_name
        if dst.exists():
            if migration_log is not None:
                _write_migration_log_line(
                    migration_log,
                    {
                        "ts_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
                        "action": "rename_model_dir_skipped_exists",
                        "src": str(model_dir),
                        "dst": str(dst),
                    },
                )
            continue
        if migration_log is not None:
            _write_migration_log_line(
                migration_log,
                {
                    "ts_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "action": "rename_model_dir",
                    "src": str(model_dir),
                    "dst": str(dst),
                },
            )
        model_dir.rename(dst)


def _stream_split_jsonl_by_outcome(*, src_path: Path, dst_root_by_outcome: dict[str, Path], filename: str) -> None:
    """Split a JSONL file into per-outcome JSONL files without loading everything."""

    handles: dict[str, Any] = {}
    try:
        with src_path.open("r", encoding="utf-8") as f:
            for line in f:
                line_s = line.strip()
                if not line_s:
                    continue
                try:
                    obj = json.loads(line_s)
                except json.JSONDecodeError:
                    continue
                if not isinstance(obj, dict):
                    continue
                outcome = _row_outcome(obj) or (obj.get("outcome") if isinstance(obj.get("outcome"), str) else None)
                if not isinstance(outcome, str) or not outcome.strip():
                    continue
                outcome = outcome.strip()
                dst_root = dst_root_by_outcome.get(outcome)
                if dst_root is None:
                    continue
                out_path = dst_root / filename
                if outcome not in handles:
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    handles[outcome] = out_path.open("a", encoding="utf-8")
                handles[outcome].write(line_s + "\n")
    finally:
        for h in handles.values():
            try:
                h.close()
            except OSError:
                pass


def ingest_prompt_tournament_root_to_runs(
    *,
    prompt_tournament_root: Path,
    output_root: Path,
    run_type: str = "perturbed",
    dry_run: bool = False,
    migration_log: Path | None = None,
    ensure_timestamp_run_id: bool = True,
) -> dict[str, int]:
    """Ingest prompt-tournament outputs into the choice_validation/perturbed/_runs tree.

    This is a COPY-based ingest. It never deletes from the source.

    Returns counts: {"leaves_seen": int, "outcomes_written": int}.
    """

    counts = {"leaves_seen": 0, "outcomes_written": 0}
    runs_root = output_root / "_runs"
    run_id_map: dict[str, str] = {}

    def _resolve_samples_path(*, base_dir: Path, strategy: str, model_slug: str) -> Path | None:
        primary = base_dir / "samples.jsonl"
        if primary.exists():
            return primary

        # Prompt-tournament runners sometimes emit strategy-scoped names.
        strategy_token = _safe_token(strategy)
        candidates: list[Path] = [
            base_dir / f"samples__{strategy_token}.jsonl",
            base_dir / f"samples__{strategy_token}__{_safe_token(model_slug)}.jsonl",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate

        # Fallback: accept a single matching strategy-scoped file.
        matches = list(base_dir.glob(f"samples__{strategy_token}__*.jsonl"))
        if len(matches) == 1:
            return matches[0]
        return None

    def _has_timestamp(run_id: str) -> bool:
        # Match YYYY-MM-DDTHH-MM-SS inside the run_id string.
        for i in range(len(run_id) - 18):
            chunk = run_id[i : i + 19]
            if (
                chunk[4:5] == "-"
                and chunk[7:8] == "-"
                and chunk[10:11] == "T"
                and chunk[13:14] == "-"
                and chunk[16:17] == "-"
            ):
                y, m, d = chunk[:4], chunk[5:7], chunk[8:10]
                hh, mm, ss = chunk[11:13], chunk[14:16], chunk[17:19]
                if all(part.isdigit() for part in (y, m, d, hh, mm, ss)):
                    return True
        return False

    def _normalize_run_id(raw: str) -> str:
        if not ensure_timestamp_run_id:
            return raw
        cached = run_id_map.get(raw)
        if cached is not None:
            return cached
        if _has_timestamp(raw):
            run_id_map[raw] = raw
            return raw
        ts = datetime.now(UTC).strftime("%Y-%m-%dT%H-%M-%S")
        normalized = f"{raw}--{ts}"
        run_id_map[raw] = normalized
        return normalized

    for leaf in _iter_prompt_tournament_leaves(prompt_tournament_root=prompt_tournament_root, run_type=run_type):
        counts["leaves_seen"] += 1

        # Normalize through the shared slugger so provider-prefixed GPT-OSS
        # (e.g., openai_gpt-oss-20b) collapses to gpt-oss-20b.
        dest_model = normalize_model_dirname(normalize_model_slug(leaf.model_slug, reasoning_effort=None))

        src_prompts = leaf.base_dir / "prompts.jsonl"
        src_samples = _resolve_samples_path(
            base_dir=leaf.base_dir,
            strategy=leaf.strategy,
            model_slug=leaf.model_slug,
        )
        src_raw_conversations = leaf.base_dir / "raw_conversations.jsonl"
        src_row_level = leaf.base_dir / "row_level.csv"
        src_ice = leaf.base_dir / "ice_llm.csv"
        src_run_meta = leaf.base_dir / "run_metadata.json"

        if src_samples is None or not src_samples.exists():
            continue

        # First pass: determine which outcomes exist (streaming) so we can build dest dirs.
        outcomes_seen: set[str] = set()
        for r in _iter_jsonl_dicts(src_samples):
            o = _row_outcome(r)
            if o:
                outcomes_seen.add(o)
        if not outcomes_seen:
            continue

        run_id = _normalize_run_id(leaf.run_id)
        dest_by_outcome: dict[str, Path] = {}
        for outcome in sorted(outcomes_seen):
            dest_leaf = runs_root / run_id / _safe_token(outcome) / dest_model / leaf.strategy / leaf.status
            dest_by_outcome[outcome] = dest_leaf

        if migration_log is not None:
            for outcome, dest_leaf in dest_by_outcome.items():
                _write_migration_log_line(
                    migration_log,
                    {
                        "ts_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
                        "source": str(leaf.base_dir),
                        "dest": str(dest_leaf),
                        "run_id": run_id,
                        "outcome": outcome,
                        "model_src": leaf.model_slug,
                        "model_dest": dest_model,
                        "strategy": leaf.strategy,
                        "status": leaf.status,
                        "action": "ingest_prompt_tournament_leaf",
                    },
                )

        if dry_run:
            counts["outcomes_written"] += len(dest_by_outcome)
            continue

        # Skip already-ingested destinations to avoid duplicating JSONL rows.
        dest_by_outcome = {
            outcome: dest for outcome, dest in dest_by_outcome.items() if not (dest / "samples.jsonl").exists()
        }
        if not dest_by_outcome:
            continue

        for dest_leaf in dest_by_outcome.values():
            dest_leaf.mkdir(parents=True, exist_ok=True)

        _stream_split_jsonl_by_outcome(
            src_path=src_samples, dst_root_by_outcome=dest_by_outcome, filename="samples.jsonl"
        )
        if src_prompts.exists():
            _stream_split_jsonl_by_outcome(
                src_path=src_prompts, dst_root_by_outcome=dest_by_outcome, filename="prompts.jsonl"
            )
        if src_raw_conversations.exists():
            _stream_split_jsonl_by_outcome(
                src_path=src_raw_conversations,
                dst_root_by_outcome=dest_by_outcome,
                filename="raw_conversations.jsonl",
            )

        # Split CSV artifacts.
        if src_row_level.exists():
            # Chunked write to avoid huge memory.
            for chunk in pd.read_csv(src_row_level, chunksize=200_000):
                if "outcome" not in chunk.columns:
                    break
                for outcome, g in chunk.groupby(chunk["outcome"].astype(str), dropna=False):
                    key = str(outcome).strip()
                    if key not in dest_by_outcome:
                        continue
                    out_path = dest_by_outcome[key] / "row_level.csv"
                    g.to_csv(out_path, mode="a", header=not out_path.exists(), index=False)

        if src_ice.exists():
            ice_df = pd.read_csv(src_ice)
            if not ice_df.empty:
                if "outcome" not in ice_df.columns:
                    unknown_dest = dest_by_outcome.get("unknown_outcome")
                    if unknown_dest is not None:
                        ice_df.to_csv(unknown_dest / "ice_llm.csv", index=False)
                else:
                    for outcome, g in ice_df.groupby(ice_df["outcome"].astype(str), dropna=False):
                        key = str(outcome).strip()
                        if key not in dest_by_outcome:
                            continue
                        g.to_csv(dest_by_outcome[key] / "ice_llm.csv", index=False)

        if src_run_meta.exists():
            for dest_leaf in dest_by_outcome.values():
                shutil.copy2(src_run_meta, dest_leaf / "run_metadata.json")

        # Preserve optional conversations/ directory.
        src_conv_dir = leaf.base_dir / "conversations"
        if src_conv_dir.exists() and src_conv_dir.is_dir():
            for dest_leaf in dest_by_outcome.values():
                dst_conv_dir = dest_leaf / "conversations"
                if not dst_conv_dir.exists():
                    shutil.copytree(src_conv_dir, dst_conv_dir)

        counts["outcomes_written"] += len(dest_by_outcome)

    return counts


__all__ = [
    "PromptTournamentLeaf",
    "normalize_model_dirname",
    "rename_prompt_tournament_model_dirs",
    "ingest_prompt_tournament_root_to_runs",
]
