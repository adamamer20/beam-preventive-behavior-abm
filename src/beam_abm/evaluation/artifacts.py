"""Output path helpers for prompt-tournament artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

RunType = Literal["perturbed", "unperturbed"]


@dataclass(slots=True)
class PromptTournamentOutputPaths:
    """Canonical output path builder for prompt tournament runs."""

    run_type: RunType
    strategy: str
    run_id: str
    outcomes: list[str] | None
    calibrated: bool = False
    model_slug: str | None = None
    root: Path = Path("evaluation/output/prompt_tournament")

    def outcome_batch_name(self) -> str:
        outcomes = [o for o in (self.outcomes or []) if isinstance(o, str) and o.strip()]
        if not outcomes:
            return "unknown_outcome"
        if len(outcomes) == 1:
            return _safe_token(outcomes[0])
        return _safe_token("_".join(outcomes))

    def base_dir(self) -> Path:
        status = "calibrated" if self.calibrated else "uncalibrated"
        model_slug = _safe_token(self.model_slug) if self.model_slug else "unknown_model"
        return self.root / model_slug / self.run_type / self.strategy / self.run_id / self.outcome_batch_name() / status

    def raw_conversations_path(self) -> Path:
        return self.base_dir() / "raw_conversations.jsonl"

    def run_metadata_path(self) -> Path:
        return self.base_dir() / "run_metadata.json"

    def row_level_path(self, *, ext: str = "csv") -> Path:
        return self.base_dir() / f"row_level.{ext}"

    def summary_metrics_path(self, *, ext: str = "csv") -> Path:
        return self.base_dir() / f"summary_metrics.{ext}"

    def prompts_path(self) -> Path:
        return self.base_dir() / "prompts.jsonl"

    def samples_path(self) -> Path:
        return self.base_dir() / "samples.jsonl"

    def ice_path(self) -> Path:
        return self.base_dir() / "ice_llm.csv"


def resolve_run_id(run_tag: str | None, *, timestamp_output: bool) -> str | None:
    if not timestamp_output:
        return run_tag
    timestamp = utc_timestamp()
    return f"{run_tag}--{timestamp}" if run_tag else timestamp


def utc_timestamp() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H-%M-%S")


def utc_timestamp_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _safe_token(value: object) -> str:
    text = str(value) if value is not None else ""
    out = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in text).strip("_")
    return out or "unknown"


def hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_prompt_id(prompt_text: str, *, template_version: str | None = None) -> str:
    payload = f"{template_version or 'v1'}::{prompt_text}"
    return f"prompt_{hash_text(payload)}"


def build_conversation_id(*, prompt_id: str, response_text: str, sample_idx: int, model: str | None) -> str:
    model_tag = model or "unknown"
    payload = f"{prompt_id}::{model_tag}::{sample_idx}::{response_text}"
    return f"conv_{hash_text(payload)}"


def infer_run_id_from_path(path: Path) -> str | None:
    parts = path.resolve().parts
    if "prompt_tournament" not in parts:
        return None
    idx = parts.index("prompt_tournament")
    if len(parts) <= idx + 4:
        return None
    return parts[idx + 4]


def build_run_metadata(
    *,
    run_id: str,
    run_type: RunType,
    strategy: str,
    outcomes: Iterable[str],
    start_time_utc: str,
    end_time_utc: str,
    runtime_seconds: float,
    config_snapshot: dict[str, object] | None = None,
    dataset_inputs: dict[str, object] | None = None,
    anchors_inputs: dict[str, object] | None = None,
    perturbation_inputs: dict[str, object] | None = None,
) -> dict[str, object]:
    outcome_list = [str(o) for o in outcomes if str(o).strip()]
    code_version = _git_commit_hash()
    payload: dict[str, object] = {
        "run_id": run_id,
        "run_type": run_type,
        "strategy": strategy,
        "outcomes": outcome_list,
        "single_outcome": outcome_list[0] if len(outcome_list) == 1 else None,
        "multi_outcome_batch_name": "_".join(outcome_list) if len(outcome_list) > 1 else None,
        "start_time_utc": start_time_utc,
        "end_time_utc": end_time_utc,
        "runtime_seconds": float(runtime_seconds),
        "code_version": code_version,
        "config_snapshot": config_snapshot or {},
        "dataset_inputs": dataset_inputs or {},
        "anchors_inputs": anchors_inputs or {},
        "perturbation_inputs": perturbation_inputs or {},
    }
    return payload


def write_run_metadata(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _git_commit_hash() -> str | None:
    try:
        root = Path(os.getenv("PROJECT_ROOT", Path.cwd())).resolve()
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(root),
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            return None
        return result.stdout.strip() or None
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


__all__ = [
    "PromptTournamentOutputPaths",
    "RunType",
    "build_conversation_id",
    "build_prompt_id",
    "build_run_metadata",
    "hash_text",
    "infer_run_id_from_path",
    "resolve_run_id",
    "utc_timestamp",
    "utc_timestamp_iso",
    "write_run_metadata",
]
