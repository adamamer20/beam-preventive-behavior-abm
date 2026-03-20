"""Repair empty/missing sample predictions using conversation logs.

This utility is intended for post-hoc recovery when the main `samples*.jsonl` output
contains empty `samples` (or missing scalar fields) but the conversation logger CSV
still contains `response_text`.
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import pandas as pd

from beam_abm.cli import parser_compat as cli
from beam_abm.evaluation.utils.regex_fallback import regex_extract_prediction
from beam_abm.llm.schemas.predictions import (
    extract_paired_scalar_predictions,
    extract_scalar_prediction_with_metadata,
)

LOGGER: Final[logging.Logger] = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ConversationBucket:
    preferred: tuple[str, ...]
    fallback: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PromptKey:
    record_id: str
    prompt_strategy: str
    prompt_hash: str


def _prompt_hash(human_prompt: str) -> str:
    return hashlib.sha1(human_prompt.encode("utf-8"), usedforsecurity=False).hexdigest()


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _needs_repair(row: dict[str, Any], *, strategies: set[str], force: bool) -> bool:
    strategy = str(row.get("strategy") or "")
    if strategy not in strategies:
        return False

    if force:
        return True

    samples = row.get("samples")
    if not isinstance(samples, list) or len(samples) == 0:
        return True

    if strategy == "paired_profile":
        for s in samples:
            if not isinstance(s, dict):
                continue
            if s.get("prediction_scalar_A") is not None or s.get("prediction_scalar_B") is not None:
                return False
        return True

    for s in samples:
        if not isinstance(s, dict):
            continue
        if s.get("prediction_scalar") is not None:
            return False
    return True


def _row_log_key(row: dict[str, Any]) -> PromptKey | None:
    record_id = row.get("id")
    prompt_strategy = row.get("strategy")
    human_prompt = row.get("human_prompt")
    if not record_id or not prompt_strategy or not isinstance(human_prompt, str) or not human_prompt.strip():
        return None
    return PromptKey(str(record_id), str(prompt_strategy), _prompt_hash(human_prompt))


def _build_conversation_index(
    *,
    conversations_csv: Path,
    needed_keys: set[PromptKey],
    prompt_by_pair: dict[tuple[str, str], dict[str, str]],
    chunksize: int,
    prefer_validation_success: bool,
) -> dict[PromptKey, ConversationBucket]:
    if not needed_keys:
        return {}

    needed_pairs = {(k.record_id, k.prompt_strategy) for k in needed_keys}

    usecols = ["record_id", "prompt_strategy", "prompt_text", "response_text", "validation_success"]
    buckets: dict[PromptKey, dict[str, list[str]]] = defaultdict(lambda: {"preferred": [], "fallback": []})

    for chunk in pd.read_csv(conversations_csv, usecols=usecols, chunksize=chunksize):
        chunk = chunk.dropna(subset=["record_id", "prompt_strategy"])
        chunk["record_id"] = chunk["record_id"].astype(str)
        chunk["prompt_strategy"] = chunk["prompt_strategy"].astype(str)

        # Filter down to only the needed (record_id, strategy) pairs first.
        pairs = list(zip(chunk["record_id"].tolist(), chunk["prompt_strategy"].tolist(), strict=False))
        mask = [p in needed_pairs for p in pairs]
        if not any(mask):
            continue

        filtered = chunk.loc[mask]
        for _, r in filtered.iterrows():
            pair = (str(r["record_id"]), str(r["prompt_strategy"]))
            prompt_text = r.get("prompt_text")
            if not isinstance(prompt_text, str) or not prompt_text:
                continue

            candidates = prompt_by_pair.get(pair)
            if not candidates:
                continue

            matched_key: PromptKey | None = None
            for ph, human_prompt in candidates.items():
                if human_prompt and human_prompt in prompt_text:
                    matched_key = PromptKey(pair[0], pair[1], ph)
                    break

            if matched_key is None:
                continue

            text = r.get("response_text")
            if not isinstance(text, str) or not text.strip():
                continue
            ok = bool(r.get("validation_success"))
            if prefer_validation_success and ok:
                buckets[matched_key]["preferred"].append(text)
            else:
                buckets[matched_key]["fallback"].append(text)

    out: dict[PromptKey, ConversationBucket] = {}
    for key, v in buckets.items():
        pref = tuple(v["preferred"])
        fallback = tuple(v["fallback"])
        out[key] = ConversationBucket(preferred=pref, fallback=fallback)
    return out


def _recover_samples_for_row(
    *,
    row: dict[str, Any],
    bucket: ConversationBucket,
    strategy: str,
    max_samples: int | None,
) -> list[dict[str, Any]]:
    texts: list[str] = list(bucket.preferred) + list(bucket.fallback)
    if max_samples is not None:
        texts = texts[:max_samples]

    recovered: list[dict[str, Any]] = []
    for t in texts:
        parsed = regex_extract_prediction(t, allow_scalar_fallback=True)
        # Normalize scalars into a dict-shaped schema.
        if isinstance(parsed, int | float):
            parsed = {"prediction": float(parsed)}

        sample: dict[str, Any] = {
            "raw_text": t,
            "raw_text_original": t,
            "raw_text_structurer": None,
            "parsed": parsed if isinstance(parsed, dict) else None,
            "error": None,
            "attempts": 1,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "recovered_from_conversations": True,
        }

        if strategy == "paired_profile":
            paired = extract_paired_scalar_predictions(sample.get("parsed"))
            if paired is None:
                continue
            a, b = paired
            sample["prediction_scalar_A"] = a
            sample["prediction_scalar_B"] = b
            if a is not None and b is not None:
                sample["prediction_scalar_delta"] = float(b) - float(a)
        else:
            scalar, meta = extract_scalar_prediction_with_metadata(sample.get("parsed"))
            sample["prediction_scalar"] = scalar
            if meta is not None:
                sample["prediction_scalar_meta"] = meta
            if scalar is None:
                continue

        recovered.append(sample)

    return recovered


def repair_samples_from_conversations_step(
    *,
    samples: str | Path,
    conversations: str | Path,
    out: str | Path,
    strategies: str = "data_only,paired_profile",
    chunksize: int = 200_000,
    max_samples_per_record: int | None = None,
    prefer_validation_success: bool = False,
    force: bool = False,
    report_out: str | Path | None = None,
) -> tuple[int, int]:
    _configure_logging()

    samples_path = Path(samples)
    conv_path = Path(conversations)
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    strategy_set = {s.strip() for s in str(strategies).split(",") if s.strip()}
    if not strategy_set:
        raise ValueError("No strategies provided")

    LOGGER.info("Scanning samples JSONL to find rows needing repair...")
    needed: set[PromptKey] = set()
    prompt_by_pair: dict[tuple[str, str], dict[str, str]] = defaultdict(dict)
    total_rows = 0
    repair_rows = 0
    for row in _iter_jsonl(samples_path):
        total_rows += 1
        if _needs_repair(row, strategies=strategy_set, force=bool(force)):
            key = _row_log_key(row)
            if key is not None:
                needed.add(key)
                pair = (key.record_id, key.prompt_strategy)
                human_prompt = row.get("human_prompt")
                if isinstance(human_prompt, str) and human_prompt.strip():
                    prompt_by_pair[pair][key.prompt_hash] = human_prompt
                repair_rows += 1
    LOGGER.info("Found %s/%s rows needing repair across strategies=%s", repair_rows, total_rows, sorted(strategy_set))

    LOGGER.info("Indexing conversation CSV for needed rows (streaming)...")
    conv_index = _build_conversation_index(
        conversations_csv=conv_path,
        needed_keys=needed,
        prompt_by_pair=dict(prompt_by_pair),
        chunksize=int(chunksize),
        prefer_validation_success=bool(prefer_validation_success),
    )
    LOGGER.info("Conversation index size: %s", len(conv_index))

    repaired_count = 0
    still_missing = 0
    report_rows: list[dict[str, Any]] = []

    LOGGER.info("Writing repaired JSONL...")
    with out_path.open("w", encoding="utf-8") as out_f:
        for row in _iter_jsonl(samples_path):
            if not _needs_repair(row, strategies=strategy_set, force=bool(force)):
                out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                continue

            key = _row_log_key(row)
            if key is None or key not in conv_index:
                still_missing += 1
                report_rows.append(
                    {
                        "id": row.get("id"),
                        "strategy": row.get("strategy"),
                        "status": "missing_conversation_rows",
                        "recovered_samples": 0,
                    }
                )
                out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                continue

            strategy = str(row.get("strategy"))
            recovered = _recover_samples_for_row(
                row=row,
                bucket=conv_index[key],
                strategy=strategy,
                max_samples=max_samples_per_record,
            )
            if recovered:
                row = dict(row)
                row["samples"] = recovered
                repaired_count += 1
                report_rows.append(
                    {
                        "id": row.get("id"),
                        "strategy": strategy,
                        "status": "repaired",
                        "recovered_samples": len(recovered),
                    }
                )
            else:
                still_missing += 1
                report_rows.append(
                    {
                        "id": row.get("id"),
                        "strategy": strategy,
                        "status": "conversation_rows_but_no_parse",
                        "recovered_samples": 0,
                    }
                )

            out_f.write(json.dumps(row, ensure_ascii=False) + "\n")

    LOGGER.info("Repaired rows: %s", repaired_count)
    LOGGER.info("Still missing after repair: %s", still_missing)

    if report_out:
        report_path = Path(report_out)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with report_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["id", "strategy", "status", "recovered_samples"])
            writer.writeheader()
            for r in report_rows:
                writer.writerow(r)

    return repaired_count, still_missing


def run_cli(argv: list[str] | None = None) -> None:
    parser = cli.ArgumentParser()
    parser.add_argument("--samples", required=True, help="Input samples JSONL (e.g. samples__direct.jsonl)")
    parser.add_argument("--conversations", required=True, help="Conversation CSV (conversations_*.csv)")
    parser.add_argument("--out", required=True, help="Output repaired JSONL")
    parser.add_argument(
        "--strategies",
        default="data_only,paired_profile",
        help="Comma-separated strategies to repair (default: data_only,paired_profile)",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=200_000,
        help="CSV chunksize for streaming reads",
    )
    parser.add_argument(
        "--max-samples-per-record",
        type=int,
        default=None,
        help="Optional cap on how many conversation rows to convert into samples per (record_id,strategy)",
    )
    parser.add_argument(
        "--prefer-validation-success",
        action="store_true",
        help="Prefer rows with validation_success=True when recovering samples",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild samples even when the JSONL already contains non-empty samples.",
    )
    parser.add_argument(
        "--report-out",
        default=None,
        help="Optional CSV report of what was repaired",
    )
    args = parser.parse_args(argv)
    repair_samples_from_conversations_step(
        samples=args.samples,
        conversations=args.conversations,
        out=args.out,
        strategies=args.strategies,
        chunksize=int(args.chunksize),
        max_samples_per_record=args.max_samples_per_record,
        prefer_validation_success=bool(args.prefer_validation_success),
        force=bool(args.force),
        report_out=args.report_out,
    )
