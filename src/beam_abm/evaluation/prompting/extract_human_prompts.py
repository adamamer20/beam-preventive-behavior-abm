"""Extract human prompts from prompt JSONL files for debugging."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from beam_abm.evaluation.utils.jsonl import read_jsonl as _read_jsonl
from beam_abm.evaluation.utils.jsonl import write_jsonl as _write_jsonl


def _pick_human_prompt(row: dict[str, Any]) -> str | None:
    human = row.get("human_prompt")
    if isinstance(human, str) and human.strip():
        return human
    narrative_user = row.get("narrative_user_prompt")
    if isinstance(narrative_user, str) and narrative_user.strip():
        return narrative_user
    narrative_human = row.get("narrative_human_prompt")
    if isinstance(narrative_human, str) and narrative_human.strip():
        return narrative_human
    legacy = row.get("prompt")
    if isinstance(legacy, str) and legacy.strip():
        return legacy
    return None


def _pick_system_prompt(row: dict[str, Any]) -> str | None:
    sys_prompt = row.get("system_prompt")
    if isinstance(sys_prompt, str) and sys_prompt.strip():
        return sys_prompt
    predict_system = row.get("predict_system_prompt")
    if isinstance(predict_system, str) and predict_system.strip():
        return predict_system
    narrative_system = row.get("narrative_system_prompt")
    if isinstance(narrative_system, str) and narrative_system.strip():
        return narrative_system
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="in_path", required=True, help="Input prompt JSONL file")
    parser.add_argument("--out", dest="out_path", required=True, help="Output JSONL file")
    parser.add_argument(
        "--include-system",
        action="store_true",
        help="Include system prompt fields when present",
    )
    parser.add_argument(
        "--include-metadata",
        action="store_true",
        help="Include metadata field when present",
    )

    args = parser.parse_args()

    rows = _read_jsonl(Path(args.in_path))
    out_rows: list[dict[str, Any]] = []
    for row in rows:
        human_prompt = _pick_human_prompt(row)
        if human_prompt is None:
            continue
        item: dict[str, Any] = {
            "id": row.get("id"),
            "strategy": row.get("strategy"),
            "human_prompt": human_prompt,
        }
        if args.include_system:
            item["system_prompt"] = _pick_system_prompt(row)
        if args.include_metadata and isinstance(row.get("metadata"), dict):
            item["metadata"] = row.get("metadata")
        out_rows.append(item)

    _write_jsonl(Path(args.out_path), out_rows)
    print(f"Wrote {len(out_rows)} prompts to {args.out_path}")


if __name__ == "__main__":
    main()
