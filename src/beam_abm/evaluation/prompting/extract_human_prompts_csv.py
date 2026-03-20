"""Extract human prompts from prompt JSONL files into a CSV."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from beam_abm.cli import parser_compat as cli
from beam_abm.evaluation.utils.jsonl import read_jsonl as _read_jsonl
from beam_abm.llm.prompts.prompts import BELIEF_UPDATE_EXPERT_PROMPTS, EXPERT_PROMPTS, format_attribute_vector


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


def _build_messages(row: dict[str, Any]) -> list[dict[str, str]] | None:
    strat = str(row.get("strategy") or "")
    narr_human = row.get("narrative_human_prompt")
    pred_sys = row.get("predict_system_prompt")
    narrative_system = row.get("narrative_system_prompt")
    narrative_assistant = row.get("narrative_assistant_prompt")
    narrative_user = row.get("narrative_user_prompt")

    if strat == "multi_expert":
        meta = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        is_belief_update = isinstance(meta, dict) and bool(meta.get("mutable_state"))
        expert_prompts = BELIEF_UPDATE_EXPERT_PROMPTS if is_belief_update else EXPERT_PROMPTS

        attr_block = None
        row_human_prompt = row.get("human_prompt")
        if isinstance(row_human_prompt, str) and row_human_prompt.strip():
            attr_block = row_human_prompt.strip()
        if attr_block is None:
            attrs = row.get("attributes") if isinstance(row.get("attributes"), dict) else {}
            attr_text = format_attribute_vector(attrs)
            attr_block = f"Attributes:\n{attr_text}" if attr_text else ""

        expert_messages: list[dict[str, Any]] = []
        for name, expert_prompt in expert_prompts.items():
            if "Output format:" in expert_prompt:
                expert_prompt = expert_prompt.split("Output format:", 1)[0].rstrip()
            system_prompt = (
                f"{expert_prompt}\n\n"
                "You are one of several experts. Provide ONLY the delta adjustment to the baseline prediction. "
                "Do NOT output full predictions."
            )
            expert_messages.append(
                {
                    "expert": name,
                    "messages": [
                        {"from": "system", "value": system_prompt},
                        {"from": "human", "value": attr_block},
                    ],
                }
            )
        return expert_messages

    if strat in {"first_person", "third_person"}:
        if isinstance(narrative_user, str) and narrative_user.strip():
            messages: list[dict[str, str]] = []
            sys_value = narrative_system if isinstance(narrative_system, str) and narrative_system.strip() else pred_sys
            if isinstance(sys_value, str) and sys_value.strip():
                messages.append({"from": "system", "value": sys_value})
            if isinstance(narrative_assistant, str) and narrative_assistant.strip():
                messages.append({"from": "assistant", "value": narrative_assistant})
            elif isinstance(narr_human, str) and narr_human.strip():
                messages.append({"from": "assistant", "value": narr_human})
            messages.append({"from": "human", "value": narrative_user})
            return messages

        if isinstance(narr_human, str) and narr_human.strip() and isinstance(pred_sys, str) and pred_sys.strip():
            pred_human = f"Narrative:\n{narr_human.strip()}\n\nReturn only JSON."
            return [
                {"from": "system", "value": pred_sys},
                {"from": "human", "value": pred_human},
            ]

    sys_prompt = _pick_system_prompt(row)
    human_prompt = _pick_human_prompt(row)
    if not isinstance(human_prompt, str) or not human_prompt.strip():
        return None
    messages = []
    if isinstance(sys_prompt, str) and sys_prompt.strip():
        messages.append({"from": "system", "value": sys_prompt})
    messages.append({"from": "human", "value": human_prompt})
    return messages


def _parse_list(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]


def extract_human_prompts_csv_step(
    *,
    in_path: str | Path,
    out_path: str | Path,
    include_system: bool = False,
    include_metadata: bool = False,
    include_messages: bool = False,
    metadata_keys: list[str] | None = None,
) -> int:
    selected_metadata_keys = metadata_keys or []
    rows = _read_jsonl(Path(in_path))
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
        if include_system:
            item["system_prompt"] = _pick_system_prompt(row)
        if include_messages:
            messages = _build_messages(row)
            item["messages"] = json.dumps(messages, ensure_ascii=False) if messages is not None else None
        metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else None
        if include_metadata:
            item["metadata"] = json.dumps(metadata, ensure_ascii=False) if metadata is not None else None
        if selected_metadata_keys:
            for key in selected_metadata_keys:
                item[key] = metadata.get(key) if isinstance(metadata, dict) else None
        out_rows.append(item)

    destination = Path(out_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in out_rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)
    return len(out_rows)


def run_cli(argv: list[str] | None = None) -> None:
    parser = cli.ArgumentParser()
    parser.add_argument("--in", dest="in_path", required=True, help="Input prompt JSONL file")
    parser.add_argument("--out", dest="out_path", required=True, help="Output CSV file")
    parser.add_argument(
        "--include-system",
        action="store_true",
        help="Include system prompt column when present",
    )
    parser.add_argument(
        "--include-metadata",
        action="store_true",
        help="Include metadata column as JSON when present",
    )
    parser.add_argument(
        "--include-messages",
        action="store_true",
        help="Include full conversation messages as a JSON column",
    )
    parser.add_argument(
        "--metadata-keys",
        default=None,
        help="Comma-separated metadata keys to extract into separate columns",
    )

    args = parser.parse_args(argv)
    metadata_keys = _parse_list(args.metadata_keys)

    count = extract_human_prompts_csv_step(
        in_path=args.in_path,
        out_path=args.out_path,
        include_system=bool(args.include_system),
        include_metadata=bool(args.include_metadata),
        include_messages=bool(args.include_messages),
        metadata_keys=metadata_keys,
    )
    print(f"Wrote {count} prompts to {args.out_path}")
