"""Clean-spec indexing helpers for prompt generation."""

from __future__ import annotations

from typing import Any


def build_clean_spec_index(clean_spec_data: dict[str, Any]) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    encodings = clean_spec_data.get("encodings", {})
    if not isinstance(encodings, dict):
        encodings = {}
    all_cols: list[dict[str, Any]] = []
    for bucket in ("old_columns", "new_columns"):
        for item in clean_spec_data.get(bucket, []) or []:
            if isinstance(item, dict):
                all_cols.append(item)
    by_id = {
        item["id"].strip(): item
        for item in all_cols
        if isinstance(item, dict) and isinstance(item.get("id"), str) and item.get("id").strip()
    }
    return encodings, by_id


__all__ = ["build_clean_spec_index"]
