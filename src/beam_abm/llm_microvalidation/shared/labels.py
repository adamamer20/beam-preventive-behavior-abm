"""Load human-readable labels/descriptions for attribute (column) keys."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_ordinal_endpoints_from_clean_spec(path: Path) -> dict[str, tuple[str, str]]:
    """Infer ordinal endpoint labels (min/max) for columns from the clean spec.

    This is used to make prompts explicit about outcome semantics, e.g.
    ``1 = Strongly disagree, 7 = Strongly agree``.

    Resolution rules:
    - Follow ``renamed`` aliases to the source column.
    - Use the column's encoding spec (inline mapping or named encoding table).
    - Prefer the column's declared answer options when selecting labels so the
      endpoints match the actual survey wording.

    Parameters
    ----------
    path:
        Path to ``preprocess/specs/5_clean_transformed_questions.json``.

    Returns
    -------
    dict[str, tuple[str, str]]
        Mapping from column id -> (min_label, max_label).
        Columns without resolvable ordinal endpoints are omitted.
    """

    data: Any = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError(f"Expected object in clean spec JSON: {path}")

    encodings = data.get("encodings", {})
    if not isinstance(encodings, dict):
        encodings = {}

    all_cols: list[dict[str, Any]] = []
    for bucket in ("old_columns", "new_columns"):
        for item in data.get(bucket, []) or []:
            if isinstance(item, dict):
                all_cols.append(item)

    by_id: dict[str, dict[str, Any]] = {}
    for item in all_cols:
        cid = item.get("id")
        if isinstance(cid, str) and cid.strip():
            by_id[cid.strip()] = item

    memo_base: dict[str, str | None] = {}
    visiting: set[str] = set()

    def _base_id(col_id: str) -> str | None:
        cid = col_id.strip()
        if not cid:
            return None
        if cid in memo_base:
            return memo_base[cid]
        if cid in visiting:
            return None
        visiting.add(cid)
        item = by_id.get(cid)
        if item is None:
            visiting.remove(cid)
            memo_base[cid] = None
            return None

        transforms = item.get("transformations") or []
        if not isinstance(transforms, list):
            transforms = []

        for tr in transforms:
            if not isinstance(tr, dict) or tr.get("type") != "renamed":
                continue
            src = tr.get("rename_from")
            if isinstance(src, str) and src.strip():
                base = _base_id(src)
                visiting.remove(cid)
                memo_base[cid] = base
                return base

        visiting.remove(cid)
        memo_base[cid] = cid
        return cid

    def _encoding_spec_for(item: dict[str, Any]) -> Any:
        transforms = item.get("transformations") or []
        if not isinstance(transforms, list):
            return None
        for tr in transforms:
            if not isinstance(tr, dict) or tr.get("type") != "encoding":
                continue
            return tr.get("encoding")
        return None

    def _mapping_from_encoding_spec(enc: Any) -> dict[str, Any] | None:
        if enc is None:
            return None
        if isinstance(enc, dict):
            return enc
        if isinstance(enc, str):
            s = enc.strip()
            if s.startswith("$reverse_"):
                # Reversal implies numeric semantics but doesn't carry labels.
                return None
            m = encodings.get(s)
            return m if isinstance(m, dict) else None
        return None

    def _as_numeric(v: Any) -> float | None:
        if v is None:
            return None
        if isinstance(v, bool):
            return 1.0 if v else 0.0
        if isinstance(v, int | float) and v == v:  # v==v filters NaN
            return float(v)
        return None

    out: dict[str, tuple[str, str]] = {}
    for cid in by_id:
        base = _base_id(cid)
        if base is None:
            continue
        base_item = by_id.get(base)
        if base_item is None:
            continue

        enc_spec = _encoding_spec_for(base_item)
        mapping = _mapping_from_encoding_spec(enc_spec)
        if not mapping:
            continue

        # Prefer declared answer options for selecting endpoint labels.
        answers = base_item.get("answer")
        candidates: list[str] = []
        if isinstance(answers, list):
            candidates = [str(a).strip() for a in answers if isinstance(a, str) and a.strip()]
        elif isinstance(answers, str) and answers.strip():
            candidates = [answers.strip()]

        values: list[tuple[str, float]] = []
        if candidates:
            for a in candidates:
                v = mapping.get(a)
                if v is None and a.lower() != a:
                    v = mapping.get(a.lower())
                if v is None and a.upper() != a:
                    v = mapping.get(a.upper())
                num = _as_numeric(v)
                if num is None:
                    continue
                values.append((a, num))

        if values:
            mn = min(n for _, n in values)
            mx = max(n for _, n in values)
            min_label = next((a for a, n in values if n == mn), "")
            max_label = next((a for a, n in values if n == mx), "")
            if min_label and max_label:
                out[cid] = (min_label, max_label)
            continue

        # Fallback: use first seen labels in the mapping for min/max numeric codes.
        inv: dict[float, str] = {}
        for k, v in mapping.items():
            if not isinstance(k, str) or not k.strip():
                continue
            num = _as_numeric(v)
            if num is None:
                continue
            inv.setdefault(num, k.strip())
        if not inv:
            continue
        mn = min(inv)
        mx = max(inv)
        out[cid] = (inv[mn], inv[mx])

    return out


def _expand_group_labels(labels: dict[str, str], groups: dict[str, Any]) -> dict[str, str]:
    """Expand grouped labels into a flat label map."""

    for spec in groups.values():
        if not isinstance(spec, dict):
            continue
        title = spec.get("title")
        members = spec.get("members")
        if not isinstance(title, str) or not isinstance(members, dict):
            continue
        for key, member_label in members.items():
            if not isinstance(key, str) or not isinstance(member_label, str):
                continue
            if key not in labels:
                labels[key] = f"{title}: {member_label}"
    return labels


def load_display_labels(path: Path) -> dict[str, str]:
    """Load thesis-style display labels from a JSON spec.

    Expected shape: ``{"labels": {"col": "Label", ...}, ...}``.
    Optionally supports ``"groups"`` with ``{"title": ..., "members": ...}``.
    """

    data: Any = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError(f"Expected object in display labels JSON: {path}")

    labels_raw = data.get("labels", {})
    if not isinstance(labels_raw, dict):
        raise TypeError(f"Expected 'labels' object in display labels JSON: {path}")

    out: dict[str, str] = {}
    for k, v in labels_raw.items():
        if isinstance(k, str) and k.strip() and isinstance(v, str) and v.strip():
            out[k.strip()] = v.strip()

    groups = data.get("groups")
    if isinstance(groups, dict):
        out = _expand_group_labels(out, groups)

    return out


def load_display_groups(path: Path) -> dict[str, dict[str, Any]]:
    """Load grouped display labels from a JSON spec."""

    data: Any = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError(f"Expected object in display labels JSON: {path}")
    groups = data.get("groups", {})
    if not isinstance(groups, dict):
        return {}
    out: dict[str, dict[str, Any]] = {}
    for key, value in groups.items():
        if not isinstance(key, str) or not isinstance(value, dict):
            continue
        title = value.get("title")
        members = value.get("members")
        if not isinstance(title, str) or not title.strip():
            continue
        if not isinstance(members, dict):
            continue
        out[key.strip()] = {
            "title": title.strip(),
            "members": {
                k.strip(): v.strip()
                for k, v in members.items()
                if isinstance(k, str) and k.strip() and isinstance(v, str) and v.strip()
            },
        }
    return out


def load_clean_spec_question_answer_maps(path: Path) -> tuple[dict[str, str], dict[str, str]]:
    """Load (question_map, answer_map) from the clean preprocessing spec."""

    data: Any = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("new_columns"), list):
        raise ValueError(f"Unsupported clean spec JSON structure: {path}")

    question_map: dict[str, str] = {}
    answer_map: dict[str, str] = {}
    for item in data["new_columns"]:
        if not isinstance(item, dict):
            continue
        key = item.get("id")
        if not isinstance(key, str) or not key.strip():
            continue
        k = key.strip()

        q = item.get("question")
        if isinstance(q, str) and q.strip():
            question_map[k] = q.strip()

        a = item.get("answer")
        if isinstance(a, str) and a.strip():
            answer_map[k] = a.strip()

    return question_map, answer_map


def load_possible_numeric_bounds_from_clean_spec(path: Path) -> dict[str, tuple[float, float]]:
    """Infer "possible" numeric bounds for columns from the clean preprocessing spec.

    This uses the encoding/transformation metadata, e.g.:
    - ``"$reverse_1_to_7"`` -> (1, 7)
    - encoding tables like ``"opinion"`` / ``"moral"`` where values are numeric
    - inline encoding dicts mapping categories -> numeric codes
    """

    data: Any = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError(f"Expected object in clean spec JSON: {path}")

    encodings = data.get("encodings", {})
    if not isinstance(encodings, dict):
        encodings = {}

    all_cols: list[dict[str, Any]] = []
    for bucket in ("old_columns", "new_columns"):
        for item in data.get(bucket, []) or []:
            if isinstance(item, dict):
                all_cols.append(item)

    by_id: dict[str, dict[str, Any]] = {}
    for item in all_cols:
        cid = item.get("id")
        if isinstance(cid, str) and cid.strip():
            by_id[cid.strip()] = item

    def _numeric_minmax_from_mapping(mapping: Any) -> tuple[float, float] | None:
        if not isinstance(mapping, dict):
            return None
        vals: list[float] = []
        for v in mapping.values():
            if v is None:
                continue
            if isinstance(v, bool):
                vals.append(1.0 if v else 0.0)
                continue
            if isinstance(v, int | float) and v == v:  # v==v filters NaN
                vals.append(float(v))
        if not vals:
            return None
        return float(min(vals)), float(max(vals))

    def _bounds_from_encoding_spec(enc: Any) -> tuple[float, float] | None:
        if enc is None:
            return None
        if isinstance(enc, dict):
            return _numeric_minmax_from_mapping(enc)
        if isinstance(enc, str):
            s = enc.strip()
            if s.startswith("$reverse_") and "_to_" in s:
                # "$reverse_1_to_7" -> 1..7
                try:
                    rest = s.removeprefix("$reverse_")
                    lo_s, hi_s = rest.split("_to_", 1)
                    return float(lo_s), float(hi_s)
                except Exception:
                    return None
            # Reference a named encoding table.
            m = encodings.get(s)
            return _numeric_minmax_from_mapping(m)
        return None

    memo: dict[str, tuple[float, float] | None] = {}
    visiting: set[str] = set()

    def _combine(bounds: list[tuple[float, float]]) -> tuple[float, float] | None:
        if not bounds:
            return None
        lo = min(b[0] for b in bounds)
        hi = max(b[1] for b in bounds)
        return float(lo), float(hi)

    def _bounds_for_col(col_id: str) -> tuple[float, float] | None:
        cid = col_id.strip()
        if not cid:
            return None
        if cid in memo:
            return memo[cid]
        if cid in visiting:
            return None
        visiting.add(cid)

        item = by_id.get(cid)
        if item is None:
            visiting.remove(cid)
            memo[cid] = None
            return None

        transforms = item.get("transformations") or []
        if not isinstance(transforms, list):
            transforms = []

        # 1) Direct encoding on this column.
        for tr in transforms:
            if not isinstance(tr, dict) or tr.get("type") != "encoding":
                continue
            b = _bounds_from_encoding_spec(tr.get("encoding"))
            if b is not None:
                visiting.remove(cid)
                memo[cid] = b
                return b

        # 2) Renamed aliases inherit bounds.
        for tr in transforms:
            if not isinstance(tr, dict) or tr.get("type") != "renamed":
                continue
            src = tr.get("rename_from")
            if isinstance(src, str) and src.strip():
                b = _bounds_for_col(src)
                visiting.remove(cid)
                memo[cid] = b
                return b

        # 3) Mean merges preserve the underlying scale range.
        for tr in transforms:
            if not isinstance(tr, dict):
                continue
            if tr.get("type") != "merged":
                continue
            if str(tr.get("merging_type") or "").strip().lower() != "mean":
                continue
            srcs = tr.get("merged_from")
            if not isinstance(srcs, list) or not srcs:
                continue
            src_bounds: list[tuple[float, float]] = []
            for s in srcs:
                if isinstance(s, str) and s.strip():
                    b = _bounds_for_col(s)
                    if b is not None:
                        src_bounds.append(b)
            b = _combine(src_bounds)
            visiting.remove(cid)
            memo[cid] = b
            return b

        visiting.remove(cid)
        memo[cid] = None
        return None

    out: dict[str, tuple[float, float]] = {}
    for cid in by_id:
        b = _bounds_for_col(cid)
        if b is not None:
            out[cid] = b
    return out
