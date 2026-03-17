"""Utilities for Quarto/Markdown thesis chapters.

These helpers are intentionally lightweight so they can be imported from
multiple `.qmd` chapters without pulling in the full application code.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from IPython.display import Markdown

_MISSING = "–"


def _normalise_var_name(value: Any) -> str:
    """Collapse whitespace and trim so labels match normalised lookups."""

    if not isinstance(value, str):
        return ""
    return re.sub(r"\s+", "", value).strip()


def _is_missing(value: Any) -> bool:
    return value is None


def fmt_int(value: Any) -> str:
    """Format an integer with thousands separators.

    Returns an em dash for missing values to keep thesis rendering robust.
    """

    if _is_missing(value):
        return _MISSING
    return f"{int(value):,}"


def fmt_float(value: Any, digits: int = 2) -> str:
    """Format a float with a fixed number of decimals.

    Returns an em dash for missing values.
    """

    if _is_missing(value):
        return _MISSING
    try:
        out = float(value)
    except (TypeError, ValueError):
        return _MISSING
    if not math.isfinite(out):
        return _MISSING
    return f"{out:.{digits}f}"


def fmt_pct(pct: Any, digits: int = 1) -> str:
    """Format a percent value already expressed in percent units.

    Returns an em dash for missing values.
    """

    if _is_missing(pct):
        return _MISSING
    try:
        out = float(pct)
    except (TypeError, ValueError):
        return _MISSING
    if not math.isfinite(out):
        return _MISSING
    return f"{out:.{digits}f}%"


def fmt_p(value: Any) -> str:
    """Format p/q-values for compact narrative display.

    Most exported results use `0.0` for underflow; display that as `<0.001`.
    """

    if _is_missing(value):
        return _MISSING
    try:
        out = float(value)
    except (TypeError, ValueError):
        return _MISSING
    if not math.isfinite(out):
        return _MISSING
    if out == 0.0:
        return "<0.001"
    return fmt_float(out, 3)


def fmt_math(value: Any) -> Markdown:
    """Format a value as inline math for Quarto.

    Important: inline `{python}` results are safest when returned as an
    `IPython.display.Markdown` object (mirroring Chapter 05 patterns).
    Returning raw strings that contain TeX commands can lead to backslashes
    being escaped/stripped on some render paths.

    - Missing/NaN values render as an en dash.
    - If the incoming string already contains `$`, it is treated as already
      delimited math/TeX and returned as-is.
    """

    if _is_missing(value):
        return Markdown(_MISSING)

    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return Markdown(_MISSING)

    if isinstance(value, Markdown):
        return value

    text = str(value).strip()
    if not text or text.lower() == "nan" or text in {"—", "–", _MISSING}:
        return Markdown(_MISSING)

    if "$" in text:
        return Markdown(text)
    return Markdown(f"${text}$")


def _expand_group_labels(
    labels: dict[str, str],
    groups: Mapping[str, Any],
) -> dict[str, str]:
    """Expand grouped labels into a flat label map."""

    for spec in groups.values():
        if not isinstance(spec, Mapping):
            continue
        title = spec.get("title")
        members = spec.get("members")
        if not isinstance(title, str) or not isinstance(members, Mapping):
            continue
        for key, member_label in members.items():
            if not isinstance(key, str) or not isinstance(member_label, str):
                continue
            if key not in labels:
                labels[key] = f"{title}: {member_label}"
    return labels


def load_display_specs(path: str | Path = Path("specs/display_labels.json")) -> dict[str, Any]:
    """Load thesis display specs from JSON and normalise label keys."""

    path = Path(path)
    # Backward compatibility: short labels now live in display_labels.json.
    if not path.exists() and path.name == "display_labels_short.json":
        path = path.with_name("display_labels.json")
    specs = json.loads(path.read_text(encoding="utf-8"))

    def _normalise_label_map(raw: Any) -> dict[str, str]:
        label_map: dict[str, str] = {}
        if isinstance(raw, Mapping):
            for key, value in raw.items():
                if isinstance(key, str) and isinstance(value, str):
                    label_map[key] = value
        if not label_map:
            return {}

        enriched: dict[str, str] = {}
        for key, value in label_map.items():
            if not isinstance(key, str):
                continue
            enriched[key] = value
            normalised = _normalise_var_name(key)
            if normalised and normalised not in enriched:
                enriched[normalised] = value
        return enriched

    labels = specs.get("labels")
    label_map = _normalise_label_map(labels)

    groups = specs.get("groups")
    if isinstance(groups, Mapping):
        label_map = _expand_group_labels(label_map, groups)

    specs["labels"] = label_map

    short_labels = specs.get("short_labels")
    short_map = _normalise_label_map(short_labels)
    if short_map:
        specs["short_labels"] = short_map
    elif "short_labels" in specs:
        specs["short_labels"] = {}

    return specs


def load_labels(
    path: str | Path = Path("specs/display_labels.json"),
    *,
    variant: str = "long",
) -> dict[str, str]:
    """Load a label map from the thesis display specs JSON.

    Parameters
    ----------
    path
        Path to the display label specs JSON.
    variant
        Label variant to load: ``"long"`` (default) or ``"short"``.
    """

    specs = load_display_specs(path)
    key = "short_labels" if variant == "short" else "labels"
    labels = specs.get(key, {})
    if not isinstance(labels, Mapping):
        return {}
    return dict(labels)


def label_for(
    var_name: str,
    *,
    labels: Mapping[str, str] | None = None,
) -> str:
    """Best-effort human label for a variable name.

    Priority:
    1) `labels` (shared across chapters, e.g. from `specs/display_labels.json`)
    2) heuristic prettification for common prefix families
    """

    # Defensive normalisation: some upstream column headers contain
    # whitespace/newlines due to source text wrapping.
    # This function is display-only, so normalising is safe.
    # Important: some upstream column headers include embedded newlines
    # (e.g., CSV line-wrapping). Remove all whitespace so the key matches
    # the intended column id.
    var_key = _normalise_var_name(var_name)

    if labels is not None and var_key in labels:
        return labels[var_key]

    if var_key.startswith("covax_reason__"):
        raw = var_key.removeprefix("covax_reason__")
        raw = raw.replace("_", " ")
        raw = raw.replace(" s ", "'s ")
        raw = re.sub(r"\s+", " ", raw).strip()
        return f"COVID vaccination reason: {raw}"

    if var_key.startswith("info_trust_vax_"):
        platform = var_key.removeprefix("info_trust_vax_").replace("_", " ")
        return f"Trust in vaccine info on {platform}".strip()

    if var_key.startswith("time_"):
        platform = var_key.removeprefix("time_").replace("_", " ")
        return f"Time spent on {platform}".strip()

    return var_key.replace("__", ": ").replace("_", " ").strip().capitalize()
