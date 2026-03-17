"""Shared display labels for model and prompting strategy ids."""

from __future__ import annotations

STRATEGY_LABELS: dict[str, str] = {
    "data_only": "Data-only",
    "third_person": "Narrative (3rd)",
    "first_person": "Narrative (1st)",
    "multi_expert": "Multi-expert",
    "paired_profile": "Paired contrast",
}


def label_strategy(strategy: str) -> str:
    """Return a human-readable strategy label."""

    raw = str(strategy)
    if raw in STRATEGY_LABELS:
        return STRATEGY_LABELS[raw]
    return raw.replace("_", " ").strip().title() or raw


def label_model(model: str) -> str:
    """Return a human-readable model label with effort tier."""

    raw = str(model).strip()
    if not raw:
        return "Unknown"

    if raw.startswith("openai_"):
        raw = raw[len("openai_") :]
    elif raw.startswith("openai-"):
        raw = raw[len("openai-") :]

    effort: str | None = None
    base = raw
    for suffix in ("_high", "_medium", "_low"):
        if base.endswith(suffix):
            effort = suffix.lstrip("_")
            base = base[: -len(suffix)]
            break

    effort = effort or "medium"
    if base == "gpt-oss-120b" and effort == "high":
        effort = "medium"

    label = base
    if base.startswith("gpt-oss-"):
        label = "GPT-OSS " + base.split("gpt-oss-", 1)[1].upper()
    elif base.startswith("qwen_"):
        rest = base.split("qwen_", 1)[1]
        label = rest.replace("-", " ").strip()
        label = label.replace("qwen", "Qwen")
        label = label[:-1] + "B" if label.lower().endswith("b") else label
    else:
        label = base.replace("_", " ")

    return f"{label} ({effort})"
