"""Central defaults for model-specific generation settings."""

from __future__ import annotations

_BASE_GENERATION_DEFAULTS: dict[str, float | int] = {
    "temperature": 0.2,
    "top_p": 0.9,
    "top_k": 50,
    "min_p": 0.0,
    "repetition_penalty": 1.0,
    "presence_penalty": 0.0,
    "frequency_penalty": 0.0,
}

_MODEL_OVERRIDES: list[tuple[str, dict[str, float | int]]] = [
    (
        "gpt-oss",
        {
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": 0,
        },
    ),
]


def get_generation_defaults(model_name: str) -> dict[str, float | int]:
    name = model_name.lower()
    defaults = dict(_BASE_GENERATION_DEFAULTS)
    for token, override in _MODEL_OVERRIDES:
        if token in name:
            defaults.update(override)
    return defaults
