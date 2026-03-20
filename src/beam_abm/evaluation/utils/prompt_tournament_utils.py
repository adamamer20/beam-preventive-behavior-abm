"""Shared helpers for prompt-tournament phase runners."""

from __future__ import annotations

import shlex
from collections.abc import Callable

from beam_abm.evaluation.artifacts import resolve_run_id


def split_args(
    raw: str | None,
    *,
    family: str | None = None,
    model: str | None = None,
    model_slug: str | None = None,
    strategy: str | None = None,
) -> list[str]:
    if not raw:
        return []
    mapping: dict[str, str] = {}
    if family is not None:
        mapping["family"] = family
    if model is not None:
        mapping["model"] = model
    if model_slug is not None:
        mapping["model_slug"] = model_slug
    if strategy is not None:
        mapping["strategy"] = strategy
    text = raw
    for key, value in mapping.items():
        text = text.replace(f"{{{key}}}", str(value))
    return shlex.split(text)


def run_step(step: Callable[..., None], argv: list[str]) -> None:
    step(argv)


def resolve_run_tag(run_tag: str | None, *, timestamp_output: bool) -> str | None:
    return resolve_run_id(run_tag, timestamp_output=timestamp_output)
