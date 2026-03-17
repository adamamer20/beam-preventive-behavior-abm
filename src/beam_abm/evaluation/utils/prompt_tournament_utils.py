"""Shared helpers for prompt-tournament phase runners."""

from __future__ import annotations

import shlex
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

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


def ensure_prompt_families(args: list[str], families: list[str]) -> list[str]:
    if "--prompt-family" in args or "--prompt-families" in args:
        return args
    return [*args, "--prompt-families", ",".join(families)]


def run_step(step: Callable[[], None], argv: list[str]) -> None:
    sys.argv = [sys.argv[0], *argv]
    step()


def run_step_subprocess(script: Path, argv: list[str], *, cwd: Path) -> None:
    cmd = [sys.executable, str(script), *argv]
    result = subprocess.run(cmd, cwd=str(cwd), check=False)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def coerce_outdir_args(args: list[str]) -> list[str]:
    if not args:
        return args
    if args[0].startswith("-"):
        return args
    return ["--outdir", args[0], *args[1:]]


def ensure_unperturbed_step(args: list[str]) -> list[str]:
    valid_steps = {"generate-prompts", "sample", "metrics", "calibrate", "apply-calibration", "all"}
    if not args:
        return args
    if args[0] in valid_steps:
        return args
    return ["all", *args]


def resolve_run_tag(run_tag: str | None, *, timestamp_output: bool) -> str | None:
    return resolve_run_id(run_tag, timestamp_output=timestamp_output)
