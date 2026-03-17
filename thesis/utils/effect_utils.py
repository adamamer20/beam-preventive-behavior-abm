"""Effect-size formatting helpers for Quarto chapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def sig_stars(p_value: float) -> str:
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return ""


def fmt_sigfig(
    value: float,
    *,
    sig: int = 3,
    signed: bool = False,
) -> str:
    """Format a float with ~`sig` significant figures (avoid scientific notation)."""

    if value == 0:
        return "+0" if signed else "0"

    magnitude = abs(value)
    s = f"{magnitude:.{sig}g}"
    if "e" in s or "E" in s:
        s = f"{magnitude:.{sig + 6}f}".rstrip("0").rstrip(".")

    prefix = ""
    if value < 0:
        prefix = "-"
    elif signed:
        prefix = "+"
    return f"{prefix}{s}"


def fmt_effect_with_sig(
    effect: float,
    p_value: float,
    *,
    metric: str,
    suppress_trivial: bool = True,
) -> str:
    if metric in {"spearman", "cohen_d"}:
        base = fmt_sigfig(effect, sig=3, signed=True)
    else:
        base = fmt_sigfig(effect, sig=3, signed=False)

    stars = sig_stars(p_value)
    if suppress_trivial and stars == "***":
        stars = ""
    return f"{base}{stars}"


@dataclass(frozen=True, slots=True)
class EffectRow:
    term: str
    effect: float
    effect_label: str
    p_value: float
    n: int


def effect_table_rows(effect_rows: list[EffectRow], *, label_for: Any, fmt_int: Any, metric: str) -> list[list[str]]:
    """Render effect rows for a Markdown table.

    Parameters are passed in to keep this module Quarto-friendly (no heavy deps).
    """

    rows: list[list[str]] = []
    for row in effect_rows:
        rows.append(
            [
                label_for(row.term),
                fmt_effect_with_sig(row.effect, row.p_value, metric=metric),
                str(row.effect_label),
                fmt_int(row.n),
            ]
        )
    return rows
