"""ID normalization helpers for microvalidation outputs."""

from __future__ import annotations

from typing import Final

__all__ = ["normalize_perturbed_dataset_id"]

_ID_DELIM: Final[str] = "__"
_ANCHOR_DELIM: Final[str] = "::"


def normalize_perturbed_dataset_id(value: object, *, outcome: str | None = None) -> str | None:
    """Normalize perturbed dataset ids to the anchor format: <anchor>::<country>::<row>.

    Some perturbed runners emit ids with an outcome prefix:
      <outcome>__<anchor>::<country>::<row>

    This helper strips the outcome prefix when it is present and the remainder
    looks like an anchor id. If the id does not match that pattern, it is
    returned unchanged (stringified + stripped).
    """

    if not isinstance(value, str):
        return None
    raw = value.strip()
    if not raw:
        return None
    if _ID_DELIM not in raw:
        return raw

    if outcome:
        prefix = f"{outcome}{_ID_DELIM}"
        if raw.startswith(prefix):
            remainder = raw[len(prefix) :].strip()
            if _ANCHOR_DELIM in remainder:
                return remainder or None

    head, tail = raw.split(_ID_DELIM, 1)
    if _ANCHOR_DELIM in tail:
        return tail.strip() or None
    return raw
