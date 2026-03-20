"""Baseline lookup helpers shared by evaluation scripts and tests."""

from __future__ import annotations


def _lookup_baseline_value(
    *,
    join_value: str | None,
    outcome: str | None,
    outcomes: list[str] | None,
    by_id: dict[str, float | dict[str, float]],
    by_id_outcome: dict[tuple[str, str], float],
) -> float | dict[str, float]:
    def _lookup_from_by_id(key: str, requested_outcome: str | None) -> float | None:
        if key not in by_id:
            return None
        val = by_id[key]
        if isinstance(val, dict):
            if requested_outcome is None or requested_outcome not in val:
                msg = f"Baseline predictions missing outcome '{requested_outcome}' for join key {key}"
                raise ValueError(msg)
            return float(val[requested_outcome])
        return float(val)

    if join_value is None:
        raise ValueError("Baseline join key missing from input row.")
    if outcomes:
        out: dict[str, float] = {}
        for o in outcomes:
            out[o] = _lookup_baseline_value(
                join_value=join_value,
                outcome=o,
                outcomes=None,
                by_id=by_id,
                by_id_outcome=by_id_outcome,
            )
        return out

    if outcome is not None:
        prefixed_join = f"{outcome}__{join_value}"
        if (prefixed_join, outcome) in by_id_outcome:
            return by_id_outcome[(prefixed_join, outcome)]
    if outcome is not None and (join_value, outcome) in by_id_outcome:
        return by_id_outcome[(join_value, outcome)]
    if outcome is not None:
        prefixed_lookup = _lookup_from_by_id(f"{outcome}__{join_value}", outcome)
        if prefixed_lookup is not None:
            return prefixed_lookup
    direct_lookup = _lookup_from_by_id(join_value, outcome)
    if direct_lookup is not None:
        return direct_lookup
    raise ValueError(f"Baseline predictions missing for join key {join_value}")
