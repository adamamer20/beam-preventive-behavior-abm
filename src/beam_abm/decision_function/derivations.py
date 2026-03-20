"""Shared helpers for handling derived columns and collinearity.

These utilities centralise derivation metadata (used by inferential and
modeling code) so we can consistently skip redundant pairs or drop
deterministic composites that cause rank deficiency.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import polars as pl
from beartype import beartype


@beartype
def load_derivations(tsv_path: Path) -> pl.DataFrame:
    """Load derivations TSV with strict column checks."""
    if not tsv_path.exists():
        raise FileNotFoundError(f"Derivations TSV not found: {tsv_path}")
    df = pl.read_csv(tsv_path, separator="\t")
    expected = {"col", "derivation_col", "method"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"derivations.tsv missing required columns: {', '.join(sorted(missing))}")
    return df.select(
        pl.col("col").cast(pl.String),
        pl.col("derivation_col").cast(pl.String),
        pl.col("method").cast(pl.String),
    )


@dataclass(slots=True)
class DerivationSkipper:
    """Pair skipper used to avoid redundant comparisons."""

    skip_pairs: dict[frozenset[str], str]
    one_hot_groups: dict[str, tuple[str, ...]]

    def reason(self, a: str, b: str) -> str | None:
        key = frozenset((a, b))
        if key in self.skip_pairs:
            return self.skip_pairs[key]
        for prefix in self.one_hot_groups:
            if a.startswith(prefix) and b.startswith(prefix):
                return "one_hot_same_prefix"
        return None


@beartype
def build_skipper(derivations: pl.DataFrame, all_columns: tuple[str, ...]) -> DerivationSkipper:
    """Build a skipper that flags derived/base or one-hot sibling pairs."""
    colset = set(all_columns)
    pairs: dict[frozenset[str], str] = {}
    groups: dict[str, tuple[str, ...]] = {}

    for row in derivations.iter_rows(named=True):
        col = (row["col"] or "").strip()
        src = (row["derivation_col"] or "").strip()
        method = (row["method"] or "").strip().lower()
        if not src:
            continue
        if method.startswith("one_hot"):
            pref = src
            members = tuple(sorted(c for c in all_columns if c.startswith(pref)))
            if members:
                groups[pref] = members
            continue
        if not col:
            continue
        if col in colset and src in colset:
            reason = method or "derivation"
            pairs[frozenset((col, src))] = reason

    return DerivationSkipper(skip_pairs=pairs, one_hot_groups=groups)


@beartype
def drop_redundant_predictors(
    predictors: Iterable[str],
    derivations: pl.DataFrame | None,
    drop_methods: tuple[str, ...] = ("merged", "composite", "complement", "index"),
) -> list[str]:
    """Remove derived predictors when both base and derived are present.

    Drops `col` if a derivation row links it to a `derivation_col` in the
    predictor set and the method starts with one of `drop_methods`.
    Order is preserved for the kept predictors.
    """
    if derivations is None or derivations.height == 0:
        return list(predictors)

    drop_prefixes = tuple(m.lower() for m in drop_methods)
    deriv_records = [
        (
            (row["col"] or "").strip(),
            (row["derivation_col"] or "").strip(),
            (row["method"] or "").strip().lower(),
        )
        for row in derivations.iter_rows(named=True)
    ]
    pred_set = set(predictors)
    to_drop: set[str] = set()

    for col, src, method in deriv_records:
        if not col or not src:
            continue
        if col in pred_set and src in pred_set and method.startswith(drop_prefixes):
            to_drop.add(col)

    return [p for p in predictors if p not in to_drop]
