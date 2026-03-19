"""Derivations map builder based on the preprocessing spec.

This script inspects ``preprocess/specs/5_clean_transformed_questions.json``
and emits a tab-separated ``derivations.tsv`` file that records pairs of
columns with deterministic relationships (aliases, indices, aggregates,
reverse-coded values, etc.). The TSV schema is::

    col \t derivation_col \t type \t role \t method \t note

Downstream we use this metadata to skip trivial correlations (e.g., a composite
index vs. its members) during inferential analysis. Keeping the derivations in
lockstep with the spec ensures we do not have to read the actual cleaned CSV
just to identify these relationships.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import polars as pl
import typer
from beartype import beartype

from beam_abm.common.logging import get_logger
from beam_abm.preprocess.cleaning.models import MergedTransformation, RenamedTransformation, SplittedTransformation
from beam_abm.preprocess.cleaning.semantics import (
    iter_raw_reference_names,
    normalize_reference_name,
    parse_transformation_payload,
)

logger = get_logger(__name__)

DEFAULT_SPEC = Path("preprocess/specs/5_clean_transformed_questions.json")
DEFAULT_OUTPUT = Path("preprocess/specs/derivations.tsv")


@dataclass(slots=True)
class DerivationRow:
    col: str
    derivation_col: str
    method: str


@beartype
def _load_spec(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Preprocessing spec not found: {path}")
    data = json.loads(path.read_text())
    new_columns = data.get("new_columns")
    if not isinstance(new_columns, list):  # pragma: no cover - defensive
        raise ValueError("Spec JSON missing 'new_columns' list")
    return [item for item in new_columns if isinstance(item, dict)]


@beartype
def _final_name(item: dict) -> str:
    """Return the final column name after any renaming steps."""

    current = item.get("id", "")
    for transform in item.get("transformations", []) or []:
        if transform.get("type") == "renaming":
            rename_to = [name for name in transform.get("rename_to", []) if name]
            if rename_to:
                current = rename_to[-1]
    return current


@beartype
def _is_removed(item: dict) -> bool:
    for transform in item.get("transformations", []) or []:
        if transform.get("type") == "removal":
            return True
    return False


@beartype
def _build_name_maps(
    items: Iterable[dict],
) -> tuple[dict[str, str], dict[str, set[str]], set[str]]:
    """Return mappings for name resolution."""

    id_to_final: dict[str, str] = {}
    raw_to_alias: dict[str, set[str]] = {}
    final_names: set[str] = set()

    for item in items:
        if _is_removed(item):
            continue
        new_id = item.get("id")
        if not new_id:
            continue
        final_name = _final_name(item)
        id_to_final[new_id] = final_name
        final_names.add(final_name)
        for transform in item.get("transformations", []) or []:
            if transform.get("type") == "renamed":
                source = transform.get("rename_from")
                if not source:
                    continue
                raw_to_alias.setdefault(source, set()).add(final_name)

    return id_to_final, raw_to_alias, final_names


@beartype
def _resolve_targets(
    source: str,
    id_to_final: dict[str, str],
    raw_to_alias: dict[str, set[str]],
    *,
    map_raw: bool,
    known_columns: set[str],
) -> list[str]:
    """Resolve a referenced column name to final dataset columns."""

    if not source:
        return []
    candidates: list[str] = []
    if source in known_columns:
        candidates.append(source)
    if source in id_to_final:
        candidates.append(id_to_final[source])
    elif map_raw and source in raw_to_alias:
        candidates.extend(sorted(raw_to_alias[source]))
    elif not map_raw:
        candidates.append(source)

    if not candidates and not map_raw:
        candidates.append(source)

    unique_candidates = list(dict.fromkeys(candidates))
    return [name for name in unique_candidates if name in known_columns]


@beartype
def _collect_derivations(
    item: dict,
    id_to_final: dict[str, str],
    raw_to_alias: dict[str, set[str]],
    known_columns: set[str],
    *,
    include_splits: bool,
) -> list[DerivationRow]:
    final_name = id_to_final.get(item.get("id", ""))
    if not final_name:
        return []
    rows: list[DerivationRow] = []

    for transform in item.get("transformations", []) or []:
        if not isinstance(transform, dict):
            continue
        typed = parse_transformation_payload(transform)
        if typed is None:
            continue

        map_raw = isinstance(typed, MergedTransformation | SplittedTransformation)
        if isinstance(typed, SplittedTransformation) and not include_splits:
            continue
        if not isinstance(typed, RenamedTransformation | MergedTransformation | SplittedTransformation):
            continue

        if isinstance(typed, RenamedTransformation):
            method_label = "alias"
        elif isinstance(typed, MergedTransformation):
            method = (typed.merging_type or "").lower()
            method_label = f"merged:{method or 'unknown'}"
        else:
            method_label = "split"

        normalized_refs = [normalize_reference_name(raw) for raw in iter_raw_reference_names(typed)]
        for src in (src for src in normalized_refs if src):
            for resolved in _resolve_targets(
                src,
                id_to_final,
                raw_to_alias,
                map_raw=map_raw,
                known_columns=known_columns,
            ):
                if not resolved or resolved == final_name:
                    continue
                rows.append(DerivationRow(col=final_name, derivation_col=resolved, method=method_label))

    return rows


@beartype
def _deduplicate(rows: Iterable[DerivationRow]) -> list[DerivationRow]:
    seen: dict[tuple[str, str, str], DerivationRow] = {}
    for row in rows:
        key = (row.col, row.derivation_col, row.method)
        if key not in seen:
            seen[key] = row
    return list(seen.values())


@beartype
def _write_tsv(rows: list[DerivationRow], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        df = pl.DataFrame([{"col": "", "derivation_col": "", "method": ""}])
    else:
        from dataclasses import asdict

        df = pl.DataFrame([asdict(row) for row in rows])
    df.write_csv(out_path, separator="\t")


@beartype
def build_derivations(
    spec_path: Path,
    out_path: Path,
    *,
    include_splits: bool = False,
) -> list[DerivationRow]:
    logger.info("Loading preprocessing spec: {p}\n", p=str(spec_path))
    entries = _load_spec(spec_path)
    id_to_final, raw_to_alias, final_names = _build_name_maps(entries)

    all_rows: list[DerivationRow] = []
    for item in entries:
        if item.get("id") not in id_to_final:
            continue
        all_rows.extend(
            _collect_derivations(
                item,
                id_to_final=id_to_final,
                raw_to_alias=raw_to_alias,
                known_columns=final_names,
                include_splits=include_splits,
            )
        )

    deduped = _deduplicate(all_rows)
    logger.info("Identified {n} derivation relationships\n", n=len(deduped))
    _write_tsv(deduped, out_path)
    logger.info("Wrote derivations TSV -> {p}\n", p=str(out_path))
    return deduped


@beartype
def main(
    spec_path: Path = typer.Option(DEFAULT_SPEC, "--spec", help="Path to preprocessing spec JSON"),
    out_path: Path = typer.Option(DEFAULT_OUTPUT, "--out", help="Output derivations TSV"),
    include_splits: bool = typer.Option(
        False,
        "--include-splits/--exclude-splits",
        help="Whether to record split-derived dummy columns",
    ),
) -> None:
    logger.remove()
    logger.add(lambda msg: print(msg, end=""))
    build_derivations(spec_path=spec_path, out_path=out_path, include_splits=include_splits)


if __name__ == "__main__":
    typer.run(main)
