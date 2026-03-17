"""Analysis helpers for Quarto thesis chapters.

These utilities provide a thin access layer over the analysis outputs/specs so
chapters don't need to duplicate path constants or hard-code outcome lists.

All paths are defined relative to the Quarto working directory (the `thesis/`
folder), which is where chapters execute by default.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path

import polars as pl

COLUMN_TYPES_TSV = Path("../empirical/specs/column_types.tsv")
ANALYSIS_OUT = Path("../thesis/artifacts/empirical")
DESCRIPTIVES_RAW = ANALYSIS_OUT / "descriptives"
INFERENTIAL = ANALYSIS_OUT / "inferential"
MODELING = ANALYSIS_OUT / "modeling"

DISPLAY_SPECS_JSON = Path("specs/display_labels.json")


def load_column_types(path: str | Path = COLUMN_TYPES_TSV) -> pl.DataFrame:
    """Load the dataset column taxonomy used across analyses."""

    return pl.read_csv(
        Path(path),
        separator="\t",
        infer_schema_length=10_000,
        quote_char='"',
    )


def outcome_type_map(
    path: str | Path = COLUMN_TYPES_TSV,
    *,
    block: str = "outcome",
) -> dict[str, str]:
    """Return mapping from outcome column -> measurement type."""

    df = load_column_types(path).filter(pl.col("block") == block).select(pl.col("column"), pl.col("type"))
    return {str(row[0]): str(row[1]) for row in df.iter_rows()}


def load_outcomes(
    path: str | Path = COLUMN_TYPES_TSV,
    *,
    block: str = "outcome",
    types: Iterable[str] | None = None,
) -> list[str]:
    """Load outcome column names from the column taxonomy.

    Parameters
    ----------
    block:
        Column-types taxonomy block; outcomes are stored under ``"outcome"``.
    types:
        Optional measurement-type filter (e.g., ``{"binary", "ordinal"}``).
    """

    df = load_column_types(path).filter(pl.col("block") == block)
    if types is not None:
        df = df.filter(pl.col("type").is_in(pl.lit(list(types))))

    # Keep file order stable (it's a human-curated taxonomy).
    return [str(value) for value in df.get_column("column").to_list()]


def load_core_outcomes(
    specs_path: str | Path = DISPLAY_SPECS_JSON,
    *,
    key: str = "core_outcomes",
) -> list[str]:
    """Load the curated "core outcomes" list used throughout the thesis.

    Stored in ``thesis/specs/display_labels.json`` so the chapter code stays
    declarative and avoids repeating hard-coded lists.
    """

    specs = json.loads(Path(specs_path).read_text(encoding="utf-8"))
    raw = specs.get(key)
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise TypeError(f"Expected '{key}' to be a list in {specs_path!s}")

    return [str(item) for item in raw if isinstance(item, str | int | float)]


def load_outcome_group(
    group: str,
    *,
    specs_path: str | Path = DISPLAY_SPECS_JSON,
    container_key: str = "outcome_groups",
) -> list[str]:
    """Load a named outcome group from thesis display specs.

    This is used for chapter-specific subsets that cannot be inferred purely
    from measurement type.
    """

    specs = json.loads(Path(specs_path).read_text(encoding="utf-8"))
    container = specs.get(container_key, {})
    if not isinstance(container, Mapping):
        raise TypeError(f"Expected '{container_key}' to be an object in {specs_path!s}")

    raw = container.get(group)
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise TypeError(f"Expected '{container_key}.{group}' to be a list in {specs_path!s}")

    return [str(item) for item in raw if isinstance(item, str | int | float)]


def ensure_labels_enriched(
    labels: Mapping[str, str],
    *,
    required: Iterable[str],
) -> None:
    """Fail fast if a chapter relies on labels not present in shared specs."""

    missing = [name for name in required if name not in labels]
    if missing:
        joined = ", ".join(missing)
        raise KeyError(f"Missing labels for: {joined}")
