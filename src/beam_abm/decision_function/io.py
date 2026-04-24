from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from pathlib import Path

import polars as pl
from beartype import beartype


@beartype
def ensure_outdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def slugify(text: str) -> str:
    """Convert a label to a safe folder name.

    Lowercase, replace non-alphanumeric with underscores, collapse repeats,
    strip leading/trailing underscores.
    """

    cleaned = re.sub(r"[^0-9a-zA-Z]+", "_", text)
    cleaned = re.sub(r"_+", "_", cleaned).strip("_").lower()
    return cleaned or "section"


@beartype
def parse_list_arg(raw: str | None, *, dedupe: bool = False) -> tuple[str, ...]:
    """Parse a comma-separated CLI arg into a tuple.

    - If `dedupe=False`, preserves order and duplicates.
    - If `dedupe=True`, deduplicates via a set (order is unspecified, matching
      existing set-style parsing in some scripts).
    """

    if not raw:
        return ()

    items = [item.strip() for item in raw.split(",") if item.strip()]
    if not items:
        return ()

    if dedupe:
        return tuple(set(items))

    return tuple(items)


@beartype
def load_column_types(path: Path, *, include_is_index: bool = False) -> pl.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Column type specification not found: {path}")

    df = pl.read_csv(path, separator="\t")
    required = {"column", "type"}
    missing = required - set(df.columns)
    if missing:
        cols = ", ".join(sorted(missing))
        raise ValueError(f"Column type spec at {path} missing required columns: {cols}")

    exprs: list[pl.Expr] = [
        pl.col("column").cast(pl.String),
        pl.col("type").cast(pl.String),
    ]

    if include_is_index:
        if "is_index" in df.columns:
            exprs.append(pl.col("is_index").cast(pl.Boolean, strict=False).fill_null(False))
        else:
            exprs.append(pl.lit(False).cast(pl.Boolean).alias("is_index"))

    if "section" in df.columns:
        exprs.append(pl.col("section").cast(pl.String, strict=False))
    else:
        exprs.append(pl.lit(None, dtype=pl.String).alias("section"))

    if "block" in df.columns:
        exprs.append(pl.col("block").cast(pl.String, strict=False))
    else:
        exprs.append(pl.lit(None, dtype=pl.String).alias("block"))

    return df.select(exprs)


@beartype
def build_type_buckets(
    df: pl.DataFrame,
    column_types: pl.DataFrame,
    overrides: Iterable[str],
) -> tuple[dict[str, list[str]], list[str], list[str], list[str]]:
    recognised: dict[str, list[str]] = {
        "continuous": [],
        "ordinal": [],
        "binary": [],
        "categorical": [],
    }

    overrides_set = {c.strip() for c in overrides if isinstance(c, str) and c.strip()}
    missing_in_data: list[str] = []
    unknown_types: list[str] = []
    available = set(df.columns)

    for record in column_types.iter_rows(named=True):
        col = record.get("column")
        if not col:
            continue
        col_type = (record.get("type") or "").strip().lower()
        if not col_type:
            unknown_types.append(f"{col}:<empty>")
            continue
        if col in overrides_set:
            col_type = "categorical"
        if col not in available:
            missing_in_data.append(col)
            continue
        if col_type not in recognised:
            unknown_types.append(f"{col}:{col_type}")
            continue
        recognised[col_type].append(col)

    for col in overrides_set:
        if col in available and col not in recognised["categorical"]:
            recognised["categorical"].append(col)

    for bucket in recognised.values():
        bucket.sort()

    typed_columns = {record.get("column") for record in column_types.iter_rows(named=True) if record.get("column")}
    untyped = sorted(col for col in df.columns if col not in typed_columns)

    return recognised, sorted(missing_in_data), sorted(unknown_types), untyped


@beartype
def write_grouped_views(
    tables: Sequence[tuple[str, pl.DataFrame]],
    mapping: pl.DataFrame,
    *,
    join_col: str,
    group_col: str,
    base_dir: Path,
) -> None:
    """Write copies of tables split into folders by `group_col`.

    `mapping` must contain columns [join_col, group_col].
    """

    if not tables or mapping.height == 0:
        return

    if join_col not in mapping.columns or group_col not in mapping.columns:
        raise ValueError(f"mapping must include columns: {join_col}, {group_col}")

    groups = mapping.select(pl.col(group_col).unique()).to_series(0).drop_nulls().to_list()
    groups = sorted(groups)
    if not groups:
        return

    for filename, data in tables:
        if data.height == 0 or join_col not in data.columns:
            continue

        joined = data.join(mapping, on=join_col, how="left")
        if group_col not in joined.columns:
            continue

        for group in groups:
            label = group or "Unspecified"
            subset = joined.filter(pl.col(group_col) == label).drop(group_col)
            if subset.height == 0:
                continue
            group_dir = base_dir / slugify(str(label))
            ensure_outdir(group_dir)
            subset.write_csv(group_dir / filename)


@beartype
def load_column_types_with_reference(path: Path) -> tuple[dict[str, str], dict[str, object]]:
    """Load column types as a dict, plus optional reference levels.

    Intended for modeling workflows that need reference categories.
    """

    if not path.exists():
        raise FileNotFoundError(f"Column type specification not found: {path}")

    df = pl.read_csv(path, separator="\t")
    required = {"column", "type"}
    missing = required - set(df.columns)
    if missing:
        cols = ", ".join(sorted(missing))
        raise ValueError(f"Column type spec at {path} missing required columns: {cols}")

    column_types: dict[str, str] = {}
    for row in df.iter_rows(named=True):
        col = row.get("column")
        if not col:
            continue
        column_types[col] = str(row.get("type") or "").strip()

    reference_levels: dict[str, object] = {}
    if "reference" in df.columns:
        for row in df.iter_rows(named=True):
            col = row.get("column")
            if not col:
                continue
            ref = row.get("reference")
            if ref is None or ref == "":
                continue
            if isinstance(ref, str):
                lowered = ref.lower()
                if lowered == "true":
                    ref = True
                elif lowered == "false":
                    ref = False
            reference_levels[col] = ref

    return column_types, reference_levels
