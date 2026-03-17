"""Generate a column-type specification for the clean survey dataset.

The script inspects the cleaned survey CSV, classifies each column into one of
four descriptive-analysis buckets (continuous, ordinal, binary, categorical),
and writes the result as a TSV. Lightweight heuristics cover the vast majority
of columns, and existing annotations can be respected via the output file
itself or an override TSV.

Key heuristics:
- Columns with "__" (multi-select expansions) are forced to ``categorical`` so
  the downstream tables treat the family of responses as a single nominal
  group rather than isolated binaries.
- Count-like fields remain ``continuous`` even when the observed support is a
  small integer range.
- Likert-style integer scales (e.g., 1–7 or -1/0/1) map to ``ordinal``.
- Strict yes/no style fields land in ``binary``.

Usage (Makefile integration recommended):
    uv run python preprocess/scripts/build_column_types.py \
        --input output/preprocess/clean_processed_survey.csv \
        --output empirical/specs/column_types.tsv
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

import polars as pl
import typer
from beartype import beartype

from beam_abm.common.logging import get_logger
from beam_abm.empirical.taxonomy import (
    DEFAULT_COLUMN_TYPES_PATH,
    OUTCOME_EXACT,
    OUTCOME_PREFIXES,
    SECTION_TO_BLOCK,
)

logger = get_logger(__name__)

DEFAULT_OUTPUT = DEFAULT_COLUMN_TYPES_PATH

_DEFAULT_CLEAN_CSV = Path(os.getenv("PREPROCESSING_OUTPUT_DIR", "preprocess/output")) / "clean_processed_survey.csv"

# Tokens that should be interpreted as missing to avoid polluting heuristics.
_NULL_TOKENS: tuple[str, ...] = (
    "",
    "NA",
    "N/A",
    "NULL",
    "Null",
    "null",
    "I prefer not to answer",
    "I don't know",
    "I do not know",
    "I don't know/I don't remember",
    "I do not know/I do not remember",
    "I'm not sure",
)

# Normalized lowercase value sets that should be treated as binary.
_BINARY_VALUE_SETS: tuple[set[str], ...] = (
    {"0", "1"},
    {"false", "true"},
    {"no", "yes"},
    {"female", "male"},
    {"f", "m"},
)

_COUNT_KEYWORDS: tuple[str, ...] = (
    "count",
    "num",
    "size",
    "total",
    "sum",
    "dose",
    "age",
)

_MULTISELECT_MARKERS: tuple[str, ...] = ("__",)

_INDEX_SUFFIXES: tuple[str, ...] = (
    "_idx",
    "_avg",
    "_timeavg",
    "_mean",
    "_index",
)


@dataclass(slots=True)
class ColumnTypeRecord:
    column: str
    type: str
    is_index: bool
    section: str
    block: str
    reference: str


@beartype
def _normalize_token(value: str) -> str:
    return value.strip().lower()


@beartype
def _is_binary_string(values: set[str]) -> bool:
    if len(values) != 2:
        return False
    return any(values <= allowed for allowed in _BINARY_VALUE_SETS)


@beartype
def _looks_like_count(column: str) -> bool:
    lowered = column.lower()
    return any(keyword in lowered for keyword in _COUNT_KEYWORDS)


@beartype
def _force_continuous(column: str) -> bool:
    lowered = column.lower()
    # Social media usage is recorded on a small numeric scale but is treated
    # as continuous for downstream summaries (mean/std).
    if lowered.startswith("time_"):
        return True
    return False


@beartype
def _is_multiselect_column(column: str) -> bool:
    lowered = column.lower()
    return any(marker in lowered for marker in _MULTISELECT_MARKERS)


@beartype
def _guess_is_index(column: str, known_indexes: set[str]) -> bool:
    if column in known_indexes:
        return True
    lowered = column.lower()
    return any(lowered.endswith(suffix) or suffix in lowered for suffix in _INDEX_SUFFIXES)


@beartype
def _classify_numeric(values: list[float], column: str) -> str:
    if not values:
        return "categorical"

    if _force_continuous(column):
        return "continuous"

    unique_vals = sorted(set(values))

    if len(unique_vals) <= 2 and set(unique_vals) <= {0.0, 1.0}:
        return "binary"

    if len(unique_vals) <= 7:
        all_int = all(float(v).is_integer() for v in unique_vals)
        min_v = min(unique_vals)
        max_v = max(unique_vals)

        if set(unique_vals) == {-1.0, 0.0, 1.0}:
            return "ordinal"

        if all_int and 1.0 <= min_v and max_v <= 7.0:
            return "ordinal"

        if all_int and min_v >= 0.0 and max_v <= 7.0 and not _looks_like_count(column):
            return "ordinal"

    return "continuous"


@beartype
def _classify_column(series: pl.Series, column: str) -> str:
    if _is_multiselect_column(column):
        # Treat exploded multi-select responses as binary indicators.
        return "binary"

    non_null = series.drop_nulls()
    if non_null.is_empty():
        return "categorical"

    numeric_cast = non_null.cast(pl.Float64, strict=False)
    numeric_non_null = numeric_cast.drop_nulls()

    # If at least 95% of non-null entries are numeric, apply numeric heuristics.
    if numeric_non_null.len() and (numeric_non_null.len() / non_null.len()) >= 0.95:
        values = numeric_non_null.to_list()
        return _classify_numeric(values, column)

    normalized = {_normalize_token(str(v)) for v in non_null.to_list()}

    if _is_binary_string(normalized):
        return "binary"

    # Occasionally 0/1 values arrive as strings; detect them explicitly.
    if normalized <= {"0", "1"}:
        return "binary"

    return "categorical"


@beartype
def _load_overrides(
    path: Path | None,
    *,
    require_type: bool = True,
) -> tuple[dict[str, str], dict[str, str], dict[str, str], dict[str, str]]:
    if path is None or not path.exists():
        return {}, {}, {}, {}

    df = pl.read_csv(path, separator="\t")
    if "column" not in df.columns:
        raise ValueError(f"Override file {path} must contain a 'column' column")
    if require_type and "type" not in df.columns:
        raise ValueError(f"Override file {path} must contain 'column' and 'type' columns")
    if "type" not in df.columns:
        # Nothing to apply yet.
        return {}, {}, {}, {}

    overrides: dict[str, str] = {}
    references: dict[str, str] = {}
    sections: dict[str, str] = {}
    blocks: dict[str, str] = {}
    for row in df.iter_rows(named=True):
        col = str(row["column"]).strip()
        value = str(row["type"]).strip()
        if col and value:
            overrides[col] = value
        if "reference" in row and col:
            ref = row.get("reference")
            if ref is not None and str(ref).strip() != "":
                references[col] = str(ref).strip()
        if "section" in row and col:
            sec = row.get("section")
            if sec is not None and str(sec).strip() != "":
                sections[col] = str(sec).strip()
        if "block" in row and col:
            blk = row.get("block")
            if blk is not None and str(blk).strip() != "":
                blocks[col] = str(blk).strip()
    return overrides, references, sections, blocks


_SECTION_DEFAULT_BLOCK: dict[str, str] = dict(SECTION_TO_BLOCK)


_REASON_PREFIXES: tuple[str, ...] = (
    "covax_reason__",
    "covax_no_child__",
    "covax_yesno_child__",
    "vax_no_child__",
)


_OUTCOME_EXACT: set[str] = set(OUTCOME_EXACT)


_OUTCOME_PREFIXES: tuple[str, ...] = OUTCOME_PREFIXES


_INFO_CHANNEL_TOKENS: tuple[str, ...] = (
    "tv",
    "television",
    "radio",
    "podcast",
    "telegram",
    "twitter",
    "x_twitter",
    "youtube",
    "tiktok",
    "facebook",
    "instagram",
    "reddit",
    "linkedin",
    "wikipedia",
    "online_news",
    "print",
    "internet_search",
    "internet_searches",
)


_SOCIAL_TOKENS: tuple[str, ...] = (
    "contact_group",
    "discussion_group",
    "vax_convo",
    "vax_freq",
    "closure",
    "size_friend_group",
    "size_colleagues_group",
)


_STRUCTURAL_TOKENS: tuple[str, ...] = (
    "booking",
    "vax_hub",
    "impediment",
)


_PSYCHOLOGICAL_TOKENS: tuple[str, ...] = (
    "fivec_",
    "mfq_",
    "att_",
    "progressive_",
    "institutional_trust",
    "vaccine_risk",
    "disease_fear",
    "vaccine_fear",
    "perceived_danger",
)


_EXPERIENCE_TOKENS: tuple[str, ...] = (
    "covid_date",
    "covax_date",
    "covid_num",
    "covax_num",
    "covid_treat",
    "vaccinated_",
    "infection_likelihood",
    "severity_if_infected",
    "side_effect",
)


@beartype
def _infer_reason_block(lowered_column: str) -> str | None:
    """Infer block for reason batteries based on content keywords."""
    if any(
        token in lowered_column
        for token in (
            "convenient_place_time",
            "did_not_have_the_time",
            "the_vaccination_was_not_offered",
            "the_vaccine_was_not_offered",
            "was_not_offered",
        )
    ):
        return "structural"

    if any(token in lowered_column for token in ("side_effect", "medical_condition")):
        return "experience"

    if any(
        token in lowered_column
        for token in (
            "lack_sufficient_information",
            "i_lack_sufficient_information",
            "i_was_not_aware",
            "health_care_professional",
        )
    ):
        return "information"

    if "trusted_person" in lowered_column:
        return "social"

    return None


@beartype
def _infer_block(column: str, section: str) -> str:
    lowered = column.lower()

    if lowered.endswith("_misflag"):
        return "qc"

    if lowered in _OUTCOME_EXACT or any(lowered.startswith(prefix) for prefix in _OUTCOME_PREFIXES):
        return "outcome"

    if lowered.startswith(("info_", "time_", "duration_")):
        return "information"
    if "trust_" in lowered and any(token in lowered for token in _INFO_CHANNEL_TOKENS):
        return "information"

    if any(token in lowered for token in _SOCIAL_TOKENS):
        return "social"
    if lowered.startswith(("family_", "friends_", "colleagues_")):
        return "social"

    if any(token in lowered for token in _STRUCTURAL_TOKENS):
        return "structural"

    if any(token in lowered for token in _PSYCHOLOGICAL_TOKENS):
        return "psychological"

    if any(token in lowered for token in _EXPERIENCE_TOKENS):
        return "experience"

    if any(lowered.startswith(prefix) for prefix in _REASON_PREFIXES):
        inferred = _infer_reason_block(lowered)
        if inferred is not None:
            return inferred

    return _SECTION_DEFAULT_BLOCK.get(section, "psychological")


@beartype
def _load_index_ids(spec_path: Path | None) -> set[str]:
    if spec_path is None or not spec_path.exists():
        return set()

    try:
        with spec_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:  # pragma: no cover - defensive
        logger.warning("Failed to parse spec for index detection: {e}", e=str(exc))
        return set()

    indexes: set[str] = set()
    new_columns = data.get("new_columns") or []
    for entry in new_columns:
        col_id = entry.get("id")
        if not col_id:
            continue
        transformations = entry.get("transformations") or []
        for trans in transformations:
            if trans.get("type") == "merged" and len(trans.get("merged_from") or []) >= 2:
                indexes.add(col_id)
                break
    return indexes


@beartype
def _load_sections(spec_path: Path | None) -> dict[str, str]:
    """Load a mapping from column id -> section from the preprocessing spec.

    Considers both old_columns and new_columns entries. If a column id is
    missing a section, it is skipped. This mapping is later used to annotate
    the column types TSV with a 'section' column. For multi-select expansions
    (columns containing '__'), we will later fall back to the base id before
    the first '__' when assigning sections.
    """
    if spec_path is None or not spec_path.exists():
        return {}

    try:
        with spec_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:  # pragma: no cover - defensive
        logger.warning("Failed to parse spec for section mapping: {e}", e=str(exc))
        return {}

    out: dict[str, str] = {}
    for key in ("old_columns", "new_columns"):
        for entry in data.get(key, []) or []:
            col_id = entry.get("id")
            sec = entry.get("section")
            if col_id and isinstance(sec, str) and sec.strip():
                out[col_id] = sec.strip()
    return out


@beartype
def _classify_columns(
    df: pl.DataFrame,
    overrides: dict[str, str],
    reference_overrides: dict[str, str],
    section_overrides: dict[str, str],
    block_overrides: dict[str, str],
    index_columns: set[str],
    section_map: dict[str, str],
) -> list[ColumnTypeRecord]:
    records: list[ColumnTypeRecord] = []

    for column in df.columns:
        series = df.get_column(column)
        inferred_type = _classify_column(series, column)
        col_type = overrides.get(column, inferred_type)

        # If an existing annotation conflicts with clear numeric evidence,
        # prefer the inferred type.
        non_null = series.drop_nulls()
        numeric_non_null = non_null.cast(pl.Float64, strict=False).drop_nulls()
        if numeric_non_null.len() and (numeric_non_null.len() / non_null.len()) >= 0.95:
            values = numeric_non_null.to_list()
            has_fractional = any(not float(v).is_integer() for v in values)
            if has_fractional and col_type == "ordinal":
                logger.info(
                    "Overriding ordinal -> continuous for '{col}' due to fractional numeric support\n",
                    col=column,
                )
                col_type = "continuous"
            if _force_continuous(column) and col_type != "continuous":
                logger.info(
                    "Overriding '{t}' -> continuous for '{col}' due to name rule\n",
                    t=col_type,
                    col=column,
                )
                col_type = "continuous"

        if col_type not in {"continuous", "ordinal", "binary", "categorical"}:
            raise ValueError(f"Unsupported type '{col_type}' for column '{column}'")

        # Resolve section, falling back to base id before '__' if needed
        section = section_map.get(column)
        section = section_overrides.get(column) or section_map.get(column)
        if section is None and column.endswith("_misflag"):
            base = column[: -len("_misflag")]
            section = section_overrides.get(base) or section_map.get(base)
        if section is None and "__" in column:
            base = column.split("__", 1)[0]
            section = section_overrides.get(base) or section_map.get(base)

        if section is None:
            raise ValueError(f"Missing section annotation for column '{column}'. Update the preprocessing spec.")

        block = block_overrides.get(column) or _infer_block(column, section)

        # Determine reference level (categoricals only, and skip multiselect expansions by default)
        reference = ""
        if column in reference_overrides:
            reference = reference_overrides[column]
        elif col_type == "categorical" and "__" not in column:
            non_null = df.get_column(column).drop_nulls()
            if not non_null.is_empty():
                counts = non_null.value_counts(sort=True)
                # value_counts returns two columns: column name and 'count'
                most_common = counts.row(0)[0]
                reference = str(most_common)

        records.append(
            ColumnTypeRecord(
                column=column,
                type=col_type,
                is_index=_guess_is_index(column, index_columns),
                section=section,
                block=block,
                reference=reference,
            )
        )

    return records


@beartype
def _write_tsv(records: Iterable[ColumnTypeRecord], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = pl.DataFrame([asdict(record) for record in records])
    df.write_csv(output_path, separator="\t")


def main(
    input_path: Path = typer.Option(
        _DEFAULT_CLEAN_CSV,
        "--input",
        help="Path to the cleaned survey CSV",
    ),
    output_path: Path = typer.Option(
        DEFAULT_OUTPUT,
        "--output",
        help="Destination TSV with inferred column types",
    ),
    override_path: Path | None = typer.Option(
        None,
        "--overrides",
        help="Optional TSV with manual overrides (column\ttype)",
    ),
    spec_path: Path | None = typer.Option(
        Path("preprocess/specs/5_clean_transformed_questions.json"),
        "--spec-path",
        help="Preprocessing spec JSON used to identify derived indexes",
    ),
    respect_existing: bool = typer.Option(
        True,
        "--respect-existing/--ignore-existing",
        help="When true, treat the output TSV (if present) as overrides",
    ),
    respect_existing_blocks: bool = typer.Option(
        False,
        "--respect-existing-blocks/--ignore-existing-blocks",
        help=(
            "When true, treat existing 'block' values in the output TSV as overrides. "
            "Defaults to false so block assignment stays deterministic and cannot drift."
        ),
    ),
) -> None:
    logger.remove()
    logger.add(lambda msg: print(msg, end=""))

    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    logger.info("Reading cleaned dataset: {p}\n", p=str(input_path))
    df = pl.read_csv(input_path, null_values=_NULL_TOKENS)

    overrides, reference_overrides, section_overrides, block_overrides = _load_overrides(override_path)

    if respect_existing and output_path.exists():
        logger.info("Applying existing annotations from: {p}\n", p=str(output_path))
        existing_types, existing_refs, existing_secs, existing_blocks = _load_overrides(output_path, require_type=False)
        overrides.update(existing_types)
        reference_overrides.update(existing_refs)
        section_overrides.update(existing_secs)
        if respect_existing_blocks:
            block_overrides.update(existing_blocks)

    index_columns = _load_index_ids(spec_path)
    section_map = _load_sections(spec_path)
    records = _classify_columns(
        df,
        overrides,
        reference_overrides,
        section_overrides,
        block_overrides,
        index_columns,
        section_map,
    )
    _write_tsv(records, output_path)

    logger.info(
        "Wrote {n} column type assignments to {p}\n",
        n=len(records),
        p=str(output_path),
    )


if __name__ == "__main__":
    typer.run(main)
