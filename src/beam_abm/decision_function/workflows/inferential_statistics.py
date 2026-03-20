"""CLI for running exploratory inferential statistics over survey data."""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path

import polars as pl
from beartype import beartype

from beam_abm.common.logging import get_logger
from beam_abm.decision_function.derivations import DerivationSkipper, build_skipper, load_derivations
from beam_abm.decision_function.inferential import (
    HighlightCriteria,
    InferentialResult,
    compute_binary_numeric_tests,
    compute_categorical_associations,
    compute_inferential_highlights,
    compute_nominal_numeric_tests,
    compute_spearman_correlations,
    generate_by_block_view,
    generate_by_outcome_view,
    generate_by_section_view,
    generate_predictor_predictor_views,
    to_canonical_raw_format,
)
from beam_abm.decision_function.io import (
    build_type_buckets,
    ensure_outdir,
    load_column_types,
    parse_list_arg,
    slugify,
)
from beam_abm.decision_function.taxonomy import (
    BLOCKS,
    DEFAULT_COLUMN_TYPES_PATH,
    DEFAULT_DERIVATIONS_PATH,
)
from beam_abm.settings import CSV_FILE_CLEAN

logger = get_logger(__name__)

DEFAULT_OUTDIR = Path("empirical/output/inferential")
_EXCLUDED_COLUMN_SUFFIXES: tuple[str, ...] = ("_misflag",)


_OUTPUT_FILE_MAP: dict[str, str] = {
    "spearman": "spearman.csv",
    "categorical": "categorical_categorical.csv",
    "binary_numeric": "binary_numeric.csv",
    "nominal_numeric": "nominal_numeric.csv",
}
_FAST_TABLES: tuple[str, ...] = ("spearman", "binary_numeric")
_SLOW_TABLES: tuple[str, ...] = ("categorical", "nominal_numeric")


@dataclass(slots=True)
class InferentialAnalysisConfig:
    input_csv: Path
    column_types_path: Path
    outdir: Path
    categorical_overrides: tuple[str, ...]
    min_nominal_levels: int = 3
    max_workers: int | None = None
    highlight_criteria: HighlightCriteria | None = None
    derivations_tsv: Path | None = None
    max_missing_fraction: float | None = 0.9


def _clear_directory(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)


def _filename_with_suffix(filename: str, suffix: str) -> str:
    if not suffix:
        return filename
    stem, ext = os.path.splitext(filename)
    return f"{stem}{suffix}{ext}"


def _build_highlight_criteria(
    enabled: bool,
    *,
    alpha: float,
    min_effect: str,
    min_spearman_n: int,
    min_categorical_n: int,
    min_binary_group_n: int,
    min_nominal_group_n: int,
) -> HighlightCriteria | None:
    if not enabled:
        return None
    return HighlightCriteria(
        alpha=alpha,
        min_effect_label=min_effect,
        min_spearman_n=min_spearman_n,
        min_categorical_n=min_categorical_n,
        min_binary_group_n=min_binary_group_n,
        min_nominal_group_n=min_nominal_group_n,
    )


@beartype
def _load_hypothesis_definitions(path: Path) -> pl.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Hypothesis definition CSV not found: {path}")
    df = pl.read_csv(path)
    if "H_ID" not in df.columns:
        raise ValueError("Hypothesis definition file must include an H_ID column")
    select_exprs = [
        pl.col("H_ID").cast(pl.String).str.strip_chars().alias("H_ID"),
    ]
    if "Title" in df.columns:
        select_exprs.append(pl.col("Title").cast(pl.String, strict=False).str.strip_chars().alias("Title"))
    else:
        select_exprs.append(pl.lit(None, dtype=pl.String).alias("Title"))
    if "Description" in df.columns:
        select_exprs.append(pl.col("Description").cast(pl.String, strict=False).str.strip_chars().alias("Description"))
    else:
        select_exprs.append(pl.lit(None, dtype=pl.String).alias("Description"))
    return df.select(select_exprs).drop_nulls(subset=["H_ID"])


@beartype
def _load_hypothesis_pairs(path: Path) -> pl.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Hypothesis variable CSV not found: {path}")
    df = pl.read_csv(path)
    required = {"H_ID", "var_x", "var_y"}
    missing = required - set(df.columns)
    if missing:
        cols = ", ".join(sorted(missing))
        raise ValueError(f"Hypothesis variable spec missing required columns: {cols}")
    select_exprs = [
        pl.col("H_ID").cast(pl.String).str.strip_chars().alias("H_ID"),
        pl.col("var_x").cast(pl.String, strict=False).str.strip_chars().alias("var_x"),
        pl.col("var_y").cast(pl.String, strict=False).str.strip_chars().alias("var_y"),
    ]
    if "expected_direction" in df.columns:
        select_exprs.append(
            pl.col("expected_direction").cast(pl.String, strict=False).str.strip_chars().alias("expected_direction")
        )
    else:
        select_exprs.append(pl.lit(None, dtype=pl.String).alias("expected_direction"))
    return df.select(select_exprs).drop_nulls(subset=["H_ID", "var_x", "var_y"])


def _inferential_order(*, fast_first: bool) -> tuple[str, ...]:
    if fast_first:
        return _FAST_TABLES + _SLOW_TABLES
    return _SLOW_TABLES + _FAST_TABLES


def _empty_inferential_result() -> InferentialResult:
    empty = pl.DataFrame({})
    return InferentialResult(
        spearman=empty,
        categorical=empty,
        binary_numeric=empty,
        nominal_numeric=empty,
    )


def _identify_high_missing_columns(
    df: pl.DataFrame,
    columns: tuple[str, ...],
    *,
    max_missing_fraction: float | None,
) -> dict[str, float]:
    if max_missing_fraction is None or not columns or df.height == 0:
        return {}
    available = [col for col in columns if col in df.columns]
    if not available:
        return {}
    exprs = [pl.col(col).null_count().alias(col) for col in available]
    counts = df.select(exprs).row(0, named=True) if exprs else {}
    denom = float(df.height)
    dropped: dict[str, float] = {}
    for col in available:
        frac = float(counts.get(col, 0) or 0) / denom
        if frac > max_missing_fraction:
            dropped[col] = frac
    return dropped


@beartype
def _build_hypothesis_label_map(definitions: pl.DataFrame | None) -> dict[str, str]:
    if definitions is None or definitions.height == 0:
        return {}
    mapping: dict[str, str] = {}
    for row in definitions.iter_rows(named=True):
        hypothesis_id = (row.get("H_ID") or "").strip()
        if not hypothesis_id:
            continue
        title = (row.get("Title") or "").strip()
        label = f"{hypothesis_id} {title}" if title else hypothesis_id
        mapping[hypothesis_id] = label
    return mapping


@beartype
def _label_hypothesis_tables(
    tables: dict[str, dict[str, pl.DataFrame]],
    label_map: dict[str, str],
) -> dict[str, dict[str, pl.DataFrame]]:
    if not tables:
        return {}
    return {label_map.get(hypothesis_id, hypothesis_id): table for hypothesis_id, table in tables.items()}


@beartype
def _bucket_columns(
    df: pl.DataFrame,
    column_types: pl.DataFrame,
    overrides: tuple[str, ...],
) -> tuple[dict[str, list[str]], list[str], list[str], list[str]]:
    # Backward-compatible alias for older name used within this CLI.
    return build_type_buckets(df, column_types, overrides)


@beartype
def _filter_column_types(column_types: pl.DataFrame) -> pl.DataFrame:
    """Remove columns with disallowed suffixes before building buckets."""

    if not _EXCLUDED_COLUMN_SUFFIXES or column_types.height == 0:
        return column_types

    excluded_cols = [
        record["column"]
        for record in column_types.iter_rows(named=True)
        if record["column"] and any(record["column"].endswith(suffix) for suffix in _EXCLUDED_COLUMN_SUFFIXES)
    ]
    if not excluded_cols:
        return column_types

    filtered = column_types.filter(~pl.col("column").is_in(pl.lit(excluded_cols)))
    logger.info(
        "Excluding columns from inferential analysis: {cols}",
        cols=", ".join(sorted(set(excluded_cols))[:15]),
    )
    return filtered


@beartype
def _detect_nominal_columns(df: pl.DataFrame, columns: tuple[str, ...], min_levels: int) -> list[str]:
    if not columns:
        return []
    available = [col for col in columns if col in df.columns]
    if not available:
        return []
    exprs = [pl.col(col).drop_nulls().n_unique().alias(col) for col in available]
    counts = df.select(exprs)
    unique_map = counts.row(0, named=True) if counts.height else {}
    return sorted(col for col in available if int(unique_map.get(col, 0) or 0) >= min_levels)


@beartype
def run_inferential_analysis(cfg: InferentialAnalysisConfig) -> InferentialResult:
    logger.info("Loading dataset: {path}", path=str(cfg.input_csv))
    df = pl.read_csv(cfg.input_csv)

    column_types = load_column_types(cfg.column_types_path)
    column_types = _filter_column_types(column_types)
    buckets, missing, unknown, untyped = _bucket_columns(df, column_types, cfg.categorical_overrides)

    if missing:
        logger.info("Columns defined in spec but missing in data: {cols}", cols=", ".join(missing[:15]))
    if unknown:
        logger.warning("Columns with unknown types ignored: {cols}", cols=", ".join(unknown[:15]))
    if untyped:
        logger.info("Dataset columns without type annotation: {cols}", cols=", ".join(untyped[:15]))

    numeric_columns = tuple(sorted(set(buckets["continuous"] + buckets["ordinal"])))
    categorical_columns = tuple(sorted(set(buckets["categorical"] + buckets["binary"])))
    binary_columns = tuple(sorted(buckets["binary"]))

    high_missing = _identify_high_missing_columns(
        df,
        tuple(sorted(set(numeric_columns + categorical_columns))),
        max_missing_fraction=cfg.max_missing_fraction,
    )
    if high_missing:
        dropped_set = set(high_missing)
        numeric_columns = tuple(col for col in numeric_columns if col not in dropped_set)
        categorical_columns = tuple(col for col in categorical_columns if col not in dropped_set)
        binary_columns = tuple(col for col in binary_columns if col not in dropped_set)
        preview = ", ".join(f"{col} ({missing:.1%})" for col, missing in sorted(high_missing.items())[:15])
        logger.info(
            "Dropping columns with missingness above {threshold:.0%}: {cols}",
            threshold=cfg.max_missing_fraction,
            cols=preview,
        )

    nominal_columns = tuple(_detect_nominal_columns(df, categorical_columns, cfg.min_nominal_levels))

    workers_to_use = cfg.max_workers if cfg.max_workers is not None else (os.cpu_count() or 1) // 2 or 1
    logger.info(
        "Running inferential tests (numeric={n}, categorical={c}, binary={b}, nominal={nom}, workers={w})",
        n=len(numeric_columns),
        c=len(categorical_columns),
        b=len(binary_columns),
        nom=len(nominal_columns),
        w=workers_to_use,
    )

    # Optional derivations-based skipper
    skipper: DerivationSkipper | None = None
    if cfg.derivations_tsv is not None:
        try:
            deriv = load_derivations(cfg.derivations_tsv)
            skipper = build_skipper(deriv, tuple(df.columns))
        except (OSError, ValueError, KeyError, pl.exceptions.PolarsError) as exc:  # pragma: no cover - defensive
            logger.warning("Failed to load derivations; proceeding without skip: {e}", e=str(exc))
            skipper = None

    def _skip_spearman(a: str, b: str) -> bool:
        return bool(skipper and skipper.reason(a, b))

    def _skip_categorical(a: str, b: str) -> bool:
        return bool(skipper and skipper.reason(a, b))

    result = _empty_inferential_result()
    for table in _inferential_order(fast_first=True):
        if table == "spearman":
            logger.info("Running spearman correlations...")
            result.spearman = compute_spearman_correlations(
                df,
                numeric_columns,
                skip=_skip_spearman if skipper else None,
            )
        elif table == "binary_numeric":
            logger.info("Running binary vs numeric tests...")
            result.binary_numeric = compute_binary_numeric_tests(
                df,
                binary_columns,
                numeric_columns,
                max_workers=workers_to_use,
            )
        elif table == "categorical":
            logger.info("Running categorical associations...")
            result.categorical = compute_categorical_associations(
                df,
                categorical_columns,
                max_workers=workers_to_use,
                skip=_skip_categorical if skipper else None,
            )
        elif table == "nominal_numeric":
            logger.info("Running nominal vs numeric tests...")
            result.nominal_numeric = compute_nominal_numeric_tests(
                df,
                nominal_columns,
                numeric_columns,
                max_workers=workers_to_use,
            )
    return result


def _write_table_group(result: InferentialResult, outdir: Path, *, suffix: str = "") -> None:
    tables = {
        "spearman": result.spearman,
        "categorical": result.categorical,
        "binary_numeric": result.binary_numeric,
        "nominal_numeric": result.nominal_numeric,
    }

    for key, data in tables.items():
        filename = _filename_with_suffix(_OUTPUT_FILE_MAP[key], suffix)
        if data.height == 0:
            logger.info("Skipping empty output: {name}", name=filename)
            continue
        path = outdir / filename
        data.write_csv(path)
        logger.info("Wrote {name} -> {path}", name=filename, path=str(path))


def _write_sectioned_tables(sectioned: dict[str, dict[str, pl.DataFrame]], base: Path, *, suffix: str = "") -> None:
    for section, section_tables in sectioned.items():
        sec_dir = base / slugify(section)
        ensure_outdir(sec_dir)
        for table_name, df in section_tables.items():
            filename = _OUTPUT_FILE_MAP.get(table_name)
            if not filename or df.height == 0:
                continue
            target = _filename_with_suffix(filename, suffix)
            df.write_csv(sec_dir / target)


@beartype
def _enumerate_pairs(cols: tuple[str, ...]) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for i, a in enumerate(cols[:-1]):
        for j in range(i + 1, len(cols)):
            out.append((a, cols[j]))
    return out


def _write_outputs(
    result: InferentialResult,
    outdir: Path,
    column_types_path: Path,
    highlights: InferentialResult | None = None,
    excluded_pairs: list[dict[str, str]] | None = None,
) -> None:
    """Write inferential results in the new layered structure."""

    # Root directory is just a container for layered outputs.
    ensure_outdir(outdir)

    # Load column types with section and block metadata
    try:
        column_types = load_column_types(column_types_path)
    except (OSError, ValueError, pl.exceptions.PolarsError) as exc:
        logger.warning("Failed to load column types: {e}", e=str(exc))
        return

    if "section" not in column_types.columns or "block" not in column_types.columns:
        logger.warning("Column types missing 'section' or 'block' columns; cannot generate new outputs")
        return

    # ==========================================
    # 1. RAW LAYER - machine-readable, complete
    # ==========================================
    raw_dir = outdir / "raw"
    ensure_outdir(raw_dir)

    # Convert to canonical format (single schema), then split into 4 raw files.
    try:
        raw_canonical = to_canonical_raw_format(result, column_types)
    except (KeyError, ValueError, pl.exceptions.PolarsError) as exc:
        logger.warning("Failed to create canonical format: {e}", e=str(exc))

        raw_canonical = pl.DataFrame({})

    if raw_canonical.height == 0:
        logger.warning("No canonical data available; skipping views generation")
        return

    for family, filename in (
        ("spearman", "spearman.csv"),
        ("categorical_categorical", "categorical_categorical.csv"),
        ("binary_numeric", "binary_numeric.csv"),
        ("nominal_numeric", "nominal_numeric.csv"),
    ):
        subset = raw_canonical.filter(pl.col("test_family") == family)
        if subset.height == 0:
            continue
        subset.write_csv(raw_dir / filename)
        logger.info("Wrote raw/{name}", name=filename)

    # Canonical highlights are used for view selection (and the original
    # highlight tables are also written under views/highlights).
    highlights_canonical: pl.DataFrame | None = None
    if highlights is not None:
        try:
            tmp = to_canonical_raw_format(highlights, column_types)
            highlights_canonical = tmp if tmp.height > 0 else None
        except (KeyError, ValueError, pl.exceptions.PolarsError) as exc:
            logger.warning("Failed to create canonical highlights: {e}", e=str(exc))

    # ==========================================
    # 2. VIEWS LAYER - human-readable, curated
    # ==========================================
    views_dir = outdir / "views"
    ensure_outdir(views_dir)

    # Store highlight tables under views/ so the inferential root directory
    # stays limited to canonical flat outputs (used by thesis + interoperability).
    if highlights is not None:
        highlights_dir = views_dir / "highlights"
        ensure_outdir(highlights_dir)
        _write_table_group(highlights, highlights_dir)

    # 2a) By outcome views
    by_outcome_dir = views_dir / "by_outcome"
    ensure_outdir(by_outcome_dir)

    try:
        outcome_vars = (
            column_types.filter(pl.col("block") == "outcome")
            .select(pl.col("column").cast(pl.String))
            .unique()
            .to_series()
            .drop_nulls()
            .to_list()
        )
        outcome_vars = tuple(sorted({str(item) for item in outcome_vars if item}))
        outcome_views = generate_by_outcome_view(
            raw_canonical,
            outcome_vars,
            highlights=highlights_canonical,
            top_k=10,
        )
        for outcome, df in outcome_views.items():
            if df.height > 0:
                filename = f"{outcome}.csv" if outcome == "support_mandatory_vaccination" else f"{outcome}_top.csv"
                df.write_csv(by_outcome_dir / filename)
                logger.info(f"Wrote views/by_outcome/{filename}")
    except (KeyError, ValueError, pl.exceptions.PolarsError) as exc:
        logger.warning("Failed to generate by_outcome views: {e}", e=str(exc))

    # 2b) By block views
    by_block_dir = views_dir / "by_block"
    ensure_outdir(by_block_dir)

    for block in BLOCKS:
        try:
            block_df = generate_by_block_view(raw_canonical, block, top_k_per_outcome=10)
            if block_df.height > 0:
                block_df.write_csv(by_block_dir / f"{block}_vs_outcomes_top.csv")
                logger.info(f"Wrote views/by_block/{block}_vs_outcomes_top.csv")
        except (KeyError, ValueError, pl.exceptions.PolarsError) as exc:
            logger.warning("Failed to generate {block} block view: {exc}", block=block, exc=str(exc))

    # 2c) By survey section
    by_section_dir = views_dir / "by_section"
    ensure_outdir(by_section_dir)

    unique_sections = column_types.select("section").unique().to_series().to_list()
    for section in unique_sections:
        if not section or section in ("metadata",):
            continue
        try:
            section_df = generate_by_section_view(raw_canonical, section, top_k_per_outcome=10)
            if section_df.height > 0:
                section_slug = slugify(section)
                section_df.write_csv(by_section_dir / f"{section_slug}_vs_outcomes_top.csv")
                logger.info(f"Wrote views/by_section/{section_slug}_vs_outcomes_top.csv")
        except (KeyError, ValueError, pl.exceptions.PolarsError) as exc:
            logger.warning("Failed to generate section view for {section}: {exc}", section=section, exc=str(exc))

    # 2d) Predictor-predictor views
    predictor_dir = views_dir / "predictor_predictor"
    ensure_outdir(predictor_dir)

    try:
        pred_views = generate_predictor_predictor_views(raw_canonical, max_within_block=50, max_cross_block=50)

        if pred_views["within_block_redundancy"].height > 0:
            pred_views["within_block_redundancy"].write_csv(predictor_dir / "within_block_redundancy_top.csv")
            logger.info("Wrote views/predictor_predictor/within_block_redundancy_top.csv")

        if pred_views["cross_block_links"].height > 0:
            pred_views["cross_block_links"].write_csv(predictor_dir / "cross_block_links_top.csv")
            logger.info("Wrote views/predictor_predictor/cross_block_links_top.csv")
    except (KeyError, ValueError, pl.exceptions.PolarsError) as exc:
        logger.warning("Failed to generate predictor-predictor views: {e}", e=str(exc))

    # Write excluded pairs list
    if excluded_pairs:
        excl_df = pl.DataFrame(excluded_pairs)
        excl_path = raw_dir / "excluded_pairs.csv"
        excl_df.write_csv(excl_path)
        logger.info("Wrote raw/excluded_pairs.csv")

    # Spec: stop organizing outputs as by_hypothesis/…
    # Hypothesis metadata can remain an optional external lens.


def run_inferential_statistics_workflow(
    *,
    input_csv: Path = CSV_FILE_CLEAN,
    column_types_path: Path = DEFAULT_COLUMN_TYPES_PATH,
    outdir: Path = DEFAULT_OUTDIR,
    categorical_overrides: str = "",
    min_nominal_levels: int = 3,
    workers: int | None = None,
    max_missing_fraction: float | None = 0.9,
    highlights: bool = True,
    highlight_alpha: float = 0.05,
    highlight_min_effect: str = "medium",
    highlight_min_spearman_n: int = 50,
    highlight_min_categorical_n: int = 50,
    highlight_min_binary_group_n: int = 20,
    highlight_min_nominal_group_n: int = 15,
    derivations_tsv: Path | None = DEFAULT_DERIVATIONS_PATH if DEFAULT_DERIVATIONS_PATH.exists() else None,
    log_level: str | None = None,
) -> None:
    """Run inferential workflow and write layered outputs."""

    effective_log_level = log_level or os.getenv("LOG_LEVEL", "INFO")
    logger.remove()
    logger.add(lambda msg: print(msg, end=""), level=effective_log_level)

    cfg = InferentialAnalysisConfig(
        input_csv=input_csv,
        column_types_path=column_types_path,
        outdir=outdir,
        categorical_overrides=parse_list_arg(categorical_overrides, dedupe=True),
        min_nominal_levels=min_nominal_levels,
        max_workers=workers,
        highlight_criteria=_build_highlight_criteria(
            highlights,
            alpha=highlight_alpha,
            min_effect=highlight_min_effect,
            min_spearman_n=highlight_min_spearman_n,
            min_categorical_n=highlight_min_categorical_n,
            min_binary_group_n=highlight_min_binary_group_n,
            min_nominal_group_n=highlight_min_nominal_group_n,
        ),
        derivations_tsv=derivations_tsv,
        max_missing_fraction=max_missing_fraction,
    )

    excluded: list[dict[str, str]] = []
    if derivations_tsv is not None and input_csv.exists():
        try:
            small_df = pl.read_csv(input_csv, n_rows=1000)
            deriv = load_derivations(derivations_tsv)
            skipper = build_skipper(deriv, tuple(small_df.columns))
            column_types = load_column_types(column_types_path)
            buckets, _, _, _ = _bucket_columns(
                small_df, column_types, parse_list_arg(categorical_overrides, dedupe=True)
            )
            numeric_columns = tuple(sorted(set(buckets["continuous"] + buckets["ordinal"])))
            categorical_columns = tuple(sorted(set(buckets["categorical"] + buckets["binary"])))
            for a, b in _enumerate_pairs(numeric_columns):
                reason = skipper.reason(a, b)
                if reason:
                    excluded.append({"table": "spearman", "var_x": a, "var_y": b, "reason": reason})
            for a, b in _enumerate_pairs(categorical_columns):
                reason = skipper.reason(a, b)
                if reason:
                    excluded.append({"table": "categorical", "var_x": a, "var_y": b, "reason": reason})
        except (OSError, ValueError, pl.exceptions.PolarsError) as exc:  # pragma: no cover - best-effort
            logger.warning("Failed to enumerate excluded pairs: {e}", e=str(exc))

    result = run_inferential_analysis(cfg)
    highlight_tables = (
        compute_inferential_highlights(result, cfg.highlight_criteria) if cfg.highlight_criteria is not None else None
    )
    _write_outputs(
        result,
        outdir,
        cfg.column_types_path,
        highlights=highlight_tables,
        excluded_pairs=excluded,
    )
