"""CLI to run the structural backbone models described in modeling.md.

Commands:
- fit: run structural backbone models and persist coefficients/summary.
- block-importance: LOBO/LOBI using statsmodels fits.
- run: convenience to execute both fit and block-importance in one call.
"""

from __future__ import annotations

# Configure BLAS and numeric library thread limits early to avoid
# oversubscription when libraries like NumPy/OpenBLAS/MKL/numexpr
# spawn threadpools on import. These are module-level on import
# intentionally so numeric libraries see the limits before init.
import json
import os

_DEFAULT_BLAS_THREADS = "2"
os.environ.setdefault("OMP_NUM_THREADS", _DEFAULT_BLAS_THREADS)
os.environ.setdefault("MKL_NUM_THREADS", _DEFAULT_BLAS_THREADS)
os.environ.setdefault("OPENBLAS_NUM_THREADS", _DEFAULT_BLAS_THREADS)
os.environ.setdefault("NUMEXPR_NUM_THREADS", _DEFAULT_BLAS_THREADS)

import sys  # noqa: E402
from collections.abc import Callable, Iterable, Sequence  # noqa: E402
from concurrent.futures import ThreadPoolExecutor  # noqa: E402
from dataclasses import replace  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import TypeVar  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import polars as pl  # noqa: E402
import statsmodels.api as sm  # noqa: E402
import typer  # noqa: E402
from beartype import beartype  # noqa: E402

from beam_abm.common.logging import get_logger  # noqa: E402
from beam_abm.empirical.derivations import drop_redundant_predictors, load_derivations  # noqa: E402  -- env set above
from beam_abm.empirical.io import load_column_types_with_reference  # noqa: E402  -- env set above
from beam_abm.empirical.missingness import (  # noqa: E402  -- env set above
    DEFAULT_MISSINGNESS_THRESHOLD,
    apply_missingness_plan_pd,
    build_missingness_plan_pd,
    drop_nonfinite_rows_pd,
    missingness_indicator_name,
)
from beam_abm.empirical.modeling import (  # noqa: E402  -- env set above
    ModelBlock,
    ModelResult,
    ModelSpec,
    _compute_pseudo_r2,
    _fit_linear,
    _fit_logit,
    _fit_ordinal,
    _prepare_formula,
    bic_consistent,
    run_model_suite,
    standardize_columns,
)
from beam_abm.empirical.taxonomy import (  # noqa: E402  -- env set above
    DEFAULT_COLUMN_TYPES_PATH,
    DEFAULT_DERIVATIONS_PATH,
)
from beam_abm.settings import CSV_FILE_CLEAN  # noqa: E402  -- env set above

logger = get_logger(__name__)

app = typer.Typer(help="Structural backbone modeling and block importance.")

DEFAULT_OUTDIR = Path("empirical/output/modeling")

T = TypeVar("T")
R = TypeVar("R")


def _append_missingness_block(specs: list[ModelSpec], missingness_cols: Sequence[str]) -> list[ModelSpec]:
    updated: list[ModelSpec] = []
    indicator_names = {missingness_indicator_name(col) for col in missingness_cols}
    for spec in specs:
        existing_names = {block.name for block in spec.blocks}
        if "missingness" in existing_names:
            updated.append(spec)
            continue
        predictors = _block_predictor_columns(spec.blocks)
        needed_indicators = sorted({name for name in indicator_names if name.removesuffix("_missing") in predictors})
        if not needed_indicators:
            updated.append(spec)
            continue
        missing_block = ModelBlock(
            "missingness",
            tuple(needed_indicators),
            sample_strategy="conditional",
        )
        updated.append(replace(spec, blocks=spec.blocks + (missing_block,)))
    return updated


def _default_max_workers() -> int:
    try:
        cpus = os.cpu_count() or 1
    except (OSError, TypeError, ValueError):  # pragma: no cover - defensive
        cpus = 1
    return max(1, cpus // 2)


def _normalize_max_workers(value: int | None) -> int:
    if value is None:
        return 1
    if value < 0:
        raise ValueError("max-workers must be >= 0")
    if value == 0:
        return _default_max_workers()
    return value


def _run_tasks(tasks: Sequence[T], worker: Callable[[T], R | None], max_workers: int) -> list[R]:
    if not tasks:
        return []
    if max_workers <= 1 or len(tasks) == 1:
        return [result for task in tasks if (result := worker(task)) is not None]
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        return [result for result in executor.map(worker, tasks) if result is not None]


@beartype
def _load_column_types(path: Path) -> tuple[dict[str, str], dict[str, object]]:
    return load_column_types_with_reference(path)


def _collect_required_columns(specs) -> set[str]:
    cols: set[str] = set()
    for spec in specs:
        cols.add(spec.outcome)
        cols.update(spec.predictor_list())
    return cols


def _block_predictor_columns(blocks: Iterable[ModelBlock]) -> set[str]:
    predictors: set[str] = set()
    for block in blocks:
        predictors.update(block.predictors)
    return predictors


def _complete_case(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    ordered_unique = list(dict.fromkeys(columns))
    required = [col for col in ordered_unique if col in df.columns]
    if not required:
        return df.copy()
    return df.dropna(subset=required)


def _core_block_names(spec: ModelSpec) -> tuple[str, ...]:
    return tuple(block.name for block in spec.blocks if block.sample_strategy == "core")


def _spec_identity(spec: ModelSpec) -> tuple[str, str, str]:
    """Key used to find comparable specifications for soaking source selection."""
    return (str(spec.group or ""), spec.outcome, spec.model_type)


def _core_block_count(spec: ModelSpec) -> int:
    return sum(1 for block in spec.blocks if block.sample_strategy == "core")


def _resolve_soaking_source_specs(
    selected_specs: Sequence[ModelSpec],
    all_specs: Sequence[ModelSpec],
) -> dict[str, ModelSpec]:
    """Map each selected spec to a richer candidate spec for soaking.

    Strategy:
    - restrict candidates to same (group, outcome, model_type)
    - prefer the spec with most core blocks, then total blocks
    - on ties, prefer non-PN_ITEMS variants (broader baseline summaries)
    """

    by_key: dict[tuple[str, str, str], list[ModelSpec]] = {}
    for spec in all_specs:
        by_key.setdefault(_spec_identity(spec), []).append(spec)

    mapping: dict[str, ModelSpec] = {}
    for spec in selected_specs:
        candidates = by_key.get(_spec_identity(spec), [])
        if not candidates:
            mapping[spec.name] = spec
            continue
        source = max(
            candidates,
            key=lambda cand: (
                _core_block_count(cand),
                len(cand.blocks),
                1 if not cand.name.endswith("_PN_ITEMS") else 0,
                -len(cand.name),
            ),
        )
        mapping[spec.name] = source
    return mapping


def _configure_logging(level: str) -> None:
    logger.remove()
    logger.add(sys.stderr, level=level.upper())


def _specs_to_records(specs: Iterable[ModelSpec]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for spec in specs:
        rows.append(
            {
                "model": spec.name,
                "group": spec.group,
                "description": spec.description,
                "outcome": spec.outcome,
                "model_type": spec.model_type,
                "blocks": [block.name for block in spec.blocks],
                "predictors": {block.name: list(block.predictors) for block in spec.blocks},
                "interactions": [list(inter) for inter in spec.interactions],
                "add_country": spec.add_country,
            }
        )
    return rows


def _term_predictors(term: str) -> set[str]:
    """Return predictor names referenced by a coefficient term (handles interactions/dummies)."""

    def _base(part: str) -> str:
        cleaned = part.strip()
        if cleaned.startswith("C("):
            inner = cleaned[2:]
            inner = inner.split(")", 1)[0]
            inner = inner.split(",", 1)[0]
            return inner.strip()
        if "[" in cleaned:
            cleaned = cleaned.split("[", 1)[0]
        return cleaned

    return {_base(piece) for piece in term.split(":") if piece}


def _filter_split_coefficients(
    coeffs: pd.DataFrame, spec: ModelSpec, specs_by_name: dict[str, ModelSpec]
) -> pd.DataFrame:
    """Keep only item-level peer norm coefficients when comparing split vs. index models."""
    if not spec.name.endswith("_PN_ITEMS"):
        return coeffs
    base_spec = specs_by_name.get(spec.name.removesuffix("_PN_ITEMS"))
    if base_spec is None:
        return coeffs
    novel_predictors = set(spec.predictor_list()) - set(base_spec.predictor_list())
    if not novel_predictors:
        return coeffs
    mask = coeffs["term"].apply(lambda term: bool(_term_predictors(str(term)) & novel_predictors))
    filtered = coeffs[mask].copy()
    return filtered if not filtered.empty else coeffs


def _model_output_dir(spec: ModelSpec, outdir: Path) -> Path:
    """Return the on-disk directory for a model, grouped when available."""
    if spec.group:
        target = outdir / spec.group / spec.name
    else:
        target = outdir / spec.name
    target.mkdir(parents=True, exist_ok=True)
    return target


def _persist_results(results: Iterable[ModelResult], specs: list[ModelSpec], outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    combined_rows: list[pd.DataFrame] = []
    summary_rows: list[dict[str, object]] = []
    specs_by_name = {spec.name: spec for spec in specs}
    for result in results:
        coeffs = result.coefficients.copy()
        coeffs = _filter_split_coefficients(coeffs, result.spec, specs_by_name)
        model_dir = _model_output_dir(result.spec, outdir)
        coeffs.to_csv(model_dir / "coefficients.csv", index=False)
        combined_rows.append(coeffs)
        summary_rows.append(
            {
                "model": result.spec.name,
                "group": result.spec.group,
                "outcome": result.spec.outcome,
                "type": result.spec.model_type,
                "nobs": result.nobs,
                "pseudo_r2": result.pseudo_r2,
                "formula": result.formula,
                "description": result.spec.description,
            }
        )
    if combined_rows:
        all_coeffs = pd.concat(combined_rows, ignore_index=True)
        all_coeffs.to_csv(outdir / "model_coefficients_all.csv", index=False)
        _write_group_coefficients(all_coeffs, outdir)
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(outdir / "model_summary.csv", index=False)
    plan_records = _specs_to_records(specs)
    pd.DataFrame(plan_records).to_json(outdir / "model_plan.json", orient="records", indent=2)
    _write_ref_choice_model_plan(plan_records, outdir / "model_plan_ref_choice.json")


def _write_ref_choice_model_plan(plan_records: list[dict[str, object]], output_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    ref_spec_path = repo_root / "evaluation" / "specs" / "belief_update_targets.json"
    if not ref_spec_path.exists():
        raise FileNotFoundError(f"ref spec JSON not found: {ref_spec_path}")

    ref_spec_raw = json.loads(ref_spec_path.read_text(encoding="utf-8"))
    ref_choice_models = ref_spec_raw.get("ref_choice_models")
    if not isinstance(ref_choice_models, dict):
        raise ValueError(f"Missing or invalid ref_choice_models in {ref_spec_path}")

    selected_names: list[str] = []
    seen_names: set[str] = set()
    for model_name in ref_choice_models.values():
        if not isinstance(model_name, str):
            continue
        cleaned = model_name.strip()
        if not cleaned or cleaned in seen_names:
            continue
        selected_names.append(cleaned)
        seen_names.add(cleaned)

    by_name = {str(entry.get("model") or entry.get("name") or "").strip(): entry for entry in plan_records}
    missing = [name for name in selected_names if name not in by_name]
    if missing:
        missing_str = ", ".join(missing)
        raise KeyError(f"Missing ref model(s) in model plan: {missing_str}")

    filtered_records = [by_name[name] for name in selected_names]
    output_path.write_text(f"{json.dumps(filtered_records, indent=2, ensure_ascii=False)}\n", encoding="utf-8")


def _write_block_table(df: pd.DataFrame, outdir: Path, filename: str) -> None:
    df.drop(columns=["group"], errors="ignore").to_csv(outdir / filename, index=False)


def _write_group_coefficients(df: pd.DataFrame, outdir: Path) -> None:
    if "group" not in df.columns:
        return
    groups = [g for g in df["group"].dropna().unique() if str(g)]
    for group in sorted(groups, key=str):
        group_dir = outdir / str(group)
        group_dir.mkdir(parents=True, exist_ok=True)
        group_df = df[df["group"] == group]
        group_df.to_csv(group_dir / "coefficients.csv", index=False)


def _write_group_block_tables(tables: list[pd.DataFrame], outdir: Path, filename: str) -> None:
    cleaned = [table for table in tables if table is not None and not table.empty]
    if not cleaned:
        return
    combined = pd.concat(cleaned, ignore_index=True)
    if "group" not in combined.columns:
        return
    groups = [g for g in combined["group"].dropna().unique() if str(g)]
    for group in sorted(groups, key=str):
        group_dir = outdir / str(group)
        group_dir.mkdir(parents=True, exist_ok=True)
        group_df = combined[combined["group"] == group]
        _write_block_table(group_df, group_dir, filename)


def _filter_specs(specs, include: set[str] | None):
    if not include:
        return specs
    chosen = [spec for spec in specs if spec.name in include]
    missing = include - {spec.name for spec in chosen}
    if missing:
        logger.warning(f"Requested models not found: {', '.join(sorted(missing))}")
    return chosen


def _prune_spec(spec: ModelSpec, derivations: pl.DataFrame | None) -> ModelSpec:
    """Drop redundant derived predictors to avoid rank deficiency."""
    dropped: list[str] = []
    pruned_blocks: list[ModelBlock] = []
    kept: set[str] = set()

    for block in spec.blocks:
        pruned = drop_redundant_predictors(block.predictors, derivations)
        removed = set(block.predictors) - set(pruned)
        if removed:
            dropped.extend(sorted(removed))
        pruned_blocks.append(ModelBlock(block.name, tuple(pruned), block.sample_strategy))
        kept.update(pruned)

    pruned_interactions = tuple(inter for inter in spec.interactions if all(term in kept for term in inter))
    if dropped:
        logger.info(f"Pruned derived predictors from {spec.name}: {', '.join(sorted(set(dropped)))}")

    return ModelSpec(
        name=spec.name,
        outcome=spec.outcome,
        model_type=spec.model_type,
        blocks=tuple(pruned_blocks),
        interactions=pruned_interactions,
        add_country=spec.add_country,
        description=spec.description,
        group=spec.group,
    )


def _enforce_dtypes(df: pl.DataFrame, column_types: dict[str, str], weight_column: str | None) -> pl.DataFrame:
    """Cast columns according to column_types to avoid accidental object dtypes."""
    type_map: dict[str, pl.DataType] = {
        "continuous": pl.Float64,
        "ordinal": pl.Float64,
        "categorical": pl.Utf8,
    }

    binary_string_map = {
        "true": "1",
        "false": "0",
        "yes": "1",
        "no": "0",
        "y": "1",
        "n": "0",
        "t": "1",
        "f": "0",
        "1": "1",
        "0": "0",
        "nan": None,
        "null": None,
        "": None,
    }

    numeric_dtypes = {
        pl.Int8,
        pl.Int16,
        pl.Int32,
        pl.Int64,
        pl.UInt8,
        pl.UInt16,
        pl.UInt32,
        pl.UInt64,
        pl.Float32,
        pl.Float64,
    }

    def _cast_or_raise(current: pl.DataFrame, col: str, expr: pl.Expr, target: str) -> pl.DataFrame:
        try:
            return current.with_columns(expr.alias(col))
        except Exception as exc:  # pragma: no cover - defensive guard for data issues
            raise ValueError(f"Failed to cast column '{col}' to {target}") from exc

    out = df
    for col in df.columns:
        if weight_column and col == weight_column:
            out = _cast_or_raise(out, col, pl.col(col).cast(pl.Float64, strict=True), "float")
            continue
        col_type = column_types.get(col, "").lower()
        if col_type == "binary":
            if out.schema.get(col) == pl.Boolean:
                # Already a boolean column; keep as-is.
                continue
            dtype = out.schema.get(col)
            cleaned = pl.col(col).cast(pl.Utf8, strict=False).str.strip_chars().str.to_lowercase()

            if dtype in numeric_dtypes:
                numeric = pl.col(col).cast(pl.Float64, strict=False)
            else:
                mapped = cleaned.replace(binary_string_map)
                numeric = mapped.cast(pl.Float64, strict=False)

            invalid_mask = (numeric.is_null() & cleaned.is_not_null()) | (
                numeric.is_not_null() & ~numeric.is_in(pl.lit([0.0, 1.0]))
            )
            invalid_values = (
                out.select(pl.when(invalid_mask).then(cleaned).alias("invalid"))
                .get_column("invalid")
                .drop_nulls()
                .unique()
                .to_list()
            )
            if invalid_values:
                raise ValueError(
                    f"Binary column '{col}' contains unsupported values: {', '.join(sorted(map(str, invalid_values)))}"
                )

            expr = numeric.cast(pl.Boolean, strict=False)
            out = _cast_or_raise(out, col, expr, "boolean (binary)")
            continue
        target = type_map.get(col_type)
        if target is not None:
            out = _cast_or_raise(out, col, pl.col(col).cast(target, strict=True), str(target))
    return out


def _load_dataset(
    input_path: Path,
    columns: set[str],
    column_types: dict[str, str],
    weight_column: str | None,
) -> pd.DataFrame:
    head = pl.read_csv(input_path, n_rows=0)
    available = set(head.columns)
    missing = columns - available
    if missing:
        logger.warning(f"Columns not found in dataset and will be skipped: {', '.join(sorted(missing))}")
    selected = sorted(columns & available)
    df = pl.read_csv(input_path, columns=selected)
    df = _enforce_dtypes(df, column_types, weight_column)
    pdf = df.to_pandas()
    # Normalise pandas dtypes for binary columns to avoid unexpected object dtypes
    # (e.g. nullable booleans with NaNs) that break downstream numeric checks.
    for col, kind in column_types.items():
        if kind.lower() == "binary" and col in pdf.columns:
            if not pd.api.types.is_numeric_dtype(pdf[col]):
                pdf[col] = pdf[col].astype("float")
    return pdf


def _parse_models(raw: str | None) -> set[str] | None:
    if not raw:
        return None
    return {item.strip() for item in raw.split(",") if item.strip()}


def _parse_baseline_blocks(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [part.strip() for part in raw.split(",") if part.strip()]


def _filter_interactions(interactions: Iterable[tuple[str, ...]], allowed: set[str]) -> tuple[tuple[str, ...], ...]:
    return tuple(inter for inter in interactions if all(term in allowed for term in inter))


def _spec_with_blocks(spec: ModelSpec, keep: set[str]) -> ModelSpec:
    kept_blocks: tuple[ModelBlock, ...] = tuple(b for b in spec.blocks if b.name in keep)
    predictors = {p for block in kept_blocks for p in block.predictors}
    pruned_interactions = _filter_interactions(spec.interactions, predictors)
    return replace(spec, blocks=kept_blocks, interactions=pruned_interactions)


def _safe_delta(base: float | None, other: float | None) -> float | None:
    if base is None or other is None:
        return None
    return float(base - other)


def _relative_delta(delta: float | None, baseline: float | None) -> float | None:
    if baseline is None:
        return None
    if baseline <= 0:
        return 0.0
    if delta is None:
        return None
    return float(delta) / float(baseline)


def _classify_block_strength(rel_delta: float | None) -> str | None:
    if rel_delta is None:
        return None
    if rel_delta < 0.05:
        return "negligible"
    if rel_delta < 0.15:
        return "small"
    if rel_delta < 0.30:
        return "medium"
    return "large"


def _load_specs_and_data(
    *,
    input_csv: Path,
    column_types_path: Path,
    derivations_path: Path | None,
    models: str | None,
    weight_column: str | None,
    standardize: bool,
    missingness_threshold: float,
    load_all_spec_columns: bool = False,
) -> tuple[
    list[ModelSpec], list[ModelSpec], pd.DataFrame, dict[str, str], dict[str, object], dict[str, pd.Series] | None
]:
    """Shared loader used by fit, block-importance and run commands."""
    # Import plan locally (kept next to the CLI for easy editing/inspection)
    script_dir = Path(__file__).resolve().parent
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))
    from modeling_plan import build_structural_backbone_plan  # type: ignore

    all_specs = list(build_structural_backbone_plan())
    all_specs = [spec for spec in all_specs if spec.blocks]  # defensive: ensure blocks exist
    requested = _parse_models(models)
    specs = all_specs
    if requested:
        specs = [spec for spec in specs if spec.name in requested]
        missing = requested - {spec.name for spec in specs}
        if missing:
            logger.warning(f"Requested models not found: {', '.join(sorted(missing))}")
    if not specs:
        raise typer.BadParameter("No model specifications selected")

    column_types, reference_levels = _load_column_types(column_types_path)
    if derivations_path is not None:
        try:
            deriv = load_derivations(derivations_path)
            all_specs = [_prune_spec(spec, deriv) for spec in all_specs]
            specs = [spec for spec in all_specs if (not requested or spec.name in requested)]
        except FileNotFoundError:
            logger.warning(f"Derivations file not found; skipping pruning: {derivations_path}")
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(f"Failed to load derivations; skipping pruning: {exc}")

    needed_specs_for_columns = all_specs if load_all_spec_columns else specs
    needed_columns = _collect_required_columns(needed_specs_for_columns)
    needed_columns_for_csv = needed_columns
    if weight_column:
        needed_columns_for_csv.add(weight_column)

    logger.info(f"Loading dataset with {len(needed_columns_for_csv)} required columns")
    df = _load_dataset(input_csv, needed_columns_for_csv, column_types, weight_column)

    predictors = {col for spec in specs for col in spec.predictor_list() if col != "country" and col in df.columns}
    missingness_plan = build_missingness_plan_pd(
        df,
        predictors=sorted(predictors),
        threshold=missingness_threshold,
        column_types=column_types,
        country_col="country",
    )
    for col in missingness_plan.columns:
        column_types.setdefault(missingness_indicator_name(col), "binary")
    specs = _append_missingness_block(specs, missingness_plan.columns)

    if standardize:
        predict_cols = {
            col for spec in specs for col in spec.predictor_list() if col != "country" and col in df.columns
        }
        df = standardize_columns(df, predict_cols, column_types)

    df = apply_missingness_plan_pd(df, missingness_plan, standardize=standardize, country_col="country")

    return specs, all_specs, df, column_types, reference_levels, None


def _fit_with_metrics(
    spec: ModelSpec,
    df: pd.DataFrame,
    column_types: dict[str, str],
    reference_levels: dict[str, object],
    weight_column: str | None,
    missingness_originals: dict[str, pd.Series] | None = None,
) -> dict[str, float | int | None]:
    """Fit a spec and return minimal metrics for block attribution."""
    formula, missing = _prepare_formula(spec, df, column_types, reference_levels)
    if missing:
        logger.debug(f"{spec.name}: missing predictors dropped: {', '.join(sorted(missing))}")

    needed_cols = [term for term in spec.predictor_list() if term in df.columns]
    needed_cols.append(spec.outcome)
    if weight_column:
        needed_cols.append(weight_column)
    # Drop duplicate columns while preserving order.
    seen: set[str] = set()
    ordered: list[str] = []
    for col in needed_cols:
        if col in seen:
            continue
        seen.add(col)
        ordered.append(col)
    working = df[ordered].copy()
    if missingness_originals and spec.outcome in missingness_originals:
        original = missingness_originals[spec.outcome]
        working[spec.outcome] = original.reindex(working.index)

    if not pd.api.types.is_numeric_dtype(working[spec.outcome]):
        raise TypeError(f"Outcome '{spec.outcome}' must be numeric; got {working[spec.outcome].dtype}")
    if weight_column and weight_column in working and not pd.api.types.is_numeric_dtype(working[weight_column]):
        raise TypeError(f"Weight column '{weight_column}' must be numeric; got {working[weight_column].dtype}")

    working = drop_nonfinite_rows_pd(working, working.columns, context=f"model:{spec.name}")
    weights = working[weight_column] if weight_column and weight_column in working.columns else None

    if spec.model_type == "linear":
        result = _fit_linear(formula, working, weights)
    elif spec.model_type == "logit":
        result = _fit_logit(formula, working, weights)
    elif spec.model_type == "ordinal":
        result = _fit_ordinal(formula, working, weights)
    else:
        raise ValueError(f"Unknown model type: {spec.model_type}")

    pseudo_r2 = _compute_pseudo_r2(result, spec.model_type)
    bic_value = bic_consistent(result)
    llf = getattr(result, "llf", None)
    llnull = getattr(result, "llnull", None)
    nobs = int(getattr(result, "nobs", working.shape[0]))
    return {
        "pseudo_r2": float(pseudo_r2) if pseudo_r2 is not None and np.isfinite(pseudo_r2) else None,
        "bic": float(bic_value) if bic_value is not None and np.isfinite(bic_value) else None,
        "llf": float(llf) if llf is not None and np.isfinite(llf) else None,
        "llnull": float(llnull) if llnull is not None and np.isfinite(llnull) else None,
        "nobs": nobs,
    }


def _block_row(
    *,
    spec: ModelSpec,
    block_name: str,
    kind: str,
    metric: str,
    baseline_value: float | None,
    delta: float | None,
    rel_delta: float | None,
    block_strength: str | None,
    baseline_nobs: int,
    baseline_blocks: str,
    sample_strategy: str,
    variant_nobs: int,
    baseline_spec: str | None = None,
    variant_spec: str | None = None,
) -> dict[str, object]:
    """Standard row representation for block-importance outputs."""

    return {
        "model": spec.name,
        "outcome": spec.outcome,
        "block": block_name,
        "kind": kind,
        "metric": metric,
        "baseline_value": baseline_value,
        "delta": delta,
        "rel_delta": rel_delta,
        "block_strength": block_strength,
        "baseline_nobs": baseline_nobs,
        "baseline_blocks": baseline_blocks,
        "sample_strategy": sample_strategy,
        "variant_nobs": variant_nobs,
        "baseline_spec": baseline_spec,
        "variant_spec": variant_spec,
        "lobo_bic_weight": None,
        "group": spec.group,
    }


def _append_lobo_bic_weights(df: pd.DataFrame | None) -> pd.DataFrame | None:
    """Add LOBO ΔBIC relative-importance weights per model/outcome.

    For LOBO BIC rows, compute:
        w_{k,o} = max(ΔBIC_{k,o}, 0) / sum_j max(ΔBIC_{j,o}, 0)
    """
    if df is None or df.empty:
        return df

    out = df.copy()
    mask = (out["kind"] == "lobo") & (out["metric"] == "bic")
    if not mask.any():
        return out

    bic_rows = out.loc[mask, ["model", "outcome", "delta"]].copy()
    bic_rows["positive_delta"] = bic_rows["delta"].clip(lower=0.0)
    denom = bic_rows.groupby(["model", "outcome"])["positive_delta"].transform("sum")
    bic_rows["lobo_bic_weight"] = np.where(
        denom > 0.0,
        bic_rows["positive_delta"] / denom,
        np.nan,
    )
    out.loc[mask, "lobo_bic_weight"] = bic_rows["lobo_bic_weight"].to_numpy()
    return out


def _compute_lobo(
    spec: ModelSpec,
    df: pd.DataFrame,
    column_types: dict[str, str],
    reference_levels: dict[str, object],
    weight_column: str | None,
    sample_strategy: str,
    spec_label: str | None = None,
    missingness_originals: dict[str, pd.Series] | None = None,
) -> pd.DataFrame:
    """Leave-one-block-out: FULL vs FULL minus each block."""
    full_metrics = _fit_with_metrics(
        spec,
        df,
        column_types,
        reference_levels,
        weight_column,
        missingness_originals,
    )
    rows: list[dict[str, object]] = []
    full_block_names = {b.name for b in spec.blocks}
    baseline_spec_name = spec_label or spec.name
    for block in spec.blocks:
        keep_blocks = full_block_names - {block.name}
        if not keep_blocks:
            continue
        variant_spec = _spec_with_blocks(spec, keep_blocks)
        try:
            m = _fit_with_metrics(
                variant_spec,
                df,
                column_types,
                reference_levels,
                weight_column,
                missingness_originals,
            )
        except Exception as exc:
            logger.warning(f"LOBO failed for {spec.name} minus {block.name}: {exc}")
            continue
        baseline_value = full_metrics["pseudo_r2"]
        delta = _safe_delta(full_metrics["pseudo_r2"], m["pseudo_r2"])
        rel_delta = _relative_delta(delta, baseline_value)
        rows.append(
            _block_row(
                spec=spec,
                block_name=block.name,
                kind="lobo",
                metric="pseudo_r2",
                baseline_value=baseline_value,
                delta=delta,
                rel_delta=rel_delta,
                block_strength=_classify_block_strength(rel_delta),
                baseline_nobs=full_metrics["nobs"],
                baseline_blocks="ALL",
                sample_strategy=sample_strategy,
                variant_nobs=int(m["nobs"]),
                baseline_spec=baseline_spec_name,
                variant_spec=f"{baseline_spec_name}-minus-{block.name}",
            )
        )

        bic_full = full_metrics.get("bic")
        bic_minus = m.get("bic")
        if bic_full is not None and bic_minus is not None:
            delta_bic = float(bic_minus - bic_full)
            rows.append(
                _block_row(
                    spec=spec,
                    block_name=block.name,
                    kind="lobo",
                    metric="bic",
                    baseline_value=bic_full,
                    delta=delta_bic,
                    rel_delta=None,
                    block_strength=None,
                    baseline_nobs=full_metrics["nobs"],
                    baseline_blocks="ALL",
                    sample_strategy=sample_strategy,
                    variant_nobs=int(m["nobs"]),
                    baseline_spec=baseline_spec_name,
                    variant_spec=f"{baseline_spec_name}-minus-{block.name}",
                )
            )
    return pd.DataFrame(rows)


def _compute_lobi(
    spec: ModelSpec,
    df: pd.DataFrame,
    column_types: dict[str, str],
    reference_levels: dict[str, object],
    weight_column: str | None,
    baseline_blocks: Iterable[str],
    sample_strategy: str,
    spec_label: str | None = None,
    missingness_originals: dict[str, pd.Series] | None = None,
) -> pd.DataFrame:
    """Leave-one-block-in: baseline S0 vs S0 + block."""
    all_block_names = [b.name for b in spec.blocks]
    baseline_list = [b for b in baseline_blocks if b in all_block_names]
    base_set = set(baseline_list)
    # Ensure deterministic order aligned with original spec.blocks
    base_spec = _spec_with_blocks(spec, set(base_set))
    base_metrics = _fit_with_metrics(
        base_spec,
        df,
        column_types,
        reference_levels,
        weight_column,
        missingness_originals,
    )
    rows: list[dict[str, object]] = []
    base_spec_name = f"{spec_label or spec.name}-baseline"
    for block in spec.blocks:
        if block.name in base_set:
            variant_set = set(base_set)
        else:
            variant_set = set(base_set | {block.name})
        variant_spec = _spec_with_blocks(spec, variant_set)
        try:
            m = _fit_with_metrics(
                variant_spec,
                df,
                column_types,
                reference_levels,
                weight_column,
                missingness_originals,
            )
        except Exception as exc:
            logger.warning(f"LOBI failed for {spec.name} with +{block.name}: {exc}")
            continue
        baseline_value = base_metrics["pseudo_r2"]
        delta = _safe_delta(m["pseudo_r2"], base_metrics["pseudo_r2"])
        rel_delta = _relative_delta(delta, baseline_value)
        rows.append(
            _block_row(
                spec=spec,
                block_name=block.name,
                kind="lobi",
                metric="pseudo_r2",
                baseline_value=baseline_value,
                delta=delta,
                rel_delta=rel_delta,
                block_strength=_classify_block_strength(rel_delta),
                baseline_nobs=base_metrics["nobs"],
                baseline_blocks=",".join(baseline_list),
                sample_strategy=sample_strategy,
                variant_nobs=int(m["nobs"]),
                baseline_spec=base_spec_name,
                variant_spec=f"{base_spec_name}+{block.name}",
            )
        )

        bic_base = base_metrics.get("bic")
        bic_plus = m.get("bic")
        if bic_base is not None and bic_plus is not None:
            delta_bic = float(bic_base - bic_plus)
            rows.append(
                _block_row(
                    spec=spec,
                    block_name=block.name,
                    kind="lobi",
                    metric="bic",
                    baseline_value=bic_base,
                    delta=delta_bic,
                    rel_delta=None,
                    block_strength=None,
                    baseline_nobs=base_metrics["nobs"],
                    baseline_blocks=",".join(baseline_list),
                    sample_strategy=sample_strategy,
                    variant_nobs=int(m["nobs"]),
                    baseline_spec=base_spec_name,
                    variant_spec=f"{base_spec_name}+{block.name}",
                )
            )
    return pd.DataFrame(rows)


def _compute_pairwise_block_rows(
    *,
    spec: ModelSpec,
    block: ModelBlock,
    df: pd.DataFrame,
    column_types: dict[str, str],
    reference_levels: dict[str, object],
    weight_column: str | None,
    baseline_blocks: Iterable[str],
    missingness_originals: dict[str, pd.Series] | None = None,
) -> pd.DataFrame:
    blocks_by_name = {b.name: b for b in spec.blocks}
    full_block_names = set(blocks_by_name)

    lobo_keep = full_block_names - {block.name}
    if not lobo_keep:
        logger.warning(f"Skipping LOBO for {spec.name} minus {block.name}: no remaining blocks")
        lobo_rows = pd.DataFrame()
    else:
        lobo_full_spec = _spec_with_blocks(spec, full_block_names)
        lobo_minus_spec = _spec_with_blocks(spec, lobo_keep)
        lobo_required = _block_predictor_columns(lobo_full_spec.blocks)
        lobo_required.add(spec.outcome)
        if spec.add_country:
            lobo_required.add("country")
        lobo_df = _complete_case(df, lobo_required)
        if lobo_df.empty:
            logger.warning(f"Skipping LOBO for {spec.name} minus {block.name}: no matched rows")
            lobo_rows = pd.DataFrame()
        else:
            try:
                full_metrics = _fit_with_metrics(
                    lobo_full_spec,
                    lobo_df,
                    column_types,
                    reference_levels,
                    weight_column,
                    missingness_originals,
                )
                minus_metrics = _fit_with_metrics(
                    lobo_minus_spec,
                    lobo_df,
                    column_types,
                    reference_levels,
                    weight_column,
                    missingness_originals,
                )
            except Exception as exc:  # pragma: no cover - defensive for data issues
                logger.warning(f"Pairwise LOBO failed for {spec.name} ({block.name}): {exc}")
                lobo_rows = pd.DataFrame()
            else:
                delta = _safe_delta(full_metrics["pseudo_r2"], minus_metrics["pseudo_r2"])
                rel_delta = _relative_delta(delta, full_metrics["pseudo_r2"])
                rows: list[dict[str, object]] = [
                    _block_row(
                        spec=spec,
                        block_name=block.name,
                        kind="lobo",
                        metric="pseudo_r2",
                        baseline_value=full_metrics["pseudo_r2"],
                        delta=delta,
                        rel_delta=rel_delta,
                        block_strength=_classify_block_strength(rel_delta),
                        baseline_nobs=full_metrics["nobs"],
                        baseline_blocks="ALL",
                        sample_strategy="pairwise_matched",
                        variant_nobs=int(minus_metrics["nobs"]),
                        baseline_spec=spec.name,
                        variant_spec=f"{spec.name}-minus-{block.name}",
                    )
                ]
                bic_full = full_metrics.get("bic")
                bic_minus = minus_metrics.get("bic")
                if bic_full is not None and bic_minus is not None:
                    rows.append(
                        _block_row(
                            spec=spec,
                            block_name=block.name,
                            kind="lobo",
                            metric="bic",
                            baseline_value=bic_full,
                            delta=float(bic_minus - bic_full),
                            rel_delta=None,
                            block_strength=None,
                            baseline_nobs=full_metrics["nobs"],
                            baseline_blocks="ALL",
                            sample_strategy="pairwise_matched",
                            variant_nobs=int(minus_metrics["nobs"]),
                            baseline_spec=spec.name,
                            variant_spec=f"{spec.name}-minus-{block.name}",
                        )
                    )
                lobo_rows = pd.DataFrame(rows)

    available = set(blocks_by_name)
    baseline_names: list[str] = []
    for name in _core_block_names(spec):
        if name != block.name and name in available:
            baseline_names.append(name)
    for name in baseline_blocks:
        if name == block.name or name not in available or name in baseline_names:
            continue
        baseline_names.append(name)

    variant_names = list(dict.fromkeys((*baseline_names, block.name)))
    base_spec = _spec_with_blocks(spec, set(baseline_names))
    variant_spec = _spec_with_blocks(spec, set(variant_names))
    lobi_required = _block_predictor_columns(variant_spec.blocks)
    lobi_required.add(spec.outcome)
    if spec.add_country:
        lobi_required.add("country")
    lobi_df = _complete_case(df, lobi_required)
    if lobi_df.empty:
        logger.warning(f"Skipping LOBI for {spec.name} plus {block.name}: no matched rows")
        lobi_rows = pd.DataFrame()
    else:
        try:
            baseline_metrics = _fit_with_metrics(
                base_spec,
                lobi_df,
                column_types,
                reference_levels,
                weight_column,
                missingness_originals,
            )
            variant_metrics = _fit_with_metrics(
                variant_spec,
                lobi_df,
                column_types,
                reference_levels,
                weight_column,
                missingness_originals,
            )
        except Exception as exc:  # pragma: no cover - defensive for data issues
            logger.warning(f"Pairwise LOBI failed for {spec.name} ({block.name}): {exc}")
            lobi_rows = pd.DataFrame()
        else:
            delta = _safe_delta(variant_metrics["pseudo_r2"], baseline_metrics["pseudo_r2"])
            rel_delta = _relative_delta(delta, baseline_metrics["pseudo_r2"])
            baseline_label = ",".join(baseline_names)
            rows = [
                _block_row(
                    spec=spec,
                    block_name=block.name,
                    kind="lobi",
                    metric="pseudo_r2",
                    baseline_value=baseline_metrics["pseudo_r2"],
                    delta=delta,
                    rel_delta=rel_delta,
                    block_strength=_classify_block_strength(rel_delta),
                    baseline_nobs=baseline_metrics["nobs"],
                    baseline_blocks=baseline_label,
                    sample_strategy="pairwise_matched",
                    variant_nobs=int(variant_metrics["nobs"]),
                    baseline_spec=f"{spec.name}-baseline",
                    variant_spec=f"{spec.name}-baseline+{block.name}",
                )
            ]
            bic_base = baseline_metrics.get("bic")
            bic_plus = variant_metrics.get("bic")
            if bic_base is not None and bic_plus is not None:
                rows.append(
                    _block_row(
                        spec=spec,
                        block_name=block.name,
                        kind="lobi",
                        metric="bic",
                        baseline_value=bic_base,
                        delta=float(bic_base - bic_plus),
                        rel_delta=None,
                        block_strength=None,
                        baseline_nobs=baseline_metrics["nobs"],
                        baseline_blocks=baseline_label,
                        sample_strategy="pairwise_matched",
                        variant_nobs=int(variant_metrics["nobs"]),
                        baseline_spec=f"{spec.name}-baseline",
                        variant_spec=f"{spec.name}-baseline+{block.name}",
                    )
                )
            lobi_rows = pd.DataFrame(rows)

    return pd.concat([lobo_rows, lobi_rows], ignore_index=True)


def _compute_soaking_scores(
    *,
    spec: ModelSpec,
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Compute per-block soaking R^2: block index explained by other block indices."""
    block_index: dict[str, pd.Series] = {}

    for block in spec.blocks:
        predictors = [col for col in block.predictors if col in df.columns]
        if not predictors:
            continue
        numeric = df[predictors].apply(pd.to_numeric, errors="coerce")
        if numeric.empty:
            continue
        index = numeric.mean(axis=1, skipna=True)
        if not np.isfinite(index.to_numpy(dtype=float)).any():
            continue
        block_index[block.name] = index

    if len(block_index) < 2:
        return pd.DataFrame()

    index_df = pd.DataFrame(block_index)
    rows: list[dict[str, object]] = []
    for target in index_df.columns:
        others = [col for col in index_df.columns if col != target]
        if not others:
            continue
        design = index_df[[target, *others]].dropna()
        if design.empty:
            continue

        y = design[target].astype(float)
        x = design[others].astype(float)
        x = sm.add_constant(x, has_constant="add")
        nobs = int(len(y))
        k = int(x.shape[1])
        if nobs <= k:
            continue
        try:
            fit = sm.OLS(y, x).fit()
        except Exception as exc:  # pragma: no cover - defensive for ill-conditioned designs
            logger.warning(f"Soaking OLS failed for {spec.name} block {target}: {exc}")
            continue

        r2 = float(getattr(fit, "rsquared", np.nan))
        rows.append(
            {
                "model": spec.name,
                "outcome": spec.outcome,
                "block": target,
                "soaking_r2": r2 if np.isfinite(r2) else None,
                "soaking_nobs": nobs,
                "group": spec.group,
            }
        )
    return pd.DataFrame(rows)


def _run_fit(
    *,
    specs: list[ModelSpec],
    df: pd.DataFrame,
    column_types: dict[str, str],
    reference_levels: dict[str, object],
    weight_column: str | None,
    outdir: Path,
    max_workers: int,
    missingness_originals: dict[str, pd.Series] | None,
) -> None:
    results = run_model_suite(
        specs,
        df,
        column_types,
        reference_levels,
        weight_column=weight_column,
        max_workers=max_workers,
        outcome_overrides=missingness_originals,
    )
    _persist_results(results, specs, outdir)
    logger.success(f"Modeling outputs written to {outdir}")


def _run_block_importance(
    *,
    specs: list[ModelSpec],
    df: pd.DataFrame,
    column_types: dict[str, str],
    reference_levels: dict[str, object],
    weight_column: str | None,
    baseline_blocks: list[str],
    outdir: Path,
    max_workers: int,
    missingness_originals: dict[str, pd.Series] | None,
    soaking_source_specs: dict[str, ModelSpec] | None = None,
) -> None:
    baseline = baseline_blocks or ["controls_demography_SES", "health_history_engine"]
    logger.info(
        "Block importance: specs={} baseline_blocks={}",
        len(specs),
        ",".join(baseline) if baseline else "(none)",
    )

    resolved_sources = soaking_source_specs or {spec.name: spec for spec in specs}
    source_specs_by_name = {source.name: source for source in resolved_sources.values()}
    source_soaking_tables: dict[str, pd.DataFrame] = {}

    for source_name, source_spec in source_specs_by_name.items():
        if not source_spec.blocks:
            source_soaking_tables[source_name] = pd.DataFrame()
            continue
        source_soaking_tables[source_name] = _compute_soaking_scores(spec=source_spec, df=df)

    def _worker(
        spec: ModelSpec,
    ) -> tuple[ModelSpec, pd.DataFrame | None, pd.DataFrame | None, pd.DataFrame | None] | None:
        logger.info("Computing LOBO/LOBI for {} (blocks={})", spec.name, len(spec.blocks))
        spec_lobo_parts: list[pd.DataFrame] = []
        spec_lobi_parts: list[pd.DataFrame] = []
        soaking_df = pd.DataFrame()

        source_spec = resolved_sources.get(spec.name, spec)
        source_soaking = source_soaking_tables.get(source_spec.name)
        if source_soaking is not None and not source_soaking.empty:
            soaking_df = source_soaking.copy()
            soaking_df["model"] = spec.name
            soaking_df["outcome"] = spec.outcome
            soaking_df["group"] = spec.group
            soaking_df["soaking_source_model"] = source_spec.name

        for block in spec.blocks:
            pairwise_df = _compute_pairwise_block_rows(
                spec=spec,
                block=block,
                df=df,
                column_types=column_types,
                reference_levels=reference_levels,
                weight_column=weight_column,
                baseline_blocks=baseline,
                missingness_originals=missingness_originals,
            )
            if pairwise_df.empty:
                continue
            pairwise_lobo = pairwise_df[pairwise_df["kind"] == "lobo"]
            pairwise_lobi = pairwise_df[pairwise_df["kind"] == "lobi"]
            if not pairwise_lobo.empty:
                if not soaking_df.empty:
                    pairwise_lobo = pairwise_lobo.merge(
                        soaking_df[["model", "outcome", "block", "soaking_r2", "soaking_nobs"]],
                        on=["model", "outcome", "block"],
                        how="left",
                    )
                spec_lobo_parts.append(pairwise_lobo)
            if not pairwise_lobi.empty:
                if not soaking_df.empty:
                    pairwise_lobi = pairwise_lobi.merge(
                        soaking_df[["model", "outcome", "block", "soaking_r2", "soaking_nobs"]],
                        on=["model", "outcome", "block"],
                        how="left",
                    )
                spec_lobi_parts.append(pairwise_lobi)

        model_outdir = _model_output_dir(spec, outdir)
        spec_lobo_df = pd.concat(spec_lobo_parts, ignore_index=True) if spec_lobo_parts else None
        spec_lobo_df = _append_lobo_bic_weights(spec_lobo_df)
        spec_lobi_df = pd.concat(spec_lobi_parts, ignore_index=True) if spec_lobi_parts else None
        if spec_lobo_df is not None:
            _write_block_table(spec_lobo_df, model_outdir, "lobo.csv")
        if spec_lobi_df is not None:
            _write_block_table(spec_lobi_df, model_outdir, "lobi.csv")
        if not soaking_df.empty:
            _write_block_table(soaking_df, model_outdir, "soaking.csv")
        if spec_lobo_df is None and spec_lobi_df is None:
            logger.warning(f"No block importance rows generated for {spec.name}")
        return spec, spec_lobo_df, spec_lobi_df, (None if soaking_df.empty else soaking_df)

    results = _run_tasks(specs, _worker, max_workers)
    all_lobo_tables: list[pd.DataFrame] = []
    all_lobi_tables: list[pd.DataFrame] = []
    all_soaking_tables: list[pd.DataFrame] = []
    for _, lobo_df, lobi_df, soaking_df in results:
        if lobo_df is not None:
            all_lobo_tables.append(lobo_df)
        if lobi_df is not None:
            all_lobi_tables.append(lobi_df)
        if soaking_df is not None:
            all_soaking_tables.append(soaking_df)

    _write_group_block_tables(all_lobo_tables, outdir, "lobo.csv")
    _write_group_block_tables(all_lobi_tables, outdir, "lobi.csv")
    _write_group_block_tables(all_soaking_tables, outdir, "soaking.csv")

    # Also write global (all-spec) tables for downstream reporting.
    if all_lobo_tables:
        _write_block_table(pd.concat(all_lobo_tables, ignore_index=True), outdir, "model_lobo_all.csv")
    if all_lobi_tables:
        _write_block_table(pd.concat(all_lobi_tables, ignore_index=True), outdir, "model_lobi_all.csv")
    if all_soaking_tables:
        _write_block_table(pd.concat(all_soaking_tables, ignore_index=True), outdir, "model_soaking_all.csv")
    logger.success(f"Block importance outputs written to {outdir}")


@app.command("fit")
def fit_cmd(
    input_csv: Path = typer.Option(CSV_FILE_CLEAN, "--input", help="Input CLEAN CSV"),
    column_types_path: Path = typer.Option(DEFAULT_COLUMN_TYPES_PATH, "--column-types", help="Column type TSV"),
    derivations_path: Path | None = typer.Option(
        DEFAULT_DERIVATIONS_PATH, "--derivations", help="Derivations TSV to drop redundant predictors"
    ),
    outdir: Path = typer.Option(DEFAULT_OUTDIR, "--outdir", help="Output directory"),
    models: str | None = typer.Option(None, "--models", help="Comma list of model names to run (defaults to all)"),
    weight_column: str | None = typer.Option(None, "--weight-column", help="Optional survey weight column"),
    standardize: bool = typer.Option(True, "--standardize/--no-standardize", help="Standardise numeric predictors"),
    missingness_threshold: float = typer.Option(
        DEFAULT_MISSINGNESS_THRESHOLD,
        "--missingness-threshold",
        help="Max missingness fraction for predictor imputation + missing indicator.",
    ),
    max_workers: int = typer.Option(6, "--max-workers", min=0, help="Worker threads (0 = auto)"),
    log_level: str = typer.Option("INFO", "--log-level"),
) -> None:
    """Run the full modeling.md backbone suite."""

    _configure_logging(log_level)
    specs, _all_specs, df, column_types, reference_levels, missingness_originals = _load_specs_and_data(
        input_csv=input_csv,
        column_types_path=column_types_path,
        derivations_path=derivations_path,
        models=models,
        weight_column=weight_column,
        standardize=standardize,
        missingness_threshold=missingness_threshold,
    )
    try:
        worker_count = _normalize_max_workers(max_workers)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    _run_fit(
        specs=specs,
        df=df,
        column_types=column_types,
        reference_levels=reference_levels,
        weight_column=weight_column,
        outdir=outdir,
        max_workers=worker_count,
        missingness_originals=missingness_originals,
    )


@app.command("block-importance")
def block_importance(
    input_csv: Path = typer.Option(CSV_FILE_CLEAN, "--input", help="Input CLEAN CSV"),
    column_types_path: Path = typer.Option(DEFAULT_COLUMN_TYPES_PATH, "--column-types", help="Column type TSV"),
    derivations_path: Path | None = typer.Option(
        DEFAULT_DERIVATIONS_PATH, "--derivations", help="Derivations TSV to drop redundant predictors"
    ),
    outdir: Path = typer.Option(Path("empirical/output/modeling"), "--outdir", help="Output directory"),
    models: str | None = typer.Option(None, "--models", help="Comma list of model names to run (defaults to all)"),
    baseline_blocks: str | None = typer.Option(
        "controls_demography_SES,health_history_engine", "--baseline-blocks", help="Comma list for LOBI S0"
    ),
    weight_column: str | None = typer.Option(None, "--weight-column", help="Optional survey weight column"),
    standardize: bool = typer.Option(True, "--standardize/--no-standardize", help="Standardise numeric predictors"),
    missingness_threshold: float = typer.Option(
        DEFAULT_MISSINGNESS_THRESHOLD,
        "--missingness-threshold",
        help="Max missingness fraction for predictor imputation + missing indicator.",
    ),
    max_workers: int = typer.Option(6, "--max-workers", min=0, help="Worker threads (0 = auto)"),
    log_level: str = typer.Option("INFO", "--log-level"),
) -> None:
    """Compute LOBO/LOBI block importance using statsmodels backbone specs."""

    _configure_logging(log_level)
    specs, all_specs, df, column_types, reference_levels, missingness_originals = _load_specs_and_data(
        input_csv=input_csv,
        column_types_path=column_types_path,
        derivations_path=derivations_path,
        models=models,
        weight_column=weight_column,
        standardize=standardize,
        missingness_threshold=missingness_threshold,
        load_all_spec_columns=True,
    )
    try:
        worker_count = _normalize_max_workers(max_workers)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    baseline = _parse_baseline_blocks(baseline_blocks) or []
    soaking_sources = _resolve_soaking_source_specs(specs, all_specs)
    _run_block_importance(
        specs=specs,
        df=df,
        column_types=column_types,
        reference_levels=reference_levels,
        weight_column=weight_column,
        baseline_blocks=baseline,
        outdir=outdir,
        max_workers=worker_count,
        missingness_originals=missingness_originals,
        soaking_source_specs=soaking_sources,
    )


@app.command("run")
def run_all(
    input_csv: Path = typer.Option(CSV_FILE_CLEAN, "--input", help="Input CLEAN CSV"),
    column_types_path: Path = typer.Option(DEFAULT_COLUMN_TYPES_PATH, "--column-types", help="Column type TSV"),
    derivations_path: Path | None = typer.Option(
        DEFAULT_DERIVATIONS_PATH, "--derivations", help="Derivations TSV to drop redundant predictors"
    ),
    outdir: Path = typer.Option(DEFAULT_OUTDIR, "--outdir", help="Output directory for fits"),
    models: str | None = typer.Option(None, "--models", help="Comma list of model names to run (defaults to all)"),
    baseline_blocks: str | None = typer.Option(
        "controls_demography_SES,health_history_engine", "--baseline-blocks", help="Comma list for LOBI S0"
    ),
    weight_column: str | None = typer.Option(None, "--weight-column", help="Optional survey weight column"),
    standardize: bool = typer.Option(True, "--standardize/--no-standardize", help="Standardise numeric predictors"),
    missingness_threshold: float = typer.Option(
        DEFAULT_MISSINGNESS_THRESHOLD,
        "--missingness-threshold",
        help="Max missingness fraction for predictor imputation + missing indicator.",
    ),
    max_workers: int = typer.Option(6, "--max-workers", min=0, help="Worker threads (0 = auto)"),
    log_level: str = typer.Option("INFO", "--log-level"),
) -> None:
    """Run fits and block-importance sequentially with shared data load."""

    _configure_logging(log_level)
    specs, all_specs, df, column_types, reference_levels, missingness_originals = _load_specs_and_data(
        input_csv=input_csv,
        column_types_path=column_types_path,
        derivations_path=derivations_path,
        models=models,
        weight_column=weight_column,
        standardize=standardize,
        missingness_threshold=missingness_threshold,
        load_all_spec_columns=True,
    )

    try:
        worker_count = _normalize_max_workers(max_workers)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    _run_fit(
        specs=specs,
        df=df,
        column_types=column_types,
        reference_levels=reference_levels,
        weight_column=weight_column,
        outdir=outdir,
        max_workers=worker_count,
        missingness_originals=missingness_originals,
    )

    baseline = _parse_baseline_blocks(baseline_blocks) or []
    soaking_sources = _resolve_soaking_source_specs(specs, all_specs)
    _run_block_importance(
        specs=specs,
        df=df,
        column_types=column_types,
        reference_levels=reference_levels,
        weight_column=weight_column,
        baseline_blocks=baseline,
        outdir=outdir,
        max_workers=worker_count,
        missingness_originals=missingness_originals,
        soaking_source_specs=soaking_sources,
    )


if __name__ == "__main__":
    app()
