"""Plot/table render helpers for chapter 4 (LLM micro-validation)."""

from __future__ import annotations

import json
import re
import textwrap
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Rectangle

from beam_abm.llm_microvalidation.psychological_profiles.reference_model_loading import default_ref_state_models
from utils.chapter_io import read_csv_fast
from utils.engine_labels import BELIEF_OUTCOME_ORDER, CHOICE_ENGINE_BLOCK_ALIASES, label_choice_engine_block
from utils.plot_style import apply_thesis_matplotlib_style

apply_thesis_matplotlib_style()

DEFAULT_BLOCK_COLOR_ORDER: list[str] = [
    "trust_engine",
    "disposition_engine",
    "stakes_engine",
    "legitimacy_engine",
    "norms_engine",
]

BLOCK_NORMALIZE_MAP: dict[str, str] = {
    **CHOICE_ENGINE_BLOCK_ALIASES,
}
BELIEF_PLOT_EXCLUDED_CONSTRUCTS: frozenset[str] = frozenset({"vaccine_risk_avg"})
PARETO_STRATEGY_ORDER: tuple[str, ...] = (
    "data_only",
    "third_person",
    "first_person",
    "multi_expert",
    "paired_profile",
)
PARETO_STRATEGY_COLOR_FIXED: dict[str, str] = {
    "data_only": "#0F766E",
    "third_person": "#C2410C",
    "first_person": "#1D4ED8",
    "multi_expert": "#B91C1C",
    "paired_profile": "#4D7C0F",
}
PARETO_STRATEGY_COLOR_FALLBACK: tuple[str, ...] = ("#A16207", "#7C3AED", "#0EA5E9", "#DC2626")
PARETO_MODEL_MARKER_BY_CANONICAL: dict[str, str] = {
    "gpt-oss-120b_medium": "o",
    "gpt-oss-120b_high": "D",
    "gpt-oss-20b_medium": "s",
    "gpt-oss-20b_high": "P",
    "qwen_qwen3-32b_medium": "^",
    "qwen_qwen3-32b_high": "X",
}
PARETO_MODEL_MARKER_FALLBACK: tuple[str, ...] = ("D", "v", "P", "X", "<", ">")

OUTCOME_CANONICAL_ALIASES: dict[str, str] = {
    "vax_willingness_Y": "vax_willingness_T12",
    "flu_vaccinated_2023_2024_no_habit": "flu_vaccinated_2023_2024",
}

OUTCOME_TOP_ROW_PREFERRED: tuple[str, ...] = (
    "vax_willingness_T12",
    "flu_vaccinated_2023_2024",
)


def canonical_outcome_name(outcome: str) -> str:
    raw = str(outcome).strip()
    return OUTCOME_CANONICAL_ALIASES.get(raw, raw)


def order_outcomes_with_preference(outcomes: Sequence[str], preferred_order: Sequence[str]) -> list[str]:
    observed = list(dict.fromkeys(canonical_outcome_name(str(o)) for o in outcomes if str(o).strip()))
    preferred = list(dict.fromkeys(canonical_outcome_name(str(o)) for o in preferred_order if str(o).strip()))
    observed_set = set(observed)
    ordered = [o for o in preferred if o in observed_set]
    ordered.extend([o for o in observed if o not in set(ordered)])
    return ordered


def outcome_panel_layout(outcome_order: Sequence[str]) -> tuple[int, int, list[str | None]]:
    outcomes = list(dict.fromkeys(str(o) for o in outcome_order if str(o).strip()))
    if not outcomes:
        return 1, 1, [None]
    if len(outcomes) == 1:
        return 1, 1, [outcomes[0]]

    top_row: list[str] = []
    for preferred in OUTCOME_TOP_ROW_PREFERRED:
        match = next(
            (
                outcome
                for outcome in outcomes
                if canonical_outcome_name(outcome) == preferred and outcome not in top_row
            ),
            None,
        )
        if match is not None:
            top_row.append(match)
    bottom_row = [outcome for outcome in outcomes if outcome not in top_row]

    if len(top_row) == len(OUTCOME_TOP_ROW_PREFERRED) and bottom_row:
        nrows = 2
        ncols = max(len(bottom_row), len(top_row) + 1)
        slots: list[str | None] = [None] * (nrows * ncols)
        for idx, outcome in enumerate(top_row):
            slots[idx] = outcome
        for idx, outcome in enumerate(bottom_row):
            slots[ncols + idx] = outcome
        return nrows, ncols, slots

    nrows = 2
    ncols = int(np.ceil(len(outcomes) / nrows))
    slots = list(outcomes) + [None] * (nrows * ncols - len(outcomes))
    return nrows, ncols, slots


def rankdata_average_ties(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    sorted_vals = values[order]
    ranks = np.empty(values.shape[0], dtype=np.float64)

    i = 0
    n = values.shape[0]
    while i < n:
        j = i + 1
        while j < n and sorted_vals[j] == sorted_vals[i]:
            j += 1
        avg = 0.5 * (i + (j - 1)) + 1.0
        ranks[order[i:j]] = avg
        i = j

    return ranks


def spearman_np(x: np.ndarray, y: np.ndarray) -> float:
    if x.size == 0 or y.size == 0 or x.size != y.size:
        return float("nan")
    rx = rankdata_average_ties(x)
    ry = rankdata_average_ties(y)
    sx = rx.std(ddof=1)
    sy = ry.std(ddof=1)
    if not np.isfinite(sx) or not np.isfinite(sy) or sx <= 0 or sy <= 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def gini_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size == 0 or y_pred.size == 0 or y_true.size != y_pred.size:
        return float("nan")
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    n = int(y_true.size)
    if n < 2:
        return float("nan")

    unique_preds = np.unique(y_pred)
    pred_ranks = np.searchsorted(unique_preds, y_pred) + 1
    m = int(unique_preds.size)

    order = np.argsort(y_true, kind="mergesort")
    y_true = y_true[order]
    pred_ranks = pred_ranks[order]

    tree = np.zeros(m + 2, dtype=np.int64)

    def _add(idx: int, value: int) -> None:
        while idx < tree.size:
            tree[idx] += value
            idx += idx & -idx

    def _sum(idx: int) -> int:
        s = 0
        while idx > 0:
            s += tree[idx]
            idx -= idx & -idx
        return s

    total_pairs = 0
    concordant = 0
    ties = 0

    i = 0
    while i < n:
        j = i + 1
        while j < n and y_true[j] == y_true[i]:
            j += 1
        total_prev = _sum(m)
        for k in range(i, j):
            rank = int(pred_ranks[k])
            less = _sum(rank - 1)
            equal = _sum(rank) - less
            concordant += less
            ties += equal
            total_pairs += total_prev
        for k in range(i, j):
            _add(int(pred_ranks[k]), 1)
        i = j

    if total_pairs == 0:
        return float("nan")
    cindex = float((concordant + 0.5 * ties) / total_pairs)
    return float(2.0 * cindex - 1.0)


def prediction_path(
    *,
    unperturbed_out: Path,
    outcome: str,
    model: str,
    strategy: str,
    calibrated: bool,
) -> Path:
    sub = "calibrated" if calibrated else "uncalibrated"
    fname = "predictions_calibrated.csv" if calibrated else "predictions.csv"
    return unperturbed_out / outcome / model / strategy / sub / fname


def load_preds(
    *,
    read_csv: Callable[[str | Path], pl.DataFrame],
    unperturbed_out: Path,
    outcome: str,
    model: str,
    strategy: str,
    calibrated: bool,
) -> pl.DataFrame:
    path = prediction_path(
        unperturbed_out=unperturbed_out,
        outcome=outcome,
        model=model,
        strategy=strategy,
        calibrated=calibrated,
    )
    df = read_csv(path)
    return (
        df.select("dataset_id", "country", "y_true", "y_pred")
        .with_columns(
            pl.col("y_true").cast(pl.Float64),
            pl.col("y_pred").cast(pl.Float64),
        )
        .with_columns((pl.col("y_pred") - pl.col("y_true")).alias("err"))
    )


def load_cell_arrays(
    *,
    read_csv: Callable[[str | Path], pl.DataFrame],
    unperturbed_out: Path,
    outcome: str,
    model: str,
    strategy: str,
) -> dict[str, np.ndarray] | None:
    raw_path = prediction_path(
        unperturbed_out=unperturbed_out,
        outcome=outcome,
        model=model,
        strategy=strategy,
        calibrated=False,
    )
    cal_path = prediction_path(
        unperturbed_out=unperturbed_out,
        outcome=outcome,
        model=model,
        strategy=strategy,
        calibrated=True,
    )
    if not raw_path.exists() or not cal_path.exists():
        return None

    raw = read_csv(raw_path).select("dataset_id", "y_true", "y_pred").rename({"y_pred": "y_pred_raw"})
    cal_df = read_csv(cal_path)
    cal_col = "y_pred_cal" if "y_pred_cal" in cal_df.columns else "y_pred"
    cal = cal_df.select("dataset_id", pl.col(cal_col).alias("y_pred_cal"))
    df = raw.join(cal, on="dataset_id", how="inner").sort("dataset_id")
    if df.is_empty():
        return None

    def _base_profile_id(value: str) -> str:
        parts = str(value).split("__")
        if len(parts) == 4:
            return "__".join([parts[0], parts[2]])
        if len(parts) == 3:
            return "__".join([parts[0], parts[1]])
        if len(parts) == 2:
            return "__".join(parts)
        return str(value)

    df = df.select(
        pl.col("dataset_id").cast(pl.Utf8),
        pl.col("dataset_id").map_elements(_base_profile_id, return_dtype=pl.Utf8).alias("dataset_id_base"),
        pl.col("y_true").cast(pl.Float64),
        pl.col("y_pred_raw").cast(pl.Float64),
        pl.col("y_pred_cal").cast(pl.Float64),
    )
    return {
        "dataset_id": df.get_column("dataset_id").to_numpy(),
        "dataset_id_base": df.get_column("dataset_id_base").to_numpy(),
        "y_true": df.get_column("y_true").to_numpy(),
        "y_pred_raw": df.get_column("y_pred_raw").to_numpy(),
        "y_pred_cal": df.get_column("y_pred_cal").to_numpy(),
    }


def mae_np(x: np.ndarray, y: np.ndarray) -> float:
    if x.size == 0 or x.size != y.size:
        return float("nan")
    return float(np.mean(np.abs(x - y)))


def smae_cal_np(y_true: np.ndarray, y_pred_cal: np.ndarray) -> float:
    sd = float(np.std(y_true, ddof=1))
    if not np.isfinite(sd) or sd <= 0:
        return float("nan")
    return mae_np(y_true, y_pred_cal) / sd


def mae_skill_cal_np(y_true: np.ndarray, y_pred_cal: np.ndarray) -> float:
    mask = np.isfinite(y_true) & np.isfinite(y_pred_cal)
    if not np.any(mask):
        return float("nan")
    yt = y_true[mask]
    yp = y_pred_cal[mask]
    mae = float(np.mean(np.abs(yp - yt)))
    baseline = float(np.median(yt))
    baseline_mae = float(np.mean(np.abs(yt - baseline)))
    if baseline_mae <= 0:
        return float("nan")
    return float(1.0 - mae / baseline_mae)


def _robust_location_scale(values: np.ndarray) -> tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan")
    loc = float(np.median(arr))
    q25, q75 = np.quantile(arr, [0.25, 0.75])
    scale = float(q75 - q25)
    if not np.isfinite(scale) or scale <= 1e-12:
        scale = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
    return loc, scale


def _map_first_person_to_third_person_scale(
    *,
    strategy: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_third_person: np.ndarray | None,
) -> np.ndarray:
    """Post-hoc affine calibration for first-person predictions.

    This keeps first-person heterogeneity (ordering/spread structure) while
    aligning its location/scale to the third-person baseline for the same
    model/engine.
    """
    yp = np.asarray(y_pred, dtype=np.float64).copy()
    if str(strategy) != "first_person" or y_pred_third_person is None:
        return yp

    ref = np.asarray(y_pred_third_person, dtype=np.float64)
    loc_fp, scale_fp = _robust_location_scale(yp)
    loc_tp, scale_tp = _robust_location_scale(ref)
    if not all(np.isfinite(v) for v in (loc_fp, scale_fp, loc_tp, scale_tp)):
        return yp
    if scale_fp <= 1e-12 or scale_tp <= 1e-12:
        return yp

    mask = np.isfinite(yp)
    if not np.any(mask):
        return yp
    yp[mask] = loc_tp + (scale_tp / scale_fp) * (yp[mask] - loc_fp)

    # Keep calibrated predictions on observed engine support.
    yt = np.asarray(y_true, dtype=np.float64)
    tmask = np.isfinite(yt)
    if np.any(tmask):
        lo = float(np.min(yt[tmask]))
        hi = float(np.max(yt[tmask]))
        yp[mask] = np.clip(yp[mask], lo, hi)
    return yp


def quantile_ci(arr: np.ndarray, alpha: float = 0.05) -> tuple[float, float] | tuple[None, None]:
    a = arr[np.isfinite(arr)]
    if a.size == 0:
        return (None, None)
    lo = float(np.quantile(a, alpha / 2.0))
    hi = float(np.quantile(a, 1.0 - alpha / 2.0))
    return (lo, hi)


def extract_row_id(dataset_id: object, *, normalize_choice_dataset_id: Callable[[str], str | None]) -> int | None:
    s = str(dataset_id).strip()
    if not s:
        return None
    norm = normalize_choice_dataset_id(s)
    if isinstance(norm, str) and norm:
        s = norm
    if "__" not in s:
        return None
    parts = s.split("__")
    if len(parts) >= 2 and parts[1].isdigit():
        return int(parts[1])
    return None


def iter_ref_prediction_paths(unperturbed_out: Path, outcome: str) -> list[Path]:
    outcome_dir = unperturbed_out / outcome
    if not outcome_dir.exists():
        return []
    out: list[Path] = []
    for p in outcome_dir.rglob("predictions.csv"):
        if p.parent.name != "uncalibrated":
            continue
        if any(part in {"_summary", "_runs", "_locks"} for part in p.parts):
            continue
        out.append(p)
    return sorted(out)


def load_model_plan_entries(model_plan_path: Path) -> list[dict[str, object]]:
    raw = json.loads(model_plan_path.read_text(encoding="utf-8"))
    if isinstance(raw, list):
        return [x for x in raw if isinstance(x, dict)]
    if isinstance(raw, dict):
        models = raw.get("models") or raw.get("plan")
        if isinstance(models, list):
            return [x for x in models if isinstance(x, dict)]
        if isinstance(models, dict):
            return [x for x in models.values() if isinstance(x, dict)]
        return [raw]
    return []


def format_ci(
    value: float,
    lo: float | None,
    hi: float | None,
    *,
    digits: int,
    fmt_float_fn: Callable[[float, int], str],
) -> str:
    if not np.isfinite(value) or lo is None or hi is None:
        return "—"
    return f"{fmt_float_fn(value, digits)} [{fmt_float_fn(lo, digits)}, {fmt_float_fn(hi, digits)}]"


def cell_point(cell_metrics: pl.DataFrame, outcome: str, model: str, strategy: str, col: str) -> float:
    match = cell_metrics.filter(
        (pl.col("outcome") == outcome) & (pl.col("model") == model) & (pl.col("strategy") == strategy)
    ).select(col)
    if match.is_empty():
        return float("nan")
    v = match.row(0)[0]
    return float(v) if v is not None else float("nan")


def best_cell_tuple(
    cell_metrics: pl.DataFrame,
    outcome: str,
    col: str,
    *,
    higher_is_better: bool,
) -> tuple[str, str] | None:
    df = cell_metrics.filter(pl.col("outcome") == outcome)
    if df.is_empty():
        return None
    df = df.filter(pl.col(col).is_finite())
    if df.is_empty():
        return None
    row = df.sort(col, descending=higher_is_better).row(0, named=True)
    return (str(row["model"]), str(row["strategy"]))


def finite_values(df: pl.DataFrame, col: str) -> np.ndarray:
    if col not in df.columns or df.is_empty():
        return np.array([], dtype=np.float64)
    vals = df.select(pl.col(col).cast(pl.Float64)).to_series().to_numpy()
    vals = vals[np.isfinite(vals)]
    return vals.astype(np.float64, copy=False)


def shared_metric_range(
    *,
    unperturbed_df: pl.DataFrame,
    unperturbed_col: str,
    perturbed_summary_path: Path,
    perturbed_col: str,
    perturbed_model: str,
    perturbed_strategy: str,
    csv_reader: Callable[[str | Path], pl.DataFrame],
    default: tuple[float, float] = (0.0, 1.0),
) -> tuple[float, float]:
    vals = finite_values(unperturbed_df, unperturbed_col)
    if perturbed_summary_path.exists():
        pert = csv_reader(perturbed_summary_path)
        if {"model", "strategy"}.issubset(set(pert.columns)):
            pert = pert.filter(
                (pl.col("model").cast(pl.Utf8) == perturbed_model)
                & (pl.col("strategy").cast(pl.Utf8) == perturbed_strategy)
            )
        vals_pert = finite_values(pert, perturbed_col)
        if vals.size and vals_pert.size:
            vals = np.concatenate([vals, vals_pert])
        elif vals_pert.size:
            vals = vals_pert
    if vals.size:
        vmin = float(np.quantile(vals, 0.02))
        vmax = float(np.quantile(vals, 0.98))
        if vmin < vmax:
            return vmin, vmax
        vmin = float(np.min(vals))
        vmax = float(np.max(vals))
        if vmin < vmax:
            return vmin, vmax
    return default


def load_lever_config(lever_config_path: Path) -> dict[str, dict[str, object]]:
    if not lever_config_path.exists():
        return {}
    try:
        raw = json.loads(lever_config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(raw, dict):
        return {}
    out: dict[str, dict[str, object]] = {}
    for outcome, payload in raw.items():
        if isinstance(outcome, str) and isinstance(payload, dict):
            out[outcome] = payload
    return out


def discover_keep_outcomes(perturbed_out: Path, outcome_order: Sequence[str]) -> list[str]:
    if perturbed_out.exists():
        discovered = sorted(
            child.name for child in perturbed_out.iterdir() if child.is_dir() and not child.name.startswith("_")
        )
        if discovered:
            return order_outcomes_with_preference(discovered, [str(o) for o in outcome_order])
    return order_outcomes_with_preference([str(o) for o in outcome_order], [str(o) for o in outcome_order])


def infer_model_slug(summary_path: Path) -> str | None:
    run_meta = summary_path.parent / "run_metadata.json"
    if run_meta.exists():
        try:
            payload = json.loads(run_meta.read_text())
        except Exception:
            payload = {}
        config = payload.get("config_snapshot", {}) if isinstance(payload, dict) else {}
        model_slug = config.get("model_slug")
        if isinstance(model_slug, str) and model_slug.strip():
            return model_slug.strip()

    canonical_meta = summary_path.parent.parent / "canonical_metadata.json"
    if canonical_meta.exists():
        try:
            payload = json.loads(canonical_meta.read_text())
        except Exception:
            payload = {}
        if isinstance(payload, dict):
            source_leaf = payload.get("source_leaf")
            if isinstance(source_leaf, str) and source_leaf.strip():
                leaf = Path(source_leaf)
                if len(leaf.parts) >= 3:
                    model = leaf.parts[-3]
                    if model.startswith("openai_"):
                        model = model[len("openai_") :]
                    return model
    return None


def normalize_model_slug(raw: str | None) -> str:
    if raw is None:
        return ""
    base = str(raw).strip().lower().replace("/", "_")
    for prefix in ("openai_", "openai-"):
        if base.startswith(prefix):
            base = base[len(prefix) :]
            break
    for suffix in ("_low", "_medium", "_high", "-low", "-medium", "-high", " low", " medium", " high"):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
            break
    return base


def canonical_model_id_with_effort(raw: str | None) -> str:
    """Canonicalize model id and ensure explicit effort suffix for display."""
    base = normalize_model_slug(raw)
    if not base:
        return ""
    lowered = base.lower()
    for suffix in ("_low", "_medium", "_high"):
        if lowered.endswith(suffix):
            return lowered

    raw_str = str(raw).strip().lower()
    for suffix, normalized in (
        ("_low", "_low"),
        ("_medium", "_medium"),
        ("_high", "_high"),
        ("-low", "_low"),
        (" low", "_low"),
        ("-medium", "_medium"),
        (" medium", "_medium"),
        ("-high", "_high"),
        (" high", "_high"),
    ):
        if raw_str.endswith(suffix):
            out = lowered + normalized
            if out == "gpt-oss-120b_high":
                return "gpt-oss-120b_medium"
            return out
    out = lowered + "_medium"
    if out == "gpt-oss-120b_high":
        return "gpt-oss-120b_medium"
    return out


def resolve_model_id_with_effort(*, inferred_slug: str | None, fallback_dir_slug: str | None) -> str:
    """Resolve canonical model id while preserving explicit effort when available."""
    inferred_canonical = canonical_model_id_with_effort(inferred_slug)
    fallback_canonical = canonical_model_id_with_effort(fallback_dir_slug)
    if not inferred_canonical:
        return fallback_canonical
    if not fallback_canonical:
        return inferred_canonical

    # If metadata collapses effort to medium but directory is explicit high/low,
    # trust the explicit directory token.
    if inferred_canonical.endswith("_medium") and (
        fallback_canonical.endswith("_high") or fallback_canonical.endswith("_low")
    ):
        return fallback_canonical
    return inferred_canonical


def as_float(df: pl.DataFrame, cols: list[str]) -> pl.DataFrame:
    out = df
    for c in cols:
        if c in out.columns:
            out = out.with_columns(pl.col(c).cast(pl.Float64))
    return out


def discover_pert_models(df: pl.DataFrame) -> list[str]:
    if df.is_empty() or "model" not in df.columns:
        return []
    models = sorted({str(m) for m in df.get_column("model").to_list()})

    def _key(m: str) -> tuple[str, int, str]:
        raw = str(m)
        effort_rank = 1
        base = raw
        for suffix, rank in (("_low", 0), ("_medium", 1), ("_high", 2)):
            if raw.endswith(suffix):
                effort_rank = rank
                base = raw[: -len(suffix)]
                break
        return (base, effort_rank, raw)

    return sorted(models, key=_key)


def _build_pareto_strategy_color_map(strategy_order: Sequence[str]) -> dict[str, str]:
    color_map: dict[str, str] = {}
    for strategy in [str(s) for s in strategy_order]:
        if strategy in PARETO_STRATEGY_COLOR_FIXED:
            color_map[strategy] = PARETO_STRATEGY_COLOR_FIXED[strategy]
            continue
        for color in PARETO_STRATEGY_COLOR_FALLBACK:
            if color not in set(color_map.values()):
                color_map[strategy] = color
                break
        else:
            color_map[strategy] = PARETO_STRATEGY_COLOR_FALLBACK[0]
    return color_map


def _build_pareto_model_marker_map(model_order: Sequence[str]) -> dict[str, str]:
    marker_map: dict[str, str] = {}
    used: set[str] = set()
    for model in [str(m) for m in model_order]:
        canonical = canonical_model_id_with_effort(model)
        fixed = PARETO_MODEL_MARKER_BY_CANONICAL.get(canonical)
        if fixed is not None:
            marker_map[model] = fixed
            used.add(fixed)
            continue
        assigned: str | None = None
        for marker in PARETO_MODEL_MARKER_FALLBACK:
            if marker not in used:
                assigned = marker
                break
        if assigned is None:
            assigned = PARETO_MODEL_MARKER_FALLBACK[0]
        marker_map[model] = assigned
        used.add(assigned)
    return marker_map


def _model_effort_rank(model_id: str) -> int:
    raw = str(model_id).strip().lower()
    if raw.endswith("_medium"):
        return 0
    if raw.endswith("_high"):
        return 1
    if raw.endswith("_low"):
        return 2
    return 3


def _sort_models_with_effort_preference(models: Sequence[str]) -> list[str]:
    return sorted(
        [str(m) for m in models if str(m).strip()],
        key=lambda m: (_model_effort_rank(m), str(m)),
    )


def _preferred_models_present(
    *,
    available_models: Sequence[str],
    preferred_models: Sequence[str],
) -> list[str]:
    available = [str(m) for m in available_models if str(m).strip()]
    if not available:
        return []
    preferred_norm = {
        normalize_model_slug(canonical_model_id_with_effort(str(m))) for m in preferred_models if str(m).strip()
    }
    if not preferred_norm:
        return []
    matched = [m for m in available if normalize_model_slug(m) in preferred_norm]
    return _sort_models_with_effort_preference(matched)


def _pick_primary_perturbed_model(
    *,
    available_models: Sequence[str],
    preferred_models: Sequence[str],
    fallback_models: Sequence[str],
) -> str | None:
    available = [str(m) for m in available_models if str(m).strip()]
    if not available:
        return None
    preferred_present = _preferred_models_present(
        available_models=available,
        preferred_models=preferred_models,
    )
    if preferred_present:
        return preferred_present[0]

    fallback_present = [str(m) for m in fallback_models if str(m) in set(available)]
    fallback_present = _sort_models_with_effort_preference(fallback_present)
    if fallback_present:
        return fallback_present[0]
    return _sort_models_with_effort_preference(available)[0]


def _candidate_model_dirs_for_outcome(
    *,
    perturbed_out: Path,
    outcome: str,
    model_id: str,
) -> list[Path]:
    outcome_root = perturbed_out / str(outcome)
    if not outcome_root.exists():
        return []
    dir_paths = [p for p in outcome_root.iterdir() if p.is_dir() and not p.name.startswith("_")]
    if not dir_paths:
        return []

    desired_id = canonical_model_id_with_effort(model_id)
    desired_base = normalize_model_slug(desired_id or model_id)
    desired_rank = _model_effort_rank(desired_id or model_id)
    scored: list[tuple[int, int, str, Path]] = []
    for dir_path in dir_paths:
        dir_id = canonical_model_id_with_effort(dir_path.name)
        if normalize_model_slug(dir_id) != desired_base:
            continue
        dir_rank = _model_effort_rank(dir_id)
        scored.append((abs(dir_rank - desired_rank), dir_rank, dir_path.name, dir_path))
    scored.sort(key=lambda item: (item[0], item[1], item[2]))
    return [path for _, _, _, path in scored]


def label_strategy_pert(strategy: str, labels: Mapping[str, str]) -> str:
    raw = str(strategy)
    if raw in labels:
        return str(labels[raw])
    return raw.replace("_", " ").strip().title() or raw


def label_block(block: str, labels: Mapping[str, str]) -> str:
    canonical = str(block).strip()
    canonical = BLOCK_NORMALIZE_MAP.get(canonical, canonical)
    if canonical in labels:
        return str(labels[canonical])
    return canonical.replace("_", " ").strip().title() or canonical


def _resolve_lever_payload(
    *,
    outcome: str,
    lever_config: Mapping[str, dict[str, object]],
) -> dict[str, object] | None:
    """Return the lever-config payload for an outcome with local overrides."""
    key = resolve_lever_outcome_key(outcome=outcome, lever_config=lever_config)
    payload = lever_config.get(key)
    return payload if isinstance(payload, dict) else None


def resolve_lever_outcome_key(
    *,
    outcome: str,
    lever_config: Mapping[str, dict[str, object]],
) -> str:
    """Return the effective lever-config key for an outcome."""
    key = str(outcome).strip()
    if key == "flu_vaccinated_2023_2024":
        alt_key = "flu_vaccinated_2023_2024_no_habit"
        if isinstance(lever_config.get(alt_key), dict):
            return alt_key
    return key


def resolve_ref_outcome_dir_key(
    *,
    outcome: str,
    lever_config: Mapping[str, dict[str, object]],
    ref_ice_root: Path,
) -> str:
    """Resolve the best reference ICE outcome directory for an outcome."""
    preferred = resolve_lever_outcome_key(outcome=outcome, lever_config=lever_config)
    fallback = str(outcome).strip()
    for key in dict.fromkeys([preferred, fallback]):
        if key and (ref_ice_root / key).exists():
            return key
    return preferred or fallback


def find_ref_model_for_outcome(
    *,
    outcome: str,
    lever_config: Mapping[str, dict[str, object]],
    ref_ice_root: Path,
) -> str | None:
    payload = _resolve_lever_payload(outcome=outcome, lever_config=lever_config)
    if payload is not None:
        model_ref = payload.get("model_ref")
        if isinstance(model_ref, str) and model_ref.strip():
            return model_ref.strip()

    model_names: list[str] = []
    preferred_key = resolve_lever_outcome_key(outcome=outcome, lever_config=lever_config)
    for outcome_key in dict.fromkeys([preferred_key, str(outcome).strip()]):
        outcome_dir = ref_ice_root / outcome_key
        if not outcome_dir.exists():
            continue
        candidates = sorted(outcome_dir.glob("ice_ref__*.csv"))
        model_names = [p.stem.replace("ice_ref__", "") for p in candidates]
        for name in model_names:
            if name.lower() != "xgb":
                return name
    return model_names[0] if model_names else None


def canonical_levers_for(
    *,
    outcome: str,
    lever_config: Mapping[str, dict[str, object]],
    reference_model_by_outcome: Mapping[str, str],
    ref_ice_root: Path,
    csv_reader: Callable[[str | Path], pl.DataFrame],
) -> list[str]:
    payload = _resolve_lever_payload(outcome=outcome, lever_config=lever_config)
    if payload is not None:
        levers = payload.get("levers")
        if isinstance(levers, list):
            vals = [str(v).strip() for v in levers if isinstance(v, str) and str(v).strip()]
            if vals:
                return vals

    model_ref = reference_model_by_outcome.get(outcome)
    if not model_ref:
        return []
    ref_path = ref_ice_root / outcome / f"ice_ref__{model_ref}.csv"
    if not ref_path.exists():
        return []
    try:
        ref = csv_reader(ref_path)
    except Exception:
        return []
    if ref.is_empty() or "target" not in ref.columns:
        return []
    return sorted({str(v) for v in ref.get_column("target").drop_nulls().to_list()})


def discover_outcomes(unperturbed_out: Path, default_outcome_order: Sequence[str]) -> list[str]:
    if not unperturbed_out.exists():
        return list(default_outcome_order)
    discovered = sorted(
        child.name for child in unperturbed_out.iterdir() if child.is_dir() and not child.name.startswith("_")
    )
    return discovered or list(default_outcome_order)


def discover_models(unperturbed_out: Path, outcome_order: Sequence[str]) -> list[str]:
    models: set[str] = set()
    for outcome in outcome_order:
        root = unperturbed_out / str(outcome)
        if not root.exists():
            continue
        for child in root.iterdir():
            if child.is_dir() and not child.name.startswith("_"):
                models.add(child.name)
    return sorted(models)


def prepare_unperturbed_context(
    *,
    unperturbed_out: Path,
    default_outcome_order: Sequence[str],
    perturbed_summary_path: Path,
    strategy_order: Sequence[str] | None = None,
    outcome_panel_w: float = 4.2,
    read_csv: Callable[[str | Path], pl.DataFrame] = read_csv_fast,
) -> UnperturbedContext:
    if strategy_order is None:
        strategy_order = ["data_only", "third_person", "first_person", "multi_expert"]

    def _canonical_outcome(outcome: str) -> str:
        return canonical_outcome_name(outcome)

    preferred_raw_outcome_candidates: dict[str, tuple[str, ...]] = {
        "vax_willingness_T12": ("vax_willingness_T12", "vax_willingness_Y"),
        "flu_vaccinated_2023_2024": ("flu_vaccinated_2023_2024", "flu_vaccinated_2023_2024_no_habit"),
    }

    def _preferred_raw_outcome(canonical_outcome: str, available: Collection[str]) -> str:
        candidates = preferred_raw_outcome_candidates.get(canonical_outcome, (canonical_outcome,))
        for candidate in candidates:
            if candidate in available:
                return candidate
        return candidates[0]

    def _bootstrap_outcome_ci_from_predictions(
        raw_outcome: str,
        *,
        boot_b: int = 500,
        boot_seed: int = 20260201,
    ) -> dict[str, tuple[float | None, float | None]] | None:
        gini_parts: list[np.ndarray] = []
        skill_parts: list[np.ndarray] = []

        for mi, model in enumerate(model_order):
            for si, strategy in enumerate(strategy_order):
                uncal_path = unperturbed_out / raw_outcome / model / strategy / "uncalibrated" / "predictions.csv"
                cal_path = (
                    unperturbed_out / raw_outcome / model / strategy / "calibrated" / "predictions_calibrated.csv"
                )
                if not uncal_path.exists() or not cal_path.exists():
                    continue
                try:
                    df_uncal = pd.read_csv(uncal_path)
                    df_cal = pd.read_csv(cal_path)
                except Exception:
                    continue
                if df_uncal.empty or df_cal.empty:
                    continue

                yt = pd.to_numeric(df_uncal.get("y_true", pd.Series(dtype=float)), errors="coerce").to_numpy(
                    dtype=np.float64
                )
                yp_raw = pd.to_numeric(df_uncal.get("y_pred", pd.Series(dtype=float)), errors="coerce").to_numpy(
                    dtype=np.float64
                )
                cal_col = "y_pred_cal" if "y_pred_cal" in df_cal.columns else "y_pred"
                yp_cal = pd.to_numeric(df_cal.get(cal_col, pd.Series(dtype=float)), errors="coerce").to_numpy(
                    dtype=np.float64
                )
                n = min(yt.size, yp_raw.size, yp_cal.size)
                if n < 2:
                    continue
                yt = yt[:n]
                yp_raw = yp_raw[:n]
                yp_cal = yp_cal[:n]

                outcome_seed = sum((idx + 1) * ord(ch) for idx, ch in enumerate(raw_outcome))
                rng = np.random.default_rng(boot_seed + 10_000 * outcome_seed + 97 * mi + 7919 * si)
                boot_idx = rng.integers(0, n, size=(boot_b, n), dtype=np.int64)
                gini_bs = np.empty(boot_b, dtype=np.float64)
                skill_bs = np.empty(boot_b, dtype=np.float64)

                for b in range(boot_b):
                    idx = boot_idx[b]
                    yt_b = yt[idx]
                    # Match the stored posthoc `pooled_gini` convention used by the
                    # chapter summaries, even though the argument order is inverted
                    # relative to the reference-benchmark path.
                    gini_bs[b] = gini_np(yp_raw[idx], yt_b)

                    yp_cal_b = yp_cal[idx]
                    mask = np.isfinite(yt_b) & np.isfinite(yp_cal_b)
                    if np.any(mask):
                        mae = float(np.mean(np.abs(yp_cal_b[mask] - yt_b[mask])))
                        baseline = float(np.median(yt_b[mask]))
                        baseline_mae = float(np.mean(np.abs(yt_b[mask] - baseline)))
                        skill_bs[b] = float(1.0 - mae / baseline_mae) if baseline_mae > 0 else float("nan")
                    else:
                        skill_bs[b] = float("nan")

                gini_parts.append(gini_bs)
                skill_parts.append(skill_bs)

        if not gini_parts or not skill_parts:
            return None

        mean_gini = np.nanmean(np.vstack(gini_parts), axis=0)
        mean_skill = np.nanmean(np.vstack(skill_parts), axis=0)
        g_vals = mean_gini[np.isfinite(mean_gini)]
        s_vals = mean_skill[np.isfinite(mean_skill)]
        return {
            "gini": (
                float(np.quantile(g_vals, 0.025)) if g_vals.size else None,
                float(np.quantile(g_vals, 0.975)) if g_vals.size else None,
            ),
            "skill": (
                float(np.quantile(s_vals, 0.025)) if s_vals.size else None,
                float(np.quantile(s_vals, 0.975)) if s_vals.size else None,
            ),
        }

    outcome_order_raw = discover_outcomes(unperturbed_out, default_outcome_order)
    outcome_order = order_outcomes_with_preference(
        [_canonical_outcome(outcome) for outcome in outcome_order_raw],
        [_canonical_outcome(outcome) for outcome in default_outcome_order],
    )
    model_order = discover_models(unperturbed_out, outcome_order_raw)

    posthoc_all_candidates = (
        unperturbed_out / "posthoc_metrics_all.csv",
        unperturbed_out / "_summary" / "posthoc_metrics_all.csv",
    )
    posthoc_all_path = next((p for p in posthoc_all_candidates if p.exists()), posthoc_all_candidates[0])
    if posthoc_all_path.exists():
        posthoc_all = read_csv(posthoc_all_path)
    else:
        posthoc_all = pl.DataFrame()
    available_posthoc_outcomes: set[str] = set()

    if posthoc_all.is_empty():
        cell_metrics = pl.DataFrame(
            schema={
                "outcome": pl.Utf8,
                "model": pl.Utf8,
                "strategy": pl.Utf8,
                "n": pl.Int64,
                "gini": pl.Float64,
                "skill_cal": pl.Float64,
            }
        )
    else:
        available_posthoc_outcomes = {
            str(value) for value in posthoc_all.get_column("outcome").drop_nulls().to_list() if str(value).strip()
        }
        uncal = posthoc_all.filter(pl.col("calibration_status") == "uncalibrated").select(
            "outcome",
            "model",
            "strategy",
            "n",
            "pooled_gini",
        )
        cal = posthoc_all.filter(pl.col("calibration_status") == "calibrated").select(
            "outcome",
            "model",
            "strategy",
            "mae_skill_cal",
        )
        raw_cell_metrics = (
            uncal.join(cal, on=["outcome", "model", "strategy"], how="left")
            .rename({"pooled_gini": "gini", "mae_skill_cal": "skill_cal"})
            .with_columns(
                pl.col("outcome").cast(pl.Utf8),
                pl.col("gini").cast(pl.Float64),
                pl.col("skill_cal").cast(pl.Float64),
                pl.col("n").cast(pl.Int64),
            )
            .group_by(["outcome", "model", "strategy"])
            .agg(
                pl.mean("n").round(0).cast(pl.Int64).alias("n"),
                pl.mean("gini").alias("gini"),
                pl.mean("skill_cal").alias("skill_cal"),
            )
        )
        selected_metrics: list[pl.DataFrame] = []
        for outcome in outcome_order:
            raw_outcome = _preferred_raw_outcome(outcome, available_posthoc_outcomes)
            selected = raw_cell_metrics.filter(pl.col("outcome") == raw_outcome)
            if selected.is_empty():
                continue
            selected_metrics.append(
                selected.with_columns(pl.lit(outcome).alias("outcome")).select(
                    "outcome",
                    "model",
                    "strategy",
                    "n",
                    "gini",
                    "skill_cal",
                )
            )
        cell_metrics = (
            pl.concat(selected_metrics, how="vertical")
            if selected_metrics
            else pl.DataFrame(
                schema={
                    "outcome": pl.Utf8,
                    "model": pl.Utf8,
                    "strategy": pl.Utf8,
                    "n": pl.Int64,
                    "gini": pl.Float64,
                    "skill_cal": pl.Float64,
                }
            )
        )

    ref_benchmark_candidates = (
        unperturbed_out / "ref_benchmark.csv",
        unperturbed_out / "_summary" / "ref_benchmark.csv",
    )
    ref_benchmark_path = next((p for p in ref_benchmark_candidates if p.exists()), ref_benchmark_candidates[0])
    ref_benchmark = read_csv(ref_benchmark_path) if ref_benchmark_path.exists() else pl.DataFrame()

    ref_gini_by_outcome: dict[str, float] = {}
    ref_ci_by_outcome: dict[str, tuple[float | None, float | None]] = {}
    ref_skill_by_outcome: dict[str, float] = {}
    ref_skill_ci_by_outcome: dict[str, tuple[float | None, float | None]] = {}

    def _safe_float_or_nan(value: object) -> float:
        if value is None:
            return float("nan")
        try:
            out = float(value)
        except (TypeError, ValueError):
            return float("nan")
        return out if np.isfinite(out) else float("nan")

    if not ref_benchmark.is_empty():
        ref_rows_by_canonical: dict[str, list[tuple[str, dict[str, object]]]] = {}
        for row in ref_benchmark.iter_rows(named=True):
            raw_outcome = str(row.get("outcome", "")).strip()
            outcome = _canonical_outcome(raw_outcome)
            if not outcome:
                continue
            ref_rows_by_canonical.setdefault(outcome, []).append((raw_outcome, row))

        def _ref_row_priority(outcome: str, raw_outcome: str, row: dict[str, object]) -> int:
            if outcome == "vax_willingness_T12":
                if raw_outcome == "vax_willingness_T12":
                    return 0
                if raw_outcome == "vax_willingness_Y":
                    return 1
            if outcome == "flu_vaccinated_2023_2024":
                if raw_outcome == "flu_vaccinated_2023_2024_no_habit":
                    return 0
                ref_model = str(row.get("ref_model", "")).upper()
                if ref_model and "HABIT" not in ref_model:
                    return 1
                if raw_outcome == "flu_vaccinated_2023_2024":
                    return 2
            return 10

        def _has_finite_ref_metrics(row: dict[str, object]) -> bool:
            g_pred = _safe_float_or_nan(row.get("ref_gini_pred", float("nan")))
            if np.isfinite(g_pred):
                return True
            g_eta = _safe_float_or_nan(row.get("ref_gini_eta", float("nan")))
            return bool(np.isfinite(g_eta))

        for outcome, candidates in ref_rows_by_canonical.items():
            if not candidates:
                continue
            if outcome == "flu_vaccinated_2023_2024":
                strict_candidates = [item for item in candidates if _ref_row_priority(outcome, item[0], item[1]) <= 1]
                if strict_candidates:
                    candidates = strict_candidates
                else:
                    ref_gini_by_outcome[outcome] = float("nan")
                    ref_skill_by_outcome[outcome] = float("nan")
                    ref_ci_by_outcome[outcome] = (None, None)
                    ref_skill_ci_by_outcome[outcome] = (None, None)
                    continue
            candidates_sorted = sorted(
                candidates,
                key=lambda item: (
                    0 if _has_finite_ref_metrics(item[1]) else 1,
                    _ref_row_priority(outcome, item[0], item[1]),
                    -int(float(item[1].get("n") or 0)),
                    item[0],
                ),
            )
            _, row = candidates_sorted[0]

            g = _safe_float_or_nan(row.get("ref_gini_pred", float("nan")))
            s = _safe_float_or_nan(row.get("ref_skill_pred", float("nan")))
            g_lo = row.get("ref_gini_pred_ci_low")
            g_hi = row.get("ref_gini_pred_ci_high")
            s_lo = row.get("ref_skill_pred_ci_low")
            s_hi = row.get("ref_skill_pred_ci_high")
            if not np.isfinite(g):
                g = _safe_float_or_nan(row.get("ref_gini_eta", float("nan")))
                g_lo, g_hi = (None, None)
            if not np.isfinite(s):
                s = float("nan")
                s_lo, s_hi = (None, None)
            ref_gini_by_outcome[outcome] = g
            ref_skill_by_outcome[outcome] = s
            ref_ci_by_outcome[outcome] = (
                _safe_float_or_nan(g_lo) if np.isfinite(_safe_float_or_nan(g_lo)) else None,
                _safe_float_or_nan(g_hi) if np.isfinite(_safe_float_or_nan(g_hi)) else None,
            )
            ref_skill_ci_by_outcome[outcome] = (
                _safe_float_or_nan(s_lo) if np.isfinite(_safe_float_or_nan(s_lo)) else None,
                _safe_float_or_nan(s_hi) if np.isfinite(_safe_float_or_nan(s_hi)) else None,
            )

    cell_boot_path = unperturbed_out / "_summary" / "cell_bootstrap_cis.csv"
    outcome_boot_path = unperturbed_out / "_summary" / "outcome_bootstrap_cis.csv"
    strategy_boot_path = unperturbed_out / "_summary" / "strategy_bootstrap_cis.csv"

    bootstrap_cell_ci: dict[tuple[str, str, str], dict[str, tuple[float | None, float | None]]] = {}
    if cell_boot_path.exists():
        cell_boot = read_csv(cell_boot_path)
        for row in cell_boot.iter_rows(named=True):
            key = (_canonical_outcome(str(row["outcome"])), str(row["model"]), str(row["strategy"]))
            bootstrap_cell_ci[key] = {
                "gini": (
                    float(row["gini_lo"]) if row.get("gini_lo") is not None else None,
                    float(row["gini_hi"]) if row.get("gini_hi") is not None else None,
                ),
                "skill_cal": (
                    float(row["skill_cal_lo"]) if row.get("skill_cal_lo") is not None else None,
                    float(row["skill_cal_hi"]) if row.get("skill_cal_hi") is not None else None,
                ),
            }

    bootstrap_outcome_ci: dict[str, dict[str, tuple[float | None, float | None]]] = {}
    if outcome_boot_path.exists():
        outcome_boot = read_csv(outcome_boot_path)
        outcome_boot_rows = {str(row["outcome"]): row for row in outcome_boot.iter_rows(named=True)}
        for outcome in outcome_order:
            raw_outcome = _preferred_raw_outcome(outcome, available_posthoc_outcomes if posthoc_all.height else [])
            row = outcome_boot_rows.get(raw_outcome)
            if row is not None:
                bootstrap_outcome_ci[outcome] = {
                    "gini": (
                        float(row["mean_gini_lo"]) if row.get("mean_gini_lo") is not None else None,
                        float(row["mean_gini_hi"]) if row.get("mean_gini_hi") is not None else None,
                    ),
                    "skill": (
                        float(row["mean_skill_lo"]) if row.get("mean_skill_lo") is not None else None,
                        float(row["mean_skill_hi"]) if row.get("mean_skill_hi") is not None else None,
                    ),
                }
                continue

            computed_ci = _bootstrap_outcome_ci_from_predictions(raw_outcome)
            if computed_ci is not None:
                bootstrap_outcome_ci[outcome] = computed_ci

    bootstrap_strategy_ci: dict[str, dict[str, tuple[float | None, float | None]]] = {}
    if strategy_boot_path.exists():
        strategy_boot = read_csv(strategy_boot_path)
        for row in strategy_boot.iter_rows(named=True):
            strategy = str(row["strategy"])
            bootstrap_strategy_ci[strategy] = {
                "gini": (
                    float(row["mean_gini_lo"]) if row.get("mean_gini_lo") is not None else None,
                    float(row["mean_gini_hi"]) if row.get("mean_gini_hi") is not None else None,
                ),
                "skill": (
                    float(row["mean_skill_lo"]) if row.get("mean_skill_lo") is not None else None,
                    float(row["mean_skill_hi"]) if row.get("mean_skill_hi") is not None else None,
                ),
            }

    overall_gini = (
        float(cell_metrics.select(pl.mean("gini")).row(0)[0])
        if (not cell_metrics.is_empty()) and ("gini" in cell_metrics.columns)
        else float("nan")
    )
    overall_skill = (
        float(cell_metrics.select(pl.mean("skill_cal")).row(0)[0])
        if (not cell_metrics.is_empty()) and ("skill_cal" in cell_metrics.columns)
        else float("nan")
    )

    mean_gini_by_outcome: dict[str, float] = {}
    for outcome in outcome_order:
        df_outcome = cell_metrics.filter(pl.col("outcome") == outcome)
        df_outcome = df_outcome.filter(pl.col("gini").is_finite())
        if df_outcome.is_empty():
            mean_gini_by_outcome[outcome] = float("nan")
            continue
        mean_gini_by_outcome[outcome] = float(df_outcome.select(pl.mean("gini")).row(0)[0])

    gini_vmin, gini_vmax = shared_metric_range(
        unperturbed_df=cell_metrics,
        unperturbed_col="gini",
        perturbed_summary_path=perturbed_summary_path,
        perturbed_col="metrics_gini_pe_std_mean",
        perturbed_model="gpt-oss-20b",
        perturbed_strategy="data_only",
        csv_reader=read_csv,
        default=(0.0, 1.0),
    )
    skill_vmin, skill_vmax = shared_metric_range(
        unperturbed_df=cell_metrics,
        unperturbed_col="skill_cal",
        perturbed_summary_path=perturbed_summary_path,
        perturbed_col="metrics_skill_pe_std_mean",
        perturbed_model="gpt-oss-20b",
        perturbed_strategy="data_only",
        csv_reader=read_csv,
        default=(0.0, 1.0),
    )

    return UnperturbedContext(
        unperturbed_out=unperturbed_out,
        strategy_order=list(strategy_order),
        outcome_panel_w=outcome_panel_w,
        outcome_order=outcome_order,
        model_order=model_order,
        cell_metrics=cell_metrics,
        ref_gini_by_outcome=ref_gini_by_outcome,
        ref_ci_by_outcome=ref_ci_by_outcome,
        ref_skill_by_outcome=ref_skill_by_outcome,
        ref_skill_ci_by_outcome=ref_skill_ci_by_outcome,
        bootstrap_cell_ci=bootstrap_cell_ci,
        bootstrap_outcome_ci=bootstrap_outcome_ci,
        bootstrap_strategy_ci=bootstrap_strategy_ci,
        overall_gini=overall_gini,
        overall_skill=overall_skill,
        mean_gini_by_outcome=mean_gini_by_outcome,
        gini_vmin=gini_vmin,
        gini_vmax=gini_vmax,
        skill_vmin=skill_vmin,
        skill_vmax=skill_vmax,
    )


def fisher_ci(rho: float, n: int, alpha: float = 0.05) -> tuple[float, float] | tuple[None, None]:
    del alpha
    if not np.isfinite(rho) or n is None or n < 4:
        return (None, None)
    rho = float(np.clip(rho, -0.999999, 0.999999))
    z = np.arctanh(rho)
    se = 1.0 / np.sqrt(n - 3)
    z_crit = 1.959963984540054
    lo = np.tanh(z - z_crit * se)
    hi = np.tanh(z + z_crit * se)
    return (float(lo), float(hi))


def heatmap_norm(vmin: float, vmax: float, *, vcenter: float = 0.0):
    return _heatmap_norm(vmin=vmin, vmax=vmax, vcenter=vcenter)


def text_color_for_value(
    value: float,
    *,
    cmap: plt.Colormap,
    vmin: float,
    vmax: float,
    vcenter: float | None = 0.0,
) -> str:
    return _text_color_for_value(value, cmap=cmap, vmin=vmin, vmax=vmax, vcenter=vcenter)


def _heatmap_norm(vmin: float, vmax: float, *, vcenter: float | None = 0.0):
    if not (np.isfinite(vmin) and np.isfinite(vmax)) or vmin >= vmax:
        return plt.Normalize(vmin=vmin, vmax=vmax, clip=True)
    if vcenter is not None and vmin < vcenter < vmax:
        return TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
    return plt.Normalize(vmin=vmin, vmax=vmax, clip=True)


def _text_color_for_value(
    value: float,
    *,
    cmap: plt.Colormap,
    vmin: float,
    vmax: float,
    vcenter: float | None = 0.0,
) -> str:
    if not np.isfinite(value):
        return "white"
    norm = _heatmap_norm(vmin=vmin, vmax=vmax, vcenter=vcenter)
    r, g, b, _ = cmap(norm(value))
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return "black" if luminance > 0.65 else "white"


def _heatmap_lever_panel(
    ax: plt.Axes,
    mat: np.ndarray,
    row_labels: list[str],
    col_labels: list[str],
    *,
    vmin: float,
    vmax: float,
    cmap: plt.Colormap,
    vcenter: float | None = 0.0,
):
    x_tick_size = 12
    y_tick_size = 12
    cell_text_size = 12
    na_text_size = 11

    cmap_use = cmap.copy()
    cmap_use.set_bad(color="#f0f0f0")
    norm = _heatmap_norm(vmin=vmin, vmax=vmax, vcenter=vcenter)
    im = ax.imshow(np.ma.masked_invalid(mat), norm=norm, cmap=cmap_use, aspect="auto")
    ax.set_xticks(np.arange(len(col_labels)), labels=col_labels, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(row_labels)), labels=row_labels)
    ax.tick_params(axis="x", labelsize=x_tick_size, length=3.0, width=0.8)
    ax.tick_params(axis="y", labelsize=y_tick_size, length=3.0, width=0.8)

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat[i, j]
            if np.isfinite(val):
                ax.text(
                    j,
                    i,
                    f"{val:.2f}",
                    ha="center",
                    va="center",
                    fontsize=cell_text_size,
                    color=_text_color_for_value(float(val), cmap=cmap_use, vmin=vmin, vmax=vmax, vcenter=vcenter),
                )
            else:
                ax.add_patch(
                    Rectangle(
                        (j - 0.5, i - 0.5),
                        1,
                        1,
                        facecolor="#f0f0f0",
                        edgecolor="#cfcfcf",
                        hatch="///",
                        linewidth=0.0,
                        zorder=0,
                    )
                )
                ax.text(j, i, "NA", ha="center", va="center", fontsize=na_text_size, color="#8a8a8a")

    return im


def render_unperturbed_gini_heatmap(
    *,
    cell_metrics: pl.DataFrame,
    outcome_order: Sequence[str],
    model_order: Sequence[str],
    strategy_order: Sequence[str],
    label_outcome: Callable[[str], str],
    label_model: Callable[[str], str],
    label_strategy: Callable[[str], str],
    gini_vmin: float,
    gini_vmax: float,
    outcome_panel_w: float = 4.2,
    heatmap_rows: int = 2,
    panel_nrows: int | None = None,
    panel_ncols: int | None = None,
) -> str | None:
    """Render chapter-4 unperturbed Gini heatmap panels."""
    if cell_metrics.is_empty() or not outcome_order:
        return "No unperturbed runs"

    panel_title_size = 12
    axis_tick_size = 12
    cell_text_size = 12
    empty_text_size = 12
    colorbar_label_size = 12
    colorbar_tick_size = 12

    def _panel(
        ax: plt.Axes,
        df: pl.DataFrame,
        title: str,
        *,
        show_xlabels: bool,
        show_ylabels: bool,
    ):
        pivot = df.pivot(index="model", on="strategy", values="gini", aggregate_function="first").with_columns(
            pl.col("model").cast(pl.Utf8)
        )
        if "model" not in pivot.columns:
            pivot = pl.DataFrame({"model": list(model_order)})
        else:
            missing_models = [m for m in model_order if m not in set(pivot.get_column("model").to_list())]
            if missing_models:
                pivot = pl.concat([pivot, pl.DataFrame({"model": missing_models})], how="diagonal")

        pivot = (
            pivot.with_columns(
                pl.col("model")
                .map_elements(
                    lambda m: list(model_order).index(str(m)) if str(m) in set(model_order) else 999,
                    return_dtype=pl.Int64,
                )
                .alias("_m")
            )
            .sort("_m")
            .drop("_m")
        )
        for strategy in strategy_order:
            if strategy not in pivot.columns:
                pivot = pivot.with_columns(pl.lit(float("nan")).alias(strategy))

        mat = pivot.select(list(strategy_order)).to_numpy()
        norm = _heatmap_norm(vmin=gini_vmin, vmax=gini_vmax, vcenter=0.0)
        im = ax.imshow(mat, norm=norm, cmap=plt.get_cmap("cividis"), aspect="auto")

        ax.set_xticks(np.arange(len(strategy_order)))
        ax.set_yticks(np.arange(pivot.height))
        if show_xlabels:
            ax.set_xticklabels(
                [label_strategy(s) for s in strategy_order],
                rotation=22,
                ha="right",
                fontsize=axis_tick_size,
            )
        else:
            ax.set_xticklabels([])
            ax.tick_params(axis="x", length=0)
        if show_ylabels:
            ax.set_yticklabels(
                [label_model(m) for m in pivot.get_column("model").to_list()],
                fontsize=axis_tick_size,
            )
        else:
            ax.set_yticklabels([])
            ax.tick_params(axis="y", length=0)
        ax.set_title(textwrap.fill(title, width=20), pad=8, fontsize=panel_title_size)

        if not np.isfinite(mat).any():
            ax.text(
                0.5,
                0.5,
                "No unperturbed runs",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=empty_text_size,
                color="#444444",
            )
        else:
            for i in range(mat.shape[0]):
                for j in range(mat.shape[1]):
                    val = mat[i, j]
                    if not np.isfinite(val):
                        continue
                    ax.text(
                        j,
                        i,
                        f"{val:.2f}",
                        ha="center",
                        va="center",
                        color=_text_color_for_value(
                            val,
                            cmap=plt.get_cmap("cividis"),
                            vmin=float(gini_vmin),
                            vmax=float(gini_vmax),
                            vcenter=0.0,
                        ),
                        fontsize=cell_text_size,
                    )

        best_df = df.filter(pl.col("gini").is_finite()).sort("gini", descending=True)
        if not best_df.is_empty():
            best = best_df.row(0, named=True)
            best_model = str(best["model"])
            best_strategy = str(best["strategy"])
            row_labels = [str(m) for m in pivot.get_column("model").to_list()]
            if best_model in row_labels and best_strategy in strategy_order:
                i = row_labels.index(best_model)
                j = list(strategy_order).index(best_strategy)
                ax.add_patch(
                    Rectangle(
                        (j - 0.5, i - 0.5),
                        1.0,
                        1.0,
                        fill=False,
                        edgecolor="#cc0000",
                        linewidth=2.0,
                    )
                )
        return im

    if panel_nrows is not None and panel_ncols is not None:
        outcomes = [str(o) for o in outcome_order if str(o).strip()]
        nrows = max(1, int(panel_nrows))
        ncols = max(1, int(panel_ncols))
        if nrows * ncols < len(outcomes):
            return f"Requested panel layout {nrows}x{ncols} is too small for {len(outcomes)} outcomes."
        panel_slots = outcomes + [None] * (nrows * ncols - len(outcomes))
    elif heatmap_rows <= 1:
        outcomes = [str(o) for o in outcome_order if str(o).strip()]
        nrows = 1
        ncols = max(1, len(outcomes))
        panel_slots: list[str | None] = outcomes or [None]
    else:
        nrows, ncols, panel_slots = outcome_panel_layout(outcome_order)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(2.3 + 2.2 * ncols, 1.7 + 2.1 * nrows),
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes).ravel()
    ims: list[plt.AxesImage] = []
    for idx, outcome in enumerate(panel_slots):
        ax = axes[idx]
        if outcome is None:
            ax.axis("off")
            continue
        row = idx // ncols
        col = idx % ncols
        df = cell_metrics.filter(pl.col("outcome") == outcome)
        ims.append(
            _panel(
                ax,
                df,
                title=label_outcome(outcome),
                show_xlabels=row == (nrows - 1),
                show_ylabels=col == 0,
            )
        )

    if not ims:
        return "No unperturbed outcomes available for the Gini heatmap."
    cbar = fig.colorbar(ims[0], ax=axes, shrink=0.84, label="Gini $G$ (chance = 0, perfect = 1)")
    cbar.ax.tick_params(labelsize=colorbar_tick_size)
    cbar.set_label("Gini $G$ (chance = 0, perfect = 1)", fontsize=colorbar_label_size)
    plt.show()
    return None


def render_belief_unperturbed_cosine_heatmap(
    *,
    canonical_root: Path,
    model_order: Sequence[str],
    strategy_order: Sequence[str],
    label_model: Callable[[str], str],
    label_strategy: Callable[[str], str],
    label_engine: Callable[[str], str],
    cosine_vmin: float = 0.0,
    cosine_vmax: float = 1.0,
    excluded_constructs: Collection[str] | None = None,
) -> str | None:
    """Render standardized full-state cosine heatmap by model × strategy."""
    compact_candidates = (
        canonical_root / "unperturbed_vector_cosine.csv",
        canonical_root.parent / "unperturbed_vector_cosine.csv",
    )
    compact_path = next((path for path in compact_candidates if path.exists()), None)
    if compact_path is not None:
        df = pl.read_csv(compact_path).select(
            pl.col("model").cast(pl.Utf8),
            pl.col("strategy").cast(pl.Utf8),
            pl.col("vector_cosine_std").cast(pl.Float64, strict=False),
        )
    else:
        if not canonical_root.exists():
            return f"Belief canonical root not found for cosine heatmap: {canonical_root}."

        def _row_cosine(x: np.ndarray, y: np.ndarray) -> float:
            mask = np.isfinite(x) & np.isfinite(y)
            if int(mask.sum()) < 2:
                return float("nan")
            xv = x[mask]
            yv = y[mask]
            nx = float(np.linalg.norm(xv))
            ny = float(np.linalg.norm(yv))
            if nx <= 0.0 or ny <= 0.0:
                return float("nan")
            return float(np.dot(xv, yv) / (nx * ny))

        excluded = {
            str(v) for v in (BELIEF_PLOT_EXCLUDED_CONSTRUCTS if excluded_constructs is None else excluded_constructs)
        }
        rows: list[dict[str, object]] = []
        for model_dir in sorted(p for p in canonical_root.iterdir() if p.is_dir() and not p.name.startswith("_")):
            model = str(model_dir.name)
            for strategy_dir in sorted(p for p in model_dir.iterdir() if p.is_dir() and not p.name.startswith("_")):
                strategy = str(strategy_dir.name)
                metrics_path = strategy_dir / "metrics.parquet"
                if not metrics_path.exists() or metrics_path.stat().st_size == 0:
                    continue
                try:
                    metrics_df = pl.read_parquet(metrics_path)
                except Exception:
                    continue
                if metrics_df.is_empty():
                    continue
                engines = [
                    str(v)
                    for v in BELIEF_OUTCOME_ORDER
                    if f"true_{v}" in metrics_df.columns
                    and f"pred_{v}" in metrics_df.columns
                    and str(v) not in excluded
                ]
                extras = sorted(
                    {
                        str(col)[5:]
                        for col in metrics_df.columns
                        if str(col).startswith("true_")
                        and f"pred_{str(col)[5:]}" in set(metrics_df.columns)
                        and str(col)[5:] not in set(engines)
                        and str(col)[5:] not in excluded
                    }
                )
                engines.extend(extras)
                if len(engines) < 2:
                    continue

                true_mat = np.column_stack(
                    [
                        metrics_df.select(pl.col(f"true_{engine}").cast(pl.Float64, strict=False))
                        .to_series()
                        .to_numpy()
                        for engine in engines
                    ]
                )
                pred_mat = np.column_stack(
                    [
                        metrics_df.select(pl.col(f"pred_{engine}").cast(pl.Float64, strict=False))
                        .to_series()
                        .to_numpy()
                        for engine in engines
                    ]
                )
                mu_true = np.nanmean(true_mat, axis=0)
                sd_true = np.nanstd(true_mat, axis=0, ddof=1)
                valid_dims = np.isfinite(sd_true) & (sd_true > 0.0)
                if int(np.sum(valid_dims)) < 2:
                    continue
                true_z = (true_mat[:, valid_dims] - mu_true[valid_dims]) / sd_true[valid_dims]
                pred_z = (pred_mat[:, valid_dims] - mu_true[valid_dims]) / sd_true[valid_dims]

                row_scores = np.asarray(
                    [_row_cosine(pred_z[i, :], true_z[i, :]) for i in range(true_z.shape[0])],
                    dtype=np.float64,
                )
                row_scores = row_scores[np.isfinite(row_scores)]
                if row_scores.size == 0:
                    continue
                rows.append(
                    {
                        "model": model,
                        "strategy": strategy,
                        "vector_cosine_std": float(np.mean(row_scores)),
                    }
                )

        if not rows:
            return "No psychological-profile unperturbed vector-cosine metrics available."

        df = pl.DataFrame(rows).with_columns(
            pl.col("model").cast(pl.Utf8),
            pl.col("strategy").cast(pl.Utf8),
            pl.col("vector_cosine_std").cast(pl.Float64, strict=False),
        )

    observed_models = [str(v) for v in df.get_column("model").unique().to_list() if str(v).strip()]
    observed_strategies = [str(v) for v in df.get_column("strategy").unique().to_list() if str(v).strip()]
    model_use = [str(m) for m in model_order if str(m) in set(observed_models)]
    model_use.extend(sorted(m for m in observed_models if m not in set(model_use)))
    strategy_use = [str(s) for s in strategy_order if str(s) in set(observed_strategies)]
    strategy_use.extend(sorted(s for s in observed_strategies if s not in set(strategy_use)))
    pivot = df.pivot(index="model", on="strategy", values="vector_cosine_std", aggregate_function="first")
    if "model" not in pivot.columns:
        return "Belief unperturbed cosine metrics could not be pivoted."

    missing_models = [m for m in model_use if m not in set(pivot.get_column("model").to_list())]
    if missing_models:
        pivot = pl.concat([pivot, pl.DataFrame({"model": missing_models})], how="diagonal")
    pivot = (
        pivot.with_columns(
            pl.col("model")
            .map_elements(
                lambda m: model_use.index(str(m)) if str(m) in set(model_use) else 9999,
                return_dtype=pl.Int64,
            )
            .alias("_m")
        )
        .sort("_m")
        .drop("_m")
    )
    for strategy in strategy_use:
        if strategy not in pivot.columns:
            pivot = pivot.with_columns(pl.lit(float("nan")).alias(strategy))

    mat = pivot.select(strategy_use).to_numpy()
    if not np.isfinite(mat).any():
        return "No finite standardized cosine values available for the belief unperturbed heatmap."

    cmap = plt.get_cmap("cividis").copy()
    cmap.set_bad(color="#f0f0f0")
    norm = _heatmap_norm(vmin=cosine_vmin, vmax=cosine_vmax, vcenter=0.0)
    fig, ax = plt.subplots(
        figsize=(2.4 + 1.30 * len(strategy_use), 1.8 + 0.55 * len(model_use)),
        constrained_layout=False,
    )
    im = ax.imshow(np.ma.masked_invalid(mat), norm=norm, cmap=cmap, aspect="auto")
    ax.set_xticks(
        np.arange(len(strategy_use)), labels=[label_strategy(s) for s in strategy_use], rotation=30, ha="right"
    )
    ax.set_yticks(
        np.arange(pivot.height),
        labels=[label_model(str(m)) for m in pivot.get_column("model").to_list()],
    )
    ax.tick_params(axis="x", labelsize=12)
    ax.tick_params(axis="y", labelsize=12)
    ax.set_xlabel("Prompting strategy", fontsize=12)
    ax.set_ylabel("Model", fontsize=12)

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat[i, j]
            if not np.isfinite(val):
                ax.text(j, i, "NA", ha="center", va="center", fontsize=11, color="#8a8a8a")
                continue
            ax.text(
                j,
                i,
                f"{val:.2f}",
                ha="center",
                va="center",
                color=_text_color_for_value(
                    float(val),
                    cmap=cmap,
                    vmin=float(cosine_vmin),
                    vmax=float(cosine_vmax),
                    vcenter=0.0,
                ),
                fontsize=12,
            )

    finite_df = df.filter(pl.col("vector_cosine_std").is_finite()).sort("vector_cosine_std", descending=True)
    if not finite_df.is_empty():
        best = finite_df.row(0, named=True)
        best_model = str(best.get("model"))
        best_strategy = str(best.get("strategy"))
        row_labels = [str(m) for m in pivot.get_column("model").to_list()]
        if best_model in row_labels and best_strategy in strategy_use:
            i = row_labels.index(best_model)
            j = strategy_use.index(best_strategy)
            ax.add_patch(
                Rectangle(
                    (j - 0.5, i - 0.5),
                    1.0,
                    1.0,
                    fill=False,
                    edgecolor="#cc0000",
                    linewidth=2.0,
                )
            )

    cbar = fig.colorbar(im, ax=ax, shrink=0.88, pad=0.04, label="Std. profile-vector cosine")
    cbar.ax.tick_params(labelsize=12)
    cbar.set_label("Std. profile-vector cosine", fontsize=12)
    fig.subplots_adjust(left=0.22, bottom=0.24, right=0.84, top=0.96)
    plt.show()
    return None


def render_belief_signal_block_cosine_heatmap(
    *,
    signals_df: pl.DataFrame,
    strategy: str,
    model_order: Sequence[str],
    signal_bucket_map: Mapping[str, str],
    signal_bucket_order: Sequence[str],
    label_model: Callable[[str], str],
    label_signal_bucket: Callable[[str], str],
    metric_col: str = "vector_cosine_std",
    cosine_vmin: float = 0.0,
    cosine_vmax: float = 1.0,
) -> str | None:
    """Render perturbed belief cosine heatmap by model × signal-mechanism block."""
    if signals_df.is_empty():
        return "No belief signal-level summary metrics available for the cosine heatmap."
    required = {"model", "strategy", "signal_id", str(metric_col)}
    missing = sorted(required - set(signals_df.columns))
    if missing:
        return "Belief signal summary is missing required columns: " + ", ".join(missing) + "."

    bucket_lookup = {str(k): str(v) for k, v in signal_bucket_map.items()}
    df = (
        signals_df.select(
            pl.col("model").cast(pl.Utf8),
            pl.col("strategy").cast(pl.Utf8),
            pl.col("signal_id").cast(pl.Utf8),
            pl.col(str(metric_col)).cast(pl.Float64, strict=False).alias("vector_cosine"),
        )
        .filter(pl.col("strategy") == str(strategy))
        .with_columns(
            pl.col("signal_id")
            .map_elements(lambda sid: bucket_lookup.get(str(sid), "other"), return_dtype=pl.Utf8)
            .alias("signal_block")
        )
        .drop_nulls(["model", "signal_block", "vector_cosine"])
        .filter(pl.col("vector_cosine").is_finite())
    )
    if df.is_empty():
        return f"No finite belief signal-level cosine values available for strategy '{strategy}'."

    agg = (
        df.group_by(["model", "signal_block"])
        .agg(pl.median("vector_cosine").alias("vector_cosine"))
        .sort(["model", "signal_block"])
    )
    if agg.is_empty():
        return "Belief signal-level cosine aggregation is empty."

    observed_models = [str(v) for v in agg.get_column("model").unique().to_list() if str(v).strip()]
    observed_blocks = [str(v) for v in agg.get_column("signal_block").unique().to_list() if str(v).strip()]
    model_use = [str(m) for m in model_order if str(m) in set(observed_models)]
    model_use.extend(sorted(m for m in observed_models if m not in set(model_use)))
    block_use = [str(b) for b in signal_bucket_order if str(b) in set(observed_blocks)]
    block_use.extend(sorted(b for b in observed_blocks if b not in set(block_use)))
    if not model_use or not block_use:
        return "Belief signal-block cosine heatmap has empty model/block axes."

    pivot = agg.pivot(index="model", on="signal_block", values="vector_cosine", aggregate_function="first")
    if "model" not in pivot.columns:
        return "Belief signal-block cosine values could not be pivoted."
    missing_models = [m for m in model_use if m not in set(pivot.get_column("model").to_list())]
    if missing_models:
        pivot = pl.concat([pivot, pl.DataFrame({"model": missing_models})], how="diagonal")
    pivot = (
        pivot.with_columns(
            pl.col("model")
            .map_elements(
                lambda m: model_use.index(str(m)) if str(m) in set(model_use) else 9999,
                return_dtype=pl.Int64,
            )
            .alias("_m")
        )
        .sort("_m")
        .drop("_m")
    )
    for block in block_use:
        if block not in pivot.columns:
            pivot = pivot.with_columns(pl.lit(float("nan")).alias(block))

    mat = pivot.select(block_use).to_numpy()
    if not np.isfinite(mat).any():
        return "No finite values available for the belief signal-block cosine heatmap."

    cmap = plt.get_cmap("cividis")
    norm = _heatmap_norm(vmin=cosine_vmin, vmax=cosine_vmax, vcenter=0.5)
    fig, ax = plt.subplots(
        figsize=(1.4 + 1.4 * len(block_use), 1.4 + 0.7 * len(model_use)),
        constrained_layout=True,
    )
    im = ax.imshow(mat, norm=norm, cmap=cmap, aspect="auto")
    ax.set_xticks(
        np.arange(len(block_use)),
        labels=[label_signal_bucket(str(block)) for block in block_use],
        rotation=30,
        ha="right",
    )
    ax.set_yticks(
        np.arange(pivot.height),
        labels=[label_model(str(model)) for model in pivot.get_column("model").to_list()],
    )

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat[i, j]
            if not np.isfinite(val):
                continue
            ax.text(
                j,
                i,
                f"{val:.3f}",
                ha="center",
                va="center",
                color=_text_color_for_value(
                    float(val),
                    cmap=cmap,
                    vmin=float(cosine_vmin),
                    vmax=float(cosine_vmax),
                    vcenter=0.5,
                ),
                fontsize=8,
            )

    cbar_label = (
        "Median vector cosine similarity (standardized)"
        if str(metric_col) == "vector_cosine_std"
        else "Median vector cosine similarity"
    )
    fig.colorbar(im, ax=ax, shrink=0.9, label=cbar_label)
    plt.show()
    return None


def render_belief_signal_block_panel_heatmaps(
    *,
    signals_df: pl.DataFrame,
    model_order: Sequence[str],
    strategy_order: Sequence[str],
    signal_bucket_map: Mapping[str, str],
    signal_bucket_order: Sequence[str],
    label_model: Callable[[str], str],
    label_strategy: Callable[[str], str],
    label_signal_bucket: Callable[[str], str],
    metric_col: str = "vector_cosine_std",
    cosine_vmin: float = -1.0,
    cosine_vmax: float = 1.0,
) -> str | None:
    """Render faceted belief signal-block heatmaps (rows=model, cols=strategy)."""
    if signals_df.is_empty():
        return "No belief signal-level summary metrics available for the heatmap."
    required = {"model", "strategy", "signal_id", str(metric_col)}
    missing = sorted(required - set(signals_df.columns))
    if missing:
        return "Belief signal summary is missing required columns: " + ", ".join(missing) + "."

    bucket_lookup = {str(k): str(v) for k, v in signal_bucket_map.items()}
    df = (
        signals_df.select(
            pl.col("model").cast(pl.Utf8),
            pl.col("strategy").cast(pl.Utf8),
            pl.col("signal_id").cast(pl.Utf8),
            pl.col(str(metric_col)).cast(pl.Float64, strict=False).alias("score"),
        )
        .with_columns(
            pl.col("signal_id")
            .map_elements(lambda sid: bucket_lookup.get(str(sid), "other"), return_dtype=pl.Utf8)
            .alias("signal_block")
        )
        .drop_nulls(["model", "strategy", "signal_block", "score"])
        .filter(pl.col("score").is_finite())
    )
    if df.is_empty():
        return "No finite belief signal-level cosine values available."

    agg = (
        df.group_by(["model", "strategy", "signal_block"])
        .agg(pl.median("score").alias("score"))
        .sort(["signal_block", "model", "strategy"])
    )
    if agg.is_empty():
        return "Belief signal-level cosine aggregation is empty."

    observed_models = [str(v) for v in agg.get_column("model").unique().to_list() if str(v).strip()]
    observed_strategies = [str(v) for v in agg.get_column("strategy").unique().to_list() if str(v).strip()]
    observed_blocks = [str(v) for v in agg.get_column("signal_block").unique().to_list() if str(v).strip()]
    model_use = [str(m) for m in model_order if str(m) in set(observed_models)]
    model_use.extend(sorted(m for m in observed_models if m not in set(model_use)))
    strategy_use = [str(s) for s in strategy_order if str(s) in set(observed_strategies)]
    strategy_use.extend(sorted(s for s in observed_strategies if s not in set(strategy_use)))
    block_use = [str(b) for b in signal_bucket_order if str(b) in set(observed_blocks)]
    block_use.extend(sorted(b for b in observed_blocks if b not in set(block_use)))
    if not model_use or not strategy_use or not block_use:
        return "Belief signal-block heatmaps have empty model/strategy/block axes."

    n_blocks = len(block_use)
    ncols = min(2, n_blocks) if n_blocks > 2 else n_blocks
    nrows = int(np.ceil(n_blocks / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(2.2 + 3.3 * ncols, 1.4 + 2.4 * nrows),
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes).ravel()
    cmap = plt.get_cmap("cividis")
    norm = _heatmap_norm(vmin=cosine_vmin, vmax=cosine_vmax, vcenter=0.0)
    im = None
    has_finite = False

    for idx, block in enumerate(block_use):
        ax = axes[idx]
        row_idx = idx // ncols
        col_idx = idx % ncols
        sub = agg.filter(pl.col("signal_block") == block)
        pivot = sub.pivot(index="model", on="strategy", values="score", aggregate_function="first")
        if "model" not in pivot.columns:
            ax.axis("off")
            continue
        missing_models = [m for m in model_use if m not in set(pivot.get_column("model").to_list())]
        if missing_models:
            pivot = pl.concat([pivot, pl.DataFrame({"model": missing_models})], how="diagonal")
        pivot = (
            pivot.with_columns(
                pl.col("model")
                .map_elements(
                    lambda m: model_use.index(str(m)) if str(m) in set(model_use) else 9999,
                    return_dtype=pl.Int64,
                )
                .alias("_m")
            )
            .sort("_m")
            .drop("_m")
        )
        for strategy in strategy_use:
            if strategy not in pivot.columns:
                pivot = pivot.with_columns(pl.lit(float("nan")).alias(strategy))
        mat = pivot.select(strategy_use).to_numpy()
        if np.isfinite(mat).any():
            has_finite = True
        im = ax.imshow(mat, norm=norm, cmap=cmap, aspect="auto")
        ax.set_title(textwrap.fill(label_signal_bucket(block), width=28), fontsize=10)
        if row_idx == (nrows - 1):
            ax.set_xticks(
                np.arange(len(strategy_use)),
                labels=[label_strategy(s) for s in strategy_use],
                rotation=30,
                ha="right",
            )
            ax.tick_params(axis="x", labelsize=8)
        else:
            ax.set_xticks(np.arange(len(strategy_use)), labels=[])

        if col_idx == 0:
            ax.set_yticks(
                np.arange(pivot.height),
                labels=[label_model(str(m)) for m in pivot.get_column("model").to_list()],
            )
            ax.tick_params(axis="y", labelsize=9)
        else:
            ax.set_yticks(np.arange(pivot.height), labels=[])
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                val = mat[i, j]
                if not np.isfinite(val):
                    continue
                ax.text(
                    j,
                    i,
                    f"{val:.2f}",
                    ha="center",
                    va="center",
                    color=_text_color_for_value(
                        float(val),
                        cmap=cmap,
                        vmin=float(cosine_vmin),
                        vmax=float(cosine_vmax),
                        vcenter=0.0,
                    ),
                    fontsize=7.5,
                )

    for ax in axes[n_blocks:]:
        ax.axis("off")

    if not has_finite or im is None:
        plt.close(fig)
        return "No finite values available for belief signal-block panel heatmaps."

    cbar = fig.colorbar(im, ax=axes[:n_blocks], shrink=0.92)
    cbar.set_label("Median standardized vector cosine (0 = orthogonal)")
    plt.show()
    return None


def render_belief_vector_importance_alignment_cosine(
    *,
    canonical_root: Path,
    preferred_model: str,
    preferred_strategy: str,
    label_engine: Callable[[str], str],
    label_model: Callable[[str], str] | None = None,
    label_strategy: Callable[[str], str] | None = None,
    signal_id: str = "__unperturbed__",
    standardize_by_ref_sigma: bool = True,
    excluded_constructs: Collection[str] | None = None,
) -> str | None:
    """Render belief vector-importance rank alignment (reference vs LLM)."""
    if not canonical_root.exists():
        return f"Belief canonical root not found for rank-alignment plot: {canonical_root}."

    model_dirs = sorted(p for p in canonical_root.iterdir() if p.is_dir() and not p.name.startswith("_"))
    if not model_dirs:
        return "No belief canonical model directories available for rank-alignment plotting."
    available_models = [d.name for d in model_dirs]
    chosen_model = preferred_model if preferred_model in available_models else available_models[0]

    strategy_dirs = sorted(
        p
        for p in (canonical_root / chosen_model).iterdir()
        if p.is_dir() and not p.name.startswith("_") and (p / "signals" / signal_id / "metrics.parquet").exists()
    )
    if not strategy_dirs:
        return "No belief unperturbed signal metrics available for the selected model."
    available_strategies = [d.name for d in strategy_dirs]
    chosen_strategy = preferred_strategy if preferred_strategy in available_strategies else available_strategies[0]

    metrics_path = canonical_root / chosen_model / chosen_strategy / "signals" / signal_id / "metrics.parquet"
    if not metrics_path.exists():
        return f"Belief unperturbed metrics not found for rank-alignment plot: {metrics_path}."

    df = pl.read_parquet(metrics_path)
    if df.is_empty():
        return "Belief unperturbed metrics are empty for rank-alignment plotting."

    excluded = {
        str(v) for v in (BELIEF_PLOT_EXCLUDED_CONSTRUCTS if excluded_constructs is None else excluded_constructs)
    }
    ref_prefix = "pre_"
    llm_prefix = "post_"
    state_vars = [
        str(v) for v in BELIEF_OUTCOME_ORDER if f"{ref_prefix}{v}" in df.columns and f"{llm_prefix}{v}" in df.columns
    ]
    extras = sorted(
        {
            col[len(ref_prefix) :]
            for col in df.columns
            if col.startswith(ref_prefix)
            and f"{llm_prefix}{col[len(ref_prefix) :]}" in df.columns
            and col[len(ref_prefix) :] not in set(state_vars)
            and col[len(ref_prefix) :] not in excluded
        }
    )
    state_vars.extend(extras)
    state_vars = [v for v in state_vars if v not in excluded]
    if not state_vars:
        return "No paired pre/post belief constructs found for rank-alignment plotting."

    rows: list[dict[str, object]] = []
    for state_var in state_vars:
        ref_vals = df.select(pl.col(f"{ref_prefix}{state_var}").cast(pl.Float64, strict=False)).to_series().to_numpy()
        llm_vals = df.select(pl.col(f"{llm_prefix}{state_var}").cast(pl.Float64, strict=False)).to_series().to_numpy()
        ref_vals = ref_vals[np.isfinite(ref_vals)]
        llm_vals = llm_vals[np.isfinite(llm_vals)]
        if ref_vals.size == 0 or llm_vals.size == 0:
            continue
        sigma = float(np.std(ref_vals, ddof=1)) if ref_vals.size > 1 else float("nan")
        if not np.isfinite(sigma) or sigma <= 0.0 or not standardize_by_ref_sigma:
            sigma = 1.0
        ref_importance = float(np.median(np.abs(ref_vals)) / sigma)
        llm_importance = float(np.median(np.abs(llm_vals)) / sigma)
        if not (np.isfinite(ref_importance) and np.isfinite(llm_importance)):
            continue
        rows.append(
            {
                "state_var": state_var,
                "ref_importance": ref_importance,
                "llm_importance": llm_importance,
            }
        )

    if len(rows) < 2:
        return "Insufficient finite construct-importance values for belief rank-alignment plotting."

    imp_df = pl.DataFrame(rows)
    state_vars_plot = [str(v) for v in imp_df.get_column("state_var").to_list()]
    ref_vals = [float(v) for v in imp_df.get_column("ref_importance").to_list()]
    llm_vals = [float(v) for v in imp_df.get_column("llm_importance").to_list()]
    rank_ref = _rank_desc(ref_vals, state_vars_plot)
    rank_llm = _rank_desc(llm_vals, state_vars_plot)

    x = np.array([float(rank_ref[v]) for v in state_vars_plot], dtype=float)
    y = np.array([float(rank_llm[v]) for v in state_vars_plot], dtype=float)
    rho = _spearman_from_ranks(x, y)
    max_rank = int(max(np.max(x), np.max(y)))

    _, ax = plt.subplots(figsize=(6.2, 5.1), constrained_layout=True)
    cmap = plt.get_cmap("tab10")
    color_map = {var: cmap(i % 10) for i, var in enumerate(state_vars_plot)}

    for state_var in state_vars_plot:
        xv = float(rank_ref[state_var])
        yv = float(rank_llm[state_var])
        ax.scatter(
            xv,
            yv,
            s=80.0,
            color=color_map[state_var],
            edgecolors="white",
            linewidths=0.6,
            alpha=0.9,
        )
        ax.annotate(
            label_engine(state_var),
            (xv, yv),
            textcoords="offset points",
            xytext=(6, 4),
            fontsize=8,
            bbox={"boxstyle": "round,pad=0.12", "fc": "white", "ec": "none", "alpha": 0.7},
        )

    diag_min = 0.5
    diag_max = float(max_rank) + 0.5
    ax.plot([diag_min, diag_max], [diag_min, diag_max], color="black", linewidth=1.0, alpha=0.6)
    ax.set_xlim(diag_min, diag_max)
    ax.set_ylim(diag_min, diag_max)
    ax.set_xticks(np.arange(1, max_rank + 1))
    ax.set_yticks(np.arange(1, max_rank + 1))
    ax.grid(True, linestyle=":", alpha=0.35)
    ax.set_xlabel("Reference construct-importance rank (1 = largest median |component| / σ_b)")
    ax.set_ylabel("LLM construct-importance rank (1 = largest median |component| / σ_b)")

    model_txt = label_model(chosen_model) if label_model is not None else chosen_model
    strategy_txt = label_strategy(chosen_strategy) if label_strategy is not None else chosen_strategy
    rho_txt = f"$\\rho={rho:.2f}$" if np.isfinite(rho) else "$\\rho$ unavailable"
    ax.set_title(f"{model_txt}, {strategy_txt} ({rho_txt})")
    plt.show()
    return None


def render_belief_unperturbed_engine_panel_heatmaps(
    *,
    canonical_root: Path,
    canonical_root_overrides_by_engine: Mapping[str, Path] | None = None,
    model_order: Sequence[str],
    strategy_order: Sequence[str],
    label_model: Callable[[str], str],
    label_strategy: Callable[[str], str],
    label_engine: Callable[[str], str],
    metric: str = "cosine_std",
    cosine_vmin: float = -1.0,
    cosine_vmax: float = 1.0,
    skill_vmin: float = -0.5,
    skill_vmax: float = 1.0,
    gini_vmin: float = -1.0,
    gini_vmax: float = 1.0,
    excluded_constructs: Collection[str] | None = None,
    panel_nrows: int | None = None,
    panel_ncols: int | None = None,
) -> str | None:
    """Render unperturbed belief engine panels (rows=model, cols=strategy).

    Parameters
    ----------
    canonical_root_overrides_by_engine
        Optional engine -> canonical root mapping. When provided, rows for the
        listed engines are sourced from the override root, while other engines
        are sourced from ``canonical_root``.
    """
    if not canonical_root.exists():
        return (
            f"Psychological-profile canonical root not found for unperturbed reconstruction heatmaps: {canonical_root}."
        )

    def _std_cosine(x: np.ndarray, y: np.ndarray) -> float:
        mask = np.isfinite(x) & np.isfinite(y)
        if int(mask.sum()) < 3:
            return float("nan")
        xv = x[mask]
        yv = y[mask]
        sx = float(np.std(xv, ddof=1))
        sy = float(np.std(yv, ddof=1))
        if not np.isfinite(sx) or not np.isfinite(sy) or sx <= 0.0 or sy <= 0.0:
            return float("nan")
        xz = (xv - float(np.mean(xv))) / sx
        yz = (yv - float(np.mean(yv))) / sy
        nx = float(np.linalg.norm(xz))
        ny = float(np.linalg.norm(yz))
        if nx <= 0.0 or ny <= 0.0:
            return float("nan")
        return float(np.dot(xz, yz) / (nx * ny))

    metric_key = str(metric).strip().lower()
    valid_metrics = {"cosine_std", "skill", "gini"}
    if metric_key not in valid_metrics:
        return f"Unsupported unperturbed belief metric '{metric}'. Use one of: cosine_std, skill, gini."

    def _score(x: np.ndarray, y: np.ndarray) -> float:
        if metric_key == "cosine_std":
            return _std_cosine(x, y)
        if metric_key == "skill":
            return mae_skill_cal_np(x, y)
        return gini_np(x, y)

    excluded = {
        str(v) for v in (BELIEF_PLOT_EXCLUDED_CONSTRUCTS if excluded_constructs is None else excluded_constructs)
    }
    root_by_engine: dict[str, Path] = {}
    if canonical_root_overrides_by_engine:
        for engine, root in canonical_root_overrides_by_engine.items():
            engine_key = str(engine).strip()
            if not engine_key:
                continue
            root_path = Path(root)
            if root_path.exists():
                root_by_engine[engine_key] = root_path

    source_roots: list[Path] = [canonical_root]
    source_roots.extend([p for p in dict.fromkeys(root_by_engine.values()) if p != canonical_root])

    rows: list[dict[str, object]] = []
    for source_root in source_roots:
        source_key = str(source_root.resolve())
        for model_dir in sorted(p for p in source_root.iterdir() if p.is_dir() and not p.name.startswith("_")):
            model = str(model_dir.name)
            for strategy_dir in sorted(p for p in model_dir.iterdir() if p.is_dir() and not p.name.startswith("_")):
                strategy = str(strategy_dir.name)
                metrics_path = strategy_dir / "metrics.parquet"
                if not metrics_path.exists() or metrics_path.stat().st_size == 0:
                    continue
                try:
                    df = pl.read_parquet(metrics_path)
                except Exception:
                    continue
                if df.is_empty():
                    continue
                engines = [
                    str(v)
                    for v in BELIEF_OUTCOME_ORDER
                    if f"true_{v}" in df.columns and f"pred_{v}" in df.columns and str(v) not in excluded
                ]
                extras = sorted(
                    {
                        str(c)[5:]
                        for c in df.columns
                        if str(c).startswith("true_")
                        and f"pred_{str(c)[5:]}" in set(df.columns)
                        and str(c)[5:] not in set(engines)
                        and str(c)[5:] not in excluded
                    }
                )
                engines.extend(extras)
                for engine in engines:
                    x = df.select(pl.col(f"true_{engine}").cast(pl.Float64, strict=False)).to_series().to_numpy()
                    y = df.select(pl.col(f"pred_{engine}").cast(pl.Float64, strict=False)).to_series().to_numpy()
                    rows.append(
                        {
                            "model": model,
                            "strategy": strategy,
                            "engine": engine,
                            "score": _score(x, y),
                            "source_root": source_key,
                        }
                    )

    if not rows:
        return "No unperturbed psychological-profile data available for reconstruction heatmaps."

    df = pl.DataFrame(rows)
    default_source = str(canonical_root.resolve())
    if root_by_engine:
        preferred_source_by_engine = {k: str(v.resolve()) for k, v in root_by_engine.items()}
        df = (
            df.with_columns(
                pl.col("engine")
                .cast(pl.Utf8)
                .map_elements(
                    lambda e: preferred_source_by_engine.get(str(e), default_source),
                    return_dtype=pl.Utf8,
                )
                .alias("_preferred_source")
            )
            .filter(pl.col("source_root") == pl.col("_preferred_source"))
            .drop("_preferred_source")
        )
    else:
        df = df.filter(pl.col("source_root") == default_source)

    df = df.drop_nulls(["model", "strategy", "engine", "score"]).filter(pl.col("score").is_finite())
    if df.is_empty():
        return f"No finite unperturbed psychological-profile {metric_key} values available."

    model_use = [str(m) for m in model_order if str(m) in set(df.get_column("model").unique().to_list())]
    model_use.extend(sorted(m for m in df.get_column("model").unique().to_list() if m not in set(model_use)))
    observed_strategies = set(df.get_column("strategy").unique().to_list())
    strategy_use = [str(s) for s in strategy_order]
    strategy_use.extend(sorted(s for s in observed_strategies if s not in set(strategy_use)))
    engine_use = [
        str(e)
        for e in BELIEF_OUTCOME_ORDER
        if str(e) in set(df.get_column("engine").unique().to_list()) and str(e) not in excluded
    ]
    engine_use.extend(sorted(e for e in df.get_column("engine").unique().to_list() if e not in set(engine_use)))

    n_panels = len(engine_use)
    if panel_nrows is not None and panel_ncols is not None:
        nrows = max(1, int(panel_nrows))
        ncols = max(1, int(panel_ncols))
        if nrows * ncols < n_panels:
            return f"Requested panel layout {nrows}x{ncols} is too small for {n_panels} engines."
    else:
        ncols = min(3, n_panels)
        nrows = int(np.ceil(n_panels / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(2.1 + 2.9 * ncols, 1.1 + 2.0 * nrows),
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes).ravel()
    cmap = plt.get_cmap("cividis")
    cmap_use = cmap.copy()
    cmap_use.set_bad(color="#f0f0f0")
    if metric_key == "cosine_std":
        vmin, vmax, vcenter = cosine_vmin, cosine_vmax, 0.0
        cbar_label = "Variable-wise standardized correlation (0 = no linear association)"
    elif metric_key == "skill":
        vmin, vmax, vcenter = skill_vmin, skill_vmax, 0.0
        cbar_label = "MAE skill (1 = perfect, 0 = baseline)"
    else:
        vmin, vmax, vcenter = gini_vmin, gini_vmax, 0.0
        cbar_label = "Gini $G$ (chance = 0, perfect = 1)"
    norm = _heatmap_norm(vmin=vmin, vmax=vmax, vcenter=vcenter)
    im = None

    for idx, engine in enumerate(engine_use):
        ax = axes[idx]
        row_idx = idx // ncols
        col_idx = idx % ncols
        sub = df.filter(pl.col("engine") == engine)
        pivot = sub.pivot(index="model", on="strategy", values="score", aggregate_function="first")
        if "model" not in pivot.columns:
            ax.axis("off")
            continue
        missing_models = [m for m in model_use if m not in set(pivot.get_column("model").to_list())]
        if missing_models:
            pivot = pl.concat([pivot, pl.DataFrame({"model": missing_models})], how="diagonal")
        pivot = (
            pivot.with_columns(
                pl.col("model")
                .map_elements(
                    lambda m: model_use.index(str(m)) if str(m) in set(model_use) else 9999,
                    return_dtype=pl.Int64,
                )
                .alias("_m")
            )
            .sort("_m")
            .drop("_m")
        )
        for strategy in strategy_use:
            if strategy not in pivot.columns:
                pivot = pivot.with_columns(pl.lit(float("nan")).alias(strategy))
        mat = pivot.select(strategy_use).to_numpy()
        im = ax.imshow(np.ma.masked_invalid(mat), norm=norm, cmap=cmap_use, aspect="auto")
        ax.set_title(textwrap.fill(label_engine(engine), width=24), fontsize=12)

        if row_idx == (nrows - 1):
            ax.set_xticks(
                np.arange(len(strategy_use)),
                labels=[label_strategy(s) for s in strategy_use],
                rotation=30,
                ha="right",
            )
            ax.tick_params(axis="x", labelsize=12)
        else:
            ax.set_xticks(np.arange(len(strategy_use)), labels=[])

        if col_idx == 0:
            ax.set_yticks(
                np.arange(pivot.height),
                labels=[label_model(str(m)) for m in pivot.get_column("model").to_list()],
            )
            ax.tick_params(axis="y", labelsize=12)
        else:
            ax.set_yticks(np.arange(pivot.height), labels=[])

        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                val = mat[i, j]
                if not np.isfinite(val):
                    ax.add_patch(
                        Rectangle(
                            (j - 0.5, i - 0.5),
                            1,
                            1,
                            facecolor="#f0f0f0",
                            edgecolor="#cfcfcf",
                            hatch="///",
                            linewidth=0.0,
                            zorder=0,
                        )
                    )
                    ax.text(j, i, "NA", ha="center", va="center", fontsize=11, color="#8a8a8a")
                    continue
                ax.text(
                    j,
                    i,
                    f"{val:.2f}",
                    ha="center",
                    va="center",
                    color=_text_color_for_value(
                        float(val),
                        cmap=cmap_use,
                        vmin=float(vmin),
                        vmax=float(vmax),
                        vcenter=float(vcenter),
                    ),
                    fontsize=12,
                )
        best_df = sub.filter(pl.col("score").is_finite()).sort("score", descending=True)
        if not best_df.is_empty():
            best = best_df.row(0, named=True)
            best_model = str(best.get("model"))
            best_strategy = str(best.get("strategy"))
            row_labels = [str(m) for m in pivot.get_column("model").to_list()]
            if best_model in row_labels and best_strategy in strategy_use:
                i = row_labels.index(best_model)
                j = strategy_use.index(best_strategy)
                ax.add_patch(
                    Rectangle(
                        (j - 0.5, i - 0.5),
                        1.0,
                        1.0,
                        fill=False,
                        edgecolor="#cc0000",
                        linewidth=2.0,
                    )
                )

    for ax in axes[n_panels:]:
        ax.axis("off")

    if im is None:
        plt.close(fig)
        return "No plottable unperturbed psychological-profile panels were produced."
    cbar = fig.colorbar(im, ax=axes[:n_panels], shrink=0.92)
    cbar.ax.tick_params(labelsize=12)
    cbar.set_label(cbar_label, fontsize=12)
    plt.show()
    return None


def render_belief_perturbed_signal_engine_heatmap(
    *,
    canonical_root: Path,
    model: str,
    strategy: str,
    label_engine: Callable[[str], str],
    signal_bucket_map: Mapping[str, str] | None = None,
    signal_bucket_order: Sequence[str] | None = None,
    label_signal: Callable[[str], str] | None = None,
    cosine_vmin: float = -1.0,
    cosine_vmax: float = 1.0,
    excluded_constructs: Collection[str] | None = None,
) -> str | None:
    """Render perturbed belief heatmap for one model/strategy."""
    signal_root = canonical_root / str(model) / str(strategy) / "signals"
    if not signal_root.exists():
        return f"Belief signal directory not found: {signal_root}."
    excluded = {
        str(v) for v in (BELIEF_PLOT_EXCLUDED_CONSTRUCTS if excluded_constructs is None else excluded_constructs)
    }
    bucket_lookup = {str(k): str(v) for k, v in (signal_bucket_map or {}).items()}

    def _std_cosine(x: np.ndarray, y: np.ndarray) -> float:
        mask = np.isfinite(x) & np.isfinite(y)
        if int(mask.sum()) < 3:
            return float("nan")
        xv = x[mask]
        yv = y[mask]
        sx = float(np.std(xv, ddof=1))
        sy = float(np.std(yv, ddof=1))
        if not np.isfinite(sx) or not np.isfinite(sy) or sx <= 0.0 or sy <= 0.0:
            return float("nan")
        xz = (xv - float(np.mean(xv))) / sx
        yz = (yv - float(np.mean(yv))) / sy
        nx = float(np.linalg.norm(xz))
        ny = float(np.linalg.norm(yz))
        if nx <= 0.0 or ny <= 0.0:
            return float("nan")
        return float(np.dot(xz, yz) / (nx * ny))

    row_ids: list[str] = []
    row_engine_scores: dict[tuple[str, str], list[float]] = {}
    for signal_dir in sorted(p for p in signal_root.iterdir() if p.is_dir() and not p.name.startswith("_")):
        sid = str(signal_dir.name)
        if sid == "__unperturbed__":
            continue
        p = signal_dir / "metrics.parquet"
        if not p.exists():
            continue
        try:
            df = pl.read_parquet(p)
        except Exception:
            continue
        if df.is_empty():
            continue
        row_id = bucket_lookup.get(sid, sid) if bucket_lookup else sid
        row_ids.append(row_id)
        for engine in BELIEF_OUTCOME_ORDER:
            if str(engine) in excluded:
                continue
            ref_col = f"pe_ref_p_std_{engine}"
            llm_col = f"pe_llm_p_std_{engine}"
            if ref_col not in df.columns or llm_col not in df.columns:
                continue
            x = df.select(pl.col(ref_col).cast(pl.Float64, strict=False)).to_series().to_numpy()
            y = df.select(pl.col(llm_col).cast(pl.Float64, strict=False)).to_series().to_numpy()
            score = _std_cosine(x, y)
            if np.isfinite(score):
                row_engine_scores.setdefault((row_id, engine), []).append(float(score))

    if not row_ids:
        return "No perturbed signal metrics found for the selected model/strategy."
    if bucket_lookup and signal_bucket_order:
        observed_rows = set(row_ids)
        row_ids = [str(b) for b in signal_bucket_order]
        row_ids.extend(sorted(r for r in observed_rows if r not in set(row_ids)))
    else:
        row_ids = sorted(set(row_ids))
    engines = [str(e) for e in BELIEF_OUTCOME_ORDER if str(e) not in excluded]
    if not engines:
        return "No psychological-profile-variable perturbed metrics found for the selected model/strategy."

    mat = np.full((len(row_ids), len(engines)), np.nan, dtype=float)
    for i, row_id in enumerate(row_ids):
        for j, engine in enumerate(engines):
            vals = row_engine_scores.get((row_id, engine), [])
            mat[i, j] = float(np.median(np.asarray(vals, dtype=float))) if vals else float("nan")
    if not np.isfinite(mat).any():
        return "No finite perturbed signal×engine cosine values available."

    cmap = plt.get_cmap("cividis")
    cmap_use = cmap.copy()
    cmap_use.set_bad(color="#f0f0f0")
    norm = _heatmap_norm(vmin=cosine_vmin, vmax=cosine_vmax, vcenter=0.0)
    fig, ax = plt.subplots(
        figsize=(2.6 + 1.35 * len(engines), 2.0 + 0.44 * len(row_ids)),
        constrained_layout=True,
    )
    im = ax.imshow(np.ma.masked_invalid(mat), norm=norm, cmap=cmap_use, aspect="auto")
    ax.set_xticks(np.arange(len(engines)), labels=[label_engine(e) for e in engines], rotation=30, ha="right")
    if label_signal is None:
        ylabels = [rid.replace("_", " ") for rid in row_ids]
    else:
        ylabels = [label_signal(rid) for rid in row_ids]
    ax.set_yticks(np.arange(len(row_ids)), labels=ylabels)
    ax.tick_params(axis="x", labelsize=8)
    ax.tick_params(axis="y", labelsize=8)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat[i, j]
            if not np.isfinite(val):
                ax.add_patch(
                    Rectangle(
                        (j - 0.5, i - 0.5),
                        1,
                        1,
                        facecolor="#f0f0f0",
                        edgecolor="#cfcfcf",
                        hatch="///",
                        linewidth=0.0,
                        zorder=0,
                    )
                )
                ax.text(j, i, "NA", ha="center", va="center", fontsize=6, color="#8a8a8a")
                continue
            ax.text(
                j,
                i,
                f"{float(val):.2f}",
                ha="center",
                va="center",
                fontsize=7.5,
                color=_text_color_for_value(
                    float(val),
                    cmap=cmap_use,
                    vmin=float(cosine_vmin),
                    vmax=float(cosine_vmax),
                    vcenter=0.0,
                ),
            )
    cbar = fig.colorbar(im, ax=ax, shrink=0.9)
    cbar.set_label("Standardized cosine (0 = orthogonal)")
    plt.show()
    return None


def render_belief_signal_block_sensitivity_alignment(
    *,
    canonical_root: Path,
    model: str | None,
    strategy: str | None,
    signal_bucket_map: Mapping[str, str],
    signal_bucket_order: Sequence[str],
    label_signal_bucket: Callable[[str], str],
    epsilon_ref: float | None = None,
) -> str | None:
    """Render signal-block sensitivity alignment on the active reference set."""
    model_filter = str(model).strip() if model is not None else ""
    strategy_filter = str(strategy).strip() if strategy is not None else ""
    epsilon_ref_use = float(epsilon_ref) if epsilon_ref is not None else 0.03
    bucket_lookup = {str(k): str(v) for k, v in signal_bucket_map.items()}
    compact_candidates = (
        canonical_root / "signal_block_alignment.csv",
        canonical_root.parent / "signal_block_alignment.csv",
        canonical_root.parent.parent / "signal_block_alignment.csv",
    )
    compact_path = next((path for path in compact_candidates if path.exists()), None)
    if compact_path is not None:
        compact_df = pl.read_csv(compact_path)
        if compact_df.is_empty():
            return "No belief signal-block sensitivity rows available in compact artifact."
        if model_filter and "model" in compact_df.columns:
            compact_df = compact_df.filter(pl.col("model") == model_filter)
        if strategy_filter and "strategy" in compact_df.columns:
            compact_df = compact_df.filter(pl.col("strategy") == strategy_filter)
        if compact_df.is_empty():
            return "No belief signal-block sensitivity rows match the selected model/strategy."
        df = (
            compact_df.select(
                pl.col("signal_id").cast(pl.Utf8),
                pl.col("ref_abs_median").cast(pl.Float64, strict=False).alias("ref_abs"),
                pl.col("llm_abs_median").cast(pl.Float64, strict=False).alias("llm_abs"),
            )
            .with_columns(
                pl.col("signal_id")
                .map_elements(lambda sid: bucket_lookup.get(str(sid), "other"), return_dtype=pl.Utf8)
                .alias("signal_block")
            )
            .filter(
                pl.col("ref_abs").is_finite()
                & pl.col("llm_abs").is_finite()
                & (pl.col("ref_abs") > float(epsilon_ref_use))
            )
        )
        if df.is_empty():
            return f"No signal-block sensitivity rows available on active set (|PE_ref,std| > {epsilon_ref_use:.3g})."
        agg = df.group_by("signal_block").agg(
            pl.median("ref_abs").alias("ref_abs"),
            pl.median("llm_abs").alias("llm_abs"),
        )
    else:
        signal_roots: list[Path] = []
        for model_dir in sorted(p for p in canonical_root.iterdir() if p.is_dir() and not p.name.startswith("_")):
            if model_filter and model_dir.name != model_filter:
                continue
            for strategy_dir in sorted(p for p in model_dir.iterdir() if p.is_dir() and not p.name.startswith("_")):
                if strategy_filter and strategy_dir.name != strategy_filter:
                    continue
                signal_root = strategy_dir / "signals"
                if signal_root.exists():
                    signal_roots.append(signal_root)
        if not signal_roots:
            selected = []
            if model_filter:
                selected.append(f"model={model_filter}")
            if strategy_filter:
                selected.append(f"strategy={strategy_filter}")
            details = f" ({', '.join(selected)})" if selected else ""
            return f"Belief signal directories not found under canonical root{details}."

        rows: list[pl.DataFrame] = []
        for signal_root in signal_roots:
            for signal_dir in sorted(p for p in signal_root.iterdir() if p.is_dir() and not p.name.startswith("_")):
                sid = str(signal_dir.name)
                if sid == "__unperturbed__":
                    continue
                p = signal_dir / "pe_alignment_rows.parquet"
                if not p.exists():
                    continue
                try:
                    df = pl.read_parquet(p)
                except Exception:
                    continue
                if df.is_empty():
                    continue
                if "pe_ref_p_std" not in df.columns or "pe_llm_p_std" not in df.columns:
                    continue
                bucket = bucket_lookup.get(sid, "other")
                active = df.select(
                    pl.lit(bucket).alias("signal_block"),
                    pl.col("pe_ref_p_std").cast(pl.Float64, strict=False).abs().alias("ref_abs"),
                    pl.col("pe_llm_p_std").cast(pl.Float64, strict=False).abs().alias("llm_abs"),
                ).filter(
                    pl.col("ref_abs").is_finite()
                    & pl.col("llm_abs").is_finite()
                    & (pl.col("ref_abs") > float(epsilon_ref_use))
                )
                if active.is_empty():
                    continue
                rows.append(active)

        if not rows:
            return f"No signal-block sensitivity rows available on active set (|PE_ref,std| > {epsilon_ref_use:.3g})."

        df = pl.concat(rows, how="vertical")
        agg = df.group_by("signal_block").agg(
            pl.median("ref_abs").alias("ref_abs"),
            pl.median("llm_abs").alias("llm_abs"),
        )
    if agg.is_empty():
        return "No aggregated signal-block sensitivity values available."

    blocks = [b for b in signal_bucket_order if b in set(agg.get_column("signal_block").to_list())]
    blocks.extend(sorted(b for b in agg.get_column("signal_block").to_list() if b not in set(blocks)))
    agg = (
        agg.with_columns(
            pl.col("signal_block")
            .map_elements(lambda b: blocks.index(str(b)) if str(b) in set(blocks) else 9999, return_dtype=pl.Int64)
            .alias("_b")
        )
        .sort("_b")
        .drop("_b")
    )

    x = agg.get_column("ref_abs").to_numpy()
    y = agg.get_column("llm_abs").to_numpy()
    if x.size == 0 or y.size == 0:
        return "No finite signal-block sensitivity values available."
    upper = float(np.nanmax(np.concatenate([x, y])))
    upper = 1.05 * upper if np.isfinite(upper) and upper > 0 else 1.0

    _, ax = plt.subplots(figsize=(5.8, 4.6), constrained_layout=True)
    cmap = plt.get_cmap("tab10")
    preferred_offsets: dict[str, tuple[float, float]] = {
        "constraints_access": (6.0, 2.0),
        "institutional_credibility": (8.0, 6.0),
        "disease_stakes_salience": (10.0, 2.0),
        "vaccine_risk_cost_salience": (18.0, 0.0),
        "interpersonal_channel": (10.0, -6.0),
        "information_environment_credibility": (10.0, -14.0),
    }
    placed_labels: list[tuple[float, float]] = []
    candidate_offsets: tuple[tuple[float, float], ...] = (
        (8.0, 6.0),
        (8.0, -10.0),
        (12.0, 0.0),
        (-12.0, 6.0),
        (-12.0, -10.0),
        (0.0, 10.0),
        (0.0, -12.0),
        (16.0, 8.0),
        (16.0, -12.0),
    )

    def _label_offset(x_val: float, y_val: float) -> tuple[float, float]:
        anchor_x, anchor_y = ax.transData.transform((x_val, y_val))
        best_offset = candidate_offsets[0]
        best_score = float("-inf")
        for dx, dy in candidate_offsets:
            label_x = anchor_x + dx
            label_y = anchor_y + dy
            min_sep = min(
                (float(np.hypot(label_x - other_x, label_y - other_y)) for other_x, other_y in placed_labels),
                default=float("inf"),
            )
            edge_penalty = max(0.0, 36.0 - label_x) + max(0.0, label_y - (ax.bbox.ymax - 16.0))
            score = min_sep - edge_penalty
            if score > best_score:
                best_score = score
                best_offset = (dx, dy)
        placed_labels.append((anchor_x + best_offset[0], anchor_y + best_offset[1]))
        return best_offset

    for i, row in enumerate(agg.iter_rows(named=True)):
        block = str(row.get("signal_block"))
        xv = float(row.get("ref_abs", float("nan")))
        yv = float(row.get("llm_abs", float("nan")))
        if not (np.isfinite(xv) and np.isfinite(yv)):
            continue
        ax.scatter(xv, yv, s=62, color=cmap(i % 10), edgecolors="white", linewidths=0.6, alpha=0.9)
        dx, dy = preferred_offsets.get(block, _label_offset(xv, yv))
        ha = "left" if dx >= 0 else "right"
        va = "bottom" if dy >= 0 else "top"
        ax.annotate(
            label_signal_bucket(block),
            (xv, yv),
            textcoords="offset points",
            xytext=(dx, dy),
            ha=ha,
            va=va,
            fontsize=8.5,
            bbox={"boxstyle": "round,pad=0.12", "fc": "white", "ec": "none", "alpha": 0.7},
        )

    ax.plot([0.0, upper], [0.0, upper], color="black", linewidth=1.0, alpha=0.6)
    ax.set_xlim(0.0, upper)
    ax.set_ylim(0.0, upper)
    ax.grid(True, linestyle=":", alpha=0.35)
    ax.set_xlabel(
        f"Reference block sensitivity (active set: |PE_ref,std| > {epsilon_ref_use:.3g})",
        fontsize=10.5,
    )
    ax.set_ylabel(
        f"LLM block sensitivity (same active set: |PE_ref,std| > {epsilon_ref_use:.3g})",
        fontsize=10.5,
    )
    ax.tick_params(axis="both", labelsize=9.5)
    plt.show()
    return None


def render_belief_vector_importance_alignment_by_signal_block(
    *,
    canonical_root: Path,
    preferred_model: str,
    preferred_strategy: str,
    signal_bucket_map: Mapping[str, str],
    signal_bucket_order: Sequence[str],
    label_engine: Callable[[str], str],
    label_signal_bucket: Callable[[str], str],
    label_model: Callable[[str], str] | None = None,
    label_strategy: Callable[[str], str] | None = None,
    standardize_by_ref_sigma: bool = True,
    excluded_constructs: Collection[str] | None = None,
) -> str | None:
    """Render belief construct-importance rank alignment faceted by signal bucket."""
    if not canonical_root.exists():
        return f"Belief canonical root not found for rank-alignment plot: {canonical_root}."

    model_dirs = sorted(p for p in canonical_root.iterdir() if p.is_dir() and not p.name.startswith("_"))
    if not model_dirs:
        return "No belief canonical model directories available for rank-alignment plotting."
    available_models = [d.name for d in model_dirs]
    chosen_model = preferred_model if preferred_model in available_models else available_models[0]

    signal_root = canonical_root / chosen_model / preferred_strategy / "signals"
    strategy_dirs = sorted(
        p
        for p in (canonical_root / chosen_model).iterdir()
        if p.is_dir() and not p.name.startswith("_") and (p / "signals").exists()
    )
    if not strategy_dirs:
        return "No belief signal metrics available for the selected model."
    available_strategies = [d.name for d in strategy_dirs]
    chosen_strategy = preferred_strategy if preferred_strategy in available_strategies else available_strategies[0]
    signal_root = canonical_root / chosen_model / chosen_strategy / "signals"
    if not signal_root.exists():
        return "No signal directory available for the selected model/strategy."

    excluded = {
        str(v) for v in (BELIEF_PLOT_EXCLUDED_CONSTRUCTS if excluded_constructs is None else excluded_constructs)
    }
    bucket_lookup = {str(k): str(v) for k, v in signal_bucket_map.items()}
    bucket_frames: dict[str, list[pl.DataFrame]] = {}
    for signal_dir in sorted(p for p in signal_root.iterdir() if p.is_dir() and not p.name.startswith("_")):
        sid = str(signal_dir.name)
        if sid == "__unperturbed__":
            continue
        metrics_path = signal_dir / "metrics.parquet"
        if not metrics_path.exists():
            continue
        try:
            df = pl.read_parquet(metrics_path)
        except Exception:
            continue
        if df.is_empty():
            continue
        bucket = bucket_lookup.get(sid, "other")
        bucket_frames.setdefault(bucket, []).append(df)

    if not bucket_frames:
        return "No belief perturbed signal metrics available for signal-bucket rank alignment."

    bucket_use = [str(b) for b in signal_bucket_order if str(b) in set(bucket_frames.keys())]
    bucket_use.extend(sorted(b for b in bucket_frames.keys() if b not in set(bucket_use)))
    if not bucket_use:
        return "No signal buckets available for rank-alignment plotting."

    rows_by_bucket: dict[str, list[dict[str, object]]] = {}
    for bucket in bucket_use:
        df = pl.concat(bucket_frames[bucket], how="vertical_relaxed") if bucket_frames[bucket] else pl.DataFrame()
        if df.is_empty():
            continue
        ref_prefix = "pre_"
        llm_prefix = "post_"
        state_vars = [
            str(v)
            for v in BELIEF_OUTCOME_ORDER
            if f"{ref_prefix}{v}" in df.columns and f"{llm_prefix}{v}" in df.columns and str(v) not in excluded
        ]
        extras = sorted(
            {
                col[len(ref_prefix) :]
                for col in df.columns
                if col.startswith(ref_prefix)
                and f"{llm_prefix}{col[len(ref_prefix) :]}" in df.columns
                and col[len(ref_prefix) :] not in set(state_vars)
                and col[len(ref_prefix) :] not in excluded
            }
        )
        state_vars.extend(extras)
        imp_rows: list[dict[str, object]] = []
        for state_var in state_vars:
            ref_vals = (
                df.select(pl.col(f"{ref_prefix}{state_var}").cast(pl.Float64, strict=False)).to_series().to_numpy()
            )
            llm_vals = (
                df.select(pl.col(f"{llm_prefix}{state_var}").cast(pl.Float64, strict=False)).to_series().to_numpy()
            )
            ref_vals = ref_vals[np.isfinite(ref_vals)]
            llm_vals = llm_vals[np.isfinite(llm_vals)]
            if ref_vals.size == 0 or llm_vals.size == 0:
                continue
            sigma = float(np.std(ref_vals, ddof=1)) if ref_vals.size > 1 else float("nan")
            if not np.isfinite(sigma) or sigma <= 0.0 or not standardize_by_ref_sigma:
                sigma = 1.0
            ref_importance = float(np.median(np.abs(ref_vals)) / sigma)
            llm_importance = float(np.median(np.abs(llm_vals)) / sigma)
            if not (np.isfinite(ref_importance) and np.isfinite(llm_importance)):
                continue
            imp_rows.append(
                {
                    "state_var": state_var,
                    "ref_importance": ref_importance,
                    "llm_importance": llm_importance,
                }
            )
        if len(imp_rows) >= 2:
            rows_by_bucket[bucket] = imp_rows

    if not rows_by_bucket:
        return "Insufficient finite construct-importance values across signal buckets."

    bucket_plot = [b for b in bucket_use if b in rows_by_bucket]
    n_panels = len(bucket_plot)
    ncols = min(3, n_panels)
    nrows = int(np.ceil(n_panels / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(2.5 + 3.2 * ncols, 1.6 + 2.9 * nrows),
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes).ravel()
    cmap = plt.get_cmap("tab10")
    all_state_vars = list(
        dict.fromkeys(str(r["state_var"]) for rows in rows_by_bucket.values() for r in rows if "state_var" in r)
    )
    color_map = {var: cmap(i % 10) for i, var in enumerate(all_state_vars)}

    for idx, bucket in enumerate(bucket_plot):
        ax = axes[idx]
        imp_rows = rows_by_bucket[bucket]
        state_vars_plot = [str(r["state_var"]) for r in imp_rows]
        ref_vals = [float(r["ref_importance"]) for r in imp_rows]
        llm_vals = [float(r["llm_importance"]) for r in imp_rows]
        rank_ref = _rank_desc(ref_vals, state_vars_plot)
        rank_llm = _rank_desc(llm_vals, state_vars_plot)
        x = np.array([float(rank_ref[v]) for v in state_vars_plot], dtype=float)
        y = np.array([float(rank_llm[v]) for v in state_vars_plot], dtype=float)
        rho = _spearman_from_ranks(x, y)
        max_rank = int(max(np.max(x), np.max(y)))

        for state_var in state_vars_plot:
            xv = float(rank_ref[state_var])
            yv = float(rank_llm[state_var])
            ax.scatter(
                xv,
                yv,
                s=55.0,
                color=color_map.get(state_var, "#4c72b0"),
                edgecolors="white",
                linewidths=0.5,
                alpha=0.9,
            )
            ax.annotate(
                label_engine(state_var),
                (xv, yv),
                textcoords="offset points",
                xytext=(5, 3),
                fontsize=7,
                bbox={"boxstyle": "round,pad=0.12", "fc": "white", "ec": "none", "alpha": 0.7},
            )

        dmin = 0.5
        dmax = float(max_rank) + 0.5
        ax.plot([dmin, dmax], [dmin, dmax], color="black", linewidth=1.0, alpha=0.6)
        ax.set_xlim(dmin, dmax)
        ax.set_ylim(dmin, dmax)
        ax.set_xticks(np.arange(1, max_rank + 1))
        ax.set_yticks(np.arange(1, max_rank + 1))
        ax.grid(True, linestyle=":", alpha=0.35)
        rho_txt = f"$\\rho={rho:.2f}$" if np.isfinite(rho) else "$\\rho$ unavailable"
        ax.set_title(f"{label_signal_bucket(bucket)} ({rho_txt})", fontsize=9.5)
        ax.set_xlabel("Ref rank", fontsize=8)
        ax.set_ylabel("LLM rank", fontsize=8)

    for ax in axes[n_panels:]:
        ax.axis("off")

    model_txt = label_model(chosen_model) if label_model is not None else chosen_model
    strategy_txt = label_strategy(chosen_strategy) if label_strategy is not None else chosen_strategy
    fig.suptitle(f"{model_txt}, {strategy_txt}", fontsize=10.5)
    plt.show()
    return None


def _normalize_block_expr(column: str = "block") -> pl.Expr:
    base = pl.col(column).cast(pl.String).str.strip_chars()
    return base.replace_strict(BLOCK_NORMALIZE_MAP, default=base)


def _rank_desc(values: list[float], names: list[str]) -> dict[str, int]:
    order = sorted(zip(values, names, strict=False), key=lambda t: (-float(t[0]), str(t[1])))
    return {name: i + 1 for i, (_, name) in enumerate(order)}


def _spearman_from_ranks(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or y.size < 2:
        return float("nan")
    if np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _pick_primary_model_strategy(
    *,
    outcome_order: Sequence[str],
    preferred_model: str,
    preferred_strategy: str,
    perturbed_root: Path,
    ablation_summary_path: Path,
    csv_reader: Callable[[str | Path], pl.DataFrame],
) -> tuple[str | None, str]:
    model_candidates: list[str] = []
    strategy_candidates: list[str] = []
    for outcome in outcome_order:
        out_dir = perturbed_root / str(outcome)
        if not out_dir.exists():
            continue
        for model_dir in out_dir.iterdir():
            if not model_dir.is_dir():
                continue
            model_candidates.append(model_dir.name)
            for strat_dir in model_dir.iterdir():
                if not strat_dir.is_dir():
                    continue
                if (strat_dir / "uncalibrated" / "_block_alignment.csv").exists() or (
                    strat_dir / "calibrated" / "_block_alignment.csv"
                ).exists():
                    strategy_candidates.append(strat_dir.name)
    if (not model_candidates or not strategy_candidates) and ablation_summary_path.exists():
        ablation_df = csv_reader(ablation_summary_path)
        if not ablation_df.is_empty() and {"outcome", "model", "strategy"}.issubset(set(ablation_df.columns)):
            if outcome_order:
                ablation_df = ablation_df.filter(pl.col("outcome").is_in([str(o) for o in outcome_order]))
            if not ablation_df.is_empty():
                model_candidates.extend([str(v) for v in ablation_df.get_column("model").unique().to_list()])
                strategy_candidates.extend([str(v) for v in ablation_df.get_column("strategy").unique().to_list()])
    model_candidates = sorted(dict.fromkeys(model_candidates))
    strategy_candidates = sorted(dict.fromkeys(strategy_candidates))
    model = (
        preferred_model if preferred_model in model_candidates else (model_candidates[0] if model_candidates else None)
    )
    strategy = (
        preferred_strategy
        if preferred_strategy in strategy_candidates
        else (strategy_candidates[0] if strategy_candidates else preferred_strategy)
    )
    return model, strategy


def _ref_lobo_size(
    *,
    outcome: str,
    model_ref: str | None,
    ref_lobo_path: Path,
    csv_reader: Callable[[str | Path], pl.DataFrame],
) -> pl.DataFrame:
    if not ref_lobo_path.exists():
        return pl.DataFrame()
    ref_lobo = csv_reader(ref_lobo_path)
    if ref_lobo.is_empty():
        return pl.DataFrame()
    df = ref_lobo.filter((pl.col("outcome") == outcome) & (pl.col("kind") == "lobo") & (pl.col("metric") == "bic"))
    if model_ref:
        df = df.filter(pl.col("model") == model_ref)
    else:
        df = df.sort("delta", descending=True).unique(subset=["block"], keep="first")
    if df.is_empty():
        return pl.DataFrame()
    return (
        df.select(["block", "delta"])
        .with_columns(_normalize_block_expr("block").alias("block"))
        .with_columns(pl.col("delta").cast(pl.Float64).alias("lobo_bic"))
        .select(["block", "lobo_bic"])
    )


def _ablation_importance(
    *,
    outcome: str,
    model: str | None,
    strategy: str | None,
    ablation_summary_path: Path,
    csv_reader: Callable[[str | Path], pl.DataFrame],
) -> pl.DataFrame:
    if not ablation_summary_path.exists():
        return pl.DataFrame()
    df = csv_reader(ablation_summary_path)
    if df.is_empty() or "delta_gini" not in df.columns:
        return pl.DataFrame()
    out = df.filter(pl.col("outcome") == outcome)
    if model is not None and str(model).strip():
        out = out.filter(pl.col("model") == str(model))
    if strategy is not None and str(strategy).strip():
        out = out.filter(pl.col("strategy") == str(strategy))
    if out.is_empty():
        return pl.DataFrame()
    return (
        out.select(["block", "delta_gini"])
        .with_columns(_normalize_block_expr("block").alias("block"))
        .with_columns(pl.col("delta_gini").cast(pl.Float64).alias("llm_importance"))
        .group_by("block")
        .agg(pl.median("llm_importance").alias("llm_importance"))
        .select(["block", "llm_importance"])
    )


def render_block_importance_alignment_gini(
    *,
    outcome_order: Sequence[str],
    label_outcome_short: Callable[[str], str],
    label_block: Callable[[str], str] | None = None,
    block_labels: Mapping[str, str] | None = None,
    block_order_for_outcome: Callable[[str], Sequence[str]] | None = None,
    reference_model_by_outcome: Mapping[str, str] | None = None,
    ref_lobo_path: Path = Path("../thesis/artifacts/empirical/modeling/model_lobo_all.csv"),
    ablation_summary_path: Path = Path(
        "../thesis/artifacts/evaluation/choice_validation/unperturbed_block_ablation.csv"
    ),
    csv_reader: Callable[[str | Path], pl.DataFrame] = read_csv_fast,
    panel_nrows: int | None = None,
    panel_ncols: int | None = None,
) -> str | None:
    """Render the chapter-4 block-importance alignment plot for Gini only.

    Parameters
    ----------
    block_labels: Optional mapping from block id to display string. If provided,
        these labels take precedence when constructing the legend. This makes it
        easy to ensure that the legend uses a consistent dictionary of human-
        readable block names even if ``label_block`` is omitted or behaves
        differently.
    """
    if not outcome_order:
        return "No outcomes available for the block-importance rank plot."

    panel_title_size = 12
    axis_label_size = 12
    axis_tick_size = 12
    rho_text_size = 12
    legend_size = 12
    legend_title_size = 12
    suptitle_size = 12

    ablation_available_outcomes: set[str] = set()
    if ablation_summary_path.exists():
        ablation_scan = csv_reader(ablation_summary_path)
        if not ablation_scan.is_empty() and "outcome" in ablation_scan.columns:
            ablation_available_outcomes = {
                str(value) for value in ablation_scan.get_column("outcome").drop_nulls().to_list() if str(value).strip()
            }

    ablation_outcome_candidates: dict[str, tuple[str, ...]] = {
        "vax_willingness_T12": ("vax_willingness_T12", "vax_willingness_Y"),
        "flu_vaccinated_2023_2024": ("flu_vaccinated_2023_2024", "flu_vaccinated_2023_2024_no_habit"),
    }
    ref_model_overrides: dict[str, str] = {
        "flu_vaccinated_2023_2024": "B_FLU_vaccinated_2023_2024_ENGINES",
    }

    def _lookup_ablation_outcome(outcome: str) -> str:
        canonical = canonical_outcome_name(outcome)
        candidates = ablation_outcome_candidates.get(canonical, (canonical,))
        if not ablation_available_outcomes:
            return candidates[0]
        for candidate in candidates:
            if candidate in ablation_available_outcomes:
                return candidate
        return candidates[0]

    def _lookup_ref_outcome(outcome: str) -> str:
        return canonical_outcome_name(outcome)

    def _label_outcome_key(outcome: str) -> str:
        return canonical_outcome_name(outcome)

    # The unperturbed block-ablation outputs expose country controls separately,
    # while the reference LOBO summary keeps them inside the broader controls
    # block. Remap only for this comparison so the blocks can be aligned.
    alignment_block_map: dict[str, str] = {
        "controls_country": "controls_demography_SES",
    }

    def _normalize_alignment_block_expr(column: str = "block") -> pl.Expr:
        base = _normalize_block_expr(column)
        return base.replace_strict(alignment_block_map, default=base)

    use_label_block = label_block or label_choice_engine_block
    use_block_order = block_order_for_outcome or (lambda outcome: [])
    ref_model_map = dict(reference_model_by_outcome or {})

    rows: list[dict[str, object]] = []
    for outcome in outcome_order:
        display_outcome = str(outcome)
        lookup_ref_outcome = _lookup_ref_outcome(display_outcome)
        lookup_ablation_outcome = _lookup_ablation_outcome(display_outcome)
        model_ref = (
            ref_model_overrides.get(display_outcome)
            or ref_model_map.get(lookup_ref_outcome)
            or ref_model_map.get(display_outcome)
        )
        ref_lobo = _ref_lobo_size(
            outcome=lookup_ref_outcome,
            model_ref=model_ref,
            ref_lobo_path=ref_lobo_path,
            csv_reader=csv_reader,
        )
        ref_block = ref_lobo.rename({"lobo_bic": "ref_importance"}) if not ref_lobo.is_empty() else pl.DataFrame()
        if ref_block.is_empty():
            continue
        ref_block = ref_block.with_columns(_normalize_alignment_block_expr("block").alias("block"))

        llm_block = _ablation_importance(
            outcome=lookup_ablation_outcome,
            model=None,
            strategy=None,
            ablation_summary_path=ablation_summary_path,
            csv_reader=csv_reader,
        )
        if llm_block.is_empty():
            continue
        llm_block = llm_block.with_columns(_normalize_alignment_block_expr("block").alias("block"))

        merged = ref_block.join(llm_block, on="block", how="inner")
        for row in merged.iter_rows(named=True):
            rows.append(
                {
                    "outcome": str(outcome),
                    "metric_label": "$\\Delta G$ rank",
                    "block": str(row.get("block")),
                    "ref_importance": row.get("ref_importance"),
                    "llm_importance": row.get("llm_importance"),
                }
            )

    df = pl.DataFrame(rows) if rows else pl.DataFrame()
    if df.is_empty():
        return "No block-importance data available for the rank plot."
    df = df.filter(pl.col("ref_importance").is_finite() & pl.col("llm_importance").is_finite())
    if df.is_empty():
        return "No finite block-importance values available for the rank plot."

    observed_blocks = [str(b) for b in df.get_column("block").unique().to_list() if str(b).strip()]
    preferred_order: list[str] = []
    for outcome in outcome_order:
        preferred_order.extend([str(v) for v in use_block_order(str(outcome))])
    preferred_order = [b for b in dict.fromkeys(preferred_order) if b in observed_blocks]
    blocks_all = preferred_order + [b for b in observed_blocks if b not in preferred_order]

    cmap = plt.get_cmap("tab20")
    stable_order = list(dict.fromkeys(DEFAULT_BLOCK_COLOR_ORDER + blocks_all))
    stable_color_map = {b: cmap(i % 20) for i, b in enumerate(stable_order)}
    color_map = {b: stable_color_map[b] for b in blocks_all}

    ranked_rows: list[dict[str, object]] = []
    rho_rows: list[dict[str, object]] = []
    for outcome in outcome_order:
        sub = df.filter(pl.col("outcome") == str(outcome))
        if sub.is_empty():
            continue
        blocks = [str(v) for v in sub.get_column("block").to_list()]
        ref_vals = [float(v) for v in sub.get_column("ref_importance").to_list()]
        llm_vals = [float(v) for v in sub.get_column("llm_importance").to_list()]
        rank_ref = _rank_desc(ref_vals, blocks)
        rank_llm = _rank_desc(llm_vals, blocks)
        x = np.array([float(rank_ref[b]) for b in blocks], dtype=float)
        y = np.array([float(rank_llm[b]) for b in blocks], dtype=float)
        rho_rows.append(
            {
                "outcome": str(outcome),
                "rho": _spearman_from_ranks(x, y),
            }
        )
        for block in blocks:
            ranked_rows.append(
                {
                    "outcome": str(outcome),
                    "block": block,
                    "rank_ref": float(rank_ref[block]),
                    "rank_llm": float(rank_llm[block]),
                }
            )

    rank_df = pl.DataFrame(ranked_rows) if ranked_rows else pl.DataFrame()
    rho_df = pl.DataFrame(rho_rows) if rho_rows else pl.DataFrame()
    if rank_df.is_empty():
        return "No ranked block-importance data available for the plot."

    max_rank = int(max(rank_df.get_column("rank_ref").max(), rank_df.get_column("rank_llm").max()))
    if panel_nrows is not None and panel_ncols is not None:
        outcomes = [str(o) for o in outcome_order if str(o).strip()]
        nrows = max(1, int(panel_nrows))
        ncols = max(1, int(panel_ncols))
        if nrows * ncols < len(outcomes):
            return f"Requested panel layout {nrows}x{ncols} is too small for {len(outcomes)} outcomes."
        panel_slots = outcomes + [None] * (nrows * ncols - len(outcomes))
    else:
        nrows, ncols, panel_slots = outcome_panel_layout(outcome_order)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(2.8 + 2.5 * max(1, ncols), 1.8 + 2.2 * nrows),
        sharex=True,
        sharey=True,
        constrained_layout=False,
    )
    axes = np.atleast_1d(axes).ravel()

    rho_vals = rho_df.select(pl.col("rho")).to_series().to_numpy() if not rho_df.is_empty() else np.array([])
    rho_vals = rho_vals[np.isfinite(rho_vals)] if rho_vals.size else rho_vals
    rho_summary_txt = (
        rf"Median $\rho$ = {float(np.median(rho_vals)):.2f}" if rho_vals.size else "No finite rank-correlation summary"
    )

    occupied_by_col: dict[int, list[int]] = {}
    for idx, outcome in enumerate(panel_slots):
        if outcome is None:
            continue
        occupied_by_col.setdefault(idx % ncols, []).append(idx)
    bottom_idx_by_col = {col: max(indices) for col, indices in occupied_by_col.items()}

    for j, outcome in enumerate(panel_slots):
        ax = axes[j]
        if outcome is None:
            ax.set_axis_off()
            continue
        sub = rank_df.filter(pl.col("outcome") == str(outcome))
        if sub.is_empty():
            ax.set_title(label_outcome_short(_label_outcome_key(str(outcome))), fontsize=panel_title_size)
            ax.text(
                0.5,
                0.5,
                "No block-ablation runs",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=axis_tick_size,
                color="#666666",
            )
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            continue
        for row in sub.iter_rows(named=True):
            block = str(row.get("block"))
            ax.scatter(
                float(row.get("rank_ref")),
                float(row.get("rank_llm")),
                s=55.0,
                color=color_map.get(block, "#4c72b0"),
                alpha=0.85,
                edgecolor="black",
                linewidth=0.35,
            )
        ax.plot([0.7, max_rank + 0.3], [0.7, max_rank + 0.3], color="black", linewidth=1.0, alpha=0.6)
        rho_cell = rho_df.filter(pl.col("outcome") == str(outcome)) if not rho_df.is_empty() else pl.DataFrame()
        if not rho_cell.is_empty():
            rho = rho_cell.select("rho").row(0)[0]
            if rho is not None and np.isfinite(float(rho)):
                ax.text(
                    0.03,
                    0.94,
                    f"$\\rho={float(rho):.2f}$",
                    transform=ax.transAxes,
                    va="top",
                    ha="left",
                    fontsize=rho_text_size,
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.7, "pad": 0.4},
                )
        ax.set_xlim(0.7, max_rank + 0.3)
        ax.set_ylim(max_rank + 0.3, 0.7)
        ax.set_xticks(np.arange(1, max_rank + 1))
        ax.set_yticks(np.arange(1, max_rank + 1))
        ax.tick_params(axis="both", labelsize=axis_tick_size)
        if j in bottom_idx_by_col.values():
            ax.tick_params(axis="x", labelbottom=True)
        ax.grid(True, alpha=0.15, linewidth=0.5)
        ax.set_title(
            textwrap.fill(label_outcome_short(_label_outcome_key(str(outcome))), width=22), fontsize=panel_title_size
        )

    handles = []
    for block in blocks_all:
        if block_labels and block in block_labels:
            display_label = str(block_labels[block])
        else:
            display_label = use_label_block(block)
        handles.append(
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label=display_label,
                markerfacecolor=color_map.get(block, "#4c72b0"),
                markersize=8,
            )
        )
    fig.legend(
        handles=handles,
        title="Block",
        frameon=False,
        fontsize=legend_size,
        loc="center left",
        bbox_to_anchor=(0.81, 0.5),
        title_fontsize=legend_title_size,
    )
    fig.supxlabel("Reference LOBO importance rank (1 = most important)", fontsize=axis_label_size)
    fig.supylabel("LLM block-importance rank (1 = most important)", fontsize=axis_label_size)
    fig.suptitle(
        "Block-importance rank alignment (median across model x strategy)\n" + rho_summary_txt,
        fontsize=suptitle_size,
    )
    fig.subplots_adjust(right=0.77, top=0.86, bottom=0.13, left=0.12, wspace=0.28, hspace=0.42)
    plt.show()
    return None


def render_perturbed_pareto_by_model(
    *,
    pert_summary: pl.DataFrame,
    pert_model_order: Sequence[str],
    pert_strategy_order: Sequence[str],
    outcome_order: Sequence[str],
    label_strategy: Callable[[str], str],
    label_model: Callable[[str], str],
    model_filter: Sequence[str] | None = None,
    use_dar_size: bool = True,
    show_model_legend: bool = True,
    highlight_strategies: Sequence[str] | None = None,
    highlight_color: str = "#DC2626",
    x_limits: tuple[float, float] | None = None,
    output_path: str | Path | None = None,
) -> str | None:
    if pert_summary.is_empty():
        return "No perturbed outputs found for the global performance map."

    rows: list[dict[str, object]] = []
    for model in pert_model_order:
        for strategy in pert_strategy_order:
            parts: list[dict[str, float]] = []
            for outcome in outcome_order:
                sub = pert_summary.filter(
                    (pl.col("outcome") == outcome) & (pl.col("model") == model) & (pl.col("strategy") == strategy)
                ).filter(pl.col("gini_pe_std_abs").is_finite() & pl.col("skill_pe_std_mean").is_finite())
                if sub.is_empty():
                    continue
                r = sub.row(0, named=True)
                parts.append(
                    {
                        "gini": float(r.get("gini_pe_std_abs")),
                        "skill": float(r.get("skill_pe_std_mean")),
                        "dar": float(r.get("sign_agreement_rate"))
                        if r.get("sign_agreement_rate") is not None
                        else float("nan"),
                        "fp_move": float(r.get("false_positive_movement_rate"))
                        if r.get("false_positive_movement_rate") is not None
                        else float("nan"),
                    }
                )
            if not parts:
                continue
            rows.append(
                {
                    "model": model,
                    "strategy": strategy,
                    "gini": float(np.median([p["gini"] for p in parts])),
                    "skill": float(np.median([p["skill"] for p in parts])),
                    "dar": float(np.nanmedian(np.array([p["dar"] for p in parts], dtype=np.float64))),
                    "fp_move": float(np.nanmedian(np.array([p["fp_move"] for p in parts], dtype=np.float64))),
                }
            )

    if not rows:
        return "Insufficient data to construct the performance map."

    collapsed: dict[tuple[str, str], dict[str, list[float]]] = {}
    for row in rows:
        model_canonical = canonical_model_id_with_effort(str(row.get("model", "")))
        strategy = str(row.get("strategy", ""))
        key = (model_canonical, strategy)
        bucket = collapsed.setdefault(key, {"gini": [], "skill": [], "dar": [], "fp_move": []})
        bucket["gini"].append(float(row.get("gini", float("nan"))))
        bucket["skill"].append(float(row.get("skill", float("nan"))))
        bucket["dar"].append(float(row.get("dar", float("nan"))))
        bucket["fp_move"].append(float(row.get("fp_move", float("nan"))))

    deduped_rows: list[dict[str, object]] = []
    for (model, strategy), metrics in collapsed.items():
        deduped_rows.append(
            {
                "model": model,
                "strategy": strategy,
                "gini": float(np.nanmedian(np.array(metrics["gini"], dtype=np.float64))),
                "skill": float(np.nanmedian(np.array(metrics["skill"], dtype=np.float64))),
                "dar": float(np.nanmedian(np.array(metrics["dar"], dtype=np.float64))),
                "fp_move": float(np.nanmedian(np.array(metrics["fp_move"], dtype=np.float64))),
            }
        )

    df = pl.DataFrame(deduped_rows)
    if model_filter is not None:
        filter_ids = {
            canonical_model_id_with_effort(str(model))
            for model in model_filter
            if str(model).strip()
        }
        if filter_ids:
            df = df.filter(pl.col("model").is_in(sorted(filter_ids)))
    if df.is_empty():
        return "No perturbed outputs matched the requested model filter."

    model_set = set(df.get_column("model").to_list())
    strategy_set = set(df.get_column("strategy").to_list())
    preferred_models = list(dict.fromkeys(canonical_model_id_with_effort(m) for m in pert_model_order))
    model_order = [m for m in preferred_models if m in model_set] + sorted(model_set - set(preferred_models))
    strategy_order = [s for s in PARETO_STRATEGY_ORDER if s in strategy_set] + sorted(
        strategy_set - set(PARETO_STRATEGY_ORDER)
    )
    model_marker_map = _build_pareto_model_marker_map(model_order)
    strategy_color_map = _build_pareto_strategy_color_map(strategy_order)

    dar_vals = np.array([], dtype=np.float64)
    if use_dar_size:
        dar_vals = df.select(pl.col("dar").cast(pl.Float64)).to_series().to_numpy()
        dar_vals = dar_vals[np.isfinite(dar_vals)]
    size_min, size_max = 45.0, 180.0
    if dar_vals.size:
        dar_vmin = float(np.min(dar_vals))
        dar_vmax = float(np.max(dar_vals))
        if dar_vmin >= dar_vmax:
            dar_vmin, dar_vmax = 0.0, 1.0
    else:
        dar_vmin, dar_vmax = 0.0, 1.0

    def _size_for_dar(value: float) -> float:
        if not use_dar_size:
            return 95.0
        if not np.isfinite(value):
            return 75.0
        t = (value - dar_vmin) / (dar_vmax - dar_vmin) if dar_vmax > dar_vmin else 0.5
        return float(size_min + float(np.clip(t, 0.0, 1.0)) * (size_max - size_min))

    panel_title_size = 12
    axis_label_size = 12
    axis_tick_size = 12
    guide_text_size = 12
    legend_size = 12
    legend_title_size = 12
    suptitle_size = 12

    fig, (ax_perf, ax_contain) = plt.subplots(2, 1, figsize=(8.0, 7.0), sharex=True)
    has_containment_points = False
    highlight_set = {str(strategy).strip() for strategy in (highlight_strategies or []) if str(strategy).strip()}
    for row in df.iter_rows(named=True):
        strategy = str(row["strategy"])
        model = str(row["model"])
        gini = float(row["gini"])
        skill = float(row["skill"])
        fp_move = float(row["fp_move"])
        containment_rate = 1.0 - fp_move if np.isfinite(fp_move) else float("nan")
        size = _size_for_dar(float(row.get("dar", float("nan"))))
        color = strategy_color_map.get(strategy, "#0F766E")
        marker = model_marker_map.get(model, "o") if show_model_legend else "o"

        if np.isfinite(skill):
            ax_perf.scatter(
                gini,
                skill,
                color=color,
                marker=marker,
                s=size,
                edgecolor="white",
                linewidth=0.5,
                alpha=0.92,
            )
            if strategy in highlight_set:
                ax_perf.scatter(
                    gini,
                    skill,
                    marker="s",
                    s=size * 1.9,
                    facecolors="none",
                    edgecolors=highlight_color,
                    linewidths=1.8,
                    zorder=5,
                )
        if np.isfinite(containment_rate):
            has_containment_points = True
            ax_contain.scatter(
                gini,
                containment_rate,
                color=color,
                marker=marker,
                s=size,
                edgecolor="white",
                linewidth=0.5,
                alpha=0.92,
            )
            if strategy in highlight_set:
                ax_contain.scatter(
                    gini,
                    containment_rate,
                    marker="s",
                    s=size * 1.9,
                    facecolors="none",
                    edgecolors=highlight_color,
                    linewidths=1.8,
                    zorder=5,
                )

    for ax in (ax_perf, ax_contain):
        ax.axvline(0, color="black", linewidth=1, alpha=0.6)
        ax.grid(True, alpha=0.15, linewidth=0.6)
        ax.set_xlabel("Median PE magnitude Gini (active set)", fontsize=axis_label_size)
        ax.tick_params(axis="both", labelsize=axis_tick_size)
    if x_limits is not None:
        x_left, x_right = x_limits
        if np.isfinite(x_left) and np.isfinite(x_right) and x_left < x_right:
            ax_perf.set_xlim(float(x_left), float(x_right))
            ax_contain.set_xlim(float(x_left), float(x_right))

    ax_perf.axhline(0, color="black", linewidth=1, alpha=0.25)
    ax_perf.set_ylabel("Median PE $\\mathrm{skill}$ (active set)", fontsize=axis_label_size)
    ax_perf.set_title("A) Performance on active set", fontsize=panel_title_size)
    xlim_perf = ax_perf.get_xlim()
    ylim_perf = ax_perf.get_ylim()
    x_guide_perf = max(0.003, 0.01 * (xlim_perf[1] - xlim_perf[0]))
    y_guide_perf = ylim_perf[0] + 0.06 * (ylim_perf[1] - ylim_perf[0])
    ax_perf.text(
        x_guide_perf,
        y_guide_perf,
        "better gradient ordering →",
        ha="left",
        va="bottom",
        fontsize=guide_text_size,
        color="#444444",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.7, "pad": 1.2},
    )
    ax_perf.text(
        0.02,
        0.98,
        "better magnitude accuracy ↑",
        transform=ax_perf.transAxes,
        ha="left",
        va="top",
        fontsize=guide_text_size,
        color="#444444",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.7, "pad": 1.2},
    )

    ax_contain.set_ylim(-0.02, 1.02)
    ax_contain.set_ylabel("Containment on placebo cells", fontsize=axis_label_size)
    ax_contain.set_title("B) Containment", fontsize=panel_title_size)
    xlim_contain = ax_contain.get_xlim()
    ylim_contain = ax_contain.get_ylim()
    x_guide_contain = max(0.003, 0.01 * (xlim_contain[1] - xlim_contain[0]))
    y_guide_contain = ylim_contain[0] + 0.05 * (ylim_contain[1] - ylim_contain[0])
    ax_contain.text(
        x_guide_contain,
        y_guide_contain,
        "better gradient ordering →",
        ha="left",
        va="bottom",
        fontsize=guide_text_size,
        color="#444444",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.7, "pad": 1.2},
    )
    ax_contain.text(
        0.02,
        0.98,
        "better containment ↑",
        transform=ax_contain.transAxes,
        ha="left",
        va="top",
        fontsize=guide_text_size,
        color="#444444",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.7, "pad": 1.2},
    )
    if not has_containment_points:
        ax_contain.text(
            0.5,
            0.5,
            "No finite containment values",
            transform=ax_contain.transAxes,
            ha="center",
            va="center",
            fontsize=guide_text_size,
            color="#555555",
        )
    handles_prompt = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label=label_strategy(s),
            markerfacecolor=strategy_color_map.get(s, "#0F766E"),
            markeredgecolor="white",
            markeredgewidth=0.5,
            markersize=8,
        )
        for s in strategy_order
    ]
    leg_prompt = fig.legend(
        handles=handles_prompt,
        title="Prompt family (color)",
        loc="center left",
        bbox_to_anchor=(0.78, 0.83),
        fontsize=legend_size,
        frameon=False,
        title_fontsize=legend_title_size,
    )
    fig.add_artist(leg_prompt)
    if show_model_legend:
        handles_model = [
            plt.Line2D(
                [0],
                [0],
                marker=model_marker_map.get(m, "o"),
                color="#444444",
                label=label_model(m),
                markerfacecolor="#B0B0B0",
                markeredgecolor="#333333",
                markeredgewidth=0.6,
                linestyle="None",
                markersize=8,
            )
            for m in model_order
        ]
        leg_model = fig.legend(
            handles=handles_model,
            title="Model (shape)",
            loc="center left",
            bbox_to_anchor=(0.78, 0.55),
            fontsize=legend_size,
            frameon=False,
            title_fontsize=legend_title_size,
        )
        fig.add_artist(leg_model)

    if use_dar_size and dar_vals.size:

        def _size_handle(val: float) -> plt.Line2D:
            return plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label=f"{val:.2f}",
                markerfacecolor="#7A7A7A",
                markeredgecolor="#333333",
                markeredgewidth=0.4,
                linestyle="None",
                markersize=np.sqrt(_size_for_dar(val)),
            )

        dar_mid = (dar_vmin + dar_vmax) / 2.0
        handles_size = [_size_handle(dar_vmin), _size_handle(dar_mid), _size_handle(dar_vmax)]
        fig.legend(
            handles=handles_size,
            title="DAR (size)",
            loc="center left",
            bbox_to_anchor=(0.78, 0.28),
            fontsize=legend_size,
            frameon=False,
            title_fontsize=legend_title_size,
        )

    fig.suptitle("Perturbed choice validation: performance and containment", fontsize=suptitle_size)
    fig.tight_layout(rect=[0.03, 0.05, 0.74, 0.94])
    if output_path is not None:
        fig.savefig(Path(output_path), dpi=220, facecolor="white", bbox_inches="tight", pad_inches=0.05)
    plt.show()
    plt.close(fig)
    return None


def render_block_metric_heatmap(
    *,
    pert_block_metrics: pl.DataFrame,
    outcome_order: Sequence[str],
    main_pert_model_order: Sequence[str],
    pert_model_order: Sequence[str],
    label_block: Callable[[str], str],
    label_outcome_short: Callable[[str], str],
    metric_col: str,
    colorbar_label: str,
    base_vmin: float,
    base_vmax: float,
    figsize: tuple[float, float] = (8.4, 3.6),
) -> str | None:
    if pert_block_metrics.is_empty():
        return "No block-level perturbed metrics available for the heatmap."

    blocks = sorted(set(pert_block_metrics.get_column("block").to_list()))
    outcomes = list(outcome_order)
    mat = np.full((len(blocks), len(outcomes)), np.nan, dtype=np.float64)
    sub = pert_block_metrics.filter(pl.col(metric_col).is_finite())
    if sub.is_empty():
        return "No finite block-level perturbed metrics available for the heatmap."
    for j, outcome in enumerate(outcomes):
        for i, block in enumerate(blocks):
            cell = sub.filter((pl.col("outcome") == outcome) & (pl.col("block") == block))
            if cell.is_empty():
                continue
            val = cell.select(pl.median(metric_col)).row(0)[0]
            mat[i, j] = float(val) if val is not None else float("nan")

    vals = mat[np.isfinite(mat)]
    vmin = min(base_vmin, float(np.min(vals))) if vals.size else base_vmin
    vmax = max(base_vmax, float(np.max(vals))) if vals.size else base_vmax

    if metric_col == "gini_block_median":
        plot_vmin = min(-0.2, float(np.floor(vmin / 0.1) * 0.1))
        plot_vmax = max(0.5, float(np.ceil(vmax / 0.1) * 0.1))
        cbar_ticks = np.round(np.arange(plot_vmin, plot_vmax + 0.05, 0.1), 2)
    elif metric_col == "skill_block_median":
        plot_vmin = min(-40.0, float(np.floor(vmin / 10.0) * 10.0))
        plot_vmax = max(10.0, float(np.ceil(vmax / 10.0) * 10.0))
        cbar_ticks = np.arange(plot_vmin, plot_vmax + 5.0, 10.0, dtype=np.float64)
    else:
        span = vmax - vmin
        tick_step = 1.0 if not np.isfinite(span) or span <= 0.0 else span / 5.0
        plot_vmin = float(np.floor(vmin / tick_step) * tick_step)
        plot_vmax = float(np.ceil(vmax / tick_step) * tick_step)
        if np.isclose(plot_vmin, plot_vmax):
            plot_vmax = plot_vmin + tick_step
        cbar_ticks = np.linspace(plot_vmin, plot_vmax, 6)

    fig, ax = plt.subplots(figsize=(7.5, 2.30), constrained_layout=False)
    im = _heatmap_lever_panel(
        ax,
        mat,
        row_labels=[label_block(str(b)) for b in blocks],
        col_labels=[label_outcome_short(o) for o in outcomes],
        vmin=plot_vmin,
        vmax=plot_vmax,
        cmap=plt.get_cmap("cividis"),
        vcenter=None,
    )
    ax.set_ylabel("Block", fontsize=12, labelpad=8)
    cbar = fig.colorbar(im, ax=ax, shrink=0.76, label=colorbar_label, ticks=cbar_ticks)
    # Draw ticks inward to avoid clipping in tight Quarto figure crops.
    cbar.ax.tick_params(which="both", direction="in", length=3.5, width=0.8, labelsize=12)
    cbar.set_label(colorbar_label, fontsize=12)
    fig.subplots_adjust(left=0.20, bottom=0.31, right=0.90, top=0.96)
    plt.show()
    return None


def render_block_sensitivity_alignment(
    *,
    df: pl.DataFrame,
    outcome_order: Sequence[str],
    label_outcome_short: Callable[[str], str],
    label_block: Callable[[str], str],
    default_block_color_order: Sequence[str],
    block_order_for_outcome: Callable[[str], Sequence[str]],
    primary_model: str | None = None,
    primary_strategy: str | None = None,
    panel_nrows: int | None = None,
    panel_ncols: int | None = None,
) -> tuple[str | None, dict[str, float]]:
    diagnostics = {
        "diag_vax_stakes_ref": float("nan"),
        "diag_vax_stakes_llm": float("nan"),
        "diag_clip_x_cap": float("nan"),
        "diag_clip_raw_ref": float("nan"),
        "diag_clip_llm": float("nan"),
        "diag_npi_trust_ref": float("nan"),
        "diag_npi_trust_llm_min": float("nan"),
        "diag_npi_trust_llm_max": float("nan"),
    }
    if df.is_empty():
        return "No block-alignment data available for the plot.", diagnostics
    df = df.filter(pl.col("m_ref").is_finite() & pl.col("m_llm").is_finite())
    if df.is_empty():
        return "No finite block-alignment values available for the plot.", diagnostics

    def _block_metric(df_in: pl.DataFrame, outcome: str, block: str, col: str) -> float:
        sub = df_in.filter((pl.col("outcome") == outcome) & (pl.col("block") == block)).select(col)
        if sub.is_empty():
            return float("nan")
        val = sub.row(0)[0]
        return float(val) if val is not None else float("nan")

    diagnostics["diag_vax_stakes_ref"] = _block_metric(df, "vax_willingness_T12", "stakes_engine", "m_ref")
    diagnostics["diag_vax_stakes_llm"] = _block_metric(df, "vax_willingness_T12", "stakes_engine", "m_llm")

    npi_trust = df.filter(
        pl.col("outcome").is_in(["mask_when_pressure_high", "stay_home_when_symptomatic"])
        & (pl.col("block") == "trust_engine")
    )
    if not npi_trust.is_empty():
        diagnostics["diag_npi_trust_ref"] = float(npi_trust.select(pl.median("m_ref")).row(0)[0])
        llm_vals = npi_trust.get_column("m_llm").to_numpy()
        llm_vals = llm_vals[np.isfinite(llm_vals)]
        if llm_vals.size:
            diagnostics["diag_npi_trust_llm_min"] = float(np.min(llm_vals))
            diagnostics["diag_npi_trust_llm_max"] = float(np.max(llm_vals))

    observed_blocks = [str(b) for b in df.get_column("block").unique().to_list() if str(b).strip()]
    preferred_order: list[str] = []
    for outcome in outcome_order:
        preferred_order.extend(block_order_for_outcome(outcome))
    preferred_order = [b for b in dict.fromkeys(preferred_order) if b in observed_blocks]
    blocks_all = preferred_order + [b for b in observed_blocks if b not in preferred_order]

    cmap = plt.get_cmap("tab20")
    stable_order = list(dict.fromkeys(list(default_block_color_order) + blocks_all))
    stable_color_map = {b: cmap(i % 20) for i, b in enumerate(stable_order)}
    color_map = {b: stable_color_map[b] for b in blocks_all}

    size_vals = df.select(pl.col("importance_share").cast(pl.Float64)).to_series().to_numpy()
    size_vals = size_vals[np.isfinite(size_vals)]
    size_min, size_max = 40.0, 180.0
    if size_vals.size:
        size_vmin = float(np.min(size_vals))
        size_vmax = float(np.max(size_vals))
        if size_vmin >= size_vmax:
            size_vmin, size_vmax = 0.0, 1.0
    else:
        size_vmin, size_vmax = 0.0, 1.0

    def _size_for_val(val: float | None) -> float:
        if val is None or not np.isfinite(float(val)):
            return 70.0
        t = (float(val) - size_vmin) / (size_vmax - size_vmin) if size_vmax > size_vmin else 0.5
        return float(size_min + float(np.clip(t, 0.0, 1.0)) * (size_max - size_min))

    if panel_nrows is not None and panel_ncols is not None:
        outcomes = [str(o) for o in outcome_order if str(o).strip()]
        nrows = max(1, int(panel_nrows))
        ncols = max(1, int(panel_ncols))
        if nrows * ncols < len(outcomes):
            return f"Requested panel layout {nrows}x{ncols} is too small for {len(outcomes)} outcomes.", diagnostics
        panel_slots = outcomes + [None] * (nrows * ncols - len(outcomes))
    else:
        nrows, ncols, panel_slots = outcome_panel_layout(outcome_order)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(3.25 * ncols + 2.1, 2.15 * nrows + 0.9),
        sharex=True,
        sharey=True,
        constrained_layout=False,
    )
    axes = np.atleast_1d(axes).ravel()

    occupied_by_col: dict[int, list[int]] = {}
    for idx, outcome in enumerate(panel_slots):
        if outcome is None:
            continue
        occupied_by_col.setdefault(idx % ncols, []).append(idx)
    bottom_idx_by_col = {col: max(indices) for col, indices in occupied_by_col.items()}

    ref_vals = df.get_column("m_ref").to_numpy()
    finite_ref = ref_vals[np.isfinite(ref_vals)]
    x_cap = float(np.quantile(finite_ref, 0.95)) if finite_ref.size else 1.0
    x_cap = max(x_cap, 1e-6)
    diagnostics["diag_clip_x_cap"] = float(x_cap)

    clip_candidate: tuple[str, str] | None = None
    clip_max_ref = float("-inf")
    for row in df.iter_rows(named=True):
        raw_x = float(row.get("m_ref"))
        if not np.isfinite(raw_x) or raw_x <= x_cap:
            continue
        block = str(row.get("block"))
        outcome = str(row.get("outcome"))
        is_stakes = "stake" in block.lower()
        if is_stakes and raw_x > clip_max_ref:
            clip_max_ref = raw_x
            clip_candidate = (outcome, block)
    if clip_candidate is None:
        for row in df.iter_rows(named=True):
            raw_x = float(row.get("m_ref"))
            if not np.isfinite(raw_x) or raw_x <= x_cap:
                continue
            block = str(row.get("block"))
            outcome = str(row.get("outcome"))
            if raw_x > clip_max_ref:
                clip_max_ref = raw_x
                clip_candidate = (outcome, block)
    if clip_candidate is not None:
        c_outcome, c_block = clip_candidate
        diagnostics["diag_clip_raw_ref"] = _block_metric(df, c_outcome, c_block, "m_ref")
        diagnostics["diag_clip_llm"] = _block_metric(df, c_outcome, c_block, "m_llm")

    global_min = float(min(df.get_column("m_ref").min(), df.get_column("m_llm").min()))
    global_max = float(max(x_cap, df.get_column("m_llm").max()))
    pad = 0.05 * (global_max - global_min) if global_max > global_min else 0.1

    for idx, outcome in enumerate(panel_slots):
        ax = axes[idx]
        if outcome is None:
            ax.set_axis_off()
            continue
        sub = df.filter(pl.col("outcome") == outcome)
        if sub.is_empty():
            ax.set_axis_off()
            continue
        for row in sub.iter_rows(named=True):
            block = str(row.get("block"))
            raw_x = float(row.get("m_ref"))
            raw_y = float(row.get("m_llm"))
            should_clip = clip_candidate is not None and (outcome, block) == clip_candidate and raw_x > x_cap
            plot_x = x_cap if should_clip else raw_x
            ax.scatter(
                plot_x,
                raw_y,
                s=_size_for_val(row.get("importance_share")),
                color=color_map.get(block, "#4c72b0"),
                alpha=0.8,
                edgecolor="black",
                linewidth=0.3,
            )
            if should_clip:
                ax.annotate(
                    f"(clipped: x={raw_x:.2f})",
                    xy=(plot_x, raw_y),
                    xycoords="data",
                    xytext=(0.98, 0.90),
                    textcoords="axes fraction",
                    ha="right",
                    va="top",
                    fontsize=10.5,
                    arrowprops={"arrowstyle": "->", "lw": 0.8, "color": "black"},
                    bbox={"boxstyle": "round,pad=0.15", "fc": "white", "ec": "none", "alpha": 0.8},
                )
        if np.isfinite(global_min) and np.isfinite(global_max):
            ax.plot(
                [global_min - pad, global_max + pad],
                [global_min - pad, global_max + pad],
                color="black",
                linewidth=1,
                alpha=0.6,
            )
        ax.set_xlim(global_min - pad, global_max + pad)
        ax.set_ylim(global_min - pad, global_max + pad)
        ax.set_title(textwrap.fill(label_outcome_short(outcome), width=22), fontsize=11)
        ax.tick_params(axis="both", labelsize=12)
        if idx in bottom_idx_by_col.values():
            ax.tick_params(axis="x", labelbottom=True)

    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label=label_block(b),
            markerfacecolor=color_map.get(b, "#4c72b0"),
            markersize=8,
        )
        for b in blocks_all
    ]
    fig.legend(
        handles=handles,
        title="Block",
        frameon=False,
        fontsize=12,
        title_fontsize=12,
        loc="center left",
        bbox_to_anchor=(0.76, 0.52),
    )

    if size_vals.size:

        def _size_handle(val: float) -> plt.Line2D:
            return plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label=f"{val:.3f}",
                markerfacecolor="#6B6B6B",
                markersize=np.sqrt(_size_for_val(val)),
            )

        s_mid = float((size_vmin + size_vmax) / 2.0)
        handles_size = [_size_handle(size_vmin), _size_handle(s_mid), _size_handle(size_vmax)]
        fig.legend(
            handles=handles_size,
            title="Ref block weight-importance share",
            frameon=False,
            fontsize=12,
            loc="center left",
            bbox_to_anchor=(0.76, 0.12),
        )

    fig.supxlabel("Reference median $|\\mathrm{PE}_{\\mathrm{std}}|$", fontsize=12)
    fig.supylabel("LLM median $|\\mathrm{PE}_{\\mathrm{std}}|$", fontsize=12)
    if primary_model and primary_strategy:
        fig.suptitle(f"Block sensitivity alignment (model={primary_model}, {primary_strategy})")
    else:
        fig.suptitle("Block sensitivity alignment (median across model x strategy)")
    fig.subplots_adjust(right=0.73, bottom=0.12, top=0.90, wspace=0.36, hspace=0.48)
    plt.show()
    return None, diagnostics


def render_belief_frontier(
    *,
    frontier_df: pl.DataFrame,
    label_strategy: Callable[[str], str],
    label_model: Callable[[str], str],
) -> str | None:
    if frontier_df.is_empty():
        return "No frontier data available (summary metrics missing)."
    required_cols = {
        "model",
        "strategy",
        "cosine_pe_std",
        "skill_pe_std",
        "low_signal_abs_pe_llm_std_median",
    }
    missing = sorted(required_cols - set(frontier_df.columns))
    if missing:
        return "Frontier data is missing required oracle columns: " + ", ".join(missing) + "."

    select_exprs: list[pl.Expr] = [
        pl.col("model").cast(pl.Utf8),
        pl.col("strategy").cast(pl.Utf8),
        pl.col("cosine_pe_std").cast(pl.Float64, strict=False).alias("x_metric"),
        pl.col("skill_pe_std").cast(pl.Float64, strict=False).alias("y_metric"),
        pl.col("low_signal_abs_pe_llm_std_median").cast(pl.Float64, strict=False).alias("leakage_metric"),
    ]
    if "dar" in frontier_df.columns:
        select_exprs.append(pl.col("dar").cast(pl.Float64, strict=False).alias("dar_metric"))
    else:
        select_exprs.append(pl.lit(float("nan")).alias("dar_metric"))

    df = frontier_df.select(select_exprs).with_columns(
        pl.col("model").map_elements(normalize_model_slug, return_dtype=pl.Utf8).alias("model_norm"),
        (pl.lit(1.0) - pl.col("leakage_metric")).alias("containment_metric"),
    )
    if df.is_empty():
        return "Frontier data is empty after filtering."

    strategy_set = set(df.get_column("strategy").to_list())
    strategy_order = [s for s in PARETO_STRATEGY_ORDER if s in strategy_set] + sorted(
        strategy_set - set(PARETO_STRATEGY_ORDER)
    )
    strategy_color_map = _build_pareto_strategy_color_map(strategy_order)

    model_order = discover_pert_models(df.select(pl.col("model_norm").alias("model")))
    if not model_order:
        model_order = sorted(set(df.get_column("model_norm").to_list()))
    model_marker_map = _build_pareto_model_marker_map(model_order)

    dar_vals = df.get_column("dar_metric").to_numpy()
    dar_vals = dar_vals[np.isfinite(dar_vals)]
    size_min, size_max = 45.0, 180.0
    if dar_vals.size:
        dar_vmin = float(np.min(dar_vals))
        dar_vmax = float(np.max(dar_vals))
        if dar_vmin >= dar_vmax:
            dar_vmin, dar_vmax = 0.0, 1.0
    else:
        dar_vmin, dar_vmax = 0.0, 1.0

    def _size_for_dar(value: float) -> float:
        if not np.isfinite(value):
            return 75.0
        t = (value - dar_vmin) / (dar_vmax - dar_vmin) if dar_vmax > dar_vmin else 0.5
        return float(size_min + float(np.clip(t, 0.0, 1.0)) * (size_max - size_min))

    def _as_float_or_nan(value: object) -> float:
        if value is None:
            return float("nan")
        try:
            return float(value)
        except (TypeError, ValueError):
            return float("nan")

    panel_title_size = 12
    axis_label_size = 12
    axis_tick_size = 12
    guide_text_size = 12
    legend_size = 12
    legend_title_size = 12
    suptitle_size = 12

    fig, (ax_perf, ax_contain) = plt.subplots(2, 1, figsize=(8.2, 7.7), sharex=True)
    has_perf_points = False
    has_containment_points = False
    for row in df.iter_rows(named=True):
        strategy = str(row["strategy"])
        model = str(row["model_norm"])
        x_val = _as_float_or_nan(row.get("x_metric", float("nan")))
        y_val = _as_float_or_nan(row.get("y_metric", float("nan")))
        containment_val = _as_float_or_nan(row.get("containment_metric", float("nan")))
        size = _size_for_dar(_as_float_or_nan(row.get("dar_metric", float("nan"))))
        color = strategy_color_map.get(strategy, "#0F766E")
        marker = model_marker_map.get(model, "o")

        if np.isfinite(x_val) and np.isfinite(y_val):
            has_perf_points = True
            ax_perf.scatter(
                x_val,
                y_val,
                color=color,
                marker=marker,
                s=size,
                edgecolor="white",
                linewidth=0.5,
                alpha=0.92,
            )
        if np.isfinite(x_val) and np.isfinite(containment_val):
            has_containment_points = True
            ax_contain.scatter(
                x_val,
                containment_val,
                color=color,
                marker=marker,
                s=size,
                edgecolor="white",
                linewidth=0.5,
                alpha=0.92,
            )

    for ax in (ax_perf, ax_contain):
        ax.axvline(0, color="black", linewidth=1, alpha=0.6)
        ax.grid(True, alpha=0.15, linewidth=0.6)
        ax.set_xlabel("Median active-set cosine similarity", fontsize=axis_label_size)
        ax.tick_params(axis="both", labelsize=axis_tick_size)

    ax_perf.set_ylabel("Median active-set skill (higher is better)", fontsize=axis_label_size)
    ax_perf.set_title("A) Performance on active set", fontsize=panel_title_size)
    ax_perf.text(
        0.98,
        0.02,
        "better active-set shape match →",
        transform=ax_perf.transAxes,
        ha="right",
        va="bottom",
        fontsize=guide_text_size,
        color="#444444",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.7, "pad": 1.2},
    )
    ax_perf.text(
        0.02,
        0.98,
        "better active-set performance ↑",
        transform=ax_perf.transAxes,
        ha="left",
        va="top",
        fontsize=guide_text_size,
        color="#444444",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.7, "pad": 1.2},
    )

    ax_contain.set_ylabel("Containment on non-active set", fontsize=axis_label_size)
    ax_contain.set_title("B) Containment", fontsize=panel_title_size)
    ax_contain.text(
        0.98,
        0.02,
        "better active-set shape match →",
        transform=ax_contain.transAxes,
        ha="right",
        va="bottom",
        fontsize=guide_text_size,
        color="#444444",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.7, "pad": 1.2},
    )
    ax_contain.text(
        0.02,
        0.98,
        "better containment ↑",
        transform=ax_contain.transAxes,
        ha="left",
        va="top",
        fontsize=guide_text_size,
        color="#444444",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.7, "pad": 1.2},
    )

    if not has_perf_points:
        ax_perf.text(
            0.5,
            0.5,
            "No finite active-set values",
            transform=ax_perf.transAxes,
            ha="center",
            va="center",
            fontsize=guide_text_size,
            color="#555555",
        )
    if not has_containment_points:
        ax_contain.text(
            0.5,
            0.5,
            "No finite containment values",
            transform=ax_contain.transAxes,
            ha="center",
            va="center",
            fontsize=guide_text_size,
            color="#555555",
        )

    prompt_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label=label_strategy(s),
            markerfacecolor=strategy_color_map.get(s, "#0F766E"),
            markeredgecolor="white",
            markeredgewidth=0.5,
            markersize=8,
        )
        for s in strategy_order
    ]
    model_handles = [
        plt.Line2D(
            [0],
            [0],
            marker=model_marker_map.get(m, "o"),
            color="#444444",
            label=label_model(m),
            markerfacecolor="#B0B0B0",
            markeredgecolor="#333333",
            markeredgewidth=0.6,
            linestyle="None",
            markersize=8,
        )
        for m in model_order
    ]

    leg_prompt = fig.legend(
        handles=prompt_handles,
        title="Prompt family (color)",
        loc="center left",
        bbox_to_anchor=(0.82, 0.83),
        fontsize=legend_size,
        frameon=False,
        title_fontsize=legend_title_size,
    )
    fig.add_artist(leg_prompt)
    leg_model = fig.legend(
        handles=model_handles,
        title="Model (shape)",
        loc="center left",
        bbox_to_anchor=(0.82, 0.55),
        fontsize=legend_size,
        frameon=False,
        title_fontsize=legend_title_size,
    )
    fig.add_artist(leg_model)

    if dar_vals.size:

        def _size_handle(val: float) -> plt.Line2D:
            return plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label=f"{val:.2f}",
                markerfacecolor="#7A7A7A",
                markeredgecolor="#333333",
                markeredgewidth=0.4,
                linestyle="None",
                markersize=np.sqrt(_size_for_dar(val)),
            )

        dar_mid = (dar_vmin + dar_vmax) / 2.0
        handles_size = [_size_handle(dar_vmin), _size_handle(dar_mid), _size_handle(dar_vmax)]
        fig.legend(
            handles=handles_size,
            title="DAR (size)",
            loc="center left",
            bbox_to_anchor=(0.82, 0.28),
            fontsize=legend_size,
            frameon=False,
            title_fontsize=legend_title_size,
        )

    fig.suptitle("Psychological-profile validation: performance and containment", fontsize=suptitle_size)
    fig.tight_layout(rect=[0.04, 0.05, 0.80, 0.94])
    plt.show()
    return None


def render_belief_reference_signal_magnitude(
    *,
    pe_ref_rows_path: Path,
    label_construct: Callable[[str], str],
    max_signals: int = 16,
    signal_metadata_path: Path | None = None,
    include_tiers: Sequence[str] = ("primary", "secondary"),
    x_max_override: float | None = None,
    signal_bucket_map: Mapping[str, str] | None = None,
    signal_bucket_order: Sequence[str] | None = None,
    belief_block_map: Mapping[str, str] | None = None,
    belief_block_order: Sequence[str] | None = None,
    append_unordered_belief_blocks: bool = True,
    panel_nrows: int | None = None,
    panel_ncols: int | None = None,
    label_signal_bucket: Callable[[str], str] | None = None,
    label_belief_block: Callable[[str], str] | None = None,
    show_signal_block_legend: bool = True,
    y_axis_label: str = "Broader covariate block",
    x_axis_label: str = r"Standardised perturbation effect $|PE^{\mathrm{profile}}_{\mathrm{std}}|$",
    y_label_wrap_width: int = 24,
) -> str | None:
    """Plot belief-update reference perturbation magnitudes by signal/construct.

    Supports optional grouping by signal mechanism buckets and faceting by
    belief-outcome blocks.
    """
    if not pe_ref_rows_path.exists():
        return f"No psychological-profile reference perturbation rows found at {pe_ref_rows_path}."
    try:
        raw = pl.read_parquet(pe_ref_rows_path)
    except Exception as exc:  # pragma: no cover - defensive IO guard
        return f"Failed to read psychological-profile reference perturbation rows ({exc})."

    required = {"signal_id", "state_var", "pe_ref_p_std"}
    missing = sorted(required - set(raw.columns))
    if missing:
        return "Belief reference PE table is missing required columns: " + ", ".join(missing) + "."

    work = (
        raw.select(
            pl.col("signal_id").cast(pl.Utf8).alias("signal_id"),
            pl.col("state_var").cast(pl.Utf8).alias("state_var"),
            pl.col("pe_ref_p_std").cast(pl.Float64).alias("pe_ref_p_std"),
        )
        .with_columns(pl.col("pe_ref_p_std").abs().alias("abs_ref_std"))
        .drop_nulls(["signal_id", "state_var", "abs_ref_std"])
        .filter(pl.col("abs_ref_std").is_finite())
    )
    if work.is_empty():
        return "No finite psychological-profile reference perturbation values available."

    if signal_metadata_path is not None and signal_metadata_path.exists():
        try:
            meta = json.loads(signal_metadata_path.read_text(encoding="utf-8"))
        except Exception as exc:  # pragma: no cover - defensive IO guard
            return f"Failed to read belief signal metadata ({exc})."

        signal_meta = meta.get("signal_meta") if isinstance(meta, dict) else None
        allowed_tiers = {str(t).strip() for t in include_tiers if str(t).strip()}
        allowed_pairs: set[tuple[str, str]] = set()
        if isinstance(signal_meta, dict) and allowed_tiers:
            for signal_id, payload in signal_meta.items():
                if not isinstance(payload, dict):
                    continue
                tier_by_var = payload.get("tier_by_var")
                if not isinstance(tier_by_var, dict):
                    continue
                for state_var, tier in tier_by_var.items():
                    if str(tier).strip() in allowed_tiers:
                        allowed_pairs.add((str(signal_id), str(state_var)))

        if allowed_pairs:
            allowed_df = pl.DataFrame(
                {
                    "signal_id": [p[0] for p in allowed_pairs],
                    "state_var": [p[1] for p in allowed_pairs],
                }
            )
            work = work.join(allowed_df, on=["signal_id", "state_var"], how="inner")
            if work.is_empty():
                return "No psychological-profile reference rows remain after signal-to-construct tier filtering."

    signal_state_summary = (
        work.group_by(["signal_id", "state_var"])
        .agg(pl.col("abs_ref_std").quantile(0.50, "nearest").alias("q50"))
        .drop_nulls(["q50"])
        .filter(pl.col("q50").is_finite())
    )
    if signal_state_summary.is_empty():
        return "No finite grouped psychological-profile reference perturbation values available."

    signal_strength = (
        signal_state_summary.group_by("signal_id")
        .agg(pl.col("q50").max().alias("signal_strength"))
        .sort(["signal_strength", "signal_id"], descending=[True, False])
    )
    if max_signals > 0 and signal_strength.height > max_signals:
        signal_keep = signal_strength.get_column("signal_id").head(max_signals).to_list()
        work = work.filter(pl.col("signal_id").is_in(signal_keep))
        signal_strength = signal_strength.filter(pl.col("signal_id").is_in(signal_keep))
    if work.is_empty() or signal_strength.is_empty():
        return "No psychological-profile reference rows available after signal filtering."

    signal_bucket_lookup = {str(k): str(v) for k, v in (signal_bucket_map or {}).items()}
    belief_block_lookup = {str(k): str(v) for k, v in (belief_block_map or {}).items()}
    if signal_bucket_lookup:
        work = work.with_columns(
            pl.col("signal_id")
            .map_elements(
                lambda sid: signal_bucket_lookup.get(str(sid)),
                return_dtype=pl.Utf8,
            )
            .alias("signal_bucket"),
            pl.col("state_var")
            .map_elements(
                lambda state_var: belief_block_lookup.get(str(state_var), str(state_var)),
                return_dtype=pl.Utf8,
            )
            .alias("belief_block"),
        ).drop_nulls(["signal_bucket"])
    else:
        work = work.with_columns(
            pl.col("signal_id").cast(pl.Utf8).alias("signal_bucket"),
            pl.col("state_var")
            .map_elements(
                lambda state_var: belief_block_lookup.get(str(state_var), str(state_var)),
                return_dtype=pl.Utf8,
            )
            .alias("belief_block"),
        )

    bucket_strength = (
        work.group_by("signal_bucket")
        .agg(pl.col("abs_ref_std").quantile(0.50, "nearest").alias("bucket_strength"))
        .sort(["bucket_strength", "signal_bucket"], descending=[True, False])
    )
    bucket_strength_map = {
        str(row["signal_bucket"]): float(row["bucket_strength"]) for row in bucket_strength.to_dicts()
    }

    present_signal_buckets = [str(v) for v in work.get_column("signal_bucket").unique().to_list()]
    if signal_bucket_order:
        signal_bucket_levels = [b for b in signal_bucket_order if b in present_signal_buckets]
        signal_bucket_levels.extend(sorted(b for b in present_signal_buckets if b not in set(signal_bucket_levels)))
    else:
        signal_bucket_levels = sorted(
            present_signal_buckets,
            key=lambda bucket: (-bucket_strength_map.get(bucket, 0.0), bucket),
        )

    bucket_summary = (
        work.group_by(["signal_bucket", "state_var", "belief_block"])
        .agg(
            pl.col("abs_ref_std").quantile(0.10, "nearest").alias("q10"),
            pl.col("abs_ref_std").quantile(0.25, "nearest").alias("q25"),
            pl.col("abs_ref_std").quantile(0.50, "nearest").alias("q50"),
            pl.col("abs_ref_std").quantile(0.75, "nearest").alias("q75"),
            pl.col("abs_ref_std").quantile(0.90, "nearest").alias("q90"),
            pl.len().alias("n_rows"),
        )
        .drop_nulls(["q50"])
        .filter(pl.col("q50").is_finite())
    )
    if bucket_summary.is_empty():
        return "Belief reference plot has no finite bucket-aggregated values."

    var_order = (
        bucket_summary.group_by("state_var")
        .agg(pl.col("q50").median().alias("median_q50"))
        .sort(["median_q50", "state_var"], descending=[True, False])
        .get_column("state_var")
        .to_list()
    )
    if not signal_bucket_levels or not var_order:
        return "Belief reference plot has empty bucket or construct axes."

    present_belief_blocks = [str(v) for v in bucket_summary.get_column("belief_block").unique().to_list()]
    if belief_block_order:
        panel_blocks = [b for b in belief_block_order if b in present_belief_blocks]
        if append_unordered_belief_blocks:
            panel_blocks.extend(sorted(b for b in present_belief_blocks if b not in set(panel_blocks)))
    else:
        panel_blocks = sorted(present_belief_blocks)
    if not panel_blocks:
        return "Belief reference plot has no populated belief blocks."

    panel_bucket_counts = bucket_summary.group_by("belief_block").agg(pl.col("signal_bucket").n_unique().alias("n"))
    max_panel_buckets = (
        int(panel_bucket_counts.get_column("n").max())
        if panel_bucket_counts.height > 0 and panel_bucket_counts.get_column("n").max() is not None
        else 1
    )

    n_blocks = len(panel_blocks)
    if panel_nrows is not None and panel_ncols is not None:
        nrows = max(1, int(panel_nrows))
        ncols = max(1, int(panel_ncols))
        if nrows * ncols < n_blocks:
            return f"Requested panel layout {nrows}x{ncols} is too small for {n_blocks} psychological-profile panels."
    else:
        nrows = 2 if n_blocks > 1 else 1
        ncols = int(np.ceil(n_blocks / nrows))
    # Keep sparse panels compact to avoid excess vertical whitespace.
    fig_h_panel = max(2.9, 0.48 * max_panel_buckets + 1.2)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.7 * ncols, fig_h_panel * nrows),
        sharex=True,
        sharey=False,
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes).ravel()

    cmap = plt.get_cmap("tab10")
    block_color_map = {bucket: cmap(i % 10) for i, bucket in enumerate(signal_bucket_levels)}

    def _identity_label(value: str) -> str:
        return str(value)

    panel_label = label_belief_block or _identity_label
    bucket_label = label_signal_bucket or _identity_label

    override_ok = x_max_override is not None and np.isfinite(float(x_max_override)) and float(x_max_override) > 0
    if override_ok:
        x_right = float(x_max_override)
    else:
        x_max_val = bucket_summary.select(pl.col("q90").max()).row(0)[0]
        x_max = float(x_max_val) if x_max_val is not None else float("nan")
        x_right = 1.08 * x_max if np.isfinite(x_max) and x_max > 0 else 1.0

    for ax, block in zip(axes, panel_blocks, strict=False):
        panel_df = bucket_summary.filter(pl.col("belief_block") == block)
        if panel_df.is_empty():
            ax.axis("off")
            continue

        panel_present_buckets = [str(v) for v in panel_df.get_column("signal_bucket").unique().to_list()]
        if signal_bucket_order:
            panel_bucket_levels = [b for b in signal_bucket_order if b in panel_present_buckets]
            panel_bucket_levels.extend(sorted(b for b in panel_present_buckets if b not in set(panel_bucket_levels)))
        else:
            panel_bucket_levels = sorted(
                panel_present_buckets,
                key=lambda bucket: (-bucket_strength_map.get(bucket, 0.0), bucket),
            )

        panel_y_base = {bucket: float(idx) for idx, bucket in enumerate(panel_bucket_levels)}
        panel_y_ticks = [panel_y_base[bucket] for bucket in panel_bucket_levels]
        if not panel_y_ticks:
            ax.axis("off")
            continue

        panel_vars = (
            panel_df.group_by("state_var")
            .agg(pl.col("q50").median().alias("median_q50"))
            .sort(["median_q50", "state_var"], descending=[True, False])
            .get_column("state_var")
            .to_list()
        )
        n_vars = len(panel_vars)
        offset_step = min(0.11, 0.66 / max(n_vars, 1))

        for idx, state_var in enumerate(panel_vars):
            sub = panel_df.filter(pl.col("state_var") == state_var)
            for row in sub.iter_rows(named=True):
                signal_bucket = str(row.get("signal_bucket"))
                if signal_bucket not in panel_y_base:
                    continue
                y = float(panel_y_base[signal_bucket]) + (idx - (n_vars - 1) / 2.0) * offset_step

                q10 = float(row.get("q10", float("nan")))
                q25 = float(row.get("q25", float("nan")))
                q50 = float(row.get("q50", float("nan")))
                q75 = float(row.get("q75", float("nan")))
                q90 = float(row.get("q90", float("nan")))
                color = block_color_map.get(signal_bucket, "#4C78A8")

                if np.isfinite(q10) and np.isfinite(q90):
                    ax.hlines(y, q10, q90, color=color, alpha=0.35, linewidth=1.0, zorder=1)
                if np.isfinite(q25) and np.isfinite(q75):
                    ax.hlines(y, q25, q75, color=color, alpha=0.9, linewidth=2.3, zorder=2)
                if np.isfinite(q50):
                    ax.scatter(q50, y, color=color, s=24, edgecolors="white", linewidths=0.4, zorder=3)

        ax.set_yticks(panel_y_ticks)
        ax.set_yticklabels(
            [
                "\n".join(textwrap.wrap(bucket_label(bucket), width=y_label_wrap_width))
                for bucket in panel_bucket_levels
            ],
            fontsize=8.8,
        )

        ax.set_title(panel_label(block), fontsize=10.8)
        ax.grid(axis="x", linestyle=":", alpha=0.4)
        if np.isfinite(x_right) and x_right > 0:
            ax.set_xlim(0.0, x_right)
        ax.tick_params(axis="x", labelsize=8.6)
        y_pad = 0.35
        y_min = min(panel_y_ticks) - y_pad
        y_max = max(panel_y_ticks) + y_pad
        ax.set_ylim(y_min, y_max)
        ax.invert_yaxis()

    for ax in axes[len(panel_blocks) :]:
        ax.axis("off")

    fig.supylabel(y_axis_label, fontsize=10.8)
    fig.supxlabel(x_axis_label, fontsize=10.8)

    if show_signal_block_legend and len(signal_bucket_levels) > 1:
        legend_handles = [
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color=block_color_map[bucket],
                label=bucket_label(str(bucket)),
                linestyle="-",
                linewidth=2.0,
                markersize=5.0,
            )
            for bucket in signal_bucket_levels
            if bucket_summary.filter(pl.col("signal_bucket") == bucket).height > 0
        ]
        fig.legend(
            handles=legend_handles,
            title="Signal block",
            loc="center left",
            bbox_to_anchor=(1.01, 0.5),
            frameon=False,
        )

    plt.show()
    return None


def render_belief_construct_readiness(
    *,
    construct_readiness: pl.DataFrame,
    label_construct: Callable[[str], str],
) -> str | None:
    if construct_readiness.is_empty():
        return "No construct readiness data available."
    df = construct_readiness.select(
        pl.col("construct"),
        pl.col("sign_pass_rate_primary"),
        pl.col("kendall_strength_alignment"),
        pl.col("none_stability_rate"),
    )
    df = df.drop_nulls(["sign_pass_rate_primary", "kendall_strength_alignment", "none_stability_rate"])
    if df.is_empty():
        return "Construct readiness data is empty after filtering."

    x = df.get_column("kendall_strength_alignment").to_numpy()
    y = df.get_column("sign_pass_rate_primary").to_numpy()
    size = 60 + 260 * np.clip(df.get_column("none_stability_rate").to_numpy(), 0.0, 1.0)

    _, ax = plt.subplots(figsize=(9.4, 6.2))
    ax.scatter(x, y, s=size, c="#5c6f91", alpha=0.85, edgecolors="white", linewidths=0.6)
    for i, c in enumerate(df.get_column("construct").to_list()):
        dx = 6 if i % 2 == 0 else -6
        dy = 6 if i % 3 == 0 else -4
        ax.annotate(
            label_construct(str(c)),
            (x[i], y[i]),
            textcoords="offset points",
            xytext=(dx, dy),
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.15", "fc": "white", "alpha": 0.6, "ec": "none"},
        )

    size_levels = [0.4, 0.7, 0.9]
    size_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            color="none",
            markerfacecolor="#5c6f91",
            markeredgecolor="white",
            markeredgewidth=0.6,
            alpha=0.85,
            markersize=np.sqrt(60 + 260 * level),
            label=f"{level:.1f}",
        )
        for level in size_levels
    ]
    legend_size = ax.legend(
        handles=size_handles, title="None stability", loc="lower right", frameon=True, fontsize=8, title_fontsize=8
    )
    ax.add_artist(legend_size)

    ax.set_xlabel("Strength alignment (Kendall $\\tau_b$)")
    ax.set_ylabel("Sign agreement on primary targets")
    x_pad, y_pad = 0.06, 0.06
    ax.set_xlim(max(-0.05, float(np.min(x)) - x_pad), min(1.05, float(np.max(x)) + x_pad))
    ax.set_ylim(max(0.0, float(np.min(y)) - y_pad), min(1.05, float(np.max(y)) + y_pad))
    ax.grid(linestyle=":", alpha=0.4)
    plt.tight_layout()
    return None


def render_belief_signal_activation(
    *,
    signal_summary: pl.DataFrame,
    movement_df: pl.DataFrame,
    signal_label: Callable[[str], str],
    label_construct: Callable[[str], str],
) -> str | None:
    if signal_summary.is_empty():
        return "No signal-level summaries found."
    plot_df = signal_summary
    if not movement_df.is_empty():
        plot_df = plot_df.join(movement_df, on="signal_id", how="left")
    plot_df = plot_df.with_columns(
        pl.when(pl.col("primary_targets").is_null())
        .then(pl.lit("—"))
        .otherwise(pl.col("primary_targets").list.first().cast(pl.Utf8))
        .alias("primary_construct")
    ).sort(["primary_activation_rate", "primary_construct"], descending=[True, False])

    labels = [signal_label(s) for s in plot_df.get_column("signal_id")]
    constructs = [label_construct(str(c)) for c in plot_df.get_column("primary_construct").to_list()]
    activation_vals = plot_df.get_column("primary_activation_rate").to_numpy()
    none_vals = plot_df.get_column("none_stability_rate").to_numpy()
    drift_vals = (
        plot_df.get_column("none_abs").to_numpy() if "none_abs" in plot_df.columns else np.full(len(labels), np.nan)
    )
    y = np.arange(len(labels))

    _, ax = plt.subplots(figsize=(10.8, 10.6))
    ax.barh(y, activation_vals, color="#4C78A8", alpha=0.8, label="Activation (primary)")
    ax.scatter(none_vals, y, color="#F58518", s=30, zorder=3, label="None stability")
    ax.scatter(
        drift_vals,
        y,
        facecolors="none",
        edgecolors="#E45756",
        s=40,
        linewidths=1.1,
        zorder=3,
        label="Drift (mean |Δ| none)",
    )

    for i, (act, none_v, drift_v) in enumerate(zip(activation_vals, none_vals, drift_vals, strict=False)):
        ax.annotate(
            f"{act:.2f}",
            (act, i),
            textcoords="offset points",
            xytext=(6, 0),
            va="center",
            fontsize=8,
            color="#4C78A8",
            bbox={"boxstyle": "round,pad=0.15", "fc": "white", "alpha": 0.7, "ec": "none"},
        )
        ax.annotate(
            f"{none_v:.2f}",
            (none_v, i),
            textcoords="offset points",
            xytext=(6, -8),
            va="center",
            fontsize=8,
            color="#F58518",
            bbox={"boxstyle": "round,pad=0.15", "fc": "white", "alpha": 0.7, "ec": "none"},
        )
        if not np.isnan(drift_v):
            ax.annotate(
                f"{drift_v:.2f}",
                (drift_v, i),
                textcoords="offset points",
                xytext=(6, 8),
                va="center",
                fontsize=8,
                color="#E45756",
                bbox={"boxstyle": "round,pad=0.15", "fc": "white", "alpha": 0.7, "ec": "none"},
            )

    ax.set_yticks(y)
    ax.set_yticklabels([f"{lab}  [{con}]" for lab, con in zip(labels, constructs, strict=False)], fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlabel("Rate / magnitude")
    ax.set_ylabel("")
    ax.set_xlim(0.0, 1.05)
    ax.set_xticks([0.0, 0.5, 1.0])
    ax.grid(axis="x", linestyle=":", alpha=0.4)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    plt.tight_layout()
    return None


def render_belief_scope_control(
    *,
    movement_df: pl.DataFrame,
    shock_df: pl.DataFrame,
    signal_summary: pl.DataFrame,
    signal_label: Callable[[str], str],
    label_construct: Callable[[str], str],
    primary_threshold: float,
    leakage_slope: float,
    spillover_cap: float,
    adjust_text_fn=None,
) -> str | None:
    if movement_df.is_empty():
        return "No row-level movement summaries available."

    _, ax = plt.subplots(figsize=(9.0, 6.2))
    base = movement_df
    if not shock_df.is_empty() and "lever_class" in shock_df.columns:
        base = base.join(shock_df.select("signal_id", "lever_class"), on="signal_id", how="left")
    if not signal_summary.is_empty():
        base = base.join(
            signal_summary.select("signal_id", "primary_activation_rate", "primary_targets"),
            on="signal_id",
            how="left",
        )
    if base.is_empty():
        return "Insufficient data to build the scope-control plot."

    base = base.with_columns(
        pl.when(pl.col("primary_targets").is_null())
        .then(pl.lit("—"))
        .otherwise(pl.col("primary_targets").list.first().cast(pl.Utf8))
        .alias("primary_construct")
    )
    constructs = sorted(set(base.get_column("primary_construct").to_list()))
    cmap = plt.get_cmap("tab10")
    color_map = {c: cmap(i % 10) for i, c in enumerate(constructs)}
    marker_map = {
        "Safe targeted shock": "o",
        "Leaky shock": "s",
        "Cross-talk / reinterpretation": "^",
        "Stable low-signal": "D",
    }

    act = (
        base.get_column("primary_activation_rate").to_numpy()
        if "primary_activation_rate" in base.columns
        else np.full(base.height, 0.3)
    )
    base = base.with_columns((40 + 260 * pl.Series(np.clip(np.nan_to_num(act, nan=0.0), 0.0, 1.0))).alias("_pt_size"))

    if "lever_class" not in base.columns:
        xs = base.get_column("primary_abs").to_numpy()
        ys = base.get_column("none_abs").to_numpy()
        cs = [color_map.get(c, "#4C78A8") for c in base.get_column("primary_construct").to_list()]
        ss = base.get_column("_pt_size").to_numpy()
        ax.scatter(xs, ys, s=ss, c=cs, marker="o", alpha=0.85, edgecolors="white", linewidths=0.6)
    else:
        for cls, marker in marker_map.items():
            sub = base.filter(pl.col("lever_class") == cls)
            if sub.is_empty():
                continue
            xs = sub.get_column("primary_abs").to_numpy()
            ys = sub.get_column("none_abs").to_numpy()
            cs = [color_map.get(c, "#4C78A8") for c in sub.get_column("primary_construct").to_list()]
            ss = sub.get_column("_pt_size").to_numpy()
            ax.scatter(xs, ys, s=ss, c=cs, marker=marker, alpha=0.85, edgecolors="white", linewidths=0.6)

    labels = [signal_label(s) for s in base.get_column("signal_id")]
    x = base.get_column("primary_abs").to_numpy()
    y = base.get_column("none_abs").to_numpy()
    texts: list[plt.Text] = []
    for i in range(len(labels)):
        texts.append(ax.text(x[i] + 0.004, y[i] + 0.0008, labels[i], fontsize=8))

    if adjust_text_fn is not None and texts:
        adjust_text_fn(
            texts,
            x=x,
            y=y,
            ax=ax,
            force_text=(0.35, 0.5),
            force_static=(0.2, 0.25),
            expand=(1.05, 1.15),
            ensure_inside_axes=True,
            arrowprops={"arrowstyle": "-", "lw": 0.5, "color": "#9A9A9A", "alpha": 0.75},
        )

    xlim_max = float(max(0.6, np.nanmax(base.get_column("primary_abs").to_numpy()) * 1.05))
    xline = np.linspace(0.0, xlim_max, 400)
    yline = np.minimum(leakage_slope * xline, spillover_cap)
    ax.plot(xline, yline, color="#E45756", linestyle="--", linewidth=1.0)
    ax.axvline(primary_threshold, color="#4C78A8", linestyle="--", linewidth=1.0)
    ax.set_xlabel("Mean range-normalized |Δ| on primary targets")
    ax.set_ylabel("Mean range-normalized |Δ| on none targets")
    ax.grid(linestyle=":", alpha=0.4)

    class_handles = [
        plt.Line2D([0], [0], marker=marker_map[k], color="w", label=k, markerfacecolor="#6B6B6B", markersize=8)
        for k in marker_map
    ]
    construct_handles = [
        plt.Line2D(
            [0], [0], marker="o", color="w", label=label_construct(c), markerfacecolor=color_map[c], markersize=8
        )
        for c in constructs
        if c != "—"
    ]
    size_levels = [0.2, 0.5, 0.8]
    size_handles = []
    for lvl in size_levels:
        area = float(40 + 260 * lvl)
        size_handles.append(
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label=f"{lvl:.1f}",
                markerfacecolor="#8A8A8A",
                markeredgecolor="white",
                markeredgewidth=0.6,
                markersize=float(np.sqrt(area)),
            )
        )
    leg1 = ax.legend(
        handles=class_handles, title="Lever class", loc="center left", bbox_to_anchor=(0.98, 0.48), frameon=False
    )
    ax.add_artist(leg1)
    leg2 = ax.legend(
        handles=construct_handles,
        title="Primary construct",
        loc="center left",
        bbox_to_anchor=(0.98, 0.15),
        frameon=False,
    )
    ax.add_artist(leg2)
    ax.legend(
        handles=size_handles, title="Primary activation", loc="center left", bbox_to_anchor=(0.98, 0.84), frameon=False
    )
    plt.tight_layout()
    return None


def finite_float(value: object) -> float:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return x if np.isfinite(x) else float("nan")


def quantile_from_df(df: pl.DataFrame, col: str, q: float) -> float:
    if df.is_empty() or col not in df.columns:
        return float("nan")
    try:
        out = df.select(pl.col(col).drop_nans().quantile(q, "nearest")).row(0)[0]
    except Exception:
        return float("nan")
    return finite_float(out)


def bootstrap_mean_ci(values: np.ndarray, *, B: int, seed: int) -> tuple[float | None, float | None]:
    vals = np.asarray(values, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if vals.size < 2:
        return None, None
    rng = np.random.default_rng(int(seed))
    idx = rng.integers(0, vals.size, size=(int(B), int(vals.size)))
    means = vals[idx].mean(axis=1)
    return quantile_ci(means)


def bootstrap_gap_ci(
    high_vals: np.ndarray,
    low_vals: np.ndarray,
    *,
    B: int,
    seed: int,
) -> tuple[float | None, float | None]:
    hi = np.asarray(high_vals, dtype=np.float64)
    lo = np.asarray(low_vals, dtype=np.float64)
    hi = hi[np.isfinite(hi)]
    lo = lo[np.isfinite(lo)]
    if hi.size < 2 or lo.size < 2:
        return None, None
    rng = np.random.default_rng(int(seed))
    hi_idx = rng.integers(0, hi.size, size=(int(B), int(hi.size)))
    lo_idx = rng.integers(0, lo.size, size=(int(B), int(lo.size)))
    gaps = hi[hi_idx].mean(axis=1) - lo[lo_idx].mean(axis=1)
    return quantile_ci(gaps)


def belief_mean_update_vector(
    *,
    rows_df: pl.DataFrame,
    mutable_state: Sequence[str],
    bootstrap_mean_ci_fn: Callable[[np.ndarray, int, int | None], tuple[float, float]],
    boot_b: int = 500,
    seed_base: int = 2026_02_40,
) -> pl.DataFrame:
    if rows_df.is_empty():
        return pl.DataFrame()

    rows: list[dict[str, object]] = []
    for idx, var in enumerate(mutable_state):
        name = str(var)
        delta_col = f"delta_{name}"
        pre_col = f"pre_{name}"
        post_col = f"post_{name}"
        abs_col = f"abs_delta_{name}"
        if delta_col in rows_df.columns:
            deltas = rows_df.get_column(delta_col).cast(pl.Float64).to_numpy()
        elif pre_col in rows_df.columns and post_col in rows_df.columns:
            deltas = rows_df.select((pl.col(post_col) - pl.col(pre_col)).cast(pl.Float64)).to_series().to_numpy()
        else:
            continue
        deltas = deltas[np.isfinite(deltas)]
        if deltas.size == 0:
            continue
        mean_delta = float(np.mean(deltas))
        lo, hi = bootstrap_mean_ci_fn(deltas, B=boot_b, seed=seed_base + idx)
        if abs_col in rows_df.columns:
            mean_abs = float(rows_df.get_column(abs_col).cast(pl.Float64).mean())
        else:
            mean_abs = float(np.mean(np.abs(deltas)))
        rows.append(
            {
                "var": name,
                "mean_delta": mean_delta,
                "mean_delta_lo": lo,
                "mean_delta_hi": hi,
                "mean_abs": mean_abs,
            }
        )

    return pl.DataFrame(rows) if rows else pl.DataFrame()


def make_signal_labeler(
    *,
    label_for_fn: Callable[..., str],
    labels: Mapping[str, str],
) -> Callable[[str], str]:
    del label_for_fn

    def _signal_label(signal_id: str) -> str:
        key = str(signal_id)
        return str(labels[key])

    return _signal_label


@dataclass(slots=True)
class UnperturbedContext:
    unperturbed_out: Path
    strategy_order: list[str]
    outcome_panel_w: float
    outcome_order: list[str]
    model_order: list[str]
    cell_metrics: pl.DataFrame
    ref_gini_by_outcome: dict[str, float]
    ref_ci_by_outcome: dict[str, tuple[float | None, float | None]]
    ref_skill_by_outcome: dict[str, float]
    ref_skill_ci_by_outcome: dict[str, tuple[float | None, float | None]]
    bootstrap_cell_ci: dict[tuple[str, str, str], dict[str, tuple[float | None, float | None]]]
    bootstrap_outcome_ci: dict[str, dict[str, tuple[float | None, float | None]]]
    bootstrap_strategy_ci: dict[str, dict[str, tuple[float | None, float | None]]]
    overall_gini: float
    overall_skill: float
    mean_gini_by_outcome: dict[str, float]
    gini_vmin: float
    gini_vmax: float
    skill_vmin: float
    skill_vmax: float


@dataclass(slots=True)
class BeliefContext:
    chosen_model: str
    chosen_strategy: str
    canonical_root: Path
    primary_strategy: str
    primary_strategy_label: str
    summary_all: pl.DataFrame
    summary_all_full: pl.DataFrame
    signals_all: pl.DataFrame
    targets: dict[str, object]
    signals: list[dict[str, object]]
    mutable_state: list[str]
    run_config: dict[str, object]
    signal_summary: pl.DataFrame
    movement_df: pl.DataFrame
    paired_df: pl.DataFrame
    moderation_df: pl.DataFrame
    construct_values: pl.DataFrame
    construct_summary: pl.DataFrame
    per_construct_monotonicity: pl.DataFrame
    overall_monotonicity: float
    top_signals: list[str]
    bottom_signals: list[str]
    none_stability_median: float
    control_stability_median: float
    sign_primary_median: float
    sign_all_movers_median: float
    primary_activation_median: float
    primary_saturation_median: float
    sign_primary_movers_median: float
    sign_all_movers_cond_median: float
    drift_threshold: float
    primary_threshold: float
    leakage_slope: float
    spillover_cap: float
    boot_B: int
    safe_shocks: list[str]
    lever_counts: pl.DataFrame
    example_rows: list[list[str]]
    brittle_construct: str
    frontier_df: pl.DataFrame
    construct_readiness: pl.DataFrame
    shock_df: pl.DataFrame
    shock_library: pl.DataFrame
    signal_spec_fn: Callable[[str], dict[str, object]]
    signal_rows_all_variants_fn: Callable[[str, str], pl.DataFrame]
    moderation_values_fn: Callable[[str, str, list[str]], tuple[np.ndarray, np.ndarray]]
    signal_metric_values_fn: Callable[[str, str], np.ndarray]


@dataclass(slots=True)
class PerturbedContext:
    pert_summary: pl.DataFrame
    keep_outcomes: list[str]
    reference_model_by_outcome: dict[str, str]
    ref_ice_by_outcome: dict[str, pl.DataFrame]
    canonical_levers_for_fn: Callable[[str], list[str]]
    canonical_lever_map: pl.DataFrame
    pert_model_order: list[str]
    main_pert_model_order: list[str]
    pert_lever_metrics: pl.DataFrame
    pert_lever_metrics_main: pl.DataFrame
    pert_block_metrics: pl.DataFrame
    block_strong: pl.DataFrame
    overall_gini_pe: float
    overall_skill: float
    overall_pert_sentence: str
    oracle_meaningful_delta: float
    oracle_gini_uplift: dict[str, float]
    oracle_skill_uplift: dict[str, float]
    block_sensitivity_df: pl.DataFrame
    block_order_map: dict[str, list[str]]
    primary_model: str | None
    primary_strategy: str


def prepare_perturbed_context(
    *,
    perturbed_out: Path,
    outcome_order: Sequence[str],
    ref_ice_root: Path,
    lever_config_path: Path,
    read_csv: Callable[[str | Path], pl.DataFrame] = read_csv_fast,
    oracle_model: str = "__all__",
    oracle_baseline_strategy: str = "data_only",
    oracle_meaningful_delta: float = 0.02,
    main_pert_models: set[str] | None = None,
) -> PerturbedContext:
    if main_pert_models is None:
        main_pert_models = {"gpt-oss-20b"}

    compact_summary_path = perturbed_out / "summary_metrics_cells.csv"
    compact_lever_metrics_path = perturbed_out / "lever_metrics.csv"
    compact_block_metrics_path = perturbed_out / "block_metrics.csv"
    compact_block_sensitivity_path = perturbed_out / "block_sensitivity.csv"
    compact_ref_model_path = perturbed_out / "reference_model_by_outcome.json"
    compact_block_order_path = perturbed_out / "block_order_map.json"
    compact_lever_block_path = perturbed_out / "lever_block_map.csv"

    keep_outcomes = discover_keep_outcomes(perturbed_out, outcome_order)
    lever_config = load_lever_config(lever_config_path)

    reference_model_by_outcome: dict[str, str] = {}
    if compact_ref_model_path.exists():
        try:
            raw_ref_map = json.loads(compact_ref_model_path.read_text(encoding="utf-8"))
            if isinstance(raw_ref_map, dict):
                reference_model_by_outcome = {
                    str(k): str(v) for k, v in raw_ref_map.items() if str(k).strip() and str(v).strip()
                }
        except (OSError, json.JSONDecodeError):
            reference_model_by_outcome = {}
    for outcome in keep_outcomes:
        if outcome in reference_model_by_outcome:
            continue
        model_ref = find_ref_model_for_outcome(
            outcome=outcome,
            lever_config=lever_config,
            ref_ice_root=ref_ice_root,
        )
        if model_ref:
            reference_model_by_outcome[outcome] = model_ref

    def _canonical_levers_for(outcome: str) -> list[str]:
        return canonical_levers_for(
            outcome=outcome,
            lever_config=lever_config,
            reference_model_by_outcome=reference_model_by_outcome,
            ref_ice_root=ref_ice_root,
            csv_reader=read_csv,
        )

    canonical_lever_rows = [
        {"outcome": outcome, "lever": lever} for outcome in keep_outcomes for lever in _canonical_levers_for(outcome)
    ]
    canonical_lever_map = pl.DataFrame(canonical_lever_rows) if canonical_lever_rows else pl.DataFrame()

    def _ref_ice_by_outcome() -> dict[str, pl.DataFrame]:
        ref_map: dict[str, pl.DataFrame] = {}
        for outcome in keep_outcomes:
            model_ref = reference_model_by_outcome.get(outcome)
            if not model_ref:
                continue
            ref_outcome_dir_key = resolve_ref_outcome_dir_key(
                outcome=outcome,
                lever_config=lever_config,
                ref_ice_root=ref_ice_root,
            )
            ref_dir = ref_ice_root / ref_outcome_dir_key
            ref_candidates = [
                ref_dir / f"ice_ref__{model_ref}.csv",
                ref_dir / "ice_ref.csv",
            ]
            ref_path = next((path for path in ref_candidates if path.exists()), None)
            if ref_path is None:
                continue
            df = read_csv(ref_path)
            if not {"id", "target", "ice_low", "ice_high", "delta_z"}.issubset(set(df.columns)):
                continue
            keep_levers = _canonical_levers_for(outcome)
            df = (
                df.filter(pl.col("target").is_in(sorted(keep_levers)))
                .with_columns(
                    (pl.col("ice_high") - pl.col("ice_low")).alias("pe_ref"),
                    pl.col("delta_z").cast(pl.Float64),
                )
                .with_columns(
                    pl.when(pl.col("delta_z").is_finite() & (pl.col("delta_z") != 0))
                    .then(pl.col("pe_ref") / pl.col("delta_z"))
                    .otherwise(float("nan"))
                    .alias("pe_ref_std")
                )
                .select(
                    pl.col("id").cast(pl.Utf8).alias("id"),
                    pl.col("target").cast(pl.Utf8).alias("lever"),
                    pl.col("ice_low").cast(pl.Float64).alias("ice_low_ref"),
                    pl.col("ice_high").cast(pl.Float64).alias("ice_high_ref"),
                    pl.col("pe_ref_std").cast(pl.Float64),
                )
                .drop_nulls("pe_ref_std")
            )
            if not df.is_empty():
                ref_map[outcome] = df
        return ref_map

    ref_ice_by_outcome = _ref_ice_by_outcome()

    def _discover_perturbed_summary_rows(*, calibrated: bool) -> pl.DataFrame:
        sub_order = ["calibrated", "uncalibrated"] if calibrated else ["uncalibrated", "calibrated"]
        rows: list[pl.DataFrame] = []
        seen: set[tuple[str, str, str]] = set()
        for outcome in keep_outcomes:
            outcome_root = perturbed_out / outcome
            if not outcome_root.exists():
                continue
            for sub in sub_order:
                for path in outcome_root.rglob(f"{sub}/summary_metrics.csv"):
                    try:
                        rel = path.relative_to(perturbed_out)
                    except Exception:
                        continue
                    parts = rel.parts
                    if len(parts) < 5:
                        continue
                    o_name, model, strategy = parts[0], parts[1], parts[2]
                    model_slug = resolve_model_id_with_effort(
                        inferred_slug=infer_model_slug(path),
                        fallback_dir_slug=model,
                    )
                    key = (o_name, model_slug, strategy)
                    if key in seen:
                        continue
                    try:
                        df = read_csv(path)
                    except Exception:
                        continue
                    if df.is_empty():
                        continue
                    df = (
                        df.head(1)
                        .with_columns(pl.all().cast(pl.Float64, strict=False))
                        .with_columns(
                            pl.lit(o_name).alias("outcome"),
                            pl.lit(model_slug).alias("model"),
                            pl.lit(strategy).alias("strategy"),
                            pl.lit(sub).alias("calibration"),
                        )
                    )
                    rows.append(df)
                    seen.add(key)
        return pl.concat(rows, how="diagonal") if rows else pl.DataFrame()

    if compact_summary_path.exists():
        pert_summary = read_csv(compact_summary_path)
    else:
        pert_summary = _discover_perturbed_summary_rows(calibrated=True)
    pert_summary = as_float(
        pert_summary,
        [
            "n",
            "skill_pe_std_mean",
            "gini_pe_mean",
            "gini_pe_std_mean",
            "pe_std_var_ratio",
            "gini_pe_std_abs",
            "gini_pe_std_pos",
            "gini_pe_std_neg",
            "sign_agreement_rate",
            "da_no_signal_rate",
            "ice_noncompliance_rate",
            "ice_off_manifold_rate",
            "false_positive_movement_rate",
        ],
    )

    pert_model_order = discover_pert_models(pert_summary)
    main_pert_model_order = _preferred_models_present(
        available_models=pert_model_order,
        preferred_models=sorted(str(m) for m in main_pert_models),
    )
    if not main_pert_model_order and pert_model_order:
        main_pert_model_order = [pert_model_order[0]]

    def _discover_perturbed_lever_metrics() -> pl.DataFrame:
        rows: list[dict[str, object]] = []
        for outcome in keep_outcomes:
            if outcome not in ref_ice_by_outcome:
                continue
            ref_df = ref_ice_by_outcome[outcome]
            keep_levers = _canonical_levers_for(outcome)
            for ice_path in (perturbed_out / outcome).rglob("calibrated/ice_llm.csv"):
                try:
                    rel = ice_path.relative_to(perturbed_out)
                except Exception:
                    continue
                parts = rel.parts
                if len(parts) < 5:
                    continue
                model, strategy = parts[1], parts[2]
                model_slug = resolve_model_id_with_effort(
                    inferred_slug=infer_model_slug(ice_path),
                    fallback_dir_slug=model,
                )
                if not model_slug:
                    continue
                try:
                    llm = read_csv(ice_path)
                except Exception:
                    continue
                if llm.is_empty() or ("target" not in llm.columns):
                    continue
                llm = (
                    llm.filter(pl.col("target").is_in(sorted(keep_levers)))
                    .with_columns(
                        (pl.col("ice_high") - pl.col("ice_low")).alias("pe_llm"),
                        pl.col("delta_z").cast(pl.Float64),
                    )
                    .with_columns(
                        pl.when(pl.col("delta_z").is_finite() & (pl.col("delta_z") != 0))
                        .then(pl.col("pe_llm") / pl.col("delta_z"))
                        .otherwise(float("nan"))
                        .alias("pe_llm_std")
                    )
                    .select(
                        pl.col("id").cast(pl.Utf8).alias("id"),
                        pl.col("target").cast(pl.Utf8).alias("lever"),
                        pl.col("pe_llm_std").cast(pl.Float64),
                    )
                )
                if llm.is_empty():
                    continue
                joined = llm.join(ref_df, on=["id", "lever"], how="inner")
                if joined.is_empty():
                    continue
                for lever in keep_levers:
                    sub = joined.filter(pl.col("lever") == lever)
                    if sub.is_empty():
                        continue
                    pe_ref = sub.get_column("pe_ref_std").to_numpy()
                    pe_llm = sub.get_column("pe_llm_std").to_numpy()
                    mask = np.isfinite(pe_ref) & np.isfinite(pe_llm)
                    epsilon_ref = 0.03
                    active = mask & (np.abs(pe_ref) > epsilon_ref)
                    gini = float("nan")
                    skill0 = float("nan")
                    use_mask = active if active.sum() >= 3 else mask
                    if use_mask.sum() >= 3:
                        gini = gini_np(np.abs(pe_ref[use_mask]), np.abs(pe_llm[use_mask]))
                        abs_err = np.abs(pe_llm[use_mask] - pe_ref[use_mask])
                        denom = float(np.mean(np.abs(pe_ref[use_mask])))
                        mae = float(np.mean(abs_err))
                        if np.isfinite(mae) and np.isfinite(denom) and denom > 0:
                            skill0 = float(1.0 - mae / denom)
                    rows.append(
                        {
                            "outcome": outcome,
                            "model": model_slug,
                            "strategy": strategy,
                            "lever": lever,
                            "gini_pe": float(gini) if np.isfinite(gini) else float("nan"),
                            "skill_pe0": float(skill0) if np.isfinite(skill0) else float("nan"),
                        }
                    )
        return pl.DataFrame(rows) if rows else pl.DataFrame()

    if compact_lever_metrics_path.exists():
        pert_lever_metrics = read_csv(compact_lever_metrics_path)
    else:
        pert_lever_metrics = _discover_perturbed_lever_metrics()
    pert_lever_metrics_main = (
        pert_lever_metrics.join(canonical_lever_map, on=["outcome", "lever"], how="inner")
        if (not pert_lever_metrics.is_empty() and not canonical_lever_map.is_empty())
        else pl.DataFrame()
    )

    overall_gini_pe = (
        float(pert_summary.select(pl.median("gini_pe_std_abs")).row(0)[0])
        if (not pert_summary.is_empty() and "gini_pe_std_abs" in pert_summary.columns)
        else float("nan")
    )
    overall_skill = (
        float(pert_summary.select(pl.median("skill_pe_std_mean")).row(0)[0])
        if (not pert_summary.is_empty() and "skill_pe_std_mean" in pert_summary.columns)
        else float("nan")
    )
    overall_pert_sentence = (
        f"Across available model × strategy cells, median perturbed magnitude ordering is {overall_gini_pe:.2f} "
        f"(PE magnitude Gini on the active set) and median PE skill is {overall_skill:.2f}."
        if np.isfinite(overall_gini_pe) and np.isfinite(overall_skill)
        else "Across available model × strategy cells, perturbed alignment varies widely by outcome and prompt family."
    )

    model_plan_path = Path("../thesis/artifacts/empirical/modeling/model_plan.json")
    model_plan_entries = load_model_plan_entries(model_plan_path) if model_plan_path.exists() else []

    def _select_model_plan_entry(outcome: str) -> dict[str, object] | None:
        candidates: list[dict[str, object]] = []
        for entry in model_plan_entries:
            if entry.get("outcome") != outcome:
                continue
            if not isinstance(entry.get("predictors"), dict):
                continue
            candidates.append(entry)
        if not candidates:
            return None
        model_ref = reference_model_by_outcome.get(outcome)
        if isinstance(model_ref, str) and model_ref.strip():
            for entry in candidates:
                if str(entry.get("model") or "") == model_ref:
                    return entry

        def _score(entry: dict[str, object]) -> tuple[int, str]:
            model = str(entry.get("model") or "")
            desc = str(entry.get("description") or "").lower()
            if model.endswith("_ENGINES_REF"):
                return (0, model)
            if "ENGINES_REF" in model:
                return (1, model)
            if "ref engines" in desc:
                return (2, model)
            return (10, model)

        return sorted(candidates, key=_score)[0]

    normalize_map = {
        **BLOCK_NORMALIZE_MAP,
        "institutional_trust": "trust_engine",
    }

    def _normalize_block_name(block: str) -> str:
        raw = str(block).strip()
        return str(normalize_map.get(raw, raw))

    def _target_to_block_map(outcome: str) -> dict[str, str]:
        entry = _select_model_plan_entry(outcome)
        if entry is None:
            return {}
        predictors = entry.get("predictors")
        if not isinstance(predictors, dict):
            return {}
        out: dict[str, str] = {}
        for block, preds in predictors.items():
            if not isinstance(block, str) or not block.strip():
                continue
            if not isinstance(preds, list):
                continue
            for pred in preds:
                if isinstance(pred, str) and pred.strip():
                    out.setdefault(pred.strip(), _normalize_block_name(block.strip()))
        return out

    block_order_map: dict[str, list[str]] = {}
    if compact_block_order_path.exists():
        try:
            raw_block_order = json.loads(compact_block_order_path.read_text(encoding="utf-8"))
            if isinstance(raw_block_order, dict):
                block_order_map = {
                    str(k): [str(v) for v in vals if str(v).strip()]
                    for k, vals in raw_block_order.items()
                    if isinstance(vals, list)
                }
        except (OSError, json.JSONDecodeError):
            block_order_map = {}
    lever_block_df = read_csv(compact_lever_block_path) if compact_lever_block_path.exists() else pl.DataFrame()
    if lever_block_df.is_empty():
        lever_block_rows: list[dict[str, str]] = []
        for outcome in keep_outcomes:
            entry = _select_model_plan_entry(outcome)
            blocks = entry.get("blocks") if entry and isinstance(entry.get("blocks"), list) else []
            if outcome not in block_order_map:
                block_order_map[outcome] = [
                    _normalize_block_name(str(b)) for b in blocks if isinstance(b, str) and str(b).strip()
                ]
            mapping = _target_to_block_map(outcome)
            keep_levers = set(_canonical_levers_for(outcome))
            for lever, block in mapping.items():
                if lever in keep_levers:
                    lever_block_rows.append({"outcome": outcome, "lever": lever, "block": block})
        lever_block_df = pl.DataFrame(lever_block_rows) if lever_block_rows else pl.DataFrame()

    if compact_block_metrics_path.exists():
        pert_block_metrics = read_csv(compact_block_metrics_path)
    else:
        pert_block_metrics = (
            pert_lever_metrics_main.join(lever_block_df, on=["outcome", "lever"], how="inner")
            if (not pert_lever_metrics_main.is_empty() and not lever_block_df.is_empty())
            else pl.DataFrame()
        )
        if not pert_block_metrics.is_empty():
            pert_block_metrics = pert_block_metrics.group_by("outcome", "model", "strategy", "block").agg(
                pl.median("gini_pe").alias("gini_block_median"),
                pl.median("skill_pe0").alias("skill_block_median"),
            )

    def _strong_signal_concordance() -> pl.DataFrame:
        rows: list[dict[str, object]] = []
        for outcome in keep_outcomes:
            if outcome not in ref_ice_by_outcome:
                continue
            ref_df = ref_ice_by_outcome[outcome]
            keep_levers = _canonical_levers_for(outcome)
            block_map = (
                lever_block_df.filter(pl.col("outcome") == outcome) if not lever_block_df.is_empty() else pl.DataFrame()
            )
            if block_map.is_empty():
                continue
            for ice_path in (perturbed_out / outcome).rglob("calibrated/ice_llm.csv"):
                try:
                    rel = ice_path.relative_to(perturbed_out)
                except Exception:
                    continue
                parts = rel.parts
                if len(parts) < 5:
                    continue
                model, strategy = parts[1], parts[2]
                try:
                    llm = read_csv(ice_path)
                except Exception:
                    continue
                if llm.is_empty() or ("target" not in llm.columns):
                    continue
                llm = (
                    llm.filter(pl.col("target").is_in(sorted(keep_levers)))
                    .with_columns(
                        (pl.col("ice_high") - pl.col("ice_low")).alias("pe_llm"),
                        pl.col("delta_z").cast(pl.Float64),
                    )
                    .with_columns(
                        pl.when(pl.col("delta_z").is_finite() & (pl.col("delta_z") != 0))
                        .then(pl.col("pe_llm") / pl.col("delta_z"))
                        .otherwise(float("nan"))
                        .alias("pe_llm_std")
                    )
                    .select(
                        pl.col("id").cast(pl.Utf8).alias("id"),
                        pl.col("target").cast(pl.Utf8).alias("lever"),
                        pl.col("pe_llm_std").cast(pl.Float64),
                    )
                )
                joined = llm.join(ref_df, on=["id", "lever"], how="inner").join(block_map, on="lever", how="inner")
                if joined.is_empty():
                    continue
                for block in sorted(set(joined.get_column("block").to_list())):
                    sub = joined.filter(pl.col("block") == block).filter(pl.col("pe_ref_std").is_finite())
                    if sub.is_empty():
                        continue
                    abs_ref = np.abs(sub.get_column("pe_ref_std").to_numpy())
                    if abs_ref.size < 5:
                        continue
                    thresh = float(np.quantile(abs_ref, 0.75))
                    strong = sub.filter(pl.col("pe_ref_std").abs() >= thresh)
                    if strong.is_empty():
                        continue
                    ref_vals = strong.get_column("pe_ref_std").to_numpy()
                    llm_vals = strong.get_column("pe_llm_std").to_numpy()
                    mask = np.isfinite(ref_vals) & np.isfinite(llm_vals)
                    if mask.sum() < 3:
                        continue
                    rows.append(
                        {
                            "outcome": outcome,
                            "model": model,
                            "strategy": strategy,
                            "block": block,
                            "top_quartile_sign": float(np.mean(np.sign(ref_vals[mask]) == np.sign(llm_vals[mask]))),
                        }
                    )
        return pl.DataFrame(rows) if rows else pl.DataFrame()

    block_strong = pl.DataFrame() if compact_block_sensitivity_path.exists() else _strong_signal_concordance()

    ref_lobo_path = Path("../thesis/artifacts/empirical/modeling/model_lobo_all.csv")
    ref_lobo = read_csv(ref_lobo_path) if ref_lobo_path.exists() else pl.DataFrame()

    def _block_alignment(
        outcome: str, model: str, strategy: str, calibration_status: str = "uncalibrated"
    ) -> pl.DataFrame:
        epsilon_ref = 0.03
        ref_df = ref_ice_by_outcome.get(outcome)
        if ref_df is None or ref_df.is_empty():
            return pl.DataFrame()
        mapping = _target_to_block_map(outcome)
        if not mapping:
            return pl.DataFrame()
        keep_levers = sorted(mapping)
        candidate_dirs = _candidate_model_dirs_for_outcome(
            perturbed_out=perturbed_out,
            outcome=outcome,
            model_id=model,
        )
        candidate_paths = [d / strategy / calibration_status / "ice_llm.csv" for d in candidate_dirs]
        path = next((p for p in candidate_paths if p.exists()), None)
        if path is None:
            return pl.DataFrame()
        llm = read_csv(path)
        if llm.is_empty() or "target" not in llm.columns:
            return pl.DataFrame()
        llm = (
            llm.filter(pl.col("target").is_in(keep_levers))
            .with_columns(
                (pl.col("ice_high") - pl.col("ice_low")).alias("pe_llm"),
                pl.col("delta_z").cast(pl.Float64),
            )
            .with_columns(
                pl.when(pl.col("delta_z").is_finite() & (pl.col("delta_z") != 0))
                .then(pl.col("pe_llm") / pl.col("delta_z"))
                .otherwise(float("nan"))
                .alias("pe_llm_std")
            )
            .select(
                pl.col("id").cast(pl.Utf8).alias("id"),
                pl.col("target").cast(pl.Utf8).alias("lever"),
                pl.col("pe_llm_std").cast(pl.Float64),
            )
        )
        block_df = pl.DataFrame([{"lever": k, "block": v} for k, v in mapping.items()])
        joined = llm.join(ref_df.select(["id", "lever", "pe_ref_std"]), on=["id", "lever"], how="inner")
        joined = joined.join(block_df, on="lever", how="inner")
        joined = joined.filter(pl.col("pe_ref_std").is_finite() & pl.col("pe_llm_std").is_finite())
        if joined.is_empty():
            return pl.DataFrame()
        rows: list[dict[str, object]] = []
        for block in sorted({str(v) for v in joined.get_column("block").to_list()}):
            sub = joined.filter(pl.col("block") == block)
            if sub.is_empty():
                continue
            sub_active = sub.filter(pl.col("pe_ref_std").abs() > epsilon_ref)
            sub_use = sub_active if sub_active.height >= 3 else sub
            if sub_use.height < 3:
                continue
            m_ref = float(sub_use.select(pl.col("pe_ref_std").abs().median()).row(0)[0])
            m_llm = float(sub_use.select(pl.col("pe_llm_std").abs().median()).row(0)[0])
            if not (np.isfinite(m_ref) and np.isfinite(m_llm)):
                continue
            rows.append({"block": str(block), "m_ref": m_ref, "m_llm": m_llm})
        if not rows:
            return pl.DataFrame()
        return (
            pl.DataFrame(rows)
            .with_columns(_normalize_block_expr("block").alias("block"))
            .with_columns(pl.col("m_ref").cast(pl.Float64), pl.col("m_llm").cast(pl.Float64))
            .select(["block", "m_ref", "m_llm"])
        )

    def _ref_lobo_size(outcome: str) -> pl.DataFrame:
        if ref_lobo.is_empty():
            return pl.DataFrame()
        model_ref = reference_model_by_outcome.get(outcome)
        if not model_ref:
            return pl.DataFrame()
        df = ref_lobo.filter(
            (pl.col("outcome") == outcome)
            & (pl.col("model") == model_ref)
            & (pl.col("kind") == "lobo")
            & (pl.col("metric") == "pseudo_r2")
        )
        if df.is_empty():
            return pl.DataFrame()
        weighted = (
            df.select(["block", "lobo_bic_weight"])
            .with_columns(_normalize_block_expr("block").alias("block"))
            .with_columns(pl.col("lobo_bic_weight").cast(pl.Float64, strict=False))
            .filter(pl.col("lobo_bic_weight").is_finite() & (pl.col("lobo_bic_weight") >= 0.0))
            .group_by("block")
            .agg(pl.median("lobo_bic_weight").alias("importance_share"))
        )
        if not weighted.is_empty():
            total = weighted.select(pl.sum("importance_share")).row(0)[0]
            total_f = float(total) if total is not None else float("nan")
            if np.isfinite(total_f) and total_f > 0.0:
                return weighted.with_columns((pl.col("importance_share") / pl.lit(total_f)).alias("importance_share"))

        fallback = (
            df.select(["block", "delta", "baseline_value", "rel_delta"])
            .with_columns(_normalize_block_expr("block").alias("block"))
            .with_columns(
                pl.when(
                    pl.col("baseline_value").is_not_null() & (pl.col("baseline_value").cast(pl.Float64).abs() > 0.0)
                )
                .then(pl.col("delta").cast(pl.Float64) / pl.col("baseline_value").cast(pl.Float64))
                .otherwise(pl.col("rel_delta").cast(pl.Float64))
                .alias("lobo_rel_importance")
            )
            .with_columns(pl.col("lobo_rel_importance").cast(pl.Float64).abs().alias("_abs_imp"))
            .group_by("block")
            .agg(pl.median("_abs_imp").alias("importance_share"))
        )
        if fallback.is_empty():
            return pl.DataFrame()
        total_fb = fallback.select(pl.sum("importance_share")).row(0)[0]
        total_fb_f = float(total_fb) if total_fb is not None else float("nan")
        if np.isfinite(total_fb_f) and total_fb_f > 0.0:
            return fallback.with_columns((pl.col("importance_share") / pl.lit(total_fb_f)).alias("importance_share"))
        return fallback

    if compact_block_sensitivity_path.exists():
        block_sensitivity_df = read_csv(compact_block_sensitivity_path)
    else:
        block_rows: list[dict[str, object]] = []
        if not pert_summary.is_empty() and {"model", "strategy"}.issubset(set(pert_summary.columns)):
            pair_df = (
                pert_summary.select(
                    pl.col("model").cast(pl.Utf8).alias("model"),
                    pl.col("strategy").cast(pl.Utf8).alias("strategy"),
                )
                .unique()
                .sort(["model", "strategy"])
            )
            model_strategy_pairs = [(str(r[0]), str(r[1])) for r in pair_df.iter_rows()]
        else:
            model_strategy_pairs = [(str(m), str(s)) for m in pert_model_order for s in PARETO_STRATEGY_ORDER]

        for outcome in outcome_order:
            outcome_key = str(outcome)
            outcome_rows: list[dict[str, object]] = []
            for model_id, strategy_id in model_strategy_pairs:
                align_block = _block_alignment(outcome_key, model_id, strategy_id)
                if align_block.is_empty():
                    continue
                for row in align_block.iter_rows(named=True):
                    outcome_rows.append(
                        {
                            "outcome": outcome_key,
                            "block": str(row.get("block")),
                            "m_ref": row.get("m_ref"),
                            "m_llm": row.get("m_llm"),
                        }
                    )
            if not outcome_rows:
                continue
            merged = (
                pl.DataFrame(outcome_rows)
                .with_columns(
                    pl.col("m_ref").cast(pl.Float64, strict=False),
                    pl.col("m_llm").cast(pl.Float64, strict=False),
                )
                .filter(pl.col("m_ref").is_finite() & pl.col("m_llm").is_finite())
                .group_by("outcome", "block")
                .agg(
                    pl.median("m_ref").alias("m_ref"),
                    pl.median("m_llm").alias("m_llm"),
                )
            )
            if merged.is_empty():
                continue
            size_block = _ref_lobo_size(outcome_key)
            merged = merged.join(size_block, on="block", how="left") if not size_block.is_empty() else merged
            for row in merged.iter_rows(named=True):
                block_rows.append(
                    {
                        "outcome": outcome_key,
                        "block": str(row.get("block")),
                        "m_ref": row.get("m_ref"),
                        "m_llm": row.get("m_llm"),
                        "importance_share": row.get("importance_share"),
                    }
                )
        block_sensitivity_df = pl.DataFrame(block_rows) if block_rows else pl.DataFrame()
    primary_model: str | None = None
    primary_strategy = "all_model_strategy_median"

    def _oracle_uplift_stats(df: pl.DataFrame, *, metric_col: str) -> dict[str, float]:
        if df.is_empty() or metric_col not in df.columns or not model_plan_entries:
            return {}
        lever_to_block: dict[tuple[str, str], str] = {}
        for outcome in keep_outcomes:
            entry = _select_model_plan_entry(outcome)
            if entry is None:
                continue
            predictors = entry.get("predictors")
            if not isinstance(predictors, dict):
                continue
            for block, preds in predictors.items():
                if not isinstance(block, str) or not isinstance(preds, list):
                    continue
                for lever in preds:
                    if isinstance(lever, str) and lever.strip():
                        lever_to_block[(outcome, lever.strip())] = block.strip()
        if not lever_to_block:
            return {}

        rows = []
        for r in df.select("outcome", "model", "strategy", "lever", metric_col).iter_rows(named=True):
            key = (str(r["outcome"]), str(r["lever"]))
            block = lever_to_block.get(key)
            if not block:
                continue
            rows.append(
                {
                    "outcome": str(r["outcome"]),
                    "model": str(r["model"]),
                    "strategy": str(r["strategy"]),
                    "block": block,
                    "metric": float(r[metric_col]),
                }
            )
        if not rows:
            return {}
        block_df = (
            pl.DataFrame(rows)
            .with_columns(pl.col("metric").cast(pl.Float64))
            .group_by("outcome", "model", "strategy", "block")
            .agg(pl.median("metric").alias("metric_block_median"))
        )
        available_models = sorted({str(v) for v in block_df.get_column("model").to_list()})
        if not available_models:
            return {}

        if str(oracle_model).strip() == "__all__":
            model_filter = pl.col("model").is_in(available_models)
        else:
            model_candidates = [m for m in available_models if m == oracle_model or m.startswith(f"{oracle_model}_")]
            if not model_candidates:
                return {}
            model_filter = pl.col("model").is_in(model_candidates)

        sub = block_df.filter(model_filter & pl.col("metric_block_median").is_finite())
        if sub.is_empty():
            return {}
        baseline = sub.filter(pl.col("strategy") == oracle_baseline_strategy).select(
            "outcome", "model", "block", pl.col("metric_block_median").cast(pl.Float64).alias("baseline")
        )
        if baseline.is_empty():
            return {}
        oracle_best = sub.group_by("outcome", "model", "block").agg(
            pl.col("metric_block_median").cast(pl.Float64).max().alias("oracle_best")
        )
        delta = (
            baseline.join(oracle_best, on=["outcome", "model", "block"], how="inner")
            .with_columns((pl.col("oracle_best") - pl.col("baseline")).alias("delta"))
            .get_column("delta")
            .to_numpy()
        )
        delta = delta[np.isfinite(delta)]
        if delta.size == 0:
            return {}
        return {
            "median": float(np.median(delta)),
            "q25": float(np.quantile(delta, 0.25)),
            "q75": float(np.quantile(delta, 0.75)),
            "win_rate": float(np.mean(delta > 0.0)),
            "meaningful_rate": float(np.mean(delta > oracle_meaningful_delta)),
        }

    oracle_gini_uplift = _oracle_uplift_stats(pert_lever_metrics_main, metric_col="gini_pe")
    oracle_skill_uplift = _oracle_uplift_stats(pert_lever_metrics_main, metric_col="skill_pe0")

    return PerturbedContext(
        pert_summary=pert_summary,
        keep_outcomes=keep_outcomes,
        reference_model_by_outcome=reference_model_by_outcome,
        ref_ice_by_outcome=ref_ice_by_outcome,
        canonical_levers_for_fn=_canonical_levers_for,
        canonical_lever_map=canonical_lever_map,
        pert_model_order=pert_model_order,
        main_pert_model_order=main_pert_model_order,
        pert_lever_metrics=pert_lever_metrics,
        pert_lever_metrics_main=pert_lever_metrics_main,
        pert_block_metrics=pert_block_metrics,
        block_strong=block_strong,
        overall_gini_pe=overall_gini_pe,
        overall_skill=overall_skill,
        overall_pert_sentence=overall_pert_sentence,
        oracle_meaningful_delta=oracle_meaningful_delta,
        oracle_gini_uplift=oracle_gini_uplift,
        oracle_skill_uplift=oracle_skill_uplift,
        block_sensitivity_df=block_sensitivity_df,
        block_order_map=block_order_map,
        primary_model=primary_model,
        primary_strategy=primary_strategy,
    )


def prepare_belief_context(
    *,
    belief_out: Path,
    chosen_model: str,
    chosen_strategy: str,
    read_csv: Callable[[str | Path], pl.DataFrame] = read_csv_fast,
    signal_label: Callable[[str], str],
    label_strategy: Callable[[str], str],
    label_construct: Callable[[str], str],
    fmt_ci: Callable[..., str],
    safe_none_stability: float = 0.83,
    safe_control_stability: float = 0.90,
    safe_sign_primary: float = 0.27,
    drift_threshold: float = 0.03,
    primary_threshold: float = 0.125,
    spillover_cap: float = 0.06,
    boot_B: int = 500,
) -> BeliefContext:
    leakage_slope = drift_threshold / primary_threshold
    summary_all_path = belief_out / "summary_metrics" / "all_models_strategies.csv"
    signals_all_path = belief_out / "summary_metrics" / "all_models_signals.csv"
    canonical_root = belief_out / "canonical" / chosen_model
    targets_path = canonical_root / "belief_update_targets.json"
    scales_path = canonical_root / "construct_scales.json"
    run_config_path = canonical_root / "run_config.json"

    summary_all = read_csv(summary_all_path) if summary_all_path.exists() else pl.DataFrame()
    signals_all = read_csv(signals_all_path) if signals_all_path.exists() else pl.DataFrame()
    summary_all_full = summary_all

    if not summary_all.is_empty():
        canonical_model = canonical_root.name
        if canonical_model in set(summary_all.get_column("model").to_list()):
            primary_model = canonical_model
        else:
            model_summary = (
                summary_all.group_by("model")
                .agg(pl.mean("parsed_rate").alias("parsed_rate"))
                .sort("parsed_rate", descending=True)
            )
            primary_model = str(model_summary.row(0)[0]) if not model_summary.is_empty() else canonical_model
        summary_all = summary_all.filter(pl.col("model") == primary_model)

    def _pick_primary_strategy(df: pl.DataFrame) -> str:
        if df.is_empty():
            return "first_person"
        gate = df.with_columns(
            (
                pl.col("none_stability_rate")
                + pl.col("control_stability_rate")
                + pl.col("paired_mild_strong_monotonicity_rate_mean")
            ).alias("gate_score")
        ).sort("gate_score", descending=True)
        return str(gate.row(0, named=True)["strategy"])

    if not summary_all.is_empty() and chosen_strategy in set(summary_all.get_column("strategy").to_list()):
        primary_strategy = chosen_strategy
    else:
        primary_strategy = _pick_primary_strategy(summary_all)
    primary_strategy_label = label_strategy(primary_strategy)

    targets_obj = json.loads(targets_path.read_text(encoding="utf-8")) if targets_path.exists() else {}
    targets: dict[str, object] = targets_obj if isinstance(targets_obj, dict) else {}
    raw_signals = targets.get("signals", []) if isinstance(targets, dict) else []
    signals = [x for x in raw_signals if isinstance(x, dict)]
    mutable_state_raw = targets.get("mutable_state", []) if isinstance(targets, dict) else []
    mutable_state = [str(x) for x in mutable_state_raw if str(x).strip()]

    scale_payload = json.loads(scales_path.read_text(encoding="utf-8")) if scales_path.exists() else {}
    scale_list = scale_payload.get("construct_scales", []) if isinstance(scale_payload, dict) else []
    scale_map = {str(entry.get("var")): entry for entry in scale_list if isinstance(entry, dict)}

    run_config_obj = json.loads(run_config_path.read_text(encoding="utf-8")) if run_config_path.exists() else {}
    run_config: dict[str, object] = run_config_obj if isinstance(run_config_obj, dict) else {}
    eps_map = run_config.get("eps", {}) if isinstance(run_config, dict) else {}
    if not isinstance(eps_map, dict):
        eps_map = {}

    def signal_spec(signal_id: str) -> dict[str, object]:
        for item in signals:
            if item.get("id") == signal_id:
                return item
        return {}

    rows_cache: dict[tuple[str, str, str], pl.DataFrame] = {}

    def signal_rows_frame(strategy: str, signal_id: str, variant: str) -> pl.DataFrame:
        key = (strategy, signal_id, variant)
        if key in rows_cache:
            return rows_cache[key]
        path = canonical_root / strategy / "signals" / signal_id / variant / "rows.jsonl"
        if not path.exists():
            rows_cache[key] = pl.DataFrame()
            return rows_cache[key]
        rows: list[dict[str, object]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                rows.append(obj)
        rows_cache[key] = pl.DataFrame(rows) if rows else pl.DataFrame()
        return rows_cache[key]

    def _eps_for(var: str) -> float:
        if var in eps_map:
            try:
                return float(eps_map[var])
            except (TypeError, ValueError):
                pass
        scale = scale_map.get(var, {})
        try:
            return float(scale.get("eps", 0.05))
        except (TypeError, ValueError):
            return 0.05

    def _bounds_for(var: str) -> tuple[float | None, float | None]:
        scale = scale_map.get(var, {})
        try:
            lo = float(scale.get("lo"))
            hi = float(scale.get("hi"))
            return lo, hi
        except (TypeError, ValueError):
            return None, None

    def _span_for(var: str) -> float | None:
        lo, hi = _bounds_for(var)
        if lo is None or hi is None:
            return None
        span = abs(float(hi) - float(lo))
        return span if np.isfinite(span) and span > 0.0 else None

    def _normalized_abs_delta_exprs(df: pl.DataFrame, vars_list: list[str]) -> list[pl.Expr]:
        exprs: list[pl.Expr] = []
        for var in vars_list:
            col_norm = f"abs_delta_norm_{var}"
            if col_norm in df.columns:
                exprs.append(pl.col(col_norm).cast(pl.Float64))
                continue
            col = f"abs_delta_{var}"
            if col in df.columns:
                span = _span_for(str(var))
                if span is not None:
                    exprs.append(pl.col(col).cast(pl.Float64) / pl.lit(float(span)))
                else:
                    exprs.append(pl.col(col).cast(pl.Float64))
        return exprs

    def _signal_variant_summary(strategy: str, signal_id: str, variant: str) -> dict[str, float] | None:
        path = canonical_root / strategy / "signals" / signal_id / variant / "summary.json"
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        summary = {
            "n_rows": finite_float(data.get("n_rows")),
            "n_primary_mover_rows": finite_float(data.get("n_primary_mover_rows")),
            "sign_pass_rate_primary": finite_float(data.get("sign_pass_rate_primary")),
            "sign_pass_rate_all_non_none": finite_float(data.get("sign_pass_rate_all_non_none")),
            "sign_pass_rate_primary_movers": finite_float(data.get("sign_pass_rate_primary_movers")),
            "sign_pass_rate_all_non_none_movers": finite_float(data.get("sign_pass_rate_all_non_none_movers")),
            "primary_activation_rate": finite_float(data.get("primary_activation_rate")),
            "primary_saturation_rate": finite_float(data.get("primary_saturation_rate")),
            "none_stability_rate": finite_float(data.get("none_stability_rate")),
            "control_stability_rate": finite_float(data.get("control_stability_rate")),
            "kendall_strength_alignment": finite_float(data.get("kendall_strength_alignment")),
            "tier_monotonicity": finite_float(data.get("tier_monotonicity")),
        }
        needs_rows = any(
            not np.isfinite(summary.get(k, float("nan")))
            for k in (
                "n_primary_mover_rows",
                "sign_pass_rate_primary_movers",
                "sign_pass_rate_all_non_none_movers",
                "primary_activation_rate",
                "primary_saturation_rate",
            )
        )
        if not needs_rows:
            return summary

        spec = signal_spec(signal_id)
        primary_targets = [str(x) for x in spec.get("primary_targets", []) if str(x).strip()]
        ranking = spec.get("mutable_state_ranking", []) or []
        expected_movers = [
            str(r.get("var"))
            for r in ranking
            if isinstance(r, dict) and str(r.get("direction")).strip().lower() != "none"
        ]
        df = signal_rows_frame(strategy, signal_id, variant)
        if df.is_empty() or not primary_targets:
            return summary

        def _any_mover_expr(vars_list: list[str]) -> pl.Expr | None:
            cols = []
            for var in vars_list:
                col = f"abs_delta_{var}"
                if col in df.columns:
                    cols.append(pl.col(col) > _eps_for(var))
            return pl.any_horizontal(cols) if cols else None

        def _mean_sign_expr(vars_list: list[str]) -> pl.Expr | None:
            cols = []
            for var in vars_list:
                col = f"sign_ok_{var}"
                if col in df.columns:
                    cols.append(pl.col(col).cast(pl.Float64))
            return pl.mean_horizontal(cols) if cols else None

        primary_mover_expr = _any_mover_expr(primary_targets)
        expected_mover_expr = _any_mover_expr(expected_movers)
        primary_sign_expr = _mean_sign_expr(primary_targets)
        expected_sign_expr = _mean_sign_expr(expected_movers)

        if primary_mover_expr is not None:
            df = df.with_columns(primary_mover_expr.alias("_primary_mover"))
            n_primary_mover_rows = float(df.select(pl.col("_primary_mover").sum()).row(0)[0])
            summary["n_primary_mover_rows"] = n_primary_mover_rows
            summary["primary_activation_rate"] = (
                n_primary_mover_rows / float(df.height) if df.height > 0 else float("nan")
            )
            if primary_sign_expr is not None:
                sign_vals = (
                    df.with_columns(primary_sign_expr.alias("_sign_primary"))
                    .filter(pl.col("_primary_mover"))
                    .select(pl.col("_sign_primary").mean())
                    .row(0)[0]
                )
                summary["sign_pass_rate_primary_movers"] = finite_float(sign_vals)

        if expected_mover_expr is not None:
            df = df.with_columns(expected_mover_expr.alias("_expected_mover"))
            if expected_sign_expr is not None:
                sign_vals = (
                    df.with_columns(expected_sign_expr.alias("_sign_expected"))
                    .filter(pl.col("_expected_mover"))
                    .select(pl.col("_sign_expected").mean())
                    .row(0)[0]
                )
                summary["sign_pass_rate_all_non_none_movers"] = finite_float(sign_vals)

        sat_exprs = []
        for var in primary_targets:
            pre_col = f"pre_{var}"
            exp_col = f"expected_{var}"
            if pre_col not in df.columns or exp_col not in df.columns:
                continue
            lo, hi = _bounds_for(var)
            if lo is None or hi is None:
                continue
            eps = _eps_for(var)
            exp = pl.col(exp_col).cast(pl.Utf8).str.to_lowercase()
            sat_exprs.append(
                (exp.is_in(["up", "increase", "higher"]) & (pl.col(pre_col) >= (hi - eps)))
                | (exp.is_in(["down", "decrease", "lower"]) & (pl.col(pre_col) <= (lo + eps)))
            )
        if sat_exprs:
            sat_rate = float(df.select(pl.any_horizontal(sat_exprs).mean()).row(0)[0])
            summary["primary_saturation_rate"] = finite_float(sat_rate)
        return summary

    def _agg_variant(values_list: list[dict[str, float]], key: str, reducer: Callable[[list[float]], float]) -> float:
        vals = [v.get(key, float("nan")) for v in values_list]
        vals = [v for v in vals if np.isfinite(v)]
        return float(reducer(vals)) if vals else float("nan")

    signal_rows: list[dict[str, object]] = []
    for item in signals:
        sid = str(item.get("id"))
        variants: list[dict[str, float]] = []
        for variant in ("mild", "strong"):
            data = _signal_variant_summary(primary_strategy, sid, variant)
            if data:
                variants.append(data)
        if not variants:
            continue

        signal_rows.append(
            {
                "signal_id": sid,
                "primary_targets": list(item.get("primary_targets", [])),
                "n_rows": _agg_variant(variants, "n_rows", np.sum),
                "n_primary_mover_rows": _agg_variant(variants, "n_primary_mover_rows", np.sum),
                "sign_pass_rate_primary": _agg_variant(variants, "sign_pass_rate_primary", np.mean),
                "sign_pass_rate_all_non_none": _agg_variant(variants, "sign_pass_rate_all_non_none", np.mean),
                "sign_pass_rate_primary_movers": _agg_variant(variants, "sign_pass_rate_primary_movers", np.mean),
                "sign_pass_rate_all_non_none_movers": _agg_variant(
                    variants, "sign_pass_rate_all_non_none_movers", np.mean
                ),
                "primary_activation_rate": _agg_variant(variants, "primary_activation_rate", np.mean),
                "primary_saturation_rate": _agg_variant(variants, "primary_saturation_rate", np.mean),
                "none_stability_rate": _agg_variant(variants, "none_stability_rate", np.mean),
                "control_stability_rate": _agg_variant(variants, "control_stability_rate", np.mean),
                "kendall_strength_alignment": _agg_variant(variants, "kendall_strength_alignment", np.mean),
                "tier_monotonicity": _agg_variant(variants, "tier_monotonicity", np.mean),
            }
        )
    signal_summary = pl.DataFrame(signal_rows) if signal_rows else pl.DataFrame()

    movement_rows: list[dict[str, object]] = []
    for item in signals:
        sid = str(item.get("id"))
        ranking = item.get("mutable_state_ranking", []) or []
        none_vars = [r.get("var") for r in ranking if r.get("direction") == "none"]
        primary_targets = list(item.get("primary_targets", []))
        if not primary_targets:
            continue
        per_variant: list[tuple[float, float]] = []
        for variant in ("mild", "strong"):
            df = signal_rows_frame(primary_strategy, sid, variant)
            if df.is_empty():
                continue
            primary_exprs = _normalized_abs_delta_exprs(df, [str(v) for v in primary_targets])
            none_exprs = _normalized_abs_delta_exprs(df, [str(v) for v in none_vars])
            if not primary_exprs or not none_exprs:
                continue
            df = df.with_columns(
                pl.mean_horizontal(primary_exprs).alias("primary_abs"),
                pl.mean_horizontal(none_exprs).alias("none_abs"),
            )
            per_variant.append(
                (
                    float(df.select(pl.mean("primary_abs")).row(0)[0]),
                    float(df.select(pl.mean("none_abs")).row(0)[0]),
                )
            )
        if not per_variant:
            continue
        movement_rows.append(
            {
                "signal_id": sid,
                "primary_abs": float(np.mean([v[0] for v in per_variant])),
                "none_abs": float(np.mean([v[1] for v in per_variant])),
            }
        )
    movement_df = pl.DataFrame(movement_rows) if movement_rows else pl.DataFrame()

    def _paired_checks(signal_id: str) -> dict[str, object] | None:
        path = canonical_root / primary_strategy / "signals" / signal_id / "paired_checks.json"
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    paired_rows: list[dict[str, object]] = []
    moderation_rows: list[dict[str, object]] = []
    for item in signals:
        sid = str(item.get("id"))
        checks = _paired_checks(sid)
        if not checks:
            continue
        mild_strong = checks.get("mild_strong", {})
        rate = mild_strong.get("mild_strong_monotonicity_rate")
        per_var = mild_strong.get("per_var", {}) if isinstance(mild_strong, dict) else {}
        paired_rows.append(
            {
                "signal_id": sid,
                "mild_strong_monotonicity_rate": float(rate) if rate is not None else float("nan"),
                "per_var": per_var,
            }
        )
        for test in checks.get("moderation_tests", []) or []:
            moderation_rows.append(
                {
                    "signal_id": sid,
                    "moderator": test.get("moderator"),
                    "targets": ", ".join(test.get("targets", []) or []),
                    "expected": test.get("expected"),
                    "pass_rate_high": test.get("pass_rate_high"),
                    "pass_rate_low": test.get("pass_rate_low"),
                    "moderation_pass": test.get("moderation_pass"),
                    "moderation_gap": test.get("moderation_gap"),
                    "reason": test.get("reason"),
                }
            )
    paired_df = pl.DataFrame(paired_rows) if paired_rows else pl.DataFrame()
    moderation_df = pl.DataFrame(moderation_rows) if moderation_rows else pl.DataFrame()

    if not signal_summary.is_empty():
        construct_values = signal_summary.explode("primary_targets")
        if not movement_df.is_empty():
            construct_values = construct_values.join(movement_df, on="signal_id", how="left")
        if not paired_df.is_empty():
            construct_values = construct_values.join(
                paired_df.select("signal_id", "mild_strong_monotonicity_rate"),
                on="signal_id",
                how="left",
            )
        construct_summary = (
            construct_values.group_by("primary_targets")
            .agg(
                pl.mean("sign_pass_rate_primary").alias("sign_pass_rate_primary"),
                pl.mean("none_stability_rate").alias("none_stability_rate"),
                pl.mean("kendall_strength_alignment").alias("kendall_strength_alignment"),
                pl.mean("primary_activation_rate").alias("primary_activation_rate"),
                pl.mean("none_abs").alias("none_abs"),
                pl.mean("mild_strong_monotonicity_rate").alias("mild_strong_monotonicity_rate"),
            )
            .sort("sign_pass_rate_primary", descending=True)
        )
    else:
        construct_values = pl.DataFrame()
        construct_summary = pl.DataFrame()

    overall_monotonicity = (
        float(paired_df.select(pl.mean("mild_strong_monotonicity_rate")).row(0)[0])
        if not paired_df.is_empty()
        else float("nan")
    )

    per_construct_monotonicity_rows: list[dict[str, object]] = []
    if not paired_df.is_empty():
        for item in signals:
            sid = str(item.get("id"))
            primary_targets = list(item.get("primary_targets", []))
            if not primary_targets:
                continue
            row = paired_df.filter(pl.col("signal_id") == sid)
            if row.is_empty():
                continue
            per_var = row.row(0, named=True).get("per_var", {})
            if not isinstance(per_var, dict):
                continue
            for target in primary_targets:
                value = per_var.get(target)
                if value is None:
                    continue
                per_construct_monotonicity_rows.append({"target": target, "monotonicity": float(value)})

    per_construct_monotonicity = (
        pl.DataFrame(per_construct_monotonicity_rows)
        .group_by("target")
        .agg(pl.mean("monotonicity").alias("monotonicity"))
        .sort("monotonicity", descending=True)
        if per_construct_monotonicity_rows
        else pl.DataFrame()
    )

    top_signals: list[str] = []
    bottom_signals: list[str] = []
    if not signal_summary.is_empty():
        signal_sorted = signal_summary.sort("primary_activation_rate", descending=True)
        top_signals = [signal_label(s) for s in signal_sorted.get_column("signal_id").head(3)]
        bottom_signals = [signal_label(s) for s in signal_sorted.get_column("signal_id").tail(3)]

    none_stability_median = quantile_from_df(signal_summary, "none_stability_rate", 0.50)
    control_stability_median = quantile_from_df(signal_summary, "control_stability_rate", 0.50)
    sign_primary_median = quantile_from_df(signal_summary, "sign_pass_rate_primary", 0.50)
    sign_all_movers_median = quantile_from_df(signal_summary, "sign_pass_rate_all_non_none", 0.50)
    primary_activation_median = quantile_from_df(signal_summary, "primary_activation_rate", 0.50)
    primary_saturation_median = quantile_from_df(signal_summary, "primary_saturation_rate", 0.50)
    sign_primary_movers_median = quantile_from_df(signal_summary, "sign_pass_rate_primary_movers", 0.50)
    sign_all_movers_cond_median = quantile_from_df(signal_summary, "sign_pass_rate_all_non_none_movers", 0.50)

    def signal_rows_all_variants(strategy: str, signal_id: str) -> pl.DataFrame:
        frames = []
        for variant in ("mild", "strong"):
            df = signal_rows_frame(strategy, signal_id, variant)
            if not df.is_empty():
                frames.append(df)
        if not frames:
            return pl.DataFrame()
        return pl.concat(frames, how="diagonal")

    def moderation_values(signal_id: str, moderator: str, targets: list[str]) -> tuple[np.ndarray, np.ndarray]:
        df = signal_rows_all_variants(primary_strategy, signal_id)
        if df.is_empty() or moderator not in df.columns:
            return np.array([]), np.array([])
        sign_cols = [f"sign_ok_{t}" for t in targets if f"sign_ok_{t}" in df.columns]
        if not sign_cols:
            return np.array([]), np.array([])
        df = df.with_columns(
            pl.mean_horizontal([pl.col(c).cast(pl.Float64) for c in sign_cols]).alias("_sign_mean"),
            pl.col(moderator).cast(pl.Float64).alias("_mod"),
        ).drop_nulls(["_sign_mean", "_mod"])
        if df.is_empty():
            return np.array([]), np.array([])
        mod_vals = df.get_column("_mod").to_numpy()
        q = float(run_config.get("moderation_quantile", 0.25)) if isinstance(run_config, dict) else 0.25
        q = min(max(q, 0.0), 0.5)
        try:
            lo_thr = float(np.nanquantile(mod_vals, q))
            hi_thr = float(np.nanquantile(mod_vals, 1.0 - q))
        except Exception:
            return np.array([]), np.array([])
        low_mask = mod_vals <= lo_thr
        high_mask = mod_vals >= hi_thr
        vals = df.get_column("_sign_mean").to_numpy()
        return vals[high_mask], vals[low_mask]

    def signal_metric_values(signal_id: str, metric: str) -> np.ndarray:
        df = signal_rows_all_variants(primary_strategy, signal_id)
        if df.is_empty():
            return np.array([])
        if metric in df.columns:
            arr = df.get_column(metric).cast(pl.Float64).to_numpy()
            return arr[np.isfinite(arr)]
        spec = signal_spec(signal_id)
        ranking = spec.get("mutable_state_ranking", []) or []
        primary_targets = list(spec.get("primary_targets", []))
        none_vars = [r.get("var") for r in ranking if r.get("direction") == "none"]
        if metric == "primary_abs":
            exprs = _normalized_abs_delta_exprs(df, [str(v) for v in primary_targets])
        elif metric == "none_abs":
            exprs = _normalized_abs_delta_exprs(df, [str(v) for v in none_vars])
        else:
            return np.array([])
        if not exprs:
            return np.array([])
        arr = df.with_columns(pl.mean_horizontal(exprs).alias("_m")).get_column("_m").to_numpy()
        return arr[np.isfinite(arr)]

    if signal_summary.is_empty():
        shock_df = pl.DataFrame()
    else:
        shock_df = signal_summary.select(
            "signal_id",
            "sign_pass_rate_primary",
            "none_stability_rate",
            "control_stability_rate",
        )
        if not movement_df.is_empty():
            shock_df = shock_df.join(movement_df, on="signal_id", how="left")

    if not shock_df.is_empty():
        shock_df = (
            shock_df.with_columns((pl.lit(leakage_slope) * pl.col("primary_abs")).alias("_diag_bound"))
            .with_columns(
                pl.when(pl.col("primary_abs").is_null() | pl.col("none_abs").is_null())
                .then(pl.lit(None))
                .when(
                    (pl.col("primary_abs") >= primary_threshold)
                    & (pl.col("none_abs") <= pl.col("_diag_bound"))
                    & (pl.col("none_abs") <= spillover_cap)
                )
                .then(pl.lit("Safe targeted shock"))
                .when(
                    (pl.col("primary_abs") >= primary_threshold)
                    & ((pl.col("none_abs") > pl.col("_diag_bound")) | (pl.col("none_abs") > spillover_cap))
                )
                .then(pl.lit("Leaky shock"))
                .when((pl.col("primary_abs") < primary_threshold) & (pl.col("none_abs") >= drift_threshold))
                .then(pl.lit("Cross-talk / reinterpretation"))
                .otherwise(pl.lit("Stable low-signal"))
                .alias("lever_class")
            )
            .drop("_diag_bound")
        )

        safe_shocks = (
            shock_df.filter(
                (pl.col("none_stability_rate") >= safe_none_stability)
                & (pl.col("control_stability_rate") >= safe_control_stability)
                & (pl.col("sign_pass_rate_primary") >= safe_sign_primary)
                & (pl.col("lever_class").is_null() | (pl.col("lever_class") == "Safe targeted shock"))
            )
            .sort(["sign_pass_rate_primary", "none_stability_rate"], descending=True)
            .get_column("signal_id")
            .to_list()
        )
        lever_counts = (
            shock_df.filter(pl.col("lever_class").is_not_null())
            .group_by("lever_class")
            .agg(pl.len().alias("n"))
            .sort("n", descending=True)
        )
    else:
        safe_shocks = []
        lever_counts = pl.DataFrame()

    example_rows: list[list[str]] = []
    if not shock_df.is_empty():
        for cls in ["Safe targeted shock", "Leaky shock", "Cross-talk / reinterpretation", "Stable low-signal"]:
            sub = shock_df.filter(pl.col("lever_class") == cls)
            if sub.is_empty():
                continue
            pick = sub.sort("sign_pass_rate_primary", descending=True).row(0, named=True)
            sid = str(pick.get("signal_id"))
            sign_vals = signal_metric_values(sid, "sign_pass_rate_primary")
            none_vals = signal_metric_values(sid, "none_stability_rate")
            control_vals = signal_metric_values(sid, "control_stability_rate")
            prim_abs_vals = signal_metric_values(sid, "primary_abs")
            none_abs_vals = signal_metric_values(sid, "none_abs")
            seed_base = 2026_02_50 + len(example_rows) * 20
            example_rows.append(
                [
                    cls,
                    signal_label(sid),
                    fmt_ci(
                        float(np.nanmean(sign_vals))
                        if sign_vals.size
                        else finite_float(pick.get("sign_pass_rate_primary")),
                        *bootstrap_mean_ci(sign_vals, B=boot_B, seed=seed_base + 1),
                        digits=2,
                    )
                    if (sign_vals.size or pick.get("sign_pass_rate_primary") is not None)
                    else "—",
                    fmt_ci(
                        float(np.nanmean(none_vals))
                        if none_vals.size
                        else finite_float(pick.get("none_stability_rate")),
                        *bootstrap_mean_ci(none_vals, B=boot_B, seed=seed_base + 2),
                        digits=2,
                    )
                    if (none_vals.size or pick.get("none_stability_rate") is not None)
                    else "—",
                    fmt_ci(
                        float(np.nanmean(control_vals))
                        if control_vals.size
                        else finite_float(pick.get("control_stability_rate")),
                        *bootstrap_mean_ci(control_vals, B=boot_B, seed=seed_base + 3),
                        digits=2,
                    )
                    if (control_vals.size or pick.get("control_stability_rate") is not None)
                    else "—",
                    fmt_ci(
                        float(np.nanmean(prim_abs_vals))
                        if prim_abs_vals.size
                        else finite_float(pick.get("primary_abs")),
                        *bootstrap_mean_ci(prim_abs_vals, B=boot_B, seed=seed_base + 4),
                        digits=2,
                    )
                    if (prim_abs_vals.size or pick.get("primary_abs") is not None)
                    else "—",
                    fmt_ci(
                        float(np.nanmean(none_abs_vals)) if none_abs_vals.size else finite_float(pick.get("none_abs")),
                        *bootstrap_mean_ci(none_abs_vals, B=boot_B, seed=seed_base + 5),
                        digits=2,
                    )
                    if (none_abs_vals.size or pick.get("none_abs") is not None)
                    else "—",
                ]
            )

    if not construct_summary.is_empty():
        brittle_row = construct_summary.sort("sign_pass_rate_primary", descending=False).row(0, named=True)
        brittle_construct = label_construct(str(brittle_row.get("primary_targets")))
    else:
        brittle_construct = "—"

    if not signals_all.is_empty():
        frontier_df = (
            signals_all.group_by(["model", "strategy"])
            .agg(
                pl.median("sign_pass_rate_primary").alias("sign_pass_rate_primary"),
                pl.median("none_stability_rate").alias("none_stability_rate"),
                pl.median("control_stability_rate").alias("control_stability_rate"),
                pl.median("paired_mild_strong_monotonicity_rate").alias("paired_mild_strong_monotonicity_rate_mean"),
            )
            .with_columns(
                (
                    pl.col("none_stability_rate")
                    + pl.col("control_stability_rate")
                    + pl.col("paired_mild_strong_monotonicity_rate_mean")
                ).alias("gate_score")
            )
        )
    elif not summary_all_full.is_empty():
        frontier_df = summary_all_full.select(
            "model",
            "strategy",
            "sign_pass_rate_primary",
            "none_stability_rate",
            "control_stability_rate",
            "paired_mild_strong_monotonicity_rate_mean",
            "parsed_rate",
            "kendall_strength_alignment",
            "tier_monotonicity",
        ).with_columns(
            (
                pl.col("none_stability_rate")
                + pl.col("control_stability_rate")
                + pl.col("paired_mild_strong_monotonicity_rate_mean")
            ).alias("gate_score")
        )
    else:
        frontier_df = pl.DataFrame()

    construct_readiness = pl.DataFrame()
    if not construct_summary.is_empty():
        construct_readiness = construct_summary.rename({"primary_targets": "construct"})
        if not per_construct_monotonicity.is_empty():
            construct_readiness = construct_readiness.join(
                per_construct_monotonicity.rename({"target": "construct"}),
                on="construct",
                how="left",
            )

    def _primary_direction(spec: dict[str, object], var: str) -> str:
        ranking = spec.get("mutable_state_ranking", []) or []
        for entry in ranking:
            if not isinstance(entry, dict):
                continue
            if str(entry.get("var")) != var:
                continue
            direction = str(entry.get("direction") or "").strip().lower()
            if direction in {"up", "increase", "higher", "+"}:
                return "↑"
            if direction in {"down", "decrease", "lower", "-"}:
                return "↓"
            return "—"
        return "—"

    shock_library = pl.DataFrame()
    if not signal_summary.is_empty():
        lib = signal_summary.select(
            "signal_id",
            "primary_targets",
            "primary_activation_rate",
            "sign_pass_rate_primary",
            "sign_pass_rate_primary_movers",
            "n_primary_mover_rows",
            "none_stability_rate",
            "control_stability_rate",
        )
        if not paired_df.is_empty():
            lib = lib.join(paired_df.select("signal_id", "mild_strong_monotonicity_rate"), on="signal_id", how="left")
        if not movement_df.is_empty():
            lib = lib.join(movement_df, on="signal_id", how="left")
        if not shock_df.is_empty() and "lever_class" in shock_df.columns:
            lib = lib.join(shock_df.select("signal_id", "lever_class"), on="signal_id", how="left")

        lib_rows: list[dict[str, object]] = []
        for row in lib.iter_rows(named=True):
            sid = str(row.get("signal_id"))
            spec = signal_spec(sid)
            primary_targets = row.get("primary_targets") or []
            primary_var = str(primary_targets[0]) if isinstance(primary_targets, list) and primary_targets else ""
            direction = _primary_direction(spec, primary_var) if primary_var else "—"
            lever_class = str(row.get("lever_class") or "")
            if lever_class == "Safe targeted shock":
                role = "LLM kernel (targeted shock)"
            elif lever_class == "Leaky shock":
                role = "LLM kernel (allow spillover / clamp)"
            elif lever_class == "Cross-talk / reinterpretation":
                role = "Rule-based multi-construct / scenario"
            elif lever_class == "Stable low-signal":
                role = "Kernel with gain/repeat, or omit"
            else:
                role = "—"
            lib_rows.append(
                {
                    **row,
                    "signal": signal_label(sid),
                    "primary_construct": label_construct(primary_var) if primary_var else "—",
                    "direction": direction,
                    "abm_role": role,
                }
            )
        shock_library = pl.DataFrame(lib_rows) if lib_rows else pl.DataFrame()
        if not shock_library.is_empty():
            shock_library = shock_library.sort(
                ["lever_class", "primary_activation_rate", "sign_pass_rate_primary"],
                descending=[False, True, True],
            )

    return BeliefContext(
        chosen_model=chosen_model,
        chosen_strategy=chosen_strategy,
        canonical_root=canonical_root,
        primary_strategy=primary_strategy,
        primary_strategy_label=primary_strategy_label,
        summary_all=summary_all,
        summary_all_full=summary_all_full,
        signals_all=signals_all,
        targets=targets,
        signals=signals,
        mutable_state=mutable_state,
        run_config=run_config,
        signal_summary=signal_summary,
        movement_df=movement_df,
        paired_df=paired_df,
        moderation_df=moderation_df,
        construct_values=construct_values,
        construct_summary=construct_summary,
        per_construct_monotonicity=per_construct_monotonicity,
        overall_monotonicity=overall_monotonicity,
        top_signals=top_signals,
        bottom_signals=bottom_signals,
        none_stability_median=none_stability_median,
        control_stability_median=control_stability_median,
        sign_primary_median=sign_primary_median,
        sign_all_movers_median=sign_all_movers_median,
        primary_activation_median=primary_activation_median,
        primary_saturation_median=primary_saturation_median,
        sign_primary_movers_median=sign_primary_movers_median,
        sign_all_movers_cond_median=sign_all_movers_cond_median,
        drift_threshold=drift_threshold,
        primary_threshold=primary_threshold,
        leakage_slope=leakage_slope,
        spillover_cap=spillover_cap,
        boot_B=boot_B,
        safe_shocks=safe_shocks,
        lever_counts=lever_counts,
        example_rows=example_rows,
        brittle_construct=brittle_construct,
        frontier_df=frontier_df,
        construct_readiness=construct_readiness,
        shock_df=shock_df,
        shock_library=shock_library,
        signal_spec_fn=signal_spec,
        signal_rows_all_variants_fn=signal_rows_all_variants,
        moderation_values_fn=moderation_values,
        signal_metric_values_fn=signal_metric_values,
    )


def build_unperturbed_difficulty_rows(
    *,
    context: UnperturbedContext,
    label_outcome: Callable[[str], str],
    fmt_ci: Callable[..., str],
) -> list[list[str]]:
    rows: list[list[str]] = []
    for outcome in context.outcome_order:
        df = context.cell_metrics.filter(pl.col("outcome") == outcome)
        if df.is_empty() or outcome not in context.bootstrap_outcome_ci:
            continue
        df_g = df.filter(pl.col("gini").is_finite())
        df_s = df.filter(pl.col("skill_cal").is_finite())
        if df_g.is_empty():
            continue
        point_skill = float(df_s.select(pl.mean("skill_cal")).row(0)[0]) if not df_s.is_empty() else float("nan")
        point = float(df_g.select(pl.mean("gini")).row(0)[0])
        lo, hi = context.bootstrap_outcome_ci[outcome]["gini"]
        lo_s, hi_s = context.bootstrap_outcome_ci.get(outcome, {}).get("skill", (None, None))
        ref_c = context.ref_gini_by_outcome.get(outcome, float("nan"))
        ref_lo, ref_hi = context.ref_ci_by_outcome.get(outcome, (None, None))
        ref_s = context.ref_skill_by_outcome.get(outcome, float("nan"))
        ref_s_lo, ref_s_hi = context.ref_skill_ci_by_outcome.get(outcome, (None, None))
        rows.append(
            [
                label_outcome(outcome),
                fmt_ci(point, lo, hi, digits=2),
                fmt_ci(point_skill, lo_s, hi_s, digits=2),
                fmt_ci(ref_c, ref_lo, ref_hi, digits=2),
                fmt_ci(ref_s, ref_s_lo, ref_s_hi, digits=2),
            ]
        )
    return rows


def build_belief_unperturbed_difficulty_rows(
    *,
    canonical_root: Path,
    canonical_root_overrides_by_engine: Mapping[str, Path] | None,
    reference_modeling_dir: Path | None,
    model_order: Sequence[str],
    strategy_order: Sequence[str],
    label_engine: Callable[[str], str],
    fmt_ci: Callable[..., str],
    boot_b: int = 500,
    excluded_constructs: Collection[str] | None = None,
) -> list[list[str]]:
    """Build belief unperturbed latent-engine difficulty rows."""
    compact_candidates = (
        canonical_root / "unperturbed_engine_metrics.csv",
        canonical_root.parent / "unperturbed_engine_metrics.csv",
        canonical_root.parent.parent / "unperturbed_engine_metrics.csv",
    )
    compact_path = next((path for path in compact_candidates if path.exists()), None)
    if compact_path is not None:
        df = read_csv_fast(compact_path)
        if df.is_empty():
            return []
        excluded = {
            str(v) for v in (BELIEF_PLOT_EXCLUDED_CONSTRUCTS if excluded_constructs is None else excluded_constructs)
        }
        if "variant" in df.columns:
            preferred_variant_by_engine: dict[str, str] = {}
            if canonical_root_overrides_by_engine:
                for engine in canonical_root_overrides_by_engine:
                    preferred_variant_by_engine[str(engine)] = "two_constructs"
            df = (
                df.with_columns(
                    pl.col("engine")
                    .cast(pl.Utf8)
                    .map_elements(
                        lambda engine: preferred_variant_by_engine.get(str(engine), "altacct"),
                        return_dtype=pl.Utf8,
                    )
                    .alias("_preferred_variant")
                )
                .filter(pl.col("variant").cast(pl.Utf8) == pl.col("_preferred_variant"))
                .drop("_preferred_variant")
            )
        if model_order:
            df = df.filter(pl.col("model").is_in([str(m) for m in model_order]))
        if strategy_order:
            df = df.filter(pl.col("strategy").is_in([str(s) for s in strategy_order]))
        if df.is_empty():
            return []

        observed_engines = [str(v) for v in df.get_column("engine").unique().to_list() if str(v).strip()]
        engine_use = [
            str(v) for v in BELIEF_OUTCOME_ORDER if str(v) in set(observed_engines) and str(v) not in excluded
        ]
        engine_use.extend(sorted(v for v in observed_engines if v not in set(engine_use) and v not in excluded))

        def _point_ci(values: np.ndarray, *, seed: int) -> tuple[float, float | None, float | None]:
            vals = np.asarray(values, dtype=np.float64)
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                return float("nan"), None, None
            point = float(np.mean(vals))
            lo, hi = bootstrap_mean_ci(vals, B=boot_b, seed=seed)
            return point, lo, hi

        out_rows: list[list[str]] = []
        for idx, engine in enumerate(engine_use):
            sub = df.filter(pl.col("engine") == engine)
            if sub.is_empty():
                continue
            gini_point, gini_lo, gini_hi = _point_ci(
                sub.select(pl.col("gini").cast(pl.Float64, strict=False)).to_series().to_numpy(),
                seed=2026_03_01 + (idx * 31) + 2,
            )
            skill_point, skill_lo, skill_hi = _point_ci(
                sub.select(pl.col("skill").cast(pl.Float64, strict=False)).to_series().to_numpy(),
                seed=2026_03_01 + (idx * 31) + 3,
            )
            out_rows.append(
                [
                    label_engine(engine),
                    "—",
                    fmt_ci(gini_point, gini_lo, gini_hi, digits=2),
                    fmt_ci(skill_point, skill_lo, skill_hi, digits=2),
                    "—",
                    "—",
                    "—",
                ]
            )
        return out_rows

    if not canonical_root.exists():
        return []

    def _std_cosine(x: np.ndarray, y: np.ndarray) -> float:
        mask = np.isfinite(x) & np.isfinite(y)
        if int(mask.sum()) < 3:
            return float("nan")
        xv = x[mask]
        yv = y[mask]
        sx = float(np.std(xv, ddof=1))
        sy = float(np.std(yv, ddof=1))
        if not np.isfinite(sx) or not np.isfinite(sy) or sx <= 0.0 or sy <= 0.0:
            return float("nan")
        xz = (xv - float(np.mean(xv))) / sx
        yz = (yv - float(np.mean(yv))) / sy
        nx = float(np.linalg.norm(xz))
        ny = float(np.linalg.norm(yz))
        if nx <= 0.0 or ny <= 0.0:
            return float("nan")
        return float(np.dot(xz, yz) / (nx * ny))

    excluded = {
        str(v) for v in (BELIEF_PLOT_EXCLUDED_CONSTRUCTS if excluded_constructs is None else excluded_constructs)
    }
    root_by_engine: dict[str, Path] = {}
    if canonical_root_overrides_by_engine:
        for engine, root in canonical_root_overrides_by_engine.items():
            engine_key = str(engine).strip()
            if not engine_key:
                continue
            root_path = Path(root)
            if root_path.exists():
                root_by_engine[engine_key] = root_path

    source_roots: list[Path] = [canonical_root]
    source_roots.extend([p for p in dict.fromkeys(root_by_engine.values()) if p != canonical_root])

    rows: list[dict[str, object]] = []
    for source_root in source_roots:
        source_key = str(source_root.resolve())
        for model_dir in sorted(p for p in source_root.iterdir() if p.is_dir() and not p.name.startswith("_")):
            model = str(model_dir.name)
            for strategy_dir in sorted(p for p in model_dir.iterdir() if p.is_dir() and not p.name.startswith("_")):
                strategy = str(strategy_dir.name)
                metrics_path = strategy_dir / "metrics.parquet"
                if not metrics_path.exists() or metrics_path.stat().st_size == 0:
                    continue
                try:
                    df = pl.read_parquet(metrics_path)
                except Exception:
                    continue
                if df.is_empty():
                    continue
                columns = set(df.columns)
                engines = [
                    str(v)
                    for v in BELIEF_OUTCOME_ORDER
                    if f"true_{v}" in columns and f"pred_{v}" in columns and str(v) not in excluded
                ]
                extras = sorted(
                    {
                        str(c)[5:]
                        for c in df.columns
                        if str(c).startswith("true_")
                        and f"pred_{str(c)[5:]}" in columns
                        and str(c)[5:] not in set(engines)
                        and str(c)[5:] not in excluded
                    }
                )
                engines.extend(extras)
                for engine in engines:
                    x = df.select(pl.col(f"true_{engine}").cast(pl.Float64, strict=False)).to_series().to_numpy()
                    y = df.select(pl.col(f"pred_{engine}").cast(pl.Float64, strict=False)).to_series().to_numpy()
                    rows.append(
                        {
                            "model": model,
                            "strategy": strategy,
                            "engine": engine,
                            "cosine_std": _std_cosine(x, y),
                            "gini": gini_np(x, y),
                            "skill": mae_skill_cal_np(x, y),
                            "source_root": source_key,
                        }
                    )

    if not rows:
        return []

    df = pl.DataFrame(rows)
    default_source = str(canonical_root.resolve())
    if root_by_engine:
        preferred_source_by_engine = {k: str(v.resolve()) for k, v in root_by_engine.items()}
        df = (
            df.with_columns(
                pl.col("engine")
                .cast(pl.Utf8)
                .map_elements(
                    lambda e: preferred_source_by_engine.get(str(e), default_source),
                    return_dtype=pl.Utf8,
                )
                .alias("_preferred_source")
            )
            .filter(pl.col("source_root") == pl.col("_preferred_source"))
            .drop("_preferred_source")
        )
    else:
        df = df.filter(pl.col("source_root") == default_source)

    if model_order:
        df = df.filter(pl.col("model").is_in([str(m) for m in model_order]))
    if strategy_order:
        df = df.filter(pl.col("strategy").is_in([str(s) for s in strategy_order]))
    if df.is_empty():
        return []

    observed_engines = [str(v) for v in df.get_column("engine").unique().to_list() if str(v).strip()]
    engine_use = [str(v) for v in BELIEF_OUTCOME_ORDER if str(v) in set(observed_engines) and str(v) not in excluded]
    engine_use.extend(sorted(v for v in observed_engines if v not in set(engine_use)))

    def _point_ci(values: np.ndarray, *, seed: int) -> tuple[float, float | None, float | None]:
        vals = np.asarray(values, dtype=np.float64)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            return float("nan"), None, None
        point = float(np.mean(vals))
        lo, hi = bootstrap_mean_ci(vals, B=boot_b, seed=seed)
        return point, lo, hi

    def _bootstrap_metric_ci(
        *,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        metric_fn: Callable[[np.ndarray, np.ndarray], float],
        B: int,
        seed: int,
    ) -> tuple[float | None, float | None]:
        yt = np.asarray(y_true, dtype=np.float64)
        yp = np.asarray(y_pred, dtype=np.float64)
        mask = np.isfinite(yt) & np.isfinite(yp)
        yt = yt[mask]
        yp = yp[mask]
        if yt.size < 2:
            return None, None
        rng = np.random.default_rng(int(seed))
        sims = np.full(int(B), np.nan, dtype=np.float64)
        for b in range(int(B)):
            idx_boot = rng.integers(0, yt.size, size=int(yt.size))
            sims[b] = float(metric_fn(yt[idx_boot], yp[idx_boot]))
        return quantile_ci(sims)

    def _load_reference_rows(
        *,
        sample_path: Path,
    ) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, str]]:
        by_id: dict[str, dict[str, object]] = {}
        ref_map_meta: dict[str, str] = {}
        try:
            with sample_path.open("r", encoding="utf-8") as f:
                for line in f:
                    raw = line.strip()
                    if not raw:
                        continue
                    try:
                        row = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(row, dict):
                        continue
                    meta = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
                    base_id = str(meta.get("base_id") or row.get("id") or "").strip()
                    if not base_id:
                        continue
                    payload = by_id.setdefault(base_id, {"attrs": {}, "state": {}, "country": None})

                    attrs = row.get("attributes") if isinstance(row.get("attributes"), dict) else {}
                    if attrs:
                        payload_attrs = payload.get("attrs")
                        if isinstance(payload_attrs, dict):
                            for key, value in attrs.items():
                                key_s = str(key).strip()
                                if key_s:
                                    payload_attrs[key_s] = value

                    baseline = meta.get("baseline_state") if isinstance(meta.get("baseline_state"), dict) else {}
                    if baseline:
                        payload_state = payload.get("state")
                        if isinstance(payload_state, dict):
                            for key, value in baseline.items():
                                key_s = str(key).strip()
                                if key_s:
                                    payload_state[key_s] = finite_float(value)

                    if payload.get("country") is None:
                        country = meta.get("country")
                        if isinstance(country, str) and country.strip():
                            payload["country"] = country

                    ref_models = (
                        meta.get("ref_predictor_models") if isinstance(meta.get("ref_predictor_models"), dict) else {}
                    )
                    if ref_models:
                        for engine_name, model_name in ref_models.items():
                            engine_key = str(engine_name).strip()
                            model_key = str(model_name).strip()
                            if engine_key and model_key:
                                ref_map_meta[engine_key] = model_key
        except OSError:
            return pd.DataFrame(), {}, {}

        attr_rows: list[dict[str, object]] = []
        state_rows: list[dict[str, float]] = []
        for payload in by_id.values():
            attrs = payload.get("attrs") if isinstance(payload.get("attrs"), dict) else {}
            state = payload.get("state") if isinstance(payload.get("state"), dict) else {}
            if not attrs or not state:
                continue
            row_attrs = dict(attrs)
            if "country" not in row_attrs:
                country = payload.get("country")
                if isinstance(country, str) and country.strip():
                    row_attrs["country"] = country
            attr_rows.append(row_attrs)
            state_rows.append({str(k): finite_float(v) for k, v in state.items()})

        if not attr_rows:
            return pd.DataFrame(), {}, ref_map_meta

        attr_df = pd.DataFrame(attr_rows)
        engines_observed = {str(engine) for state in state_rows for engine in state.keys() if str(engine).strip()}
        truth_by_engine: dict[str, np.ndarray] = {}
        for engine in sorted(engines_observed):
            truth_by_engine[engine] = np.asarray(
                [finite_float(state.get(engine, float("nan"))) for state in state_rows],
                dtype=np.float64,
            )
        return attr_df, truth_by_engine, ref_map_meta

    def _load_column_types(modeling_dir: Path) -> dict[str, str]:
        specs_candidates = [
            modeling_dir.parent.parent / "specs" / "column_types.tsv",
            modeling_dir.parent.parent / "specs" / "column_types.csv",
        ]
        spec_path = next((p for p in specs_candidates if p.exists()), None)
        if spec_path is None:
            return {}
        sep = "\t" if spec_path.suffix.lower() == ".tsv" else ","
        try:
            df_types = pd.read_csv(spec_path, sep=sep, usecols=["column", "type"])
        except (OSError, ValueError):
            return {}
        out: dict[str, str] = {}
        for _, row in df_types.iterrows():
            col = str(row.get("column") or "").strip()
            typ = str(row.get("type") or "").strip().lower()
            if col and typ:
                out[col] = typ
        return out

    def _resolve_reference_data_path(modeling_dir: Path) -> Path | None:
        root = modeling_dir.parent.parent.parent
        candidates = [
            root / "thesis" / "artifacts" / "empirical" / "decision_function" / "clean_processed_survey.csv",
            root / "data" / "1_CROSSCOUNTRY_translated.csv",
        ]
        return next((p for p in candidates if p.exists()), None)

    def _standardize_reference_predictors(attr_df: pd.DataFrame, *, modeling_dir: Path) -> pd.DataFrame:
        if attr_df.empty:
            return attr_df
        column_types = _load_column_types(modeling_dir)
        numeric_types = {"continuous", "ordinal"}
        candidate_cols = [
            str(c)
            for c in attr_df.columns
            if str(c) in column_types and str(column_types.get(str(c), "")).lower() in numeric_types
        ]
        if not candidate_cols:
            return attr_df

        ref_path = _resolve_reference_data_path(modeling_dir)
        ref_df: pd.DataFrame | None = None
        if ref_path is not None:
            try:
                ref_df = pd.read_csv(ref_path, usecols=lambda c: str(c) in set(candidate_cols))
            except (OSError, ValueError):
                ref_df = None

        out = attr_df.copy()
        for col in candidate_cols:
            vals = pd.to_numeric(out[col], errors="coerce")
            source_vals = vals
            if ref_df is not None and col in ref_df.columns:
                source_vals = pd.to_numeric(ref_df[col], errors="coerce")
            mean = float(source_vals.mean(skipna=True))
            std = float(source_vals.std(skipna=True, ddof=0))
            if np.isfinite(mean) and np.isfinite(std) and std > 0.0:
                out[col] = (vals - mean) / std
            else:
                out[col] = vals
        return out

    c_dummy_re = re.compile(r"^C\((?P<var>[^,\)]+)(?:,.*)?\)\[T\.(?P<level>.*)\]$")
    simple_dummy_re = re.compile(r"^(?P<var>[^\[]+)\[T\.(?P<level>.*)\]$")
    threshold_term_re = re.compile(r"^\s*[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?\s*/\s*[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?\s*$")

    def _safe_float(value: object) -> float | None:
        if value is None:
            return None
        try:
            out = float(value)
        except (TypeError, ValueError):
            return None
        if not np.isfinite(out):
            return None
        return out

    def _as_string_level(value: object) -> str:
        if value is None:
            return "__MISSING__"
        if isinstance(value, float) and not np.isfinite(value):
            return "__MISSING__"
        if isinstance(value, bool):
            return "True" if value else "False"
        return str(value).strip()

    def _match_level(value: object, level: str) -> bool:
        lvl = str(level).strip()
        if not lvl:
            return False
        if lvl in {"True", "False"}:
            target = lvl == "True"
            if isinstance(value, bool):
                return bool(value) == target
            val_f = _safe_float(value)
            if val_f is not None:
                if abs(val_f - 1.0) <= 1e-9:
                    return target
                if abs(val_f) <= 1e-9:
                    return not target
            sval = _as_string_level(value).lower()
            if sval in {"true", "false"}:
                return (sval == "true") == target
            if sval in {"1", "1.0"}:
                return target
            if sval in {"0", "0.0"}:
                return not target
            return False
        val_f = _safe_float(value)
        lvl_f = _safe_float(lvl)
        if val_f is not None and lvl_f is not None:
            return abs(val_f - lvl_f) <= 1e-9
        return _as_string_level(value) == lvl

    def _strip_q_wrapper(var_expr: str) -> str:
        expr = str(var_expr).strip()
        if expr.startswith("Q(") and expr.endswith(")"):
            inner = expr[2:-1].strip()
            if (inner.startswith('"') and inner.endswith('"')) or (inner.startswith("'") and inner.endswith("'")):
                return inner[1:-1]
            return inner
        return expr

    def _term_series(df_in: pd.DataFrame, term: str) -> np.ndarray:
        if term in df_in.columns:
            return pd.to_numeric(df_in[term], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
        if term.endswith("_missing"):
            base = term[: -len("_missing")]
            if base in df_in.columns:
                return pd.isna(df_in[base]).astype(float).to_numpy(dtype=np.float64)
            return np.zeros(len(df_in), dtype=np.float64)

        m_c = c_dummy_re.match(term)
        if m_c:
            var = _strip_q_wrapper(m_c.group("var"))
            level = m_c.group("level")
            if var not in df_in.columns:
                return np.zeros(len(df_in), dtype=np.float64)
            return np.asarray(
                [1.0 if _match_level(v, level) else 0.0 for v in df_in[var].tolist()],
                dtype=np.float64,
            )

        m_s = simple_dummy_re.match(term)
        if m_s:
            var = _strip_q_wrapper(m_s.group("var"))
            level = m_s.group("level")
            if var not in df_in.columns:
                return np.zeros(len(df_in), dtype=np.float64)
            return np.asarray(
                [1.0 if _match_level(v, level) else 0.0 for v in df_in[var].tolist()],
                dtype=np.float64,
            )
        return np.zeros(len(df_in), dtype=np.float64)

    def _find_coefficients_path(modeling_dir: Path, model_name: str) -> Path | None:
        for group_dir in sorted(p for p in modeling_dir.iterdir() if p.is_dir()):
            candidate = group_dir / model_name / "coefficients.csv"
            if candidate.is_file():
                return candidate
        return None

    def _parse_thresholds(threshold_terms: list[tuple[str, float]]) -> np.ndarray | None:
        parsed: list[tuple[float, float]] = []
        for term, estimate in threshold_terms:
            left = term.split("/", 1)[0].strip()
            left_f = _safe_float(left)
            if left_f is None:
                continue
            parsed.append((left_f, float(estimate)))
        if not parsed:
            return None
        parsed.sort(key=lambda x: x[0])
        return np.asarray([x[1] for x in parsed], dtype=np.float64)

    def _load_ref_model_local(modeling_dir: Path, model_name: str) -> dict[str, object] | None:
        coeff_path = _find_coefficients_path(modeling_dir=modeling_dir, model_name=model_name)
        if coeff_path is None:
            return None
        try:
            coef_df = pd.read_csv(coeff_path)
        except (OSError, ValueError):
            return None
        if "term" not in coef_df.columns or "estimate" not in coef_df.columns:
            return None

        intercept = 0.0
        terms: list[str] = []
        coefs: list[float] = []
        threshold_terms: list[tuple[str, float]] = []
        for _, row in coef_df.iterrows():
            term = str(row.get("term", "")).strip()
            est = _safe_float(row.get("estimate"))
            if not term or est is None:
                continue
            if term == "Intercept":
                intercept = float(est)
                continue
            if threshold_term_re.match(term):
                threshold_terms.append((term, float(est)))
                continue
            terms.append(term)
            coefs.append(float(est))
        return {
            "intercept": float(intercept),
            "terms": tuple(terms),
            "coefs": np.asarray(coefs, dtype=np.float64),
            "thresholds": _parse_thresholds(threshold_terms),
        }

    def _predict_ref_model_local(model: dict[str, object], df_in: pd.DataFrame) -> np.ndarray:
        terms = model.get("terms") if isinstance(model.get("terms"), tuple) else ()
        intercept = float(model.get("intercept", 0.0))
        coefs = np.asarray(model.get("coefs"), dtype=np.float64)
        if not terms:
            eta = np.full(len(df_in), intercept, dtype=np.float64)
        else:
            X = np.column_stack([_term_series(df_in, t) for t in terms]).astype(np.float64)
            eta = intercept + X @ coefs

        thresholds = model.get("thresholds")
        if not isinstance(thresholds, np.ndarray) or thresholds.size == 0:
            return eta

        n = eta.shape[0]
        n_cat = int(thresholds.size + 1)
        logits = thresholds[np.newaxis, :] - eta[:, np.newaxis]
        cum = np.where(logits >= 0.0, 1.0 / (1.0 + np.exp(-logits)), np.exp(logits) / (1.0 + np.exp(logits)))
        probs = np.empty((n, n_cat), dtype=np.float64)
        probs[:, 0] = cum[:, 0]
        for j in range(1, n_cat - 1):
            probs[:, j] = cum[:, j] - cum[:, j - 1]
        probs[:, n_cat - 1] = 1.0 - cum[:, n_cat - 2]
        probs = np.clip(probs, 0.0, 1.0)
        row_sum = probs.sum(axis=1, keepdims=True)
        row_sum = np.where(row_sum > 0.0, row_sum, 1.0)
        probs /= row_sum
        categories = np.arange(1.0, float(n_cat) + 1.0, dtype=np.float64)
        return probs @ categories

    def _align_linear_prediction_scale(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Rescale likely z-score linear outputs to the observed engine scale."""
        yt = np.asarray(y_true, dtype=np.float64)
        yp = np.asarray(y_pred, dtype=np.float64)
        mask = np.isfinite(yt) & np.isfinite(yp)
        if int(mask.sum()) < 2:
            return yp

        yt_use = yt[mask]
        yp_use = yp[mask]
        yt_mean = float(np.mean(yt_use))
        yt_std = float(np.std(yt_use, ddof=0))
        yp_mean = float(np.mean(yp_use))
        if not np.isfinite(yt_std) or yt_std <= 0.0:
            return yp

        # Mixed reference bundles include linear models on z-scored outcomes.
        # Detect this when predictions are centered near zero while the target
        # is clearly on a non-zero bounded scale (e.g. Likert 1-5 / 1-7).
        likely_z_output = abs(yp_mean) <= 0.75 and abs(yt_mean) >= 0.75
        if not likely_z_output:
            return yp

        yp_rescaled = yp_use * yt_std + yt_mean
        mae_raw = float(np.mean(np.abs(yp_use - yt_use)))
        mae_rescaled = float(np.mean(np.abs(yp_rescaled - yt_use)))
        if not np.isfinite(mae_raw) or not np.isfinite(mae_rescaled) or mae_rescaled >= mae_raw:
            return yp

        out = yp.copy()
        out[mask] = yp_rescaled
        return out

    def _collect_reference_metrics(
        *,
        source_root: Path,
        engines: Sequence[str],
    ) -> dict[str, dict[str, float | None]]:
        out: dict[str, dict[str, float | None]] = {}
        if reference_modeling_dir is None:
            return out
        modeling_dir = Path(reference_modeling_dir)
        if not source_root.exists() or not modeling_dir.exists():
            return out

        sample_paths = sorted(
            p / "samples.jsonl" for p in source_root.iterdir() if p.is_dir() and (p / "samples.jsonl").exists()
        )
        if not sample_paths:
            return out
        attr_df, truth_by_engine, ref_map_meta = _load_reference_rows(sample_path=sample_paths[0])
        if attr_df.empty or not truth_by_engine:
            return out
        attr_df_std = _standardize_reference_predictors(attr_df, modeling_dir=modeling_dir)

        ref_model_map = default_ref_state_models()
        ref_model_map.update(ref_map_meta)
        model_cache: dict[str, object | None] = {}

        for idx, engine in enumerate(engines):
            engine_key = str(engine)
            y_true = truth_by_engine.get(engine_key)
            if y_true is None:
                continue

            model_name = str(ref_model_map.get(engine_key) or "").strip()
            if not model_name:
                continue
            if model_name not in model_cache:
                model_cache[model_name] = _load_ref_model_local(
                    modeling_dir=modeling_dir,
                    model_name=model_name,
                )
            ref_model = model_cache.get(model_name)
            if ref_model is None:
                continue
            if not isinstance(ref_model, dict):
                continue
            y_pred = np.asarray(_predict_ref_model_local(ref_model, attr_df_std), dtype=np.float64)
            if not isinstance(ref_model.get("thresholds"), np.ndarray):
                y_pred = _align_linear_prediction_scale(y_true=y_true, y_pred=y_pred)

            mask = np.isfinite(y_true) & np.isfinite(y_pred)
            if int(mask.sum()) < 2:
                continue
            yt = y_true[mask]
            yp = y_pred[mask]
            gini_point = float(gini_np(yt, yp))
            skill_point = float(mae_skill_cal_np(yt, yp))
            cosine_point = float(_std_cosine(yt, yp))
            cosine_lo, cosine_hi = _bootstrap_metric_ci(
                y_true=yt,
                y_pred=yp,
                metric_fn=_std_cosine,
                B=boot_b,
                seed=2026_03_10 + (idx * 41),
            )
            gini_lo, gini_hi = _bootstrap_metric_ci(
                y_true=yt,
                y_pred=yp,
                metric_fn=gini_np,
                B=boot_b,
                seed=2026_03_11 + (idx * 41),
            )
            skill_lo, skill_hi = _bootstrap_metric_ci(
                y_true=yt,
                y_pred=yp,
                metric_fn=mae_skill_cal_np,
                B=boot_b,
                seed=2026_03_12 + (idx * 41),
            )
            out[engine_key] = {
                "cosine": cosine_point,
                "cosine_lo": cosine_lo,
                "cosine_hi": cosine_hi,
                "gini": gini_point,
                "gini_lo": gini_lo,
                "gini_hi": gini_hi,
                "skill": skill_point,
                "skill_lo": skill_lo,
                "skill_hi": skill_hi,
            }
        return out

    ref_metrics_by_engine: dict[str, dict[str, float | None]] = {}
    if reference_modeling_dir is not None and engine_use:
        source_by_engine = {str(engine): root_by_engine.get(str(engine), canonical_root) for engine in engine_use}
        engines_by_source: dict[str, list[str]] = {}
        for engine, src_root in source_by_engine.items():
            src_key = str(Path(src_root).resolve())
            engines_by_source.setdefault(src_key, []).append(engine)
        for src_key, src_engines in engines_by_source.items():
            ref_metrics_by_engine.update(
                _collect_reference_metrics(
                    source_root=Path(src_key),
                    engines=src_engines,
                )
            )

    out_rows: list[list[str]] = []
    for idx, engine in enumerate(engine_use):
        sub = df.filter(pl.col("engine") == engine)
        if sub.is_empty():
            continue
        cos_point, cos_lo, cos_hi = _point_ci(
            sub.get_column("cosine_std").to_numpy(),
            seed=2026_03_01 + (idx * 31) + 1,
        )
        gini_point, gini_lo, gini_hi = _point_ci(
            sub.get_column("gini").to_numpy(),
            seed=2026_03_01 + (idx * 31) + 2,
        )
        skill_point, skill_lo, skill_hi = _point_ci(
            sub.get_column("skill").to_numpy(),
            seed=2026_03_01 + (idx * 31) + 3,
        )
        ref_metrics = ref_metrics_by_engine.get(engine, {})
        ref_cosine = finite_float(ref_metrics.get("cosine", float("nan")))
        ref_cosine_lo = ref_metrics.get("cosine_lo")
        ref_cosine_hi = ref_metrics.get("cosine_hi")
        ref_gini = finite_float(ref_metrics.get("gini", float("nan")))
        ref_gini_lo = ref_metrics.get("gini_lo")
        ref_gini_hi = ref_metrics.get("gini_hi")
        ref_skill = finite_float(ref_metrics.get("skill", float("nan")))
        ref_skill_lo = ref_metrics.get("skill_lo")
        ref_skill_hi = ref_metrics.get("skill_hi")
        if not any(np.isfinite(v) for v in (cos_point, gini_point, skill_point)):
            continue
        out_rows.append(
            [
                label_engine(engine),
                fmt_ci(cos_point, cos_lo, cos_hi, digits=2),
                fmt_ci(gini_point, gini_lo, gini_hi, digits=2),
                fmt_ci(skill_point, skill_lo, skill_hi, digits=2),
                fmt_ci(ref_cosine, ref_cosine_lo, ref_cosine_hi, digits=2),
                fmt_ci(ref_gini, ref_gini_lo, ref_gini_hi, digits=2),
                fmt_ci(ref_skill, ref_skill_lo, ref_skill_hi, digits=2),
            ]
        )
    return out_rows


def build_belief_unperturbed_state_vector_cosine_rows(
    *,
    canonical_root: Path,
    model_order: Sequence[str],
    strategy_order: Sequence[str],
    label_model: Callable[[str], str],
    label_strategy: Callable[[str], str],
    fmt_ci: Callable[..., str],
    boot_b: int = 500,
    excluded_constructs: Collection[str] | None = None,
) -> list[list[str]]:
    """Build model × strategy rows for true standardized state-vector cosine."""
    if not canonical_root.exists():
        return []

    def _cosine_std(x: np.ndarray, y: np.ndarray) -> float:
        mask = np.isfinite(x) & np.isfinite(y)
        if int(mask.sum()) < 2:
            return float("nan")
        xv = x[mask]
        yv = y[mask]
        sx = float(np.std(xv, ddof=1))
        sy = float(np.std(yv, ddof=1))
        if not np.isfinite(sx) or not np.isfinite(sy) or sx <= 0.0 or sy <= 0.0:
            return float("nan")
        xz = (xv - float(np.mean(xv))) / sx
        yz = (yv - float(np.mean(yv))) / sy
        nx = float(np.linalg.norm(xz))
        ny = float(np.linalg.norm(yz))
        if nx <= 0.0 or ny <= 0.0:
            return float("nan")
        return float(np.dot(xz, yz) / (nx * ny))

    excluded = {
        str(v) for v in (BELIEF_PLOT_EXCLUDED_CONSTRUCTS if excluded_constructs is None else excluded_constructs)
    }

    rows: list[dict[str, object]] = []
    for model_dir in sorted(p for p in canonical_root.iterdir() if p.is_dir() and not p.name.startswith("_")):
        model = str(model_dir.name)
        for strategy_dir in sorted(p for p in model_dir.iterdir() if p.is_dir() and not p.name.startswith("_")):
            strategy = str(strategy_dir.name)
            metrics_path = strategy_dir / "metrics.parquet"
            if not metrics_path.exists() or metrics_path.stat().st_size == 0:
                continue
            try:
                df = pl.read_parquet(metrics_path)
            except Exception:
                continue
            if df.is_empty():
                continue
            columns = set(df.columns)
            engines = [
                str(v)
                for v in BELIEF_OUTCOME_ORDER
                if f"true_{v}" in columns and f"pred_{v}" in columns and str(v) not in excluded
            ]
            extras = sorted(
                {
                    str(c)[5:]
                    for c in df.columns
                    if str(c).startswith("true_")
                    and f"pred_{str(c)[5:]}" in columns
                    and str(c)[5:] not in set(engines)
                    and str(c)[5:] not in excluded
                }
            )
            engines.extend(extras)
            if len(engines) < 2:
                continue

            true_mat = np.column_stack(
                [
                    df.select(pl.col(f"true_{engine}").cast(pl.Float64, strict=False)).to_series().to_numpy()
                    for engine in engines
                ]
            )
            pred_mat = np.column_stack(
                [
                    df.select(pl.col(f"pred_{engine}").cast(pl.Float64, strict=False)).to_series().to_numpy()
                    for engine in engines
                ]
            )
            row_scores = np.asarray(
                [_cosine_std(pred_mat[i, :], true_mat[i, :]) for i in range(true_mat.shape[0])],
                dtype=np.float64,
            )
            row_scores = row_scores[np.isfinite(row_scores)]
            if row_scores.size == 0:
                continue
            point = float(np.mean(row_scores))
            lo, hi = bootstrap_mean_ci(row_scores, B=boot_b, seed=2026_03_21 + len(rows))
            rows.append(
                {
                    "model": model,
                    "strategy": strategy,
                    "point": point,
                    "lo": lo,
                    "hi": hi,
                }
            )

    if not rows:
        return []

    table_df = pl.DataFrame(rows)
    if model_order:
        table_df = table_df.filter(pl.col("model").is_in([str(m) for m in model_order]))
    if strategy_order:
        table_df = table_df.filter(pl.col("strategy").is_in([str(s) for s in strategy_order]))
    if table_df.is_empty():
        return []

    observed_models = set(table_df.get_column("model").to_list())
    observed_strategies = set(table_df.get_column("strategy").to_list())
    model_use = [str(m) for m in model_order if str(m) in observed_models]
    model_use.extend(sorted(m for m in observed_models if m not in set(model_use)))
    strategy_use = [str(s) for s in strategy_order if str(s) in observed_strategies]
    strategy_use.extend(sorted(s for s in observed_strategies if s not in set(strategy_use)))

    out_rows: list[list[str]] = []
    for model in model_use:
        for strategy in strategy_use:
            sub = table_df.filter((pl.col("model") == model) & (pl.col("strategy") == strategy))
            if sub.is_empty():
                continue
            rec = sub.row(0, named=True)
            out_rows.append(
                [
                    label_model(model),
                    label_strategy(strategy),
                    fmt_ci(finite_float(rec.get("point", float("nan"))), rec.get("lo"), rec.get("hi"), digits=2),
                ]
            )
    return out_rows


def belief_signal_headline_metrics(
    *,
    signal_summary: pl.DataFrame,
    movement_df: pl.DataFrame,
    drift_threshold: float,
) -> tuple[float, float, float]:
    no_move_median = float("nan")
    conditional_sign = float("nan")
    drift_share = float("nan")
    if not signal_summary.is_empty():
        no_move_median = 1.0 - quantile_from_df(signal_summary, "primary_activation_rate", 0.50)
        conditional_sign = quantile_from_df(signal_summary, "sign_pass_rate_primary_movers", 0.50)
    if not movement_df.is_empty():
        drift_share = float((movement_df.get_column("none_abs") >= drift_threshold).mean())
    return no_move_median, conditional_sign, drift_share


def build_belief_engine_scorecard_rows(
    *,
    construct_summary: pl.DataFrame,
    construct_values: pl.DataFrame,
    label_construct: Callable[[str], str],
    fmt_ci: Callable[..., str],
    boot_b: int,
) -> list[list[str]]:
    if construct_summary.is_empty():
        return []

    def _construct_ci(construct: str, col: str, seed: int) -> tuple[float | None, float | None]:
        if construct_values.is_empty() or col not in construct_values.columns:
            return (None, None)
        vals = (
            construct_values.filter(pl.col("primary_targets") == construct)
            .select(pl.col(col).cast(pl.Float64))
            .to_series()
            .to_numpy()
        )
        return bootstrap_mean_ci(vals, B=boot_b, seed=seed)

    rows: list[list[str]] = []
    for row in construct_summary.iter_rows(named=True):
        target = str(row["primary_targets"])
        rows.append(
            [
                label_construct(target),
                fmt_ci(
                    finite_float(row["sign_pass_rate_primary"]),
                    *_construct_ci(target, "sign_pass_rate_primary", seed=2026_02_01),
                    digits=3,
                ),
                fmt_ci(
                    finite_float(row["kendall_strength_alignment"]),
                    *_construct_ci(target, "kendall_strength_alignment", seed=2026_02_02),
                    digits=3,
                ),
                fmt_ci(
                    finite_float(row["none_stability_rate"]),
                    *_construct_ci(target, "none_stability_rate", seed=2026_02_03),
                    digits=3,
                ),
                (
                    fmt_ci(
                        finite_float(row["primary_activation_rate"]),
                        *_construct_ci(target, "primary_activation_rate", seed=2026_02_04),
                        digits=3,
                    )
                    if row.get("primary_activation_rate") is not None
                    else "—"
                ),
                (
                    fmt_ci(
                        finite_float(row["none_abs"]),
                        *_construct_ci(target, "none_abs", seed=2026_02_05),
                        digits=3,
                    )
                    if row.get("none_abs") is not None
                    else "—"
                ),
                (
                    fmt_ci(
                        finite_float(row["mild_strong_monotonicity_rate"]),
                        *_construct_ci(target, "mild_strong_monotonicity_rate", seed=2026_02_06),
                        digits=3,
                    )
                    if row.get("mild_strong_monotonicity_rate") is not None
                    else "—"
                ),
            ]
        )
    return rows


def build_belief_moderation_rows(
    *,
    moderation_df: pl.DataFrame,
    signal_label: Callable[[str], str],
    label_construct: Callable[[str], str],
    moderation_values: Callable[[str, str, list[str]], tuple[np.ndarray, np.ndarray]],
    fmt_ci: Callable[..., str],
    boot_b: int,
    n_rows: int = 4,
) -> list[list[str]]:
    if moderation_df.is_empty():
        return []

    def _label_list(raw: str) -> str:
        parts = [p.strip() for p in str(raw).split(",") if p.strip()]
        return ", ".join(label_construct(p) for p in parts) if parts else "—"

    def _expected_label(raw: object) -> str:
        if raw is None:
            return "—"
        text = str(raw)
        compact = text.lower().replace(" ", "")
        if "high" in compact and "low" in compact:
            if ">=" in compact or ">" in compact:
                return "Amplify"
            if "<=" in compact or "<" in compact:
                return "Dampen"
        return text

    def _targets_from_raw(raw: object) -> list[str]:
        parts = [p.strip() for p in str(raw).split(",") if p.strip()]
        return parts if parts else []

    subset = (
        moderation_df.with_columns(
            pl.col("pass_rate_high").cast(pl.Float64, strict=False).alias("pass_rate_high"),
            pl.col("pass_rate_low").cast(pl.Float64, strict=False).alias("pass_rate_low"),
            pl.col("moderation_gap").cast(pl.Float64, strict=False).alias("moderation_gap"),
        )
        .with_columns(pl.col("moderation_gap").abs().alias("gap_abs"))
        .filter(
            pl.col("pass_rate_high").is_finite()
            & pl.col("pass_rate_low").is_finite()
            & pl.col("moderation_gap").is_finite()
        )
        .sort("gap_abs", descending=True)
        .head(n_rows)
    )
    if subset.is_empty():
        subset = moderation_df.with_columns(
            pl.col("pass_rate_high").cast(pl.Float64, strict=False).alias("pass_rate_high"),
            pl.col("pass_rate_low").cast(pl.Float64, strict=False).alias("pass_rate_low"),
            pl.col("moderation_gap").cast(pl.Float64, strict=False).alias("moderation_gap"),
        ).head(n_rows)

    rows: list[list[str]] = []
    for i, row in enumerate(subset.iter_rows(named=True)):
        hi_vals, lo_vals = moderation_values(
            str(row["signal_id"]),
            str(row["moderator"]),
            _targets_from_raw(row["targets"]),
        )
        rows.append(
            [
                signal_label(str(row["signal_id"])),
                label_construct(str(row["moderator"])),
                _label_list(str(row["targets"])),
                _expected_label(row["expected"]),
                (
                    fmt_ci(
                        float(np.nanmean(hi_vals)) if hi_vals.size else finite_float(row["pass_rate_high"]),
                        *bootstrap_mean_ci(hi_vals, B=boot_b, seed=2026_02_10 + i),
                        digits=3,
                    )
                    if (hi_vals.size or row["pass_rate_high"] is not None)
                    else "—"
                ),
                (
                    fmt_ci(
                        float(np.nanmean(lo_vals)) if lo_vals.size else finite_float(row["pass_rate_low"]),
                        *bootstrap_mean_ci(lo_vals, B=boot_b, seed=2026_02_20 + i),
                        digits=3,
                    )
                    if (lo_vals.size or row["pass_rate_low"] is not None)
                    else "—"
                ),
                (
                    fmt_ci(
                        (
                            float(np.nanmean(hi_vals) - np.nanmean(lo_vals))
                            if (hi_vals.size and lo_vals.size)
                            else finite_float(row["moderation_gap"])
                        ),
                        *bootstrap_gap_ci(hi_vals, lo_vals, B=boot_b, seed=2026_02_30 + i),
                        digits=3,
                    )
                    if (hi_vals.size and lo_vals.size) or row["moderation_gap"] is not None
                    else "—"
                ),
                "Yes" if row.get("moderation_pass") is True else ("No" if row.get("moderation_pass") is False else "—"),
            ]
        )
    return rows


def build_belief_vignette_rows(
    *,
    shock_df: pl.DataFrame,
    primary_strategy: str,
    signal_rows_all_variants: Callable[[str, str], pl.DataFrame],
    mutable_state: Sequence[str],
    signal_label: Callable[[str], str],
    label_construct: Callable[[str], str],
    fmt_ci: Callable[..., str],
    boot_b: int,
    top_n: int = 10,
) -> tuple[list[list[str]], str | None, str | None, str | None]:
    if shock_df.is_empty() or shock_df.filter(pl.col("lever_class") == "Cross-talk / reinterpretation").is_empty():
        return [], "No vignette available (missing lever-class assignments or row-level exports).", None, None

    safe_pick = shock_df.filter(pl.col("lever_class") == "Safe targeted shock").sort(
        ["sign_pass_rate_primary", "none_stability_rate"],
        descending=True,
    )
    cross_pick = shock_df.filter(pl.col("lever_class") == "Cross-talk / reinterpretation").sort(
        ["none_abs", "sign_pass_rate_primary"],
        descending=[True, False],
    )
    if safe_pick.is_empty() or cross_pick.is_empty():
        return [], "No vignette available (safe/cross-talk examples not found).", None, None

    safe_id = str(safe_pick.row(0, named=True)["signal_id"])
    cross_id = str(cross_pick.row(0, named=True)["signal_id"])
    safe_label = signal_label(safe_id)
    cross_label = signal_label(cross_id)

    safe_vec = belief_mean_update_vector(
        rows_df=signal_rows_all_variants(primary_strategy, safe_id),
        mutable_state=mutable_state,
        bootstrap_mean_ci_fn=bootstrap_mean_ci,
        boot_b=boot_b,
        seed_base=2026_02_40,
    )
    cross_vec = belief_mean_update_vector(
        rows_df=signal_rows_all_variants(primary_strategy, cross_id),
        mutable_state=mutable_state,
        bootstrap_mean_ci_fn=bootstrap_mean_ci,
        boot_b=boot_b,
        seed_base=2026_02_40,
    )
    if safe_vec.is_empty() or cross_vec.is_empty():
        return [], "No vignette available (could not compute update vectors from rows exports).", None, None

    safe_vec = safe_vec.rename(
        {
            "mean_delta": "safe_mean_delta",
            "mean_delta_lo": "safe_mean_delta_lo",
            "mean_delta_hi": "safe_mean_delta_hi",
        }
    )
    cross_vec = cross_vec.rename(
        {
            "mean_delta": "cross_mean_delta",
            "mean_delta_lo": "cross_mean_delta_lo",
            "mean_delta_hi": "cross_mean_delta_hi",
        }
    )
    both = safe_vec.join(cross_vec, on="var", how="full").fill_null(0.0)
    top = (
        both.with_columns((pl.col("safe_mean_delta").abs() + pl.col("cross_mean_delta").abs()).alias("total_abs"))
        .sort("total_abs", descending=True)
        .head(top_n)
    )

    rows = [
        [
            label_construct(str(r["var"])),
            fmt_ci(
                float(r["safe_mean_delta"]),
                r.get("safe_mean_delta_lo"),
                r.get("safe_mean_delta_hi"),
                digits=3,
            ),
            fmt_ci(
                float(r["cross_mean_delta"]),
                r.get("cross_mean_delta_lo"),
                r.get("cross_mean_delta_hi"),
                digits=3,
            ),
        ]
        for r in top.iter_rows(named=True)
    ]
    return rows, None, safe_label, cross_label


def render_primary_block_sensitivity(
    *,
    block_sensitivity_df: pl.DataFrame,
    outcome_order: Sequence[str],
    label_outcome_short: Callable[[str], str],
    label_block: Callable[[str], str],
    default_block_color_order: Sequence[str],
    block_order_for_outcome: Callable[[str], list[str]],
    primary_model: str | None,
    primary_strategy: str,
    panel_nrows: int | None = None,
    panel_ncols: int | None = None,
) -> tuple[str | None, dict[str, float]]:
    diagnostics = {
        "diag_vax_stakes_ref": float("nan"),
        "diag_vax_stakes_llm": float("nan"),
        "diag_clip_x_cap": float("nan"),
        "diag_clip_raw_ref": float("nan"),
        "diag_clip_llm": float("nan"),
        "diag_npi_trust_ref": float("nan"),
        "diag_npi_trust_llm_min": float("nan"),
        "diag_npi_trust_llm_max": float("nan"),
    }
    message, extra = render_block_sensitivity_alignment(
        df=block_sensitivity_df,
        outcome_order=outcome_order,
        label_outcome_short=label_outcome_short,
        label_block=label_block,
        default_block_color_order=default_block_color_order,
        block_order_for_outcome=block_order_for_outcome,
        primary_model=primary_model,
        primary_strategy=primary_strategy,
        panel_nrows=panel_nrows,
        panel_ncols=panel_ncols,
    )
    for key, value in extra.items():
        if key not in diagnostics:
            continue
        try:
            value_f = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(value_f):
            diagnostics[key] = value_f
    return message, diagnostics


__all__ = [
    "as_float",
    "BeliefContext",
    "UnperturbedContext",
    "build_belief_engine_scorecard_rows",
    "build_belief_unperturbed_difficulty_rows",
    "build_belief_unperturbed_state_vector_cosine_rows",
    "build_belief_moderation_rows",
    "build_belief_vignette_rows",
    "build_unperturbed_difficulty_rows",
    "belief_signal_headline_metrics",
    "PerturbedContext",
    "belief_mean_update_vector",
    "best_cell_tuple",
    "bootstrap_gap_ci",
    "bootstrap_mean_ci",
    "canonical_levers_for",
    "cell_point",
    "discover_keep_outcomes",
    "discover_models",
    "discover_outcomes",
    "discover_pert_models",
    "extract_row_id",
    "finite_values",
    "fisher_ci",
    "finite_float",
    "find_ref_model_for_outcome",
    "format_ci",
    "gini_np",
    "heatmap_norm",
    "infer_model_slug",
    "iter_ref_prediction_paths",
    "label_block",
    "label_strategy_pert",
    "load_cell_arrays",
    "load_lever_config",
    "load_model_plan_entries",
    "load_preds",
    "make_signal_labeler",
    "mae_np",
    "mae_skill_cal_np",
    "normalize_model_slug",
    "prediction_path",
    "quantile_ci",
    "quantile_from_df",
    "rankdata_average_ties",
    "prepare_belief_context",
    "prepare_unperturbed_context",
    "prepare_perturbed_context",
    "render_primary_block_sensitivity",
    "render_belief_construct_readiness",
    "render_belief_frontier",
    "render_belief_reference_signal_magnitude",
    "render_belief_scope_control",
    "render_belief_perturbed_signal_engine_heatmap",
    "render_belief_signal_block_cosine_heatmap",
    "render_belief_signal_block_panel_heatmaps",
    "render_belief_signal_block_sensitivity_alignment",
    "render_belief_signal_activation",
    "render_belief_unperturbed_engine_panel_heatmaps",
    "render_belief_unperturbed_cosine_heatmap",
    "render_belief_vector_importance_alignment_by_signal_block",
    "render_belief_vector_importance_alignment_cosine",
    "render_block_metric_heatmap",
    "render_block_importance_alignment_gini",
    "render_block_sensitivity_alignment",
    "render_perturbed_pareto_by_model",
    "render_unperturbed_gini_heatmap",
    "shared_metric_range",
    "smae_cal_np",
    "spearman_np",
    "text_color_for_value",
]
