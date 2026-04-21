"""Plot/table helpers for chapter 5 (ABM counterfactual results)."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from IPython.display import Markdown
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from beam_abm.abm.scenario_defs import get_scenario, get_scenario_label
from utils.chapter_io import md_table
from utils.plot_style import apply_thesis_matplotlib_style

apply_thesis_matplotlib_style()

ABM_ARTIFACT_ROOT = Path("../thesis/artifacts/abm")
ABM_RUN_CONTEXT_ROOT = ABM_ARTIFACT_ROOT / "run_context"

RUN_NAME_RE = re.compile(r"^(?P<scenario>.+?)_rep(?P<rep>\d+)_seed(?P<seed>\d+)$")
CHAPTER_BASELINE_SCENARIO = "baseline_importation"
CHAPTER_NO_COMM_SCENARIO = "baseline_no_communication"

LEADERBOARD_SCENARIOS: tuple[str, ...] = (
    "access_intervention",
    "norm_campaign",
    "targeted_norm_seed",
    "misinfo_shock",
    "trust_scandal_repair",
    "outbreak_global",
    "outbreak_local",
    "outbreak_misinfo_combined",
)

AUC_LEADERBOARD_SCENARIOS: tuple[str, ...] = (
    "trust_repair_institutional",
    "trust_scandal_institutional",
    "trust_repair_legitimacy",
    "trust_scandal_legitimacy",
    "misinfo_trust_erosion_only",
    "norm_shift_campaign",
    "outbreak_salience_global",
    "outbreak_global_incidence_strong",
    "access_intervention",
    "access_barrier_shock",
)

VERIFICATION_SCENARIOS: tuple[str, ...] = (
    "baseline",
    "baseline_no_communication",
    "baseline_importation",
    "baseline_importation_no_communication",
    "social_only",
    "baseline_no_observed_norms",
    "baseline_no_peer_pull",
    "baseline_importation_no_observed_norms",
    "baseline_importation_no_peer_pull",
    "outbreak_local_incidence",
)

COUNTRY_SCENARIOS: tuple[str, ...] = (
    "access_intervention",
    "outbreak_global",
    "trust_scandal_repair",
)

SENSITIVITY_PREFERRED_REGIMES: tuple[str, ...] = ("shock", "perturbation")

FLAGSHIP_MORRIS_FOCUS: tuple[tuple[str, str], ...] = (
    ("norm_campaign", "mask_when_pressure_high_mean"),
    ("outbreak_global_incidence_strong", "mask_when_pressure_high_mean"),
    ("outbreak_salience_global", "mask_when_pressure_high_mean"),
    ("trust_repair_institutional", "vax_willingness_T12_mean"),
    ("trust_repair_legitimacy", "vax_willingness_T12_mean"),
    ("access_intervention", "vax_willingness_T12_mean"),
    ("trust_scandal_legitimacy", "mask_when_pressure_high_mean"),
)

FLAGSHIP_MORRIS_SCENARIO_ALIASES: dict[str, str] = {
    "trust_scandal_repair": "trust_scandal_legitimacy",
    "norm_shift_campaign": "norm_campaign",
}

FLAGSHIP_MORRIS_ROW_LABELS: dict[tuple[str, str], str] = {
    ("norm_campaign", "mask_when_pressure_high_mean"): "Norm shift\nNPI index",
    ("outbreak_global_incidence_strong", "mask_when_pressure_high_mean"): "Outbreak wave\nNPI index",
    ("outbreak_salience_global", "mask_when_pressure_high_mean"): "Outbreak salience\nNPI index",
    ("trust_repair_institutional", "vax_willingness_T12_mean"): "Institutional trust repair\nVaccination willingness",
    ("trust_repair_legitimacy", "vax_willingness_T12_mean"): "Legitimacy repair\nVaccination willingness",
    ("access_intervention", "vax_willingness_T12_mean"): "Access facilitation\nVaccination willingness",
    ("trust_scandal_legitimacy", "mask_when_pressure_high_mean"): "Legitimacy scandal\nNPI index",
}

FLAGSHIP_MORRIS_FAMILY_ORDER: tuple[str, ...] = (
    "Persistence",
    "Social diffusion",
    "Incidence loop",
    "Stochasticity",
)

FLAGSHIP_MORRIS_FAMILY_LABELS: dict[str, str] = {
    "Persistence": "Persistence",
    "Social diffusion": "Social\ndiffusion",
    "Incidence loop": "Incidence\nloop",
    "Stochasticity": "Stochasticity",
}

PAIRWISE_ROBUSTNESS_CLAIMS: tuple[dict[str, object], ...] = (
    {
        "label": "Norm shift > institutional trust repair on vaccination final effect",
        "left_scenario": "norm_campaign",
        "right_scenario": "trust_repair_institutional",
        "output": "vax_willingness_T12_mean",
        "value_column": "delta_final",
        "comparison": "gt",
    },
    {
        "label": "Norm shift > institutional trust repair on NPI final effect",
        "left_scenario": "norm_campaign",
        "right_scenario": "trust_repair_institutional",
        "output": "mask_when_pressure_high_mean",
        "value_column": "delta_final",
        "comparison": "gt",
    },
    {
        "label": "Norm shift > legitimacy repair on vaccination AUC",
        "left_scenario": "norm_campaign",
        "right_scenario": "trust_repair_legitimacy",
        "output": "vax_willingness_T12_mean",
        "value_column": "delta_auc",
        "comparison": "gt",
    },
    {
        "label": "Norm shift > legitimacy repair on descriptive-norm final effect",
        "left_scenario": "norm_campaign",
        "right_scenario": "trust_repair_legitimacy",
        "output": "social_norms_descriptive_vax_mean",
        "value_column": "delta_final",
        "comparison": "gt",
    },
    {
        "label": "Legitimacy repair > legitimacy scandal on NPI AUC",
        "left_scenario": "trust_repair_legitimacy",
        "right_scenario": "trust_scandal_legitimacy",
        "output": "mask_when_pressure_high_mean",
        "value_column": "delta_auc",
        "comparison": "gt",
    },
)

_VAX_OUTCOME_CANDIDATES: tuple[str, ...] = ("vax_willingness_T12", "vax_willingness_Y")
_BEHAVIORAL_PROJECTION_CACHE: dict[str, pl.DataFrame] = {}


@dataclass(slots=True)
class ABMChapterContext:
    """Container for chapter-level ABM outputs.

    Attributes
    ----------
    runs_root
        Resolved run directory used as source.
    run_index
        One row per run with scenario and metadata fields.
    series
        Tick-level merged trajectories for incidence, outcomes, and mediators.
    grouped_outcomes
        Group-level outcomes from ``outcomes_by_group`` for heterogeneity.
    grouped_mediators
        Group-level mediators from ``mediators_by_group`` for targeted/local
        harness diagnostics.
    """

    runs_root: Path
    run_index: pl.DataFrame
    series: pl.DataFrame
    grouped_outcomes: pl.DataFrame
    grouped_mediators: pl.DataFrame


def label_scenario(scenario: str) -> str:
    """Return human-readable scenario label."""

    key = str(scenario).strip()
    return get_scenario_label(key)


def resolve_runs_root(explicit_root: str | Path | None = None) -> Path:
    """Resolve ABM run-context root used in chapter plots."""

    if explicit_root is not None:
        root = Path(explicit_root)
        if not root.exists():
            raise FileNotFoundError(f"ABM runs root does not exist: {root!s}")
        return root

    # Prefer tracked thesis artifact contract.
    candidates = (
        ABM_RUN_CONTEXT_ROOT,
        Path("thesis/artifacts/abm/run_context"),
    )
    for root in candidates:
        if root.exists():
            return root

    raise FileNotFoundError("No ABM run-context artifacts found under thesis/artifacts/abm/run_context.")


def _parse_run_name(name: str) -> tuple[str, int, int] | None:
    match = RUN_NAME_RE.match(name)
    if match is None:
        return None
    return str(match.group("scenario")), int(match.group("rep")), int(match.group("seed"))


def _safe_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _expr_or_nan(frame: pl.DataFrame, column: str, alias: str) -> pl.Expr:
    """Return column expression if present, otherwise NaN literal."""

    if column in frame.columns:
        return pl.col(column).alias(alias)
    return pl.lit(float("nan")).alias(alias)


def _expr_or_nan_by_name(columns: set[str], column: str, alias: str) -> pl.Expr:
    """Return column expression if present in schema names, otherwise NaN."""

    if column in columns:
        return pl.col(column).alias(alias)
    return pl.lit(float("nan")).alias(alias)


def _first_existing(columns: set[str], candidates: tuple[str, ...]) -> str | None:
    """Return first candidate column name found in a schema-name set."""

    for name in candidates:
        if name in columns:
            return name
    return None


def _mean_ci_from_series(values: pl.Series) -> tuple[float, float, float, int]:
    arr = np.asarray(values.to_list(), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    n = int(arr.size)
    if n == 0:
        return float("nan"), float("nan"), float("nan"), 0
    mean = float(arr.mean())
    if n == 1:
        return mean, mean, mean, n
    sd = float(arr.std(ddof=1))
    half = 1.96 * sd / np.sqrt(n)
    return mean, mean - half, mean + half, n


def load_abm_context(
    runs_root: str | Path | None = None,
    scenarios: tuple[str, ...] | list[str] | None = None,
    effect_regime: str | None = None,
) -> ABMChapterContext:
    """Load run index and time-series outputs required by chapter 5.

    Parameters
    ----------
    runs_root
        Optional explicit root for ABM runs.
    scenarios
        Optional scenario allow-list. When provided, only matching runs are
        loaded to reduce I/O and startup time.
    effect_regime
        Optional effect-regime filter (for example ``\"shock\"`` or
        ``\"perturbation\"``). Baseline scenarios with missing regime metadata
        are retained to preserve matched-baseline deltas.
    """

    root = resolve_runs_root(runs_root)
    scenario_filter = {str(s) for s in scenarios} if scenarios is not None else None
    requested_regime = str(effect_regime).strip().lower() if effect_regime is not None else None
    baseline_scenarios = {
        "baseline",
        "baseline_importation",
        "baseline_no_communication",
        "baseline_importation_no_communication",
        "baseline_no_observed_norms",
        "baseline_importation_no_observed_norms",
        "baseline_no_peer_pull",
        "baseline_importation_no_peer_pull",
        "social_only",
    }
    # Always include verification scenarios to keep chapter diagnostics available.
    if scenario_filter is not None:
        scenario_filter.update(VERIFICATION_SCENARIOS)

    run_index_path = root / "run_index.parquet"
    series_path = root / "series.parquet"
    grouped_outcomes_path = root / "grouped_outcomes.parquet"
    grouped_mediators_path = root / "grouped_mediators.parquet"
    if run_index_path.exists() and series_path.exists():
        run_index = pl.read_parquet(run_index_path)
        series = pl.read_parquet(series_path)
        grouped_outcomes = pl.read_parquet(grouped_outcomes_path) if grouped_outcomes_path.exists() else pl.DataFrame()
        grouped_mediators = (
            pl.read_parquet(grouped_mediators_path) if grouped_mediators_path.exists() else pl.DataFrame()
        )

        if scenario_filter is not None and "scenario" in run_index.columns:
            run_index = run_index.filter(pl.col("scenario").is_in(sorted(scenario_filter)))
        if requested_regime is not None and "effect_regime" in run_index.columns:
            run_index = run_index.filter(
                (pl.col("effect_regime").cast(pl.Utf8).str.to_lowercase() == requested_regime)
                | (
                    pl.col("scenario").is_in(sorted(baseline_scenarios))
                    & (pl.col("effect_regime").cast(pl.Utf8).str.to_lowercase() == "unknown")
                )
            )
        if run_index.is_empty():
            raise RuntimeError(f"No ABM run-context rows found in {root!s}")

        valid_run_ids = run_index.get_column("run_id").cast(pl.Utf8).to_list()
        series = series.filter(pl.col("run_id").cast(pl.Utf8).is_in(valid_run_ids))
        if not grouped_outcomes.is_empty() and "run_id" in grouped_outcomes.columns:
            grouped_outcomes = grouped_outcomes.filter(pl.col("run_id").cast(pl.Utf8).is_in(valid_run_ids))
        if not grouped_mediators.is_empty() and "run_id" in grouped_mediators.columns:
            grouped_mediators = grouped_mediators.filter(pl.col("run_id").cast(pl.Utf8).is_in(valid_run_ids))

        return ABMChapterContext(
            runs_root=root,
            run_index=run_index.sort(["scenario", "rep", "seed"]),
            series=series.sort(["scenario", "seed", "rep", "tick"]),
            grouped_outcomes=grouped_outcomes,
            grouped_mediators=grouped_mediators,
        )

    baseline_requested_available: set[str] = set()
    if requested_regime is not None:
        for run_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            parsed = _parse_run_name(run_dir.name)
            if parsed is None:
                continue
            metadata_path = run_dir / "metadata.json"
            if not metadata_path.exists():
                continue
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
            scenario_cfg = payload.get("scenario_config", {})
            scenario_name = str(scenario_cfg.get("name") or scenario_cfg.get("scenario_key") or parsed[0])
            run_effect_regime = (
                str(
                    payload.get("effect_regime")
                    or payload.get("run_metadata", {}).get("effect_regime")
                    or payload.get("simulation_config", {}).get("effect_regime")
                    or "unknown"
                )
                .strip()
                .lower()
            )
            if scenario_name in baseline_scenarios and run_effect_regime == requested_regime:
                baseline_requested_available.add(scenario_name)

    run_rows: list[dict[str, object]] = []
    incidence_paths: list[str] = []
    outcomes_paths: list[str] = []
    mediators_paths: list[str] = []
    grouped_paths: list[str] = []
    mediators_grouped_paths: list[str] = []

    for run_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        parsed = _parse_run_name(run_dir.name)
        if parsed is None:
            continue

        scenario_from_dir, rep, seed = parsed
        metadata_path = run_dir / "metadata.json"
        incidence_path = run_dir / "trajectories" / "incidence.parquet"
        outcomes_path = run_dir / "trajectories" / "outcomes.parquet"
        mediators_path = run_dir / "trajectories" / "mediators.parquet"
        grouped_path = run_dir / "trajectories" / "outcomes_by_group.parquet"

        if not (
            metadata_path.exists() and incidence_path.exists() and outcomes_path.exists() and mediators_path.exists()
        ):
            continue

        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        scenario_cfg = payload.get("scenario_config", {})
        scenario = str(scenario_cfg.get("name") or scenario_cfg.get("scenario_key") or scenario_from_dir)
        run_effect_regime = (
            str(
                payload.get("effect_regime")
                or payload.get("run_metadata", {}).get("effect_regime")
                or payload.get("simulation_config", {}).get("effect_regime")
                or "unknown"
            )
            .strip()
            .lower()
        )
        incidence_mode = str(scenario_cfg.get("incidence_mode", "none"))
        communication_enabled = bool(scenario_cfg.get("communication_enabled", True))
        burn_in_ticks = int(scenario_cfg.get("burn_in_ticks", 0))
        if scenario_filter is not None and scenario not in scenario_filter:
            continue
        if requested_regime is not None:
            is_baseline_unknown = run_effect_regime == "unknown" and scenario in baseline_scenarios
            if is_baseline_unknown and scenario in baseline_requested_available:
                continue
            if run_effect_regime != requested_regime and not is_baseline_unknown:
                continue

        run_id = run_dir.name
        run_rows.append(
            {
                "run_id": run_id,
                "run_dir": str(run_dir),
                "scenario": scenario,
                "rep": rep,
                "seed": seed,
                "effect_regime": run_effect_regime,
                "incidence_mode": incidence_mode,
                "communication_enabled": communication_enabled,
                "burn_in_ticks": burn_in_ticks,
            }
        )

        incidence_paths.append(str(incidence_path))
        outcomes_paths.append(str(outcomes_path))
        mediators_paths.append(str(mediators_path))

        if grouped_path.exists():
            grouped_paths.append(str(grouped_path))

        mediators_grouped_path = run_dir / "trajectories" / "mediators_by_group.parquet"
        if mediators_grouped_path.exists():
            mediators_grouped_paths.append(str(mediators_grouped_path))

    if not run_rows:
        raise RuntimeError(f"No valid ABM runs found in {root!s}")

    run_index = pl.DataFrame(run_rows).sort(["scenario", "rep", "seed"])
    run_meta = run_index.select(
        "run_id",
        "scenario",
        "seed",
        "rep",
        "effect_regime",
        "incidence_mode",
        "communication_enabled",
        "burn_in_ticks",
    ).lazy()

    outcomes_schema = pl.scan_parquet(outcomes_paths[0]).collect_schema()
    outcomes_cols = set(outcomes_schema.names())
    mediators_schema = pl.scan_parquet(mediators_paths[0]).collect_schema()
    mediators_cols = set(mediators_schema.names())

    vax_mean_col = _first_existing(outcomes_cols, ("vax_willingness_T12_mean", "vax_willingness_Y_mean"))
    vax_std_col = _first_existing(outcomes_cols, ("vax_willingness_T12_std", "vax_willingness_Y_std"))
    danger_mean_col = _first_existing(
        mediators_cols, ("covid_perceived_danger_T12_mean", "covid_perceived_danger_Y_mean")
    )
    danger_std_col = _first_existing(mediators_cols, ("covid_perceived_danger_T12_std", "covid_perceived_danger_Y_std"))

    incidence = pl.scan_parquet(
        incidence_paths,
        include_file_paths="_path",
        missing_columns="insert",
        extra_columns="ignore",
    ).select(
        pl.col("_path").str.extract(r"/([^/]+)/trajectories/incidence\.parquet$").alias("run_id"),
        pl.col("tick"),
        pl.col("mean_incidence").alias("incidence_mean"),
    )
    outcomes = pl.scan_parquet(
        outcomes_paths,
        include_file_paths="_path",
        missing_columns="insert",
        extra_columns="ignore",
    ).select(
        pl.col("_path").str.extract(r"/([^/]+)/trajectories/outcomes\.parquet$").alias("run_id"),
        pl.col("tick"),
        _expr_or_nan_by_name(outcomes_cols, vax_mean_col or "", "vax_willingness"),
        _expr_or_nan_by_name(outcomes_cols, vax_std_col or "", "vax_willingness_std"),
        pl.col("mask_when_pressure_high_mean").alias("mask_pressure"),
        pl.col("mask_when_symptomatic_crowded_mean").alias("mask_symptomatic"),
        _expr_or_nan_by_name(outcomes_cols, "mask_when_symptomatic_crowded_std", "mask_symptomatic_std"),
        pl.col("stay_home_when_symptomatic_mean").alias("stay_home_symptomatic"),
    )
    mediators = pl.scan_parquet(
        mediators_paths,
        include_file_paths="_path",
        missing_columns="insert",
        extra_columns="ignore",
    ).select(
        pl.col("_path").str.extract(r"/([^/]+)/trajectories/mediators\.parquet$").alias("run_id"),
        pl.col("tick"),
        _expr_or_nan_by_name(mediators_cols, danger_mean_col or "", "danger"),
        _expr_or_nan_by_name(mediators_cols, danger_std_col or "", "danger_std"),
        pl.col("disease_fear_avg_mean").alias("fear"),
        _expr_or_nan_by_name(mediators_cols, "disease_fear_avg_std", "fear_std"),
        pl.col("fivec_low_constraints_mean").alias("low_constraints"),
        pl.col("institutional_trust_avg_mean").alias("institutional_trust"),
        _expr_or_nan_by_name(mediators_cols, "institutional_trust_avg_std", "institutional_trust_std"),
        _expr_or_nan_by_name(mediators_cols, "vaccine_risk_avg_std", "vaccine_risk_std"),
        pl.col("social_norms_descriptive_vax_mean").alias("descriptive_norm"),
        _expr_or_nan_by_name(mediators_cols, "social_norms_descriptive_vax_std", "descriptive_norm_std"),
    )

    series = (
        incidence.join(outcomes, on=["run_id", "tick"], how="inner")
        .join(mediators, on=["run_id", "tick"], how="inner")
        .join(run_meta, on="run_id", how="inner")
        .select(
            "run_id",
            "scenario",
            "seed",
            "rep",
            "effect_regime",
            "incidence_mode",
            "communication_enabled",
            "burn_in_ticks",
            "tick",
            "incidence_mean",
            "vax_willingness",
            "vax_willingness_std",
            "mask_pressure",
            "mask_symptomatic",
            "mask_symptomatic_std",
            "stay_home_symptomatic",
            "danger",
            "danger_std",
            "fear",
            "fear_std",
            "low_constraints",
            "institutional_trust",
            "institutional_trust_std",
            "vaccine_risk_std",
            "descriptive_norm",
            "descriptive_norm_std",
        )
        .with_columns(_npi_index_expr())
        .collect()
        .sort(["scenario", "seed", "rep", "tick"])
    )

    grouped_outcomes = pl.DataFrame()
    if grouped_paths:
        grouped_outcomes = (
            pl.scan_parquet(
                grouped_paths,
                include_file_paths="_path",
                missing_columns="insert",
                extra_columns="ignore",
            )
            .select(
                pl.col("_path").str.extract(r"/([^/]+)/trajectories/outcomes_by_group\.parquet$").alias("run_id"),
                "tick",
                "group_type",
                "group_value",
                "variable",
                "mean",
            )
            .filter(pl.col("variable") == "vax_willingness_T12")
            .join(run_meta, on="run_id", how="inner")
            .select(
                "run_id",
                "scenario",
                "seed",
                "rep",
                "effect_regime",
                "incidence_mode",
                "communication_enabled",
                "burn_in_ticks",
                "tick",
                "group_type",
                "group_value",
                "variable",
                "mean",
            )
            .collect()
        )

    grouped_mediators = pl.DataFrame()
    if mediators_grouped_paths:
        grouped_mediators = (
            pl.scan_parquet(
                mediators_grouped_paths,
                include_file_paths="_path",
                missing_columns="insert",
                extra_columns="ignore",
            )
            .select(
                pl.col("_path").str.extract(r"/([^/]+)/trajectories/mediators_by_group\.parquet$").alias("run_id"),
                "tick",
                "group_type",
                "group_value",
                "variable",
                "mean",
                "std",
            )
            .join(run_meta, on="run_id", how="inner")
            .select(
                "run_id",
                "scenario",
                "seed",
                "rep",
                "effect_regime",
                "incidence_mode",
                "communication_enabled",
                "burn_in_ticks",
                "tick",
                "group_type",
                "group_value",
                "variable",
                "mean",
                "std",
            )
            .collect()
        )

    # Filter to consistent runs (same tick count - use the most common tick count)
    run_ticks = series.group_by("run_id").agg(pl.col("tick").max().alias("max_tick"))
    tick_mode = (
        run_ticks.group_by("max_tick")
        .agg(pl.len().alias("count"))
        .sort("count", descending=True)
        .head(1)
        .get_column("max_tick")
        .item()
    )
    consistent_runs = run_ticks.filter(pl.col("max_tick") == tick_mode).get_column("run_id").to_list()

    series = series.filter(pl.col("run_id").is_in(consistent_runs))
    run_index = run_index.filter(pl.col("run_id").is_in(consistent_runs))
    if not grouped_outcomes.is_empty():
        grouped_outcomes = grouped_outcomes.filter(pl.col("run_id").is_in(consistent_runs))
    if not grouped_mediators.is_empty():
        grouped_mediators = grouped_mediators.filter(pl.col("run_id").is_in(consistent_runs))

    return ABMChapterContext(
        runs_root=root,
        run_index=run_index,
        series=series,
        grouped_outcomes=grouped_outcomes,
        grouped_mediators=grouped_mediators,
    )


def _baseline_for_row(incidence_mode: str, communication_enabled: bool) -> str:
    if str(incidence_mode) == "importation":
        return "baseline_importation"
    if not bool(communication_enabled):
        return "baseline"
    return "baseline"


def _npi_index_expr() -> pl.Expr:
    """Return mean NPI index across the three prevention outcomes."""

    return ((pl.col("mask_pressure") + pl.col("mask_symptomatic") + pl.col("stay_home_symptomatic")) / 3.0).alias(
        "npi_index"
    )


def _run_endpoints(context: ABMChapterContext) -> pl.DataFrame:
    return (
        context.series.sort("tick")
        .group_by(
            [
                "scenario",
                "seed",
                "rep",
                "incidence_mode",
                "communication_enabled",
            ]
        )
        .agg(
            pl.col("tick").last().alias("tick_end"),
            pl.col("incidence_mean").sum().alias("incidence_auc"),
            pl.col("incidence_mean").max().alias("incidence_peak"),
            pl.col("incidence_mean").last().alias("incidence_end"),
            pl.col("vax_willingness").last().alias("vax_end"),
            pl.col("mask_pressure").last().alias("mask_end"),
            pl.col("npi_index").last().alias("npi_end"),
            pl.col("danger").last().alias("danger_end"),
            pl.col("low_constraints").last().alias("constraints_end"),
            pl.col("institutional_trust").last().alias("trust_end"),
            pl.col("descriptive_norm").last().alias("descriptive_norm_end"),
        )
        .with_columns(
            pl.struct(["incidence_mode", "communication_enabled"])
            .map_elements(
                lambda row: _baseline_for_row(
                    str(row["incidence_mode"]),
                    bool(row["communication_enabled"]),
                ),
                return_dtype=pl.String,
            )
            .alias("baseline_scenario")
        )
        .sort(["scenario", "seed", "rep"])
    )


def _trajectory_deltas(context: ABMChapterContext) -> pl.DataFrame:
    """Compute per-tick deltas against matched baseline for AUC/peak metrics.

    Returns a DataFrame with per-run, per-tick deltas for outcomes and
    mediators, plus aggregate AUC and peak statistics per run.
    """
    series = context.series.sort("tick")

    # Attach baseline_scenario label to each row
    series = series.with_columns(
        pl.struct(["incidence_mode", "communication_enabled"])
        .map_elements(
            lambda row: _baseline_for_row(
                str(row["incidence_mode"]),
                bool(row["communication_enabled"]),
            ),
            return_dtype=pl.String,
        )
        .alias("baseline_scenario")
    )

    # Get baseline tick-level data
    value_cols = [
        "incidence_mean",
        "vax_willingness",
        "mask_pressure",
        "npi_index",
        "danger",
        "low_constraints",
        "institutional_trust",
        "descriptive_norm",
    ]
    baseline_ticks = series.select(
        pl.col("scenario").alias("baseline_scenario"),
        "seed",
        "rep",
        "tick",
        *[pl.col(c).alias(f"base_{c}") for c in value_cols],
    )

    # Join scenario tick data with matched baseline ticks
    joined = series.join(
        baseline_ticks,
        on=["baseline_scenario", "seed", "rep", "tick"],
        how="left",
    )

    # Compute per-tick deltas
    for col in value_cols:
        joined = joined.with_columns((pl.col(col) - pl.col(f"base_{col}")).alias(f"delta_{col}"))

    # Aggregate AUC/peak over the post-burn-in window for each run.
    delta_cols = [f"delta_{c}" for c in value_cols]

    def _auc_peak_aggs(col: str) -> list[pl.Expr]:
        """Polars aggregation expressions for AUC and peak of a delta column."""
        return [
            # AUC via sum (approximate trapezoid with unit tick spacing)
            pl.col(col).sum().alias(f"{col}_auc"),
            # Peak: value with largest absolute magnitude
            pl.col(col).sort_by(pl.col(col).abs(), descending=True).first().alias(f"{col}_peak"),
            # Time to peak
            pl.col("tick").gather(pl.col(col).abs().arg_max()).first().alias(f"{col}_time_to_peak"),
        ]

    agg_exprs: list[pl.Expr] = []
    for dc in delta_cols:
        agg_exprs.extend(_auc_peak_aggs(dc))

    run_agg = (
        joined.filter(
            (pl.col("tick") >= pl.col("burn_in_ticks"))
            & ~pl.col("scenario").is_in(
                [
                    "baseline",
                    "baseline_importation",
                    "baseline_no_communication",
                ]
            )
        )
        .sort("tick")
        .group_by(["scenario", "seed", "rep", "incidence_mode", "communication_enabled", "baseline_scenario"])
        .agg(*agg_exprs)
    )

    return run_agg


@lru_cache(maxsize=1)
def _behavioral_projection_weights() -> dict[str, dict[str, float]]:
    from beam_abm.abm.config import SimulationConfig
    from beam_abm.abm.data_contracts import load_all_backbone_models, load_levers_config
    from beam_abm.abm.io import load_survey_data
    from beam_abm.abm.metrics import build_behavioral_projection_weights

    cfg = SimulationConfig()
    survey_df = load_survey_data(cfg.survey_csv, cfg.countries)
    levers_cfg = load_levers_config(cfg.levers_path)
    models = load_all_backbone_models(cfg.modeling_dir, levers_cfg, survey_df=survey_df)
    return build_behavioral_projection_weights(models)


def _behavioral_projection_run_effects(context: ABMChapterContext) -> pl.DataFrame:
    cache_key = str(context.runs_root.resolve())
    if cache_key in _BEHAVIORAL_PROJECTION_CACHE:
        return _BEHAVIORAL_PROJECTION_CACHE[cache_key]

    from beam_abm.abm.metrics import compute_behavioral_projection_effects

    weights = _behavioral_projection_weights()
    vax_outcome = next((name for name in _VAX_OUTCOME_CANDIDATES if name in weights), None)
    if vax_outcome is None:
        result = pl.DataFrame()
        _BEHAVIORAL_PROJECTION_CACHE[cache_key] = result
        return result

    vax_weights = {vax_outcome: weights[vax_outcome]}
    run_lookup: dict[tuple[str, int, int], str] = {
        (str(row["scenario"]), int(row["seed"]), int(row["rep"])): str(row["run_dir"])
        for row in context.run_index.iter_rows(named=True)
    }

    rows: list[dict[str, float | int | str]] = []
    for row in context.run_index.iter_rows(named=True):
        scenario = str(row["scenario"])
        seed = int(row["seed"])
        rep = int(row["rep"])
        baseline_scenario = _baseline_for_row(str(row["incidence_mode"]), bool(row["communication_enabled"]))
        if scenario == baseline_scenario:
            continue

        intervention_run = Path(str(row["run_dir"]))
        baseline_run_dir = run_lookup.get((baseline_scenario, seed, rep))
        if baseline_run_dir is None:
            continue
        baseline_run = Path(baseline_run_dir)

        intervention_path = intervention_run / "trajectories" / "mediators.parquet"
        baseline_path = baseline_run / "trajectories" / "mediators.parquet"
        if not intervention_path.exists() or not baseline_path.exists():
            continue

        intervention_mediators = pl.read_parquet(intervention_path).sort("tick")
        baseline_mediators = pl.read_parquet(baseline_path).sort("tick")
        projection = compute_behavioral_projection_effects(
            baseline_mediator_traj=baseline_mediators,
            intervention_mediator_traj=intervention_mediators,
            projection_weights=vax_weights,
        )
        if projection.is_empty():
            continue

        key = f"eta_projection::{vax_outcome}"
        proj_row = projection.filter(pl.col("outcome") == key)
        if proj_row.is_empty():
            continue

        rows.append(
            {
                "scenario": scenario,
                "seed": seed,
                "rep": rep,
                "delta_eta_vax_end": float(proj_row.select("delta_final").item()),
                "delta_eta_vax_auc": float(proj_row.select("delta_auc").item()),
                "delta_eta_vax_abs_auc": float(proj_row.select("delta_abs_auc").item()),
                "delta_eta_vax_peak": float(proj_row.select("peak_delta").item()),
            }
        )

    result = pl.DataFrame(rows) if rows else pl.DataFrame()
    _BEHAVIORAL_PROJECTION_CACHE[cache_key] = result
    return result


def matched_effects(context: ABMChapterContext) -> pl.DataFrame:
    """Compute run-level deltas against matched baseline runs.

    Includes both endpoint deltas and AUC/peak trajectory metrics.
    """

    endpoints = _run_endpoints(context)
    baseline = endpoints.select(
        pl.col("scenario").alias("baseline_scenario"),
        "seed",
        "rep",
        pl.col("incidence_auc").alias("base_incidence_auc"),
        pl.col("incidence_peak").alias("base_incidence_peak"),
        pl.col("incidence_end").alias("base_incidence_end"),
        pl.col("vax_end").alias("base_vax_end"),
        pl.col("mask_end").alias("base_mask_end"),
        pl.col("npi_end").alias("base_npi_end"),
        pl.col("danger_end").alias("base_danger_end"),
        pl.col("constraints_end").alias("base_constraints_end"),
        pl.col("trust_end").alias("base_trust_end"),
        pl.col("descriptive_norm_end").alias("base_descriptive_norm_end"),
    )

    joined = endpoints.join(
        baseline,
        on=["baseline_scenario", "seed", "rep"],
        how="left",
    )

    endpoint_effects = joined.with_columns(
        (pl.col("vax_end") - pl.col("base_vax_end")).alias("delta_vax_end"),
        (pl.col("mask_end") - pl.col("base_mask_end")).alias("delta_mask_end"),
        (pl.col("npi_end") - pl.col("base_npi_end")).alias("delta_npi_end"),
        (pl.col("incidence_peak") - pl.col("base_incidence_peak")).alias("delta_incidence_peak"),
        (pl.col("incidence_auc") - pl.col("base_incidence_auc")).alias("delta_incidence_auc"),
        (pl.col("danger_end") - pl.col("base_danger_end")).alias("delta_danger_end"),
        (pl.col("constraints_end") - pl.col("base_constraints_end")).alias("delta_constraints_end"),
        (pl.col("trust_end") - pl.col("base_trust_end")).alias("delta_trust_end"),
        (pl.col("descriptive_norm_end") - pl.col("base_descriptive_norm_end")).alias("delta_descriptive_norm_end"),
    )

    # Merge trajectory-level AUC/peak metrics
    traj = _trajectory_deltas(context)
    traj_selected = traj.select(
        "scenario",
        "seed",
        "rep",
        pl.col("delta_vax_willingness_auc").alias("delta_vax_auc"),
        pl.col("delta_vax_willingness_peak").alias("delta_vax_peak"),
        pl.col("delta_vax_willingness_time_to_peak").alias("vax_time_to_peak"),
        pl.col("delta_mask_pressure_auc").alias("delta_mask_auc"),
        pl.col("delta_mask_pressure_peak").alias("delta_mask_peak"),
        pl.col("delta_npi_index_auc").alias("delta_npi_auc"),
        pl.col("delta_npi_index_peak").alias("delta_npi_peak"),
        pl.col("delta_incidence_mean_auc").alias("delta_incidence_traj_auc"),
        pl.col("delta_incidence_mean_peak").alias("delta_incidence_traj_peak"),
        pl.col("delta_institutional_trust_auc").alias("delta_trust_auc"),
        pl.col("delta_institutional_trust_peak").alias("delta_trust_peak"),
        pl.col("delta_danger_auc"),
        pl.col("delta_danger_peak"),
        pl.col("delta_low_constraints_auc").alias("delta_constraints_auc"),
        pl.col("delta_low_constraints_peak").alias("delta_constraints_peak"),
        pl.col("delta_descriptive_norm_auc"),
        pl.col("delta_descriptive_norm_peak"),
    )

    matched = endpoint_effects.join(
        traj_selected,
        on=["scenario", "seed", "rep"],
        how="left",
    )
    eta_effects = _behavioral_projection_run_effects(context)
    if eta_effects.is_empty():
        return matched
    return matched.join(
        eta_effects,
        on=["scenario", "seed", "rep"],
        how="left",
    )


def summarize_effects(
    context: ABMChapterContext,
    scenarios: tuple[str, ...] | list[str] | None = None,
) -> pl.DataFrame:
    """Scenario-level mean effects and normal-approximation 95% CIs.

    Includes both endpoint and trajectory-based (AUC/peak) metrics.
    """

    effects = matched_effects(context)
    if scenarios is not None:
        effects = effects.filter(pl.col("scenario").is_in(list(scenarios)))

    metric_cols = [
        # Endpoint metrics
        "delta_vax_end",
        "delta_mask_end",
        "delta_npi_end",
        "delta_incidence_peak",
        "delta_incidence_auc",
        # Trajectory AUC metrics
        "delta_vax_auc",
        "delta_mask_auc",
        "delta_npi_auc",
        "delta_incidence_traj_auc",
        "delta_trust_auc",
        "delta_danger_auc",
        "delta_constraints_auc",
        "delta_descriptive_norm_auc",
        "delta_eta_vax_auc",
        "delta_eta_vax_abs_auc",
        "delta_eta_vax_end",
        # Trajectory peak metrics
        "delta_vax_peak",
        "delta_mask_peak",
        "delta_npi_peak",
        "delta_incidence_traj_peak",
        "delta_trust_peak",
        "delta_danger_peak",
        "delta_constraints_peak",
        "delta_descriptive_norm_peak",
        "delta_eta_vax_peak",
    ]

    # Only include columns that exist in the effects DataFrame
    available_cols = [c for c in metric_cols if c in effects.columns]

    long = effects.select("scenario", *available_cols).unpivot(
        index="scenario",
        variable_name="metric",
        value_name="delta",
    )

    summary = (
        long.group_by(["scenario", "metric"])
        .agg(
            pl.col("delta").count().alias("n"),
            pl.col("delta").mean().alias("mean"),
            pl.col("delta").std().alias("sd"),
        )
        .with_columns(
            pl.when(pl.col("n") > 1)
            .then(1.96 * pl.col("sd") / pl.col("n").cast(pl.Float64).sqrt())
            .otherwise(0.0)
            .alias("ci_half")
        )
        .with_columns(
            (pl.col("mean") - pl.col("ci_half")).alias("ci_low"),
            (pl.col("mean") + pl.col("ci_half")).alias("ci_high"),
        )
    )

    precomputed_auc = load_precomputed_scenario_delta_auc()
    if precomputed_auc is None:
        return summary.sort(["metric", "scenario"])

    scenario_filter: set[str] | None = set(map(str, scenarios)) if scenarios is not None else None
    pre_rows: list[dict[str, object]] = []
    for row in precomputed_auc.iter_rows(named=True):
        scenario = str(row["scenario"])
        if scenario_filter is not None and scenario not in scenario_filter:
            continue
        for metric, mean_col, low_col, high_col, n_col in (
            ("delta_vax_auc", "delta_vax_auc", "delta_vax_auc_ci_low", "delta_vax_auc_ci_high", "n_vax_auc"),
            ("delta_mask_auc", "delta_mask_auc", "delta_mask_auc_ci_low", "delta_mask_auc_ci_high", "n_mask_auc"),
        ):
            mean = _safe_float(row.get(mean_col))
            ci_low = _safe_float(row.get(low_col))
            ci_high = _safe_float(row.get(high_col))
            n_val = int(row.get(n_col) or 0)
            if not np.isfinite(mean) or n_val <= 0:
                continue
            pre_rows.append(
                {
                    "scenario": scenario,
                    "metric": metric,
                    "n": n_val,
                    "mean": mean,
                    "sd": float("nan"),
                    "ci_half": float("nan"),
                    "ci_low": ci_low if np.isfinite(ci_low) else float("nan"),
                    "ci_high": ci_high if np.isfinite(ci_high) else float("nan"),
                }
            )

    if not pre_rows:
        return summary.sort(["metric", "scenario"])

    pre_df = pl.DataFrame(pre_rows)
    summary_filtered = summary.join(
        pre_df.select("scenario", "metric").unique(),
        on=["scenario", "metric"],
        how="anti",
    )
    return pl.concat([summary_filtered, pre_df], how="vertical_relaxed").sort(["metric", "scenario"])


def load_precomputed_scenario_delta_auc() -> pl.DataFrame | None:
    """Load precomputed post-burn matched AUC table if available."""
    candidates = (
        ABM_ARTIFACT_ROOT / "scenario_delta_auc_table.csv",
        Path("thesis/artifacts/abm/scenario_delta_auc_table.csv"),
    )
    for path in candidates:
        if path.exists():
            return pl.read_csv(path)
    return None


def scenario_coverage_table(context: ABMChapterContext) -> Markdown:
    """Return compact scenario coverage diagnostics table."""

    horizon = (
        context.series.group_by("scenario")
        .agg(
            pl.col("tick").max().alias("max_tick"),
            pl.col("run_id").n_unique().alias("n_runs"),
            pl.col("seed").n_unique().alias("n_seeds"),
        )
        .sort("scenario")
    )

    effects = matched_effects(context)
    matched = effects.group_by("scenario").agg(
        pl.col("base_vax_end").is_not_null().sum().alias("matched_runs"),
    )
    table = horizon.join(matched, on="scenario", how="left").with_columns(pl.col("matched_runs").fill_null(0))

    rows = [
        [
            label_scenario(str(row[0])),
            str(int(row[1])),
            str(int(row[2])),
            str(int(row[3])),
            str(int(row[4])),
        ]
        for row in table.iter_rows()
    ]
    return md_table(
        ["Scenario", "Runs", "Seeds", "Max tick", "Matched vs baseline"],
        rows,
    )


def baseline_endogenous_metrics(context: ABMChapterContext) -> dict[str, float]:
    """Headline communication-loop metrics in baseline vs no-communication."""

    endpoints = _run_endpoints(context)

    def _mean_for(scenario: str, column: str) -> float:
        frame = endpoints.filter(pl.col("scenario") == scenario)
        if frame.is_empty():
            return float("nan")
        return _safe_float(frame.select(pl.col(column).mean()).item())

    metrics = {
        "baseline_vax_end": _mean_for(CHAPTER_BASELINE_SCENARIO, "vax_end"),
        "no_comm_vax_end": _mean_for(CHAPTER_NO_COMM_SCENARIO, "vax_end"),
        "baseline_mask_end": _mean_for(CHAPTER_BASELINE_SCENARIO, "mask_end"),
        "no_comm_mask_end": _mean_for(CHAPTER_NO_COMM_SCENARIO, "mask_end"),
        "baseline_trust_end": _mean_for(CHAPTER_BASELINE_SCENARIO, "trust_end"),
        "no_comm_trust_end": _mean_for(CHAPTER_NO_COMM_SCENARIO, "trust_end"),
        "baseline_desc_norm_end": _mean_for(CHAPTER_BASELINE_SCENARIO, "descriptive_norm_end"),
        "no_comm_desc_norm_end": _mean_for(CHAPTER_NO_COMM_SCENARIO, "descriptive_norm_end"),
    }
    metrics["delta_vax_comm"] = metrics["baseline_vax_end"] - metrics["no_comm_vax_end"]
    metrics["delta_mask_comm"] = metrics["baseline_mask_end"] - metrics["no_comm_mask_end"]
    metrics["delta_trust_comm"] = metrics["baseline_trust_end"] - metrics["no_comm_trust_end"]
    metrics["delta_desc_norm_comm"] = metrics["baseline_desc_norm_end"] - metrics["no_comm_desc_norm_end"]

    return metrics


def baseline_state_table(context: ABMChapterContext) -> Markdown:
    """Baseline and no-communication table for $t=1$ and end states."""

    variables = [
        ("incidence_mean", "Incidence"),
        ("vax_willingness", "Vaccination willingness"),
        ("mask_pressure", "Masking (high pressure)"),
        ("descriptive_norm", "Descriptive norm"),
        ("institutional_trust", "Institutional trust"),
        ("danger", "Perceived danger"),
        ("low_constraints", "Low constraints"),
    ]

    per_run = (
        context.series.filter(pl.col("tick") >= 1)
        .sort("tick")
        .group_by("scenario", "seed", "rep")
        .agg(
            pl.col("tick").first().alias("tick1"),
            pl.col("tick").last().alias("tick_end"),
            *[pl.col(col).first().alias(f"{col}_t1") for col, _ in variables],
            *[pl.col(col).last().alias(f"{col}_tend") for col, _ in variables],
        )
    )

    def _fmt(scenario: str, col: str) -> str:
        frame = per_run.filter(pl.col("scenario") == scenario)
        if frame.is_empty():
            return "—"
        values = frame.get_column(col)
        mean, _, _, n = _mean_ci_from_series(values)
        sd = float("nan") if n < 2 else _safe_float(values.std())
        if not np.isfinite(mean):
            return "—"
        if n < 2 or not np.isfinite(sd):
            return f"{mean:.3f}"
        return f"{mean:.3f} ± {sd:.3f}"

    rows: list[list[str]] = []
    for col, label in variables:
        rows.append(
            [
                label,
                _fmt(CHAPTER_BASELINE_SCENARIO, f"{col}_t1"),
                _fmt(CHAPTER_BASELINE_SCENARIO, f"{col}_tend"),
                _fmt(CHAPTER_NO_COMM_SCENARIO, f"{col}_tend"),
            ]
        )

    return md_table(
        ["Variable", "Baseline $t=1$", "Baseline $t=end$", "No-communication $t=end$"],
        rows,
    )


def _trajectory_ci(frame: pl.DataFrame, scenario: str, variable: str) -> pl.DataFrame:
    subset = frame.filter(pl.col("scenario") == scenario)
    if subset.is_empty():
        return pl.DataFrame({"tick": [], "mean": [], "low": [], "high": []})

    summary = (
        subset.group_by("tick")
        .agg(
            pl.col(variable).count().alias("n"),
            pl.col(variable).mean().alias("mean"),
            pl.col(variable).std().alias("sd"),
        )
        .with_columns(
            pl.when(pl.col("n") > 1)
            .then(1.96 * pl.col("sd") / pl.col("n").cast(pl.Float64).sqrt())
            .otherwise(0.0)
            .alias("ci_half")
        )
        .with_columns(
            (pl.col("mean") - pl.col("ci_half")).alias("low"),
            (pl.col("mean") + pl.col("ci_half")).alias("high"),
        )
        .sort("tick")
        .select("tick", "mean", "low", "high")
    )
    return summary


def plot_baseline_sanity(context: ABMChapterContext) -> plt.Figure:
    """Baseline trajectories for incidence and willingness."""

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 3.8), sharex=True)
    specs = [
        ("incidence_mean", "Mean incidence"),
        ("vax_willingness", "Vaccination willingness (mean)"),
    ]
    scenarios = [CHAPTER_BASELINE_SCENARIO, CHAPTER_NO_COMM_SCENARIO]

    for ax, (var, title) in zip(axes, specs, strict=True):
        for idx, scenario in enumerate(scenarios):
            summary = _trajectory_ci(context.series, scenario, var)
            if summary.is_empty():
                continue
            x = np.asarray(summary.get_column("tick").to_list(), dtype=np.float64)
            mean = np.asarray(summary.get_column("mean").to_list(), dtype=np.float64)
            lo = np.asarray(summary.get_column("low").to_list(), dtype=np.float64)
            hi = np.asarray(summary.get_column("high").to_list(), dtype=np.float64)
            color = f"C{idx}"
            ax.plot(x, mean, lw=2.0, color=color, label=label_scenario(scenario))
            ax.fill_between(x, lo, hi, color=color, alpha=0.18)

        ax.set_title(title)
        ax.set_xlabel("Tick")
        ax.grid(alpha=0.25)

    axes[0].set_ylabel("Level")
    axes[1].legend(frameon=False, loc="best")
    fig.suptitle("Baseline sanity check: stable dynamics with/without communication", y=1.02)
    fig.tight_layout()
    return fig


def plot_baseline_closure_diagnostic(context: ABMChapterContext) -> plt.Figure:
    """Four-panel endogenous channel verification plot for baseline dynamics."""

    panel_title_size = 10.5
    axis_label_size = 10.0
    tick_label_size = 10.0
    legend_size = 9.5
    annotation_size = 9.0

    scenarios_present = set(context.series.get_column("scenario").unique().to_list())
    ref_scenario = "baseline" if "baseline" in scenarios_present else sorted(scenarios_present)[0]
    ref_runs = context.run_index.filter(pl.col("scenario") == ref_scenario)
    burn_in_tick = 0
    if not ref_runs.is_empty() and "burn_in_ticks" in ref_runs.columns:
        burn_in_tick = int(
            ref_runs.group_by("burn_in_ticks")
            .agg(pl.len().alias("n"))
            .sort("n", descending=True)
            .head(1)
            .get_column("burn_in_ticks")
            .item()
        )
    # If metadata burn-in is unset (often 0), use a conservative plotting cutoff
    # to avoid initialization transients in the chapter verification figure.
    burn_in_tick = max(burn_in_tick, 12)

    def _trajectory_ci(data: pl.DataFrame, scenario: str, variable: str) -> pl.DataFrame:
        subset = data.filter(pl.col("scenario") == scenario).sort(["run_id", "tick"])
        if subset.is_empty() or variable not in subset.columns:
            return pl.DataFrame({"tick": [], "mean": [], "low": [], "high": []})
        return (
            subset.group_by("tick")
            .agg(
                pl.col(variable).mean().alias("mean"),
                pl.col(variable).quantile(0.05).alias("low"),
                pl.col(variable).quantile(0.95).alias("high"),
            )
            .sort("tick")
            .select("tick", "mean", "low", "high")
        )

    def _reference_params(
        data: pl.DataFrame,
        scenario: str,
        variable: str,
        ref_tick: int,
        sd_column: str | None = None,
    ) -> tuple[float, float]:
        base = data.filter((pl.col("scenario") == scenario) & (pl.col("tick") == ref_tick))
        if base.is_empty():
            return 0.0, 1.0

        mean1 = _safe_float(base.select(pl.col(variable).mean()).item()) if variable in data.columns else 0.0
        if np.isfinite(mean1) is False:
            mean1 = 0.0

        s1 = float("nan")
        if sd_column is not None and sd_column in data.columns:
            s1 = _safe_float(base.select(pl.col(sd_column).mean()).item())
        if not np.isfinite(s1) or s1 <= 0.0:
            # Fallback for variables lacking cross-agent SD in saved trajectories
            s1 = _safe_float(base.select(pl.col(variable).std()).item()) if variable in data.columns else float("nan")
        if not np.isfinite(s1) or s1 <= 0.0:
            s1 = 1.0
        return mean1, s1

    def _trajectory_z_ci(
        data: pl.DataFrame,
        scenario: str,
        variable: str,
        ref_scenario_name: str,
        ref_tick: int,
        sd_column: str | None = None,
    ) -> pl.DataFrame:
        summary = _trajectory_ci(data, scenario, variable)
        if summary.is_empty():
            return summary
        mean1, s1 = _reference_params(
            data,
            scenario=ref_scenario_name,
            variable=variable,
            ref_tick=ref_tick,
            sd_column=sd_column,
        )
        return summary.with_columns(
            ((pl.col("mean") - pl.lit(mean1)) / pl.lit(s1)).alias("mean"),
            ((pl.col("low") - pl.lit(mean1)) / pl.lit(s1)).alias("low"),
            ((pl.col("high") - pl.lit(mean1)) / pl.lit(s1)).alias("high"),
        )

    def _trajectory_ratio_ci(data: pl.DataFrame, scenario: str, variable: str, ref_tick: int) -> pl.DataFrame:
        summary = _trajectory_ci(data, scenario, variable)
        if summary.is_empty():
            return summary
        ref_level, _ = _reference_params(data, scenario=scenario, variable=variable, ref_tick=ref_tick)
        denom = ref_level if np.isfinite(ref_level) and abs(ref_level) > 1e-12 else 1.0
        return summary.with_columns(
            (pl.col("mean") / pl.lit(denom)).alias("mean"),
            (pl.col("low") / pl.lit(denom)).alias("low"),
            (pl.col("high") / pl.lit(denom)).alias("high"),
        )

    def _trajectory_delta_ci(data: pl.DataFrame, scenario: str, variable: str, ref_tick: int) -> pl.DataFrame:
        summary = _trajectory_ci(data, scenario, variable)
        if summary.is_empty():
            return summary
        ref_level, _ = _reference_params(data, scenario=scenario, variable=variable, ref_tick=ref_tick)
        return summary.with_columns(
            (pl.col("mean") - pl.lit(ref_level)).alias("mean"),
            (pl.col("low") - pl.lit(ref_level)).alias("low"),
            (pl.col("high") - pl.lit(ref_level)).alias("high"),
        )

    def _ticks_for_panel(data: pl.DataFrame) -> np.ndarray:
        if data.is_empty():
            return np.asarray([], dtype=np.float64)
        return np.asarray(data.get_column("tick").to_list(), dtype=np.float64)

    def _first_diff_summary(data: pl.DataFrame) -> pl.DataFrame:
        if data.is_empty():
            return data
        sorted_df = data.sort("tick")
        return (
            sorted_df.with_columns(
                (pl.col("mean") - pl.col("mean").shift(1)).alias("mean"),
                (pl.col("low") - pl.col("low").shift(1)).alias("low"),
                (pl.col("high") - pl.col("high").shift(1)).alias("high"),
            )
            .drop_nulls(["mean", "low", "high"])
            .select("tick", "mean", "low", "high")
        )

    def _plot_line_ci(
        ax: plt.Axes,
        data: pl.DataFrame,
        *,
        color: str,
        label: str,
        lw: float = 2.0,
        ls: str = "-",
        alpha_fill: float = 0.12,
        zorder: int = 3,
    ) -> None:
        if data.is_empty():
            return
        x = _ticks_for_panel(data)
        y = np.asarray(data.get_column("mean").to_list(), dtype=np.float64)
        lo = np.asarray(data.get_column("low").to_list(), dtype=np.float64)
        hi = np.asarray(data.get_column("high").to_list(), dtype=np.float64)
        ax.plot(x, y, lw=lw, ls=ls, color=color, label=label, zorder=zorder)
        ax.fill_between(x, lo, hi, color=color, alpha=alpha_fill, zorder=max(1, zorder - 1))

    frame = context.series.filter(pl.col("tick") >= burn_in_tick)

    fig, axes = plt.subplots(3, 1, figsize=(6.2, 5.6), sharex=True)
    ax_a, ax_b, ax_c = axes

    # Panel A — incidence regimes with communication ablations.
    panel_a_specs = [
        ("baseline", "C0", "-", "Baseline"),
        ("baseline_no_communication", "C0", "--", "Baseline (no communication)"),
        ("baseline_importation", "C3", "-", "Baseline + importation"),
        ("baseline_importation_no_communication", "C3", "--", "Baseline + importation (no communication)"),
    ]
    for scenario, color, ls, label in panel_a_specs:
        if scenario not in scenarios_present:
            continue
        _plot_line_ci(
            ax_a,
            _trajectory_ci(frame, scenario, "incidence_mean"),
            color=color,
            ls=ls,
            label=label,
            lw=2.0,
            alpha_fill=0.14,
        )
    ax_a.set_title("A. Incidence dynamics under regime toggles", fontsize=panel_title_size)
    ax_a.set_ylabel("Mean incidence", fontsize=axis_label_size)
    ax_a.grid(alpha=0.25)
    combined_handles = [
        Line2D([0], [0], color="C0", lw=2.0, ls="-", label="Baseline"),
        Line2D([0], [0], color="C3", lw=2.0, ls="-", label="Baseline + importation"),
        Line2D([0], [0], color="0.25", lw=2.0, ls="-", label="Communication on"),
        Line2D([0], [0], color="0.25", lw=2.0, ls="--", label="Communication off"),
    ]
    ax_a.legend(
        handles=combined_handles,
        frameon=True,
        facecolor="white",
        edgecolor="none",
        framealpha=0.82,
        fontsize=legend_size,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        borderaxespad=0.0,
        ncol=2,
        handlelength=1.8,
        handletextpad=0.5,
        columnspacing=1.0,
        labelspacing=0.25,
    )

    # Panel B — one incidence-driven construct (baseline vs importation).
    mediator_var = "fear" if frame.filter(pl.col("fear").is_not_null()).height > 0 else "danger"
    mediator_label = "Disease fear" if mediator_var == "fear" else "Perceived danger"
    for scenario, color, label in [
        ("baseline", "C0", "Baseline"),
        ("baseline_importation", "C3", "Baseline + importation"),
    ]:
        if scenario not in scenarios_present:
            continue
        _plot_line_ci(
            ax_b,
            _trajectory_ci(frame, scenario, mediator_var),
            color=color,
            label=label,
            lw=2.1,
            alpha_fill=0.12,
        )
    ax_b.set_title("B. Salience response", fontsize=panel_title_size)
    # use mediator_label to keep y-axis dynamic
    ax_b.set_ylabel(f"Mean {mediator_label.lower()}", fontsize=axis_label_size)
    ax_b.grid(alpha=0.25)
    ax_b.legend(
        frameon=True,
        facecolor="white",
        edgecolor="none",
        framealpha=0.82,
        fontsize=legend_size,
        loc="lower right",
        bbox_to_anchor=(0.99, 0.02),
        borderaxespad=0.0,
        handlelength=1.8,
        handletextpad=0.5,
        labelspacing=0.25,
    )

    # Effect helper: ON - OFF matched by seed/rep/tick.
    def _matched_effect_ci(on_scenario: str, off_scenario: str, variable: str) -> pl.DataFrame:
        on = frame.filter(pl.col("scenario") == on_scenario).select("seed", "rep", "tick", pl.col(variable).alias("on"))
        off = frame.filter(pl.col("scenario") == off_scenario).select(
            "seed", "rep", "tick", pl.col(variable).alias("off")
        )
        if on.is_empty() or off.is_empty():
            return pl.DataFrame({"tick": [], "mean": [], "low": [], "high": []})
        delta = (
            on.join(off, on=["seed", "rep", "tick"], how="inner")
            .with_columns((pl.col("on") - pl.col("off")).alias("delta"))
            .select("tick", "delta")
        )
        if delta.is_empty():
            return pl.DataFrame({"tick": [], "mean": [], "low": [], "high": []})
        return (
            delta.group_by("tick")
            .agg(
                pl.col("delta").mean().alias("mean"),
                pl.col("delta").quantile(0.05).alias("low"),
                pl.col("delta").quantile(0.95).alias("high"),
            )
            .sort("tick")
            .select("tick", "mean", "low", "high")
        )

    # Panel C — direct social-channel targets (effects).
    trust_peer = _matched_effect_ci("baseline", "baseline_no_peer_pull", "institutional_trust")
    desc_obs = _matched_effect_ci("baseline", "baseline_no_observed_norms", "descriptive_norm")
    _plot_line_ci(
        ax_c,
        trust_peer,
        color="C0",
        label="Peer pull effect on trust (baseline - no peer pull)",
        lw=2.0,
        alpha_fill=0.12,
    )
    _plot_line_ci(
        ax_c,
        desc_obs,
        color="C2",
        label="Observed norms effect on descriptive norm (baseline - no observed norms)",
        lw=2.0,
        alpha_fill=0.12,
    )
    ax_c.set_title("C. Social-channel direct targets", fontsize=panel_title_size)
    ax_c.set_ylabel("Effect (ON - OFF)", fontsize=axis_label_size)
    ax_c.axhline(0.0, color="0.35", lw=0.9, ls="-", alpha=0.55, zorder=0)
    ax_c.grid(alpha=0.25)

    def _label_curve_end(
        ax: plt.Axes,
        data: pl.DataFrame,
        text: str,
        *,
        color: str,
        dy: float = 0.0,
    ) -> None:
        if data.is_empty():
            return
        tick_vals = np.asarray(data.get_column("tick").to_list(), dtype=np.float64)
        mean_vals = np.asarray(data.get_column("mean").to_list(), dtype=np.float64)
        if tick_vals.size == 0 or mean_vals.size == 0:
            return
        x_last = float(tick_vals[-1])
        y_last = float(mean_vals[-1])
        ax.annotate(
            text,
            xy=(x_last, y_last),
            xytext=(-6, int(dy)),
            textcoords="offset points",
            ha="right",
            va="top" if dy < 0 else "bottom",
            color=color,
            fontsize=annotation_size,
            bbox={"boxstyle": "round,pad=0.12", "fc": "white", "ec": "none", "alpha": 0.7},
        )

    _label_curve_end(
        ax_c,
        trust_peer,
        "Peer pull \u2192 institutional trust",
        color="C0",
        dy=-18.0,
    )
    _label_curve_end(
        ax_c,
        desc_obs,
        "Observed norms \u2192 descriptive norm",
        color="C2",
        dy=12.0,
    )

    max_tick = int(frame.get_column("tick").max()) if frame.height > 0 else burn_in_tick
    common_xticks = np.arange(burn_in_tick, max_tick + 1, 4, dtype=int)
    for ax in (ax_a, ax_b, ax_c):
        ax.set_xticks(common_xticks)
        ax.set_xlim(burn_in_tick, max_tick)
        ax.tick_params(axis="both", labelsize=tick_label_size)

    ax_a.set_xlabel("")
    ax_b.set_xlabel("")
    ax_c.set_xlabel("Month (tick)", fontsize=axis_label_size, labelpad=2)
    fig.tight_layout()
    return fig


def compute_norm_channel_baseline_checks(
    context: ABMChapterContext,
    *,
    baseline_scenario: str = CHAPTER_BASELINE_SCENARIO,
) -> dict[str, float]:
    """Compute lightweight baseline diagnostics for norm/trust channels."""
    from beam_abm.abm.engine import _agent_peer_pull_rates_per_substep
    from beam_abm.abm.network import build_contact_network
    from beam_abm.abm.outcome_scales import normalize_outcome_to_unit_interval

    baseline_runs = context.run_index.filter(pl.col("scenario") == baseline_scenario).sort(["seed", "rep"])
    if baseline_runs.is_empty():
        return {}
    run = baseline_runs.row(0, named=True)

    run_dir = Path(str(run["run_dir"]))
    t0_path = run_dir / "populations" / "t0.parquet"
    meta_path = run_dir / "metadata.json"
    if not t0_path.exists() or not meta_path.exists():
        return {}

    agents_t0 = pl.read_parquet(t0_path)
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    constants = metadata.get("simulation_config", {}).get("constants", {})
    bridge_prob = float(constants.get("bridge_prob", 0.08))
    base_rate = float(constants.get("peer_pull_interactions_per_agent", 0.3))
    household_enabled = bool(metadata.get("simulation_config", {}).get("household_enabled", True))
    seed = int(run.get("seed", 42))

    network = build_contact_network(
        agents_t0,
        bridge_prob=bridge_prob,
        household_enabled=household_enabled,
        seed=seed,
    )

    degree = network.core_degree.astype(np.float64, copy=False)
    peer_rates = _agent_peer_pull_rates_per_substep(agents_t0, base_rate=base_rate)

    trust = (
        agents_t0["institutional_trust_avg"].cast(pl.Float64).to_numpy().astype(np.float64)
        if "institutional_trust_avg" in agents_t0.columns
        else np.full(agents_t0.height, np.nan, dtype=np.float64)
    )
    risk = (
        agents_t0["vaccine_risk_avg"].cast(pl.Float64).to_numpy().astype(np.float64)
        if "vaccine_risk_avg" in agents_t0.columns
        else np.full(agents_t0.height, np.nan, dtype=np.float64)
    )

    def _corr(a: np.ndarray, b: np.ndarray) -> float:
        mask = np.isfinite(a) & np.isfinite(b)
        if np.sum(mask) < 3:
            return float("nan")
        aa = a[mask]
        bb = b[mask]
        if float(np.std(aa)) <= 0.0 or float(np.std(bb)) <= 0.0:
            return float("nan")
        return float(np.corrcoef(aa, bb)[0, 1])

    y_col = "vax_willingness_T12" if "vax_willingness_T12" in agents_t0.columns else "vax_willingness_Y"
    y = normalize_outcome_to_unit_interval(
        agents_t0[y_col].cast(pl.Float64).to_numpy().astype(np.float64),
        "vax_willingness_T12",
    )
    perceived = (
        agents_t0["norm_share_hat_vax"].cast(pl.Float64).to_numpy().astype(np.float64)
        if "norm_share_hat_vax" in agents_t0.columns
        else np.full(agents_t0.height, np.nan, dtype=np.float64)
    )

    src = network.recurring_edge_src
    dst = network.recurring_edge_dst
    counts = (
        np.bincount(src, minlength=agents_t0.height).astype(np.float64) if src.size > 0 else np.zeros(agents_t0.height)
    )
    sums = (
        np.bincount(src, weights=y[dst], minlength=agents_t0.height).astype(np.float64)
        if src.size > 0
        else np.zeros(agents_t0.height)
    )
    true_neighbor_share = y.copy()
    has_neighbors = counts > 0.0
    true_neighbor_share[has_neighbors] = sums[has_neighbors] / counts[has_neighbors]
    true_neighbor_share = np.clip(true_neighbor_share, 0.0, 1.0)

    return {
        "corr_degree_peer_rate": _corr(degree, peer_rates),
        "corr_degree_trust": _corr(degree, trust),
        "corr_degree_risk": _corr(degree, risk),
        "corr_perceived_true_neighbor_share": _corr(perceived, true_neighbor_share),
        "perceived_mean": float(np.nanmean(perceived)),
        "true_neighbor_share_mean": float(np.nanmean(true_neighbor_share)),
    }


def plot_norm_channel_baseline_checks(
    context: ABMChapterContext,
    *,
    baseline_scenario: str = CHAPTER_BASELINE_SCENARIO,
) -> plt.Figure:
    """Appendix-ready diagnostics for norm channels under baseline."""
    from beam_abm.abm.engine import _agent_peer_pull_rates_per_substep
    from beam_abm.abm.network import build_contact_network
    from beam_abm.abm.outcome_scales import normalize_outcome_to_unit_interval

    baseline_runs = context.run_index.filter(pl.col("scenario") == baseline_scenario).sort(["seed", "rep"])
    if baseline_runs.is_empty():
        fig, ax = plt.subplots(1, 1, figsize=(8.0, 3.2))
        ax.text(0.5, 0.5, "No baseline run available", ha="center", va="center")
        ax.set_axis_off()
        return fig
    run = baseline_runs.row(0, named=True)

    run_dir = Path(str(run["run_dir"]))
    agents_t0 = pl.read_parquet(run_dir / "populations" / "t0.parquet")
    metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    constants = metadata.get("simulation_config", {}).get("constants", {})
    bridge_prob = float(constants.get("bridge_prob", 0.08))
    base_rate = float(constants.get("peer_pull_interactions_per_agent", 0.3))
    household_enabled = bool(metadata.get("simulation_config", {}).get("household_enabled", True))
    seed = int(run.get("seed", 42))

    network = build_contact_network(
        agents_t0,
        bridge_prob=bridge_prob,
        household_enabled=household_enabled,
        seed=seed,
    )
    degree = network.core_degree.astype(np.float64, copy=False)
    peer_rates = _agent_peer_pull_rates_per_substep(agents_t0, base_rate=base_rate)

    y_col = "vax_willingness_T12" if "vax_willingness_T12" in agents_t0.columns else "vax_willingness_Y"
    y = normalize_outcome_to_unit_interval(
        agents_t0[y_col].cast(pl.Float64).to_numpy().astype(np.float64),
        "vax_willingness_T12",
    )
    perceived = agents_t0["norm_share_hat_vax"].cast(pl.Float64).to_numpy().astype(np.float64)

    src = network.recurring_edge_src
    dst = network.recurring_edge_dst
    counts = (
        np.bincount(src, minlength=agents_t0.height).astype(np.float64) if src.size > 0 else np.zeros(agents_t0.height)
    )
    sums = (
        np.bincount(src, weights=y[dst], minlength=agents_t0.height).astype(np.float64)
        if src.size > 0
        else np.zeros(agents_t0.height)
    )
    true_neighbor_share = y.copy()
    has_neighbors = counts > 0.0
    true_neighbor_share[has_neighbors] = sums[has_neighbors] / counts[has_neighbors]
    true_neighbor_share = np.clip(true_neighbor_share, 0.0, 1.0)

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.2))

    ax = axes[0]
    sample_n = min(4000, degree.size)
    rng = np.random.default_rng(123)
    idx = rng.choice(degree.size, size=sample_n, replace=False) if sample_n > 0 else np.array([], dtype=np.int64)
    ax.scatter(degree[idx], peer_rates[idx], s=7, alpha=0.25, color="C0", edgecolors="none")
    corr_deg_rate = float(np.corrcoef(degree, peer_rates)[0, 1]) if degree.size > 2 else float("nan")
    ax.set_xlabel("Core degree")
    ax.set_ylabel("Peer-pull updates per substep")
    ax.set_title(f"Degree vs peer-pull participation (r={corr_deg_rate:.3f})")
    ax.grid(alpha=0.2)

    ax = axes[1]
    bins = np.linspace(0.0, 1.0, 31)
    ax.hist(perceived, bins=bins, density=True, alpha=0.55, color="C1", label="Perceived exposure proxy")
    ax.hist(true_neighbor_share, bins=bins, density=True, alpha=0.45, color="C2", label="True neighbor Y-share")
    corr_exposure = float(np.corrcoef(perceived, true_neighbor_share)[0, 1]) if perceived.size > 2 else float("nan")
    ax.set_xlabel("Share")
    ax.set_ylabel("Density")
    ax.set_title(f"Perceived vs true neighbor share (r={corr_exposure:.3f})")
    ax.legend(loc="upper center", ncols=1, frameon=False)
    ax.grid(alpha=0.2)

    fig.suptitle("Baseline checks for norm channels (single baseline run)", fontsize=12)
    fig.tight_layout()
    return fig


def plot_scenario_leaderboard(
    context: ABMChapterContext,
    scenarios: tuple[str, ...] | list[str] = LEADERBOARD_SCENARIOS,
) -> plt.Figure:
    """Dot-and-CI leaderboard for core macro deltas vs matched baseline.

    Uses cumulative AUC and peak-delta metrics as primary evaluation,
    with endpoint deltas shown for reference.
    """

    summary = summarize_effects(context, scenarios=scenarios)
    metric_specs = [
        ("delta_vax_auc", "Δ willingness (AUC)"),
        ("delta_vax_peak", "Δ willingness (peak)"),
        ("delta_mask_auc", "Δ masking (AUC)"),
        ("delta_incidence_traj_peak", "Δ peak incidence"),
    ]

    # Fallback: if AUC/peak metrics are missing, use endpoint metrics
    available_metrics = set(summary.get_column("metric").unique().to_list())
    fallback_specs = [
        ("delta_vax_end", "Δ willingness (end)"),
        ("delta_mask_end", "Δ masking under pressure (end)"),
        ("delta_incidence_peak", "Δ peak incidence"),
    ]
    active_specs = [spec for spec in metric_specs if spec[0] in available_metrics]
    if not active_specs:
        active_specs = [spec for spec in fallback_specs if spec[0] in available_metrics]

    n_panels = len(active_specs)
    if n_panels == 0:
        fig, ax = plt.subplots(figsize=(6.0, 3.0))
        ax.text(0.5, 0.5, "No effect metrics available", ha="center", va="center")
        ax.axis("off")
        return fig

    fig, axes = plt.subplots(1, n_panels, figsize=(3.3 * n_panels, 4.2), sharey=True)
    if n_panels == 1:
        axes = [axes]
    ordered = [s for s in scenarios if s in summary.get_column("scenario").unique().to_list()]

    y = np.arange(len(ordered), dtype=np.float64)
    labels = [label_scenario(s) for s in ordered]

    for ax, (metric, title) in zip(axes, active_specs, strict=True):
        sub = summary.filter(pl.col("metric") == metric)
        lookup = {
            str(row[0]): (float(row[1]), float(row[2]), float(row[3]))
            for row in sub.select("scenario", "mean", "ci_low", "ci_high").iter_rows()
        }
        mean = np.asarray([lookup.get(s, (np.nan, np.nan, np.nan))[0] for s in ordered], dtype=np.float64)
        low = np.asarray([lookup.get(s, (np.nan, np.nan, np.nan))[1] for s in ordered], dtype=np.float64)
        high = np.asarray([lookup.get(s, (np.nan, np.nan, np.nan))[2] for s in ordered], dtype=np.float64)
        err_low = mean - low
        err_high = high - mean

        ax.errorbar(
            mean,
            y,
            xerr=np.vstack([err_low, err_high]),
            fmt="o",
            color="C0",
            ecolor="C0",
            elinewidth=1.6,
            capsize=3,
        )
        ax.axvline(0.0, color="black", lw=1.0, alpha=0.65)
        ax.set_title(title)
        ax.grid(axis="x", alpha=0.25)
        ax.set_xlabel("Effect vs matched baseline")

    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels)
    for ax in axes[1:]:
        ax.set_yticks(y)
        ax.set_yticklabels([])

    fig.suptitle("Scenario leaderboard: cumulative and peak effects (mean ± 95% CI)", y=1.02)
    fig.tight_layout()
    return fig


def _scenario_baseline(context: ABMChapterContext, scenario: str) -> str:
    row = (
        context.run_index.filter(pl.col("scenario") == scenario)
        .select("incidence_mode", "communication_enabled")
        .head(1)
    )
    if row.is_empty():
        return CHAPTER_BASELINE_SCENARIO
    incidence_mode, communication_enabled = row.row(0)
    return _baseline_for_row(str(incidence_mode), bool(communication_enabled))


def plot_mechanism_panel(
    context: ABMChapterContext,
    scenario: str,
    mediator: str,
    outcome: str,
) -> plt.Figure:
    """Three-row mechanism panel: incidence, mediator, and outcome trajectories."""

    baseline = _scenario_baseline(context, scenario)
    frame = context.series.filter(pl.col("scenario").is_in([scenario, baseline]))

    tracks = [
        ("incidence_mean", "Mean incidence"),
        (mediator, "Mediator"),
        (outcome, "Outcome"),
    ]

    fig, axes = plt.subplots(3, 1, figsize=(8.2, 8.2), sharex=True)
    for ax, (var, ylabel) in zip(axes, tracks, strict=True):
        for idx, name in enumerate([baseline, scenario]):
            summary = _trajectory_ci(frame, name, var)
            if summary.is_empty():
                continue
            x = np.asarray(summary.get_column("tick").to_list(), dtype=np.float64)
            mean = np.asarray(summary.get_column("mean").to_list(), dtype=np.float64)
            lo = np.asarray(summary.get_column("low").to_list(), dtype=np.float64)
            hi = np.asarray(summary.get_column("high").to_list(), dtype=np.float64)
            color = f"C{idx}"
            ax.plot(x, mean, lw=2.0, color=color, label=label_scenario(name))
            ax.fill_between(x, lo, hi, color=color, alpha=0.18)

        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)

    axes[2].set_xlabel("Tick")
    axes[0].legend(frameon=False, loc="best")
    fig.suptitle(f"Mechanism trajectory: {label_scenario(scenario)}", y=1.01)
    fig.tight_layout()
    return fig


def anchor_effects(
    context: ABMChapterContext,
    scenarios: tuple[str, ...] | list[str],
) -> pl.DataFrame:
    """Anchor-level end-point deltas in willingness vs matched baseline."""

    if context.grouped_outcomes.is_empty():
        return pl.DataFrame()

    grouped = context.grouped_outcomes.filter(pl.col("group_type") == "anchor_profile")
    end_values = (
        grouped.sort("tick")
        .group_by(
            [
                "scenario",
                "seed",
                "rep",
                "incidence_mode",
                "communication_enabled",
                "group_value",
            ]
        )
        .agg(pl.col("mean").last().alias("vax_end"))
        .with_columns(
            pl.struct(["incidence_mode", "communication_enabled"])
            .map_elements(
                lambda row: _baseline_for_row(
                    str(row["incidence_mode"]),
                    bool(row["communication_enabled"]),
                ),
                return_dtype=pl.String,
            )
            .alias("baseline_scenario")
        )
    )

    baseline = end_values.select(
        pl.col("scenario").alias("baseline_scenario"),
        "seed",
        "rep",
        "group_value",
        pl.col("vax_end").alias("base_vax_end"),
    )

    deltas = (
        end_values.join(
            baseline,
            on=["baseline_scenario", "seed", "rep", "group_value"],
            how="left",
        )
        .with_columns((pl.col("vax_end") - pl.col("base_vax_end")).alias("delta_vax_end"))
        .filter(pl.col("scenario").is_in(list(scenarios)))
    )

    return (
        deltas.group_by("scenario", "group_value")
        .agg(
            pl.col("delta_vax_end").mean().alias("mean"),
            pl.col("delta_vax_end").std().alias("sd"),
            pl.len().alias("n"),
        )
        .sort(["scenario", "group_value"])
    )


def plot_anchor_effect_heatmap(
    context: ABMChapterContext,
    scenarios: tuple[str, ...] | list[str] = ("access_intervention", "outbreak_global"),
) -> plt.Figure:
    """Heatmap of anchor-level willingness effects for selected scenarios."""

    agg = anchor_effects(context, scenarios)
    if agg.is_empty():
        fig, ax = plt.subplots(figsize=(6.0, 3.0))
        ax.text(0.5, 0.5, "No anchor-level outputs found", ha="center", va="center")
        ax.axis("off")
        return fig

    anchors = sorted(
        agg.get_column("group_value").unique().to_list(),
        key=lambda item: (int(item) if str(item).isdigit() else str(item)),
    )
    scenario_order = [s for s in scenarios if s in agg.get_column("scenario").unique().to_list()]

    pivot = agg.select("group_value", "scenario", "mean").pivot(
        index="group_value",
        on="scenario",
        values="mean",
    )
    pivot = pivot.sort(
        pl.col("group_value").cast(pl.Int64, strict=False).fill_null(10_000),
        "group_value",
    )

    matrix = np.zeros((len(anchors), len(scenario_order)), dtype=np.float64)
    matrix.fill(np.nan)

    table_lookup: dict[tuple[str, str], float] = {
        (str(row[0]), str(row[1])): float(row[2]) for row in agg.select("group_value", "scenario", "mean").iter_rows()
    }
    for i, anchor in enumerate(anchors):
        for j, scenario in enumerate(scenario_order):
            matrix[i, j] = table_lookup.get((str(anchor), scenario), np.nan)

    finite = matrix[np.isfinite(matrix)]
    vmax = float(np.max(np.abs(finite))) if finite.size else 0.1
    vmax = max(vmax, 0.01)

    fig, ax = plt.subplots(figsize=(7.4, max(3.8, 0.24 * len(anchors) + 2.0)))
    image = ax.imshow(
        matrix,
        aspect="auto",
        cmap="coolwarm",
        norm=TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax),
    )
    ax.set_xticks(np.arange(len(scenario_order), dtype=np.float64))
    ax.set_xticklabels([label_scenario(s) for s in scenario_order], rotation=25, ha="right")
    ax.set_yticks(np.arange(len(anchors), dtype=np.float64))
    ax.set_yticklabels([f"Anchor {a}" for a in anchors])
    ax.set_title("Anchor-level Δ willingness vs matched baseline")

    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Δ willingness")

    fig.tight_layout()
    return fig


def country_contrast_table(
    context: ABMChapterContext,
    scenarios: tuple[str, ...] | list[str] = COUNTRY_SCENARIOS,
) -> Markdown:
    """Country heterogeneity table for willingness end-point deltas."""

    if context.grouped_outcomes.is_empty():
        return md_table(["Scenario", "Mean Δ", "Lowest country", "Highest country"], [])

    grouped = context.grouped_outcomes.filter(pl.col("group_type") == "country")
    end_values = (
        grouped.sort("tick")
        .group_by(
            [
                "scenario",
                "seed",
                "rep",
                "incidence_mode",
                "communication_enabled",
                "group_value",
            ]
        )
        .agg(pl.col("mean").last().alias("vax_end"))
        .with_columns(
            pl.struct(["incidence_mode", "communication_enabled"])
            .map_elements(
                lambda row: _baseline_for_row(
                    str(row["incidence_mode"]),
                    bool(row["communication_enabled"]),
                ),
                return_dtype=pl.String,
            )
            .alias("baseline_scenario")
        )
    )

    baseline = end_values.select(
        pl.col("scenario").alias("baseline_scenario"),
        "seed",
        "rep",
        "group_value",
        pl.col("vax_end").alias("base_vax_end"),
    )

    deltas = (
        end_values.join(
            baseline,
            on=["baseline_scenario", "seed", "rep", "group_value"],
            how="left",
        )
        .with_columns((pl.col("vax_end") - pl.col("base_vax_end")).alias("delta_vax_end"))
        .filter(pl.col("scenario").is_in(list(scenarios)))
    )

    rows: list[list[str]] = []
    for scenario in scenarios:
        sub = deltas.filter(pl.col("scenario") == scenario)
        if sub.is_empty():
            continue

        country_stats = sub.group_by("group_value").agg(pl.col("delta_vax_end").mean().alias("mean")).sort("mean")
        overall_mean = _safe_float(sub.select(pl.col("delta_vax_end").mean()).item())
        low_country, low_value = country_stats.row(0)
        high_country, high_value = country_stats.row(country_stats.height - 1)

        rows.append(
            [
                label_scenario(scenario),
                f"{overall_mean:.3f}",
                f"{low_country} ({_safe_float(low_value):.3f})",
                f"{high_country} ({_safe_float(high_value):.3f})",
            ]
        )

    return md_table(["Scenario", "Mean Δ", "Lowest country", "Highest country"], rows)


def interaction_summary(context: ABMChapterContext) -> pl.DataFrame:
    """Observed combined vs additive single-shock effects."""

    effects = matched_effects(context)
    required = ["outbreak_global", "misinfo_shock", "outbreak_misinfo_combined"]
    sub = effects.filter(pl.col("scenario").is_in(required))

    metrics = [
        "delta_vax_end",
        "delta_mask_end",
        "delta_incidence_peak",
        "delta_incidence_auc",
    ]

    rows: list[dict[str, object]] = []
    for metric in metrics:
        pivot = sub.select("seed", "rep", "scenario", metric).pivot(
            index=["seed", "rep"],
            on="scenario",
            values=metric,
        )
        needed = set(required)
        if not needed.issubset(set(pivot.columns)):
            continue

        additive = pivot.get_column("outbreak_global") + pivot.get_column("misinfo_shock")
        observed = pivot.get_column("outbreak_misinfo_combined")
        gap = observed - additive
        obs_mean, obs_lo, obs_hi, obs_n = _mean_ci_from_series(observed)
        add_mean, add_lo, add_hi, add_n = _mean_ci_from_series(additive)
        gap_mean, gap_lo, gap_hi, gap_n = _mean_ci_from_series(gap)

        rows.append(
            {
                "metric": metric,
                "observed_mean": obs_mean,
                "observed_low": obs_lo,
                "observed_high": obs_hi,
                "additive_mean": add_mean,
                "additive_low": add_lo,
                "additive_high": add_hi,
                "gap_mean": gap_mean,
                "gap_low": gap_lo,
                "gap_high": gap_hi,
                "n": int(min(obs_n, add_n, gap_n)),
            }
        )

    return pl.DataFrame(rows)


def plot_interaction_observed_vs_additive(context: ABMChapterContext) -> plt.Figure:
    """Observed combined effects vs additive counterfactual prediction."""

    summary = interaction_summary(context)
    metric_labels = {
        "delta_vax_end": "Δ willingness",
        "delta_mask_end": "Δ masking",
        "delta_incidence_peak": "Δ peak incidence",
        "delta_incidence_auc": "Δ incidence AUC",
    }

    if summary.is_empty():
        fig, ax = plt.subplots(figsize=(6.4, 3.2))
        ax.text(0.5, 0.5, "No interaction runs available", ha="center", va="center")
        ax.axis("off")
        return fig

    ordered = [
        metric
        for metric in [
            "delta_vax_end",
            "delta_mask_end",
            "delta_incidence_peak",
            "delta_incidence_auc",
        ]
        if metric in summary.get_column("metric").to_list()
    ]
    sub = summary.with_columns(pl.col("metric").cast(pl.Categorical)).sort("metric")
    lookup = {str(row[0]): row[1:] for row in sub.iter_rows()}

    y = np.arange(len(ordered), dtype=np.float64)

    fig, ax = plt.subplots(figsize=(8.6, 4.0))
    for i, metric in enumerate(ordered):
        values = lookup.get(metric)
        if values is None:
            continue
        observed_mean, observed_low, observed_high, additive_mean, additive_low, additive_high, *_ = values

        ax.plot([additive_mean, observed_mean], [i, i], color="0.65", lw=2.0)
        ax.errorbar(
            additive_mean,
            i,
            xerr=[[additive_mean - additive_low], [additive_high - additive_mean]],
            fmt="o",
            color="C1",
            ecolor="C1",
            capsize=3,
            label="Additive prediction" if i == 0 else None,
        )
        ax.errorbar(
            observed_mean,
            i,
            xerr=[[observed_mean - observed_low], [observed_high - observed_mean]],
            fmt="o",
            color="C0",
            ecolor="C0",
            capsize=3,
            label="Observed combined" if i == 0 else None,
        )

    ax.axvline(0.0, color="black", lw=1.0, alpha=0.65)
    ax.set_yticks(y)
    ax.set_yticklabels([metric_labels.get(metric, metric) for metric in ordered])
    ax.set_xlabel("Effect vs matched baseline (outbreak-importation baseline)")
    ax.set_title("Combined-shock interaction: observed vs additive")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    return fig


def plot_targeting_comparison(context: ABMChapterContext) -> plt.Figure:
    """Compare norm campaign vs targeted seeding under matched baseline."""

    scenarios = ["norm_campaign", "targeted_norm_seed"]
    baseline = CHAPTER_BASELINE_SCENARIO

    subset = context.series.filter(pl.col("scenario").is_in(scenarios + [baseline]))
    baseline_tick = (
        subset.filter(pl.col("scenario") == baseline)
        .select("seed", "rep", "tick", "vax_willingness", "descriptive_norm")
        .rename(
            {
                "vax_willingness": "base_vax",
                "descriptive_norm": "base_descriptive_norm",
            }
        )
    )

    treated = (
        subset.filter(pl.col("scenario").is_in(scenarios))
        .join(baseline_tick, on=["seed", "rep", "tick"], how="left")
        .with_columns(
            (pl.col("vax_willingness") - pl.col("base_vax")).alias("delta_vax"),
            (pl.col("descriptive_norm") - pl.col("base_descriptive_norm")).alias("delta_descriptive_norm"),
        )
    )

    fig, axes = plt.subplots(2, 1, figsize=(8.3, 6.6), sharex=True)
    tracks = [
        ("delta_descriptive_norm", "Δ descriptive norm"),
        ("delta_vax", "Δ willingness"),
    ]

    for ax, (var, ylabel) in zip(axes, tracks, strict=True):
        for idx, scenario in enumerate(scenarios):
            summary = _trajectory_ci(treated, scenario, var)
            if summary.is_empty():
                continue
            x = np.asarray(summary.get_column("tick").to_list(), dtype=np.float64)
            mean = np.asarray(summary.get_column("mean").to_list(), dtype=np.float64)
            lo = np.asarray(summary.get_column("low").to_list(), dtype=np.float64)
            hi = np.asarray(summary.get_column("high").to_list(), dtype=np.float64)
            color = f"C{idx}"
            ax.plot(x, mean, lw=2.0, color=color, label=label_scenario(scenario))
            ax.fill_between(x, lo, hi, color=color, alpha=0.18)

        ax.axhline(0.0, color="black", lw=1.0, alpha=0.65)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)

    axes[0].legend(frameon=False, loc="best")
    axes[1].set_xlabel("Tick")
    fig.suptitle("Targeting comparison: same norm family, different seeding", y=1.01)
    fig.tight_layout()
    return fig


def plot_auc_leaderboard_horizontal(
    context: ABMChapterContext,
    scenarios: tuple[str, ...] | list[str] = AUC_LEADERBOARD_SCENARIOS,
) -> plt.Figure:
    """Consolidated horizontal bar chart for cumulative AUC effects.

    Shows only cumulative AUC for vaccine willingness and masking,
    sorted from highest to lowest impact.
    """
    summary = summarize_effects(context, scenarios=scenarios)

    # Focus on AUC metrics only
    metrics = [
        ("delta_vax_auc", "Vaccine willingness (cumulative)"),
        ("delta_npi_auc", "NPI index (cumulative)"),
    ]

    # Check available metrics
    available = set(summary.get_column("metric").unique().to_list())
    active_metrics = [(m, label) for m, label in metrics if m in available]

    if not active_metrics:
        fig, ax = plt.subplots(figsize=(8.0, 4.0))
        ax.text(0.5, 0.5, "No AUC metrics available", ha="center", va="center")
        ax.axis("off")
        return fig

    # Each panel has its own ranking, so y-axes must not be shared.
    fig, axes = plt.subplots(1, len(active_metrics), figsize=(5.0 * len(active_metrics), 5.0), sharey=False)
    if len(active_metrics) == 1:
        axes = [axes]

    for ax, (metric, title) in zip(axes, active_metrics, strict=True):
        sub = summary.filter(pl.col("metric") == metric).sort("mean", descending=True)

        if sub.is_empty():
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            continue

        scenario_order = sub.get_column("scenario").to_list()
        labels = [label_scenario(s) for s in scenario_order]
        means = np.asarray(sub.get_column("mean").to_list(), dtype=np.float64)
        q_lows = np.asarray(sub.get_column("ci_low").to_list(), dtype=np.float64)
        q_highs = np.asarray(sub.get_column("ci_high").to_list(), dtype=np.float64)

        y = np.arange(len(scenario_order), dtype=np.float64)
        err_low = means - q_lows
        err_high = q_highs - means

        # Color positive effects blue, negative red
        colors = ["#2166ac" if m >= 0 else "#b2182b" for m in means]

        ax.barh(y, means, color=colors, alpha=0.8, edgecolor="none")
        ax.errorbar(
            means,
            y,
            xerr=np.vstack([err_low, err_high]),
            fmt="none",
            color="0.3",
            elinewidth=1.5,
            capsize=3,
        )

        ax.axvline(0.0, color="black", lw=1.0, alpha=0.7)
        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        ax.set_xlabel("Cumulative effect vs baseline")
        ax.set_title(title)
        ax.grid(axis="x", alpha=0.3)
        ax.invert_yaxis()  # Highest at top

    fig.tight_layout()
    return fig


def plot_mechanism_grid_3x3(
    context: ABMChapterContext,
    scenarios: tuple[str, str, str] = ("outbreak_global", "trust_scandal_repair", "access_intervention"),
    mediators: tuple[str, str, str] = ("danger", "institutional_trust", "low_constraints"),
    outcomes: tuple[str, str, str] = ("mask_pressure", "vax_willingness", "vax_willingness"),
) -> plt.Figure:
    """3x3 mechanism comparison grid.

    Rows: Incidence, Mediating psychological profile variable, behavioural outcome
    Columns: Outbreak scenario, Trust repair scenario, Access scenario
    """
    scenario_labels = [label_scenario(s) for s in scenarios]
    row_labels = ["Incidence", "Psychological profile variable", "Behavioural outcome"]

    fig, axes = plt.subplots(
        3,
        3,
        figsize=(12.0, 9.0),
        sharex=True,
    )

    for col_idx, (scenario, mediator, outcome) in enumerate(zip(scenarios, mediators, outcomes, strict=True)):
        baseline = _scenario_baseline(context, scenario)
        frame = context.series.filter(pl.col("scenario").is_in([scenario, baseline]))

        tracks = [
            ("incidence_mean", "Incidence"),
            (mediator, mediator.replace("_", " ").title()),
            (outcome, outcome.replace("_", " ").title()),
        ]

        for row_idx, (var, _ylabel) in enumerate(tracks):
            ax = axes[row_idx, col_idx]

            for line_idx, name in enumerate([baseline, scenario]):
                summary = _trajectory_ci(frame, name, var)
                if summary.is_empty():
                    continue
                x = np.asarray(summary.get_column("tick").to_list(), dtype=np.float64)
                mean = np.asarray(summary.get_column("mean").to_list(), dtype=np.float64)
                lo = np.asarray(summary.get_column("low").to_list(), dtype=np.float64)
                hi = np.asarray(summary.get_column("high").to_list(), dtype=np.float64)
                color = "0.5" if line_idx == 0 else f"C{col_idx}"
                ls = "--" if line_idx == 0 else "-"
                lw = 1.5 if line_idx == 0 else 2.0
                label = "Baseline" if line_idx == 0 else label_scenario(name)
                ax.plot(x, mean, lw=lw, ls=ls, color=color, label=label)
                ax.fill_between(x, lo, hi, color=color, alpha=0.12)

            ax.grid(alpha=0.25)
            if row_idx == 0:
                ax.set_title(scenario_labels[col_idx], fontsize=11, fontweight="bold")
            if col_idx == 0:
                ax.set_ylabel(row_labels[row_idx])
            if row_idx == 2:
                ax.set_xlabel("Tick")
            if row_idx == 0 and col_idx == 2:
                ax.legend(frameon=False, loc="upper right", fontsize=8)

    fig.suptitle(
        "Mechanism dynamics: outbreak salience vs trust repair vs access removal",
        y=1.01,
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout()
    return fig


def compute_trust_vs_fear_ratio(context: ABMChapterContext) -> dict[str, float]:
    """Compute the Trust > Fear ratio and key effect sizes."""
    effects = summarize_effects(context, scenarios=LEADERBOARD_SCENARIOS)

    def _get_mean(scenario: str, metric: str) -> float:
        row = effects.filter((pl.col("scenario") == scenario) & (pl.col("metric") == metric)).select("mean")
        if row.is_empty():
            return float("nan")
        return float(row.item())

    trust_auc = _get_mean("trust_scandal_repair", "delta_vax_auc")
    outbreak_auc = _get_mean("outbreak_global", "delta_vax_auc")
    access_mask_auc = _get_mean("access_intervention", "delta_mask_auc")

    return {
        "trust_vax_auc": trust_auc,
        "outbreak_vax_auc": outbreak_auc,
        "trust_vs_fear_ratio": trust_auc / outbreak_auc if outbreak_auc != 0 else float("nan"),
        "access_mask_auc": access_mask_auc,
    }


def anchor_effect_variance_table(
    context: ABMChapterContext,
    scenarios: tuple[str, ...] | list[str] = ("access_intervention", "outbreak_global"),
) -> Markdown:
    """Simple variance table for anchor-level heterogeneity (replaces heatmap)."""
    agg = anchor_effects(context, scenarios)
    if agg.is_empty():
        return md_table(["Scenario", "Min effect", "Max effect", "Spread"], [])

    rows: list[list[str]] = []
    for scenario in scenarios:
        sub = agg.filter(pl.col("scenario") == scenario)
        if sub.is_empty():
            continue
        means = sub.get_column("mean")
        min_val = float(means.min()) if means.len() > 0 else float("nan")
        max_val = float(means.max()) if means.len() > 0 else float("nan")
        spread = max_val - min_val

        rows.append(
            [
                label_scenario(scenario),
                f"{min_val:.3f}",
                f"{max_val:.3f}",
                f"{spread:.3f}",
            ]
        )

    return md_table(
        ["Scenario", "Min Δ willingness", "Max Δ willingness", "Spread"],
        rows,
    )


def compute_network_effects(context: ABMChapterContext) -> dict[str, float]:
    """Compute network targeting effects: targeted (high-degree) vs other nodes."""
    targeted = context.grouped_outcomes.filter(pl.col("group_type") == "targeted_flag")
    if targeted.is_empty():
        return {}

    # Get end-of-horizon values per run, then average by group.
    end_vals = (
        targeted.sort("tick")
        .group_by(["scenario", "group_value", "run_id"])
        .agg(pl.col("mean").last().alias("vax_end"))
        .group_by(["scenario", "group_value"])
        .agg(pl.col("vax_end").mean().alias("vax_end"))
    )

    metrics: dict[str, float] = {}
    for scenario in ["outbreak_local", "targeted_norm_seed"]:
        sub = end_vals.filter(pl.col("scenario") == scenario)
        if sub.is_empty():
            continue
        targeted_val = sub.filter(pl.col("group_value") == "targeted").select("vax_end")
        other_val = sub.filter(pl.col("group_value") == "other").select("vax_end")
        if not targeted_val.is_empty() and not other_val.is_empty():
            t_val = float(targeted_val.item())
            o_val = float(other_val.item())
            metrics[f"{scenario}_targeted"] = t_val
            metrics[f"{scenario}_other"] = o_val
            metrics[f"{scenario}_diff"] = t_val - o_val

    return metrics


def _compute_shadow_target_mask(
    agents_t0: pl.DataFrame,
    *,
    seed_rule: str,
    high_degree_target_percentile: float,
    high_exposure_target_percentile: float,
    low_legitimacy_target_percentile: float,
) -> np.ndarray | None:
    """Reconstruct scenario targeting mask from baseline t0 population."""
    n = agents_t0.height
    if n == 0:
        return None

    if seed_rule == "high_degree":
        if "core_degree" not in agents_t0.columns:
            return None
        degree = agents_t0["core_degree"].cast(pl.Float64).to_numpy()
        cutoff = float(np.percentile(degree, high_degree_target_percentile))
        return degree >= cutoff

    if seed_rule in {"high_exposure", "high_degree_high_exposure", "community_targeted"}:
        if "activity_score" not in agents_t0.columns:
            score = np.zeros(n, dtype=np.float64)
        else:
            score = agents_t0["activity_score"].cast(pl.Float64).to_numpy()
        local = np.ones(n, dtype=bool)
        if seed_rule == "community_targeted" and "locality_id" in agents_t0.columns:
            communities = agents_t0["locality_id"].to_numpy().astype(str)
            unique = np.unique(communities)
            means_by_comm: dict[str, float] = {}
            for comm in unique:
                idx = communities == comm
                if not np.any(idx):
                    continue
                means_by_comm[comm] = float(np.nanmean(score[idx]))
            if means_by_comm:
                target_community = max(means_by_comm.items(), key=lambda item: item[1])[0]
                local = communities == target_community
        local_scores = score[local]
        if local_scores.size == 0:
            return None
        threshold = float(np.nanpercentile(local_scores, high_exposure_target_percentile))
        mask = np.zeros(n, dtype=bool)
        mask[local] = score[local] >= threshold
        return mask

    if seed_rule in {"low_legitimacy", "low_legitimacy_high_degree"}:
        legitimacy_score: np.ndarray | None = None
        if "legitimacy_engine" in agents_t0.columns:
            legitimacy_score = agents_t0["legitimacy_engine"].cast(pl.Float64).to_numpy()
        elif "covax_legitimacy_scepticism_idx" in agents_t0.columns:
            scepticism = agents_t0["covax_legitimacy_scepticism_idx"].cast(pl.Float64).to_numpy()
            legitimacy_score = -scepticism
        if legitimacy_score is None:
            return None

        cutoff = float(np.nanpercentile(legitimacy_score, low_legitimacy_target_percentile))
        if not np.isfinite(cutoff):
            return None
        low_legitimacy_mask = legitimacy_score <= cutoff

        if seed_rule == "low_legitimacy":
            return low_legitimacy_mask
        if "core_degree" not in agents_t0.columns:
            return low_legitimacy_mask
        degree = agents_t0["core_degree"].cast(pl.Float64).to_numpy()
        degree_cutoff = float(np.percentile(degree, high_degree_target_percentile))
        return low_legitimacy_mask & (degree >= degree_cutoff)

    return None


def compute_local_diffusion_did(
    context: ABMChapterContext,
    *,
    scenarios: tuple[str, ...] | list[str] = (
        "outbreak_local",
        "misinfo_shock_targeted",
        "targeted_norm_seed",
        "trust_repair_targeted_low_legitimacy",
        "trust_repair_targeted_low_legitimacy_high_degree",
        "trust_repair_targeted_high_exposure",
    ),
    outcomes: tuple[str, ...] | list[str] = (
        "vax_willingness_T12",
        "mask_when_pressure_high",
    ),
) -> pl.DataFrame:
    """Compute local-shock diffusion using shadow-target Diff-in-Diff.

    For each run and scenario:
    DID = [(yT - y0)_targeted - (yT - y0)_other]_scenario
          -[(yT - y0)_targeted - (yT - y0)_other]_baseline_shadow
    """
    run_lookup: dict[tuple[str, int, int], str] = {
        (str(row["scenario"]), int(row["seed"]), int(row["rep"])): str(row["run_dir"])
        for row in context.run_index.iter_rows(named=True)
    }
    rows: list[dict[str, float | int | str]] = []

    for row in context.run_index.iter_rows(named=True):
        scenario = str(row["scenario"])
        if scenario not in scenarios:
            continue
        seed = int(row["seed"])
        rep = int(row["rep"])
        baseline_scenario = _baseline_for_row(str(row["incidence_mode"]), bool(row["communication_enabled"]))
        baseline_run_dir = run_lookup.get((baseline_scenario, seed, rep))
        if baseline_run_dir is None:
            continue

        scenario_run = Path(str(row["run_dir"]))
        baseline_run = Path(baseline_run_dir)
        scenario_meta_path = scenario_run / "metadata.json"
        if not scenario_meta_path.exists():
            continue

        payload = json.loads(scenario_meta_path.read_text(encoding="utf-8"))
        scenario_cfg = payload.get("scenario_config", {})
        sim_cfg = payload.get("simulation_config", {})
        where = str(scenario_cfg.get("where", "global"))
        targeting = str(scenario_cfg.get("targeting", "uniform"))
        seed_rule = str(scenario_cfg.get("seed_rule", "uniform"))
        if where != "local" or targeting != "targeted":
            continue

        constants = sim_cfg.get("constants", {})
        high_degree_target_percentile = float(constants.get("high_degree_target_percentile", 75.0))
        high_exposure_target_percentile = float(constants.get("high_exposure_target_percentile", 75.0))
        low_legitimacy_target_percentile = float(constants.get("low_legitimacy_target_percentile", 25.0))

        s_t0_path = scenario_run / "populations" / "t0.parquet"
        s_t1_path = scenario_run / "populations" / "final.parquet"
        b_t0_path = baseline_run / "populations" / "t0.parquet"
        b_t1_path = baseline_run / "populations" / "final.parquet"
        if not (s_t0_path.exists() and s_t1_path.exists() and b_t0_path.exists() and b_t1_path.exists()):
            continue

        s_t0 = pl.read_parquet(s_t0_path).sort("unique_id")
        s_t1 = pl.read_parquet(s_t1_path).sort("unique_id")
        b_t0 = pl.read_parquet(b_t0_path).sort("unique_id")
        b_t1 = pl.read_parquet(b_t1_path).sort("unique_id")

        mask = _compute_shadow_target_mask(
            b_t0,
            seed_rule=seed_rule,
            high_degree_target_percentile=high_degree_target_percentile,
            high_exposure_target_percentile=high_exposure_target_percentile,
            low_legitimacy_target_percentile=low_legitimacy_target_percentile,
        )
        if mask is None:
            continue
        if int(np.count_nonzero(mask)) == 0 or int(np.count_nonzero(~mask)) == 0:
            continue

        for outcome in outcomes:
            if outcome not in s_t0.columns or outcome not in s_t1.columns:
                continue
            if outcome not in b_t0.columns or outcome not in b_t1.columns:
                continue

            s0 = s_t0[outcome].cast(pl.Float64).to_numpy()
            s1 = s_t1[outcome].cast(pl.Float64).to_numpy()
            b0 = b_t0[outcome].cast(pl.Float64).to_numpy()
            b1 = b_t1[outcome].cast(pl.Float64).to_numpy()

            delta_s = s1 - s0
            delta_b = b1 - b0

            scenario_targeted_delta = float(np.mean(delta_s[mask]))
            scenario_other_delta = float(np.mean(delta_s[~mask]))
            baseline_targeted_delta = float(np.mean(delta_b[mask]))
            baseline_other_delta = float(np.mean(delta_b[~mask]))

            scenario_gap_change = scenario_targeted_delta - scenario_other_delta
            baseline_shadow_gap_change = baseline_targeted_delta - baseline_other_delta
            did = scenario_gap_change - baseline_shadow_gap_change

            rows.append(
                {
                    "scenario": scenario,
                    "baseline_scenario": baseline_scenario,
                    "seed": seed,
                    "rep": rep,
                    "outcome": outcome,
                    "scenario_targeted_delta": scenario_targeted_delta,
                    "scenario_other_delta": scenario_other_delta,
                    "baseline_targeted_delta": baseline_targeted_delta,
                    "baseline_other_delta": baseline_other_delta,
                    "scenario_gap_change": scenario_gap_change,
                    "baseline_shadow_gap_change": baseline_shadow_gap_change,
                    "did": did,
                    "spillover_ratio_scenario": (
                        scenario_other_delta / scenario_targeted_delta
                        if abs(scenario_targeted_delta) > 1e-12
                        else float("nan")
                    ),
                }
            )

    if not rows:
        return pl.DataFrame()
    return pl.DataFrame(rows).sort(["scenario", "outcome", "rep", "seed"])


def summarize_local_diffusion_did(
    context: ABMChapterContext,
    *,
    scenarios: tuple[str, ...] | list[str] = (
        "outbreak_local",
        "misinfo_shock_targeted",
        "targeted_norm_seed",
    ),
    outcomes: tuple[str, ...] | list[str] = (
        "vax_willingness_T12",
        "mask_when_pressure_high",
    ),
) -> pl.DataFrame:
    """Aggregate shadow-target Diff-in-Diff metrics by scenario and outcome."""
    rows = compute_local_diffusion_did(context, scenarios=scenarios, outcomes=outcomes)
    if rows.is_empty():
        return rows
    return (
        rows.group_by(["scenario", "outcome"])
        .agg(
            pl.len().alias("n_runs"),
            pl.col("did").mean().alias("did_mean"),
            pl.col("did").std().alias("did_sd"),
            pl.col("scenario_gap_change").mean().alias("scenario_gap_change_mean"),
            pl.col("baseline_shadow_gap_change").mean().alias("baseline_shadow_gap_change_mean"),
            pl.col("scenario_targeted_delta").mean().alias("scenario_targeted_delta_mean"),
            pl.col("scenario_other_delta").mean().alias("scenario_other_delta_mean"),
            pl.col("spillover_ratio_scenario").mean().alias("spillover_ratio_mean"),
        )
        .sort(["scenario", "outcome"])
    )


def plot_network_effects(context: ABMChapterContext) -> plt.Figure:
    """Bar chart comparing targeted vs other nodes across scenarios."""
    targeted = context.grouped_outcomes.filter(pl.col("group_type") == "targeted_flag")
    if targeted.is_empty():
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No network targeting data available", ha="center", va="center")
        ax.axis("off")
        return fig

    # Get end-of-horizon values by scenario and group
    end_vals = (
        targeted.sort("tick")
        .group_by(["scenario", "group_value", "run_id"])
        .agg(pl.col("mean").last().alias("vax_end"))
    )

    # Aggregate across runs
    summary = (
        end_vals.group_by(["scenario", "group_value"])
        .agg(
            pl.col("vax_end").mean().alias("mean"),
            pl.col("vax_end").std().alias("sd"),
            pl.len().alias("n"),
        )
        .with_columns((1.96 * pl.col("sd") / pl.col("n").cast(pl.Float64).sqrt()).alias("ci_half"))
    )

    scenarios = ["outbreak_local", "targeted_norm_seed"]
    scenarios = [s for s in scenarios if s in summary.get_column("scenario").unique().to_list()]

    if not scenarios:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No relevant network scenarios", ha="center", va="center")
        ax.axis("off")
        return fig

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(scenarios))
    width = 0.35

    for idx, group in enumerate(["other", "targeted"]):
        sub = summary.filter(pl.col("group_value") == group)
        means = []
        errs = []
        for scenario in scenarios:
            row = sub.filter(pl.col("scenario") == scenario)
            if row.is_empty():
                means.append(0)
                errs.append(0)
            else:
                means.append(float(row.get_column("mean").item()))
                errs.append(float(row.get_column("ci_half").item()))

        offset = (idx - 0.5) * width
        label = "High-degree (targeted)" if group == "targeted" else "Other nodes"
        color = "C0" if group == "other" else "C1"
        ax.bar(x + offset, means, width, yerr=errs, label=label, color=color, alpha=0.8, capsize=4)

    ax.set_ylabel("Vaccination willingness (end of horizon)")
    ax.set_xticks(x)
    ax.set_xticklabels([label_scenario(s) for s in scenarios])
    ax.legend(frameon=False, loc="upper right")
    ax.set_title("Network targeting effects: high-degree vs other nodes")
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    return fig


def plot_shock_dynamics_panel(
    context: ABMChapterContext,
    scenarios: tuple[str, ...] = ("misinfo_shock", "outbreak_global", "trust_scandal_repair"),
) -> plt.Figure:
    """Plot shock dynamics showing how different scenarios unfold over time.

    Shows the trajectory of key constructs to reveal temporal dynamics:
    - Fast shocks (outbreak) vs slow buildup (trust)
    - Peak timing and decay patterns
    """
    available = context.series.get_column("scenario").unique().to_list()
    scenarios = [s for s in scenarios if s in available]

    if len(scenarios) < 2:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "Insufficient scenarios for comparison", ha="center", va="center")
        ax.axis("off")
        return fig

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)

    # Compute deltas vs baseline
    baseline = _scenario_baseline(context, "outbreak_global")  # Use importation baseline

    variables = [
        ("vax_willingness", "Vaccine willingness"),
        ("institutional_trust", "Institutional trust"),
        ("danger", "Perceived danger"),
    ]

    for ax, (var, title) in zip(axes, variables, strict=True):
        baseline_traj = _trajectory_ci(context.series, baseline, var)

        for idx, scenario in enumerate(scenarios):
            scenario_traj = _trajectory_ci(context.series, scenario, var)

            if scenario_traj.is_empty() or baseline_traj.is_empty():
                continue

            # Compute delta vs baseline
            merged = scenario_traj.join(
                baseline_traj.select(pl.col("tick"), pl.col("mean").alias("base_mean")),
                on="tick",
                how="inner",
            ).with_columns((pl.col("mean") - pl.col("base_mean")).alias("delta"))

            x = np.asarray(merged.get_column("tick").to_list(), dtype=np.float64)
            delta = np.asarray(merged.get_column("delta").to_list(), dtype=np.float64)

            color = f"C{idx}"
            ax.plot(x, delta, lw=2, color=color, label=label_scenario(scenario))

        ax.axhline(0, color="black", lw=0.8, ls="--", alpha=0.5)
        ax.set_title(title)
        ax.set_xlabel("Tick")
        ax.grid(alpha=0.25)

    axes[0].set_ylabel("Δ vs baseline")
    axes[2].legend(frameon=False, loc="upper right", fontsize=8)

    fig.suptitle("Shock dynamics: differential trajectories vs baseline", y=1.02)
    fig.tight_layout()
    return fig


def _burn_in_tick_for_scenario(context: ABMChapterContext, scenario: str) -> int:
    runs = context.run_index.filter(pl.col("scenario") == scenario)
    if runs.is_empty() or "burn_in_ticks" not in runs.columns:
        return 0
    return int(
        runs.group_by("burn_in_ticks")
        .agg(pl.len().alias("n"))
        .sort("n", descending=True)
        .head(1)
        .get_column("burn_in_ticks")
        .item()
    )


def _delta_trajectory_ci(context: ABMChapterContext, scenario: str, variable: str) -> pl.DataFrame:
    if variable not in context.series.columns:
        return pl.DataFrame({"tick": [], "mean": [], "low": [], "high": []})

    baseline = _scenario_baseline(context, scenario)
    scenario_ticks = context.series.filter(pl.col("scenario") == scenario).select("seed", "rep", "tick", variable)
    baseline_ticks = context.series.filter(pl.col("scenario") == baseline).select(
        "seed",
        "rep",
        "tick",
        pl.col(variable).alias("base_value"),
    )

    joined = scenario_ticks.join(
        baseline_ticks,
        on=["seed", "rep", "tick"],
        how="inner",
    )
    if joined.is_empty():
        return pl.DataFrame({"tick": [], "mean": [], "low": [], "high": []})

    return (
        joined.with_columns((pl.col(variable) - pl.col("base_value")).alias("delta"))
        .group_by("tick")
        .agg(
            pl.col("delta").mean().alias("mean"),
            pl.col("delta").quantile(0.05).alias("low"),
            pl.col("delta").quantile(0.95).alias("high"),
        )
        .sort("tick")
    )


def plot_mechanism_impulse_response_panel(
    context: ABMChapterContext,
    scenarios: tuple[str, ...] = ("trust_repair_institutional", "outbreak_global_incidence_strong"),
    signals: tuple[tuple[str, str], ...] = (
        ("incidence_mean", "Incidence (locality mean)"),
        ("fear", "Disease fear / perceived danger"),
        ("institutional_trust", "Institutional trust"),
        ("vax_willingness", "Vaccine willingness"),
        ("mask_pressure", "Masking (mask pressure)"),
        ("descriptive_norm", "Descriptive vax norm"),
    ),
) -> plt.Figure:
    available = set(context.series.get_column("scenario").unique().to_list())
    active_scenarios = [scenario for scenario in scenarios if scenario in available]
    if not active_scenarios:
        fig, ax = plt.subplots(figsize=(7, 3.5))
        ax.text(0.5, 0.5, "No scenarios available for impulse-response panel", ha="center", va="center")
        ax.axis("off")
        return fig

    n_rows = len(active_scenarios)
    n_cols = len(signals)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(3.1 * n_cols, 2.5 * n_rows),
        sharex=True,
    )
    if n_rows == 1:
        axes = np.asarray([axes])

    for row_idx, scenario in enumerate(active_scenarios):
        for col_idx, (variable, panel_title) in enumerate(signals):
            ax = axes[row_idx, col_idx]
            summary = _delta_trajectory_ci(context, scenario, variable)
            if summary.is_empty():
                ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=8)
                ax.axis("off")
                continue

            x = np.asarray(summary.get_column("tick").to_list(), dtype=np.float64)
            mean = np.asarray(summary.get_column("mean").to_list(), dtype=np.float64)
            low = np.asarray(summary.get_column("low").to_list(), dtype=np.float64)
            high = np.asarray(summary.get_column("high").to_list(), dtype=np.float64)
            color = f"C{row_idx}"

            ax.plot(x, mean, lw=2.0, color=color)
            ax.fill_between(x, low, high, color=color, alpha=0.16)
            ax.axhline(0.0, color="black", lw=0.8, ls="--", alpha=0.65)
            ax.grid(alpha=0.25)

            if row_idx == 0:
                ax.set_title(panel_title, fontsize=9.5)
            if col_idx == 0:
                ax.set_ylabel(f"{label_scenario(scenario)}\nΔ vs baseline")
            if row_idx == n_rows - 1:
                ax.set_xlabel("Tick")

    fig.suptitle("Paired impulse-response panel: trust repair vs outbreak forcing", y=1.01)
    fig.tight_layout()
    return fig


def plot_dynamic_fingerprints_panel(
    context: ABMChapterContext,
    rows: tuple[tuple[str, str, str, str], ...] = (
        (
            "norm_shift_campaign",
            "Norm shift campaign (C)",
            "descriptive_norm",
            "Norms",
        ),
        (
            "outbreak_global_incidence_strong",
            "Outbreak wave (S)",
            "fear",
            "Perceived Danger",
        ),
        (
            "misinfo_trust_erosion_only",
            "Misinfo trust erosion (P)",
            "institutional_trust",
            "Institutional trust",
        ),
    ),
) -> plt.Figure:
    """3x3 dynamic response signatures grid for key policy levers.

    Rows map to intervention levers. Columns are fixed to:
    - mediator targeted by the lever
    - delta vaccine willingness vs matched baseline
    - delta NPI index vs matched baseline
    """
    available = set(context.series.get_column("scenario").unique().to_list())
    active_rows = [row for row in rows if row[0] in available]
    if not active_rows:
        fig, ax = plt.subplots(figsize=(8, 3.5))
        ax.text(0.5, 0.5, "No scenarios available for dynamic-response-signatures panel", ha="center", va="center")
        ax.axis("off")
        return fig

    n_rows = len(active_rows)
    fig, axes = plt.subplots(n_rows, 3, figsize=(13.2, 3.0 * n_rows), sharex=False)
    if n_rows == 1:
        axes = np.asarray([axes])

    panel_titles = (
        "Mediator construct",
        "Δ vaccine willingness vs baseline",
        "Δ NPI index vs baseline",
    )

    # Precompute summaries so we can apply shared y-limits within behavioural columns.
    cache: dict[tuple[str, str], pl.DataFrame] = {}
    for scenario, _, mediator_var, _ in active_rows:
        for variable in (mediator_var, "vax_willingness", "npi_index"):
            summary = _delta_trajectory_ci(context, scenario, variable)
            burn_in_tick = _burn_in_tick_for_scenario(context, scenario)
            cache[(scenario, variable)] = summary.filter(pl.col("tick") >= burn_in_tick).sort("tick")

    def _column_limits(variable: str) -> tuple[float, float] | None:
        vals: list[np.ndarray] = []
        for scenario, _, _, _ in active_rows:
            summary = cache.get((scenario, variable))
            if summary is None or summary.is_empty():
                continue
            mean = np.asarray(summary.get_column("mean").to_list(), dtype=np.float64)
            low = np.asarray(summary.get_column("low").to_list(), dtype=np.float64)
            high = np.asarray(summary.get_column("high").to_list(), dtype=np.float64)
            if mean.size:
                vals.extend([mean, low, high])
        if not vals:
            return None
        joined = np.concatenate(vals)
        joined = joined[np.isfinite(joined)]
        if joined.size == 0:
            return None
        lo = float(np.min(joined))
        hi = float(np.max(joined))
        span = hi - lo
        if span <= 1e-12:
            span = max(abs(hi), 1.0) * 0.2
            return lo - span, hi + span
        pad = 0.08 * span
        return lo - pad, hi + pad

    vax_lims = _column_limits("vax_willingness")
    npi_lims = _column_limits("npi_index")

    for row_idx, (scenario, row_label, mediator_var, mediator_label) in enumerate(active_rows):
        tracks = (
            (mediator_var, mediator_label),
            ("vax_willingness", "Vaccine willingness"),
            ("npi_index", "NPI index"),
        )

        for col_idx, (variable, fallback_label) in enumerate(tracks):
            ax = axes[row_idx, col_idx]
            summary = cache.get((scenario, variable))

            if summary is None or summary.is_empty():
                ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=8.8)
                ax.axis("off")
                continue

            x = np.asarray(summary.get_column("tick").to_list(), dtype=np.float64)
            mean = np.asarray(summary.get_column("mean").to_list(), dtype=np.float64)
            low = np.asarray(summary.get_column("low").to_list(), dtype=np.float64)
            high = np.asarray(summary.get_column("high").to_list(), dtype=np.float64)
            color = f"C{row_idx}"

            ax.plot(x, mean, lw=2.0, color=color)
            ax.fill_between(x, low, high, color=color, alpha=0.16)
            ax.axhline(0.0, color="black", lw=0.85, ls="--", alpha=0.7)
            ax.grid(alpha=0.25)
            ax.tick_params(axis="both", labelsize=8.8)

            if col_idx == 0:
                ax.set_title(f"Mediator construct\n({fallback_label})", fontsize=10.2)
            elif row_idx == 0:
                ax.set_title(panel_titles[col_idx], fontsize=10.2)
            if col_idx == 0:
                ax.set_ylabel(f"{row_label}\nΔ vs baseline", fontsize=9.6)
            if row_idx == n_rows - 1:
                ax.set_xlabel("Month (tick)", fontsize=9.6)

            if col_idx == 1 and vax_lims is not None:
                ax.set_ylim(vax_lims)
            if col_idx == 2 and npi_lims is not None:
                ax.set_ylim(npi_lims)

    fig.tight_layout(pad=0.6, w_pad=0.8, h_pad=1.0)
    return fig


def export_dynamic_fingerprints_presentation_assets(
    context: ABMChapterContext,
    *,
    output_dir: str | Path = Path("../thesis/presentation/assets"),
    stem: str = "fig-05-dynamic-fingerprints",
    dpi: int = 220,
    highlight_color: str = "#d63031",
    highlight_linewidth: float = 2.3,
    highlight_pad_x: float = 0.004,
    highlight_pad_y: float = 0.006,
) -> list[Path]:
    """Export base and stepped presentation assets for the dynamic fingerprints panel.

    The stepped variants are generated from the same Matplotlib figure by adding
    a row-level rectangle in figure coordinates, so the highlight aligns exactly
    with the underlying subplot geometry.
    """
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    fig = plot_dynamic_fingerprints_panel(context)
    axes = [ax for ax in fig.axes if ax.get_visible()]
    if len(axes) < 9:
        raise RuntimeError(f"Expected at least 9 visible axes for the dynamic fingerprints panel, found {len(axes)}.")

    step_paths = [
        output_root / f"{stem}.png",
        output_root / f"{stem}-step0.png",
        output_root / f"{stem}-step1.png",
        output_root / f"{stem}-step2.png",
        output_root / f"{stem}-step3.png",
    ]

    save_kwargs = {"dpi": dpi, "facecolor": "white", "bbox_inches": "tight", "pad_inches": 0.02}

    # Base full figure and step0 are identical unhighlighted variants.
    fig.savefig(step_paths[0], **save_kwargs)
    fig.savefig(step_paths[1], **save_kwargs)

    row_rectangles: list[Rectangle] = []
    for row_idx in range(3):
        row_axes = axes[row_idx * 3 : (row_idx + 1) * 3]
        x0 = min(ax.get_position().x0 for ax in row_axes) - highlight_pad_x
        y0 = min(ax.get_position().y0 for ax in row_axes) - highlight_pad_y
        x1 = max(ax.get_position().x1 for ax in row_axes) + highlight_pad_x
        y1 = max(ax.get_position().y1 for ax in row_axes) + highlight_pad_y
        row_rectangles.append(
            Rectangle(
                (x0, y0),
                x1 - x0,
                y1 - y0,
                transform=fig.transFigure,
                fill=False,
                edgecolor=highlight_color,
                linewidth=highlight_linewidth,
                zorder=1000,
                clip_on=False,
                joinstyle="round",
            )
        )

    exported = step_paths[:2]
    for row_idx, rect in enumerate(row_rectangles, start=1):
        fig.add_artist(rect)
        out_path = output_root / f"{stem}-step{row_idx}.png"
        fig.savefig(out_path, **save_kwargs)
        rect.remove()
        exported.append(out_path)

    plt.close(fig)
    return exported


def _corr_safe(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 3 or y.size < 3:
        return float("nan")
    mask = np.isfinite(x) & np.isfinite(y)
    if int(np.sum(mask)) < 3:
        return float("nan")
    x_valid = x[mask]
    y_valid = y[mask]
    if float(np.std(x_valid)) < 1e-12 or float(np.std(y_valid)) < 1e-12:
        return float("nan")
    return float(np.corrcoef(x_valid, y_valid)[0, 1])


def _slope_safe(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 3 or y.size < 3:
        return float("nan")
    mask = np.isfinite(x) & np.isfinite(y)
    if int(np.sum(mask)) < 3:
        return float("nan")
    x_valid = x[mask]
    y_valid = y[mask]
    x_var = float(np.var(x_valid))
    if x_var < 1e-12:
        return float("nan")
    cov = float(np.mean((x_valid - np.mean(x_valid)) * (y_valid - np.mean(y_valid))))
    return cov / x_var


def _peak_tick_and_value(ticks: np.ndarray, values: np.ndarray) -> tuple[float, float]:
    if ticks.size == 0 or values.size == 0 or np.all(~np.isfinite(values)):
        return float("nan"), float("nan")
    idx = int(np.nanargmax(np.abs(values)))
    return float(ticks[idx]), float(values[idx])


def _persistence_share(values: np.ndarray, threshold: float) -> float:
    if values.size == 0 or np.all(~np.isfinite(values)):
        return float("nan")
    idx = int(np.nanargmax(np.abs(values)))
    peak_abs = abs(float(values[idx]))
    if peak_abs <= 1e-12:
        return float("nan")
    tail = np.abs(values[idx:])
    return float(np.mean(tail >= threshold * peak_abs))


def _intended_channel_spec(scenario: str) -> tuple[str, str, str]:
    scenario_key = str(scenario)
    explicit: dict[str, tuple[str, str, str]] = {
        "trust_repair_institutional": ("trust", "institutional_trust", "vax_willingness"),
        "trust_scandal_institutional": ("trust", "institutional_trust", "vax_willingness"),
        "misinfo_trust_erosion_only": ("trust", "institutional_trust", "vax_willingness"),
        "norm_shift_campaign": ("norm", "descriptive_norm", "vax_willingness"),
        "access_intervention": ("constraints", "low_constraints", "vax_willingness"),
        "access_barrier_shock": ("constraints", "low_constraints", "vax_willingness"),
        "outbreak_global_incidence_strong": ("fear", "fear", "mask_pressure"),
        "outbreak_salience_global": ("fear", "fear", "mask_pressure"),
    }
    if scenario_key in explicit:
        return explicit[scenario_key]
    if "outbreak" in scenario_key:
        return ("fear", "fear", "mask_pressure")
    if "norm" in scenario_key:
        return ("norm", "descriptive_norm", "vax_willingness")
    if "access" in scenario_key or "constraint" in scenario_key:
        return ("constraints", "low_constraints", "vax_willingness")
    return ("trust", "institutional_trust", "vax_willingness")


def compute_shock_coupling_diagnostics(
    context: ABMChapterContext,
    scenarios: tuple[str, ...] | list[str] = AUC_LEADERBOARD_SCENARIOS,
    persistence_threshold: float = 0.25,
    lag_min_amplitude: float = 0.005,
    lag_min_relative_amplitude: float = 0.4,
) -> pl.DataFrame:
    """Compute coupling diagnostics for shock trajectories.

    Returns one row per scenario with:
    - paired AUC signs (vaccination vs NPI)
    - intended mediator channel assignment
    - behaviour co-movement sign (reinforcing/substitution)
    - peak timing and persistence diagnostics
    """
    available = set(context.series.get_column("scenario").unique().to_list())
    rows: list[dict[str, object]] = []

    for scenario in scenarios:
        if scenario not in available:
            continue

        burn_in_tick = _burn_in_tick_for_scenario(context, scenario)
        series_by_var: dict[str, tuple[np.ndarray, np.ndarray]] = {}

        for variable in (
            "vax_willingness",
            "npi_index",
            "mask_pressure",
            "institutional_trust",
            "fear",
            "descriptive_norm",
            "low_constraints",
        ):
            summary = _delta_trajectory_ci(context, scenario, variable)
            if summary.is_empty():
                series_by_var[variable] = (
                    np.asarray([], dtype=np.float64),
                    np.asarray([], dtype=np.float64),
                )
                continue
            post = summary.filter(pl.col("tick") >= burn_in_tick).sort("tick")
            ticks = np.asarray(post.get_column("tick").to_list(), dtype=np.float64)
            values = np.asarray(post.get_column("mean").to_list(), dtype=np.float64)
            series_by_var[variable] = ticks, values

        vax_ticks, vax_vals = series_by_var["vax_willingness"]
        npi_ticks, npi_vals = series_by_var["npi_index"]
        mask_ticks, mask_vals = series_by_var["mask_pressure"]
        _, trust_vals = series_by_var["institutional_trust"]
        _, fear_vals = series_by_var["fear"]
        _, norm_vals = series_by_var["descriptive_norm"]
        _, constraints_vals = series_by_var["low_constraints"]
        channel_name, channel_mediator, channel_outcome = _intended_channel_spec(scenario)
        _, intended_mediator_vals = series_by_var.get(
            channel_mediator,
            (np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)),
        )
        _, intended_outcome_vals = series_by_var.get(
            channel_outcome,
            (np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)),
        )

        vax_peak_tick, vax_peak_delta = _peak_tick_and_value(vax_ticks, vax_vals)
        npi_peak_tick, npi_peak_delta = _peak_tick_and_value(npi_ticks, npi_vals)
        mask_peak_tick, mask_peak_delta = _peak_tick_and_value(mask_ticks, mask_vals)

        trust_vax_corr = _corr_safe(trust_vals, vax_vals)
        fear_npi_corr = _corr_safe(fear_vals, npi_vals)
        fear_mask_corr = _corr_safe(fear_vals, mask_vals)
        norm_vax_corr = _corr_safe(norm_vals, vax_vals)
        constraints_vax_corr = _corr_safe(constraints_vals, vax_vals)
        vax_npi_corr = _corr_safe(vax_vals, npi_vals)
        vax_mask_corr = _corr_safe(vax_vals, mask_vals)
        trust_vax_slope = _slope_safe(trust_vals, vax_vals)
        fear_npi_slope = _slope_safe(fear_vals, npi_vals)
        fear_mask_slope = _slope_safe(fear_vals, mask_vals)
        norm_vax_slope = _slope_safe(norm_vals, vax_vals)
        constraints_vax_slope = _slope_safe(constraints_vals, vax_vals)
        vax_npi_slope = _slope_safe(vax_vals, npi_vals)
        vax_mask_slope = _slope_safe(vax_vals, mask_vals)
        intended_channel_corr = _corr_safe(intended_mediator_vals, intended_outcome_vals)
        intended_channel_slope = _slope_safe(intended_mediator_vals, intended_outcome_vals)
        intended_channel_beta_std = intended_channel_corr

        vax_auc = float(np.nansum(vax_vals)) if vax_vals.size else float("nan")
        npi_auc = float(np.nansum(npi_vals)) if npi_vals.size else float("nan")
        mask_auc = float(np.nansum(mask_vals)) if mask_vals.size else float("nan")
        vax_peak_abs = float(np.nanmax(np.abs(vax_vals))) if vax_vals.size else float("nan")
        npi_peak_abs = float(np.nanmax(np.abs(npi_vals))) if npi_vals.size else float("nan")
        mask_peak_abs = float(np.nanmax(np.abs(mask_vals))) if mask_vals.size else float("nan")
        peak_balance = (
            min(vax_peak_abs, npi_peak_abs) / max(vax_peak_abs, npi_peak_abs)
            if np.isfinite(vax_peak_abs) and np.isfinite(npi_peak_abs) and max(vax_peak_abs, npi_peak_abs) > 0.0
            else float("nan")
        )
        lag_identifiable = (
            np.isfinite(vax_peak_abs)
            and np.isfinite(npi_peak_abs)
            and np.isfinite(peak_balance)
            and vax_peak_abs >= lag_min_amplitude
            and npi_peak_abs >= lag_min_amplitude
            and peak_balance >= lag_min_relative_amplitude
        )
        if np.isfinite(vax_npi_corr) and vax_npi_corr > 0.0:
            coupling_type = "reinforcing"
        elif np.isfinite(vax_npi_corr) and vax_npi_corr < 0.0:
            coupling_type = "substitution"
        else:
            coupling_type = "undetermined"

        rows.append(
            {
                "scenario": scenario,
                "scenario_label": label_scenario(scenario),
                "intended_channel": channel_name,
                "intended_mediator": channel_mediator,
                "intended_outcome": channel_outcome,
                "coupling_type": coupling_type,
                "intended_channel_corr": intended_channel_corr,
                "intended_channel_slope": intended_channel_slope,
                "intended_channel_beta_std": intended_channel_beta_std,
                "vax_auc": vax_auc,
                "npi_auc": npi_auc,
                "mask_auc": mask_auc,
                "vax_npi_corr": vax_npi_corr,
                "vax_npi_slope": vax_npi_slope,
                "vax_mask_corr": vax_mask_corr,
                "vax_mask_slope": vax_mask_slope,
                "trust_vax_corr": trust_vax_corr,
                "trust_vax_slope": trust_vax_slope,
                "fear_npi_corr": fear_npi_corr,
                "fear_npi_slope": fear_npi_slope,
                "fear_mask_corr": fear_mask_corr,
                "fear_mask_slope": fear_mask_slope,
                "norm_vax_corr": norm_vax_corr,
                "norm_vax_slope": norm_vax_slope,
                "constraints_vax_corr": constraints_vax_corr,
                "constraints_vax_slope": constraints_vax_slope,
                "vax_peak_tick": vax_peak_tick,
                "npi_peak_tick": npi_peak_tick,
                "mask_peak_tick": mask_peak_tick,
                "peak_lag_npi_minus_vax": (
                    float(npi_peak_tick - vax_peak_tick)
                    if lag_identifiable and np.isfinite(npi_peak_tick) and np.isfinite(vax_peak_tick)
                    else float("nan")
                ),
                "peak_lag_mask_minus_vax": (
                    float(mask_peak_tick - vax_peak_tick)
                    if lag_identifiable and np.isfinite(mask_peak_tick) and np.isfinite(vax_peak_tick)
                    else float("nan")
                ),
                "vax_peak_delta": vax_peak_delta,
                "npi_peak_delta": npi_peak_delta,
                "mask_peak_delta": mask_peak_delta,
                "vax_peak_abs": vax_peak_abs,
                "npi_peak_abs": npi_peak_abs,
                "mask_peak_abs": mask_peak_abs,
                "peak_balance": peak_balance,
                "lag_identifiable": bool(lag_identifiable),
                "lag_min_amplitude": float(lag_min_amplitude),
                "lag_min_relative_amplitude": float(lag_min_relative_amplitude),
                "vax_persistence": _persistence_share(vax_vals, persistence_threshold),
                "mask_persistence": _persistence_share(mask_vals, persistence_threshold),
                "persistence_threshold": float(persistence_threshold),
            }
        )

    if not rows:
        return pl.DataFrame()

    return (
        pl.DataFrame(rows)
        .with_columns((pl.col("vax_auc").abs() + pl.col("mask_auc").abs()).alias("_effect_magnitude"))
        .sort("_effect_magnitude", descending=True)
        .drop("_effect_magnitude")
    )


def plot_shock_coupling_trajectories(context: ABMChapterContext) -> plt.Figure:
    """State-space trajectories highlighting reinforcement and substitution."""
    panel_specs: tuple[tuple[str, str, str, tuple[str, ...]], ...] = (
        (
            "Trust reinforcement",
            "institutional_trust",
            "vax_willingness",
            ("trust_repair_institutional",),
        ),
        (
            "Fear reinforcement",
            "fear",
            "mask_pressure",
            ("outbreak_global_incidence_strong",),
        ),
        (
            "Access substitution",
            "vax_willingness",
            "mask_pressure",
            ("access_intervention", "access_barrier_shock"),
        ),
    )
    pretty_label = {
        "institutional_trust": "institutional trust",
        "vax_willingness": "vaccination willingness",
        "fear": "disease fear",
        "mask_pressure": "masking",
    }
    available = set(context.series.get_column("scenario").unique().to_list())

    fig, axes = plt.subplots(1, len(panel_specs), figsize=(13.2, 4.1))
    if len(panel_specs) == 1:
        axes = np.asarray([axes])

    for ax, (title, x_var, y_var, scenarios) in zip(axes, panel_specs, strict=True):
        any_drawn = False
        for idx, scenario in enumerate(scenarios):
            if scenario not in available:
                continue

            x_summary = _delta_trajectory_ci(context, scenario, x_var)
            y_summary = _delta_trajectory_ci(context, scenario, y_var)
            if x_summary.is_empty() or y_summary.is_empty():
                continue

            burn_in_tick = _burn_in_tick_for_scenario(context, scenario)
            merged = (
                x_summary.select("tick", pl.col("mean").alias("x"))
                .join(
                    y_summary.select("tick", pl.col("mean").alias("y")),
                    on="tick",
                    how="inner",
                )
                .filter(pl.col("tick") >= burn_in_tick)
                .sort("tick")
            )
            if merged.is_empty():
                continue

            x = np.asarray(merged.get_column("x").to_list(), dtype=np.float64)
            y = np.asarray(merged.get_column("y").to_list(), dtype=np.float64)
            color = f"C{idx}"

            ax.plot(x, y, lw=2.0, color=color, label=label_scenario(scenario))
            ax.scatter([x[0]], [y[0]], color=color, s=26, marker="o", alpha=0.85)
            ax.scatter([x[-1]], [y[-1]], color=color, s=30, marker="s", alpha=0.9)
            any_drawn = True

        if not any_drawn:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            ax.axis("off")
            continue

        ax.axhline(0.0, color="black", lw=0.8, ls="--", alpha=0.65)
        ax.axvline(0.0, color="black", lw=0.8, ls="--", alpha=0.65)
        ax.grid(alpha=0.25)
        ax.set_title(title)
        ax.set_xlabel(f"Δ {pretty_label.get(x_var, x_var)}")
        ax.set_ylabel(f"Δ {pretty_label.get(y_var, y_var)}")
        ax.legend(frameon=False, fontsize=8, loc="best")

    fig.suptitle("Shock trajectories in psychological-profile space (post-burn, mean deltas)", y=1.01)
    fig.tight_layout()
    return fig


def _best_future_delta_lag(source: np.ndarray, target: np.ndarray, *, max_lag: int = 24) -> tuple[float, float]:
    if source.size < 3 or target.size < 3:
        return float("nan"), float("nan")

    d_source = np.diff(source)
    d_target = np.diff(target)
    best_lag = float("nan")
    best_corr = float("-inf")

    for lag in range(0, max_lag + 1):
        if lag == 0:
            x = d_source
            y = d_target
        else:
            if d_source.size <= lag or d_target.size <= lag:
                continue
            x = d_source[:-lag]
            y = d_target[lag:]

        mask = np.isfinite(x) & np.isfinite(y)
        if int(np.sum(mask)) < 3:
            continue

        x_valid = x[mask]
        y_valid = y[mask]
        if float(np.std(x_valid)) < 1e-12 or float(np.std(y_valid)) < 1e-12:
            continue

        corr = float(np.corrcoef(x_valid, y_valid)[0, 1])
        if corr > best_corr:
            best_corr = corr
            best_lag = float(lag)

    if best_corr == float("-inf"):
        return float("nan"), float("nan")
    return best_lag, best_corr


def compute_temporal_fingerprint_metrics(
    context: ABMChapterContext,
    scenarios: tuple[str, ...] = (
        "trust_repair_institutional",
        "outbreak_global_incidence_strong",
        "norm_shift_campaign",
    ),
    persistence_threshold: float = 0.25,
) -> pl.DataFrame:
    signals: tuple[tuple[str, str, str], ...] = (
        ("incidence_mean", "Incidence", "incidence"),
        ("fear", "Fear", "fear"),
        ("institutional_trust", "Trust", "trust"),
        ("descriptive_norm", "Norms", "norms"),
        ("vax_willingness", "Vaccine", "vax"),
        ("mask_pressure", "Mask", "mask"),
    )

    available = set(context.series.get_column("scenario").unique().to_list())
    rows: list[dict[str, object]] = []
    for scenario in scenarios:
        if scenario not in available:
            continue
        burn_in_tick = _burn_in_tick_for_scenario(context, scenario)

        for variable, signal_label, signal_type in signals:
            summary = _delta_trajectory_ci(context, scenario, variable)
            if summary.is_empty():
                continue

            post = summary.filter(pl.col("tick") >= burn_in_tick).sort("tick")
            if post.is_empty():
                continue

            ticks = np.asarray(post.get_column("tick").to_list(), dtype=np.float64)
            values = np.asarray(post.get_column("mean").to_list(), dtype=np.float64)
            if values.size == 0 or np.all(~np.isfinite(values)):
                continue

            peak_idx = int(np.nanargmax(np.abs(values)))
            peak_tick = float(ticks[peak_idx])
            peak_delta = float(values[peak_idx])
            peak_abs = abs(peak_delta)
            if peak_abs <= 1e-12:
                persistence = float("nan")
            else:
                persistence = float(np.mean(np.abs(values[peak_idx:]) >= persistence_threshold * peak_abs))

            rows.append(
                {
                    "scenario": scenario,
                    "scenario_label": label_scenario(scenario),
                    "signal": variable,
                    "signal_label": signal_label,
                    "signal_type": signal_type,
                    "peak_tick": peak_tick,
                    "peak_delta": peak_delta,
                    "persistence": persistence,
                    "persistence_threshold": persistence_threshold,
                }
            )

    return pl.DataFrame(rows)


def compute_temporal_lag_signatures(context: ABMChapterContext) -> dict[str, float]:
    def _mean_delta_values(scenario: str, variable: str) -> np.ndarray:
        summary = _delta_trajectory_ci(context, scenario, variable)
        if summary.is_empty():
            return np.asarray([], dtype=np.float64)
        burn_in_tick = _burn_in_tick_for_scenario(context, scenario)
        post = summary.filter(pl.col("tick") >= burn_in_tick).sort("tick")
        return np.asarray(post.get_column("mean").to_list(), dtype=np.float64)

    outbreak = "outbreak_global_incidence_strong"
    trust_repair = "trust_repair_institutional"
    norm_shift = "norm_shift_campaign"

    fear_vals = _mean_delta_values(outbreak, "fear")
    mask_vals = _mean_delta_values(outbreak, "mask_pressure")
    trust_vals = _mean_delta_values(trust_repair, "institutional_trust")
    vax_vals = _mean_delta_values(trust_repair, "vax_willingness")
    norm_vals = _mean_delta_values(norm_shift, "descriptive_norm")
    vax_norm_vals = _mean_delta_values(norm_shift, "vax_willingness")

    fear_mask_lag, fear_mask_corr = _best_future_delta_lag(fear_vals, mask_vals, max_lag=24)
    trust_vax_lag, trust_vax_corr = _best_future_delta_lag(trust_vals, vax_vals, max_lag=24)
    norm_vax_lag, norm_vax_corr = _best_future_delta_lag(norm_vals, vax_norm_vals, max_lag=24)

    return {
        "fear_to_mask_lag": fear_mask_lag,
        "fear_to_mask_corr": fear_mask_corr,
        "trust_to_vax_lag": trust_vax_lag,
        "trust_to_vax_corr": trust_vax_corr,
        "norm_to_vax_lag": norm_vax_lag,
        "norm_to_vax_corr": norm_vax_corr,
    }


def plot_temporal_fingerprint_heatmaps(
    context: ABMChapterContext,
    scenarios: tuple[str, ...] = (
        "trust_repair_institutional",
        "outbreak_global_incidence_strong",
        "norm_shift_campaign",
    ),
    persistence_threshold: float = 0.25,
) -> plt.Figure:
    summary = compute_temporal_fingerprint_metrics(
        context,
        scenarios=scenarios,
        persistence_threshold=persistence_threshold,
    )
    if summary.is_empty():
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.text(0.5, 0.5, "No temporal-fingerprint data available", ha="center", va="center")
        ax.axis("off")
        return fig

    signal_labels = {
        "incidence": "Incidence",
        "fear": "Fear",
        "trust": "Trust",
        "norms": "Norms",
        "vax": "Vaccine",
        "mask": "Mask",
    }
    signal_order = ["incidence", "fear", "trust", "norms", "vax", "mask"]
    row_labels = [signal_labels[sig] for sig in signal_order]
    scenario_order = [
        scenario for scenario in scenarios if scenario in summary.get_column("scenario").unique().to_list()
    ]
    scenario_labels = [label_scenario(scenario) for scenario in scenario_order]
    n_rows = len(row_labels)
    n_cols = len(scenario_order)
    peak_data = np.full((n_rows, n_cols), np.nan, dtype=np.float64)
    persistence_data = np.full((n_rows, n_cols), np.nan, dtype=np.float64)
    signal_index = {signal: idx for idx, signal in enumerate(signal_order)}
    scenario_index = {scenario: idx for idx, scenario in enumerate(scenario_order)}

    for row in summary.iter_rows(named=True):
        signal = str(row["signal_type"])
        scenario = str(row["scenario"])
        if signal not in signal_index or scenario not in scenario_index:
            continue
        i = signal_index[signal]
        j = scenario_index[scenario]
        peak_data[i, j] = float(row["peak_tick"])
        persistence_data[i, j] = float(row["persistence"])

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.9), constrained_layout=True)
    peak_img = axes[0].imshow(np.ma.masked_invalid(peak_data), aspect="auto", cmap="cividis")
    persistence_img = axes[1].imshow(
        np.ma.masked_invalid(persistence_data),
        aspect="auto",
        cmap="cividis",
        vmin=0.0,
        vmax=1.0,
    )

    def _format_cell(value: float, decimals: int) -> str:
        if not np.isfinite(value):
            return "NA"
        return f"{value:.{decimals}f}"

    for ax, values, title, decimals in [
        (axes[0], peak_data, "Peak timing (ticks since shock onset)", 0),
        (axes[1], persistence_data, "Persistence", 2),
    ]:
        ax.set_title(title)
        ax.set_xticks(np.arange(n_cols))
        ax.set_xticklabels(scenario_labels, rotation=18, ha="right")
        ax.set_yticks(np.arange(n_rows))
        ax.set_yticklabels(row_labels)
        ax.set_xlabel("Scenario")
        for i in range(n_rows):
            for j in range(n_cols):
                ax.text(
                    j,
                    i,
                    _format_cell(values[i, j], decimals=decimals),
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="black",
                )
        ax.set_xticks(np.arange(-0.5, n_cols, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
        ax.grid(which="minor", color="white", linestyle="-", linewidth=0.8, alpha=0.9)
        ax.tick_params(which="minor", bottom=False, left=False)

    fig.colorbar(peak_img, ax=axes[0], shrink=0.9, label="Ticks")
    fig.colorbar(persistence_img, ax=axes[1], shrink=0.9, label="Share")
    return fig


def plot_temporal_fingerprint_scatter(
    context: ABMChapterContext,
    scenarios: tuple[str, ...] = (
        "trust_repair_institutional",
        "outbreak_global_incidence_strong",
        "norm_shift_campaign",
    ),
    persistence_threshold: float = 0.25,
) -> plt.Figure:
    """Backward-compatible wrapper for legacy call sites."""
    return plot_temporal_fingerprint_heatmaps(
        context=context,
        scenarios=scenarios,
        persistence_threshold=persistence_threshold,
    )


def compute_phase_inversion_case(
    context: ABMChapterContext,
    scenarios: tuple[str, ...] | list[str] | None = None,
    variable: str = "stay_home_symptomatic",
) -> dict[str, object]:
    if variable not in context.series.columns:
        return {}

    if scenarios is None:
        baseline_set = {
            "baseline",
            "baseline_importation",
            "baseline_no_communication",
            "baseline_importation_no_communication",
            "baseline_no_observed_norms",
            "baseline_importation_no_observed_norms",
            "baseline_no_peer_pull",
            "baseline_importation_no_peer_pull",
            "social_only",
        }
        scenarios = sorted(s for s in context.series.get_column("scenario").unique().to_list() if s not in baseline_set)

    candidates: list[dict[str, object]] = []
    for scenario in scenarios:
        summary = _delta_trajectory_ci(context, str(scenario), variable)
        if summary.is_empty():
            continue

        burn_in_tick = _burn_in_tick_for_scenario(context, str(scenario))
        post = summary.filter(pl.col("tick") >= burn_in_tick).sort("tick")
        if post.is_empty():
            continue

        ticks = np.asarray(post.get_column("tick").to_list(), dtype=np.float64)
        deltas = np.asarray(post.get_column("mean").to_list(), dtype=np.float64)
        if deltas.size < 3 or np.all(~np.isfinite(deltas)):
            continue

        endpoint = float(deltas[-1])
        auc = float(np.nansum(deltas))
        if endpoint <= 0.0 or auc >= 0.0:
            continue

        trough_idx = int(np.nanargmin(deltas))
        peak_idx = int(np.nanargmax(deltas))
        candidates.append(
            {
                "scenario": str(scenario),
                "scenario_label": label_scenario(str(scenario)),
                "tick": ticks,
                "delta": deltas,
                "running_auc": np.nancumsum(deltas),
                "endpoint": endpoint,
                "auc": auc,
                "trough": float(deltas[trough_idx]),
                "trough_tick": float(ticks[trough_idx]),
                "peak": float(deltas[peak_idx]),
                "peak_tick": float(ticks[peak_idx]),
                "burn_in_tick": float(burn_in_tick),
            }
        )

    if not candidates:
        return {}
    return max(candidates, key=lambda item: abs(float(item["auc"])))


def plot_phase_inversion_case(
    context: ABMChapterContext,
    scenarios: tuple[str, ...] | list[str] | None = None,
    variable: str = "stay_home_symptomatic",
) -> plt.Figure:
    case = compute_phase_inversion_case(context, scenarios=scenarios, variable=variable)
    if not case:
        fig, ax = plt.subplots(figsize=(7.2, 3.8))
        ax.text(0.5, 0.5, "No phase-inversion case found in loaded scenarios", ha="center", va="center")
        ax.axis("off")
        return fig

    x = np.asarray(case["tick"], dtype=np.float64)
    delta = np.asarray(case["delta"], dtype=np.float64)
    running_auc = np.asarray(case["running_auc"], dtype=np.float64)

    fig, axes = plt.subplots(2, 1, figsize=(8.2, 5.4), sharex=True)
    axes[0].plot(x, delta, color="C3", lw=2.0)
    axes[0].axhline(0.0, color="black", lw=0.8, ls="--", alpha=0.65)
    axes[0].set_ylabel("Δ stay-home when symptomatic")
    axes[0].grid(alpha=0.25)

    axes[1].plot(x, running_auc, color="C0", lw=2.0)
    axes[1].axhline(0.0, color="black", lw=0.8, ls="--", alpha=0.65)
    axes[1].set_ylabel("Running ΔAUC")
    axes[1].set_xlabel("Tick")
    axes[1].grid(alpha=0.25)

    axes[0].set_title(str(case["scenario_label"]))
    fig.suptitle("AUC vs endpoint mismatch: phase inversion case", y=1.01)
    fig.tight_layout()
    return fig


def compute_mechanism_summary(context: ABMChapterContext) -> dict[str, dict[str, float]]:
    """Compute summary statistics for mechanism dynamics explanation."""
    results = {}

    for scenario in ["outbreak_global", "trust_scandal_repair", "access_intervention"]:
        baseline = _scenario_baseline(context, scenario)
        frame = context.series

        scenario_data = frame.filter(pl.col("scenario") == scenario)
        baseline_data = frame.filter(pl.col("scenario") == baseline)

        if scenario_data.is_empty() or baseline_data.is_empty():
            continue

        # Get scenario trajectory stats
        scenario_agg = scenario_data.group_by("tick").agg(
            pl.col("vax_willingness").mean().alias("vax_mean"),
            pl.col("mask_pressure").mean().alias("mask_mean"),
            pl.col("institutional_trust").mean().alias("trust_mean"),
            pl.col("danger").mean().alias("danger_mean"),
        )

        baseline_agg = baseline_data.group_by("tick").agg(
            pl.col("vax_willingness").mean().alias("vax_mean"),
            pl.col("mask_pressure").mean().alias("mask_mean"),
        )

        # Compute peak differences
        joined = scenario_agg.join(
            baseline_agg.select(
                pl.col("tick"),
                pl.col("vax_mean").alias("base_vax"),
                pl.col("mask_mean").alias("base_mask"),
            ),
            on="tick",
            how="inner",
        ).with_columns(
            (pl.col("vax_mean") - pl.col("base_vax")).alias("delta_vax"),
            (pl.col("mask_mean") - pl.col("base_mask")).alias("delta_mask"),
        )

        if joined.is_empty():
            continue

        # Find peak time
        max_vax_row = joined.sort("delta_vax", descending=True).head(1)
        max_mask_row = joined.sort("delta_mask", descending=True).head(1)

        results[scenario] = {
            "peak_vax_tick": float(max_vax_row.get_column("tick").item()),
            "peak_vax_delta": float(max_vax_row.get_column("delta_vax").item()),
            "peak_mask_tick": float(max_mask_row.get_column("tick").item()),
            "peak_mask_delta": float(max_mask_row.get_column("delta_mask").item()),
            "end_vax_delta": float(joined.sort("tick").tail(1).get_column("delta_vax").item()),
            "end_mask_delta": float(joined.sort("tick").tail(1).get_column("delta_mask").item()),
        }

    return results


def load_k_social_sensitivity() -> pl.DataFrame | None:
    """Load k_social sensitivity analysis results if available."""
    candidates = (
        ABM_ARTIFACT_ROOT / "k_social_sensitivity" / "sensitivity_ates.csv",
        Path("thesis/artifacts/abm/k_social_sensitivity/sensitivity_ates.csv"),
    )
    for path in candidates:
        if path.exists():
            return pl.read_csv(path)
    return None


def compute_k_social_sensitivity_metrics() -> dict[str, float]:
    """Compute summary metrics from k_social sensitivity analysis.

    Returns
    -------
    dict with keys:
        misinfo_ate_k0: ATE for misinfo at k_social=0
        misinfo_ate_k05: ATE for misinfo at k_social=0.5
        misinfo_ate_k1: ATE for misinfo at k_social=1.0
        misinfo_ate_range: max - min ATE for misinfo
        nocomm_ate_k0: ATE for no-comm at k_social=0
        nocomm_ate_k05: ATE for no-comm at k_social=0.5
        nocomm_ate_k1: ATE for no-comm at k_social=1.0
        nocomm_ate_range: max - min ATE for no-comm
    """
    df = load_k_social_sensitivity()
    if df is None:
        return {
            k: float("nan")
            for k in [
                "misinfo_ate_k0",
                "misinfo_ate_k05",
                "misinfo_ate_k1",
                "misinfo_ate_range",
                "nocomm_ate_k0",
                "nocomm_ate_k05",
                "nocomm_ate_k1",
                "nocomm_ate_range",
            ]
        }

    results: dict[str, float] = {}

    for scenario, prefix in [("misinfo_shock", "misinfo"), ("baseline_no_communication", "nocomm")]:
        scen_df = df.filter(pl.col("scenario") == scenario)
        if scen_df.is_empty():
            for k in [0.0, 0.5, 1.0]:
                results[f"{prefix}_ate_k{str(k).replace('.', '')}"] = float("nan")
            results[f"{prefix}_ate_range"] = float("nan")
            continue

        for k in [0.0, 0.5, 1.0]:
            row = scen_df.filter(pl.col("k_social") == k)
            key = f"{prefix}_ate_k{int(k * 10) if k != 0 else 0}"
            if k == 0.5:
                key = f"{prefix}_ate_k05"
            elif k == 1.0:
                key = f"{prefix}_ate_k1"
            else:
                key = f"{prefix}_ate_k0"

            if row.is_empty():
                results[key] = float("nan")
            else:
                results[key] = float(row.get_column("ate_vax_willingness").item())

        ates = scen_df.get_column("ate_vax_willingness")
        results[f"{prefix}_ate_range"] = float(ates.max() - ates.min())

    return results


def plot_k_social_sensitivity() -> plt.Figure:
    """Plot ATEs across k_social values for key scenarios."""
    df = load_k_social_sensitivity()
    if df is None:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "Sensitivity data not available", ha="center", va="center", fontsize=12)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        return fig

    fig, ax = plt.subplots(figsize=(8, 4))

    scenarios = {
        "misinfo_shock": ("Misinformation shock", "C3"),
        "baseline_no_communication": ("Communication disabled", "C1"),
    }

    for scenario, (label, color) in scenarios.items():
        scen_df = df.filter(pl.col("scenario") == scenario).sort("k_social")
        if scen_df.is_empty():
            continue

        k_vals = np.asarray(scen_df.get_column("k_social").to_list(), dtype=np.float64)
        ate_vals = np.asarray(scen_df.get_column("ate_vax_willingness").to_list(), dtype=np.float64)

        ax.plot(k_vals, ate_vals, "o-", color=color, label=label, lw=2, markersize=8)

    ax.axhline(0.0, color="black", lw=1.0, ls="--", alpha=0.5)
    ax.axvline(0.5, color="gray", lw=1.0, ls=":", alpha=0.5, label="Default k_social")
    ax.set_xlabel("k_social (social feedback magnitude)")
    ax.set_ylabel("ATE on vaccine willingness (vs baseline)")
    ax.set_title("Parameter Sensitivity: Effect of k_social on Scenario ATEs")
    ax.legend(frameon=False, loc="best")
    ax.grid(alpha=0.25)

    fig.tight_layout()
    return fig


def resolve_full_sensitivity_root(explicit_root: str | Path | None = None) -> Path:
    """Resolve the staged global-sensitivity output directory."""

    if explicit_root is not None:
        root = Path(explicit_root)
        if not root.exists():
            raise FileNotFoundError(f"Full sensitivity output does not exist: {root!s}")
        return root

    candidates = (
        ABM_ARTIFACT_ROOT / "full_sensitivity_pipeline",
        Path("thesis/artifacts/abm/full_sensitivity_pipeline"),
    )
    for root in candidates:
        if (root / "sobol" / "indices.csv").exists() or (root / "morris" / "indices_per_scenario.csv").exists():
            return root

    raise FileNotFoundError("No full sensitivity pipeline output found under thesis/artifacts/abm/.")


@lru_cache(maxsize=2)
def load_full_sensitivity_sobol(explicit_root: str | None = None) -> pl.DataFrame | None:
    """Load total-order Sobol indices from the staged sensitivity pipeline."""

    try:
        root = resolve_full_sensitivity_root(explicit_root=explicit_root)
    except FileNotFoundError:
        return None
    return pl.read_csv(root / "sobol" / "indices.csv")


@lru_cache(maxsize=2)
def load_full_sensitivity_morris(explicit_root: str | None = None) -> pl.DataFrame | None:
    """Load per-scenario Morris indices from the staged sensitivity pipeline."""

    try:
        root = resolve_full_sensitivity_root(explicit_root=explicit_root)
    except FileNotFoundError:
        return None

    indices_path = root / "morris" / "indices_per_scenario.csv"
    if not indices_path.exists():
        return None
    return pl.read_csv(indices_path)


@lru_cache(maxsize=2)
def load_full_sensitivity_lhc_delta(explicit_root: str | None = None) -> pl.DataFrame | None:
    """Load LHC delta outputs from the staged sensitivity pipeline."""

    try:
        root = resolve_full_sensitivity_root(explicit_root=explicit_root)
    except FileNotFoundError:
        return None

    delta_path = root / "lhc" / "delta.parquet"
    if not delta_path.exists():
        return None
    return pl.read_parquet(delta_path)


def _scenario_duration_months_for_sensitivity(scenario: str) -> float:
    """Return scenario duration in months for duration-normalised claims."""

    try:
        cfg = get_scenario(scenario)
    except KeyError:
        return float("nan")

    duration = int(getattr(cfg, "duration", 0))
    if duration > 0:
        return float(duration)

    incidence_schedule = tuple(getattr(cfg, "incidence_schedule", ()) or ())
    schedule_durations: list[int] = []
    for spec in incidence_schedule:
        start = int(getattr(spec, "start_tick", 0))
        end = getattr(spec, "end_tick", None)
        if end is None:
            continue
        schedule_durations.append(max(int(end) - start, 1))

    if schedule_durations:
        return float(max(schedule_durations))
    return float("nan")


def _resolve_sensitivity_pair(
    frame: pl.DataFrame,
    *,
    left_scenario: str,
    right_scenario: str,
    output: str,
) -> tuple[str, str] | None:
    """Resolve a comparable pair of sensitivity scenarios for one output."""

    available_pairs = {
        (str(row["scenario"]), str(row["output"]))
        for row in frame.select("scenario", "output").unique().iter_rows(named=True)
    }
    left_base = _normalize_flagship_morris_scenario_name(left_scenario)
    right_base = _normalize_flagship_morris_scenario_name(right_scenario)

    for regime in SENSITIVITY_PREFERRED_REGIMES:
        left_label = f"{left_base}__{regime}"
        right_label = f"{right_base}__{regime}"
        if (left_label, output) in available_pairs and (right_label, output) in available_pairs:
            return left_label, right_label

    return None


def pairwise_rank_robustness_table(explicit_root: str | None = None) -> str:
    """Generate pairwise rank-robustness claims from LHC sensitivity draws."""

    delta_df = load_full_sensitivity_lhc_delta(explicit_root=explicit_root)
    if delta_df is None or delta_df.is_empty():
        return "_Full sensitivity LHC outputs not available._"

    rows: list[list[str]] = []
    for claim in PAIRWISE_ROBUSTNESS_CLAIMS:
        label = str(claim["label"])
        left_scenario = str(claim["left_scenario"])
        right_scenario = str(claim["right_scenario"])
        output = str(claim["output"])
        value_column = str(claim.get("value_column", "delta_auc"))
        comparison = str(claim.get("comparison", "gt"))
        normalize_duration = bool(claim.get("normalize_duration", False))

        resolved_pair = _resolve_sensitivity_pair(
            delta_df,
            left_scenario=left_scenario,
            right_scenario=right_scenario,
            output=output,
        )
        if resolved_pair is None:
            rows.append([label, "N/A"])
            continue
        left_run_label, right_run_label = resolved_pair

        subset = delta_df.filter(
            pl.col("scenario").is_in([left_run_label, right_run_label]) & (pl.col("output") == output)
        )

        left_df = subset.filter(pl.col("scenario") == left_run_label).select(
            "point_id",
            pl.col(value_column).cast(pl.Float64).alias("left_value"),
        )
        right_df = subset.filter(pl.col("scenario") == right_run_label).select(
            "point_id",
            pl.col(value_column).cast(pl.Float64).alias("right_value"),
        )

        paired = left_df.join(right_df, on="point_id", how="inner")
        if paired.is_empty():
            rows.append([label, "N/A"])
            continue

        if normalize_duration:
            left_duration = _scenario_duration_months_for_sensitivity(left_scenario)
            right_duration = _scenario_duration_months_for_sensitivity(right_scenario)
            if not (
                np.isfinite(left_duration)
                and np.isfinite(right_duration)
                and left_duration > 0.0
                and right_duration > 0.0
            ):
                rows.append([label, "N/A"])
                continue
            paired = paired.with_columns(
                (pl.col("left_value") / left_duration).alias("left_value"),
                (pl.col("right_value") / right_duration).alias("right_value"),
            )

        paired = paired.drop_nulls(["left_value", "right_value"])
        if paired.is_empty():
            rows.append([label, "N/A"])
            continue

        if comparison == "lt":
            claim_holds = paired.select((pl.col("left_value") < pl.col("right_value")).mean()).item()
        else:
            claim_holds = paired.select((pl.col("left_value") > pl.col("right_value")).mean()).item()

        if claim_holds is None:
            rows.append([label, "N/A"])
            continue
        rows.append([label, f"{float(claim_holds):.2f}"])

    return md_table(
        headers=[
            "Comparison",
            "Fraction of sampled parameter space where claim holds",
        ],
        rows=rows,
    )


def _label_flagship_morris_row(scenario: str, output: str) -> str:
    """Return a compact row label for the flagship Morris heatmap."""

    custom = FLAGSHIP_MORRIS_ROW_LABELS.get((scenario, output))
    if custom is not None:
        return custom

    output_label = {
        "vax_willingness_T12_mean": "Vaccination willingness",
        "mask_when_pressure_high_mean": "NPI index",
        "social_norms_descriptive_vax_mean": "Descriptive norm",
    }.get(output, output)
    return f"{label_scenario(scenario)}\n{output_label}"


def _normalize_flagship_morris_scenario_name(scenario: str) -> str:
    """Map legacy thesis scenario names to current sensitivity labels."""

    return FLAGSHIP_MORRIS_SCENARIO_ALIASES.get(scenario, scenario)


def _resolve_flagship_morris_focus(
    morris_df: pl.DataFrame,
) -> list[tuple[str, str, str]]:
    """Resolve flagship focus pairs against retained Morris outputs."""

    available_pairs = {
        (str(row["scenario"]), str(row["output"]))
        for row in morris_df.select("scenario", "output").unique().iter_rows(named=True)
    }

    resolved: list[tuple[str, str, str]] = []
    for focus_scenario, output in FLAGSHIP_MORRIS_FOCUS:
        base_scenario = _normalize_flagship_morris_scenario_name(focus_scenario)
        for regime in SENSITIVITY_PREFERRED_REGIMES:
            scenario_label = f"{base_scenario}__{regime}"
            if (scenario_label, output) in available_pairs:
                resolved.append((focus_scenario, scenario_label, output))
                break

    return resolved


def _morris_family_for_parameter(parameter: str) -> str | None:
    """Map a Morris parameter to a broad mechanism family."""

    key = str(parameter).strip()
    if key == "state_update_noise_std":
        return "Stochasticity"
    if key.startswith("state_relaxation_"):
        return "Persistence"
    if key in {"k_social", "bridge_prob", "norm_threshold_min", "norm_threshold_max"}:
        return "Social diffusion"
    if (
        key.startswith("incidence_")
        or key.startswith("importation_")
        or key in {"community_incidence_baseline", "k_incidence", "k_feedback_stakes", "stakes_to_infection_multiplier"}
    ):
        return "Incidence loop"
    return None


def plot_flagship_morris_heatmap() -> plt.Figure:
    """Plot a family-aggregated Morris heatmap for flagship scenario-endpoint pairs."""

    morris_df = load_full_sensitivity_morris()
    if morris_df is None or morris_df.is_empty():
        fig, ax = plt.subplots(figsize=(10, 4.5))
        ax.text(
            0.5,
            0.5,
            "Full sensitivity pipeline outputs not available",
            ha="center",
            va="center",
            fontsize=12,
        )
        ax.set_axis_off()
        return fig

    family_df = (
        morris_df.with_columns(
            pl.col("parameter").map_elements(_morris_family_for_parameter, return_dtype=pl.String).alias("family")
        )
        .drop_nulls(["family", "mu_star"])
        .group_by("scenario", "output", "family")
        .agg(pl.col("mu_star").sum().alias("mu_star_family"))
    )

    available_focus = [
        pair
        for pair in _resolve_flagship_morris_focus(family_df)
        if not family_df.filter((pl.col("scenario") == pair[1]) & (pl.col("output") == pair[2])).is_empty()
    ]
    if not available_focus:
        fig, ax = plt.subplots(figsize=(10, 4.5))
        ax.text(
            0.5,
            0.5,
            "No retained flagship Morris summaries found",
            ha="center",
            va="center",
            fontsize=12,
        )
        ax.set_axis_off()
        return fig

    lookup = {
        (str(row["scenario"]), str(row["output"]), str(row["family"])): float(row["mu_star_family"])
        for row in family_df.iter_rows(named=True)
    }
    data = np.full((len(available_focus), len(FLAGSHIP_MORRIS_FAMILY_ORDER)), np.nan, dtype=np.float64)
    row_labels: list[str] = []
    for row_idx, (label_scenario_name, morris_scenario_name, output) in enumerate(available_focus):
        row_labels.append(_label_flagship_morris_row(label_scenario_name, output))
        row_total = 0.0
        for col_idx, family in enumerate(FLAGSHIP_MORRIS_FAMILY_ORDER):
            value = lookup.get((morris_scenario_name, output, family), 0.0)
            data[row_idx, col_idx] = value
            row_total += value
        if row_total > 0.0 and np.isfinite(row_total):
            data[row_idx, :] = data[row_idx, :] / row_total

    fig_height = max(4.8, 1.4 + 0.75 * len(available_focus))
    fig, ax = plt.subplots(figsize=(8.2, fig_height))

    masked = np.ma.masked_invalid(data)
    finite_vals = np.asarray(masked.compressed(), dtype=np.float64)
    if finite_vals.size:
        vmin = float(np.min(finite_vals))
        vmax = float(np.max(finite_vals))
        if not (np.isfinite(vmin) and np.isfinite(vmax)) or vmin >= vmax:
            vmin -= 0.01
            vmax += 0.01
    else:
        vmin, vmax = 0.0, 1.0
    text_threshold = (vmin + vmax) / 2.0
    image = ax.imshow(masked, aspect="auto", cmap="cividis", vmin=vmin, vmax=vmax)

    for row_idx in range(data.shape[0]):
        for col_idx in range(data.shape[1]):
            value = data[row_idx, col_idx]
            label = "--" if not np.isfinite(value) else f"{value:.2f}"
            color = "white" if np.isfinite(value) and value <= text_threshold else "black"
            ax.text(
                col_idx,
                row_idx,
                label,
                ha="center",
                va="center",
                fontsize=8.5,
                color=color,
                fontweight="bold" if np.isfinite(value) and value >= 0.5 else None,
            )

    ax.set_xticks(np.arange(len(FLAGSHIP_MORRIS_FAMILY_ORDER)))
    ax.set_xticklabels([FLAGSHIP_MORRIS_FAMILY_LABELS[name] for name in FLAGSHIP_MORRIS_FAMILY_ORDER])
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.set_xlabel("Mechanism family")
    ax.set_ylabel("Scenario-outcome pair")
    ax.tick_params(axis="x", pad=10)
    ax.set_facecolor("#f2f2f2")
    for boundary in np.arange(0.5, len(FLAGSHIP_MORRIS_FAMILY_ORDER) - 0.5, 1.0):
        ax.axvline(boundary, color="white", lw=2.0, alpha=0.85)

    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Row-normalised Morris $\\mu^*$ share")

    fig.tight_layout()
    return fig


def load_comprehensive_sensitivity() -> tuple[pl.DataFrame | None, pl.DataFrame | None]:
    """Load comprehensive OAT and 2D sensitivity results.

    Returns
    -------
    tuple of (oat_df, grid2d_df), each may be None if unavailable.
    """
    oat_candidates = (
        ABM_ARTIFACT_ROOT / "comprehensive_sensitivity" / "oat_results.parquet",
        Path("thesis/artifacts/abm/comprehensive_sensitivity/oat_results.parquet"),
    )
    grid_candidates = (
        ABM_ARTIFACT_ROOT / "comprehensive_sensitivity" / "2d_results.parquet",
        Path("thesis/artifacts/abm/comprehensive_sensitivity/2d_results.parquet"),
    )

    oat_df = None
    for path in oat_candidates:
        if path.exists():
            oat_df = pl.read_parquet(path)
            break

    grid_df = None
    for path in grid_candidates:
        if path.exists():
            grid_df = pl.read_parquet(path)
            break

    return oat_df, grid_df


def compute_comprehensive_sensitivity_metrics() -> dict[str, float]:
    """Compute summary metrics from comprehensive sensitivity analysis.

    Returns dictionary with ATE ranges for each parameter:
        {param}_misinfo_ate_min, {param}_misinfo_ate_max, {param}_misinfo_ate_range
        {param}_nocomm_ate_min, {param}_nocomm_ate_max, {param}_nocomm_ate_range
    """
    oat_df, _ = load_comprehensive_sensitivity()
    if oat_df is None:
        return {}

    results: dict[str, float] = {}
    param_col_map = {
        "k_social": "k_social",
        "k_incidence": "k_incidence",
        "incidence_gamma": "incidence_gamma",
        "k_feedback_stakes": "k_feedback_stakes",
        "contact_rate_multiplier": "contact_rate_multiplier",
        "norm_response_slope": "norm_response_slope",
        "norm_threshold_beta_alpha": "norm_threshold_beta_alpha",
        "state_relaxation_slow": "state_relaxation_slow",
        "state_relaxation_fast": "state_relaxation_fast",
        "bridge_prob": "bridge_prob",
    }

    for param, col in param_col_map.items():
        param_df = oat_df.filter(pl.col("varied_param") == param)
        if param_df.is_empty():
            continue

        baseline = param_df.filter(pl.col("scenario") == "baseline_importation")

        for scenario, prefix in [("misinfo_shock", "misinfo"), ("baseline_no_communication", "nocomm")]:
            scen_df = param_df.filter(pl.col("scenario") == scenario)
            if scen_df.is_empty() or baseline.is_empty():
                continue

            # Group by param value
            bl_means = baseline.group_by(col).agg(pl.col("vax_willingness_t24").mean().alias("bl_vax"))
            sc_means = scen_df.group_by(col).agg(pl.col("vax_willingness_t24").mean().alias("sc_vax"))
            merged = bl_means.join(sc_means, on=col)
            merged = merged.with_columns((pl.col("sc_vax") - pl.col("bl_vax")).alias("ate"))

            if merged.is_empty():
                continue

            ate_min = float(merged["ate"].min())
            ate_max = float(merged["ate"].max())
            ate_range = ate_max - ate_min

            results[f"{param}_{prefix}_ate_min"] = ate_min
            results[f"{param}_{prefix}_ate_max"] = ate_max
            results[f"{param}_{prefix}_ate_range"] = ate_range

    return results


def plot_comprehensive_sensitivity() -> plt.Figure:
    """Plot OAT sensitivity: ATE ranges by parameter for both scenarios."""
    oat_df, _ = load_comprehensive_sensitivity()
    if oat_df is None:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.text(0.5, 0.5, "Comprehensive sensitivity data not available", ha="center", va="center", fontsize=12)
        return fig

    # Compute ATE ranges for each parameter
    param_order = [
        "state_relaxation_slow",
        "k_social",
        "k_feedback_stakes",
        "k_incidence",
        "incidence_gamma",
        "contact_rate_multiplier",
        "state_relaxation_fast",
        "bridge_prob",
        "norm_response_slope",
        "norm_threshold_beta_alpha",
    ]
    param_labels = {
        "k_social": "k_social\n(social feedback)",
        "k_incidence": "k_incidence\n(incidence growth)",
        "incidence_gamma": "incidence_gamma\n(incidence memory)",
        "k_feedback_stakes": "k_feedback_stakes\n(stakes feedback)",
        "contact_rate_multiplier": "contact_rate_multiplier\n(contact intensity)",
        "norm_response_slope": "norm_response_slope\n(logistic steepness)",
        "norm_threshold_beta_alpha": "norm_threshold_beta_alpha\n(theta heterogeneity)",
        "state_relaxation_slow": "state_relaxation_slow\n(trust/norm decay)",
        "state_relaxation_fast": "state_relaxation_fast\n(fast construct decay)",
        "bridge_prob": "bridge_prob\n(network bridging)",
    }

    misinfo_ranges = []
    nocomm_ranges = []
    misinfo_centers = []
    nocomm_centers = []

    for param in param_order:
        param_df = oat_df.filter(pl.col("varied_param") == param)
        if param_df.is_empty():
            misinfo_ranges.append((0, 0))
            nocomm_ranges.append((0, 0))
            misinfo_centers.append(0)
            nocomm_centers.append(0)
            continue

        baseline = param_df.filter(pl.col("scenario") == "baseline_importation")

        for scenario, ranges, centers in [
            ("misinfo_shock", misinfo_ranges, misinfo_centers),
            ("baseline_no_communication", nocomm_ranges, nocomm_centers),
        ]:
            scen_df = param_df.filter(pl.col("scenario") == scenario)
            if scen_df.is_empty() or baseline.is_empty():
                ranges.append((0, 0))
                centers.append(0)
                continue

            bl_means = baseline.group_by(param).agg(pl.col("vax_willingness_t24").mean().alias("bl_vax"))
            sc_means = scen_df.group_by(param).agg(pl.col("vax_willingness_t24").mean().alias("sc_vax"))
            merged = bl_means.join(sc_means, on=param)
            merged = merged.with_columns((pl.col("sc_vax") - pl.col("bl_vax")).alias("ate"))

            ate_min = float(merged["ate"].min())
            ate_max = float(merged["ate"].max())
            ranges.append((ate_min, ate_max))
            centers.append((ate_min + ate_max) / 2)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    y_pos = np.arange(len(param_order))
    # bar_height = 0.35  # unused, retained for reference

    # Misinfo shock panel
    ax = axes[0]
    for i, ((ate_min, ate_max), _param) in enumerate(zip(misinfo_ranges, param_order, strict=False)):
        range_width = ate_max - ate_min
        ax.barh(i, range_width, left=ate_min, height=0.6, color="C3", alpha=0.7, edgecolor="C3")
        ax.plot([ate_min, ate_max], [i, i], "|-", color="C3", lw=2, markersize=10)

    ax.axvline(0.0, color="black", lw=1.0, ls="--", alpha=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([param_labels.get(p, p) for p in param_order])
    ax.set_xlabel("ATE on vaccine willingness")
    ax.set_title("Misinformation shock", fontweight="bold")
    ax.grid(alpha=0.25, axis="x")

    # No-communication panel
    ax = axes[1]
    for i, ((ate_min, ate_max), _param) in enumerate(zip(nocomm_ranges, param_order, strict=False)):
        range_width = ate_max - ate_min
        ax.barh(i, range_width, left=ate_min, height=0.6, color="C1", alpha=0.7, edgecolor="C1")
        ax.plot([ate_min, ate_max], [i, i], "|-", color="C1", lw=2, markersize=10)

    ax.axvline(0.0, color="black", lw=1.0, ls="--", alpha=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([])
    ax.set_xlabel("ATE on vaccine willingness")
    ax.set_title("Communication disabled", fontweight="bold")
    ax.grid(alpha=0.25, axis="x")

    fig.suptitle("Parameter Sensitivity: ATE Ranges Across Hand-Coded Parameters", fontweight="bold", fontsize=12)
    fig.tight_layout()
    return fig


def sensitivity_summary_table() -> str:
    """Generate markdown table summarizing parameter sensitivity."""
    metrics = compute_comprehensive_sensitivity_metrics()
    if not metrics:
        return "_Comprehensive sensitivity data not available._"

    param_order = [
        "state_relaxation_slow",
        "k_social",
        "k_feedback_stakes",
        "k_incidence",
        "incidence_gamma",
        "contact_rate_multiplier",
        "state_relaxation_fast",
        "bridge_prob",
        "norm_response_slope",
        "norm_threshold_beta_alpha",
    ]
    param_labels = {
        "k_social": "k_social",
        "k_incidence": "k_incidence",
        "incidence_gamma": "incidence_gamma",
        "k_feedback_stakes": "k_feedback_stakes",
        "contact_rate_multiplier": "contact_rate_multiplier",
        "norm_response_slope": "norm_response_slope",
        "norm_threshold_beta_alpha": "norm_threshold_beta_alpha",
        "state_relaxation_slow": "state_relaxation_slow",
        "state_relaxation_fast": "state_relaxation_fast",
        "bridge_prob": "bridge_prob",
    }

    rows = []
    for param in param_order:
        misinfo_range = metrics.get(f"{param}_misinfo_ate_range", float("nan"))
        nocomm_range = metrics.get(f"{param}_nocomm_ate_range", float("nan"))
        misinfo_min = metrics.get(f"{param}_misinfo_ate_min", float("nan"))
        misinfo_max = metrics.get(f"{param}_misinfo_ate_max", float("nan"))
        nocomm_min = metrics.get(f"{param}_nocomm_ate_min", float("nan"))
        nocomm_max = metrics.get(f"{param}_nocomm_ate_max", float("nan"))

        rows.append(
            [
                param_labels.get(param, param),
                f"[{misinfo_min:.3f}, {misinfo_max:.3f}]",
                f"{misinfo_range:.4f}",
                f"[{nocomm_min:.3f}, {nocomm_max:.3f}]",
                f"{nocomm_range:.4f}",
            ]
        )

    return md_table(
        headers=["Parameter", "Misinfo ATE range", "Spread", "No-comm ATE range", "Spread"],
        rows=rows,
    )
