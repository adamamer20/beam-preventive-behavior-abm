"""Core benchmarking contracts, helpers, and baseline gates."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import polars as pl
from loguru import logger
from scipy.spatial.distance import jensenshannon

from beam_abm.abm.data_contracts import (
    AnchorSystem,
    BackboneModel,
    LeversConfig,
    get_abm_levers_config,
)
from beam_abm.abm.outcome_scales import OUTCOME_BOUNDS, normalize_outcome_to_unit_interval
from beam_abm.abm.outcomes import predict_all_outcomes
from beam_abm.abm.state_schema import CONSTRUCT_BOUNDS, MUTABLE_COLUMNS, OUTCOME_COLUMNS

from . import non_degeneracy as _non_degeneracy
from . import population_shift as _population_shift

build_non_degeneracy_report = _non_degeneracy.build_non_degeneracy_report
run_population_shift_validation_gate = _population_shift.run_population_shift_validation_gate

CheckDetails = dict[str, float | int | str | bool | None]


@dataclass(slots=True)
class GateCheck:
    """Single gate check outcome."""

    name: str
    passed: bool
    details: CheckDetails = field(default_factory=dict)


@dataclass(slots=True)
class GateResult:
    """Result of a validation gate suite."""

    gate_name: str = ""
    passed: bool = False
    checks: list[GateCheck] = field(default_factory=list)

    def add_check(
        self,
        name: str,
        passed: bool,
        details: CheckDetails | None = None,
    ) -> None:
        self.checks.append(GateCheck(name=name, passed=passed, details=details or {}))

    def evaluate(self) -> bool:
        self.passed = all(c.passed for c in self.checks)
        return self.passed

    def to_dict(self) -> dict[str, object]:
        return {
            "gate_name": self.gate_name,
            "passed": self.passed,
            "n_checks": len(self.checks),
            "n_passed": sum(1 for c in self.checks if c.passed),
            "checks": [{"name": c.name, "passed": c.passed, "details": c.details} for c in self.checks],
        }


@dataclass(slots=True)
class GateSuiteResult:
    """Container for all validation gates."""

    gates: tuple[GateResult, ...]

    @property
    def passed(self) -> bool:
        return all(gate.passed for gate in self.gates)


@dataclass(slots=True)
class NonDegeneracyDomainSummary:
    """Domain-level summary for non-degeneracy diagnostics."""

    domain: str
    description: str
    status: str
    n_checks: int
    n_passed: int
    failing_checks: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "domain": self.domain,
            "description": self.description,
            "status": self.status,
            "n_checks": self.n_checks,
            "n_passed": self.n_passed,
            "failing_checks": self.failing_checks,
        }


@dataclass(slots=True)
class NonDegeneracyReport:
    """Compact report over key non-degeneracy domains."""

    status: str
    n_domains: int
    n_failed_domains: int
    n_incomplete_domains: int
    domains: list[NonDegeneracyDomainSummary] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "n_domains": self.n_domains,
            "n_failed_domains": self.n_failed_domains,
            "n_incomplete_domains": self.n_incomplete_domains,
            "domains": [domain.to_dict() for domain in self.domains],
        }


@dataclass(slots=True)
class PerformanceBenchmarkEntry:
    """Performance benchmark row."""

    n_agents: int
    n_steps: int
    elapsed_s: float
    peak_memory_mb: float
    per_step_ms: float


@dataclass(slots=True)
class PerformanceReport:
    """Performance benchmark collection."""

    benchmarks: list[PerformanceBenchmarkEntry] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "benchmarks": [
                {
                    "n_agents": entry.n_agents,
                    "n_steps": entry.n_steps,
                    "elapsed_s": entry.elapsed_s,
                    "peak_memory_mb": entry.peak_memory_mb,
                    "per_step_ms": entry.per_step_ms,
                }
                for entry in self.benchmarks
            ]
        }


@dataclass(slots=True)
class MutableSummaryRow:
    """Summary of a mutable construct vs survey."""

    construct: str
    agent_mean: float
    survey_mean: float
    abs_diff: float

    def to_dict(self) -> dict[str, object]:
        return {
            "construct": self.construct,
            "agent_mean": self.agent_mean,
            "survey_mean": self.survey_mean,
            "abs_diff": self.abs_diff,
        }


@dataclass(slots=True)
class QuantileSummaryRow:
    """Quantile summary for shape checks."""

    construct: str
    agent_q10: float
    agent_q50: float
    agent_q90: float
    survey_q10: float
    survey_q50: float
    survey_q90: float

    def to_dict(self) -> dict[str, object]:
        return {
            "construct": self.construct,
            "agent_q10": self.agent_q10,
            "agent_q50": self.agent_q50,
            "agent_q90": self.agent_q90,
            "survey_q10": self.survey_q10,
            "survey_q50": self.survey_q50,
            "survey_q90": self.survey_q90,
        }


@dataclass(slots=True)
class PopulationDiagnosticsReport:
    """Diagnostics emitted by build-pop."""

    gate: GateResult
    mutable_summary: list[MutableSummaryRow]
    quantile_summary: list[QuantileSummaryRow]
    duplication_rate: float
    anchor_coverage_rate: float
    n_agents: int
    countries: list[str]
    source_population: str

    def to_dict(self) -> dict[str, object]:
        return {
            "gate": self.gate.to_dict(),
            "mutable_summary": [row.to_dict() for row in self.mutable_summary],
            "quantile_summary": [row.to_dict() for row in self.quantile_summary],
            "duplication_rate": self.duplication_rate,
            "anchor_coverage_rate": self.anchor_coverage_rate,
            "n_agents": self.n_agents,
            "countries": self.countries,
            "source_population": self.source_population,
        }


@dataclass(slots=True)
class BenchmarkReport:
    """Full benchmark report bundle."""

    gates: tuple[GateResult, ...]
    performance: PerformanceReport
    non_degeneracy: NonDegeneracyReport | None = None

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "gates": [gate.to_dict() for gate in self.gates],
            "performance": self.performance.to_dict(),
        }
        if self.non_degeneracy is not None:
            payload["non_degeneracy"] = self.non_degeneracy.to_dict()
        return payload


_SUPPLEMENTARY_BASELINE_OUTCOME_CANDIDATES: tuple[str, ...] = (
    "ever_vaccinated_covid",
    "mandate_support",
    "protection_index",
)
_VAX_WILLINGNESS_MEAN_CANDIDATES: tuple[str, ...] = ("vax_willingness_T12_mean",)
_ABM_OUTCOME_MEAN_COLUMNS: frozenset[str] = frozenset(f"{name}_mean" for name in OUTCOME_COLUMNS)


def _is_abm_outcome_mean_column(col: str) -> bool:
    return col in _ABM_OUTCOME_MEAN_COLUMNS


def _abm_outcome_mean_columns(columns: list[str]) -> list[str]:
    return [col for col in columns if _is_abm_outcome_mean_column(col)]


def _safe_quantiles(values: np.ndarray) -> tuple[float, float, float]:
    clean = values[np.isfinite(values)]
    if clean.size == 0:
        return 0.0, 0.0, 0.0
    q10, q50, q90 = np.nanpercentile(clean, [10, 50, 90])
    return float(q10), float(q50), float(q90)


def _first_present_column(columns: list[str], candidates: tuple[str, ...]) -> str | None:
    for candidate in candidates:
        if candidate in columns:
            return candidate
    return None


def _derive_engine_quantile_columns(
    agents_t0: pl.DataFrame,
    survey_df: pl.DataFrame,
    models: dict[str, BackboneModel],
    levers_cfg: LeversConfig,
) -> tuple[str, ...]:
    """Derive engine columns for quantile checks from loaded model terms."""
    available = set(agents_t0.columns) & set(survey_df.columns)
    ordered_cols: list[str] = []
    seen: set[str] = set()

    abm_levers_cfg = get_abm_levers_config(levers_cfg)
    for outcome_name in abm_levers_cfg.outcomes:
        model = models.get(outcome_name)
        if model is None:
            continue
        for term in model.terms:
            if term not in MUTABLE_COLUMNS:
                continue
            if term not in available or term in seen:
                continue
            ordered_cols.append(term)
            seen.add(term)
    return tuple(ordered_cols)


def load_gradient_specs_from_modeling(
    model_coefficients_path: Path,
    levers_cfg: LeversConfig,
    *,
    min_abs_estimate: float = 0.05,
) -> tuple[tuple[str, str, float], ...]:
    """Build gradient specs from empirical modeling coefficients."""
    if not model_coefficients_path.is_file():
        msg = f"Model coefficients file not found: {model_coefficients_path}"
        raise FileNotFoundError(msg)

    coeff_df = pl.read_csv(model_coefficients_path)
    required = {"model", "term", "estimate"}
    missing = required - set(coeff_df.columns)
    if missing:
        msg = f"Model coefficients file missing required columns: {sorted(missing)}"
        raise ValueError(msg)

    specs: list[tuple[str, str, float]] = []
    seen: set[tuple[str, str]] = set()
    abm_levers_cfg = get_abm_levers_config(levers_cfg)
    for outcome_name, lever_spec in abm_levers_cfg.outcomes.items():
        model_rows = coeff_df.filter(pl.col("model") == lever_spec.model_ref)
        if model_rows.height == 0:
            msg = (
                "No rows found in model coefficients file for model_ref="
                f"'{lever_spec.model_ref}' (outcome='{outcome_name}')."
            )
            raise ValueError(msg)

        model_terms = set(model_rows["term"].to_list())
        matched = False
        for lever in lever_spec.levers:
            if lever not in model_terms:
                continue
            estimate = float(
                model_rows.filter(pl.col("term") == lever)["estimate"][0]  # type: ignore[index]
            )
            if abs(estimate) < min_abs_estimate:
                continue
            sign = float(np.sign(estimate))
            if sign == 0.0:
                continue
            key = (lever, outcome_name)
            if key in seen:
                continue
            specs.append((lever, outcome_name, sign))
            seen.add(key)
            matched = True
        if not matched:
            msg = f"No gradient specs derived for outcome='{outcome_name}' from model_ref='{lever_spec.model_ref}'."
            raise ValueError(msg)
    return tuple(specs)


def _anchor_share_tolerance(
    n_country: int,
    survey_share: float,
    *,
    z_score: float,
    abs_floor: float,
) -> float:
    n_eff = max(n_country, 1)
    p = float(np.clip(survey_share, 0.0, 1.0))
    se = np.sqrt(max(p * (1.0 - p), 1e-8) / n_eff)
    return max(abs_floor, float(z_score * se + (1.0 / n_eff)))


def _tertile_gradient(df: pl.DataFrame, predictor: str, outcome: str) -> tuple[float, int, int]:
    pred = df[predictor].to_numpy().astype(np.float64)
    out = df[outcome].to_numpy().astype(np.float64)

    valid = np.isfinite(pred) & np.isfinite(out)
    pred = pred[valid]
    out = out[valid]
    if pred.size == 0:
        return 0.0, 0, 0

    q33, q67 = np.nanpercentile(pred, [33.0, 67.0])
    low_mask = pred <= q33
    high_mask = pred >= q67
    n_low = int(np.sum(low_mask))
    n_high = int(np.sum(high_mask))
    if n_low == 0 or n_high == 0:
        return 0.0, n_low, n_high

    low_mean = float(np.nanmean(out[low_mask]))
    high_mean = float(np.nanmean(out[high_mask]))
    return high_mean - low_mean, n_low, n_high


def _scenario_key(result: object) -> str:
    """Resolve scenario key from stored scenario name metadata."""
    from beam_abm.abm.scenario_defs import SCENARIO_LIBRARY

    raw = str(getattr(result, "scenario_name", "")).strip().lower()
    for key, scenario in SCENARIO_LIBRARY.items():
        if raw == key.lower() or raw == scenario.name.strip().lower():
            return key
    return raw


def _scenario_effect_regime(result: object) -> str:
    """Resolve effect regime from simulation metadata."""
    metadata = getattr(result, "metadata", None)
    effect_regime = str(getattr(metadata, "effect_regime", "")).strip().lower()
    if effect_regime in {"perturbation", "shock"}:
        return effect_regime

    raw_name = str(getattr(result, "scenario_name", "")).strip().lower()
    if raw_name.endswith("__shock"):
        return "shock"
    return "perturbation"


def _scenario_config_for_result(result: object):
    """Resolve canonical scenario config for a stored simulation result."""
    from beam_abm.abm.scenario_defs import SCENARIO_LIBRARY

    key = _scenario_key(result)
    return SCENARIO_LIBRARY.get(key)


def _scenario_has_no_signals(scenario_cfg: object) -> bool:
    state_signals = tuple(getattr(scenario_cfg, "state_signals", ()))
    choice_signals = tuple(getattr(scenario_cfg, "choice_signals", ()))
    return len(state_signals) == 0 and len(choice_signals) == 0


def _late_window_slope(series: np.ndarray, window: int) -> float:
    """Estimate linear slope on the final ``window`` observations."""
    clean = series[np.isfinite(series)]
    if clean.size < 2:
        return 0.0
    w = int(max(2, min(window, clean.size)))
    y = clean[-w:]
    x = np.arange(w, dtype=np.float64)
    return float(np.polyfit(x, y, 1)[0])


def _protection_to_next_delta_incidence_signature(incidence: np.ndarray, protection: np.ndarray) -> tuple[float, float]:
    """Return corr/slope for protection_t -> Δincidence_{t+1}."""
    n = int(min(incidence.size, protection.size))
    if n <= 1:
        return 0.0, 0.0
    x = protection[: n - 1]
    y = incidence[1:n] - incidence[: n - 1]
    finite = np.isfinite(x) & np.isfinite(y)
    if np.sum(finite) <= 1:
        return 0.0, 0.0

    x_f = x[finite]
    y_f = y[finite]
    corr = float(np.corrcoef(x_f, y_f)[0, 1])
    corr = 0.0 if np.isnan(corr) else corr

    x_centered = x_f - float(np.mean(x_f))
    denom = float(np.sum(x_centered * x_centered))
    if denom <= 0.0:
        slope = 0.0
    else:
        slope = float(np.sum(x_centered * y_f) / denom)
    return corr, slope


def _protection_index_from_outcome_trajectory(outcome_trajectory: pl.DataFrame) -> np.ndarray | None:
    """Build mean protection index from vax willingness + available NPI outcomes."""
    required = {"vax_willingness_T12_mean"}
    if not required.issubset(set(outcome_trajectory.columns)):
        return None

    vax = normalize_outcome_to_unit_interval(
        outcome_trajectory["vax_willingness_T12_mean"].to_numpy().astype(np.float64),
        "vax_willingness_T12",
    )
    components: list[np.ndarray] = [vax]
    for col_name, outcome_name in (
        ("mask_when_symptomatic_crowded_mean", "mask_when_symptomatic_crowded"),
        ("stay_home_when_symptomatic_mean", "stay_home_when_symptomatic"),
        ("mask_when_pressure_high_mean", "mask_when_pressure_high"),
    ):
        if col_name not in outcome_trajectory.columns:
            continue
        comp = normalize_outcome_to_unit_interval(
            outcome_trajectory[col_name].to_numpy().astype(np.float64),
            outcome_name,
        )
        components.append(comp)
    if not components:
        return None
    return np.mean(np.vstack(components), axis=0)


def _is_null_world_scenario_cfg(scenario_cfg: object) -> bool:
    if not _scenario_has_no_signals(scenario_cfg):
        return False
    return (
        not bool(getattr(scenario_cfg, "communication_enabled", True))
        and not bool(getattr(scenario_cfg, "enable_endogenous_loop", True))
        and str(getattr(scenario_cfg, "incidence_mode", "none")) == "none"
    )


def _is_social_only_scenario_cfg(scenario_cfg: object) -> bool:
    if not _scenario_has_no_signals(scenario_cfg):
        return False
    return (
        bool(getattr(scenario_cfg, "communication_enabled", True))
        and not bool(getattr(scenario_cfg, "enable_endogenous_loop", True))
        and str(getattr(scenario_cfg, "incidence_mode", "none")) == "none"
    )


def _mean_column_support_span(column: str, values: np.ndarray | None = None) -> float:
    """Return bounded support width for a trajectory mean column when available."""
    base = column.removesuffix("_mean")
    bounds = CONSTRUCT_BOUNDS.get(base) or OUTCOME_BOUNDS.get(base)
    if bounds is not None:
        lo, hi = bounds
        if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
            return float(hi - lo)

    if values is not None:
        clean = values[np.isfinite(values)]
        if clean.size > 0:
            observed = float(np.nanmax(clean) - np.nanmin(clean))
            if np.isfinite(observed) and observed > 0.0:
                return observed

    return 1.0


def _partial_effect(
    agents: pl.DataFrame,
    models: dict[str, BackboneModel],
    levers_cfg: LeversConfig,
    *,
    outcome_name: str,
    predictor: str,
    delta: float,
) -> tuple[float, float]:
    """Compute reference and realised one-step finite-difference effects."""
    if predictor not in agents.columns:
        return 0.0, 0.0

    base_det = predict_all_outcomes(
        agents,
        models,
        levers_cfg,
        deterministic=True,
        rng=np.random.default_rng(42),
    )[outcome_name]
    perturbed = agents.with_columns((pl.col(predictor) + delta).alias(predictor))
    pert_det = predict_all_outcomes(
        perturbed,
        models,
        levers_cfg,
        deterministic=True,
        rng=np.random.default_rng(42),
    )[outcome_name]
    pe_ref = float(np.mean(pert_det) - np.mean(base_det))

    # ABM one-step effect in deterministic expectation (no action sampling).
    base_draw = predict_all_outcomes(
        agents,
        models,
        levers_cfg,
        deterministic=True,
        rng=np.random.default_rng(42),
    )[outcome_name]
    pert_draw = predict_all_outcomes(
        perturbed,
        models,
        levers_cfg,
        deterministic=True,
        rng=np.random.default_rng(43),
    )[outcome_name]
    pe_abm = float(np.mean(pert_draw) - np.mean(base_draw))
    return pe_ref, pe_abm


def _shock_unit_denominator(values: np.ndarray, intensity: float) -> float:
    """Robust denominator for survey-scaled shock units."""
    clean = values[np.isfinite(values)]
    if clean.size == 0:
        return 1.0
    q10, q50, q70, q90 = np.percentile(clean, [10, 50, 70, 90])
    if intensity >= 0.7:
        base = float(abs(q90 - q50))
    elif intensity >= 0.0:
        base = float(abs(q70 - q50))
    else:
        base = float(abs(q50 - q10))
    robust_span = float(abs(q90 - q10))
    robust_std = float(np.std(clean))
    floor = max(0.05, 0.10 * max(robust_span, robust_std, 1.0))
    return max(base, floor)


def _estimate_dispersion_half_life_ticks(
    std_series: np.ndarray,
) -> tuple[float | None, float | None, float | None, int]:
    """Estimate dispersion half-life from a std(t) series.

    Fits ``std(t) ≈ a * exp(-t/tau) + b`` via log-linear regression on
    ``std(t) - b`` where ``b`` is the late-run baseline level.
    """
    vals = np.asarray(std_series, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if vals.size < 4:
        return None, None, None, 0

    tail = max(3, int(vals.size // 4))
    baseline = float(np.nanmedian(vals[-tail:]))
    baseline = max(baseline, 0.0)
    excess = vals - baseline
    eps = max(1e-8, 1e-4 * max(float(np.nanmax(vals)), 1.0))
    mask = excess > eps
    n_fit = int(np.count_nonzero(mask))
    if n_fit < 3:
        return None, None, baseline, n_fit

    t = np.arange(vals.size, dtype=np.float64)[mask]
    y = np.log(excess[mask])
    x = np.column_stack([np.ones_like(t), t])
    coeffs, *_ = np.linalg.lstsq(x, y, rcond=None)
    slope = float(coeffs[1])
    if not np.isfinite(slope) or slope >= 0.0:
        return None, None, baseline, n_fit

    tau = -1.0 / slope
    half_life = float(tau * np.log(2.0))

    y_hat = x @ coeffs
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - float(np.mean(y))) ** 2))
    r2 = None if ss_tot <= 1e-12 else float(max(0.0, 1.0 - (ss_res / ss_tot)))
    return half_life, r2, baseline, n_fit


def _chapter6_like_delta(values: np.ndarray) -> float:
    """Approximate Chapter-6 perturbation sizing from empirical quantiles."""
    clean = values[np.isfinite(values)]
    if clean.size == 0:
        return 0.25
    q25, q75 = np.percentile(clean, [25.0, 75.0])
    delta = float(q75 - q25)
    if not np.isfinite(delta) or delta <= 0.0:
        q10, q90 = np.percentile(clean, [10.0, 90.0])
        delta = float(q90 - q10)
    if (not np.isfinite(delta) or delta <= 0.0) and np.any(clean == 0.0):
        nonzero = clean[clean != 0.0]
        if nonzero.size > 0:
            delta = float(np.percentile(nonzero, 50.0))
    if not np.isfinite(delta) or delta <= 0.0:
        sd = float(np.std(clean))
        delta = 0.5 * max(sd, 0.25)
    return float(max(delta, 0.05))


def _safe_jsd(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    p_clean = np.nan_to_num(p.astype(np.float64), nan=0.0)
    q_clean = np.nan_to_num(q.astype(np.float64), nan=0.0)
    p_smooth = p_clean + eps
    q_smooth = q_clean + eps
    p_norm = p_smooth / np.sum(p_smooth)
    q_norm = q_smooth / np.sum(q_smooth)
    return float(jensenshannon(p_norm, q_norm, base=2.0))


def _adaptive_jsd_tolerance(
    n_obs: int,
    *,
    base: float = 0.08,
    min_tol: float = 0.05,
    max_tol: float = 0.22,
    reference_n: int = 200,
) -> float:
    n_eff = max(1, int(n_obs))
    scaled = base * np.sqrt(reference_n / n_eff)
    return float(np.clip(scaled, min_tol, max_tol))


def _pearson_corr_safe(x: np.ndarray, y: np.ndarray) -> float:
    x64 = x.astype(np.float64)
    y64 = y.astype(np.float64)
    finite = np.isfinite(x64) & np.isfinite(y64)
    if int(np.sum(finite)) < 2:
        return 0.0
    x64 = x64[finite]
    y64 = y64[finite]
    x_std = float(np.std(x64))
    y_std = float(np.std(y64))
    if x_std < 1e-12 or y_std < 1e-12:
        return 0.0
    corr = float(np.corrcoef(x64, y64)[0, 1])
    return corr if np.isfinite(corr) else 0.0


# =========================================================================
# 2.  Gate 1: Baseline replication
# =========================================================================


def run_baseline_replication_gate(
    agents_t0: pl.DataFrame,
    survey_df: pl.DataFrame,
    models: dict[str, BackboneModel],
    levers_cfg: LeversConfig,
    anchor_system: AnchorSystem,
    countries: list[str],
    tolerance: float = 0.15,
    outcome_tolerance: float = 0.12,
    quantile_tolerance: float = 0.75,
    anchor_z_score: float = 2.58,
    anchor_abs_floor: float = 0.05,
    gradient_specs: tuple[tuple[str, str, float], ...] | None = None,
    model_coefficients_path: Path | None = None,
    gradient_min_abs_estimate: float = 0.05,
) -> GateResult:
    """Check that t=0 agent distributions match survey marginals.

    Parameters
    ----------
    agents_t0 : pl.DataFrame
        Initial agent population.
    survey_df : pl.DataFrame
        Original survey data.
    models : dict[str, BackboneModel]
        Loaded backbone models.
    levers_cfg : LeversConfig
        Lever configuration.
    anchor_system : AnchorSystem
        Loaded anchor system for strata checks.
    countries : list[str]
        Countries to check.
    tolerance : float
        Maximum allowed relative deviation in means.

    Returns
    -------
    GateResult
    """
    gate = GateResult(gate_name="baseline_replication")

    gradient_source = "explicit_argument"
    if gradient_specs is None:
        if model_coefficients_path is None:
            msg = "run_baseline_replication_gate requires either explicit gradient_specs or model_coefficients_path."
            raise ValueError(msg)
        gradient_specs = load_gradient_specs_from_modeling(
            model_coefficients_path=model_coefficients_path,
            levers_cfg=levers_cfg,
            min_abs_estimate=gradient_min_abs_estimate,
        )
        gradient_source = "empirical/model_coefficients_all.csv"

    # Check overall construct distributions
    for col in sorted(MUTABLE_COLUMNS):
        if col not in agents_t0.columns or col not in survey_df.columns:
            continue

        agent_mean = float(agents_t0[col].mean())  # type: ignore[arg-type]
        survey_mean = float(survey_df[col].mean())  # type: ignore[arg-type]

        if abs(survey_mean) > 0.01:
            rel_dev = abs(agent_mean - survey_mean) / abs(survey_mean)
        else:
            rel_dev = abs(agent_mean - survey_mean)

        gate.add_check(
            f"marginal_{col}",
            passed=rel_dev < tolerance,
            details={
                "agent_mean": agent_mean,
                "survey_mean": survey_mean,
                "rel_deviation": rel_dev,
            },
        )

    # Outcome marginals at t=0 (prevalence / mean level checks)
    # Primary outcomes are ABM-native outcomes from the shared schema.
    for col in sorted(OUTCOME_COLUMNS):
        if col not in agents_t0.columns or col not in survey_df.columns:
            continue
        agent_mean = float(agents_t0[col].mean())  # type: ignore[arg-type]
        survey_mean = float(survey_df[col].mean())  # type: ignore[arg-type]
        abs_diff = abs(agent_mean - survey_mean)
        gate.add_check(
            f"outcome_marginal_primary_{col}",
            passed=abs_diff < outcome_tolerance,
            details={
                "agent_mean": agent_mean,
                "survey_mean": survey_mean,
                "abs_diff": abs_diff,
                "outcome_source": "primary_outcome_columns",
            },
        )

    # Supplementary outcomes are diagnostics run by default, but separate
    # from required ABM outcomes to keep report semantics explicit.
    for col in _SUPPLEMENTARY_BASELINE_OUTCOME_CANDIDATES:
        if col not in agents_t0.columns or col not in survey_df.columns:
            continue
        agent_mean = float(agents_t0[col].mean())  # type: ignore[arg-type]
        survey_mean = float(survey_df[col].mean())  # type: ignore[arg-type]
        abs_diff = abs(agent_mean - survey_mean)
        gate.add_check(
            f"outcome_marginal_supplementary_{col}",
            passed=abs_diff < outcome_tolerance,
            details={
                "agent_mean": agent_mean,
                "survey_mean": survey_mean,
                "abs_diff": abs_diff,
                "outcome_source": "supplementary_baseline_candidates",
            },
        )

    # Distribution shape checks for engine predictors inferred from model refs.
    engine_quantile_cols = _derive_engine_quantile_columns(
        agents_t0=agents_t0,
        survey_df=survey_df,
        models=models,
        levers_cfg=levers_cfg,
    )
    for col in engine_quantile_cols:
        if col not in agents_t0.columns or col not in survey_df.columns:
            continue
        agent_q10, agent_q50, agent_q90 = _safe_quantiles(agents_t0[col].to_numpy().astype(np.float64))
        survey_q10, survey_q50, survey_q90 = _safe_quantiles(survey_df[col].to_numpy().astype(np.float64))
        lo, hi = CONSTRUCT_BOUNDS.get(col, (-np.inf, np.inf))
        local_quantile_tolerance = quantile_tolerance
        # Ordinal 1-7 constructs are discrete; allow one Likert step.
        if np.isfinite(lo) and np.isfinite(hi) and (hi - lo) <= 6.5:
            local_quantile_tolerance = max(local_quantile_tolerance, 1.0)
        q_diffs = [
            abs(agent_q10 - survey_q10),
            abs(agent_q50 - survey_q50),
            abs(agent_q90 - survey_q90),
        ]
        max_q_diff = max(q_diffs)
        gate.add_check(
            f"engine_quantiles_{col}",
            passed=max_q_diff <= local_quantile_tolerance,
            details={
                "agent_q10": agent_q10,
                "agent_q50": agent_q50,
                "agent_q90": agent_q90,
                "survey_q10": survey_q10,
                "survey_q50": survey_q50,
                "survey_q90": survey_q90,
                "max_abs_quantile_diff": max_q_diff,
                "quantile_tolerance_used": local_quantile_tolerance,
            },
        )

    # Bivariate gradients for baseline realism (Chapter 4 style).
    for predictor, outcome, expected_sign in gradient_specs:
        if predictor not in agents_t0.columns or outcome not in agents_t0.columns:
            continue
        if predictor not in survey_df.columns or outcome not in survey_df.columns:
            continue

        survey_delta, survey_n_low, survey_n_high = _tertile_gradient(survey_df, predictor, outcome)
        agent_delta, agent_n_low, agent_n_high = _tertile_gradient(agents_t0, predictor, outcome)

        if survey_n_low < 5 or survey_n_high < 5 or agent_n_low < 5 or agent_n_high < 5:
            continue

        strong_survey_effect_threshold = 0.10
        survey_sign = float(np.sign(survey_delta))
        model_sign = float(np.sign(expected_sign))
        mag_ratio = abs(agent_delta) / max(abs(survey_delta), 1e-8)
        if abs(survey_delta) >= strong_survey_effect_threshold:
            target_sign = survey_sign
            observed_ok = np.sign(agent_delta) == target_sign
            magnitude_ok = 0.25 <= mag_ratio <= 4.0
            validation_regime = "survey_strong_sign_enforced"
        else:
            # Weak survey gradients are treated as diagnostic magnitude checks.
            target_sign = model_sign
            observed_ok = True
            magnitude_ok = mag_ratio <= 6.0
            validation_regime = "survey_weak_magnitude_only"
        gate.add_check(
            f"bivar_gradient_{predictor}_to_{outcome}",
            passed=bool(observed_ok and magnitude_ok),
            details={
                "agent_delta_high_low": float(agent_delta),
                "survey_delta_high_low": float(survey_delta),
                "expected_sign": float(target_sign),
                "model_expected_sign": float(expected_sign),
                "target_sign_used": float(target_sign),
                "validation_regime": validation_regime,
                "magnitude_ratio_agent_to_survey": float(mag_ratio),
                "agent_n_low": int(agent_n_low),
                "agent_n_high": int(agent_n_high),
                "gradient_source": gradient_source,
            },
        )
        gate.add_check(
            f"gradient_{predictor}_to_{outcome}",
            passed=bool(observed_ok and magnitude_ok),
            details={
                "alias_for": f"bivar_gradient_{predictor}_to_{outcome}",
                "agent_delta_high_low": float(agent_delta),
                "survey_delta_high_low": float(survey_delta),
                "expected_sign": float(target_sign),
                "model_expected_sign": float(expected_sign),
            },
        )

    # Check by country
    for country in countries:
        agent_c = agents_t0.filter(pl.col("country") == country)
        survey_c = survey_df.filter(pl.col("country") == country)
        if agent_c.height == 0 or survey_c.height == 0:
            continue

        agent_frac = agent_c.height / agents_t0.height
        survey_frac = survey_c.height / survey_df.height
        gate.add_check(
            f"country_proportion_{country}",
            passed=abs(agent_frac - survey_frac) < 0.1,
            details={
                "agent_frac": agent_frac,
                "survey_frac": survey_frac,
            },
        )

    # Anchor strata replication (overall + by country)
    if "anchor_id" in agents_t0.columns:
        memberships = anchor_system.memberships.select(["row_id", "anchor_id", "weight"]).join(
            survey_df.select(["row_id", "country"]),
            on="row_id",
            how="left",
        )
        survey_anchor = (
            memberships.group_by(["anchor_id", "country"])
            .agg(pl.col("weight").sum().alias("weight_sum"))
            .with_columns((pl.col("weight_sum") / pl.col("weight_sum").sum().over("country")).alias("share"))
            .select(["anchor_id", "country", "share"])
        )
        agent_anchor = (
            agents_t0.group_by(["anchor_id", "country"])
            .agg(pl.len().alias("count"))
            .with_columns((pl.col("count") / pl.col("count").sum().over("country")).alias("share"))
            .select(["anchor_id", "country", "share"])
        )

        anchor_join = agent_anchor.join(
            survey_anchor,
            on=["anchor_id", "country"],
            how="full",
            suffix="_survey",
        ).fill_null(0.0)

        for row in anchor_join.iter_rows(named=True):
            anchor_id = int(row["anchor_id"]) if row["anchor_id"] is not None else -1
            country = str(row["country"]) if row["country"] is not None else "NA"
            agent_share = float(row.get("share", 0.0))
            survey_share = float(row.get("share_survey", 0.0))
            diff = abs(agent_share - survey_share)
            n_country = max(1, agents_t0.filter(pl.col("country") == country).height)
            tol = _anchor_share_tolerance(
                n_country,
                survey_share,
                z_score=anchor_z_score,
                abs_floor=anchor_abs_floor,
            )
            gate.add_check(
                f"anchor_share_{anchor_id}_{country}",
                passed=diff < tol,
                details={
                    "agent_share": agent_share,
                    "survey_share": survey_share,
                    "abs_diff": diff,
                    "adaptive_tolerance": tol,
                    "n_country": n_country,
                },
            )

    # Propensity strata replication via histogram distance (overall + country).
    propensity_cols = [
        "propensity_slice",
        "propensity_bin",
        "propensity_decile",
        "propensity_quintile",
    ]
    for col in propensity_cols:
        if col not in agents_t0.columns or col not in survey_df.columns:
            continue
        agent_dist = (
            agents_t0.group_by(col)
            .agg((pl.len() / agents_t0.height).alias("share"))
            .with_columns(pl.col(col).cast(pl.Utf8))
        )
        survey_dist = (
            survey_df.group_by(col)
            .agg((pl.len() / survey_df.height).alias("share"))
            .with_columns(pl.col(col).cast(pl.Utf8))
        )
        joined = agent_dist.join(survey_dist, on=col, how="full", suffix="_survey").fill_null(0.0)
        agent_shares = [float(v) for v in joined["share"].to_list()]
        survey_shares = [float(v) for v in joined["share_survey"].to_list()]
        jsd = _safe_jsd(np.asarray(agent_shares), np.asarray(survey_shares))
        n_eff = min(agents_t0.height, survey_df.height)
        tol = _adaptive_jsd_tolerance(n_eff)
        gate.add_check(
            f"propensity_hist_distance_{col}",
            passed=jsd <= tol,
            details={
                "jsd": jsd,
                "adaptive_tolerance": tol,
                "n_effective": n_eff,
            },
        )
        gate.add_check(
            f"propensity_jsd_{col}",
            passed=jsd <= tol,
            details={"alias_for": f"propensity_hist_distance_{col}", "jsd": jsd},
        )
        if "country" not in agents_t0.columns or "country" not in survey_df.columns:
            continue
        for country in countries:
            agent_c = agents_t0.filter(pl.col("country") == country)
            survey_c = survey_df.filter(pl.col("country") == country)
            if agent_c.height == 0 or survey_c.height == 0:
                continue
            agent_dist_c = (
                agent_c.group_by(col)
                .agg((pl.len() / agent_c.height).alias("share"))
                .with_columns(pl.col(col).cast(pl.Utf8))
            )
            survey_dist_c = (
                survey_c.group_by(col)
                .agg((pl.len() / survey_c.height).alias("share"))
                .with_columns(pl.col(col).cast(pl.Utf8))
            )
            joined_c = agent_dist_c.join(survey_dist_c, on=col, how="full", suffix="_survey").fill_null(0.0)
            agent_shares_c = [float(v) for v in joined_c["share"].to_list()]
            survey_shares_c = [float(v) for v in joined_c["share_survey"].to_list()]
            jsd_c = _safe_jsd(np.asarray(agent_shares_c), np.asarray(survey_shares_c))
            tol_c = _adaptive_jsd_tolerance(min(agent_c.height, survey_c.height))
            gate.add_check(
                f"propensity_hist_distance_{col}_{country}",
                passed=jsd_c <= tol_c,
                details={
                    "country": country,
                    "jsd": jsd_c,
                    "adaptive_tolerance": tol_c,
                    "n_effective": min(agent_c.height, survey_c.height),
                },
            )

    gate.evaluate()
    logger.info(
        "Baseline replication gate: {status} ({n_pass}/{n_total} checks).",
        status="PASSED" if gate.passed else "FAILED",
        n_pass=sum(1 for c in gate.checks if c.passed),
        n_total=len(gate.checks),
    )
    return gate
