"""ABM credibility diagnostics suite.

Phases 1-3 of the credibility-first plan:
  Phase 1 — Instrumentation (no model changes): runtime harness logging
            additive per-substep contributions and key intermediates.
  Phase 2 — Diagnosis with controlled worlds: attractor ON/OFF, 2x2 social
            ablation, incidence pulse response.
  Phase 3 — Fix identification via decision tree, fix application, and re-run.

Outputs are orchestrator-managed under the suite output directory.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from typing import Any

import polars as pl

from beam_abm.abm.diagnostics.contracts import DiagnosticsSuiteResult
from beam_abm.abm.diagnostics.runtime import DiagnosticsRuntime
from beam_abm.abm.engine import VaccineBehaviourModel
from beam_abm.abm.scenario_defs import SCENARIO_LIBRARY, ScenarioConfig

from . import credibility_analysis as _analysis
from . import credibility_experiments as _experiments
from . import credibility_instrumentation as _instrumentation
from . import credibility_shared as _shared
from . import credibility_suite as _suite

DEFAULT_SEED = _shared.DEFAULT_SEED
LONG_HORIZON_STEPS = _shared.LONG_HORIZON_STEPS
DRIFT_ABS_THRESHOLD = _shared.DRIFT_ABS_THRESHOLD
DRIFT_PER_MONTH_THRESHOLD = _shared.DRIFT_PER_MONTH_THRESHOLD
PULSE_MILD_STD_THRESHOLD = _shared.PULSE_MILD_STD_THRESHOLD
PULSE_MILD_RANGE_THRESHOLD = _shared.PULSE_MILD_RANGE_THRESHOLD
PULSE_MILD_MAX_THRESHOLD = _shared.PULSE_MILD_MAX_THRESHOLD
POLICY_RANK_TIE_TOLERANCE = _shared.POLICY_RANK_TIE_TOLERANCE
POLICY_RANKING_SCENARIOS = _shared.POLICY_RANKING_SCENARIOS
TRACKED_CONSTRUCTS = _shared.TRACKED_CONSTRUCTS

SharedData = _shared.SharedData
SubstepRecord = _instrumentation.SubstepRecord


@contextmanager
def _patched_module_attrs(module: ModuleType, **updates: Any) -> Iterator[None]:
    originals = {name: getattr(module, name) for name in updates}
    for name, value in updates.items():
        setattr(module, name, value)
    try:
        yield
    finally:
        for name, value in originals.items():
            setattr(module, name, value)


def load_shared_data(
    runtime: DiagnosticsRuntime,
    *,
    n_steps: int,
    seed: int = DEFAULT_SEED,
) -> SharedData:
    """Build shared data bundle from diagnostics runtime contracts."""
    return _shared.load_shared_data(runtime, n_steps=n_steps, seed=seed)


def build_model(
    shared: SharedData,
    scenario: ScenarioConfig,
    *,
    attractor_on: bool = True,
    seed: int = DEFAULT_SEED,
) -> VaccineBehaviourModel:
    """Build a model from shared data with optional attractor control."""
    local_cfg = shared.cfg.model_copy(deep=True)
    local_cfg.seed = int(seed)
    model = VaccineBehaviourModel.from_preloaded(
        cfg=local_cfg,
        scenario=scenario,
        models=shared.models,
        levers_cfg=shared.levers_cfg,
        agent_df=shared.agent_df.clone(),
        network=shared.network,
        seed=seed,
    )
    if not attractor_on:
        model._state_ref_attractor = None
    return model


_safe_logit = _instrumentation._safe_logit
_run_instrumented_tick = _instrumentation._run_instrumented_tick


def run_instrumented(
    shared: SharedData,
    scenario: ScenarioConfig,
    *,
    attractor_on: bool = True,
    label: str = "instrumented",
) -> list[SubstepRecord]:
    with _patched_module_attrs(_instrumentation, build_model=build_model):
        return _instrumentation.run_instrumented(
            shared,
            scenario,
            attractor_on=attractor_on,
            label=label,
        )


def run_instrumented_detailed(
    shared: SharedData,
    scenario: ScenarioConfig,
    *,
    attractor_on: bool = True,
    label: str = "detailed",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    with _patched_module_attrs(_instrumentation, build_model=build_model):
        return _instrumentation.run_instrumented_detailed(
            shared,
            scenario,
            attractor_on=attractor_on,
            label=label,
        )


def run_2x2_social_ablation(shared: SharedData) -> pl.DataFrame:
    with _patched_module_attrs(_experiments, run_instrumented_detailed=run_instrumented_detailed):
        return _experiments.run_2x2_social_ablation(shared)


def run_pulse_response(shared: SharedData) -> pl.DataFrame:
    with _patched_module_attrs(_experiments, build_model=build_model):
        return _experiments.run_pulse_response(shared)


def run_long_horizon_stationarity(
    shared: SharedData,
    *,
    steps: int = LONG_HORIZON_STEPS,
) -> pl.DataFrame:
    with _patched_module_attrs(_experiments, build_model=build_model):
        return _experiments.run_long_horizon_stationarity(shared, steps=steps)


summarize_pulse_band = _experiments.summarize_pulse_band
_pairwise_rank_agreement = _experiments._pairwise_rank_agreement
_top_tied_scenarios = _experiments._top_tied_scenarios


def run_policy_ranking_stability(
    shared: SharedData,
    *,
    horizons: tuple[int, int] | None = None,
    scenario_names: tuple[str, ...] = POLICY_RANKING_SCENARIOS,
    tie_tolerance: float = POLICY_RANK_TIE_TOLERANCE,
) -> dict[str, Any]:
    with _patched_module_attrs(
        _experiments,
        build_model=build_model,
        SCENARIO_LIBRARY=SCENARIO_LIBRARY,
    ):
        return _experiments.run_policy_ranking_stability(
            shared,
            horizons=horizons,
            scenario_names=scenario_names,
            tie_tolerance=tie_tolerance,
        )


analyze_decision_tree = _analysis.analyze_decision_tree
generate_report = _analysis.generate_report


def run_credibility_suite(
    *,
    runtime: DiagnosticsRuntime,
    suite_output_dir: Path,
    seeds: list[int],
    n_steps: int,
) -> DiagnosticsSuiteResult:
    with _patched_module_attrs(
        _suite,
        load_shared_data=load_shared_data,
        run_instrumented_detailed=run_instrumented_detailed,
        run_2x2_social_ablation=run_2x2_social_ablation,
        run_pulse_response=run_pulse_response,
        run_long_horizon_stationarity=run_long_horizon_stationarity,
        run_policy_ranking_stability=run_policy_ranking_stability,
        analyze_decision_tree=analyze_decision_tree,
        generate_report=generate_report,
    ):
        return _suite.run_credibility_suite(
            runtime=runtime,
            suite_output_dir=suite_output_dir,
            seeds=seeds,
            n_steps=n_steps,
        )


__all__ = ["run_credibility_suite"]
