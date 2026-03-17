"""Main orchestrator for the credibility diagnostics suite."""

from __future__ import annotations

import time
from pathlib import Path

import polars as pl
from loguru import logger

from beam_abm.abm.diagnostics.contracts import (
    CredibilityDiagnosticsSummary,
    DiagnosticsSuiteResult,
    summary_to_dict,
)
from beam_abm.abm.diagnostics.runtime import DiagnosticsRuntime
from beam_abm.abm.io import write_json
from beam_abm.abm.scenario_defs import ScenarioConfig

from .credibility_analysis import analyze_decision_tree, generate_report
from .credibility_experiments import (
    run_2x2_social_ablation,
    run_long_horizon_stationarity,
    run_policy_ranking_stability,
    run_pulse_response,
)
from .credibility_instrumentation import run_instrumented_detailed
from .credibility_shared import LONG_HORIZON_STEPS, load_shared_data


def run_credibility_suite(
    *,
    runtime: DiagnosticsRuntime,
    suite_output_dir: Path,
    seeds: list[int],
    n_steps: int,
) -> DiagnosticsSuiteResult:
    """Run the full credibility diagnostics suite."""
    logger.info("=" * 60)
    logger.info("ABM Credibility Diagnostics — Starting")
    logger.info("=" * 60)

    suite_output_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    seed = int(seeds[0]) if seeds else int(runtime.cfg.seed)

    # --- Load shared data ---
    logger.info("Loading shared data contracts...")
    shared = load_shared_data(runtime, n_steps=n_steps, seed=seed)
    logger.info(f"Shared data loaded: {shared.agent_df.height} agents")

    # --- Phase 1+2: Instrumented runs ---

    # Run 1: Social-only, attractor OFF
    logger.info("Phase 1+2: social_only, attractor OFF")
    social_only = ScenarioConfig(
        name="social_only",
        family="diagnostic",
        communication_enabled=True,
        peer_pull_enabled=True,
        observed_norms_enabled=True,
        enable_endogenous_loop=False,
        incidence_mode="none",
    )
    substep_off, tick_off = run_instrumented_detailed(shared, social_only, attractor_on=False, label="social_att_off")
    logger.info(f"  → {len(substep_off)} substep records, {len(tick_off)} tick records")

    # Run 2: Social-only, attractor ON
    logger.info("Phase 1+2: social_only, attractor ON")
    substep_on, tick_on = run_instrumented_detailed(shared, social_only, attractor_on=True, label="social_att_on")
    logger.info(f"  → {len(substep_on)} substep records, {len(tick_on)} tick records")

    # Combine
    substep_df = pl.DataFrame(substep_off + substep_on)
    tick_df = pl.DataFrame(tick_off + tick_on)

    substep_df.write_csv(str(suite_output_dir / "substep_timeseries.csv"))
    tick_df.write_csv(str(suite_output_dir / "tick_summaries.csv"))
    logger.info("Substep and tick CSVs written.")

    # Run 3: 2x2 social ablation
    logger.info("Phase 2: 2×2 social ablation grid")
    ablation_df = run_2x2_social_ablation(shared)
    ablation_df.write_csv(str(suite_output_dir / "ablation_2x2.csv"))
    logger.info(f"  → {ablation_df.height} ablation records")

    # Run 4: Pulse response
    logger.info("Phase 2: Incidence pulse response")
    pulse_df = run_pulse_response(shared)
    pulse_df.write_csv(str(suite_output_dir / "pulse_response.csv"))
    logger.info(f"  → {pulse_df.height} pulse response records")

    logger.info(f"Phase 2: Long-horizon stationarity ({LONG_HORIZON_STEPS} months)")
    long_horizon_df = run_long_horizon_stationarity(shared, steps=LONG_HORIZON_STEPS)
    long_horizon_df.write_csv(str(suite_output_dir / "long_horizon_stationarity.csv"))
    logger.info(f"  → {long_horizon_df.height} long-horizon records")

    logger.info("Phase 2: Policy ranking stability (short vs long horizon)")
    policy_ranking = run_policy_ranking_stability(
        shared,
        horizons=(int(shared.cfg.n_steps), LONG_HORIZON_STEPS),
    )
    write_json(policy_ranking, suite_output_dir / "policy_ranking.json")
    logger.info(
        "  → policy top stable={top}, full stable={full}, agreement={agree:.3f}",
        top=bool(policy_ranking.get("top_scenario_stable", False)),
        full=bool(policy_ranking.get("full_ranking_stable", False)),
        agree=float(policy_ranking.get("pairwise_rank_agreement", float("nan"))),
    )

    # --- Phase 3: Decision tree ---
    logger.info("Phase 3: Decision tree analysis")
    findings = analyze_decision_tree(
        substep_df,
        tick_df,
        ablation_df,
        short_horizon_steps=int(shared.cfg.n_steps),
        pulse_df=pulse_df,
        long_horizon_df=long_horizon_df,
        policy_ranking=policy_ranking,
    )

    write_json(findings, suite_output_dir / "findings.json")
    logger.info(f"Buckets: {findings.get('buckets', [])}")
    logger.info(f"Fix recommendations: {findings.get('fix_recommendations', [])}")

    # --- Generate report ---
    report = generate_report(
        substep_df,
        tick_df,
        ablation_df,
        pulse_df,
        findings,
        n_agents=int(shared.cfg.n_agents),
        n_steps=int(shared.cfg.n_steps),
        seed=seed,
        output_dir=suite_output_dir,
    )
    report_path = suite_output_dir / "report.md"
    report_path.write_text(report, encoding="utf-8")

    elapsed = time.perf_counter() - t0
    logger.info(f"Diagnostics complete in {elapsed:.1f}s")
    logger.info(f"Report: {report_path}")

    invariant_checks = findings.get("invariant_checks", {})
    checks: list[tuple[str, bool]] = [
        ("substep_rows", substep_df.height > 0),
        ("tick_rows", tick_df.height > 0),
        ("ablation_rows", ablation_df.height > 0),
        ("pulse_rows", pulse_df.height > 0),
        ("long_horizon_rows", long_horizon_df.height > 0),
    ]
    checks.extend((f"invariant_{name}", bool(passed)) for name, passed in invariant_checks.items())
    n_failed = sum(1 for _, passed in checks if not passed)
    passed = n_failed == 0

    summary = CredibilityDiagnosticsSummary(
        n_substep_rows=int(substep_df.height),
        n_tick_rows=int(tick_df.height),
        n_ablation_rows=int(ablation_df.height),
        n_pulse_rows=int(pulse_df.height),
        n_long_horizon_rows=int(long_horizon_df.height),
        n_invariant_checks=len(invariant_checks),
        n_failed_invariants=sum(1 for value in invariant_checks.values() if not value),
    )

    suite_report_payload = {
        "suite": "credibility",
        "passed": passed,
        "checks": [{"name": name, "passed": ok} for name, ok in checks],
        "summary": summary.to_dict(),
    }
    suite_report_path = suite_output_dir / "report.json"
    write_json(suite_report_payload, suite_report_path)

    return DiagnosticsSuiteResult(
        suite="credibility",
        passed=passed,
        elapsed_seconds=float(elapsed),
        n_checks=len(checks),
        n_failed=n_failed,
        artifacts=(
            "substep_timeseries.csv",
            "tick_summaries.csv",
            "ablation_2x2.csv",
            "pulse_response.csv",
            "long_horizon_stationarity.csv",
            "policy_ranking.json",
            "findings.json",
            "report.md",
            "report.json",
        ),
        report_path=str(report_path),
        summary=summary_to_dict(summary),
    )
