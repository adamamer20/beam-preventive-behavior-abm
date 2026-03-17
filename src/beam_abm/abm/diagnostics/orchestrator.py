"""Orchestrator for unified ABM diagnostics suites."""

from __future__ import annotations

import hashlib
import time
from datetime import UTC, datetime
from pathlib import Path

from loguru import logger

from beam_abm.abm.diagnostics.contracts import (
    DiagnosticsConfig,
    DiagnosticsReport,
    DiagnosticsSuiteResult,
    SuiteName,
    SuiteSelector,
)
from beam_abm.abm.diagnostics.core import run_core_suite
from beam_abm.abm.diagnostics.credibility import run_credibility_suite
from beam_abm.abm.diagnostics.importation import run_importation_suite
from beam_abm.abm.diagnostics.runtime import load_shared_runtime
from beam_abm.abm.io import create_diagnostics_run_dirs, write_json

__all__ = ["run_diagnostics"]

SUITE_ORDER: tuple[SuiteName, ...] = ("core", "importation", "credibility")


def _resolve_suites(selector: SuiteSelector) -> tuple[SuiteName, ...]:
    if selector == "all":
        return SUITE_ORDER
    return (selector,)


def _build_run_id(seeds: list[int]) -> tuple[str, str]:
    timestamp = datetime.now(tz=UTC)
    timestamp_utc = timestamp.strftime("%Y-%m-%dT%H:%M:%SZ")
    stamp = timestamp.strftime("%Y%m%dT%H%M%SZ")
    seed_hash = hashlib.sha1(",".join(str(seed) for seed in seeds).encode("utf-8")).hexdigest()[:8]
    return f"{stamp}_{seed_hash}", timestamp_utc


def run_diagnostics(
    *,
    suite: SuiteSelector,
    n_agents: int,
    n_steps: int,
    n_reps: int,
    seeds: list[int],
    output_dir: Path,
) -> DiagnosticsReport:
    """Run selected diagnostics suites and write unified ``summary.json``."""
    if not seeds:
        msg = "At least one seed is required for diagnostics."
        raise ValueError(msg)

    suite_order = _resolve_suites(suite)
    run_id, timestamp_utc = _build_run_id(seeds)
    run_paths = create_diagnostics_run_dirs(output_dir, run_id, list(suite_order))

    runtime = load_shared_runtime(
        n_agents=n_agents,
        n_steps=n_steps,
        n_reps=n_reps,
        seed=int(seeds[0]),
        deterministic=True,
    )

    suite_results: list[DiagnosticsSuiteResult] = []
    start_total = time.perf_counter()

    for suite_name in suite_order:
        logger.info("Diagnostics suite start: {suite}", suite=suite_name)
        suite_start = time.perf_counter()
        suite_dir = run_paths.suites[suite_name]
        try:
            if suite_name == "core":
                result = run_core_suite(
                    runtime=runtime,
                    suite_output_dir=suite_dir,
                    n_agents=n_agents,
                    n_steps=n_steps,
                    n_reps=n_reps,
                    seeds=seeds,
                )
            elif suite_name == "importation":
                result = run_importation_suite(
                    runtime=runtime,
                    suite_output_dir=suite_dir,
                    seeds=seeds,
                )
            else:
                result = run_credibility_suite(
                    runtime=runtime,
                    suite_output_dir=suite_dir,
                    seeds=seeds,
                    n_steps=n_steps,
                )
        except Exception as exc:
            elapsed = time.perf_counter() - suite_start
            logger.exception("Diagnostics suite failed: {suite}", suite=suite_name)
            result = DiagnosticsSuiteResult(
                suite=suite_name,
                passed=False,
                elapsed_seconds=float(elapsed),
                n_checks=1,
                n_failed=1,
                artifacts=(),
                report_path=str(suite_dir / "report.json"),
                error=str(exc),
                summary=None,
            )
            write_json(result.to_dict(), suite_dir / "report.json")

        suite_results.append(result)
        logger.info(
            "Diagnostics suite finished: {suite} passed={passed} n_failed={n_failed}",
            suite=suite_name,
            passed=result.passed,
            n_failed=result.n_failed,
        )

    elapsed_total = time.perf_counter() - start_total
    report = DiagnosticsReport(
        run_id=run_id,
        timestamp_utc=timestamp_utc,
        config=DiagnosticsConfig(
            suite=suite,
            n_agents=n_agents,
            n_steps=n_steps,
            n_reps=n_reps,
            seeds=tuple(int(seed) for seed in seeds),
            output_dir=str(output_dir),
        ),
        suite_order=suite_order,
        suites=tuple(suite_results),
        elapsed_seconds_total=float(elapsed_total),
    )

    write_json(report.to_dict(), run_paths.summary)
    return report
