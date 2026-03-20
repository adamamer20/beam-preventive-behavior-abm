from __future__ import annotations

import re
from pathlib import Path
from typing import Annotated, Literal

import typer

from beam_abm.abm.workflows import (
    run_benchmark_workflow,
    run_build_population_workflow,
    run_diagnostics_workflow,
    run_report_workflow,
    run_scenarios_workflow,
)

app = typer.Typer(add_completion=False, no_args_is_help=True)


def _split_tokens(raw: str) -> tuple[str, ...]:
    values = tuple(token.strip() for token in re.split(r"[\s,]+", raw.strip()) if token.strip())
    if not values:
        raise typer.BadParameter("Expected a non-empty space/comma separated list.")
    return values


def _split_int_tokens(raw: str) -> tuple[int, ...]:
    tokens = _split_tokens(raw)
    try:
        return tuple(int(token) for token in tokens)
    except ValueError as exc:
        raise typer.BadParameter(f"Invalid integer list: {raw}") from exc


@app.command("build-pop")
def build_pop(
    n_agents: Annotated[int, typer.Option("--n-agents")] = 10_000,
    seed: Annotated[int, typer.Option("--seed")] = 42,
    output: Annotated[Path | None, typer.Option("--output")] = None,
) -> None:
    run_build_population_workflow(n_agents=n_agents, seed=seed, output=output)


@app.command("benchmark")
def benchmark(
    n_agents: Annotated[str, typer.Option("--n-agents")] = "1000,5000,10000",
    n_steps: Annotated[int, typer.Option("--n-steps")] = 52,
    n_reps: Annotated[int, typer.Option("--n-reps")] = 3,
    output: Annotated[Path | None, typer.Option("--output")] = None,
) -> None:
    run_benchmark_workflow(
        n_agents=_split_int_tokens(n_agents),
        n_steps=n_steps,
        n_reps=n_reps,
        output=output,
    )


@app.command("diagnostics")
def diagnostics(
    suite: Annotated[Literal["all", "core", "importation", "credibility"], typer.Option("--suite")] = "all",
    n_agents: Annotated[int, typer.Option("--n-agents")] = 2_000,
    n_steps: Annotated[int, typer.Option("--n-steps")] = 24,
    n_reps: Annotated[int, typer.Option("--n-reps")] = 3,
    seeds: Annotated[str, typer.Option("--seeds")] = "42,43,44,45,46",
    output_dir: Annotated[Path, typer.Option("--output-dir")] = Path("abm/output/diagnostics"),
) -> None:
    report = run_diagnostics_workflow(
        suite=suite,
        n_agents=n_agents,
        n_steps=n_steps,
        n_reps=n_reps,
        seeds=_split_int_tokens(seeds),
        output_dir=output_dir,
    )
    if not report.overall_passed:
        raise typer.Exit(code=1)


@app.command("run")
def run(
    scenario: Annotated[str, typer.Option("--scenario")] = "baseline",
    n_agents: Annotated[int, typer.Option("--n-agents")] = 10_000,
    n_steps: Annotated[int, typer.Option("--n-steps")] = 52,
    n_reps: Annotated[int, typer.Option("--n-reps")] = 10,
    rep_workers: Annotated[int, typer.Option("--rep-workers")] = 0,
    seed: Annotated[int, typer.Option("--seed")] = 42,
    society: Annotated[str | None, typer.Option("--society")] = None,
    pop_mode: Annotated[Literal["survey", "resample_on_manifold", "parquet"] | None, typer.Option("--pop-mode")] = None,
    pop_n: Annotated[int | None, typer.Option("--pop-n")] = None,
    pop_parquet: Annotated[Path | None, typer.Option("--pop-parquet")] = None,
    deterministic: Annotated[bool, typer.Option("--deterministic")] = False,
    belief_surrogate_dir: Annotated[Path | None, typer.Option("--belief-surrogate-dir")] = None,
    effect_regime: Annotated[
        Literal["perturbation", "shock", "both"], typer.Option("--effect-regime")
    ] = "perturbation",
) -> None:
    run_scenarios_workflow(
        scenario=_split_tokens(scenario),
        n_agents=n_agents,
        n_steps=n_steps,
        n_reps=n_reps,
        rep_workers=rep_workers,
        seed=seed,
        society=society,
        pop_mode=pop_mode,
        pop_n=pop_n,
        pop_parquet=pop_parquet,
        deterministic=deterministic,
        belief_surrogate_dir=belief_surrogate_dir,
        effect_regime=effect_regime,
    )


@app.command("report")
def report(
    run_dir: Annotated[Path | None, typer.Option("--run-dir")] = None,
) -> None:
    run_report_workflow(run_dir=run_dir)


if __name__ == "__main__":
    app()
