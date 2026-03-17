# ABM

This directory contains the agent-based modelling stage of BEAM ABM. It covers
simulation runs, diagnostics, reporting, and sensitivity analysis for the
behavioural epidemic model.

Use the top-level `Makefile` for all routine commands.

## What lives here

- `scripts/` contains the simulation and reporting entrypoints.
- `output/` contains run outputs, diagnostics, summaries, and sensitivity
  results.

Reusable package code for this stage lives mainly in:

- `src/beam_abm/abm/`

## Main commands

```bash
make abm-diagnostics
make abm-run-scenarios
make abm-sensitivity-full
make abm-export-thesis
```

- `make abm-diagnostics` runs the ABM diagnostics suites.
- `make abm-run-scenarios` runs the scenario set resolved from `scenario_defs`.
- `make abm-sensitivity-full` runs the full sensitivity pipeline.
- `make abm-export-thesis` exports the ABM outputs used by the thesis.

## Expected outputs

ABM runs and post-processing write under `abm/output/`, especially:

- `abm/output/runs/`
- `abm/output/diagnostics/`
- `abm/output/full_sensitivity_pipeline/`

The thesis-facing export writes under:

- `thesis/artifacts/abm/`
