# Evaluation

This directory contains the LLM micro-validation stage of BEAM ABM. It covers
the evaluation workflows used to compare LLM outputs against survey-grounded
empirical references for both behavioural outcomes and psychological profile
updates.

Use the top-level `Makefile` for all routine commands.

## What lives here

- `scripts/` contains the evaluation orchestration entrypoints.
- `specs/` contains the evaluation target and signal specifications.
- `output/` contains runtime outputs from evaluation runs.
- `logs/` contains evaluation logs.

Reusable package code for this stage lives in:

- `src/beam_abm/llm_microvalidation/`
- `src/beam_abm/llm/`

## Main commands

```bash
make llm-microvalidation-run-behavioural-outcomes-baseline
make llm-microvalidation-run-behavioural-outcomes-perturbed
make llm-microvalidation-run-psychological-profiles-baseline
make llm-microvalidation-run-psychological-profiles-perturbed
make llm-microvalidation-export-thesis
make thesis-artifacts
```

- `make llm-microvalidation-export-thesis` exports the evaluation outputs needed by the
  thesis.
- `make thesis-artifacts` refreshes the full tracked thesis artefact contract,
  including the evaluation exports.

## Expected outputs

Runtime evaluation runs write under:

- `evaluation/output/choice_validation/`
- `evaluation/output/belief_update_validation/`

The thesis-facing export writes under:

- `thesis/artifacts/evaluation/`

The runtime script surface is intentionally minimal:

- `evaluation/scripts/run_behavioural_outcomes_baseline.py`
- `evaluation/scripts/run_behavioural_outcomes_perturbed.py`
- `evaluation/scripts/run_psychological_profiles_baseline.py`
- `evaluation/scripts/run_psychological_profiles_perturbed.py`
- `evaluation/scripts/export_thesis_artifacts.py`
