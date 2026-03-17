# Preprocessing

This directory contains the survey-cleaning stage of BEAM ABM. It holds the
transformation specifications and scripts used to rebuild the cleaned survey
dataset consumed by the empirical, evaluation, and ABM stages.

Use the top-level `Makefile` for all routine commands.

## What lives here

- `specs/` contains versioned transformation specifications.
- `scripts/` contains the preprocessing pipeline and helper builders.
- `output/` contains generated datasets and reports.

## Main commands

```bash
make fetch-ppp
make preprocess-clean
```

- `make fetch-ppp` downloads the PPP inputs used by the preprocessing stage.
- `make preprocess-clean` rebuilds the cleaned survey dataset and supporting
  derived specifications.

## Expected outputs

Running `make preprocess-clean` should produce:

- `preprocess/output/clean_processed_survey.csv`
- `empirical/specs/column_types.tsv`
- `preprocess/specs/derivations.tsv`

Other generated survey variants and reports may also appear under
`preprocess/output/`, but the cleaned survey dataset is the main downstream
input.
