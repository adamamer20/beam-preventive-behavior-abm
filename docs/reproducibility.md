# Reproducibility

BEAM ABM is reproducible at the level of code, tracked thesis artefacts, and
documented workflows. The full survey data are not public, so the steps below
reproduce the public pipeline and its exported outputs rather than a full
raw-data replay from scratch.

## 1. Set up the environment

This creates the project environment and installs the development tools.

```bash
make help
make setup
```

You should end up with a working local environment and pre-commit hooks
installed.

## 2. Rebuild the cleaned survey dataset

This regenerates the cleaned analysis dataset and supporting specifications used
by later stages.

```bash
make preprocess-clean
```

Expected outputs:

- `preprocess/output/clean_processed_survey.csv`
- `empirical/specs/column_types.tsv`
- `preprocess/specs/derivations.tsv`

## 3. Run the empirical analysis

These commands build the descriptive summaries, inferential outputs, backbone
models, and anchoring artefacts used downstream.

```bash
make descriptives
make inferential
make modeling
make anchors
```

Expected outputs:

- `empirical/output/descriptives/`
- `empirical/output/inferential/`
- `empirical/output/modeling/`
- `empirical/output/anchors/`

## 4. Export thesis-facing artefacts

These commands copy the relevant runtime outputs into the tracked thesis
contract under `thesis/artifacts/`.

```bash
make decision-function-export-thesis
make llm-microvalidation-export-thesis
make abm-export-thesis
make thesis-artifacts
```

Expected outputs:

- `thesis/artifacts/empirical/`
- `thesis/artifacts/evaluation/`
- `thesis/artifacts/abm/`

## 5. Build the documentation

This checks that the documentation site renders correctly.

```bash
make docs-build
```

Expected output:

- the built site under `site/`

## 6. Render the thesis

This renders the thesis from the tracked artefacts contract.

```bash
make thesis-render
```

Expected outputs:

- rendered thesis files under `thesis/final_output/`

For a focused check on a single chapter, run:

```bash
make thesis-render-one FILE=04-llm-micro-validation.qmd
```
