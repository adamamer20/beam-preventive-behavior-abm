# Empirical

This directory contains the empirical analysis stage of BEAM ABM. It turns the
cleaned survey dataset into descriptive summaries, inferential outputs,
reference models, and anchoring artefacts used by the later evaluation and ABM
stages.

Use the top-level `Makefile` for all routine commands.

## What lives here

- `scripts/` contains the empirical analysis entrypoints.
- `specs/` contains analysis specifications such as column types and
  derivations.
- `output/` contains generated summaries, models, and anchoring artefacts.
- `plots/` is reserved for generated figures.

## Main commands

```bash
make descriptives
make inferential
make modeling
make block-importance
make anchors
```

- `make descriptives` builds descriptive summaries for the cleaned survey.
- `make inferential` runs the inferential analysis suite.
- `make modeling` fits the structural backbone models and block-importance
  outputs.
- `make block-importance` recomputes block importance without full refits.
- `make anchors` builds the anchor system and associated perturbation-reference
  artefacts.

## Expected outputs

These commands write under `empirical/output/`, especially:

- `empirical/output/descriptives/`
- `empirical/output/inferential/`
- `empirical/output/modeling/`
- `empirical/output/anchors/`

To export the empirical outputs used by the thesis, run:

```bash
make decision-function-export-thesis
```
