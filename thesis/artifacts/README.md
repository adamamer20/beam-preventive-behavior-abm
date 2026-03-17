# Thesis Artifacts Contract

This directory is the only thesis-readable generated input contract.

Thesis chapters and thesis utility modules must read generated inputs from
`thesis/artifacts/**` only, not from runtime output trees such as:

- `preprocess/output/**`
- `empirical/output/**`
- `evaluation/output/**`
- `abm/output/**`

Runtime output trees remain local working state and are intentionally ignored.
To refresh this contract after running domain pipelines, use the explicit
export targets:

- `make empirical-export-thesis`
- `make evaluation-export-thesis`
- `make abm-export-thesis`
- `make thesis-artifacts`

The export scripts validate required runtime inputs and write compact,
chapter-facing CSV/JSON/Parquet files with stable names under:

- `thesis/artifacts/empirical/`
- `thesis/artifacts/evaluation/`
- `thesis/artifacts/abm/`
