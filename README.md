# Behavioural Epidemic Agent-based Modelling (BEAM)

**Behavioural Epidemic Agent-based Modelling (BEAM)** is the codebase accompanying my master's research thesis on **survey-grounded behavioural epidemic modelling**.

The project studies how preventive behaviour can be carried from empirical survey evidence into counterfactual simulation through a single workflow. It combines three connected layers:

1. **empirical modelling from survey data**, to estimate a compact behavioural backbone,
2. **LLM micro-validation**, to test whether off-the-shelf language models reproduce that structure under baseline and controlled perturbations,
3. **agent-based simulation**, to study how behaviour evolves under interventions, social influence, and epidemic feedback.

## Repository layout

```text
preprocess/            Survey cleaning, recoding, and transformation specs
empirical/             Descriptives, modelling, anchoring, and diagnostics
evaluation/            LLM micro-validation prompts, runs, and summaries
abm/                   Agent-based simulation, scenarios, metrics, sensitivity
src/beam_abm/          Shared package code for survey, empirical, LLM, and ABM logic
thesis/                Quarto thesis source and thesis-facing artefacts
docs/                  Project documentation source
```

Within `src/beam_abm/`, the main package structure is:

```text
survey/               Survey loading and cleaning helpers
empirical/            Shared empirical-model utilities
anchoring/            Anchor construction and resampling logic
evaluation/           Choice and belief micro-validation code
llm/                  Backends, prompts, schemas, and processors
abm/                  Simulation engine, scenarios, diagnostics, and benchmarking
common/               Shared configuration and utility code
```

## Installation

### Recommended: `uv`

Install `uv`:

#### macOS / Linux

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

#### Windows (PowerShell)

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Clone the repository and create the environment:

```bash
git clone https://github.com/adamamer20/beam-abm.git
cd beam-abm
uv sync
```

### Alternative: `pip`

```bash
git clone https://github.com/adamamer20/beam-abm.git
cd beam-abm
python -m venv .venv
source .venv/bin/activate    # On Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -e .
```

## Common workflows

### Preprocessing

```bash
make preprocess-clean
```

### Empirical modelling

```bash
make descriptives
make inferential
make modeling
make anchors
```

### Thesis artefact export

```bash
make empirical-export-thesis
make evaluation-export-thesis
make abm-export-thesis
make thesis-artifacts
```

### Thesis rendering

```bash
make thesis-render
make thesis-preview
make thesis-render-one FILE=04-llm-micro-validation.qmd
```

## Data and reproducibility

The full multi-country survey dataset used in the thesis is **not public**.

The repository still provides a public reproducibility contract through the versioned codebase, the tracked thesis-facing artefacts, and the thesis source itself. This makes the modelling pipeline, exported outputs, and simulation workflow inspectable and reproducible at the artefact level, even though the restricted raw survey data cannot be redistributed.

## License

This project is released under the [MIT License](LICENSE.txt)
