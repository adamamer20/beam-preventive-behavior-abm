.PHONY: help setup dev dev-install test quick-test test-cov lint format check quality pre-commit docs docs-build clean build publish upgrade sync paper-render paper-preview paper-check thesis-render thesis-render-cached thesis-preview thesis-artifacts empirical-export-thesis evaluation-run-choice-unperturbed evaluation-run-choice-perturbed evaluation-run-belief-unperturbed evaluation-run-belief-perturbed evaluation-export-thesis abm-export-thesis clean-spec-post-general preprocess-clean descriptives fetch-ppp inferential modeling-aggregate anchors anchors-pe-reduced anchors-pe-full anchors-pe-ref-p anchors-build anchors-diagnostics abm-diagnostics abm-run-scenarios abm-sensitivity-full abm-sensitivity-full-posthoc

MODELING_WORKERS ?= 8
ANCHOR_K ?= 10
NEIGHBOR_K ?= 200
ANCHORS_OUTDIR ?= empirical/output/anchors/system
GLOBALITY_MIN_COUNTRIES ?= 3
GLOBALITY_PI_THRESHOLD ?= 0.05
GLOBALITY_PI_FACTOR ?=
TAU ?=
TAU_METHOD ?= target_keff
TAU_KEFF_TARGET ?= 2.0
TAU_KNN_K ?= 10
TAU_QUANTILE ?= 0.5
TAU_SAMPLE_N ?= 5000
PCA_VARIANCE_TARGET ?= 0.97
ANCHOR_GLOBALITY_FACTOR_ARG := $(if $(GLOBALITY_PI_FACTOR),--globality-pi-factor $(GLOBALITY_PI_FACTOR),)
ANCHOR_TAU_ARG := $(if $(TAU),--tau $(TAU),)
ANCHOR_TAU_KEFF_ARG := $(if $(TAU_KEFF_TARGET),--tau-keff-target $(TAU_KEFF_TARGET),)

help: ## Show this help message and available commands
	@echo "🚀 beam-abm - Development Commands"
	@echo "=================================================="
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "💡 Use 'make <command>' to run any command above"
	@echo "💡 All commands use the Makefile for consistency and ease of use"

setup: ## Complete development environment setup
	@echo "🔧 Setting up development environment..."
	uv sync --all-extras --dev
	uv run pre-commit install
	@echo "✅ Development environment setup completed!"

dev: ## Start development environment
	@echo "🚀 Starting development environment..."
	uv sync --all-extras --dev

dev-install: ## Install the package in development mode with all dependencies
	@echo "📦 Installing development dependencies..."
	uv sync --all-extras --dev

test: ## Run tests with type checking
	@echo "🧪 Running tests..."
	uv run env DEV_TYPECHECK=1 pytest

quick-test: ## Fast tests without coverage
	@echo "🧪 Running quick tests..."
	uv run pytest

test-cov: ## Run tests with coverage report
	@echo "🧪 Running tests with coverage..."
	uv run pytest --cov=beam_abm --cov-report=html --cov-report=xml
	@echo "📊 Coverage report generated in htmlcov/"

lint: ## Run linting and fix issues
	@echo "🔍 Running linter..."
	uv run ruff check --fix .

format: ## Format code with ruff
	@echo "🎨 Formatting code..."
	uv run ruff format .

check: ## Run all quality checks + tests
	@echo "🔍 Running all quality checks and tests..."
	$(MAKE) lint
	$(MAKE) format
	$(MAKE) test
	@echo "✅ All checks completed!"

quality: lint format pre-commit ## Run all quality checks (lint, format)
	@echo "✅ All quality checks completed!"

pre-commit: ## Run pre-commit hooks on all files
	@echo "🔄 Running pre-commit hooks..."
	uv run pre-commit run --all-files

docs: ## Serve documentation locally
	@echo "📚 Starting documentation server..."
	uv run mkdocs serve

docs-build: ## Build documentation
	@echo "🏗️  Building documentation..."
	uv run mkdocs build

clean: ## Clean build artifacts and cache
	@echo "🧹 Cleaning build artifacts..."
	rm -rf build/ dist/ *.egg-info/
	rm -rf .pytest_cache/ .coverage htmlcov/ .mypy_cache/ .ruff_cache/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	@echo "✅ Cleanup completed!"

build: ## Build the package
	@echo "🏗️  Building package..."
	uv build

publish: ## Publish package to PyPI (requires authentication)
	@echo "🚀 Publishing to PyPI..."
	uv publish

# Development workflow commands

sync: ## Sync dependencies with lockfile
	@echo "🔄 Syncing dependencies..."
	uv sync --all-extras --dev

upgrade: ## Upgrade all dependencies
	@echo "⬆️  Upgrading dependencies..."
	uv sync --upgrade

# PREPROCESSING

fetch-ppp: ## Fetch 2024 Eurostat PPP values for the default country list
	@echo "🌍 Fetching Eurostat PPP indicators from Eurostat..."
	uv run python preprocess/scripts/fetch_ppp.py
	@echo "✅ PPP data saved to preprocess/output"

# Regenerate the CLEAN dataset from the current clean transform spec
preprocess-clean: ## Rebuild output/preprocess/clean_processed_survey.csv from preprocess/specs/5_clean_transformed_questions.json
	@echo "🧼 Rebuilding CLEAN dataset from transform spec..."
	@echo "🧼 Updating clean transform spec (raw stage) before building dataset..."
	uv run python preprocess/scripts/build_clean_spec.py --stage raw
	@echo "✅ Clean spec updated at preprocess/specs/5_clean_transformed_questions.json"
	@echo "🗑️  Removing existing clean_processed_survey.csv (candidate paths)"; \
	rm -f preprocess/output/clean_processed_survey.csv output/preprocess/clean_processed_survey.csv || true
	uv run python preprocess/scripts/main.py
	@echo "✅ CLEAN dataset rebuilt: preprocess/output/clean_processed_survey.csv"
	@echo "🧮 Generating column type specification..."
	uv run python preprocess/scripts/build_column_types.py
	@echo "✅ Column types written to empirical/specs/column_types.tsv"
	@echo "🧮 Generating derivations TSV from transform spec..."
	uv run python preprocess/scripts/build_derivations.py --out preprocess/specs/derivations.tsv
	@echo "✅ Derivations written to preprocess/specs/derivations.tsv"

# DATASET ANALYSIS

descriptives: ## Compute descriptive statistics for cleaned survey (writes empirical/output/descriptives)
	@echo "📊 Computing descriptive statistics..."
	uv run python empirical/scripts/compute_descriptives.py --input preprocess/output/clean_processed_survey.csv --outdir empirical/output/descriptives
	@echo "✅ Descriptives written to empirical/output/descriptives"

inferential: ## Compute inferential statistics for cleaned survey (writes empirical/output/inferential)
	@echo "📈 Running inferential statistics..."
	uv run python empirical/scripts/inferential_analysis.py --input preprocess/output/clean_processed_survey.csv --outdir empirical/output/inferential
	@echo "✅ Inferential analysis written to empirical/output/inferential"

# MODELING

modeling: ## Run structural backbone models and block importance (LOBO/LOBI)
	@echo "🧠 Running structural backbone models and block importance..."
	uv run python empirical/scripts/modeling.py run --input preprocess/output/clean_processed_survey.csv --outdir empirical/output/modeling --max-workers $(MODELING_WORKERS)
	@echo "✅ Modeling outputs written under empirical/output/modeling (per-model coefficients/lobo/lobi)"

block-importance: ## Compute LOBO/LOBI block importance only
	@echo "🧩 Computing LOBO/LOBI block importance (no refits)..."
	uv run python empirical/scripts/modeling.py block-importance --input preprocess/output/clean_processed_survey.csv --outdir empirical/output/modeling --max-workers $(MODELING_WORKERS)
	@echo "✅ Block-importance outputs written under empirical/output/modeling"


# LLM MICROVALIDATION

anchors: anchors-build anchors-diagnostics anchors-pe-reduced anchors-pe-full anchors-pe-ref-p ## Build anchors and compute all PE_ref artifacts (B/P x mutable/stable x reduced/full)
	@echo "✅ Anchor pipeline complete"

anchors-build: ## Build anchor system
	@echo "🧩 Building anchors..."
	uv run python empirical/scripts/anchors.py build --input preprocess/output/clean_processed_survey.csv --outdir $(ANCHORS_OUTDIR) --anchor-k $(ANCHOR_K) --neighbor-k $(NEIGHBOR_K) --globality-min-countries $(GLOBALITY_MIN_COUNTRIES) --globality-pi-threshold $(GLOBALITY_PI_THRESHOLD) $(ANCHOR_GLOBALITY_FACTOR_ARG) --pca-variance-target $(PCA_VARIANCE_TARGET) --tau-method $(TAU_METHOD) $(ANCHOR_TAU_ARG) $(ANCHOR_TAU_KEFF_ARG) --tau-knn-k $(TAU_KNN_K) --tau-quantile $(TAU_QUANTILE) --tau-sample-n $(TAU_SAMPLE_N)
	@echo "✅ Anchor outputs written to $(ANCHORS_OUTDIR)"

anchors-diagnostics: ## Run anchor diagnostics
	@echo "🧪 Running anchor diagnostics..."
	uv run python empirical/scripts/anchors.py diagnostics --input preprocess/output/clean_processed_survey.csv --anchors-dir $(ANCHORS_OUTDIR) --outdir empirical/output/anchors/diagnostics
	@echo "✅ Diagnostics written to empirical/output/anchors/diagnostics"

anchors-pe-reduced: ## Compute reduced-anchor PE refs for B models (mutable engines + stable stratifiers)
	@echo "🧪 Computing reduced-anchor PE refs for B models (mutable + stable)..."
	@for target in vax_willingness_T12 flu_vaccinated_2023_2024 flu_vaccinated_2023_2024_no_habit mask_when_symptomatic_crowded stay_home_when_symptomatic mask_when_pressure_high; do \
		target_col="$$target"; \
		if [ "$$target" = "flu_vaccinated_2023_2024_no_habit" ]; then target_col="flu_vaccinated_2023_2024"; fi; \
		uv run python empirical/scripts/anchors.py pe --input preprocess/output/clean_processed_survey.csv --anchors-dir $(ANCHORS_OUTDIR) --outdir empirical/output/anchors/pe/mutable_engines/reduced/$$target --target-col $$target_col --lever-config config/levers.json --lever-key $$target --lever-scope policy; \
		uv run python empirical/scripts/anchors.py pe --input preprocess/output/clean_processed_survey.csv --anchors-dir $(ANCHORS_OUTDIR) --outdir empirical/output/anchors/pe/stable_stratifiers/reduced/$$target --target-col $$target_col --lever-config config/levers.json --lever-key $$target --lever-scope non_policy; \
	done
	@echo "✅ Reduced mutable-engine PE refs written under empirical/output/anchors/pe/mutable_engines/reduced/*"
	@echo "✅ Reduced stable-stratifier PE refs written under empirical/output/anchors/pe/stable_stratifiers/reduced/*"

anchors-pe-full: ## Compute full-respondent PE refs for B models (mutable engines + stable stratifiers)
	@echo "🧪 Computing full-respondent PE refs for B models (mutable + stable)..."
	@for target in vax_willingness_T12 flu_vaccinated_2023_2024 flu_vaccinated_2023_2024_no_habit mask_when_symptomatic_crowded stay_home_when_symptomatic mask_when_pressure_high; do \
		target_col="$$target"; \
		if [ "$$target" = "flu_vaccinated_2023_2024_no_habit" ]; then target_col="flu_vaccinated_2023_2024"; fi; \
		uv run python empirical/scripts/anchors.py pe-full --input preprocess/output/clean_processed_survey.csv --anchors-dir $(ANCHORS_OUTDIR) --outdir empirical/output/anchors/pe/mutable_engines/full/$$target --target-col $$target_col --lever-config config/levers.json --lever-key $$target --lever-scope policy; \
		uv run python empirical/scripts/anchors.py pe-full --input preprocess/output/clean_processed_survey.csv --anchors-dir $(ANCHORS_OUTDIR) --outdir empirical/output/anchors/pe/stable_stratifiers/full/$$target --target-col $$target_col --lever-config config/levers.json --lever-key $$target --lever-scope non_policy; \
	done
	@echo "✅ Full mutable-engine PE refs written under empirical/output/anchors/pe/mutable_engines/full/*"
	@echo "✅ Full stable-stratifier PE refs written under empirical/output/anchors/pe/stable_stratifiers/full/*"

anchors-pe-ref-p: ## Compute PE_ref_P refs for P models (upstream signals + background traits, reduced/full)
	@echo "🧪 Computing PE_ref_P for upstream signals (reduced + full)..."
	@uv run python evaluation/scripts/run_belief_pe_ref_p.py \
		--data empirical/output/anchors/eval/eval_profiles.parquet \
		--spec evaluation/specs/belief_update_targets.json \
		--out empirical/output/anchors/pe/upstream_signals/reduced/pe_ref_p_rows.parquet \
		--meta-out empirical/output/anchors/pe/upstream_signals/reduced/pe_ref_p_signal_metadata.json \
		--id-col row_id --country-col country
	@uv run python evaluation/scripts/run_belief_pe_ref_p.py \
		--data preprocess/output/clean_processed_survey.csv \
		--spec evaluation/specs/belief_update_targets.json \
		--out empirical/output/anchors/pe/upstream_signals/full/pe_ref_p_rows.parquet \
		--meta-out empirical/output/anchors/pe/upstream_signals/full/pe_ref_p_signal_metadata.json \
		--id-col row_id --country-col country
	@echo "🧪 Computing PE_ref_P for background traits (reduced + full)..."
	@uv run python evaluation/scripts/run_belief_pe_ref_p.py \
		--data empirical/output/anchors/eval/eval_profiles.parquet \
		--spec evaluation/specs/belief_update_background_traits.json \
		--out empirical/output/anchors/pe/background_traits/reduced/pe_ref_p_rows.parquet \
		--meta-out empirical/output/anchors/pe/background_traits/reduced/pe_ref_p_signal_metadata.json \
		--id-col row_id --country-col country
	@uv run python evaluation/scripts/run_belief_pe_ref_p.py \
		--data preprocess/output/clean_processed_survey.csv \
		--spec evaluation/specs/belief_update_background_traits.json \
		--out empirical/output/anchors/pe/background_traits/full/pe_ref_p_rows.parquet \
		--meta-out empirical/output/anchors/pe/background_traits/full/pe_ref_p_signal_metadata.json \
		--id-col row_id --country-col country
	@echo "✅ Upstream-signal PE_ref_P written under empirical/output/anchors/pe/upstream_signals/{reduced,full}"
	@echo "✅ Background-trait PE_ref_P written under empirical/output/anchors/pe/background_traits/{reduced,full}"


# Thesis (Quarto book) commands
empirical-export-thesis: ## Export compact empirical artifacts to thesis/artifacts/empirical
	@echo "📦 Exporting empirical thesis artifacts..."
	@uv run python empirical/scripts/export_thesis_artifacts.py
	@echo "✅ Empirical thesis artifacts exported."

evaluation-run-choice-unperturbed: ## Run choice-validation unperturbed workflow
	@uv run python evaluation/scripts/run_choice_validation_unperturbed.py

evaluation-run-choice-perturbed: ## Run choice-validation perturbed workflow
	@uv run python evaluation/scripts/run_choice_validation_perturbed.py

evaluation-run-belief-unperturbed: ## Run belief-update unperturbed workflow
	@uv run python evaluation/scripts/run_belief_update_unperturbed.py

evaluation-run-belief-perturbed: ## Run belief-update perturbed workflow
	@uv run python evaluation/scripts/run_belief_update_perturbed.py

evaluation-export-thesis: ## Export compact evaluation artifacts to thesis/artifacts/evaluation
	@echo "📦 Exporting evaluation thesis artifacts..."
	@uv run python evaluation/scripts/export_thesis_artifacts.py
	@echo "✅ Evaluation thesis artifacts exported."

abm-export-thesis: ## Export compact ABM artifacts to thesis/artifacts/abm
	@echo "📦 Exporting ABM thesis artifacts..."
	@uv run python abm/scripts/export_thesis_artifacts.py
	@echo "✅ ABM thesis artifacts exported."

thesis-artifacts: empirical-export-thesis evaluation-export-thesis abm-export-thesis ## Build all tracked thesis artifact contracts
	@echo "✅ All thesis artifacts exported under thesis/artifacts/."

thesis-render: ## Render thesis from tracked thesis/artifacts contract (HTML, PDF, DOCX); CACHE=1 to reuse cache
	@echo "📚 Rendering thesis (HTML, PDF, DOCX)..."
	@if [ ! -d thesis/node_modules ]; then \
		echo "📦 Installing thesis Mermaid pre-render dependencies (npm)..."; \
		cd thesis && npm install; \
	fi
	@CACHE_FLAG=""; \
	if [ "$(CACHE)" = "1" ]; then CACHE_FLAG="--cache"; fi; \
	if [ -n "$$CACHE_FLAG" ]; then echo "Using Quarto execution cache"; fi; \
	@THESIS_PY=$$(uv run python -c 'import sys; print(sys.executable)'); \
		echo "Using QUARTO_PYTHON=$$THESIS_PY"; \
		cd thesis && LOG_TO_CONSOLE=0 LOG_LEVEL=CRITICAL QUARTO_PYTHON=$$THESIS_PY quarto render $$CACHE_FLAG
	@echo "✅ Thesis rendered successfully in all formats!"

thesis-render-cached: ## Render thesis book with CACHE=1 (backward-compatible alias)
	@$(MAKE) thesis-render CACHE=1

thesis-preview: ## Preview thesis book (HTML)
	@echo "👀 Previewing thesis (HTML)..."
	@THESIS_PY=$$(uv run python -c 'import sys; print(sys.executable)'); \
		echo "Using QUARTO_PYTHON=$$THESIS_PY"; \
		cd thesis && LOG_TO_CONSOLE=0 LOG_LEVEL=CRITICAL QUARTO_PYTHON=$$THESIS_PY quarto preview --to html --no-browser

thesis-render-one: ## Render a single thesis page (HTML + PDF): make thesis-render-one FILE=chapters/06_llm_microvalidation.qmd
	@echo "📄 Rendering one page: $(FILE)"
	@if [ -z "$(FILE)" ]; then \
		echo "❌ Please provide FILE=path/to/file.qmd"; \
		exit 1; \
	fi
	@if [ ! -d thesis/node_modules ]; then \
		echo "📦 Installing thesis Mermaid pre-render dependencies (npm)..."; \
		cd thesis && npm install; \
	fi
	@THESIS_PY=$$(uv run python -c 'import sys; print(sys.executable)'); \
		echo "Using QUARTO_PYTHON=$$THESIS_PY"; \
		cd thesis && LOG_TO_CONSOLE=0 LOG_LEVEL=CRITICAL QUARTO_PYTHON=$$THESIS_PY quarto render "$(FILE)" --to html && \
		LOG_TO_CONSOLE=0 LOG_LEVEL=CRITICAL QUARTO_PYTHON=$$THESIS_PY quarto render "$(FILE)" --to pdf
	@echo "✅ Single page rendered (HTML + PDF)!"

# =============================================================================
# ABM targets
# =============================================================================

ABM_N_AGENTS ?= 10000
ABM_N_STEPS  ?= 52
ABM_N_REPS   ?= 10
ABM_SEED     ?= 42
ABM_SCENARIO ?= baseline
ABM_EFFECT_REGIME ?= perturbation
ABM_DIAG_N_AGENTS ?= 2000
ABM_DIAG_N_STEPS  ?= 24
ABM_DIAG_N_REPS   ?= 3
ABM_DIAG_SEEDS    ?= 42 43 44 45 46
ABM_DIAG_OUTDIR   ?= abm/output/diagnostics
ABM_SENSITIVITY_MODE ?= oat
ABM_SENSITIVITY_RUN_2D ?= false
ABM_SENSITIVITY_FULL_OUTDIR ?= abm/output/full_sensitivity_pipeline
ABM_SENSITIVITY_FULL_N_AGENTS ?= 2000
ABM_SENSITIVITY_FULL_N_STEPS ?= 24
ABM_SENSITIVITY_FULL_SEED_OFFSETS ?= 0 1
ABM_SENSITIVITY_FULL_MORRIS_TRAJ ?= 6
ABM_SENSITIVITY_FULL_MORRIS_LEVELS ?= 6
ABM_SENSITIVITY_FULL_LHC_SAMPLES ?= 100
ABM_SENSITIVITY_FULL_SOBOL_N ?= 2048
ABM_SENSITIVITY_FULL_SHORTLIST_TARGET ?= 8
ABM_SENSITIVITY_FULL_SHORTLIST_MAX ?= 10
ABM_SENSITIVITY_FULL_SIGMA_Q ?= 0.8
ABM_SENSITIVITY_FULL_MIN_R2 ?= 0.30
ABM_SENSITIVITY_FULL_MAX_WORKERS ?= 0
ABM_SENSITIVITY_FULL_PERSIST_RUN_DATA ?= false
ABM_SENSITIVITY_FULL_RESUME_MORRIS_RAW ?= false
ABM_SENSITIVITY_FULL_FLAGSHIP_SCENARIOS ?=
ABM_SENSITIVITY_REFRESH_FLAGSHIP_SCENARIOS ?=

abm-diagnostics: ## Run unified diagnostics (all suites)
	@echo "🧪 Running ABM diagnostics (all suites)..."
	uv run python abm/scripts/abm.py diagnostics \
		--suite all \
		--n-agents $(ABM_DIAG_N_AGENTS) \
		--n-steps $(ABM_DIAG_N_STEPS) \
		--n-reps $(ABM_DIAG_N_REPS) \
		--seeds "$(ABM_DIAG_SEEDS)" \
		--output-dir $(ABM_DIAG_OUTDIR)
	@echo "✅ Diagnostics complete!"

abm-run-scenarios: ## Execute scenarios resolved from scenario_defs (default: all parametric-supported scenarios)
	@SCENS=$$(uv run python -c "from beam_abm.abm.scenario_defs import get_scenario_library; include_mutable = '$(ABM_SCENARIO_INCLUDE_MUTABLE)'.strip().lower() not in {'0', 'false', 'no'}; kernel = '$(ABM_SCENARIO_KERNEL)'.strip() or 'parametric'; raw_families = '$(ABM_SCENARIO_FAMILIES)'.replace(',', ' '); families = {item.strip() for item in raw_families.split() if item.strip()}; scenarios = get_scenario_library(include_mutable=include_mutable); keys = [key for key, spec in sorted(scenarios.items()) if kernel in spec.supported_kernels and (not families or spec.family in families)]; print(' '.join(keys))"); \
	if [ -z "$$SCENS" ]; then \
		echo "❌ No scenarios matched scenario_defs filters."; \
		echo "   ABM_SCENARIO_KERNEL=$(ABM_SCENARIO_KERNEL) ABM_SCENARIO_INCLUDE_MUTABLE=$(ABM_SCENARIO_INCLUDE_MUTABLE) ABM_SCENARIO_FAMILIES=$(ABM_SCENARIO_FAMILIES)"; \
		exit 1; \
	fi; \
	echo "🚀 Running scenario_defs scenarios: $$SCENS"; \
	uv run python abm/scripts/abm.py run \
		--scenario "$$SCENS" \
		--effect-regime $(ABM_EFFECT_REGIME) \
		--n-agents $(ABM_N_AGENTS) \
		--n-steps $(ABM_N_STEPS) \
		--n-reps $(ABM_N_REPS) \
		--seed $(ABM_SEED)

abm-sensitivity-full: ## Run full ABM sensitivity pipeline (Morris->LHC->Surrogate->Sobol)
	@echo "🧪 Running full ABM sensitivity pipeline..."
	uv run python abm/scripts/run_full_sensitivity_pipeline.py \
		--output-dir $(ABM_SENSITIVITY_FULL_OUTDIR) \
		--n-agents $(ABM_SENSITIVITY_FULL_N_AGENTS) \
		--n-steps $(ABM_SENSITIVITY_FULL_N_STEPS) \
		--seed $(ABM_SEED) \
		--seed-offsets $(ABM_SENSITIVITY_FULL_SEED_OFFSETS) \
		--morris-trajectories $(ABM_SENSITIVITY_FULL_MORRIS_TRAJ) \
		--morris-levels $(ABM_SENSITIVITY_FULL_MORRIS_LEVELS) \
		--lhc-samples $(ABM_SENSITIVITY_FULL_LHC_SAMPLES) \
		--sobol-base-samples $(ABM_SENSITIVITY_FULL_SOBOL_N) \
		--shortlist-target $(ABM_SENSITIVITY_FULL_SHORTLIST_TARGET) \
		--shortlist-max $(ABM_SENSITIVITY_FULL_SHORTLIST_MAX) \
		--sigma-quantile $(ABM_SENSITIVITY_FULL_SIGMA_Q) \
		--surrogate-min-r2 $(ABM_SENSITIVITY_FULL_MIN_R2) \
		--max-workers $(ABM_SENSITIVITY_FULL_MAX_WORKERS) \
		$(if $(ABM_SENSITIVITY_FULL_FLAGSHIP_SCENARIOS),--flagship-scenarios $(ABM_SENSITIVITY_FULL_FLAGSHIP_SCENARIOS),) \
		$(if $(filter true,$(ABM_SENSITIVITY_FULL_PERSIST_RUN_DATA)),--persist-run-data,--no-persist-run-data) \
		$(if $(filter true,$(ABM_SENSITIVITY_FULL_RESUME_MORRIS_RAW)),--resume-from-morris-raw,)
	@echo "✅ Full ABM sensitivity pipeline complete!"
