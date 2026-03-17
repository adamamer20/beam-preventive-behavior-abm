# Project Instructions

This file summarizes coding and project guidelines. Apply it to this template and any projects generated from it.

## Development Environment Setup

**CRITICAL: Always use the Makefile for development tasks!**

The Makefile is the CENTRAL hub for all development operations. Every command should go through make targets for consistency and ease of use.

### Core Development Setup

1. **Project Generation**: Install `cookiecutter` via `uv tool install cookiecutter`
2. **Generate project**:
   ```bash
   cookiecutter https://github.com/adamamer20/pythonic-template
   ```
3. **Initial setup** (ALWAYS use make):
   ```bash
   cd your-new-project
   make setup  # This installs dependencies and sets up pre-commit
   ```

### Essential Make Commands (Use These Daily!)

- `make help` - Show all available commands (ALWAYS start here)
- `make setup` - Complete development environment setup
- `make dev` - Start development environment
- `make test` - Run tests with type checking
- `make quick-test` - Fast tests without coverage
- `make lint` - Run linting and fix issues
- `make format` - Format code
- `make check` - Run all quality checks + tests
- `make clean` - Clean build artifacts

### AI Agent Integration

This template supports multiple AI development assistants:

- **Claude Code CLI**: Anthropic's official CLI (`make ai-setup`)
- **OpenAI Codex**: Uses AGENTS.md instructions
- **Roo Code**: Advanced AI with `.roo/rules-code/rules.md`

The devcontainer automatically installs your selected AI agents and sets up:
- Qdrant vector database (for Roo Code)
- Ollama local LLM runtime (for Roo Code)
- VS Code extensions for each agent

Keep a copy of this file in child projects so these rules apply everywhere.

## Philosophy and General Principles

- Prefer clarity over cleverness.
- Keep a single type per variable.
- Use PEP 484/695 type hints throughout.
- Never pass untyped `dict` payloads across modules/functions.
- Use `pydantic.BaseModel` v2 or `dataclasses.dataclass` for all data contracts.
- Write small, modular functions and avoid duplication.
- Do not hard-code paths, GPUs or URLs; read from environment variables.
- Inline comments should explain why, and stay under 80 characters.

## Environment and Tooling

**CRITICAL: USE MAKEFILE COMMANDS INSTEAD OF DIRECT TOOL CALLS!**

- Manage dependencies with `uv`; let it handle lockfiles.
- Assume Pop OS 24.04 and CUDA 12 on an RTX 3090. If CUDA is missing, raise an error.
- Load configuration from `.env` or `pyproject.toml`.
- Enforce formatting and linting with ruff via pre-commit.
- **Use beartype for runtime type checking** (no mypy needed).

### Package Management (via Make)

- `make sync` - Sync dependencies with lockfile
- `make upgrade` - Upgrade all dependencies
- `make requirements` - Export requirements.txt

### Direct uv commands (when Make doesn't cover it)

- `uv add <package>` adds a dependency and updates the lockfile.
- `uv remove <package>` removes a dependency and updates the lockfile.
- `uv run <package>` runs a package from the current environment.

## Testing and Continuous Integration

**CRITICAL: ALWAYS USE MAKE FOR TESTING!**

- Follow test-driven development using pytest.
- `make test` - Run full test suite with type checking
- `make test-cov` - Run tests with coverage reports
- `make quick-test` - Fast tests for development
- `make ci` - Run full CI pipeline locally
- GitHub Actions workflow uses `uv sync --all-extras --dev`, then `make check`
- Enable pre-commit.ci for automatic lint fixes on pull requests.
- **CRITICAL** Run `make thesis-render` for major thesis-impacting changes (substantive code/logic updates in `thesis/*.qmd` or `thesis/utils/*.py` that could break outputs). For text-only edits or slight code tweaks, full re-render is not required.

## Data and Performance

- Use polars instead of pandas when possible.
- Vectorise heavy numerical work with polars; fallback to numpy otherwise; benchmark loops before keeping them.
- For ABM work, use the latest `mesa-frames` from the `main` branch (not a stale release pin).

## Documentation and Structure

- Control the public API with explicit `__all__` exports.
- Write NumPy-style docstrings with parameters, returns, raises and examples.
- Keep examples small and runnable with `pytest -q`.
- Use argparse for CLI interfaces and add a `--help` option.
- Place imports at the top of each file.

### Quarto authoring conventions 🔧

- **Math rendering:** in Quarto `.qmd` files use **MathJax** syntax: **inline math with `$...$`** and **display/block math with `$$...$$`**. Avoid LaTeX-style wrappers `\(...\)` and `\[...\]` (these are less reliably handled by Quarto/MathJax).
- **Embed results, don’t hard-code:** do **not** hardcode numeric results or estimates into prose. Instead, embed computed values and tables using Python code chunks or inline execution (e.g., `` `{python} compute_value()` ``) so text is reproducible and stays in sync with analysis outputs.

## Safety and Best Practices

- Avoid `eval` and `exec`; use parameterised queries for SQL.
- Keep secrets in environment variables.
- Add a module-level logger with configurable log format.

## Extras

- Use `dataclasses.dataclass(slots=True, frozen=False)` or `pydantic.BaseModel` v2 for configuration.
- **The Makefile IS PROVIDED and is MANDATORY** - use it for all development tasks.
- For paper projects: `make paper-render`, `make paper-preview`, `make paper-check`

## Template Features

### Project Types
- **Standard**: Regular Python package
- **Paper**: Academic paper with Quarto, includes paper/ directory with paper.qmd

### AI Agent Support
- Choose from Claude Code CLI, OpenAI Codex, Roo Code, or combinations
- Automatic devcontainer configuration for selected agents
- Qdrant + Ollama setup for advanced AI workflows

### Development Containers
- Multi-stage Dockerfile with development, builder, and runtime stages
- Docker Compose with service orchestration
- VS Code devcontainer with proper extensions and settings
- Port forwarding for development servers and AI services

Remember: **MAKEFILE IS KING** - always check `make help` first!
