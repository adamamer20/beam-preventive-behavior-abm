# `beam_abm.llm` package map

This package contains reusable LLM integration components used by BEAM ABM
evaluation workflows.

## Current structure

```text
src/beam_abm/llm/
├── backends/     # backend adapters and backend factory
├── clients/      # lower-level clients (e.g. async vLLM manager)
├── prompts/      # prompt builders/templates
├── schemas/      # pydantic model configs and prediction schemas
├── strategies/   # strategy logic helpers
├── utils/        # token, logging, and helper utilities
├── config.py     # runtime settings resolver
└── inference.py  # backend-agnostic async inference helpers
```

## Stable imports

```python
from beam_abm.llm.schemas import ModelConfig, ModelOptions
from beam_abm.llm.backends import get_backend, AsyncLLMBackend, LLMRequest
```

## Backend selection

Select backend with `ModelConfig.backend`:

- `vllm`
- `azure_openai` / `openai_v1`-compatible route

Settings are resolved from environment-backed project config (for example:
`OPENAI_API_KEY`, `OPENAI_BASE_URL`, `AZURE_OPENAI_API_KEY`,
`AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_DEPLOYMENT`, `VLLM_BASE_URL`,
`VLLM_API_KEY`).

## Notes

- This package is library code; orchestration CLIs live in `evaluation/scripts/`.
- Legacy `llm_behaviour/` module paths are obsolete in this repository.
