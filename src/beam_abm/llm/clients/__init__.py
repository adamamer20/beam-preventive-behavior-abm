"""Model managers for LLM ABM package.

This package retains the vLLM engine manager implementation. Higher-level code
should use the backend abstraction in `beam_abm.llm.backends`.
"""

from beam_abm.llm.clients.vllm import AsyncVLLMManager

__all__ = ["AsyncVLLMManager"]
