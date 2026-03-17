"""Model configuration Pydantic models for LLM ABM package."""

from pydantic import BaseModel, ConfigDict, Field


class AzureOpenAIOptions(BaseModel):
    """Azure OpenAI-specific configuration.

    Notes
    -----
    Azure OpenAI identifies chat models by deployment name.
    Endpoint/key/version are generally sourced from environment variables.
    """

    deployment: str | None = Field(None, min_length=1, description="Azure OpenAI deployment name")
    endpoint: str | None = Field(None, min_length=1, description="Azure OpenAI endpoint")
    api_version: str | None = Field(None, min_length=1, description="Azure OpenAI API version")
    model_hint: str | None = Field(
        None,
        description=(
            "Optional tokenization hint for tiktoken (e.g., 'gpt-4o-mini'). "
            "Deployment names are not stable tokenization IDs."
        ),
    )

    model_config = ConfigDict(extra="forbid")


class ReasoningOptions(BaseModel):
    """Reasoning controls for OpenAI-compatible backends."""

    effort: str | None = Field(None, description="Reasoning effort level (e.g., low/medium/high)")
    summary: str | None = Field(None, description="Reasoning summary verbosity")

    model_config = ConfigDict(extra="forbid")


class ModelOptions(BaseModel):
    """Configuration options for model loading and inference."""

    load_format: str | None = Field(None, alias="load-format")
    quantization: str | None = None
    max_model_len: int | None = Field(None, alias="max-model-len")
    max_num_seqs: int | None = Field(None, alias="max-num-seqs")
    max_num_batched_tokens: int | None = Field(None, alias="max-num-batched-tokens")
    enable_chunked_prefill: bool | None = Field(None, alias="enable-chunked-prefill")
    enable_prefix_caching: bool | None = Field(None, alias="enable-prefix-caching")
    tensor_parallel_size: int | None = Field(None, alias="tensor-parallel-size")
    use_optimization: bool | None = Field(None, alias="use-optimization")
    temperature: float | None = Field(None, ge=0.0, le=2.0)
    top_p: float | None = Field(None, ge=0.0, le=1.0)
    top_k: int | None = Field(None, ge=1)
    min_p: float | None = Field(None, ge=0.0, le=1.0)
    repetition_penalty: float | None = Field(None, ge=0.0)
    gpu_memory_utilization: float | None = Field(None, ge=0.0, le=1.0, alias="gpu-memory-utilization")
    trust_remote_code: bool | None = Field(None, alias="trust-remote-code")
    reasoning: ReasoningOptions | None = None
    text_verbosity: str | None = Field(None, alias="text-verbosity")

    model_config = ConfigDict(
        populate_by_name=True,  # Allow both field names and aliases
        extra="forbid",  # Prevent extra fields
    )


class ModelConfig(BaseModel):
    """Model configuration with validation and direct attribute access."""

    name: str = Field(..., min_length=1, description="Model name")
    backend: str = Field(
        default="vllm",
        description="LLM backend selector (e.g., 'vllm' or 'azure_openai')",
    )
    options: ModelOptions | None = Field(None, description="Model configuration options")
    azure: AzureOpenAIOptions | None = Field(
        None, description="Azure OpenAI options (required when backend='azure_openai')"
    )

    model_config = ConfigDict(extra="forbid")  # Prevent extra fields
