"""
Simple vLLM configuration optimizer for Qwen models.
Calculates optimal settings based on model size, prompt length, and quantization.
"""

import re

import torch
from pydantic import BaseModel, Field

from beam_abm.common.logging import get_logger

logger = get_logger(__name__)


def extract_model_params(model_name: str) -> int:
    """Extract parameter count from model name."""
    model_lower = model_name.lower()

    # Match patterns like "1.7b", "4b", "8b", "10b"
    param_match = re.search(r"(\d+(?:\.\d+)?)b", model_lower)
    if param_match:
        params = float(param_match.group(1))
        return int(params * 1000) if params < 100 else int(params)  # Convert to millions

    # Default fallback
    return 7000  # 7B default


def get_gpu_memory_gb(device_index: int = 0) -> float:
    """
    Auto-detect GPU memory in GB using PyTorch.

    Args:
        device_index: GPU device index (default: 0)

    Returns:
        GPU memory in GB, or fallback value if detection fails
    """
    try:
        if torch is not None and torch.cuda.is_available():
            # Get device properties for the specified GPU
            device_props = torch.cuda.get_device_properties(device_index)
            # Convert from bytes to GB
            memory_gb = device_props.total_memory / (1024**3)
            logger.info(f"Auto-detected GPU memory: {memory_gb:.1f} GB")
            return memory_gb
        else:
            # Fallback for CPU-only systems or when torch is not available
            logger.warning("PyTorch or CUDA not available, using fallback GPU memory")
            return 16.0  # Conservative default
    except Exception as e:
        # Fallback if detection fails
        logger.warning(f"GPU memory detection failed: {e}, using fallback")
        return 24.0  # Current RTX 3090 default


def calculate_qwen_kv_memory(model_params: int, sequence_length: int) -> float:
    """
    Calculate KV cache memory for Qwen models.
    Qwen uses Grouped-Query Attention with 8 KV heads.
    """
    # Qwen architecture mapping (approximate)
    layer_mapping = {
        1700: 28,  # 1.7B
        4000: 36,  # 4B
        8000: 36,  # 8B
        10000: 40,  # 10B
    }

    # Find closest match
    closest_size = min(layer_mapping.keys(), key=lambda x: abs(x - model_params))
    num_layers = layer_mapping[closest_size]

    # Qwen uses 8 KV heads, 128 dim per head
    # Memory = layers × 8_heads × 128_dim × sequence_length × 2_bytes × 2_(K+V)
    kv_memory_bytes = num_layers * 8 * 128 * sequence_length * 2 * 2
    kv_memory_gb = kv_memory_bytes / (1024**3)

    return kv_memory_gb


class OptimizationInfo(BaseModel):
    """Metadata about the optimization process."""

    model_params_millions: int = Field(description="Model parameters in millions")
    avg_prompt_length: int = Field(description="Average prompt length in tokens")
    quantization: str = Field(description="Quantization type")
    kv_memory_per_seq_gb: float = Field(description="KV cache memory per sequence in GB")
    estimated_concurrent_seqs: int = Field(description="Estimated concurrent sequences")
    weight_memory_gb: float = Field(description="Model weight memory in GB")


class VLLMConfig(BaseModel):
    """vLLM configuration settings."""

    max_num_seqs: int = Field(description="Maximum number of sequences")
    max_num_batched_tokens: int = Field(description="Maximum number of batched tokens")
    gpu_memory_utilization: float = Field(description="GPU memory utilization ratio")
    max_model_len: int = Field(description="Maximum model length")
    enable_chunked_prefill: bool = Field(description="Enable chunked prefill")
    block_size: int = Field(description="Block size for memory allocation")
    swap_space: int = Field(description="Swap space in GB")
    optimization_info: OptimizationInfo = Field(description="Optimization metadata")


class PerformanceEstimate(BaseModel):
    """Performance estimation metrics."""

    estimated_tokens_per_sec: float = Field(description="Estimated tokens per second")
    estimated_concurrent_seqs: int = Field(description="Estimated concurrent sequences")
    estimated_memory_efficiency: float = Field(description="Estimated memory efficiency")


def calculate_optimal_config(
    model_name: str, avg_prompt_length: int = 4000, quantization: str = "4bit", gpu_memory_gb: float | None = None
) -> VLLMConfig:
    """
    Calculate optimal vLLM configuration for a model.

    Args:
        model_name: Model identifier (e.g., "Qwen/Qwen3-8B-Chat")
        avg_prompt_length: Average prompt length in tokens
        quantization: Quantization type ("4bit", "8bit", "fp16")
        gpu_memory_gb: Available GPU memory in GB (auto-detected if None)

    Returns:
        VLLMConfig with optimal vLLM configuration
    """
    # Auto-detect GPU memory if not provided
    if gpu_memory_gb is None:
        gpu_memory_gb = get_gpu_memory_gb()

    model_params = extract_model_params(model_name)

    # Calculate KV cache memory per sequence
    kv_memory_per_seq = calculate_qwen_kv_memory(model_params, avg_prompt_length)

    # Estimate model weight memory
    if quantization == "4bit":
        weight_memory = model_params * 0.5 / 1000  # 0.5 bytes per param
    elif quantization == "8bit":
        weight_memory = model_params * 1.0 / 1000  # 1 byte per param
    else:  # fp16
        weight_memory = model_params * 2.0 / 1000  # 2 bytes per param

    # Reserve memory for model weights + misc overhead
    overhead_memory = weight_memory + 2.0  # 2GB misc overhead
    available_kv_memory = gpu_memory_gb - overhead_memory

    # Calculate max concurrent sequences
    max_concurrent_seqs = int(available_kv_memory / kv_memory_per_seq)

    # Apply safety margin and reasonable limits
    max_concurrent_seqs = max(8, min(max_concurrent_seqs, 256))

    # Calculate optimal batched tokens based on model size and context
    if avg_prompt_length > 4000:
        # Long context - favor throughput
        optimal_batched_tokens = max(8192, min(16384, avg_prompt_length * 2))
    else:
        # Short context - favor latency
        optimal_batched_tokens = max(4096, min(8192, avg_prompt_length * 2))

    # GPU memory utilization - higher for quantized models
    if quantization == "4bit":
        gpu_memory_utilization = 0.96
    elif quantization == "8bit":
        gpu_memory_utilization = 0.94
    else:
        gpu_memory_utilization = 0.90

    # Context window - with some buffer
    max_model_len = min(8192, max(4096, int(avg_prompt_length * 1.5)))

    # Chunked prefill settings
    # TODO: To be refactored and adapted dynamically based on the GPU

    return VLLMConfig(
        max_num_seqs=max_concurrent_seqs,
        max_num_batched_tokens=optimal_batched_tokens,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        enable_chunked_prefill=False,
        block_size=16,
        swap_space=min(8, max(2, int(gpu_memory_gb * 0.1))),
        optimization_info=OptimizationInfo(
            model_params_millions=model_params,
            avg_prompt_length=avg_prompt_length,
            quantization=quantization,
            kv_memory_per_seq_gb=kv_memory_per_seq,
            estimated_concurrent_seqs=max_concurrent_seqs,
            weight_memory_gb=weight_memory,
        ),
    )


def get_performance_estimate(model_name: str, config: VLLMConfig) -> PerformanceEstimate:
    """Estimate performance metrics for a configuration."""
    model_params = extract_model_params(model_name)

    # Base throughput estimates for RTX 3090 (tokens/sec)
    base_throughput = {
        1700: 30.0,  # 1.7B
        4000: 22.0,  # 4B
        8000: 15.0,  # 8B
        10000: 11.0,  # 10B
    }

    closest_size = min(base_throughput.keys(), key=lambda x: abs(x - model_params))
    estimated_throughput = base_throughput[closest_size]

    # Adjust for quantization
    if config.optimization_info.quantization == "4bit":
        estimated_throughput *= 0.85  # 4-bit overhead
    elif config.optimization_info.quantization == "8bit":
        estimated_throughput *= 0.95  # 8-bit overhead

    # Adjust for context length
    if config.optimization_info.avg_prompt_length > 4000:
        estimated_throughput *= 0.8  # Long context penalty

    return PerformanceEstimate(
        estimated_tokens_per_sec=estimated_throughput,
        estimated_concurrent_seqs=config.max_num_seqs,
        estimated_memory_efficiency=min(0.95, config.gpu_memory_utilization),
    )
