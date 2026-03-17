"""Multi-expert prompting execution for micro-validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from beam_abm.llm.config import get_llm_runtime_settings
from beam_abm.llm.inference import SampleResult, sample_many_concurrent
from beam_abm.llm.prompts.prompts import build_multi_expert_prompts
from beam_abm.llm.schemas.model_config import ModelConfig


@dataclass(frozen=True)
class MultiExpertResult:
    baseline_y0: float | dict[str, float] | None
    expert_deltas: dict[str, dict[str, Any] | None]
    y_pred: float | dict[str, float] | None


async def run_multi_expert(
    *,
    model: ModelConfig,
    base_prompt: str,
    attributes: dict[str, Any],
    task_instructions: str,
    response_model: type | None,
    use_structured_output: bool,
    temperature: float,
    top_p: float,
    baseline_y0: float | dict[str, float] | None = None,
    outcome_type: str | None = None,
    ordinal_levels: list[str] | None = None,
    record_index: int | None = None,
    method: str = "multi_expert",
) -> MultiExpertResult:
    expert_prompts = build_multi_expert_prompts(
        base_prompt=base_prompt,
        attributes=attributes,
        task_instructions=task_instructions,
    )

    prompts = ["" for _ in expert_prompts]
    messages_list = [[{"from": "human", "value": p}] for p in expert_prompts.values()]
    batches = await sample_many_concurrent(
        model=model,
        prompts=prompts,
        messages_list=messages_list,
        response_model=response_model,
        use_structured_output=use_structured_output,
        temperature=temperature,
        top_p=top_p,
        k=1,
        max_retries=get_llm_runtime_settings().service.llm_max_retries,
        method=method,
        methods=[f"{method}::expert::{name}" for name in expert_prompts],
        start_record_index=record_index or 0,
    )

    def _numeric_bounds(levels: list[str] | None) -> tuple[float, float] | None:
        if not levels:
            return None
        numeric: list[float] = []
        for lv in levels:
            try:
                numeric.append(float(lv))
            except (TypeError, ValueError):
                return None
        if not numeric:
            return None
        return min(numeric), max(numeric)

    def _clip(value: float, t: str | None) -> float:
        if t == "binary":
            return max(0.0, min(1.0, value))
        if t == "ordinal":
            bounds = _numeric_bounds(ordinal_levels)
            if bounds is not None:
                lo, hi = bounds
                return max(float(lo), min(float(hi), value))
        return value

    expert_deltas: dict[str, dict[str, Any] | None] = {}
    delta_sum = 0.0
    ok = True
    for (name, _), sample_batch in zip(expert_prompts.items(), batches, strict=False):
        sample: SampleResult | None = sample_batch[0] if sample_batch else None
        parsed = sample.parsed if sample is not None else None
        expert_deltas[name] = parsed if isinstance(parsed, dict) else None
        try:
            delta_raw = parsed.get("delta_raw") if isinstance(parsed, dict) else None
            delta_sum += float(delta_raw)
        except (TypeError, ValueError):
            ok = False

    y_pred: float | dict[str, float] | None = None
    if ok and isinstance(baseline_y0, int | float):
        y_pred = _clip(float(baseline_y0) + delta_sum, outcome_type)

    return MultiExpertResult(baseline_y0=baseline_y0, expert_deltas=expert_deltas, y_pred=y_pred)
