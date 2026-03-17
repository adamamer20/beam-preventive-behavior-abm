"""OpenAI v1-compatible backend (async) using Responses API.

This backend intentionally uses ONLY the OpenAI Python SDK v1 client (`AsyncOpenAI`)
with the Responses API (`client.responses.create`).

It supports Azure OpenAI v1 GA endpoints by setting `base_url` to:
    https://<resource>.openai.azure.com/openai/v1/

Auth:
- API key: set OPENAI_API_KEY (or AZURE_OPENAI_API_KEY as fallback)
- Base URL: set OPENAI_BASE_URL (or AZURE_OPENAI_ENDPOINT as fallback)

Structured output:
- We do best-effort extraction of a JSON object from model output and validate it
  against a supplied Pydantic `response_model`.

Notes
-----
We intentionally do not support legacy Azure `api-version` query params or the
`AsyncAzureOpenAI` client.
"""

from __future__ import annotations

import asyncio
import json
import random
import re
import time
import uuid
from typing import Any

from beam_abm.common.logging import get_logger
from beam_abm.llm.backends.base import AsyncLLMBackend, LLMRequest, gather_concurrent
from beam_abm.llm.config import get_llm_runtime_settings
from beam_abm.llm.schemas.model_config import ModelConfig
from beam_abm.llm.utils.conversation_logger import ConversationLogger
from beam_abm.llm.utils.tokens import count_tokens

logger = get_logger(__name__)


_JSON_SCHEMA_PROMPT_MARKER = "LLM_ABM_JSON_SCHEMA_INSTRUCTIONS"


def _matches_qwen3_32b(model_name: str) -> bool:
    s = str(model_name or "").strip().lower()
    return s == "qwen/qwen3-32b" or s.endswith("/qwen3-32b") or "qwen3-32b" in s


def _has_json_schema_prompt(openai_messages: list[dict[str, str]]) -> bool:
    for m in openai_messages:
        if not isinstance(m, dict):
            continue
        content = str(m.get("content") or "")
        if _JSON_SCHEMA_PROMPT_MARKER in content:
            return True
    return False


def _inject_json_schema_prompt(
    openai_messages: list[dict[str, str]],
    *,
    response_format: dict[str, Any],
) -> None:
    if _has_json_schema_prompt(openai_messages):
        return

    schema = response_format.get("schema")
    if not isinstance(schema, dict):
        return
    name = str(response_format.get("name") or "ResponseSchema")

    schema_json = json.dumps(schema, indent=2, ensure_ascii=False)
    instruction = (
        f"[{_JSON_SCHEMA_PROMPT_MARKER}]\n"
        "Return ONLY a single JSON object (no prose, no markdown, no code fences). "
        "The JSON must validate against the following JSON Schema. "
        "Do not include additional keys beyond the schema.\n\n"
        f"Schema name: {name}\n"
        f"JSON Schema:\n{schema_json}\n"
    )

    # Prefer appending to an existing system message to preserve ordering.
    if openai_messages and openai_messages[0].get("role") == "system":
        existing = str(openai_messages[0].get("content") or "")
        joiner = "\n\n" if existing.strip() else ""
        openai_messages[0]["content"] = f"{existing}{joiner}{instruction}"
        return

    openai_messages.insert(0, {"role": "system", "content": instruction})


def _messages_to_openai(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for msg in messages:
        frm = msg.get("from") or msg.get("role") or "human"
        value = msg.get("value") or msg.get("content") or ""

        if frm in {"human", "user"}:
            role = "user"
        elif frm in {"assistant", "ai"}:
            role = "assistant"
        elif frm in {"system"}:
            role = "system"
        else:
            role = "user"
        out.append({"role": role, "content": value})
    return out


def _format_prompt_for_logging(openai_messages: list[dict[str, str]]) -> str:
    parts: list[str] = []
    for m in openai_messages:
        parts.append(f"[{m['role']}]\n{m['content']}")
    return "\n\n".join(parts)


def _is_retryable_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    # JSON decode errors commonly indicate an empty/invalid HTTP body from the provider.
    # Treat as transient/retryable (seen intermittently under load).
    if isinstance(exc, json.JSONDecodeError):
        return True
    if "expecting value" in msg and "line 1 column 1" in msg:
        return True
    if "timed out" in msg or "timeout" in msg:
        return True
    if "rate limit" in msg or "too many requests" in msg or "429" in msg or "498" in msg:
        return True
    if "capacity exceeded" in msg or "capacity_exceeded" in msg:
        return True
    if "service unavailable" in msg or "503" in msg:
        return True
    if "gateway" in msg and ("502" in msg or "504" in msg):
        return True
    return False


def _is_groq_base_url(base_url: str) -> bool:
    s = str(base_url).strip().lower()
    return "groq" in s


def _try_parse_rate_limit_retry_after_seconds(exc: Exception) -> float | None:
    """Best-effort parse of provider-specific retry hints.

    Groq commonly returns messages like: "Please try again in 109.92ms.".
    """

    msg = str(exc)
    m = re.search(r"try again in\s*([0-9]+(?:\.[0-9]+)?)\s*ms", msg, flags=re.IGNORECASE)
    if m:
        try:
            return float(m.group(1)) / 1000.0
        except ValueError:
            return None

    m = re.search(r"try again in\s*([0-9]+(?:\.[0-9]+)?)\s*s", msg, flags=re.IGNORECASE)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            return None

    return None


def _retry_sleep_seconds(exc: Exception, attempt_index: int) -> float:
    base = 0.5
    cap = 30.0
    msg = str(exc).lower()
    if "rate limit" in msg or "too many requests" in msg or "429" in msg or "498" in msg:
        hinted = _try_parse_rate_limit_retry_after_seconds(exc)
        if hinted is not None and hinted > 0:
            hinted = max(0.05, hinted)
            return hinted * (0.9 + 0.2 * random.random())

        base = 2.0
        cap = 60.0

    delay = min(cap, base * (2.0**attempt_index))
    return delay * (0.5 + random.random())


def _extract_json_object(text: str) -> str:
    s = text.strip()
    if not s:
        return s
    if s.startswith("{") and s.endswith("}"):
        return s
    start = s.find("{")
    end = s.rfind("}")
    if start >= 0 and end > start:
        return s[start : end + 1]
    return s


def _extract_output_text_from_response_dict(d: dict[str, Any]) -> str:
    out: list[str] = []
    for item in d.get("output", []) or []:
        if not isinstance(item, dict):
            continue
        if item.get("type") != "message":
            continue
        for part in item.get("content", []) or []:
            if not isinstance(part, dict):
                continue
            if part.get("type") == "output_text":
                txt = part.get("text")
                if txt:
                    out.append(str(txt))
    return "".join(out)


def _extract_parsed_from_response_dict(d: dict[str, Any]) -> dict[str, Any] | None:
    parsed = d.get("output_parsed")
    if isinstance(parsed, dict):
        return parsed
    for item in d.get("output", []) or []:
        if not isinstance(item, dict):
            continue
        if item.get("type") != "message":
            continue
        for part in item.get("content", []) or []:
            if not isinstance(part, dict):
                continue
            p = part.get("parsed")
            if isinstance(p, dict):
                return p
    return None


def _response_format_from_pydantic(response_model: type) -> dict[str, Any] | None:
    if not hasattr(response_model, "model_json_schema"):
        return None
    try:
        schema = response_model.model_json_schema()
    except (TypeError, ValueError):
        return None
    name = getattr(response_model, "__name__", "ResponseSchema")

    def _enforce_additional_properties_false(obj: object) -> None:
        """Recursively set additionalProperties=false on any object schema.

        Responses API requires `additionalProperties: false` on object schemas.
        We also force `required` to include every property key; this is required
        by strict JSON Schema validation in the Responses API.
        """
        if isinstance(obj, dict):
            typ = obj.get("type")
            # Treat schemas with `properties` as objects as well
            if typ == "object" or "properties" in obj:
                props = obj.get("properties")
                if isinstance(props, dict):
                    keys = list(props.keys())
                    # Strict mode: require every property key.
                    obj["required"] = keys
                if "additionalProperties" not in obj:
                    obj["additionalProperties"] = False
            # Recurse into common schema containers
            for v in list(obj.values()):
                if isinstance(v, dict | list):
                    _enforce_additional_properties_false(v)
        elif isinstance(obj, list):
            for item in obj:
                _enforce_additional_properties_false(item)

    # Enforce additionalProperties:false across the returned schema to satisfy Responses API checks
    try:
        _enforce_additional_properties_false(schema)
    except (TypeError, AttributeError) as exc:
        logger.debug(
            "Failed to enforce additionalProperties on response schema; proceeding with original schema: %s",
            exc,
        )

    return {
        "type": "json_schema",
        "name": name,
        "strict": True,
        "schema": schema,
    }


def _extract_reasoning_summary_from_response_dict(d: dict[str, Any]) -> str:
    out: list[str] = []
    for item in d.get("output", []) or []:
        if not isinstance(item, dict):
            continue
        if item.get("type") != "reasoning":
            continue
        for part in item.get("summary", []) or []:
            if isinstance(part, dict):
                txt = part.get("text")
                if txt:
                    out.append(str(txt))
            else:
                out.append(str(part))
    return "\n".join([t for t in out if t.strip()])


def _extract_usage_counts_from_response_dict(d: dict[str, Any]) -> tuple[int | None, int | None, int | None]:
    usage = d.get("usage")
    if not isinstance(usage, dict):
        return None, None, None

    def _to_int(x: Any) -> int | None:
        try:
            return int(x) if x is not None else None
        except (TypeError, ValueError):
            return None

    return _to_int(usage.get("input_tokens")), _to_int(usage.get("output_tokens")), _to_int(usage.get("total_tokens"))


def _extract_reasoning_tokens_from_usage_dict(usage: Any) -> int | None:
    if not isinstance(usage, dict):
        return None
    details = usage.get("output_tokens_details") or usage.get("completion_tokens_details")
    if not isinstance(details, dict):
        return None
    val = details.get("reasoning_tokens")
    try:
        return int(val) if val is not None else None
    except (TypeError, ValueError):
        return None


class OpenAIV1Backend(AsyncLLMBackend):
    """Backend using OpenAI Python SDK v1 (AsyncOpenAI) + Responses API."""

    def __init__(self, conversation_logger: ConversationLogger | None = None):
        self._client = None
        self.conversation_logger = conversation_logger

    def set_conversation_logger(self, conversation_logger: ConversationLogger | None) -> None:
        self.conversation_logger = conversation_logger

    def _get_client(self):
        if self._client is not None:
            return self._client

        from openai import AsyncOpenAI

        svc = get_llm_runtime_settings().service
        base_url = svc.openai_base_url or svc.azure_openai_endpoint
        api_key = svc.openai_api_key or svc.azure_openai_api_key

        if not base_url or not api_key:
            raise RuntimeError(
                "OpenAI v1 backend requires base_url and api_key. "
                "Set OPENAI_BASE_URL and OPENAI_API_KEY (or AZURE_OPENAI_ENDPOINT/AZURE_OPENAI_API_KEY)."
            )

        base_url_s = str(base_url).rstrip("/")
        # Azure OpenAI v1 GA expects /openai/v1
        if not base_url_s.endswith("/openai/v1") and not base_url_s.endswith("/openai/v1/"):
            base_url_s = base_url_s + "/openai/v1/"
        elif not base_url_s.endswith("/"):
            base_url_s = base_url_s + "/"

        timeout_seconds = svc.azure_openai_timeout_seconds
        sdk_max_retries = svc.azure_openai_sdk_max_retries

        self._client = AsyncOpenAI(
            base_url=base_url_s,
            api_key=api_key,
            timeout=timeout_seconds,
            max_retries=sdk_max_retries,
        )
        return self._client

    async def generate_response_async(
        self,
        *,
        model: ModelConfig,
        messages: list[dict[str, str]],
        max_model_len: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        min_p: float | None = None,
        repetition_penalty: float | None = None,
        presence_penalty: float | None = None,
        frequency_penalty: float | None = None,
        enable_thinking: bool = False,
        use_structured_output: bool = False,
        response_model: type | None = None,
        max_retries: int | None = None,
        record_index: int | None = None,
        record_id: str | None = None,
        prompt_strategy: str | None = None,
        perturbation: str | None = None,
        method: str = "unknown",
        **kwargs: Any,
    ) -> str:
        _ = (
            temperature,
            top_p,
            top_k,
            min_p,
            repetition_penalty,
            presence_penalty,
            frequency_penalty,
            enable_thinking,
        )

        # Model identifier: use Azure deployment if provided, else model.name.
        deployment = None
        if model.azure and model.azure.deployment:
            deployment = model.azure.deployment
        elif get_llm_runtime_settings().service.azure_openai_deployment:
            deployment = get_llm_runtime_settings().service.azure_openai_deployment
        if not deployment:
            deployment = model.name

        openai_messages = _messages_to_openai(messages)

        resolved_max_model_len = (
            int(max_model_len)
            if max_model_len is not None
            else (model.options.max_model_len if model.options and model.options.max_model_len else 8192)
        )

        response_format_for_request: dict[str, Any] | None = None
        if use_structured_output and response_model is not None:
            response_format_for_request = _response_format_from_pydantic(response_model)

            # Some OpenAI-compatible providers/models (e.g. Groq-hosted Qwen3) do not
            # support `response_format: json_schema` on the Responses API.
            # For qwen/qwen3-32b specifically, inject the JSON schema into a system
            # prompt instead of sending `text.format`.
            model_id = deployment or model.name
            if response_format_for_request is not None and _matches_qwen3_32b(model_id):
                _inject_json_schema_prompt(openai_messages, response_format=response_format_for_request)
                response_format_for_request = None

        prompt_for_logging = _format_prompt_for_logging(openai_messages)
        prompt_tokens = self.estimate_tokens(prompt_for_logging, model)
        available_for_output = max(1, resolved_max_model_len - prompt_tokens)

        retryable_exc_types: tuple[type[BaseException], ...] = (
            RuntimeError,
            ValueError,
            TypeError,
            OSError,
            TimeoutError,
            asyncio.TimeoutError,
        )
        try:
            from openai import OpenAIError  # type: ignore

            retryable_exc_types = (*retryable_exc_types, OpenAIError)
        except ImportError:
            pass

        request_id = str(uuid.uuid4())

        reasoning_effort = str(
            kwargs.get("reasoning_effort") or get_llm_runtime_settings().service.azure_openai_reasoning_effort or ""
        ).strip()
        reasoning_summary_req = str(
            kwargs.get("reasoning_summary") or get_llm_runtime_settings().service.azure_openai_reasoning_summary or ""
        ).strip()
        text_verbosity = str(
            kwargs.get("text_verbosity") or get_llm_runtime_settings().service.azure_openai_text_verbosity or ""
        ).strip()

        base_url_cfg = str(
            get_llm_runtime_settings().service.openai_base_url
            or get_llm_runtime_settings().service.azure_openai_endpoint
            or ""
        )
        base_url_cfg_l = base_url_cfg.lower()
        is_groq = _is_groq_base_url(base_url_cfg_l)
        groq_service_tier = get_llm_runtime_settings().service.groq_service_tier.strip()

        min_rate_limit_retries = get_llm_runtime_settings().service.openai_v1_min_rate_limit_retries
        if max_retries is None:
            max_retries = get_llm_runtime_settings().service.llm_max_retries
        max_attempts = int(max_retries)

        attempt = 0
        while attempt < max_attempts:
            start = time.time()
            try:
                client = self._get_client()

                request_kwargs: dict[str, Any] = {
                    "model": deployment,
                    "input": openai_messages,
                    "max_output_tokens": available_for_output,
                }

                if response_format_for_request is not None:
                    # Responses API expects structured outputs under `text.format`.
                    text_dict = request_kwargs.get("text") or {}
                    text_dict["format"] = response_format_for_request
                    request_kwargs["text"] = text_dict

                # Groq OpenAI-compatible endpoints support `service_tier`.
                if is_groq and groq_service_tier:
                    request_kwargs["service_tier"] = groq_service_tier

                reasoning_obj: dict[str, Any] = {}
                if reasoning_effort:
                    reasoning_obj["effort"] = reasoning_effort
                if reasoning_summary_req:
                    allow_reasoning_summary = not is_groq
                    if allow_reasoning_summary:
                        reasoning_obj["summary"] = reasoning_summary_req
                    else:
                        logger.debug(
                            "Skipping reasoning.summary because base_url indicates Groq endpoint: %s",
                            base_url_cfg_l,
                        )
                if reasoning_obj:
                    request_kwargs["reasoning"] = reasoning_obj

                allow_text_verbosity = not is_groq
                if text_verbosity and allow_text_verbosity:
                    text_dict = request_kwargs.get("text") or {}
                    if isinstance(text_dict, dict):
                        text_dict["verbosity"] = text_verbosity
                        request_kwargs["text"] = text_dict
                    else:
                        request_kwargs["text"] = {"verbosity": text_verbosity}
                elif text_verbosity and not allow_text_verbosity:
                    # Groq Responses API does not accept `text.verbosity` in request body; skip to avoid 400
                    logger.debug(
                        "Skipping text verbosity because base_url indicates Groq endpoint: %s",
                        base_url_cfg_l,
                    )

                try:
                    resp = await client.responses.create(**request_kwargs)
                except TypeError as exc:
                    msg = str(exc).lower()
                    if response_format_for_request is not None and (
                        "format" in msg or "text" in msg or "response_format" in msg
                    ):
                        # Backend doesn't accept structured format under `text.format`; retry without it.
                        if isinstance(request_kwargs.get("text"), dict):
                            request_kwargs["text"].pop("format", None)
                            resp = await client.responses.create(**request_kwargs)
                        else:
                            raise
                    else:
                        raise
                d = resp.model_dump() if hasattr(resp, "model_dump") else {}

                parsed_payload = _extract_parsed_from_response_dict(d)
                response_text = (
                    json.dumps(parsed_payload)
                    if isinstance(parsed_payload, dict)
                    else _extract_output_text_from_response_dict(d)
                )
                reasoning_summary_text = _extract_reasoning_summary_from_response_dict(d)

                usage = d.get("usage") if isinstance(d, dict) else None
                reasoning_tokens = _extract_reasoning_tokens_from_usage_dict(usage)
                in_toks, out_toks, total_toks = _extract_usage_counts_from_response_dict(d)

                raw_response_text = response_text
                if use_structured_output and response_model is not None:
                    try:
                        if isinstance(parsed_payload, dict):
                            data = parsed_payload
                        else:
                            json_text = _extract_json_object(response_text)
                            data = json.loads(json_text)
                        if hasattr(response_model, "model_validate"):
                            response_model.model_validate(data)
                        else:
                            response_model(**data)
                        response_text = json.dumps(data)
                    except Exception as parse_exc:
                        # This is a *model output* problem (or our JSON extraction), not a transport failure.
                        # Log the raw text so we can confirm whether the API returned something.
                        if self.conversation_logger is not None and record_index is not None:
                            generation_time = time.time() - start
                            input_tokens = self.estimate_tokens(prompt_for_logging, model)
                            output_tokens = self.estimate_tokens(raw_response_text or "", model)
                            total_tokens = input_tokens + output_tokens
                            tps = total_tokens / generation_time if generation_time > 0 else 0.0
                            self.conversation_logger.log_conversation(
                                record_index=record_index,
                                record_id=record_id,
                                prompt_strategy=prompt_strategy,
                                perturbation=perturbation,
                                model_name=str(deployment),
                                method=method,
                                prompt_text=prompt_for_logging,
                                response_text=raw_response_text,
                                thinking_content="",
                                reasoning_effort=reasoning_effort,
                                reasoning_tokens=reasoning_tokens,
                                reasoning_summary=reasoning_summary_text,
                                input_tokens=int(input_tokens),
                                output_tokens=int(output_tokens),
                                total_tokens=int(total_tokens),
                                validation_success=False,
                                validation_error=f"{type(parse_exc).__name__}: {parse_exc}",
                                attempt_number=attempt + 1,
                                request_id=getattr(resp, "_request_id", request_id),
                                inference_time_seconds=generation_time,
                                tokens_per_second=tps,
                            )
                        raise

                input_tokens = in_toks if in_toks is not None else self.estimate_tokens(prompt_for_logging, model)
                output_tokens = out_toks if out_toks is not None else self.estimate_tokens(response_text, model)
                total_tokens = total_toks if total_toks is not None else input_tokens + output_tokens

                if self.conversation_logger is not None and record_index is not None:
                    generation_time = time.time() - start
                    tps = total_tokens / generation_time if generation_time > 0 else 0.0
                    self.conversation_logger.log_conversation(
                        record_index=record_index,
                        record_id=record_id,
                        prompt_strategy=prompt_strategy,
                        perturbation=perturbation,
                        model_name=str(deployment),
                        method=method,
                        prompt_text=prompt_for_logging,
                        response_text=response_text,
                        thinking_content="",
                        reasoning_effort=reasoning_effort,
                        reasoning_tokens=reasoning_tokens,
                        reasoning_summary=reasoning_summary_text,
                        input_tokens=int(input_tokens),
                        output_tokens=int(output_tokens),
                        total_tokens=int(total_tokens),
                        validation_success=True,
                        attempt_number=attempt + 1,
                        request_id=getattr(resp, "_request_id", request_id),
                        inference_time_seconds=generation_time,
                        tokens_per_second=tps,
                    )

                return response_text

            except retryable_exc_types as e:
                # If caller requested fewer retries, still ensure enough attempts for rate limits.
                if _is_retryable_error(e):
                    msg_l = str(e).lower()
                    is_rate_limit = (
                        "rate limit" in msg_l
                        or "too many requests" in msg_l
                        or "429" in msg_l
                        or "capacity exceeded" in msg_l
                        or "capacity_exceeded" in msg_l
                    )
                    if is_rate_limit:
                        max_attempts = max(max_attempts, min_rate_limit_retries)

                logger.warning(f"OpenAIV1Backend attempt {attempt + 1}/{max_attempts} failed: {type(e).__name__}: {e}")

                if self.conversation_logger is not None and record_index is not None:
                    input_tokens = self.estimate_tokens(prompt_for_logging, model)
                    self.conversation_logger.log_conversation(
                        record_index=record_index,
                        record_id=record_id,
                        prompt_strategy=prompt_strategy,
                        perturbation=perturbation,
                        model_name=str(deployment),
                        method=method,
                        prompt_text=prompt_for_logging,
                        response_text="",
                        thinking_content="",
                        reasoning_effort=reasoning_effort,
                        reasoning_tokens=None,
                        reasoning_summary="",
                        input_tokens=input_tokens,
                        output_tokens=0,
                        total_tokens=input_tokens,
                        validation_success=False,
                        validation_error=f"{type(e).__name__}: {e}",
                        attempt_number=attempt + 1,
                        request_id=request_id,
                    )

                if attempt >= max_attempts - 1:
                    raise

                if _is_retryable_error(e):
                    await asyncio.sleep(_retry_sleep_seconds(e, attempt))
                else:
                    raise

            except Exception as e:
                logger.warning(f"OpenAIV1Backend attempt {attempt + 1}/{max_attempts} failed: {type(e).__name__}: {e}")

                if self.conversation_logger is not None and record_index is not None:
                    input_tokens = self.estimate_tokens(prompt_for_logging, model)
                    self.conversation_logger.log_conversation(
                        record_index=record_index,
                        record_id=record_id,
                        prompt_strategy=prompt_strategy,
                        perturbation=perturbation,
                        model_name=str(deployment),
                        method=method,
                        prompt_text=prompt_for_logging,
                        response_text="",
                        thinking_content="",
                        reasoning_effort=reasoning_effort,
                        reasoning_tokens=None,
                        reasoning_summary="",
                        input_tokens=input_tokens,
                        output_tokens=0,
                        total_tokens=input_tokens,
                        validation_success=False,
                        validation_error=f"{type(e).__name__}: {e}",
                        attempt_number=attempt + 1,
                        request_id=request_id,
                    )

                raise

            attempt += 1

        return ""

    async def generate_responses_concurrent(self, requests: list[LLMRequest]) -> list[str]:
        return await gather_concurrent(self, requests)

    def estimate_tokens(self, text: str, model: ModelConfig) -> int:
        hint = None
        if model.azure and model.azure.model_hint:
            hint = model.azure.model_hint
        return count_tokens(text, model=hint or "gpt-4o-mini")
