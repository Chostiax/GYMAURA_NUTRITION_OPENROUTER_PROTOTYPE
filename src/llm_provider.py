from __future__ import annotations

import os
import time
from typing import Any

import httpx
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

PROVIDER_CONFIG = {
    "deepseek_v32": {
        "model_env": "DEEPSEEK_V32_MODEL",
        "default_model": "deepseek/deepseek-v3.2",
        "input_cost_env": "DEEPSEEK_V32_INPUT_COST_PER_1M",
        "output_cost_env": "DEEPSEEK_V32_OUTPUT_COST_PER_1M",
        "default_input_cost": 0.259,
        "default_output_cost": 0.42,
    },
    "deepseek_v4_flash": {
        "model_env": "DEEPSEEK_V4_FLASH_MODEL",
        "default_model": "deepseek/deepseek-v4-flash",
        "input_cost_env": "DEEPSEEK_V4_FLASH_INPUT_COST_PER_1M",
        "output_cost_env": "DEEPSEEK_V4_FLASH_OUTPUT_COST_PER_1M",
        "default_input_cost": 0.14,
        "default_output_cost": 0.28,
    },
}


def _get_float_env(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except Exception:
        return default


def _estimate_cost_usd(provider: str, usage: dict[str, Any] | None) -> float | None:
    if not usage:
        return None

    config = PROVIDER_CONFIG[provider]

    prompt_tokens = usage.get("prompt_tokens") or usage.get("input_tokens") or 0
    completion_tokens = usage.get("completion_tokens") or usage.get("output_tokens") or 0

    input_cost = _get_float_env(config["input_cost_env"], config["default_input_cost"])
    output_cost = _get_float_env(config["output_cost_env"], config["default_output_cost"])

    cost = (prompt_tokens / 1_000_000) * input_cost
    cost += (completion_tokens / 1_000_000) * output_cost

    return round(cost, 10)


def _extract_cached_tokens(usage: dict[str, Any] | None) -> int:
    if not usage:
        return 0

    details = usage.get("prompt_tokens_details") or {}
    return int(details.get("cached_tokens") or 0)


def _extract_cache_write_tokens(usage: dict[str, Any] | None) -> int:
    if not usage:
        return 0

    details = usage.get("prompt_tokens_details") or {}
    return int(details.get("cache_write_tokens") or 0)


def _extract_content(payload: dict[str, Any]) -> str:
    choices = payload.get("choices") or []
    if not choices:
        return ""

    message = choices[0].get("message") or {}
    content = message.get("content", "")

    if content is None:
        return ""

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    parts.append(str(block.get("text", "")))
                elif "text" in block:
                    parts.append(str(block.get("text", "")))
        return "\n".join(parts).strip()

    return str(content).strip()


def chat_completion(
    *,
    provider: str,
    model: str | None,
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.0,
    max_tokens: int = 200,
    response_format: dict[str, Any] | None = None,
    extra_body: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if provider not in PROVIDER_CONFIG:
        raise ValueError(f"Unsupported provider: {provider}")

    if not OPENROUTER_API_KEY:
        raise ValueError("Missing OPENROUTER_API_KEY in environment.")

    config = PROVIDER_CONFIG[provider]
    selected_model = model or os.getenv(config["model_env"], config["default_model"])

    body: dict[str, Any] = {
        "model": selected_model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    if response_format is not None:
        body["response_format"] = response_format

    if extra_body:
        body.update(extra_body)

    started_at = time.perf_counter()

    response = httpx.post(
        f"{OPENROUTER_BASE_URL}/chat/completions",
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        },
        json=body,
        timeout=90.0,
    )

    elapsed_ms = round((time.perf_counter() - started_at) * 1000, 2)

    if response.status_code >= 400:
        raise RuntimeError(
            f"{provider} request failed with status {response.status_code}: {response.text}"
        )

    payload = response.json()
    usage = payload.get("usage") or {}
    raw_output = _extract_content(payload) or ""

    return {
        "raw_output": raw_output,
        "usage": usage,
        "estimated_cost_usd": _estimate_cost_usd(provider, usage),
        "provider": provider,
        "model": selected_model,
        "elapsed_ms": elapsed_ms,
        "prompt_tokens": usage.get("prompt_tokens") or usage.get("input_tokens") or 0,
        "completion_tokens": usage.get("completion_tokens") or usage.get("output_tokens") or 0,
        "cached_tokens": _extract_cached_tokens(usage),
        "cache_write_tokens": _extract_cache_write_tokens(usage),
        "payload_id": payload.get("id"),
    }