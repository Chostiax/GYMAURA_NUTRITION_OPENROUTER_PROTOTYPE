from __future__ import annotations

import os
import time
from dataclasses import dataclass

import httpx
from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class ProviderConfig:
    name: str
    api_key_env: str
    base_url_env: str
    default_base_url: str
    default_model_env: str
    input_cost_env: str
    output_cost_env: str


PROVIDER_CONFIGS = {
    "openai": ProviderConfig(
        name="openai",
        api_key_env="OPENAI_API_KEY",
        base_url_env="OPENAI_BASE_URL",
        default_base_url="https://api.openai.com/v1",
        default_model_env="OPENAI_SMART_MODEL",
        input_cost_env="OPENAI_INPUT_COST_PER_1M",
        output_cost_env="OPENAI_OUTPUT_COST_PER_1M",
    ),
    "openrouter_deepseek": ProviderConfig(
        name="openrouter_deepseek",
        api_key_env="OPENROUTER_API_KEY",
        base_url_env="OPENROUTER_BASE_URL",
        default_base_url="https://openrouter.ai/api/v1",
        default_model_env="OPENROUTER_DEEPSEEK_MODEL",
        input_cost_env="OPENROUTER_DEEPSEEK_INPUT_COST_PER_1M",
        output_cost_env="OPENROUTER_DEEPSEEK_OUTPUT_COST_PER_1M",
    ),
}


def _get_provider_config(provider: str) -> ProviderConfig:
    if provider not in PROVIDER_CONFIGS:
        raise ValueError(f"Unsupported provider: {provider}")
    return PROVIDER_CONFIGS[provider]


def _normalize_usage_openai(payload: dict) -> dict | None:
    usage = payload.get("usage")
    if not isinstance(usage, dict):
        return None

    input_tokens = usage.get("input_tokens")
    output_tokens = usage.get("output_tokens")
    total_tokens = usage.get("total_tokens")

    return {
        "prompt_tokens": input_tokens,
        "completion_tokens": output_tokens,
        "total_tokens": total_tokens,
    }


def _normalize_usage_chat(payload: dict) -> dict | None:
    usage = payload.get("usage")
    if not isinstance(usage, dict):
        return None

    return {
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "total_tokens": usage.get("total_tokens"),
    }


def _estimate_cost_usd(provider: str, usage: dict | None) -> float | None:
    if usage is None:
        return None

    config = _get_provider_config(provider)

    try:
        prompt_tokens = float(usage.get("prompt_tokens") or 0)
        completion_tokens = float(usage.get("completion_tokens") or 0)

        input_cost_per_1m = float(os.getenv(config.input_cost_env, "0"))
        output_cost_per_1m = float(os.getenv(config.output_cost_env, "0"))

        if input_cost_per_1m <= 0 and output_cost_per_1m <= 0:
            return None

        cost = (
            (prompt_tokens / 1_000_000) * input_cost_per_1m
            + (completion_tokens / 1_000_000) * output_cost_per_1m
        )
        return round(cost, 8)
    except Exception:
        return None


def _openai_response_text(data: dict) -> str:
    output_text = data.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()

    output = data.get("output", [])
    parts = []
    for item in output:
        if not isinstance(item, dict):
            continue
        content = item.get("content", [])
        for block in content:
            if isinstance(block, dict) and block.get("type") == "output_text":
                text = block.get("text")
                if text:
                    parts.append(str(text))
    return "\n".join(parts).strip()


def chat_completion(
    provider: str,
    system_prompt: str,
    user_prompt: str,
    model: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = 300,
    timeout: float = 60.0,
    retries: int = 2,
) -> dict:
    config = _get_provider_config(provider)

    api_key = os.getenv(config.api_key_env)
    if not api_key:
        raise ValueError(f"Missing API key: {config.api_key_env}")

    base_url = os.getenv(config.base_url_env, config.default_base_url).rstrip("/")
    selected_model = model or os.getenv(config.default_model_env)

    if not selected_model:
        raise ValueError(f"Missing default model env: {config.default_model_env}")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    if provider == "openrouter_deepseek":
        referer = os.getenv("OPENROUTER_HTTP_REFERER")
        title = os.getenv("OPENROUTER_APP_TITLE")
        if referer:
            headers["HTTP-Referer"] = referer
        if title:
            headers["X-Title"] = title

    last_error = None

    for attempt in range(retries + 1):
        try:
            if provider == "openai":
                url = f"{base_url}/responses"
                payload = {
                    "model": selected_model,
                    "input": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "max_output_tokens": max_tokens,
                }

                response = httpx.post(url, headers=headers, json=payload, timeout=timeout)
                response.raise_for_status()
                data = response.json()

                raw_output = _openai_response_text(data)
                usage = _normalize_usage_openai(data)

            else:
                url = f"{base_url}/chat/completions"
                payload = {
                    "model": selected_model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                }

                response = httpx.post(url, headers=headers, json=payload, timeout=timeout)
                response.raise_for_status()
                data = response.json()

                choices = data.get("choices", [])
                if not choices:
                    raise ValueError("No choices returned from provider")

                raw_output = choices[0].get("message", {}).get("content", "")
                raw_output = str(raw_output).strip()
                usage = _normalize_usage_chat(data)

            estimated_cost_usd = _estimate_cost_usd(provider, usage)

            return {
                "provider": provider,
                "model": selected_model,
                "raw_output": raw_output,
                "usage": usage,
                "estimated_cost_usd": estimated_cost_usd,
                "payload": data,
            }

        except Exception as e:
            last_error = e
            try:
                if isinstance(e, httpx.HTTPStatusError):
                    response_text = e.response.text
                    last_error = RuntimeError(
                        f"{provider} request failed with status {e.response.status_code}: {response_text}"
                    )
            except Exception:
                pass

            if attempt < retries:
                time.sleep(1.0 * (attempt + 1))
            else:
                raise last_error