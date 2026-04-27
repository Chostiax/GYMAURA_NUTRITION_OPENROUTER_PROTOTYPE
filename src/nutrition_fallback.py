from __future__ import annotations

import json
import os
import re
from dotenv import load_dotenv

from src.llm_provider import chat_completion

load_dotenv()

DEFAULT_PROVIDER = os.getenv("SMART_FALLBACK_PROVIDER", "deepseek_v32")

SYSTEM_PROMPT = """
You estimate nutrition for one food item.

Return nutrition for the TOTAL quantity only.

Do not include explanations.
""".strip()


NUTRITION_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "nutrition_estimate",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "calories": {"type": "number"},
                "protein_g": {"type": "number"},
                "carbs_g": {"type": "number"},
                "fat_g": {"type": "number"},
            },
            "required": ["calories", "protein_g", "carbs_g", "fat_g"],
            "additionalProperties": False,
        },
    },
}


def _parse_json_nutrition(raw_text: str) -> dict | None:
    text = (raw_text or "").strip()
    if not text:
        return None

    try:
        data = json.loads(text)
    except Exception:
        return None

    try:
        values = {
            "calories": float(data["calories"]),
            "protein_g": float(data["protein_g"]),
            "carbs_g": float(data["carbs_g"]),
            "fat_g": float(data["fat_g"]),
        }
    except Exception:
        return None

    if min(values.values()) < 0:
        return None

    return {k: round(v, 2) for k, v in values.items()}


def _parse_semicolon_or_numbers(raw_text: str) -> dict | None:
    text = (raw_text or "").strip()
    if not text:
        return None

    lowered = text.lower()
    if "error" in lowered and "message" in lowered:
        return None

    text = text.replace("```", "").replace(",", ".").strip()

    for line in text.splitlines():
        parts = [p.strip() for p in line.split(";")]
        if len(parts) == 4:
            nums = []
            for part in parts:
                match = re.search(r"-?\d+(?:\.\d+)?", part)
                if not match:
                    nums = []
                    break
                nums.append(float(match.group(0)))

            if len(nums) == 4 and min(nums) >= 0:
                return {
                    "calories": round(nums[0], 2),
                    "protein_g": round(nums[1], 2),
                    "carbs_g": round(nums[2], 2),
                    "fat_g": round(nums[3], 2),
                }

    nums = re.findall(r"-?\d+(?:\.\d+)?", text)
    if len(nums) >= 4:
        values = list(map(float, nums[:4]))
        if min(values) >= 0:
            return {
                "calories": round(values[0], 2),
                "protein_g": round(values[1], 2),
                "carbs_g": round(values[2], 2),
                "fat_g": round(values[3], 2),
            }

    return None


def _parse_nutrition(raw_text: str) -> dict | None:
    return _parse_json_nutrition(raw_text) or _parse_semicolon_or_numbers(raw_text)


def _user_prompt(food_text: str, grams: float | None, original_text: str, category_id: int | None, category_name: str | None) -> str:
    return f"""
Food: {food_text}
Quantity: {grams if grams is not None else "unknown"} grams
Category ID: {category_id if category_id is not None else "unknown"}
Category name: {category_name if category_name else "unknown"}
Original input: {original_text}

Estimate TOTAL nutrition for the food quantity.
""".strip()


def estimate_nutrition_with_llm(
    food_text: str,
    grams: float | None,
    original_text: str,
    provider: str | None = None,
    model: str | None = None,
    category_id: int | None = None,
    category_name: str | None = None,
) -> dict:
    selected_provider = provider or DEFAULT_PROVIDER
    selected_model = model

    prompt = _user_prompt(food_text, grams, original_text, category_id, category_name)

    total_cost = 0.0
    total_elapsed_ms = 0.0
    last_raw_output = ""
    last_usage = None
    last_provider = selected_provider
    last_model = selected_model

    # Attempt 1: structured output
    try:
        response = chat_completion(
            provider=selected_provider,
            model=selected_model,
            system_prompt=SYSTEM_PROMPT,
            user_prompt=prompt,
            temperature=0.0,
            max_tokens=120,
            response_format=NUTRITION_RESPONSE_FORMAT,
        )

        last_raw_output = response.get("raw_output", "")
        last_usage = response.get("usage")
        last_provider = response.get("provider")
        last_model = response.get("model")
        total_cost += response.get("estimated_cost_usd") or 0.0
        total_elapsed_ms += response.get("elapsed_ms") or 0.0

        parsed = _parse_nutrition(last_raw_output)
        if parsed is not None:
            return {
                "nutrition": parsed,
                "raw_output": last_raw_output,
                "usage": last_usage,
                "estimated_cost_usd": total_cost,
                "provider": last_provider,
                "model": last_model,
                "elapsed_ms": round(total_elapsed_ms, 2),
                "used_structured_output": True,
                "used_second_attempt": False,
            }

    except Exception as structured_error:
        last_raw_output = str(structured_error)

    # Attempt 2: semicolon strict fallback
    strict_prompt = f"""
Estimate TOTAL nutrition for:
{grams if grams is not None else "unknown"} grams of {food_text}

Return exactly:
calories;protein_g;carbs_g;fat_g

Only numbers and semicolons. No words. No JSON. No units.
""".strip()

    response = chat_completion(
        provider=selected_provider,
        model=selected_model,
        system_prompt="Return exactly four numeric values separated by semicolons.",
        user_prompt=strict_prompt,
        temperature=0.0,
        max_tokens=60,
    )

    last_raw_output = response.get("raw_output", "")
    last_usage = response.get("usage")
    last_provider = response.get("provider")
    last_model = response.get("model")
    total_cost += response.get("estimated_cost_usd") or 0.0
    total_elapsed_ms += response.get("elapsed_ms") or 0.0

    parsed = _parse_nutrition(last_raw_output)
    if parsed is not None:
        return {
            "nutrition": parsed,
            "raw_output": last_raw_output,
            "usage": last_usage,
            "estimated_cost_usd": total_cost,
            "provider": last_provider,
            "model": last_model,
            "elapsed_ms": round(total_elapsed_ms, 2),
            "used_structured_output": False,
            "used_second_attempt": True,
        }

    raise RuntimeError(
        f"Fallback nutrition LLM failed to return parseable nutrition. Last raw output: {last_raw_output}"
    )