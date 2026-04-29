from __future__ import annotations

import json
import os
import re

from dotenv import load_dotenv

from src.llm_provider import chat_completion

load_dotenv()

DEFAULT_PROVIDER = os.getenv("SMART_FALLBACK_PROVIDER", "deepseek_v32")

SYSTEM_PROMPT_JSON = """
You estimate nutrition for one food item.

Return ONLY valid JSON matching the required schema.
Estimate nutrition for the TOTAL quantity, not per 100g.
Do not include explanations, markdown, units, or extra keys.
""".strip()

SYSTEM_PROMPT_SEMICOLON = """
You estimate nutrition for one food item.

Return ONLY one line:
calories;protein_g;carbs_g;fat_g

Rules:
- numeric values only
- estimate TOTAL nutrition for the full quantity
- no units
- no labels
- no JSON
- no markdown
- no explanation
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


def _valid_nutrition(values: dict) -> dict | None:
    try:
        nutrition = {
            "calories": float(values["calories"]),
            "protein_g": float(values["protein_g"]),
            "carbs_g": float(values["carbs_g"]),
            "fat_g": float(values["fat_g"]),
        }
    except Exception:
        return None

    if min(nutrition.values()) < 0:
        return None

    # Guard against obvious broken outputs.
    # This is not hardcoded nutrition; it only rejects invalid model responses.
    if nutrition["calories"] == 0 and (
        nutrition["protein_g"] > 0 or nutrition["carbs_g"] > 0 or nutrition["fat_g"] > 0
    ):
        return None

    if nutrition["calories"] > 5000:
        return None

    return {key: round(value, 2) for key, value in nutrition.items()}


def _parse_json_nutrition(raw_text: str) -> dict | None:
    text = (raw_text or "").strip()
    if not text:
        return None

    lowered = text.lower()
    if "error" in lowered and "message" in lowered:
        return None

    try:
        data = json.loads(text)
    except Exception:
        return None

    if not isinstance(data, dict):
        return None

    return _valid_nutrition(data)


def _parse_semicolon_nutrition(raw_text: str) -> dict | None:
    text = (raw_text or "").strip()
    if not text:
        return None

    lowered = text.lower()
    if "error" in lowered and "message" in lowered:
        return None

    text = text.replace("```", "").replace(",", ".").strip()

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue

        parts = [part.strip() for part in line.split(";")]
        if len(parts) != 4:
            continue

        numbers = []
        for part in parts:
            match = re.search(r"-?\d+(?:\.\d+)?", part)
            if not match:
                numbers = []
                break
            numbers.append(float(match.group(0)))

        if len(numbers) == 4:
            return _valid_nutrition(
                {
                    "calories": numbers[0],
                    "protein_g": numbers[1],
                    "carbs_g": numbers[2],
                    "fat_g": numbers[3],
                }
            )

    # Last tolerant parser: first 4 numbers anywhere.
    numbers = re.findall(r"-?\d+(?:\.\d+)?", text)
    if len(numbers) >= 4:
        values = list(map(float, numbers[:4]))
        return _valid_nutrition(
            {
                "calories": values[0],
                "protein_g": values[1],
                "carbs_g": values[2],
                "fat_g": values[3],
            }
        )

    return None


def _build_json_prompt(
    *,
    food_text: str,
    grams: float | None,
    original_text: str,
    category_id: int | None,
    category_name: str | None,
) -> str:
    return f"""
Food: {food_text}
Quantity: {grams if grams is not None else "unknown"} grams
Category ID: {category_id if category_id is not None else "unknown"}
Category name: {category_name if category_name else "unknown"}
Original input: {original_text}

Estimate TOTAL nutrition for this full quantity.
Return only JSON with:
calories, protein_g, carbs_g, fat_g
""".strip()


def _build_semicolon_prompt(food_text: str, grams: float | None) -> str:
    return f"""
Estimate TOTAL nutrition for:
{grams if grams is not None else "unknown"} grams of {food_text}

Return exactly:
calories;protein_g;carbs_g;fat_g

Only numbers and semicolons.
No words.
No JSON.
No units.
""".strip()


def _merge_usage_and_cost(responses: list[dict]) -> dict:
    total_cost = 0.0
    total_elapsed_ms = 0.0
    prompt_tokens = 0
    completion_tokens = 0
    cached_tokens = 0
    cache_write_tokens = 0

    for response in responses:
        total_cost += response.get("estimated_cost_usd") or 0.0
        total_elapsed_ms += response.get("elapsed_ms") or 0.0
        prompt_tokens += response.get("prompt_tokens") or 0
        completion_tokens += response.get("completion_tokens") or 0
        cached_tokens += response.get("cached_tokens") or 0
        cache_write_tokens += response.get("cache_write_tokens") or 0

    return {
        "estimated_cost_usd": round(total_cost, 10),
        "elapsed_ms": round(total_elapsed_ms, 2),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "cached_tokens": cached_tokens,
        "cache_write_tokens": cache_write_tokens,
    }


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

    responses: list[dict] = []
    last_raw_output = ""

    json_prompt = _build_json_prompt(
        food_text=food_text,
        grams=grams,
        original_text=original_text,
        category_id=category_id,
        category_name=category_name,
    )

    # Attempt 1: JSON schema structured output.
    try:
        response = chat_completion(
            provider=selected_provider,
            model=selected_model,
            system_prompt=SYSTEM_PROMPT_JSON,
            user_prompt=json_prompt,
            temperature=0.0,
            max_tokens=120,
            response_format=NUTRITION_RESPONSE_FORMAT,
        )

        responses.append(response)
        last_raw_output = response.get("raw_output") or ""

        parsed = _parse_json_nutrition(last_raw_output)
        if parsed is not None:
            merged = _merge_usage_and_cost(responses)
            return {
                "nutrition": parsed,
                "raw_output": last_raw_output,
                "usage": response.get("usage"),
                "provider": response.get("provider"),
                "model": response.get("model"),
                "used_structured_output": True,
                "used_second_attempt": False,
                **merged,
            }

    except Exception as e:
        last_raw_output = str(e)

    # Attempt 2: strict semicolon backup.
    try:
        response = chat_completion(
            provider=selected_provider,
            model=selected_model,
            system_prompt=SYSTEM_PROMPT_SEMICOLON,
            user_prompt=_build_semicolon_prompt(food_text, grams),
            temperature=0.0,
            max_tokens=60,
        )

        responses.append(response)
        last_raw_output = response.get("raw_output") or ""

        parsed = _parse_semicolon_nutrition(last_raw_output)
        if parsed is not None:
            merged = _merge_usage_and_cost(responses)
            return {
                "nutrition": parsed,
                "raw_output": last_raw_output,
                "usage": response.get("usage"),
                "provider": response.get("provider"),
                "model": response.get("model"),
                "used_structured_output": False,
                "used_second_attempt": True,
                **merged,
            }

    except Exception as e:
        last_raw_output = str(e)

    merged = _merge_usage_and_cost(responses)

    raise RuntimeError(
        f"Fallback nutrition LLM failed to return parseable nutrition. Last raw output: {last_raw_output}. "
        f"Elapsed: {merged['elapsed_ms']} ms. Cost: ${merged['estimated_cost_usd']:.10f}"
    )