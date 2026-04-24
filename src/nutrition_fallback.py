from __future__ import annotations

import os
import re
from dotenv import load_dotenv

from src.llm_provider import chat_completion

load_dotenv()

DEFAULT_PROVIDER = os.getenv("SMART_FALLBACK_PROVIDER", "openrouter_deepseek")

SYSTEM_PROMPT = """
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

Example:
180;5;22;8
""".strip()


def _parse_nutrition(raw_text: str) -> dict | None:
    text = (raw_text or "").strip()
    if not text:
        return None

    text = text.replace("```", "").strip()
    text = text.replace(",", ".")

    # First: strict semicolon format
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue

        parts = [p.strip() for p in line.split(";")]
        if len(parts) == 4:
            values = []
            for part in parts:
                match = re.search(r"-?\d+(?:\.\d+)?", part)
                if not match:
                    values = []
                    break
                values.append(float(match.group(0)))

            if len(values) == 4:
                calories, protein_g, carbs_g, fat_g = values
                if min(values) >= 0:
                    return {
                        "calories": round(calories, 2),
                        "protein_g": round(protein_g, 2),
                        "carbs_g": round(carbs_g, 2),
                        "fat_g": round(fat_g, 2),
                    }

    # Second: tolerate labelled / JSON-ish / multiline output
    numbers = re.findall(r"-?\d+(?:\.\d+)?", text)
    if len(numbers) >= 4:
        values = list(map(float, numbers[:4]))
        calories, protein_g, carbs_g, fat_g = values

        if min(values) >= 0:
            return {
                "calories": round(calories, 2),
                "protein_g": round(protein_g, 2),
                "carbs_g": round(carbs_g, 2),
                "fat_g": round(fat_g, 2),
            }

    return None


def _call_llm(
    *,
    food_text: str,
    grams: float | None,
    original_text: str,
    provider: str,
    model: str | None,
    category_id: int | None,
    category_name: str | None,
) -> dict:
    user_prompt = f"""
Food: {food_text}
Quantity: {grams if grams is not None else "unknown"} grams
Category ID: {category_id if category_id is not None else "unknown"}
Category name: {category_name if category_name else "unknown"}
Original input: {original_text}

Return only:
calories;protein_g;carbs_g;fat_g
""".strip()

    return chat_completion(
        provider=provider,
        model=model,
        system_prompt=SYSTEM_PROMPT,
        user_prompt=user_prompt,
        temperature=0.0,
        max_tokens=80,
    )


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
    selected_model = model  # let llm_provider choose provider-specific model

    total_cost = 0.0
    last_raw_output = ""
    last_usage = None
    last_provider = selected_provider
    last_model = selected_model

    # Attempt 1
    response = _call_llm(
        food_text=food_text,
        grams=grams,
        original_text=original_text,
        provider=selected_provider,
        model=selected_model,
        category_id=category_id,
        category_name=category_name,
    )

    last_raw_output = response.get("raw_output", "")
    last_usage = response.get("usage")
    last_provider = response.get("provider")
    last_model = response.get("model")
    total_cost += response.get("estimated_cost_usd") or 0.0

    parsed = _parse_nutrition(last_raw_output)
    if parsed is not None:
        return {
            "nutrition": parsed,
            "raw_output": last_raw_output,
            "usage": last_usage,
            "estimated_cost_usd": total_cost,
            "provider": last_provider,
            "model": last_model,
            "used_repair_pass": False,
            "used_second_attempt": False,
        }

    # Attempt 2 with stricter prompt
    retry_prompt = f"""
Estimate TOTAL nutrition for:
{grams if grams is not None else "unknown"} grams of {food_text}

Return exactly 4 numbers separated by semicolons:
calories;protein_g;carbs_g;fat_g

No words. No JSON. No units.
""".strip()

    response = chat_completion(
        provider=selected_provider,
        model=selected_model,
        system_prompt=SYSTEM_PROMPT,
        user_prompt=retry_prompt,
        temperature=0.0,
        max_tokens=50,
    )

    last_raw_output = response.get("raw_output", "")
    last_usage = response.get("usage")
    last_provider = response.get("provider")
    last_model = response.get("model")
    total_cost += response.get("estimated_cost_usd") or 0.0

    parsed = _parse_nutrition(last_raw_output)
    if parsed is not None:
        return {
            "nutrition": parsed,
            "raw_output": last_raw_output,
            "usage": last_usage,
            "estimated_cost_usd": total_cost,
            "provider": last_provider,
            "model": last_model,
            "used_repair_pass": False,
            "used_second_attempt": True,
        }

    raise RuntimeError(
        f"Fallback nutrition LLM failed to return parseable nutrition. Last raw output: {last_raw_output}"
    )