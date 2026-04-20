from __future__ import annotations

import os
import re
from dotenv import load_dotenv

from src.llm_provider import chat_completion

load_dotenv()

DEFAULT_PROVIDER = os.getenv("SMART_FALLBACK_PROVIDER", "openai")

SYSTEM_PROMPT = """
You estimate nutrition values for one food item.

You will receive:
- food name
- estimated grams if available
- original meal description for context
- category id if available
- category name if available

Return ONLY one line in this exact format:
calories;protein_g;carbs_g;fat_g

Rules:
- numeric values only
- estimate TOTAL nutrition for the full item quantity, not per 100g
- no units
- no labels
- no JSON
- no explanation
- no markdown

Example:
420;18;52;15
""".strip()

REPAIR_PROMPT = """
You are a formatting repair assistant.

You will receive a model output that was supposed to contain nutrition values.

Extract or infer the intended 4 nutrition values and return ONLY:
calories;protein_g;carbs_g;fat_g

Rules:
- numeric values only
- no units
- no labels
- no JSON
- no explanation
- no markdown
- exactly one line

Example:
420;18;52;15
""".strip()


def _extract_first_number(value: str) -> float | None:
    if value is None:
        return None

    text = str(value).strip().lower().replace(",", ".")
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    if not match:
        return None

    try:
        return float(match.group(0))
    except Exception:
        return None


def _parse_nutrition_line(raw_text: str) -> dict | None:
    """
    Accepts:
    - exact semicolon line
    - semicolon line with units
    - multi-line output where one line contains the values
    - extra text / code fences
    - fallback: first 4 numeric values found anywhere
    """
    text = (raw_text or "").strip()
    if not text:
        return None

    text = text.replace("```", "").strip()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    candidate_lines = [text] + lines

    for line in candidate_lines:
        parts = [p.strip() for p in line.split(";")]
        if len(parts) != 4:
            continue

        values = [_extract_first_number(part) for part in parts]
        if any(v is None for v in values):
            continue

        calories, protein_g, carbs_g, fat_g = values

        if min(calories, protein_g, carbs_g, fat_g) < 0:
            continue

        return {
            "calories": round(calories, 2),
            "protein_g": round(protein_g, 2),
            "carbs_g": round(carbs_g, 2),
            "fat_g": round(fat_g, 2),
        }

    numbers = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", "."))
    if len(numbers) >= 4:
        try:
            calories, protein_g, carbs_g, fat_g = map(float, numbers[:4])

            if min(calories, protein_g, carbs_g, fat_g) < 0:
                return None

            return {
                "calories": round(calories, 2),
                "protein_g": round(protein_g, 2),
                "carbs_g": round(carbs_g, 2),
                "fat_g": round(fat_g, 2),
            }
        except Exception:
            return None

    return None


def _call_nutrition_llm(
    *,
    provider: str,
    model: str | None,
    food_text: str,
    grams: float | None,
    original_text: str,
    category_id: int | None,
    category_name: str | None,
) -> dict:
    user_prompt = f"""
Food: {food_text}
Estimated grams: {grams if grams is not None else "unknown"}
Category ID: {category_id if category_id is not None else "unknown"}
Category name: {category_name if category_name else "unknown"}
Original meal description: {original_text}
""".strip()

    return chat_completion(
        provider=provider,
        model=model,
        system_prompt=SYSTEM_PROMPT,
        user_prompt=user_prompt,
        temperature=0.0,
        max_tokens=120,
    )


def _repair_with_llm(
    *,
    provider: str,
    model: str | None,
    raw_output: str,
) -> dict:
    user_prompt = f"""
Broken model output:
{raw_output}
""".strip()

    return chat_completion(
        provider=provider,
        model=model,
        system_prompt=REPAIR_PROMPT,
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
    """
    No hardcoded nutrition backup.
    Flow:
    1) main nutrition call
    2) parse directly
    3) retry once if needed
    4) repair pass with same provider/model if needed
    5) if all fail, raise explicit error
    """
    selected_provider = provider or DEFAULT_PROVIDER
    selected_model = model  # let llm_provider choose provider-specific default

    total_estimated_cost = 0.0
    last_usage = None

    # First attempt
    first = _call_nutrition_llm(
        provider=selected_provider,
        model=selected_model,
        food_text=food_text,
        grams=grams,
        original_text=original_text,
        category_id=category_id,
        category_name=category_name,
    )
    raw_output = first.get("raw_output", "")
    total_estimated_cost += first.get("estimated_cost_usd") or 0.0
    last_usage = first.get("usage")

    parsed = _parse_nutrition_line(raw_output)
    if parsed is not None:
        return {
            "nutrition": parsed,
            "raw_output": raw_output,
            "usage": last_usage,
            "estimated_cost_usd": total_estimated_cost,
            "provider": first.get("provider"),
            "model": first.get("model"),
            "used_repair_pass": False,
            "used_second_attempt": False,
        }

    # Second attempt
    second = _call_nutrition_llm(
        provider=selected_provider,
        model=selected_model,
        food_text=food_text,
        grams=grams,
        original_text=original_text,
        category_id=category_id,
        category_name=category_name,
    )
    raw_output_second = second.get("raw_output", "")
    total_estimated_cost += second.get("estimated_cost_usd") or 0.0
    last_usage = second.get("usage")

    parsed = _parse_nutrition_line(raw_output_second)
    if parsed is not None:
        return {
            "nutrition": parsed,
            "raw_output": raw_output_second,
            "usage": last_usage,
            "estimated_cost_usd": total_estimated_cost,
            "provider": second.get("provider"),
            "model": second.get("model"),
            "used_repair_pass": False,
            "used_second_attempt": True,
        }

    # Repair pass using same provider/model
    repair_input = raw_output_second or raw_output
    repair = _repair_with_llm(
        provider=selected_provider,
        model=selected_model,
        raw_output=repair_input,
    )
    repaired_output = repair.get("raw_output", "")
    total_estimated_cost += repair.get("estimated_cost_usd") or 0.0
    last_usage = repair.get("usage")

    parsed = _parse_nutrition_line(repaired_output)
    if parsed is not None:
        return {
            "nutrition": parsed,
            "raw_output": repaired_output,
            "usage": last_usage,
            "estimated_cost_usd": total_estimated_cost,
            "provider": repair.get("provider"),
            "model": repair.get("model"),
            "used_repair_pass": True,
            "used_second_attempt": True,
        }

    raise RuntimeError(
        "Fallback nutrition LLM failed to return parseable nutrition after retry and repair pass."
    )