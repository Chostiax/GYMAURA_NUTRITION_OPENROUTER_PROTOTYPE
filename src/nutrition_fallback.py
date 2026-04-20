from __future__ import annotations

import os

from dotenv import load_dotenv

from src.llm_provider import chat_completion

load_dotenv()

DEFAULT_PROVIDER = os.getenv("SMART_FALLBACK_PROVIDER", "openai")
DEFAULT_MODEL = os.getenv("SMART_FALLBACK_MODEL")


SYSTEM_PROMPT = """
You estimate nutrition values for one food item.

You will receive:
- food name
- estimated grams if available
- original meal description for context

Return ONLY this exact format:
calories;protein_g;carbs_g;fat_g

Rules:
- All values must be numeric.
- Estimate TOTAL nutrition for the given item and quantity.
- If grams are missing, infer a realistic serving size from context.
- No explanation.
- No JSON.
- No extra text.
""".strip()


def _is_numeric_nutrition_line(line: str) -> bool:
    parts = [p.strip() for p in line.split(";")]
    if len(parts) != 4:
        return False

    for part in parts:
        try:
            float(part)
        except ValueError:
            return False

    return True


def _parse_nutrition_line(raw_text: str) -> dict | None:
    """
    Handles:
    - single-line responses
    - header + data responses
    - extra blank lines
    """
    text = (raw_text or "").strip()
    if not text:
        return None

    lines = [line.strip() for line in text.splitlines() if line.strip()]

    if _is_numeric_nutrition_line(text):
        selected = text
    else:
        selected = None
        for line in lines:
            if _is_numeric_nutrition_line(line):
                selected = line
                break

    if selected is None:
        return None

    try:
        calories, protein_g, carbs_g, fat_g = [float(p.strip()) for p in selected.split(";")]
        return {
            "calories": round(calories, 2),
            "protein_g": round(protein_g, 2),
            "carbs_g": round(carbs_g, 2),
            "fat_g": round(fat_g, 2),
        }
    except Exception:
        return None


def estimate_nutrition_with_llm(
    food_text: str,
    grams: float | None,
    original_text: str,
    provider: str | None = None,
    model: str | None = None,
) -> dict:
    selected_provider = provider or DEFAULT_PROVIDER
    selected_model = model or DEFAULT_MODEL

    user_prompt = f"""
Food: {food_text}
Estimated grams: {grams if grams is not None else "unknown"}
Original meal description: {original_text}
""".strip()

    response = chat_completion(
        provider=selected_provider,
        model=selected_model,
        system_prompt=SYSTEM_PROMPT,
        user_prompt=user_prompt,
        temperature=0.0,
        max_tokens=120,
    )

    parsed = _parse_nutrition_line(response["raw_output"])

    if parsed is None:
        parsed = {
            "calories": 0.0,
            "protein_g": 0.0,
            "carbs_g": 0.0,
            "fat_g": 0.0,
        }

    return {
        "nutrition": parsed,
        "raw_output": response["raw_output"],
        "usage": response["usage"],
        "estimated_cost_usd": response["estimated_cost_usd"],
        "provider": response["provider"],
        "model": response["model"],
    }