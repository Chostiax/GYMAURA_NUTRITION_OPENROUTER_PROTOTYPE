from __future__ import annotations

import os

import httpx
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
DEFAULT_MODEL = os.getenv("OPENROUTER_MODEL", "google/gemma-3-4b-it")

SYSTEM_PROMPT = """
You estimate nutrition values for one food item.

You will receive:
- food name
- estimated grams if available
- original user meal description for context

Return ONLY this exact format:
calories;protein_g;carbs_g;fat_g

Rules:
- All values must be numeric.
- Estimate total nutrition for the given item and quantity.
- If grams are missing, infer a realistic serving size from context.
- No explanation.
- No JSON.
- No extra text.
""".strip()


def _parse_nutrition_line(raw_text: str) -> dict | None:
    text = (raw_text or "").strip()
    parts = [p.strip() for p in text.split(";")]

    if len(parts) != 4:
        return None

    try:
        return {
            "calories": round(float(parts[0]), 2),
            "protein_g": round(float(parts[1]), 2),
            "carbs_g": round(float(parts[2]), 2),
            "fat_g": round(float(parts[3]), 2),
        }
    except ValueError:
        return None


def estimate_nutrition_with_llm(
    food_text: str,
    grams: float | None,
    original_text: str,
    model: str | None = None,
) -> dict:
    if not OPENROUTER_API_KEY:
        raise ValueError("Missing OPENROUTER_API_KEY in environment.")

    selected_model = model or DEFAULT_MODEL

    user_prompt = f"""
Food: {food_text}
Estimated grams: {grams if grams is not None else "unknown"}
Original meal description: {original_text}
""".strip()

    response = httpx.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": selected_model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0,
            "max_tokens": 80,
        },
        timeout=60.0,
    )

    response.raise_for_status()
    payload = response.json()

    raw_output = (
        payload.get("choices", [{}])[0]
        .get("message", {})
        .get("content", "")
        .strip()
    )

    parsed = _parse_nutrition_line(raw_output)

    if parsed is None:
        # emergency fallback: never return None to user
        parsed = {
            "calories": 0.0,
            "protein_g": 0.0,
            "carbs_g": 0.0,
            "fat_g": 0.0,
        }

    return {
        "nutrition": parsed,
        "raw_output": raw_output,
        "usage": payload.get("usage"),
        "model": selected_model,
    }