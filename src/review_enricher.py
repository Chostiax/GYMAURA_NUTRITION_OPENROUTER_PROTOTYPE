from __future__ import annotations

import os

from dotenv import load_dotenv

from src.llm_provider import chat_completion

load_dotenv()

DEFAULT_PROVIDER = os.getenv("SMART_GROWTH_PROVIDER", "openai")
DEFAULT_MODEL = os.getenv("SMART_GROWTH_MODEL")

SYSTEM_PROMPT = """
You enrich one unmatched food candidate for a nutrition application.

You will receive:
- original user meal description
- extracted food candidate
- estimated grams if available

Return ONLY this exact format:
canonical_food_text;category_id;grams;calories;protein_g;carbs_g;fat_g

Rules:
- canonical_food_text must be in English
- keep dishes and recipes as ONE item
- do not decompose
- choose the most reasonable USDA-style category_id
- if grams are unrealistic or missing, correct them to a realistic serving size
- all numeric values must be realistic
- no JSON
- no explanation
- no extra text
""".strip()


def _is_valid_line(line: str) -> bool:
    parts = [p.strip() for p in line.split(";")]
    if len(parts) != 7:
        return False

    if not parts[0]:
        return False

    try:
        int(float(parts[1]))
        float(parts[2])
        float(parts[3])
        float(parts[4])
        float(parts[5])
        float(parts[6])
        return True
    except Exception:
        return False


def _parse_output(raw_text: str) -> dict | None:
    text = (raw_text or "").strip()
    if not text:
        return None

    lines = [line.strip() for line in text.splitlines() if line.strip()]

    selected = None
    if _is_valid_line(text):
        selected = text
    else:
        for line in lines:
            if _is_valid_line(line):
                selected = line
                break

    if selected is None:
        return None

    parts = [p.strip() for p in selected.split(";")]

    try:
        return {
            "canonical_food_text": parts[0],
            "category_id": int(float(parts[1])),
            "grams": round(float(parts[2]), 2),
            "nutrition": {
                "calories": round(float(parts[3]), 2),
                "protein_g": round(float(parts[4]), 2),
                "carbs_g": round(float(parts[5]), 2),
                "fat_g": round(float(parts[6]), 2),
            },
        }
    except Exception:
        return None


def enrich_unmatched_food_with_llm(
    food_text: str,
    grams: float | None,
    original_text: str,
    provider: str | None = None,
    model: str | None = None,
) -> dict:
    selected_provider = provider or DEFAULT_PROVIDER
    selected_model = model or DEFAULT_MODEL

    user_prompt = f"""
Original meal description: {original_text}
Extracted food candidate: {food_text}
Estimated grams: {grams if grams is not None else "unknown"}
""".strip()

    response = chat_completion(
        provider=selected_provider,
        model=selected_model,
        system_prompt=SYSTEM_PROMPT,
        user_prompt=user_prompt,
        temperature=0.0,
        max_tokens=180,
    )

    parsed = _parse_output(response["raw_output"])

    return {
        "parsed": parsed,
        "raw_output": response["raw_output"],
        "usage": response["usage"],
        "estimated_cost_usd": response["estimated_cost_usd"],
        "provider": response["provider"],
        "model": response["model"],
    }