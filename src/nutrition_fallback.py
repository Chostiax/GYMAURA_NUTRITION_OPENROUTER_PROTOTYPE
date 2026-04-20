from __future__ import annotations

import os
import re
from dotenv import load_dotenv

from src.llm_provider import chat_completion

load_dotenv()

# Keep only provider as a default fallback if the UI/code does not pass one.
# DO NOT set a global default model here.
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


def _extract_first_number(value: str) -> float | None:
    if value is None:
        return None

    text = str(value).strip().lower()
    text = text.replace(",", ".")

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
    - semicolon line with units, e.g. '320 kcal; 12 g; 38 g; 9 g'
    - multi-line output where one line contains the values
    - code fences / extra text
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

        if calories < 0 or protein_g < 0 or carbs_g < 0 or fat_g < 0:
            continue

        return {
            "calories": round(calories, 2),
            "protein_g": round(protein_g, 2),
            "carbs_g": round(carbs_g, 2),
            "fat_g": round(fat_g, 2),
        }

    return None


def _heuristic_backup(
    food_text: str,
    grams: float | None,
    category_id: int | None = None,
    category_name: str | None = None,
) -> dict:
    """
    Last-resort deterministic estimate.
    This guarantees the app never returns empty nutrition.
    """
    g = float(grams) if grams is not None and grams > 0 else 200.0
    factor = g / 100.0

    lowered = (food_text or "").strip().lower()
    cat = (category_name or "").strip().lower()

    # Default profile = mixed prepared dish
    kcal_100 = 180.0
    protein_100 = 7.0
    carbs_100 = 18.0
    fat_100 = 7.0

    # Near-zero / very light beverages
    if "tea" in lowered or "coffee" in lowered:
        kcal_100 = 8.0
        protein_100 = 0.2
        carbs_100 = 1.5
        fat_100 = 0.1

    # Sweet / milky beverages
    elif any(x in lowered for x in ["lassi", "horchata", "bubble tea", "smoothie", "juice"]):
        kcal_100 = 60.0
        protein_100 = 1.5
        carbs_100 = 12.0
        fat_100 = 1.0

    elif "beverage" in cat:
        kcal_100 = 25.0
        protein_100 = 0.3
        carbs_100 = 5.0
        fat_100 = 0.1

    # Desserts / sweets
    elif any(x in lowered for x in ["halva", "donut", "mochi", "cake", "dessert", "affogato"]):
        kcal_100 = 360.0
        protein_100 = 5.0
        carbs_100 = 45.0
        fat_100 = 16.0

    # Common savory mixed dishes
    elif any(
        x in lowered
        for x in [
            "tagine",
            "shawarma",
            "biryani",
            "risotto",
            "butter chicken",
            "dal makhani",
            "bibimbap",
            "poutine",
            "fish and chips",
            "okonomiyaki",
            "couscous",
            "empanada",
            "salad",
            "pizza",
            "ramen",
            "pad thai",
            "pho",
            "dumplings",
            "poke bowl",
            "burrito",
            "wrap",
        ]
    ):
        kcal_100 = 175.0
        protein_100 = 8.0
        carbs_100 = 17.0
        fat_100 = 8.0

    return {
        "calories": round(kcal_100 * factor, 2),
        "protein_g": round(protein_100 * factor, 2),
        "carbs_g": round(carbs_100 * factor, 2),
        "fat_g": round(fat_100 * factor, 2),
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
    """
    Important behavior:
    - provider can come from UI or caller
    - model should usually be None
    - llm_provider.py should decide the correct model based on provider
    """
    selected_provider = provider or DEFAULT_PROVIDER
    selected_model = model  # DO NOT override with a global env fallback model

    user_prompt = f"""
Food: {food_text}
Estimated grams: {grams if grams is not None else "unknown"}
Category ID: {category_id if category_id is not None else "unknown"}
Category name: {category_name if category_name else "unknown"}
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

    raw_output = response.get("raw_output", "")
    parsed = _parse_nutrition_line(raw_output)

    used_heuristic_backup = False
    if parsed is None:
        parsed = _heuristic_backup(
            food_text=food_text,
            grams=grams,
            category_id=category_id,
            category_name=category_name,
        )
        used_heuristic_backup = True

    return {
        "nutrition": parsed,
        "raw_output": raw_output,
        "usage": response.get("usage"),
        "estimated_cost_usd": response.get("estimated_cost_usd"),
        "provider": response.get("provider"),
        "model": response.get("model"),
        "used_heuristic_backup": used_heuristic_backup,
    }