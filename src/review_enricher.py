from __future__ import annotations

from src.nutrition_fallback import estimate_nutrition_with_llm


def enrich_unmatched_food_with_llm(
    food_text: str,
    grams: float | None,
    original_text: str,
    provider: str | None = None,
    model: str | None = None,
    category_id: int | None = None,
    category_name: str | None = None,
) -> dict:
    """
    Unified fallback + growth path.

    We keep one LLM call for nutrition fallback and use the same result
    for dataset-growth review. Canonical text remains the extracted English
    food_text for now, which is consistent with the current architecture.
    """

    fallback = estimate_nutrition_with_llm(
        food_text=food_text,
        grams=grams,
        original_text=original_text,
        provider=provider,
        model=model,
        category_id=category_id,
        category_name=category_name,
    )

    parsed = {
        "canonical_food_text": food_text,
        "category_id": category_id,
        "grams": grams,
        "nutrition": fallback["nutrition"],
    }

    return {
        "parsed": parsed,
        "raw_output": fallback.get("raw_output"),
        "usage": fallback.get("usage"),
        "estimated_cost_usd": fallback.get("estimated_cost_usd"),
        "provider": fallback.get("provider"),
        "model": fallback.get("model"),
        "used_heuristic_backup": fallback.get("used_heuristic_backup", False),
    }