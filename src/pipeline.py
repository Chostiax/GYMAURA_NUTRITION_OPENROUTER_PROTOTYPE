from __future__ import annotations

from src.dataset_growth import append_proposed_row
from src.matcher import match_food_to_dataset
from src.nutrition import compute_item_nutrition, sum_nutrition
from src.nutrition_fallback import estimate_nutrition_with_llm
from src.openrouter_extractor import extract_foods_with_openrouter


LOW_CONFIDENCE_THRESHOLD = 75


def _resolve_known_food_grams(portions: float | None, dataset_row) -> tuple[float | None, str | None]:
    if portions is None or dataset_row is None:
        return None, None

    default_portion_grams = dataset_row.get("default_portion_grams")
    if default_portion_grams is None:
        return None, None

    try:
        grams = float(portions) * float(default_portion_grams)
        return grams, "portion_x_dataset_default"
    except Exception:
        return None, None


def run_pipeline(
    text: str,
    dataset,
    model: str | None = None,
    save_unmatched_candidates: bool = True,
) -> dict:
    """
    New product logic:
    - Never decompose dishes
    - Known foods -> value interpreted as portions
    - Unknown foods -> value interpreted as grams
    - If unmatched -> second LLM call for nutrition fallback
    - User should never see nutrition = None
    """
    extraction = extract_foods_with_openrouter(text, model=model)

    if extraction["no_food"]:
        return {
            "input": text,
            "items": [],
            "totals": {
                "calories": 0.0,
                "protein_g": 0.0,
                "carbs_g": 0.0,
                "fat_g": 0.0,
            },
            "ai_usage": extraction["usage"],
            "ai_raw_output": extraction["raw_output"],
            "parse_errors": extraction["parse_errors"],
            "model": extraction["model"],
        }

    final_items = []

    for item in extraction["items"]:
        match_result = match_food_to_dataset(
            food_text=item["food_text"],
            dataset=dataset,
            category_id=item.get("category_id"),
        )

        dataset_row = None
        nutrition = None
        nutrition_source = None
        needs_clarification = False
        fallback_raw_output = None

        raw_value = item.get("value")

        if match_result["matched"]:
            dataset_row = dataset.loc[match_result["row_index"]]

            # Known food => value = portions
            portions = raw_value
            grams, grams_source = _resolve_known_food_grams(
                portions=portions,
                dataset_row=dataset_row,
            )

            if grams is None:
                # fallback to dataset default portion as 1 portion if model gave nothing
                default_portion_grams = dataset_row.get("default_portion_grams")
                grams = float(default_portion_grams) if default_portion_grams is not None else None
                grams_source = "dataset_default_fallback" if grams is not None else None

            nutrition = compute_item_nutrition(dataset_row, grams)
            nutrition_source = "dataset"

            if nutrition is None:
                # emergency fallback, should be rare
                fallback = estimate_nutrition_with_llm(
                    food_text=item["food_text"],
                    grams=grams,
                    original_text=text,
                    model=model,
                )
                nutrition = fallback["nutrition"]
                nutrition_source = "llm_fallback_after_dataset_failure"
                fallback_raw_output = fallback["raw_output"]

            if grams is None:
                needs_clarification = True

            if save_unmatched_candidates and match_result["score"] < LOW_CONFIDENCE_THRESHOLD:
                append_proposed_row(
                    food_text=item["food_text"],
                    grams=grams,
                    category_id=item.get("category_id"),
                    category_name=item.get("category_name"),
                    source_input=text,
                    match_score=match_result.get("score", 0.0),
                )

        else:
            # Unknown food => value = grams
            portions = None
            grams = raw_value
            grams_source = "llm_unknown_food_grams" if grams is not None else None
            needs_clarification = True

            if save_unmatched_candidates:
                append_proposed_row(
                    food_text=item["food_text"],
                    grams=grams,
                    category_id=item.get("category_id"),
                    category_name=item.get("category_name"),
                    source_input=text,
                    match_score=match_result.get("score", 0.0),
                )

            fallback = estimate_nutrition_with_llm(
                food_text=item["food_text"],
                grams=grams,
                original_text=text,
                model=model,
            )
            nutrition = fallback["nutrition"]
            nutrition_source = "llm_fallback"
            fallback_raw_output = fallback["raw_output"]

        final_items.append(
            {
                "raw_segment": item["food_text"],
                "food_text": item["food_text"],
                "value": raw_value,
                "portions": portions,
                "grams": grams,
                "grams_source": grams_source,
                "llm_category_id": item.get("category_id"),
                "llm_category_name": item.get("category_name"),
                "matched": match_result["matched"],
                "match_type": match_result["match_type"],
                "match_score": match_result["score"],
                "search_scope": match_result.get("search_scope"),
                "normalized_query": match_result["normalized_query"],
                "matched_description": match_result["description"],
                "matched_category": match_result.get("matched_category"),
                "matched_category_id": match_result.get("matched_category_id"),
                "dataset_default_portion_grams": dataset_row.get("default_portion_grams") if dataset_row is not None else None,
                "dataset_default_portion_label": dataset_row.get("default_portion_label") if dataset_row is not None else None,
                "nutrition": nutrition,
                "nutrition_source": nutrition_source,
                "fallback_nutrition_raw_output": fallback_raw_output,
                "needs_clarification": needs_clarification,
            }
        )

    totals = sum_nutrition(final_items)

    return {
        "input": text,
        "items": final_items,
        "totals": totals,
        "ai_usage": extraction["usage"],
        "ai_raw_output": extraction["raw_output"],
        "parse_errors": extraction["parse_errors"],
        "model": extraction["model"],
    }