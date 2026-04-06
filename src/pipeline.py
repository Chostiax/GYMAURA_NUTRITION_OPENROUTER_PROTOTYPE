from __future__ import annotations

from src.dataset_growth import append_proposed_row
from src.matcher import match_food_to_dataset
from src.nutrition import resolve_item_grams, compute_item_nutrition, sum_nutrition
from src.openrouter_extractor import extract_foods_with_openrouter


LOW_CONFIDENCE_THRESHOLD = 75


def run_pipeline(
    text: str,
    dataset,
    model: str | None = None,
    save_unmatched_candidates: bool = True,
) -> dict:
    """
    Prototype pipeline:
    text -> OpenRouter -> food;grams;category_id -> parser
    -> category-aware matcher -> nutrition

    Behavior:
    - keeps LLM grams even when matching fails
    - computes nutrition only when a dataset row is matched
    - logs unmatched items for future dataset growth
    - also logs weak matches below LOW_CONFIDENCE_THRESHOLD
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
        needs_clarification = False

        # Keep the LLM grams even if matching fails
        llm_grams = item.get("grams")
        grams = llm_grams
        grams_source = "llm" if llm_grams is not None else None

        if match_result["matched"]:
            dataset_row = dataset.loc[match_result["row_index"]]

            # If the LLM did not provide grams, fall back to dataset default portion
            if grams is None:
                grams, grams_source = resolve_item_grams(
                    item_grams=None,
                    dataset_row=dataset_row,
                )

            nutrition = compute_item_nutrition(dataset_row, grams)

            if grams is None:
                needs_clarification = True

            # Log weak matches too, so they can be reviewed later
            if save_unmatched_candidates and match_result["score"] < LOW_CONFIDENCE_THRESHOLD:
                append_proposed_row(
                    food_text=item["food_text"],
                    grams=item.get("grams"),
                    category_id=item.get("category_id"),
                    category_name=item.get("category_name"),
                    source_input=text,
                    match_score=match_result.get("score", 0.0),
                )

        else:
            needs_clarification = True

            # Log unmatched items for dataset growth
            if save_unmatched_candidates:
                append_proposed_row(
                    food_text=item["food_text"],
                    grams=item.get("grams"),
                    category_id=item.get("category_id"),
                    category_name=item.get("category_name"),
                    source_input=text,
                    match_score=match_result.get("score", 0.0),
                )

        final_items.append(
            {
                "raw_segment": item["food_text"],
                "food_text": item["food_text"],
                "quantity": None,
                "unit": None,
                "grams": grams,
                "grams_source": grams_source,
                "llm_grams": item.get("grams"),
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