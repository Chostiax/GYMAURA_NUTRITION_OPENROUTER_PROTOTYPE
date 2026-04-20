from __future__ import annotations

from src.dataset_growth import append_proposed_row
from src.matcher import match_food_to_dataset
from src.nutrition import compute_item_nutrition, sum_nutrition
from src.nutrition_fallback import estimate_nutrition_with_llm
from src.openrouter_extractor import extract_foods_with_openrouter

LOW_CONFIDENCE_THRESHOLD = 75
SMALL_WEIGHT_ALLOWED_CATEGORIES = {
    10,  # fats and oils
    22,  # soups, sauces, gravies
    23,  # spices and herbs
}
DEFAULT_UNMATCHED_MIN_GRAMS = 200.0
MAX_REASONABLE_PORTIONS = 8.0


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


def _apply_unmatched_grams_guard(grams: float | None, category_id: int | None) -> tuple[float | None, str | None]:
    """
    For unmatched foods:
    - value should already be grams according to your new rule
    - we still guard against absurd tiny values like 1g for a full dish
    """
    if grams is None:
        return DEFAULT_UNMATCHED_MIN_GRAMS, "minimum_unknown_default"

    if category_id in SMALL_WEIGHT_ALLOWED_CATEGORIES:
        return grams, "llm_unknown_food_grams"

    if grams < 20:
        return DEFAULT_UNMATCHED_MIN_GRAMS, "minimum_unknown_guard"

    return grams, "llm_unknown_food_grams"


def _should_reject_match_due_to_category_mismatch(item: dict, match_result: dict) -> bool:
    llm_category_id = item.get("category_id")
    matched_category_id = match_result.get("matched_category_id")

    if llm_category_id is None or matched_category_id is None:
        return False

    return llm_category_id != matched_category_id


def _should_reject_match_due_to_absurd_portions(item: dict) -> bool:
    value = item.get("value")
    if value is None:
        return False

    try:
        return float(value) > MAX_REASONABLE_PORTIONS
    except Exception:
        return False


def _reject_match(match_result: dict, reason: str) -> dict:
    return {
        "matched": False,
        "match_type": None,
        "score": 0.0,
        "row_index": None,
        "description": None,
        "normalized_query": match_result.get("normalized_query"),
        "matched_category": None,
        "matched_category_id": None,
        "search_scope": reason,
    }


def _build_zero_totals() -> dict:
    return {
        "calories": 0.0,
        "protein_g": 0.0,
        "carbs_g": 0.0,
        "fat_g": 0.0,
    }


def run_pipeline(
    text: str,
    dataset,
    model: str | None = None,
    save_unmatched_candidates: bool = True,
    smart_provider: str | None = None,
    smart_model: str | None = None,
) -> dict:
    extraction = extract_foods_with_openrouter(text, model=model)

    if extraction["no_food"]:
        return {
            "input": text,
            "items": [],
            "totals": _build_zero_totals(),
            "ai_usage": extraction.get("usage"),
            "ai_raw_output": extraction.get("raw_output"),
            "parse_errors": extraction.get("parse_errors"),
            "model": extraction.get("model"),
            "estimated_cost_usd": None,
        }

    final_items = []

    extraction_cost = None
    if isinstance(extraction.get("usage"), dict):
        extraction_cost = extraction["usage"].get("cost")

    fallback_cost_total = 0.0

    for item in extraction["items"]:
        match_result = match_food_to_dataset(
            food_text=item["food_text"],
            dataset=dataset,
            category_id=item.get("category_id"),
        )

        if match_result["matched"]:
            if _should_reject_match_due_to_category_mismatch(item, match_result):
                match_result = _reject_match(match_result, "rejected_category_mismatch")
            elif _should_reject_match_due_to_absurd_portions(item):
                match_result = _reject_match(match_result, "rejected_absurd_portions")

        dataset_row = None
        nutrition = None
        nutrition_source = None
        needs_clarification = False

        fallback_raw_output = None
        fallback_estimated_cost_usd = None
        fallback_provider_used = None
        fallback_model_used = None
        fallback_used_repair_pass = False
        fallback_used_second_attempt = False
        fallback_error = None

        raw_value = item.get("value")
        portions = None
        grams = None
        grams_source = None

        if match_result["matched"]:
            dataset_row = dataset.loc[match_result["row_index"]]

            portions = raw_value
            grams, grams_source = _resolve_known_food_grams(
                portions=portions,
                dataset_row=dataset_row,
            )

            if grams is None:
                default_portion_grams = dataset_row.get("default_portion_grams")
                if default_portion_grams is not None:
                    grams = float(default_portion_grams)
                    grams_source = "dataset_default_fallback"

            nutrition = compute_item_nutrition(dataset_row, grams)
            nutrition_source = "dataset"

            if nutrition is None:
                try:
                    fallback = estimate_nutrition_with_llm(
                        food_text=item["food_text"],
                        grams=grams,
                        original_text=text,
                        provider=smart_provider,
                        model=smart_model,
                        category_id=item.get("category_id"),
                        category_name=item.get("category_name"),
                    )

                    nutrition = fallback["nutrition"]
                    fallback_raw_output = fallback.get("raw_output")
                    fallback_estimated_cost_usd = fallback.get("estimated_cost_usd") or 0.0
                    fallback_provider_used = fallback.get("provider")
                    fallback_model_used = fallback.get("model")
                    fallback_used_repair_pass = fallback.get("used_repair_pass", False)
                    fallback_used_second_attempt = fallback.get("used_second_attempt", False)
                    fallback_cost_total += fallback_estimated_cost_usd

                    nutrition_source = (
                        "llm_fallback_after_dataset_failure_repair"
                        if fallback_used_repair_pass
                        else "llm_fallback_after_dataset_failure"
                    )
                except Exception as e:
                    nutrition = None
                    nutrition_source = "llm_failure_after_dataset_failure"
                    fallback_error = str(e)

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
                    status="low_confidence_match",
                    fallback_provider=fallback_provider_used,
                    fallback_model=fallback_model_used,
                    fallback_nutrition=nutrition if nutrition_source != "dataset" else None,
                    fallback_raw_output=fallback_raw_output,
                )

        else:
            grams, grams_source = _apply_unmatched_grams_guard(
                grams=raw_value,
                category_id=item.get("category_id"),
            )
            needs_clarification = True

            try:
                fallback = estimate_nutrition_with_llm(
                    food_text=item["food_text"],
                    grams=grams,
                    original_text=text,
                    provider=smart_provider,
                    model=smart_model,
                    category_id=item.get("category_id"),
                    category_name=item.get("category_name"),
                )

                nutrition = fallback["nutrition"]
                fallback_raw_output = fallback.get("raw_output")
                fallback_estimated_cost_usd = fallback.get("estimated_cost_usd") or 0.0
                fallback_provider_used = fallback.get("provider")
                fallback_model_used = fallback.get("model")
                fallback_used_repair_pass = fallback.get("used_repair_pass", False)
                fallback_used_second_attempt = fallback.get("used_second_attempt", False)
                fallback_cost_total += fallback_estimated_cost_usd

                nutrition_source = "llm_fallback_repair" if fallback_used_repair_pass else "llm_fallback"

            except Exception as e:
                nutrition = None
                nutrition_source = "llm_failure"
                fallback_error = str(e)

            if save_unmatched_candidates:
                append_proposed_row(
                    food_text=item["food_text"],
                    grams=grams,
                    category_id=item.get("category_id"),
                    category_name=item.get("category_name"),
                    source_input=text,
                    match_score=match_result.get("score", 0.0),
                    status="pending_review" if nutrition is not None else "pending_review_llm_failed",
                    fallback_provider=fallback_provider_used,
                    fallback_model=fallback_model_used,
                    fallback_nutrition=nutrition,
                    fallback_raw_output=fallback_raw_output,
                )

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
                "match_type": match_result.get("match_type"),
                "match_score": match_result.get("score"),
                "search_scope": match_result.get("search_scope"),
                "normalized_query": match_result.get("normalized_query"),
                "matched_description": match_result.get("description"),
                "matched_category": match_result.get("matched_category"),
                "matched_category_id": match_result.get("matched_category_id"),
                "dataset_default_portion_grams": (
                    dataset_row.get("default_portion_grams") if dataset_row is not None else None
                ),
                "dataset_default_portion_label": (
                    dataset_row.get("default_portion_label") if dataset_row is not None else None
                ),
                "nutrition": nutrition,
                "nutrition_source": nutrition_source,
                "fallback_nutrition_raw_output": fallback_raw_output,
                "fallback_estimated_cost_usd": fallback_estimated_cost_usd,
                "fallback_provider": fallback_provider_used,
                "fallback_model": fallback_model_used,
                "fallback_used_repair_pass": fallback_used_repair_pass,
                "fallback_used_second_attempt": fallback_used_second_attempt,
                "fallback_error": fallback_error,
                "needs_clarification": needs_clarification,
            }
        )

    items_with_nutrition = [item for item in final_items if item.get("nutrition") is not None]

    totals = sum_nutrition(items_with_nutrition) if items_with_nutrition else _build_zero_totals()

    total_estimated_cost_usd = None
    if extraction_cost is not None or fallback_cost_total > 0:
        total_estimated_cost_usd = round((extraction_cost or 0.0) + fallback_cost_total, 8)

    return {
        "input": text,
        "items": final_items,
        "totals": totals,
        "ai_usage": extraction.get("usage"),
        "ai_raw_output": extraction.get("raw_output"),
        "parse_errors": extraction.get("parse_errors"),
        "model": extraction.get("model"),
        "estimated_cost_usd": total_estimated_cost_usd,
    }