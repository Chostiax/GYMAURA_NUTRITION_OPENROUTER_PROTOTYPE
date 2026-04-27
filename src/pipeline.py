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
MAX_REASONABLE_GRAMS = 5000.0


def _build_zero_totals() -> dict:
    return {
        "calories": 0.0,
        "protein_g": 0.0,
        "carbs_g": 0.0,
        "fat_g": 0.0,
    }


def _safe_float(value) -> float | None:
    if value is None:
        return None

    try:
        parsed = float(value)
        if parsed <= 0:
            return None
        return parsed
    except Exception:
        return None


def _resolve_matched_food_grams(
    *,
    value: float | None,
    unit: str | None,
    dataset_row,
) -> tuple[float | None, float | None, str | None]:
    numeric_value = _safe_float(value)

    if numeric_value is None:
        return None, None, None

    if unit == "g":
        return None, numeric_value, "explicit_or_extracted_grams"

    default_portion_grams = dataset_row.get("default_portion_grams") if dataset_row is not None else None

    if unit == "portion" or unit is None:
        if default_portion_grams is None:
            return numeric_value, None, "portion_missing_dataset_default"

        try:
            grams = numeric_value * float(default_portion_grams)
            return numeric_value, grams, "portion_x_dataset_default"
        except Exception:
            return numeric_value, None, "portion_conversion_failed"

    return None, None, "unknown_unit"


def _resolve_unmatched_food_grams(
    *,
    value: float | None,
    unit: str | None,
    category_id: int | None,
) -> tuple[float | None, float | None, str | None]:
    numeric_value = _safe_float(value)

    if numeric_value is None:
        return None, DEFAULT_UNMATCHED_MIN_GRAMS, "minimum_unknown_default"

    if unit == "g":
        grams = numeric_value
        grams_source = "explicit_or_extracted_grams"

    elif unit == "portion":
        if category_id in SMALL_WEIGHT_ALLOWED_CATEGORIES:
            grams = numeric_value
            grams_source = "unmatched_portion_small_category_as_grams"
        else:
            grams = max(DEFAULT_UNMATCHED_MIN_GRAMS, numeric_value)
            grams_source = "unmatched_portion_context_guard"

    else:
        grams = numeric_value
        grams_source = "missing_unit_assumed_grams_with_guard"

    if category_id not in SMALL_WEIGHT_ALLOWED_CATEGORIES and grams < 20:
        grams = DEFAULT_UNMATCHED_MIN_GRAMS
        grams_source = "minimum_unknown_guard"

    return None, grams, grams_source


def _should_reject_match_due_to_category_mismatch(item: dict, match_result: dict) -> bool:
    llm_category_id = item.get("category_id")
    matched_category_id = match_result.get("matched_category_id")

    if llm_category_id is None or matched_category_id is None:
        return False

    return llm_category_id != matched_category_id


def _should_reject_match_due_to_absurd_quantity(
    *,
    item: dict,
    unit: str | None,
) -> bool:
    value = _safe_float(item.get("value"))
    if value is None:
        return False

    if unit == "g":
        return value > MAX_REASONABLE_GRAMS

    if unit == "portion":
        return value > MAX_REASONABLE_PORTIONS

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
        "reject_reason": reason,
    }


def _empty_fallback_meta() -> dict:
    return {
        "raw_output": None,
        "estimated_cost_usd": None,
        "provider": None,
        "model": None,
        "elapsed_ms": None,
        "used_structured_output": False,
        "used_second_attempt": False,
        "error": None,
    }


def _call_fallback(
    *,
    item: dict,
    grams: float | None,
    text: str,
    smart_provider: str | None,
    smart_model: str | None,
) -> tuple[dict | None, dict]:
    meta = _empty_fallback_meta()

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

        meta["raw_output"] = fallback.get("raw_output")
        meta["estimated_cost_usd"] = fallback.get("estimated_cost_usd") or 0.0
        meta["provider"] = fallback.get("provider")
        meta["model"] = fallback.get("model")
        meta["elapsed_ms"] = fallback.get("elapsed_ms")
        meta["used_structured_output"] = fallback.get("used_structured_output", False)
        meta["used_second_attempt"] = fallback.get("used_second_attempt", False)

        return fallback["nutrition"], meta

    except Exception as e:
        meta["error"] = str(e)
        return None, meta


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
            "fallback_elapsed_total_ms": 0.0,
            "fallback_count": 0,
            "dataset_count": 0,
        }

    final_items = []

    extraction_cost = None
    if isinstance(extraction.get("usage"), dict):
        extraction_cost = extraction["usage"].get("cost")

    fallback_cost_total = 0.0
    fallback_elapsed_total_ms = 0.0
    fallback_count = 0
    dataset_count = 0

    for item in extraction["items"]:
        raw_value = item.get("value")
        unit = item.get("unit")

        match_result = match_food_to_dataset(
            food_text=item["food_text"],
            dataset=dataset,
            category_id=item.get("category_id"),
        )

        if match_result["matched"]:
            if _should_reject_match_due_to_category_mismatch(item, match_result):
                match_result = _reject_match(match_result, "rejected_category_mismatch")
            elif _should_reject_match_due_to_absurd_quantity(item=item, unit=unit):
                match_result = _reject_match(match_result, "rejected_absurd_quantity")

        dataset_row = None
        nutrition = None
        nutrition_source = None
        needs_clarification = False
        fallback_meta = _empty_fallback_meta()

        portions = None
        grams = None
        grams_source = None

        if match_result["matched"]:
            dataset_row = dataset.loc[match_result["row_index"]]

            portions, grams, grams_source = _resolve_matched_food_grams(
                value=raw_value,
                unit=unit,
                dataset_row=dataset_row,
            )

            if grams is None:
                default_portion_grams = dataset_row.get("default_portion_grams")
                if default_portion_grams is not None:
                    grams = float(default_portion_grams)
                    grams_source = "dataset_default_fallback"
                    needs_clarification = True

            nutrition = compute_item_nutrition(dataset_row, grams)
            nutrition_source = "dataset"

            if nutrition is not None:
                dataset_count += 1

            if nutrition is None:
                nutrition, fallback_meta = _call_fallback(
                    item=item,
                    grams=grams,
                    text=text,
                    smart_provider=smart_provider,
                    smart_model=smart_model,
                )

                if nutrition is not None:
                    nutrition_source = (
                        "llm_fallback_after_dataset_failure_structured"
                        if fallback_meta["used_structured_output"]
                        else "llm_fallback_after_dataset_failure"
                    )
                    fallback_cost_total += fallback_meta["estimated_cost_usd"] or 0.0
                    fallback_elapsed_total_ms += fallback_meta["elapsed_ms"] or 0.0
                    fallback_count += 1
                else:
                    nutrition_source = "llm_failure_after_dataset_failure"

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
                    fallback_provider=fallback_meta["provider"],
                    fallback_model=fallback_meta["model"],
                    fallback_nutrition=nutrition if nutrition_source != "dataset" else None,
                    fallback_raw_output=fallback_meta["raw_output"],
                )

        else:
            portions, grams, grams_source = _resolve_unmatched_food_grams(
                value=raw_value,
                unit=unit,
                category_id=item.get("category_id"),
            )
            needs_clarification = True

            nutrition, fallback_meta = _call_fallback(
                item=item,
                grams=grams,
                text=text,
                smart_provider=smart_provider,
                smart_model=smart_model,
            )

            if nutrition is not None:
                nutrition_source = (
                    "llm_fallback_structured"
                    if fallback_meta["used_structured_output"]
                    else "llm_fallback"
                )
                fallback_cost_total += fallback_meta["estimated_cost_usd"] or 0.0
                fallback_elapsed_total_ms += fallback_meta["elapsed_ms"] or 0.0
                fallback_count += 1
            else:
                nutrition_source = "llm_failure"

            if save_unmatched_candidates:
                append_proposed_row(
                    food_text=item["food_text"],
                    grams=grams,
                    category_id=item.get("category_id"),
                    category_name=item.get("category_name"),
                    source_input=text,
                    match_score=match_result.get("score", 0.0),
                    status="pending_review" if nutrition is not None else "pending_review_llm_failed",
                    fallback_provider=fallback_meta["provider"],
                    fallback_model=fallback_meta["model"],
                    fallback_nutrition=nutrition,
                    fallback_raw_output=fallback_meta["raw_output"],
                )

        final_items.append(
            {
                "raw_segment": item["food_text"],
                "food_text": item["food_text"],
                "value": raw_value,
                "unit": unit,
                "portions": portions,
                "grams": grams,
                "grams_source": grams_source,
                "llm_category_id": item.get("category_id"),
                "llm_category_name": item.get("category_name"),
                "matched": match_result["matched"],
                "match_type": match_result.get("match_type"),
                "match_score": match_result.get("score"),
                "search_scope": match_result.get("search_scope"),
                "match_reject_reason": match_result.get("reject_reason"),
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
                "fallback_nutrition_raw_output": fallback_meta["raw_output"],
                "fallback_estimated_cost_usd": fallback_meta["estimated_cost_usd"],
                "fallback_provider": fallback_meta["provider"],
                "fallback_model": fallback_meta["model"],
                "fallback_elapsed_ms": fallback_meta["elapsed_ms"],
                "fallback_used_structured_output": fallback_meta["used_structured_output"],
                "fallback_used_second_attempt": fallback_meta["used_second_attempt"],
                "fallback_error": fallback_meta["error"],
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
        "fallback_elapsed_total_ms": round(fallback_elapsed_total_ms, 2),
        "fallback_count": fallback_count,
        "dataset_count": dataset_count,
    }