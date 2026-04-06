from __future__ import annotations

from src.matcher import apply_alias
from src.pipeline import run_pipeline


def normalize_expected_food(food_name: str) -> str:
    return apply_alias(food_name)


def foods_match(expected_food: str, predicted_food: str) -> bool:
    return expected_food == predicted_food


def evaluate_pipeline(test_cases: list[dict], dataset, model: str | None = None) -> dict:
    """
    Updated evaluation for the new prototype.
    We no longer evaluate explicit quantity extraction because the new prototype
    works primarily in grams and uses LLM estimation + dataset default portions.
    """
    expected_food_total = 0
    correct_food_total = 0

    extracted_item_total = 0
    matched_item_total = 0
    grams_available_total = 0

    successful_case_total = 0
    total_cases = len(test_cases)

    details = []

    for case in test_cases:
        text = case["input"]
        expected = case["expected"]

        result = run_pipeline(text, dataset, model=model)

        predicted_items = result["items"]
        predicted_foods = [apply_alias(item["food_text"]) for item in predicted_items]
        expected_foods = [normalize_expected_food(item["food"]) for item in expected]

        expected_food_total += len(expected_foods)
        case_food_hits = 0
        used_predicted_indexes = set()

        for expected_food in expected_foods:
            for idx, predicted_food in enumerate(predicted_foods):
                if idx in used_predicted_indexes:
                    continue
                if foods_match(expected_food, predicted_food):
                    correct_food_total += 1
                    case_food_hits += 1
                    used_predicted_indexes.add(idx)
                    break

        extracted_item_total += len(predicted_items)
        matched_item_total += sum(1 for item in predicted_items if item["matched"])
        grams_available_total += sum(1 for item in predicted_items if item.get("grams") is not None)

        all_detected = case_food_hits == len(expected_foods)
        all_matched = all(item["matched"] for item in predicted_items) if predicted_items else len(expected_foods) == 0

        if all_detected and all_matched:
            successful_case_total += 1

        details.append({
            "input": text,
            "expected": expected,
            "result": result,
            "all_detected": all_detected,
            "all_matched": all_matched,
        })

    metrics = {
        "food_detection_accuracy": round(correct_food_total / expected_food_total, 3) if expected_food_total else 0.0,
        "dataset_matching_success": round(matched_item_total / extracted_item_total, 3) if extracted_item_total else 0.0,
        "grams_coverage": round(grams_available_total / extracted_item_total, 3) if extracted_item_total else 0.0,
        "end_to_end_success": round(successful_case_total / total_cases, 3) if total_cases else 0.0,
        "total_cases": total_cases,
    }

    return {
        "metrics": metrics,
        "details": details,
    }