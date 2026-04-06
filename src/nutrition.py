from __future__ import annotations


def resolve_item_grams(item_grams: float | None, dataset_row) -> tuple[float | None, str | None]:
    if item_grams is not None:
        return float(item_grams), "llm"

    if dataset_row is not None:
        default_grams = dataset_row.get("default_portion_grams")
        try:
            if default_grams is not None:
                return float(default_grams), "dataset_default"
        except Exception:
            pass

    return None, None


def compute_item_nutrition(dataset_row, grams: float | None) -> dict | None:
    if dataset_row is None or grams is None:
        return None

    factor = grams / 100.0

    def scaled(value):
        try:
            if value is None:
                return 0.0
            return round(float(value) * factor, 2)
        except Exception:
            return 0.0

    return {
        "calories": scaled(dataset_row.get("calories_per_100g")),
        "protein_g": scaled(dataset_row.get("protein_per_100g")),
        "carbs_g": scaled(dataset_row.get("carbs_per_100g")),
        "fat_g": scaled(dataset_row.get("fat_per_100g")),
    }


def sum_nutrition(items: list[dict]) -> dict:
    totals = {"calories": 0.0, "protein_g": 0.0, "carbs_g": 0.0, "fat_g": 0.0}

    for item in items:
        nutrition = item.get("nutrition")
        if not nutrition:
            continue
        totals["calories"] += nutrition.get("calories", 0.0)
        totals["protein_g"] += nutrition.get("protein_g", 0.0)
        totals["carbs_g"] += nutrition.get("carbs_g", 0.0)
        totals["fat_g"] += nutrition.get("fat_g", 0.0)

    for key in totals:
        totals[key] = round(totals[key], 2)

    return totals