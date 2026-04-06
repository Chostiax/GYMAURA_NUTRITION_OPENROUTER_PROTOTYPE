import json
import re
from pathlib import Path
from typing import Any

import pandas as pd


CATEGORY_ID_TO_NAME = {
    1: "American Indian/Alaska Native Foods",
    2: "Baby Foods",
    3: "Baked Products",
    4: "Beef Products",
    5: "Beverages",
    6: "Breakfast Cereals",
    7: "Cereal Grains and Pasta",
    8: "Dairy and Egg Products",
    9: "Fast Foods",
    10: "Fats and Oils",
    11: "Finfish and Shellfish Products",
    12: "Fruits and Fruit Juices",
    13: "Lamb, Veal, and Game Products",
    14: "Legumes and Legume Products",
    15: "Meals, Entrees, and Side Dishes",
    16: "Nut and Seed Products",
    17: "Pork Products",
    18: "Poultry Products",
    19: "Restaurant Foods",
    20: "Sausages and Luncheon Meats",
    21: "Snacks",
    22: "Soups, Sauces, and Gravies",
    23: "Spices and Herbs",
    24: "Sweets",
    25: "Vegetables and Vegetable Products",
}

CATEGORY_NAME_TO_ID = {v: k for k, v in CATEGORY_ID_TO_NAME.items()}

TARGET_NUTRIENTS = {
    "calories": ["energy", "kcal", "calorie", "calories"],
    "protein": ["protein"],
    "carbs": ["carbohydrate", "carbohydrates", "carbs"],
    "fat": ["fat", "total lipid", "lipid", "total fat"],
}

GENERIC_DESCRIPTION_WORDS = {
    "raw", "cooked", "fresh", "regular", "unenriched", "without", "with",
    "salt", "boiled", "steamed", "grilled", "roasted", "fried",
    "broiled", "baked", "dry", "form", "frozen", "prepared", "plain",
    "drained", "solids", "pack", "commercial", "unheated", "canned",
}

PREFERRED_LANG_ORDER = ["en", "fr", "de", "it", "es"]


def normalize_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return " ".join(text.split()).strip()


def singularize_simple(word: str) -> str:
    irregular = {
        "eggs": "egg",
        "bananas": "banana",
        "apples": "apple",
        "dates": "date",
        "almonds": "almond",
        "cereals": "cereal",
        "berries": "berry",
        "vegetables": "vegetable",
        "fries": "fry",
        "cookies": "cookie",
        "strawberries": "strawberry",
        "blueberries": "blueberry",
        "cappuccinos": "cappuccino",
        "asparagus": "asparagus",
        "chips": "chips",
        "tomatoes": "tomato",
    }
    if word in irregular:
        return irregular[word]
    if word.endswith("ies") and len(word) > 3:
        return word[:-3] + "y"
    if word.endswith("s") and len(word) > 3 and not word.endswith(("ss", "us")):
        return word[:-1]
    return word


def normalize_food_text(text: str, drop_generic_words: bool = False) -> str:
    text = normalize_text(text)
    tokens = [singularize_simple(tok) for tok in text.split()]
    if drop_generic_words:
        tokens = [tok for tok in tokens if tok not in GENERIC_DESCRIPTION_WORDS]
    return " ".join(tokens).strip()


def extract_english_description(description_value: Any) -> str:
    if isinstance(description_value, dict):
        if description_value.get("en"):
            return str(description_value["en"]).strip().lower()
        for lang in PREFERRED_LANG_ORDER:
            if description_value.get(lang):
                return str(description_value[lang]).strip().lower()
        if description_value:
            first_value = next(iter(description_value.values()))
            return str(first_value).strip().lower()

    if isinstance(description_value, str):
        return description_value.strip().lower()

    return ""


def iter_nutrient_items(nutrients_value: Any):
    if isinstance(nutrients_value, dict):
        for tier_items in nutrients_value.values():
            if isinstance(tier_items, list):
                for item in tier_items:
                    if isinstance(item, dict):
                        yield item
    elif isinstance(nutrients_value, list):
        for item in nutrients_value:
            if isinstance(item, dict):
                yield item


def extract_macro_value(nutrients_value: Any, target_macro: str) -> float | None:
    aliases = TARGET_NUTRIENTS[target_macro]
    for item in iter_nutrient_items(nutrients_value):
        nutrient_name = str(item.get("name", "")).lower()
        amount = item.get("amount")
        if amount is None:
            continue
        if any(alias in nutrient_name for alias in aliases):
            try:
                return float(amount)
            except Exception:
                return None
    return None


def extract_default_portion_grams(portions_value: Any) -> float | None:
    if not isinstance(portions_value, list):
        return None

    default_item = None
    first_item = None

    for item in portions_value:
        if not isinstance(item, dict):
            continue
        if first_item is None:
            first_item = item
        if item.get("portionType") == "default":
            default_item = item
            break

    chosen = default_item or first_item
    if not chosen:
        return None

    grams = chosen.get("gramWeight")
    try:
        return float(grams) if grams is not None else None
    except Exception:
        return None


def extract_default_portion_label(portions_value: Any) -> str | None:
    if not isinstance(portions_value, list):
        return None

    default_item = None
    first_item = None

    for item in portions_value:
        if not isinstance(item, dict):
            continue
        if first_item is None:
            first_item = item
        if item.get("portionType") == "default":
            default_item = item
            break

    chosen = default_item or first_item
    if not chosen:
        return None

    label = chosen.get("label")
    if isinstance(label, dict):
        return label.get("en") or next(iter(label.values()), None)
    if isinstance(label, str):
        return label
    return None


def load_and_prepare_dataset(json_path: str) -> pd.DataFrame:
    path = Path(json_path)
    with path.open("r", encoding="utf-8") as f:
        raw_items = json.load(f)

    rows = []

    for item in raw_items:
        english_description = extract_english_description(item.get("description"))
        clean_description = normalize_food_text(english_description, drop_generic_words=True)

        food_category = item.get("food_category")
        category_id = CATEGORY_NAME_TO_ID.get(food_category)

        row = {
            "fdc_id": str(item.get("fdcId", "")).strip(),
            "english_description": english_description,
            "clean_description": clean_description,
            "food_category": food_category,
            "category_id": category_id,
            "default_portion_grams": extract_default_portion_grams(item.get("portions")),
            "default_portion_label": extract_default_portion_label(item.get("portions")),
            "calories_per_100g": extract_macro_value(item.get("nutrients"), "calories"),
            "protein_per_100g": extract_macro_value(item.get("nutrients"), "protein"),
            "carbs_per_100g": extract_macro_value(item.get("nutrients"), "carbs"),
            "fat_per_100g": extract_macro_value(item.get("nutrients"), "fat"),
        }

        if row["clean_description"]:
            rows.append(row)

    df = pd.DataFrame(rows).drop_duplicates(subset=["clean_description", "food_category"]).reset_index(drop=True)
    return df