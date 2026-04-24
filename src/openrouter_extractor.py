from __future__ import annotations

import os
import re

import httpx
from dotenv import load_dotenv

from src.semicolon_parser import parse_semicolon_output
from src.data_prep import CATEGORY_ID_TO_NAME

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
DEFAULT_MODEL = os.getenv("OPENROUTER_MODEL", "google/gemma-3-4b-it")

CATEGORY_BLOCK = """
Valid category IDs:
1 = American Indian/Alaska Native Foods
2 = Baby Foods
3 = Baked Products
4 = Beef Products
5 = Beverages
6 = Breakfast Cereals
7 = Cereal Grains and Pasta
8 = Dairy and Egg Products
9 = Fast Foods
10 = Fats and Oils
11 = Finfish and Shellfish Products
12 = Fruits and Fruit Juices
13 = Lamb, Veal, and Game Products
14 = Legumes and Legume Products
15 = Meals, Entrees, and Side Dishes
16 = Nut and Seed Products
17 = Pork Products
18 = Poultry Products
19 = Restaurant Foods
20 = Sausages and Luncheon Meats
21 = Snacks
22 = Soups, Sauces, and Gravies
23 = Spices and Herbs
24 = Sweets
25 = Vegetables and Vegetable Products
""".strip()


SYSTEM_PROMPT = f"""
You extract foods from ONE user meal description.

The user's input may be in any language.
Your output must always be in ENGLISH.

Return ONLY foods explicitly consumed in the USER INPUT.
Do NOT infer extra ingredients.
Do NOT replace one food with another.
Do NOT split sauces, recipes, dishes, or prepared foods into components.
Do NOT output explanations.
Do NOT output JSON.

Output format:
food_text;value;unit;category_id

Allowed units:
- g
- portion

Rules:
1. If the user explicitly gives grams, preserve the grams exactly.
2. If the user gives kg, convert to grams.
3. For drinks, convert ml to grams approximately 1 ml = 1 g.
4. If the user gives a number before a countable food WITHOUT grams/ml/kg, that number means portions, NOT grams.
5. Do NOT estimate grams when the user gives a countable quantity.
6. For vague quantities like "some sauce", estimate a realistic gram amount, never 1g.
7. Never return tiny weights like 1g, 2g, or 5g unless the food is clearly a spice, herb, or tiny garnish.
8. NEVER decompose dishes, recipes, sauces, or prepared meals.
9. If the user writes a compound food as one word, normalize it into English words.
10. Preserve the user's food phrase. Do NOT replace it with a more common food.
11. "hamburger" is hamburger. Do NOT change it to chicken burger.
12. "calamari pizza" is calamari pizza. Do NOT change it to chicken pizza.
13. "chickenburger" or "chickenburgers" means "chicken burger".
14. "bolognese sauce" is one item. Do NOT output "bologna".
15. Always output a category_id from the list below.
16. If the user ate nothing, return exactly: NO_FOOD.
17. Output only valid semicolon-separated lines.

Examples:

User input:
I ate a hamburger and a vanilla milkshake

Output:
hamburger;1;portion;19
vanilla milkshake;1;portion;5

User input:
I ate 2 slices of calamari pizza

Output:
calamari pizza;2;portion;19

User input:
I had 300g of chicken breast with 200g of rice

Output:
chicken breast;300;g;18
white rice;200;g;7

User input:
I ate 10 apples

Output:
apple;10;portion;12

User input:
I ate 2 chickenburgers and a cup of pineapple juice

Output:
chicken burger;2;portion;19
pineapple juice;1;portion;5

User input:
I ate dragon fruit pizza

Output:
dragon fruit pizza;1;portion;19

User input:
I ate 150 grams of spaghetti with some sauce bolognaise and a glass of martini

Output:
spaghetti;150;g;7
bolognese sauce;80;g;22
martini;150;g;5

User input:
J'ai mangé un tagine et un verre de thé marocain

Output:
tagine;350;g;15
moroccan tea;250;g;5

User input:
I didn't eat anything

Output:
NO_FOOD

{CATEGORY_BLOCK}
""".strip()


MASS_OR_VOLUME_UNITS = {
    "g",
    "gram",
    "grams",
    "kg",
    "kilogram",
    "kilograms",
    "ml",
    "milliliter",
    "milliliters",
    "l",
    "liter",
    "liters",
}

SMALL_GRAM_ALLOWED_CATEGORY_IDS = {
    10,  # fats and oils
    23,  # spices and herbs
}

MIN_REASONABLE_GRAMS_FOR_EXTRACTED_FOOD = 20.0


HEAD_WORDS = {
    "pizza": "pizza",
    "pizzas": "pizza",
    "burger": "burger",
    "burgers": "burger",
    "hamburger": "burger",
    "hamburgers": "burger",
    "milkshake": "milkshake",
    "milkshakes": "milkshake",
    "juice": "juice",
    "juices": "juice",
    "sandwich": "sandwich",
    "sandwiches": "sandwich",
    "wrap": "wrap",
    "wraps": "wrap",
    "salad": "salad",
    "salads": "salad",
    "taco": "taco",
    "tacos": "taco",
    "burrito": "burrito",
    "burritos": "burrito",
    "risotto": "risotto",
    "ramen": "ramen",
    "sauce": "sauce",
    "soup": "soup",
    "soups": "soup",
}


LEADING_NOISE_WORDS = {
    "i",
    "ate",
    "eat",
    "eaten",
    "had",
    "have",
    "a",
    "an",
    "the",
    "some",
    "of",
    "with",
    "and",
    "or",
    "cup",
    "glass",
    "bowl",
    "plate",
    "slice",
    "slices",
    "serving",
    "servings",
    "piece",
    "pieces",
    "grams",
    "gram",
    "g",
    "ml",
    "kg",
}


def _normalize_compound_food_words(text: str) -> str:
    text = (text or "").lower()

    replacements = {
        "chickenburgers": "chicken burgers",
        "chickenburger": "chicken burger",
        "beefburgers": "beef burgers",
        "beefburger": "beef burger",
        "cheeseburgers": "cheese burgers",
        "cheeseburger": "cheese burger",
        "pineapplejuice": "pineapple juice",
        "orangejuice": "orange juice",
        "applejuice": "apple juice",
        "chickenpizza": "chicken pizza",
        "seafoodpizza": "seafood pizza",
    }

    for old, new in replacements.items():
        text = re.sub(rf"\b{re.escape(old)}\b", new, text)

    return text


def _simple_singular(text: str) -> str:
    text = (text or "").lower().strip()

    if text.endswith("ies") and len(text) > 3:
        return text[:-3] + "y"

    if text.endswith("es") and len(text) > 2:
        return text[:-2]

    if text.endswith("s") and not text.endswith("ss") and len(text) > 1:
        return text[:-1]

    return text


def _simple_plural(text: str) -> str:
    text = (text or "").lower().strip()

    if not text:
        return text

    if text.endswith("y") and len(text) > 1:
        return text[:-1] + "ies"

    if text.endswith(("s", "x", "ch", "sh")):
        return text + "es"

    return text + "s"


def _singularize_phrase(text: str) -> str:
    tokens = text.split()
    if not tokens:
        return text

    tokens[-1] = _simple_singular(tokens[-1])
    return " ".join(tokens)


def _number_word_to_float(text: str) -> float | None:
    cleaned = (text or "").lower().strip()

    words = {
        "a": 1.0,
        "an": 1.0,
        "one": 1.0,
        "two": 2.0,
        "three": 3.0,
        "four": 4.0,
        "five": 5.0,
        "six": 6.0,
        "seven": 7.0,
        "eight": 8.0,
        "nine": 9.0,
        "ten": 10.0,
        "eleven": 11.0,
        "twelve": 12.0,
    }

    if cleaned in words:
        return words[cleaned]

    try:
        return float(cleaned)
    except Exception:
        return None


def _looks_english(text: str) -> bool:
    text = text or ""
    if not text.strip():
        return False

    ascii_letters = sum(1 for ch in text if ch.isascii() and ch.isalpha())
    total_letters = sum(1 for ch in text if ch.isalpha())

    return total_letters > 0 and ascii_letters / total_letters > 0.9


def _has_whole_word(text: str, word: str) -> bool:
    return re.search(rf"\b{re.escape(word)}\b", text, flags=re.IGNORECASE) is not None


def _category_name(category_id: int | None) -> str | None:
    if category_id is None:
        return None
    return CATEGORY_ID_TO_NAME.get(category_id)


def _clean_original_phrase(phrase: str) -> str:
    tokens = re.findall(r"[a-z0-9]+", phrase.lower())

    while tokens and (tokens[0] in LEADING_NOISE_WORDS or tokens[0].isdigit()):
        tokens.pop(0)

    return _singularize_phrase(" ".join(tokens).strip())


def _head_group(food_text: str) -> str | None:
    tokens = re.findall(r"[a-z0-9]+", (food_text or "").lower())

    for token in reversed(tokens):
        if token in HEAD_WORDS:
            return HEAD_WORDS[token]

    return None


def _extract_original_head_phrases(original_text: str) -> list[tuple[str, str]]:
    """
    Extract likely dish phrases from English input.

    Example:
    'I ate 2 slices of calamari pizza'
    -> [('calamari pizza', 'pizza')]

    'I ate a hamburger and a vanilla milkshake'
    -> [('hamburger', 'burger'), ('vanilla milkshake', 'milkshake')]
    """
    normalized = _normalize_compound_food_words(original_text)
    normalized = re.sub(r"[^a-z0-9\s]", " ", normalized)
    normalized = " ".join(normalized.split())

    phrases: list[tuple[str, str]] = []

    for head, group in HEAD_WORDS.items():
        pattern = rf"\b((?:[a-z0-9]+\s+){{0,4}}{re.escape(head)})\b"

        for match in re.finditer(pattern, normalized):
            phrase = _clean_original_phrase(match.group(1))
            if phrase:
                phrases.append((phrase, group))

    seen = set()
    deduped = []

    for phrase, group in phrases:
        key = (phrase, group)
        if key not in seen:
            deduped.append((phrase, group))
            seen.add(key)

    return deduped


def _category_for_head_group(group: str) -> int:
    if group in {"juice", "milkshake"}:
        return 5

    if group in {"burger", "pizza", "sandwich", "wrap", "taco", "burrito"}:
        return 19

    if group in {"salad", "risotto", "ramen", "sauce", "soup"}:
        return 15

    return 15


def _repair_wrong_head_modifier_hallucinations(original_text: str, items: list[dict]) -> list[dict]:
    """
    Repairs wrong same-head substitutions.

    Examples:
    original: hamburger
    model: chicken burger
    -> hamburger

    original: calamari pizza
    model: chicken pizza
    -> calamari pizza
    """
    if not _looks_english(original_text):
        return items

    original_phrases = _extract_original_head_phrases(original_text)
    if not original_phrases:
        return items

    normalized_original = _normalize_compound_food_words(original_text)

    for item in items:
        food_text = (item.get("food_text") or "").lower().strip()
        group = _head_group(food_text)

        if group is None:
            continue

        candidates = [phrase for phrase, phrase_group in original_phrases if phrase_group == group]
        if not candidates:
            continue

        if food_text in normalized_original:
            continue

        food_tokens = set(re.findall(r"[a-z0-9]+", food_text))
        original_tokens = set(re.findall(r"[a-z0-9]+", normalized_original))

        introduced_tokens = food_tokens - original_tokens

        if introduced_tokens:
            repaired_phrase = candidates[0]
            item["food_text"] = repaired_phrase
            item["category_id"] = _category_for_head_group(group)
            item["category_name"] = _category_name(item["category_id"])

    return items


def _remove_substring_hallucinations(original_text: str, items: list[dict]) -> list[dict]:
    """
    Removes obvious English substring hallucinations.

    Example:
    original: "sauce bolognaise"
    bad item: "bologna"
    """
    if not _looks_english(original_text):
        return items

    lowered = _normalize_compound_food_words(original_text)
    cleaned_items = []

    for item in items:
        food_text = (item.get("food_text") or "").lower().strip()
        tokens = [tok for tok in re.findall(r"[a-z]+", food_text) if len(tok) >= 3]

        if not tokens:
            cleaned_items.append(item)
            continue

        if any(_has_whole_word(lowered, token) for token in tokens):
            cleaned_items.append(item)
            continue

        if "bolognese" in food_text and ("bolognaise" in lowered or "bolognese" in lowered):
            cleaned_items.append(item)
            continue

    return cleaned_items


def _remove_items_contained_in_larger_dishes(items: list[dict]) -> list[dict]:
    """
    Removes decomposed ingredients when the model also extracted the full dish.

    Examples:
    - chicken pizza + chicken -> keep chicken pizza only
    - dragon fruit pizza + dragon fruit -> keep dragon fruit pizza only
    - beef tacos + beef -> keep beef tacos only
    """
    cleaned = []

    normalized_items = []
    for item in items:
        food_text = (item.get("food_text") or "").lower().strip()
        tokens = set(re.findall(r"[a-z0-9]+", food_text))
        normalized_items.append((item, food_text, tokens))

    for item, food_text, tokens in normalized_items:
        if not food_text or not tokens:
            cleaned.append(item)
            continue

        should_drop = False

        for other_item, other_food_text, other_tokens in normalized_items:
            if other_item is item:
                continue

            if not other_food_text or other_food_text == food_text:
                continue

            if tokens.issubset(other_tokens) and len(other_tokens) > len(tokens):
                should_drop = True
                break

            if f" {food_text} " in f" {other_food_text} " and len(other_food_text) > len(food_text):
                should_drop = True
                break

        if not should_drop:
            cleaned.append(item)

    return cleaned


def _add_missing_original_head_phrases(original_text: str, items: list[dict]) -> list[dict]:
    """
    If Gemma completely drops an obvious food phrase from English input, add it back.

    Example:
    original: "i ate a hamburger and a vanilla milkshake"
    model output: "vanilla milkshake"
    fixed output: "hamburger", "vanilla milkshake"
    """
    if not _looks_english(original_text):
        return items

    original_phrases = _extract_original_head_phrases(original_text)
    if not original_phrases:
        return items

    existing = {(item.get("food_text") or "").lower().strip() for item in items}

    for phrase, group in original_phrases:
        if not phrase:
            continue

        already_present = any(
            phrase == existing_food
            or phrase in existing_food
            or existing_food in phrase
            for existing_food in existing
        )

        if already_present:
            continue

        category_id = _category_for_head_group(group)

        items.append(
            {
                "food_text": phrase,
                "value": 1.0,
                "unit": "portion",
                "category_id": category_id,
                "category_name": _category_name(category_id),
            }
        )

        existing.add(phrase)

    return items


def _has_mass_or_volume_unit_between(number_text: str, food_text: str, original_text: str) -> bool:
    escaped_number = re.escape(number_text)
    escaped_food = re.escape(food_text)
    units = "|".join(re.escape(unit) for unit in MASS_OR_VOLUME_UNITS)

    pattern = rf"\b{escaped_number}\s*({units})\s+(?:of\s+)?{escaped_food}\b"
    return re.search(pattern, original_text, flags=re.IGNORECASE) is not None


def _force_countable_quantities_from_input(original_text: str, items: list[dict]) -> list[dict]:
    lowered = _normalize_compound_food_words(original_text)

    for item in items:
        food_text = (item.get("food_text") or "").lower().strip()
        if not food_text:
            continue

        food_tokens = food_text.split()
        main_food = food_tokens[-1] if food_tokens else food_text

        singular = _simple_singular(main_food)
        plural = _simple_plural(singular)

        possible_food_forms = {
            main_food,
            singular,
            plural,
            food_text,
            _simple_plural(food_text),
        }

        possible_food_forms = {form for form in possible_food_forms if form and len(form) >= 2}

        for form in sorted(possible_food_forms, key=len, reverse=True):
            quantity_pattern = (
                rf"\b(\d+(?:\.\d+)?|a|an|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)"
                rf"\s+{re.escape(form)}\b"
            )

            match = re.search(quantity_pattern, lowered)
            if not match:
                continue

            number_text = match.group(1)
            number = _number_word_to_float(number_text)
            if number is None:
                continue

            if _has_mass_or_volume_unit_between(number_text, form, lowered):
                continue

            item["value"] = number
            item["unit"] = "portion"
            break

    return items


def _guard_tiny_gram_estimates(items: list[dict]) -> list[dict]:
    """
    Quantity sanity guard only. This is not a nutrition heuristic.
    """
    for item in items:
        unit = item.get("unit")
        value = item.get("value")
        category_id = item.get("category_id")

        if unit != "g":
            continue

        try:
            numeric_value = float(value)
        except Exception:
            continue

        if numeric_value >= MIN_REASONABLE_GRAMS_FOR_EXTRACTED_FOOD:
            continue

        if category_id in SMALL_GRAM_ALLOWED_CATEGORY_IDS:
            continue

        item["value"] = MIN_REASONABLE_GRAMS_FOR_EXTRACTED_FOOD

    return items


def _postprocess_items(original_text: str, items: list[dict]) -> list[dict]:
    items = _repair_wrong_head_modifier_hallucinations(original_text, items)
    items = _remove_substring_hallucinations(original_text, items)
    items = _remove_items_contained_in_larger_dishes(items)
    items = _add_missing_original_head_phrases(original_text, items)
    items = _force_countable_quantities_from_input(original_text, items)
    items = _guard_tiny_gram_estimates(items)
    return items


def extract_foods_with_openrouter(text: str, model: str | None = None) -> dict:
    if not OPENROUTER_API_KEY:
        raise ValueError("Missing OPENROUTER_API_KEY in environment.")

    selected_model = model or DEFAULT_MODEL

    response = httpx.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": selected_model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": text},
            ],
            "temperature": 0,
            "max_tokens": 250,
        },
        timeout=60.0,
    )

    response.raise_for_status()
    payload = response.json()

    raw_output = (
        payload.get("choices", [{}])[0]
        .get("message", {})
        .get("content", "")
        .strip()
    )

    parsed = parse_semicolon_output(raw_output)

    items = parsed["items"]
    if not parsed["no_food"]:
        items = _postprocess_items(text, items)

    return {
        "no_food": parsed["no_food"],
        "items": items,
        "usage": payload.get("usage"),
        "raw_output": raw_output,
        "parse_errors": parsed["parse_errors"],
        "model": selected_model,
    }