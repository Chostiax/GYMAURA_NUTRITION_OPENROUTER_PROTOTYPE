from __future__ import annotations

import os

import httpx
from dotenv import load_dotenv

from src.semicolon_parser import parse_semicolon_output

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

Return ONLY the foods from the USER INPUT.
Do NOT repeat examples.
Do NOT output the input sentence.
Do NOT output explanations.
Do NOT output JSON.

Output format:
food_text;value;category_id

Meaning of value:
- For foods that are likely to exist in a standard food database, value = number of PORTIONS
- For foods that are likely NEW / uncommon / not in a standard food database, value = GRAMS

Rules:
1. NEVER decompose dishes, recipes, or prepared meals.
2. Always keep dishes / recipes / branded foods / simple foods as ONE item.
3. Always use English food names.
4. Always output a category_id from the list below.
5. If the food is common or likely known in a food database, estimate portions.
6. If the food is uncommon, very specific, regional, or likely not in a food database, estimate REALISTIC grams instead.
7. For unknown foods, NEVER return tiny weights like 1g, 2g, or 5g unless the food is clearly a condiment, spice, or garnish.
8. For meals, drinks, and plated foods, use realistic serving-level gram estimates.
9. If the user ate nothing, return exactly:
NO_FOOD
10. Output only valid semicolon-separated lines.

Examples:

User input: I ate a pizza
Output:
pizza;1;19

User input: J'ai mangé du poulet avec du riz
Output:
chicken breast;1;18
white rice;1;7

User input: أكلت حمص
Output:
hummus;1;14

User input: I ate dragon fruit pizza
Output:
dragon fruit pizza;250;19

User input: J'ai mangé un tagine et un verre de thé marocain
Output:
tagine;350;15
moroccan tea;250;5

User input: I didn't eat anything
Output:
NO_FOOD

{CATEGORY_BLOCK}
""".strip()


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
            "max_tokens": 200,
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

    return {
        "no_food": parsed["no_food"],
        "items": parsed["items"],
        "usage": payload.get("usage"),
        "raw_output": raw_output,
        "parse_errors": parsed["parse_errors"],
        "model": selected_model,
    }