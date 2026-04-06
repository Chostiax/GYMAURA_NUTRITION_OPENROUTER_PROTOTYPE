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
food_text;grams;category_id

Rules:
- One food per line.
- Always estimate grams directly.
- Always use English food names.
- Always output a category_id from the list below.
- If the user ate nothing, return exactly: NO_FOOD

Food extraction rules:
1. SIMPLE FOOD
- If it is a simple food, keep it as one item.

2. BRANDED OR PACKAGED PRODUCT
- Keep it as one item.
- Never decompose it.

3. DISH / RECIPE / PREPARED MEAL
- You may decompose it into its standard core ingredients only if you are reasonably confident.
- If you decompose a dish, the decomposition must be reasonably complete at the core-ingredient level.
- Do NOT return a partial decomposition with only one or two obvious ingredients from a larger dish.
- If you are not confident that you can provide a reasonably complete core decomposition, keep the dish as one item.

4. CONTEXT
- If context suggests a specific edible form, prefer that over unrelated processed products.
- Example: chicken with rice -> chicken breast or chicken thigh, not chicken spread.

5. GENERAL
- Do not invent unrelated ingredients.
- Do not add non-standard extras.
- Prefer one correct whole dish over a wrong or incomplete decomposition.

{CATEGORY_BLOCK}
""".strip()


def extract_foods_with_openrouter(text: str, model: str | None = None) -> dict:
    if not OPENROUTER_API_KEY:
        raise ValueError("Missing OPENROUTER_API_KEY in environment.")

    selected_model = model or DEFAULT_MODEL
    print("DEBUG USER TEXT SENT TO MODEL:", repr(text))
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