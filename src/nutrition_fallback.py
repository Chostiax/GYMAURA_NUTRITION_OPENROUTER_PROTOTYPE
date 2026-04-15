from __future__ import annotations

import os
import re

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DEFAULT_MODEL = os.getenv("OPENAI_NUTRITION_FALLBACK_MODEL", "gpt-5-mini")

client = OpenAI(api_key=OPENAI_API_KEY)

SYSTEM_PROMPT = """
You estimate nutrition values for one food item.

You will receive:
- food name
- estimated grams if available
- original meal description for context

Return ONLY this exact format:
calories;protein_g;carbs_g;fat_g

Rules:
- All values must be numeric.
- Estimate total nutrition for the given item and quantity.
- If grams are missing, infer a realistic serving size from context.
- No explanation.
- No JSON.
- No extra text.
""".strip()


def _looks_numeric_line(line: str) -> bool:
    parts = [p.strip() for p in line.split(";")]
    if len(parts) != 4:
        return False

    for part in parts:
        try:
            float(part)
        except ValueError:
            return False
    return True


def _parse_nutrition_line(raw_text: str) -> dict | None:
    """
    Robust parser:
    - supports single-line responses
    - supports header + data on two lines
    - scans all lines and picks the first valid numeric line
    """
    text = (raw_text or "").strip()
    if not text:
        return None

    lines = [line.strip() for line in text.splitlines() if line.strip()]

    # First try whole text as one line
    if _looks_numeric_line(text):
        line = text
    else:
        line = None
        for candidate in lines:
            if _looks_numeric_line(candidate):
                line = candidate
                break

    if line is None:
        return None

    parts = [p.strip() for p in line.split(";")]

    try:
        return {
            "calories": round(float(parts[0]), 2),
            "protein_g": round(float(parts[1]), 2),
            "carbs_g": round(float(parts[2]), 2),
            "fat_g": round(float(parts[3]), 2),
        }
    except ValueError:
        return None


def _estimate_openai_cost_usd(usage) -> float | None:
    """
    Keep this lightweight and safe.
    If usage structure does not expose token counts in the expected way,
    just return None.
    """
    if usage is None:
        return None

    # Some SDK responses expose usage as an object, others as dict-like
    try:
        input_tokens = getattr(usage, "input_tokens", None)
        output_tokens = getattr(usage, "output_tokens", None)
        if input_tokens is None and hasattr(usage, "to_dict"):
            usage_dict = usage.to_dict()
            input_tokens = usage_dict.get("input_tokens")
            output_tokens = usage_dict.get("output_tokens")
    except Exception:
        input_tokens = None
        output_tokens = None

    # Rough placeholder based on your supervisor’s requested budget envelope.
    # Replace later with exact model pricing if you want stricter accounting.
    if input_tokens is None or output_tokens is None:
        return None

    # Conservative simple estimate in USD:
    # input: $0.30 / 1M tokens
    # output: $0.80 / 1M tokens
    cost = (input_tokens / 1_000_000) * 0.30 + (output_tokens / 1_000_000) * 0.80
    return round(cost, 8)


def estimate_nutrition_with_llm(
    food_text: str,
    grams: float | None,
    original_text: str,
    model: str | None = None,
) -> dict:
    if not OPENAI_API_KEY:
        raise ValueError("Missing OPENAI_API_KEY in environment.")

    selected_model = model or DEFAULT_MODEL

    user_prompt = f"""
Food: {food_text}
Estimated grams: {grams if grams is not None else "unknown"}
Original meal description: {original_text}
""".strip()

    response = client.responses.create(
        model=selected_model,
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )

    raw_output = getattr(response, "output_text", "") or ""
    parsed = _parse_nutrition_line(raw_output)

    if parsed is None:
        parsed = {
            "calories": 0.0,
            "protein_g": 0.0,
            "carbs_g": 0.0,
            "fat_g": 0.0,
        }

    usage = getattr(response, "usage", None)
    estimated_cost_usd = _estimate_openai_cost_usd(usage)

    return {
        "nutrition": parsed,
        "raw_output": raw_output,
        "usage": usage,
        "estimated_cost_usd": estimated_cost_usd,
        "model": selected_model,
    }