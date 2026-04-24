from __future__ import annotations

from typing import Any

from src.data_prep import CATEGORY_ID_TO_NAME, CATEGORY_NAME_TO_ID


VALID_UNITS = {"g", "portion"}


def _parse_value(value: str | None) -> float | None:
    if value is None:
        return None

    cleaned = value.strip().lower()
    if not cleaned:
        return None

    cleaned = cleaned.replace(",", ".")
    cleaned = cleaned.replace("grams", "")
    cleaned = cleaned.replace("gram", "")
    cleaned = cleaned.replace("g", "")
    cleaned = cleaned.strip()

    try:
        parsed = float(cleaned)
        if parsed <= 0:
            return None
        return parsed
    except ValueError:
        return None


def _parse_unit(unit: str | None) -> str | None:
    if unit is None:
        return None

    cleaned = unit.strip().lower()

    if cleaned in {"g", "gram", "grams"}:
        return "g"

    if cleaned in {"portion", "portions", "serving", "servings", "piece", "pieces", "unit", "units"}:
        return "portion"

    return None


def _parse_category(value: str | None) -> tuple[int | None, str | None]:
    if value is None:
        return None, None

    cleaned = value.strip()
    if not cleaned:
        return None, None

    if cleaned.isdigit():
        category_id = int(cleaned)
        return category_id if category_id in CATEGORY_ID_TO_NAME else None, CATEGORY_ID_TO_NAME.get(category_id)

    category_name = cleaned
    category_id = CATEGORY_NAME_TO_ID.get(category_name)
    return category_id, category_name if category_id is not None else None


def parse_semicolon_output(raw_text: str) -> dict[str, Any]:
    """
    Expected output:
    food_text;value;unit;category_id

    Examples:
    chicken breast;300;g;18
    apple;2;portion;12
    tagine;350;g;15

    Backward compatible with old:
    food_text;value;category_id
    """

    text = (raw_text or "").strip()

    if not text:
        return {
            "no_food": False,
            "items": [],
            "raw_output": raw_text,
            "parse_errors": ["Empty model output"],
        }

    if text.upper() == "NO_FOOD":
        return {
            "no_food": True,
            "items": [],
            "raw_output": raw_text,
            "parse_errors": [],
        }

    items: list[dict[str, Any]] = []
    parse_errors: list[str] = []

    lines = [line.strip() for line in text.splitlines() if line.strip()]

    for idx, line in enumerate(lines, start=1):
        parts = [part.strip() for part in line.split(";")]

        if len(parts) == 4:
            food_text, value_text, unit_text, category_text = parts
            unit = _parse_unit(unit_text)

        elif len(parts) == 3:
            # Backward compatibility with old extractor output
            food_text, value_text, category_text = parts
            unit = None
            parse_errors.append(
                f"Line {idx}: old 3-field format detected. Unit is missing -> {line}"
            )

        else:
            parse_errors.append(
                f"Line {idx}: expected 4 fields separated by ';' but got {len(parts)} -> {line}"
            )
            continue

        if not food_text:
            parse_errors.append(f"Line {idx}: missing food_text -> {line}")
            continue

        value = _parse_value(value_text)
        category_id, category_name = _parse_category(category_text)

        if unit is None and len(parts) == 4:
            parse_errors.append(f"Line {idx}: invalid unit '{unit_text}' -> {line}")

        items.append(
            {
                "food_text": food_text.lower().strip(),
                "value": value,
                "unit": unit,
                "category_id": category_id,
                "category_name": category_name,
            }
        )

    return {
        "no_food": False,
        "items": items,
        "raw_output": raw_text,
        "parse_errors": parse_errors,
    }