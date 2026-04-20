from __future__ import annotations

import csv
import json
import re
import unicodedata
from datetime import datetime
from pathlib import Path

PROPOSED_ROWS_PATH = Path("data/proposed_rows.csv")

HEADER = [
    "timestamp",
    "raw_food_text",
    "normalized_food_text",
    "grams",
    "category_id",
    "category_name",
    "source_input",
    "match_score",
    "status",
    "fallback_provider",
    "fallback_model",
    "fallback_nutrition_json",
    "fallback_raw_output",
]


def _normalize_food_text_for_queue(text: str) -> str:
    text = (text or "").strip().lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = " ".join(text.split())
    return text


def _ensure_file_schema() -> None:
    if not PROPOSED_ROWS_PATH.exists():
        PROPOSED_ROWS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with PROPOSED_ROWS_PATH.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(HEADER)
        return

    with PROPOSED_ROWS_PATH.open("r", encoding="utf-8", newline="") as f:
        first_line = f.readline().strip()

    expected = ",".join(HEADER)
    if first_line != expected:
        backup_path = PROPOSED_ROWS_PATH.with_suffix(".backup.csv")
        PROPOSED_ROWS_PATH.replace(backup_path)
        with PROPOSED_ROWS_PATH.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(HEADER)


def append_proposed_row(
    food_text: str,
    grams: float | None,
    category_id: int | None,
    category_name: str | None,
    source_input: str,
    match_score: float,
    status: str = "pending_review",
    fallback_provider: str | None = None,
    fallback_model: str | None = None,
    fallback_nutrition: dict | None = None,
    fallback_raw_output: str | None = None,
) -> bool:
    """
    Deduplicate by normalized_food_text.
    If already present, do not add another row.
    Returns True if appended, False if skipped.
    """
    _ensure_file_schema()

    normalized_food_text = _normalize_food_text_for_queue(food_text)

    existing_normalized = set()
    with PROPOSED_ROWS_PATH.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            existing_normalized.add((row.get("normalized_food_text") or "").strip())

    if normalized_food_text in existing_normalized:
        return False

    row = {
        "timestamp": datetime.utcnow().isoformat(),
        "raw_food_text": food_text,
        "normalized_food_text": normalized_food_text,
        "grams": "" if grams is None else round(float(grams), 2),
        "category_id": "" if category_id is None else category_id,
        "category_name": category_name or "",
        "source_input": source_input,
        "match_score": round(float(match_score or 0.0), 4),
        "status": status,
        "fallback_provider": fallback_provider or "",
        "fallback_model": fallback_model or "",
        "fallback_nutrition_json": json.dumps(fallback_nutrition or {}, ensure_ascii=False),
        "fallback_raw_output": fallback_raw_output or "",
    }

    with PROPOSED_ROWS_PATH.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=HEADER)
        writer.writerow(row)

    return True