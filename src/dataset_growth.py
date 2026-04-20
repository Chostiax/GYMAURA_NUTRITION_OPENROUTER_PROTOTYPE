from __future__ import annotations

import csv
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


def _row_already_exists(normalized_food_text: str, category_id: int | None) -> bool:
    if not PROPOSED_ROWS_PATH.exists():
        return False

    with PROPOSED_ROWS_PATH.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            existing_norm = row.get("normalized_food_text", "").strip().lower()
            existing_cat = row.get("category_id")
            try:
                existing_cat = int(existing_cat) if existing_cat not in (None, "", "None") else None
            except Exception:
                existing_cat = None

            if existing_norm == normalized_food_text and existing_cat == category_id:
                return True

    return False


def append_proposed_row(
    food_text: str,
    grams: float | None,
    category_id: int | None,
    category_name: str | None,
    source_input: str,
    match_score: float,
) -> None:
    _ensure_file_schema()

    normalized_food_text = _normalize_food_text_for_queue(food_text)

    if _row_already_exists(normalized_food_text, category_id):
        return

    with PROPOSED_ROWS_PATH.open("a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.utcnow().isoformat(),
            food_text,
            normalized_food_text,
            grams,
            category_id,
            category_name,
            source_input,
            match_score,
            "pending_review",
        ])