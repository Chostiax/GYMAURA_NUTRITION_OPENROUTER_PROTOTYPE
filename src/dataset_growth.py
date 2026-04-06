from __future__ import annotations

import csv
from pathlib import Path
from datetime import datetime


PROPOSED_ROWS_PATH = Path("data/proposed_rows.csv")


def append_proposed_row(
    food_text: str,
    grams: float | None,
    category_id: int | None,
    category_name: str | None,
    source_input: str,
    match_score: float,
) -> None:
    PROPOSED_ROWS_PATH.parent.mkdir(parents=True, exist_ok=True)

    file_exists = PROPOSED_ROWS_PATH.exists()

    with PROPOSED_ROWS_PATH.open("a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow([
                "timestamp",
                "food_text",
                "grams",
                "category_id",
                "category_name",
                "source_input",
                "match_score",
                "status",
            ])

        writer.writerow([
            datetime.utcnow().isoformat(),
            food_text,
            grams,
            category_id,
            category_name,
            source_input,
            match_score,
            "pending_review",
        ])