from __future__ import annotations

import csv
import re
import unicodedata
from pathlib import Path

from src.review_enricher import enrich_unmatched_food_with_llm
from tests.fallback_growth_benchmark_cases import BENCHMARK_CASES


OUTPUT_PATH = Path("benchmark_fallback_growth_results.csv")


def normalize_text(text: str) -> str:
    text = (text or "").strip().lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = " ".join(text.split())
    return text


def score_case(case: dict, parsed: dict | None) -> dict:
    if parsed is None:
        return {
            "canonical_score": 0.0,
            "category_score": 0.0,
            "calorie_score": 0.0,
            "total_score": 0.0,
        }

    expected_name = normalize_text(case["expected_canonical"])
    got_name = normalize_text(parsed["canonical_food_text"])

    if got_name == expected_name:
        canonical_score = 2.0
    elif expected_name in got_name or got_name in expected_name:
        canonical_score = 1.0
    else:
        canonical_score = 0.0

    category_score = 1.0 if parsed["category_id"] == case["expected_category_id"] else 0.0

    calories = parsed["nutrition"]["calories"]
    calorie_score = 1.0 if case["expected_calories_min"] <= calories <= case["expected_calories_max"] else 0.0

    total_score = canonical_score + category_score + calorie_score

    return {
        "canonical_score": canonical_score,
        "category_score": category_score,
        "calorie_score": calorie_score,
        "total_score": total_score,
    }


def run_provider(provider: str) -> list[dict]:
    rows = []

    for case in BENCHMARK_CASES:
        result = enrich_unmatched_food_with_llm(
            food_text=case["food_text"],
            grams=case["grams"],
            original_text=case["input"],
            provider=provider,
        )

        parsed = result["parsed"]
        score = score_case(case, parsed)

        rows.append({
            "provider": provider,
            "case_id": case["id"],
            "input": case["input"],
            "food_text": case["food_text"],
            "grams_input": case["grams"],
            "expected_canonical": case["expected_canonical"],
            "expected_category_id": case["expected_category_id"],
            "expected_calories_min": case["expected_calories_min"],
            "expected_calories_max": case["expected_calories_max"],
            "got_canonical": parsed["canonical_food_text"] if parsed else None,
            "got_category_id": parsed["category_id"] if parsed else None,
            "got_grams": parsed["grams"] if parsed else None,
            "got_calories": parsed["nutrition"]["calories"] if parsed else None,
            "canonical_score": score["canonical_score"],
            "category_score": score["category_score"],
            "calorie_score": score["calorie_score"],
            "total_score": score["total_score"],
            "estimated_cost_usd": result["estimated_cost_usd"],
            "raw_output": result["raw_output"],
            "model": result["model"],
        })

    return rows


def write_csv(rows: list[dict]) -> None:
    if not rows:
        return

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows: list[dict], provider: str) -> None:
    provider_rows = [r for r in rows if r["provider"] == provider]
    if not provider_rows:
        return

    total_score = sum(r["total_score"] for r in provider_rows)
    max_score = len(provider_rows) * 4.0
    avg_score = total_score / len(provider_rows)

    total_cost = sum((r["estimated_cost_usd"] or 0.0) for r in provider_rows)

    print("=" * 70)
    print(f"PROVIDER: {provider}")
    print(f"Cases: {len(provider_rows)}")
    print(f"Average score: {avg_score:.2f} / 4.00")
    print(f"Total score: {total_score:.2f} / {max_score:.2f}")
    print(f"Estimated total cost: ${total_cost:.8f}")

    worst = sorted(provider_rows, key=lambda r: r["total_score"])[:5]
    print("Worst 5 cases:")
    for row in worst:
        print(f"- {row['case_id']} | score={row['total_score']} | input={row['input']}")


def main():
    all_rows = []

    for provider in ["openai", "openrouter_deepseek"]:
        rows = run_provider(provider)
        all_rows.extend(rows)

    write_csv(all_rows)

    for provider in ["openai", "openrouter_deepseek"]:
        print_summary(all_rows, provider)

    print("=" * 70)
    print(f"Detailed results written to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()