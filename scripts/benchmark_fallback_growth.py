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
    got_name = normalize_text(parsed.get("canonical_food_text"))

    if got_name == expected_name:
        canonical_score = 2.0
    elif expected_name in got_name or got_name in expected_name:
        canonical_score = 1.0
    else:
        canonical_score = 0.0

    got_category_id = parsed.get("category_id")
    category_score = 1.0 if got_category_id == case["expected_category_id"] else 0.0

    nutrition = parsed.get("nutrition") or {}
    calories = nutrition.get("calories")

    calorie_score = 0.0
    if calories is not None and case["expected_calories_min"] <= calories <= case["expected_calories_max"]:
        calorie_score = 1.0

    total_score = canonical_score + category_score + calorie_score

    return {
        "canonical_score": canonical_score,
        "category_score": category_score,
        "calorie_score": calorie_score,
        "total_score": total_score,
    }


def init_csv() -> list[str]:
    fieldnames = [
        "provider",
        "case_id",
        "input",
        "food_text",
        "grams_input",
        "expected_canonical",
        "expected_category_id",
        "expected_calories_min",
        "expected_calories_max",
        "got_canonical",
        "got_category_id",
        "got_grams",
        "got_calories",
        "canonical_score",
        "category_score",
        "calorie_score",
        "total_score",
        "estimated_cost_usd",
        "raw_output",
        "model",
        "used_repair_pass",
        "used_second_attempt",
        "success",
        "error",
    ]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

    return fieldnames


def append_row(row: dict, fieldnames: list[str]) -> None:
    with OUTPUT_PATH.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writerow(row)


def run_provider(provider: str, fieldnames: list[str]) -> list[dict]:
    rows = []

    print(f"\nRunning provider: {provider}", flush=True)

    for idx, case in enumerate(BENCHMARK_CASES, start=1):
        print(f"[{provider}] Case {idx}/{len(BENCHMARK_CASES)} -> {case['id']}", flush=True)

        parsed = None
        raw_output = None
        model = None
        estimated_cost_usd = None
        used_repair_pass = False
        used_second_attempt = False
        success = False
        error = None

        try:
            result = enrich_unmatched_food_with_llm(
                food_text=case["food_text"],
                grams=case["grams"],
                original_text=case["input"],
                provider=provider,
                model=None,
                category_id=case.get("expected_category_id"),
                category_name=case.get("expected_category_name"),
            )

            parsed = result.get("parsed")
            raw_output = result.get("raw_output")
            model = result.get("model")
            estimated_cost_usd = result.get("estimated_cost_usd")
            used_repair_pass = result.get("used_repair_pass", False)
            used_second_attempt = result.get("used_second_attempt", False)
            success = parsed is not None

        except Exception as e:
            error = str(e)

        score = score_case(case, parsed)

        row = {
            "provider": provider,
            "case_id": case["id"],
            "input": case["input"],
            "food_text": case["food_text"],
            "grams_input": case["grams"],
            "expected_canonical": case["expected_canonical"],
            "expected_category_id": case["expected_category_id"],
            "expected_calories_min": case["expected_calories_min"],
            "expected_calories_max": case["expected_calories_max"],
            "got_canonical": parsed.get("canonical_food_text") if parsed else None,
            "got_category_id": parsed.get("category_id") if parsed else None,
            "got_grams": parsed.get("grams") if parsed else None,
            "got_calories": (parsed.get("nutrition") or {}).get("calories") if parsed else None,
            "canonical_score": score["canonical_score"],
            "category_score": score["category_score"],
            "calorie_score": score["calorie_score"],
            "total_score": score["total_score"],
            "estimated_cost_usd": estimated_cost_usd,
            "raw_output": raw_output,
            "model": model,
            "used_repair_pass": used_repair_pass,
            "used_second_attempt": used_second_attempt,
            "success": success,
            "error": error,
        }

        rows.append(row)
        append_row(row, fieldnames)

    return rows


def print_summary(rows: list[dict], provider: str) -> None:
    provider_rows = [r for r in rows if r["provider"] == provider]
    if not provider_rows:
        return

    total_cases = len(provider_rows)
    success_count = sum(1 for r in provider_rows if r["success"])
    failure_count = total_cases - success_count
    total_score = sum(r["total_score"] for r in provider_rows)
    avg_score = total_score / total_cases if total_cases else 0.0
    total_cost = sum((r["estimated_cost_usd"] or 0.0) for r in provider_rows)
    repair_count = sum(1 for r in provider_rows if r["used_repair_pass"])
    second_attempt_count = sum(1 for r in provider_rows if r["used_second_attempt"])

    print("=" * 70)
    print(f"PROVIDER: {provider}")
    print(f"Cases: {total_cases}")
    print(f"Successful cases: {success_count}")
    print(f"Failed cases: {failure_count}")
    print(f"Average score: {avg_score:.2f} / 4.00")
    print(f"Total score: {total_score:.2f}")
    print(f"Estimated total cost: ${total_cost:.8f}")
    print(f"Used second attempt: {second_attempt_count}")
    print(f"Used repair pass: {repair_count}")


def main():
    fieldnames = init_csv()
    all_rows = []

    for provider in ["openai", "openrouter_deepseek"]:
        rows = run_provider(provider, fieldnames)
        all_rows.extend(rows)

    for provider in ["openai", "openrouter_deepseek"]:
        print_summary(all_rows, provider)

    print("=" * 70)
    print(f"Detailed results written to: {OUTPUT_PATH.resolve()}")


if __name__ == "__main__":
    main()