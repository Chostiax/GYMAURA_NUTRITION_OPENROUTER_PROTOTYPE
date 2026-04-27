from __future__ import annotations

import csv
import json
import re
import unicodedata
from datetime import datetime
from pathlib import Path

from src.nutrition_fallback import estimate_nutrition_with_llm
from tests.fallback_growth_benchmark_cases import BENCHMARK_CASES

RESULTS_DIR = Path("benchmark_results")


def normalize_text(text: str) -> str:
    text = (text or "").strip().lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return " ".join(text.split())


def score_case(case: dict, nutrition: dict | None) -> dict:
    if nutrition is None:
        return {
            "calorie_score": 0.0,
            "total_score": 0.0,
        }

    calories = nutrition.get("calories")

    calorie_score = 0.0
    if calories is not None:
        if case["expected_calories_min"] <= calories <= case["expected_calories_max"]:
            calorie_score = 1.0

    return {
        "calorie_score": calorie_score,
        "total_score": calorie_score,
    }


def run_case(provider: str, case: dict) -> dict:
    nutrition = None
    raw_output = None
    model = None
    estimated_cost_usd = None
    elapsed_ms = None
    prompt_tokens = None
    completion_tokens = None
    cached_tokens = None
    cache_write_tokens = None
    used_structured_output = False
    used_second_attempt = False
    success = False
    error = None

    try:
        result = estimate_nutrition_with_llm(
            food_text=case["food_text"],
            grams=case["grams"],
            original_text=case["input"],
            provider=provider,
            model=None,
            category_id=case.get("expected_category_id"),
            category_name=case.get("expected_category_name"),
        )

        nutrition = result.get("nutrition")
        raw_output = result.get("raw_output")
        model = result.get("model")
        estimated_cost_usd = result.get("estimated_cost_usd")
        elapsed_ms = result.get("elapsed_ms")
        prompt_tokens = result.get("prompt_tokens")
        completion_tokens = result.get("completion_tokens")
        cached_tokens = result.get("cached_tokens")
        cache_write_tokens = result.get("cache_write_tokens")
        used_structured_output = result.get("used_structured_output", False)
        used_second_attempt = result.get("used_second_attempt", False)
        success = nutrition is not None

    except Exception as e:
        error = str(e)

    score = score_case(case, nutrition)

    return {
        "provider": provider,
        "case_id": case["id"],
        "input": case["input"],
        "food_text": case["food_text"],
        "grams_input": case["grams"],
        "expected_canonical": case["expected_canonical"],
        "expected_category_id": case["expected_category_id"],
        "expected_calories_min": case["expected_calories_min"],
        "expected_calories_max": case["expected_calories_max"],
        "got_calories": nutrition.get("calories") if nutrition else None,
        "got_protein_g": nutrition.get("protein_g") if nutrition else None,
        "got_carbs_g": nutrition.get("carbs_g") if nutrition else None,
        "got_fat_g": nutrition.get("fat_g") if nutrition else None,
        "calorie_score": score["calorie_score"],
        "total_score": score["total_score"],
        "success": success,
        "estimated_cost_usd": estimated_cost_usd,
        "elapsed_ms": elapsed_ms,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "cached_tokens": cached_tokens,
        "cache_write_tokens": cache_write_tokens,
        "used_structured_output": used_structured_output,
        "used_second_attempt": used_second_attempt,
        "raw_output": raw_output,
        "model": model,
        "error": error,
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict]) -> dict:
    providers = sorted({row["provider"] for row in rows})
    summary = {}

    for provider in providers:
        provider_rows = [row for row in rows if row["provider"] == provider]
        n = len(provider_rows)

        success_count = sum(1 for row in provider_rows if row["success"])
        total_score = sum(float(row["total_score"] or 0) for row in provider_rows)
        total_cost = sum(float(row["estimated_cost_usd"] or 0) for row in provider_rows)

        elapsed_values = [
            float(row["elapsed_ms"])
            for row in provider_rows
            if row["elapsed_ms"] not in (None, "")
        ]

        structured_count = sum(1 for row in provider_rows if row["used_structured_output"])
        second_attempt_count = sum(1 for row in provider_rows if row["used_second_attempt"])
        cached_total = sum(int(row["cached_tokens"] or 0) for row in provider_rows)
        cache_write_total = sum(int(row["cache_write_tokens"] or 0) for row in provider_rows)

        summary[provider] = {
            "cases": n,
            "success_count": success_count,
            "success_rate": round(success_count / n, 4) if n else 0,
            "avg_total_score": round(total_score / n, 4) if n else 0,
            "sum_total_score": round(total_score, 4),
            "estimated_total_cost_usd": round(total_cost, 10),
            "avg_elapsed_ms": round(sum(elapsed_values) / len(elapsed_values), 2) if elapsed_values else None,
            "min_elapsed_ms": round(min(elapsed_values), 2) if elapsed_values else None,
            "max_elapsed_ms": round(max(elapsed_values), 2) if elapsed_values else None,
            "structured_output_rate": round(structured_count / n, 4) if n else 0,
            "second_attempt_rate": round(second_attempt_count / n, 4) if n else 0,
            "cached_tokens_total": cached_total,
            "cache_write_tokens_total": cache_write_total,
        }

    return summary


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    csv_path = RESULTS_DIR / f"v32_vs_v4flash_{timestamp}.csv"
    summary_path = RESULTS_DIR / f"v32_vs_v4flash_{timestamp}_summary.json"

    providers = ["deepseek_v32", "deepseek_v4_flash"]
    rows = []

    for provider in providers:
        print(f"\nRunning benchmark for {provider}", flush=True)

        for idx, case in enumerate(BENCHMARK_CASES, start=1):
            print(f"[{provider}] case {idx}/{len(BENCHMARK_CASES)}: {case['id']}", flush=True)
            row = run_case(provider, case)
            rows.append(row)

            write_csv(csv_path, rows)

    summary = summarize(rows)
    write_json(summary_path, summary)

    print("\nBenchmark complete.")
    print(f"CSV: {csv_path}")
    print(f"Summary: {summary_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()