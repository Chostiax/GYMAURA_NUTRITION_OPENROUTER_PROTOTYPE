from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

RESULTS_DIR = Path("benchmark_results")


def latest_csv() -> Path | None:
    files = sorted(RESULTS_DIR.glob("v32_vs_v4flash_*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None


def main() -> None:
    path = latest_csv()

    if path is None:
        print("No benchmark CSV found in benchmark_results/.")
        return

    df = pd.read_csv(path)

    print(f"Analyzing: {path}")
    print()

    summary_rows = []

    for provider, group in df.groupby("provider"):
        elapsed = group["elapsed_ms"].dropna()

        summary_rows.append(
            {
                "provider": provider,
                "cases": len(group),
                "success_rate": round(group["success"].mean(), 4),
                "avg_total_score": round(group["total_score"].mean(), 4),
                "total_cost_usd": round(group["estimated_cost_usd"].fillna(0).sum(), 10),
                "avg_elapsed_ms": round(elapsed.mean(), 2) if not elapsed.empty else None,
                "min_elapsed_ms": round(elapsed.min(), 2) if not elapsed.empty else None,
                "max_elapsed_ms": round(elapsed.max(), 2) if not elapsed.empty else None,
                "structured_output_rate": round(group["used_structured_output"].mean(), 4)
                if "used_structured_output" in group
                else None,
                "second_attempt_rate": round(group["used_second_attempt"].mean(), 4)
                if "used_second_attempt" in group
                else None,
                "cached_tokens_total": int(group["cached_tokens"].fillna(0).sum())
                if "cached_tokens" in group
                else 0,
                "cache_write_tokens_total": int(group["cache_write_tokens"].fillna(0).sum())
                if "cache_write_tokens" in group
                else 0,
            }
        )

    summary_df = pd.DataFrame(summary_rows)

    print(summary_df.to_string(index=False))
    print()

    print("Worst cases by provider:")
    for provider, group in df.groupby("provider"):
        print(f"\n{provider}")
        worst = group.sort_values(["success", "total_score"], ascending=[True, True]).head(5)

        for _, row in worst.iterrows():
            print(
                f"- {row['case_id']} | success={row['success']} | "
                f"score={row['total_score']} | elapsed={row.get('elapsed_ms')} ms | "
                f"error={row.get('error')}"
            )

    output_path = RESULTS_DIR / f"{path.stem}_analysis_summary.json"
    output_path.write_text(
        json.dumps(summary_rows, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print()
    print(f"Analysis written to: {output_path}")


if __name__ == "__main__":
    main()