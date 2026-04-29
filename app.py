from pathlib import Path

import pandas as pd
import streamlit as st

from src.data_prep import CATEGORY_ID_TO_NAME, load_and_prepare_dataset
from src.nutrition import compute_item_nutrition
from src.pipeline import run_pipeline

DATA_PATH = "data/USDA_V2_merged.json"
PROPOSED_ROWS_PATH = Path("data/proposed_rows.csv")
BENCHMARK_RESULTS_DIR = Path("benchmark_results")


@st.cache_data
def get_dataset():
    return load_and_prepare_dataset(DATA_PATH)


@st.cache_data
def load_review_queue():
    if not PROPOSED_ROWS_PATH.exists():
        return pd.DataFrame()

    try:
        return pd.read_csv(PROPOSED_ROWS_PATH, engine="python", on_bad_lines="skip")
    except Exception:
        return pd.DataFrame()


def latest_benchmark_csv() -> Path | None:
    files = sorted(
        BENCHMARK_RESULTS_DIR.glob("v32_vs_v4flash_*.csv"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return files[0] if files else None


@st.cache_data
def load_latest_benchmark_results(path_str: str | None):
    if path_str is None:
        return pd.DataFrame()

    path = Path(path_str)
    if not path.exists():
        return pd.DataFrame()

    try:
        return pd.read_csv(path, engine="python", on_bad_lines="skip")
    except Exception:
        return pd.DataFrame()


def nutrition_to_flat_dict(nutrition: dict | None) -> dict:
    nutrition = nutrition or {}

    return {
        "calories": float(nutrition.get("calories") or 0.0),
        "protein_g": float(nutrition.get("protein_g") or 0.0),
        "carbs_g": float(nutrition.get("carbs_g") or 0.0),
        "fat_g": float(nutrition.get("fat_g") or 0.0),
    }


def build_editable_items_df(items: list[dict]) -> pd.DataFrame:
    rows = []

    for idx, item in enumerate(items):
        nutrition = nutrition_to_flat_dict(item.get("nutrition"))
        grams = item.get("grams")

        rows.append(
            {
                "include": True,
                "item_index": idx,
                "food_text": item.get("food_text"),
                "nutrition_source": item.get("nutrition_source"),
                "matched": item.get("matched"),
                "grams": float(grams) if grams is not None else 0.0,
                "base_grams": float(grams) if grams is not None else 0.0,
                "calories": nutrition["calories"],
                "protein_g": nutrition["protein_g"],
                "carbs_g": nutrition["carbs_g"],
                "fat_g": nutrition["fat_g"],
                "base_calories": nutrition["calories"],
                "base_protein_g": nutrition["protein_g"],
                "base_carbs_g": nutrition["carbs_g"],
                "base_fat_g": nutrition["fat_g"],
            }
        )

    return pd.DataFrame(rows)


def recompute_macros_from_edited_df(df: pd.DataFrame) -> pd.DataFrame:
    edited = df.copy()

    for idx, row in edited.iterrows():
        include = bool(row.get("include", True))
        grams = float(row.get("grams") or 0.0)
        base_grams = float(row.get("base_grams") or 0.0)

        if not include:
            factor = 0.0
        elif base_grams > 0:
            factor = grams / base_grams
        else:
            factor = 1.0

        edited.at[idx, "calories"] = round(float(row.get("base_calories") or 0.0) * factor, 2)
        edited.at[idx, "protein_g"] = round(float(row.get("base_protein_g") or 0.0) * factor, 2)
        edited.at[idx, "carbs_g"] = round(float(row.get("base_carbs_g") or 0.0) * factor, 2)
        edited.at[idx, "fat_g"] = round(float(row.get("base_fat_g") or 0.0) * factor, 2)

    return edited


def compute_totals_from_edited_df(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"calories": 0.0, "protein_g": 0.0, "carbs_g": 0.0, "fat_g": 0.0}

    active = df[df["include"] == True].copy()

    return {
        "calories": round(float(active["calories"].sum()), 2),
        "protein_g": round(float(active["protein_g"].sum()), 2),
        "carbs_g": round(float(active["carbs_g"].sum()), 2),
        "fat_g": round(float(active["fat_g"].sum()), 2),
    }


def search_dataset(dataset: pd.DataFrame, query: str, max_results: int = 25) -> pd.DataFrame:
    query = (query or "").strip().lower()
    if not query:
        return pd.DataFrame()

    english = dataset["english_description"].fillna("").str.lower()
    clean = dataset["clean_description"].fillna("").str.lower()

    exact = dataset[(english == query) | (clean == query)].copy()
    contains = dataset[
        english.str.contains(query, na=False, regex=False)
        | clean.str.contains(query, na=False, regex=False)
    ].copy()

    results = pd.concat([exact, contains]).drop_duplicates().head(max_results)
    return results


def add_manual_item_to_editable_df(
    editable_df: pd.DataFrame,
    dataset_row,
    grams: float,
) -> pd.DataFrame:
    nutrition = compute_item_nutrition(dataset_row, grams)
    nutrition = nutrition_to_flat_dict(nutrition)

    new_row = {
        "include": True,
        "item_index": len(editable_df),
        "food_text": dataset_row.get("english_description"),
        "nutrition_source": "manual_dataset",
        "matched": True,
        "grams": float(grams),
        "base_grams": float(grams),
        "calories": nutrition["calories"],
        "protein_g": nutrition["protein_g"],
        "carbs_g": nutrition["carbs_g"],
        "fat_g": nutrition["fat_g"],
        "base_calories": nutrition["calories"],
        "base_protein_g": nutrition["protein_g"],
        "base_carbs_g": nutrition["carbs_g"],
        "base_fat_g": nutrition["fat_g"],
    }

    return pd.concat([editable_df, pd.DataFrame([new_row])], ignore_index=True)


st.set_page_config(page_title="GymAura Nutrition Prototype", layout="wide")
st.title("GymAura Nutrition Prototype")

st.write(
    "Flow: text → Gemma/OpenRouter extraction → matcher → dataset nutrition or DeepSeek fallback/growth"
)

dataset = get_dataset()

extraction_model = st.text_input(
    "OpenRouter extraction model",
    value="google/gemma-3-4b-it",
)

smart_provider = st.selectbox(
    "Fallback + dataset growth provider",
    ["deepseek_v32", "deepseek_v4_flash"],
    index=0,
)

if "meal_input" not in st.session_state:
    st.session_state.meal_input = ""

if "pipeline_result" not in st.session_state:
    st.session_state.pipeline_result = None

if "editable_items_df" not in st.session_state:
    st.session_state.editable_items_df = pd.DataFrame()

if "validated_items_df" not in st.session_state:
    st.session_state.validated_items_df = pd.DataFrame()

st.text_area(
    "Meal description",
    key="meal_input",
    height=120,
    placeholder="Example: I had 300g of chicken breast with 200g of rice",
)

with st.expander("Category IDs"):
    st.json(CATEGORY_ID_TO_NAME)

if st.button("Run Prototype"):
    current_input = st.session_state.meal_input.strip()

    if not current_input:
        st.warning("Please enter a sentence first.")
    else:
        result = run_pipeline(
            text=current_input,
            dataset=dataset,
            model=extraction_model,
            save_unmatched_candidates=True,
            smart_provider=smart_provider,
            smart_model=None,
        )

        st.session_state.pipeline_result = result
        st.session_state.editable_items_df = build_editable_items_df(result["items"])
        st.session_state.validated_items_df = pd.DataFrame()

result = st.session_state.pipeline_result

if result is not None:
    st.subheader("Input")
    st.write(result["input"])

    st.subheader("Pipeline Summary")
    col1, col2, col3 = st.columns(3)
    col1.metric("Dataset items", result.get("dataset_count", 0))
    col2.metric("Fallback items", result.get("fallback_count", 0))
    col3.metric(
        "Fallback latency",
        f"{result.get('fallback_elapsed_total_ms', 0.0):.2f} ms",
    )

    st.subheader("Detected Items")

    if not result["items"]:
        st.info("No food items detected.")
    else:
        for idx, item in enumerate(result["items"], start=1):
            label = item.get("food_text", "Unknown")
            source = item.get("nutrition_source", "unknown")

            with st.expander(f"Item {idx}: {label} — {source}"):
                st.write(f"**Food:** {item.get('food_text')}")
                st.write(f"**Value:** {item.get('value')}")
                st.write(f"**Unit:** {item.get('unit')}")
                st.write(f"**Portions:** {item.get('portions')}")
                st.write(f"**Grams:** {item.get('grams')} g")
                st.write(f"**Grams source:** {item.get('grams_source')}")

                st.write(f"**Matched:** {item.get('matched')}")
                st.write(f"**Match type:** {item.get('match_type')}")
                st.write(f"**Match score:** {item.get('match_score')}")
                st.write(f"**Search scope:** {item.get('search_scope')}")
                st.write(f"**Match reject reason:** {item.get('match_reject_reason')}")
                st.write(f"**Normalized query:** {item.get('normalized_query')}")
                st.write(f"**Matched description:** {item.get('matched_description')}")
                st.write(f"**LLM category ID:** {item.get('llm_category_id')}")
                st.write(f"**LLM category name:** {item.get('llm_category_name')}")
                st.write(f"**Matched category:** {item.get('matched_category')}")
                st.write(f"**Dataset default portion grams:** {item.get('dataset_default_portion_grams')}")
                st.write(f"**Dataset default portion label:** {item.get('dataset_default_portion_label')}")

                nutrition_source = item.get("nutrition_source")
                if nutrition_source == "dataset":
                    st.success("Nutrition source: dataset")
                elif nutrition_source and "llm_fallback" in nutrition_source:
                    st.warning(f"Nutrition source: {nutrition_source}")
                elif nutrition_source and "llm_failure" in nutrition_source:
                    st.error(f"Nutrition source: {nutrition_source}")
                else:
                    st.info(f"Nutrition source: {nutrition_source}")

                st.subheader("Nutrition")
                st.json(item.get("nutrition"))

                if item.get("fallback_provider"):
                    st.subheader("Fallback Debug")
                    st.write(f"**Fallback provider:** {item.get('fallback_provider')}")
                    st.write(f"**Fallback model:** {item.get('fallback_model')}")

                    if item.get("fallback_elapsed_ms") is not None:
                        st.write(f"**Fallback elapsed time:** {item.get('fallback_elapsed_ms'):.2f} ms")

                    st.write(f"**Used structured output:** {item.get('fallback_used_structured_output')}")
                    st.write(f"**Used second attempt:** {item.get('fallback_used_second_attempt')}")

                    if item.get("fallback_estimated_cost_usd") is not None:
                        st.write(
                            f"**Fallback estimated cost:** ${item.get('fallback_estimated_cost_usd'):.8f}"
                        )

                if item.get("fallback_nutrition_raw_output"):
                    st.subheader("Fallback Raw Output")
                    st.code(item.get("fallback_nutrition_raw_output"), language="text")

                if item.get("fallback_error"):
                    st.subheader("Fallback Error")
                    st.error(item.get("fallback_error"))

                if item.get("needs_clarification"):
                    st.info("This item may need clarification or review.")

                st.subheader("Full Item Debug")
                st.json(item)

    st.subheader("Review / Edit Detected Items")
    st.write(
        "Uncheck wrong items, edit grams, or manually add missing foods from the dataset. Macros recalculate dynamically."
    )

    editable_df = st.session_state.editable_items_df

    if editable_df.empty:
        st.info("No editable items available.")
    else:
        display_columns = [
            "include",
            "food_text",
            "nutrition_source",
            "matched",
            "grams",
            "calories",
            "protein_g",
            "carbs_g",
            "fat_g",
        ]

        edited_df = st.data_editor(
            editable_df[display_columns],
            key="items_editor",
            use_container_width=True,
            hide_index=True,
            disabled=[
                "food_text",
                "nutrition_source",
                "matched",
                "calories",
                "protein_g",
                "carbs_g",
                "fat_g",
            ],
            column_config={
                "include": st.column_config.CheckboxColumn("Include", help="Uncheck to remove this item"),
                "food_text": st.column_config.TextColumn("Food"),
                "nutrition_source": st.column_config.TextColumn("Source"),
                "matched": st.column_config.CheckboxColumn("Matched"),
                "grams": st.column_config.NumberColumn("Grams", min_value=0.0, step=1.0),
                "calories": st.column_config.NumberColumn("Calories"),
                "protein_g": st.column_config.NumberColumn("Protein (g)"),
                "carbs_g": st.column_config.NumberColumn("Carbs (g)"),
                "fat_g": st.column_config.NumberColumn("Fat (g)"),
            },
        )

        full_df = editable_df.copy()
        for col in ["include", "grams"]:
            full_df[col] = edited_df[col]

        st.session_state.editable_items_df = full_df

        st.subheader("Manually Add Item from Dataset")

        manual_query = st.text_input(
            "Search food in USDA dataset",
            placeholder="Example: apple, rice, chicken breast",
        )

        manual_results = search_dataset(dataset, manual_query)

        if manual_query.strip() and manual_results.empty:
            st.info("No matching food found in the dataset.")

        if not manual_results.empty:
            manual_options = {
                f"{row['english_description']} — {row.get('food_category', 'Unknown category')}": idx
                for idx, row in manual_results.iterrows()
            }

            selected_label = st.selectbox(
                "Select food item",
                list(manual_options.keys()),
            )

            manual_grams = st.number_input(
                "Manual item grams",
                min_value=1.0,
                value=100.0,
                step=1.0,
            )

            if st.button("Add Manual Item"):
                selected_idx = manual_options[selected_label]
                selected_row = dataset.loc[selected_idx]

                st.session_state.editable_items_df = add_manual_item_to_editable_df(
                    st.session_state.editable_items_df,
                    selected_row,
                    manual_grams,
                )

                st.success(f"Added {selected_row['english_description']} ({manual_grams}g).")
                st.rerun()

        recalculated_df = recompute_macros_from_edited_df(st.session_state.editable_items_df)
        edited_totals = compute_totals_from_edited_df(recalculated_df)

        st.subheader("Edited Meal Totals")
        st.json(edited_totals)

        st.subheader("Edited Items Preview")
        st.dataframe(
            recalculated_df[
                [
                    "include",
                    "food_text",
                    "nutrition_source",
                    "matched",
                    "grams",
                    "calories",
                    "protein_g",
                    "carbs_g",
                    "fat_g",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )

        if st.button("Validate Edited Meal"):
            st.session_state.validated_items_df = recalculated_df[recalculated_df["include"] == True].copy()
            st.success("Meal validated.")

        if not st.session_state.validated_items_df.empty:
            st.subheader("Validated Meal")
            st.dataframe(st.session_state.validated_items_df, use_container_width=True, hide_index=True)

    st.subheader("Original Meal Totals")
    st.json(result["totals"])

    st.subheader("Raw Extraction Output")
    st.code(result["ai_raw_output"], language="text")

    if result.get("parse_errors"):
        st.subheader("Parse Errors")
        st.json(result["parse_errors"])

    if result.get("ai_usage") is not None:
        st.subheader("Extraction Usage")
        st.write(result["ai_usage"])

        if isinstance(result["ai_usage"], dict):
            cost = result["ai_usage"].get("cost")
            if cost is not None:
                st.write(f"**Extraction cost:** ${cost:.8f}")

    if result.get("estimated_cost_usd") is not None:
        st.subheader("Estimated Total Cost")
        st.write(f"${result['estimated_cost_usd']:.8f}")

st.subheader("Dataset Review Queue")
if PROPOSED_ROWS_PATH.exists():
    review_df = load_review_queue()
    if review_df.empty:
        st.info("No items pending review.")
    else:
        st.write(f"{len(review_df)} item(s) pending review")
        st.dataframe(review_df, use_container_width=True)
else:
    st.info("No proposed_rows.csv file yet. It will be created automatically.")

st.subheader("DeepSeek V3.2 vs V4 Flash Benchmark Results")

latest_path = latest_benchmark_csv()
benchmark_df = load_latest_benchmark_results(str(latest_path) if latest_path else None)

if latest_path:
    st.caption(f"Latest benchmark file: {latest_path}")
else:
    st.caption("No benchmark file found yet.")

if benchmark_df.empty:
    st.info("No benchmark results found yet. Run: python -m scripts.benchmark_v32_vs_v4flash")
else:
    st.write(f"{len(benchmark_df)} benchmark row(s) loaded")

    providers = sorted(benchmark_df["provider"].dropna().unique().tolist()) if "provider" in benchmark_df.columns else []

    selected_provider_filter = st.selectbox(
        "Filter benchmark provider",
        ["All"] + providers,
        index=0,
    )

    filtered_df = benchmark_df.copy()
    if selected_provider_filter != "All" and "provider" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["provider"] == selected_provider_filter]

    st.dataframe(filtered_df, use_container_width=True)

    if providers and "total_score" in benchmark_df.columns:
        summary_rows = []

        for provider in providers:
            provider_df = benchmark_df[benchmark_df["provider"] == provider]

            elapsed = provider_df["elapsed_ms"].dropna() if "elapsed_ms" in provider_df.columns else pd.Series(dtype=float)

            summary_rows.append(
                {
                    "provider": provider,
                    "cases": len(provider_df),
                    "success_rate": round(provider_df["success"].mean(), 3)
                    if "success" in provider_df.columns
                    else None,
                    "avg_total_score": round(provider_df["total_score"].mean(), 3),
                    "sum_total_score": round(provider_df["total_score"].sum(), 3),
                    "avg_elapsed_ms": round(elapsed.mean(), 2) if not elapsed.empty else None,
                    "min_elapsed_ms": round(elapsed.min(), 2) if not elapsed.empty else None,
                    "max_elapsed_ms": round(elapsed.max(), 2) if not elapsed.empty else None,
                    "structured_output_rate": round(provider_df["used_structured_output"].mean(), 3)
                    if "used_structured_output" in provider_df.columns
                    else None,
                    "second_attempt_rate": round(provider_df["used_second_attempt"].mean(), 3)
                    if "used_second_attempt" in provider_df.columns
                    else None,
                    "cached_tokens_total": int(provider_df["cached_tokens"].fillna(0).sum())
                    if "cached_tokens" in provider_df.columns
                    else 0,
                    "cache_write_tokens_total": int(provider_df["cache_write_tokens"].fillna(0).sum())
                    if "cache_write_tokens" in provider_df.columns
                    else 0,
                    "estimated_total_cost_usd": round(provider_df["estimated_cost_usd"].fillna(0).sum(), 8)
                    if "estimated_cost_usd" in provider_df.columns
                    else 0.0,
                }
            )

        st.subheader("V3.2 vs V4 Flash Summary")
        st.dataframe(pd.DataFrame(summary_rows), use_container_width=True)