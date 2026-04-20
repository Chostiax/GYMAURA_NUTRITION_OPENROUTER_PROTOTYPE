import pandas as pd
import streamlit as st
from pathlib import Path

from src.data_prep import CATEGORY_ID_TO_NAME, load_and_prepare_dataset
from src.pipeline import run_pipeline

DATA_PATH = "data/USDA_V2_merged.json"
PROPOSED_ROWS_PATH = Path("data/proposed_rows.csv")
BENCHMARK_RESULTS_PATH = Path("benchmark_fallback_growth_results.csv")


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
        return pd.DataFrame(
            columns=[
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
        )


@st.cache_data
def load_benchmark_results():
    if not BENCHMARK_RESULTS_PATH.exists():
        return pd.DataFrame()

    try:
        return pd.read_csv(BENCHMARK_RESULTS_PATH, engine="python", on_bad_lines="skip")
    except Exception:
        return pd.DataFrame()


st.set_page_config(page_title="GymAura Nutrition Prototype", layout="wide")

st.title("GymAura Nutrition Prototype")
st.write(
    "Flow: text (any language) → Gemma/OpenRouter extraction → matcher → dataset nutrition or smart-model fallback"
)

dataset = get_dataset()

extraction_model = st.text_input("OpenRouter extraction model", value="google/gemma-3-4b-it")

smart_provider = st.selectbox(
    "Fallback + dataset growth provider",
    ["openai", "openrouter_deepseek"],
    index=0,
)

smart_model_override = st.text_input(
    "Optional smart model override",
    value="",
    placeholder="Leave blank to use .env default",
)

save_unmatched = st.checkbox("Save unmatched / low-confidence items to review queue", value=True)

example_sentences = [
    "I had chicken with rice",
    "J'ai mangé du poulet avec du riz",
    "أكلت حمص",
    "أكلت بيتزا بيبروني",
    "I ate dragon fruit pizza",
    "I had a caesar salad",
    "J'ai mangé un tagine et un verre de thé marocain",
    "I had beef dumplings",
    "I didn't eat anything today",
]

if "meal_input" not in st.session_state:
    st.session_state.meal_input = ""

selected_example = st.selectbox(
    "Choose an example sentence",
    [""] + example_sentences,
    index=0,
)

if st.button("Load Example"):
    st.session_state.meal_input = selected_example

st.text_area(
    "Meal description",
    key="meal_input",
    height=120,
    placeholder="Example: J'ai mangé du poulet avec du riz",
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
            save_unmatched_candidates=save_unmatched,
            smart_provider=smart_provider,
            smart_model=smart_model_override.strip() or None,
        )

        st.subheader("Input")
        st.write(result["input"])

        st.subheader("Detected Items")
        if not result["items"]:
            st.info("No food items detected.")
        else:
            for idx, item in enumerate(result["items"], start=1):
                with st.expander(f"Item {idx}: {item.get('food_text', 'Unknown')}"):
                    st.write(f"**Food:** {item.get('food_text')}")

                    if item.get("portions") is not None:
                        st.write(f"**Portions:** {item.get('portions')}")

                    if item.get("grams") is not None:
                        st.write(f"**Grams:** {item.get('grams')} g")

                    st.write(f"**Matched:** {item.get('matched')}")
                    st.write(f"**Match type:** {item.get('match_type')}")
                    st.write(f"**Match score:** {item.get('match_score')}")
                    st.write(f"**Search scope:** {item.get('search_scope')}")
                    st.write(f"**Normalized query:** {item.get('normalized_query')}")
                    st.write(f"**Matched description:** {item.get('matched_description')}")
                    st.write(f"**LLM category ID:** {item.get('llm_category_id')}")
                    st.write(f"**LLM category name:** {item.get('llm_category_name')}")
                    st.write(f"**Matched category:** {item.get('matched_category')}")
                    st.write(f"**Dataset default portion grams:** {item.get('dataset_default_portion_grams')}")
                    st.write(f"**Dataset default portion label:** {item.get('dataset_default_portion_label')}")

                    nutrition_source = item.get("nutrition_source")
                    if nutrition_source == "dataset":
                        st.success("Nutrition source: Dataset")
                    elif nutrition_source:
                        st.warning(f"Nutrition source: {nutrition_source}")
                    else:
                        st.warning("Nutrition source: Unknown")

                    if not item.get("matched"):
                        st.warning("This food was not found in the dataset and was added to the review queue.")

                    if item.get("needs_clarification"):
                        st.info("This item may need clarification or review.")

                    st.subheader("Nutrition")
                    st.json(item.get("nutrition"))

                    if item.get("fallback_nutrition_raw_output"):
                        st.subheader("Fallback Nutrition Raw Output")
                        st.code(item.get("fallback_nutrition_raw_output"), language="text")

                    if item.get("fallback_estimated_cost_usd") is not None:
                        st.write(
                            f"**Fallback nutrition estimated cost:** ${item.get('fallback_estimated_cost_usd'):.8f}"
                        )

                    st.subheader("Full Item Debug")
                    st.json(item)

        st.subheader("Meal Totals")
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
    st.info(
        "No proposed_rows.csv file yet. It will be created automatically when an unmatched or low-confidence item is logged."
    )

st.subheader("Fallback / Growth Benchmark Results")

benchmark_df = load_benchmark_results()

if benchmark_df.empty:
    st.info("No benchmark results file found yet. Run: uv run python -m scripts.benchmark_fallback_growth")
else:
    st.write(f"{len(benchmark_df)} benchmark row(s) loaded")

    providers = sorted(benchmark_df["provider"].dropna().unique().tolist())
    selected_provider_filter = st.selectbox(
        "Filter benchmark provider",
        ["All"] + providers,
        index=0,
    )

    filtered_df = benchmark_df.copy()
    if selected_provider_filter != "All":
        filtered_df = filtered_df[filtered_df["provider"] == selected_provider_filter]

    st.dataframe(filtered_df, use_container_width=True)

    if "provider" in benchmark_df.columns and "total_score" in benchmark_df.columns:
        summary_rows = []
        for provider in providers:
            provider_df = benchmark_df[benchmark_df["provider"] == provider]
            summary_rows.append(
                {
                    "provider": provider,
                    "cases": len(provider_df),
                    "avg_total_score": round(provider_df["total_score"].mean(), 3),
                    "sum_total_score": round(provider_df["total_score"].sum(), 3),
                    "estimated_total_cost_usd": round(provider_df["estimated_cost_usd"].fillna(0).sum(), 8),
                }
            )

        st.subheader("Benchmark Summary")
        st.dataframe(pd.DataFrame(summary_rows), use_container_width=True)