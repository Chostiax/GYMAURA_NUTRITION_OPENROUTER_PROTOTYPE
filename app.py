import pandas as pd
import streamlit as st
from pathlib import Path

from src.data_prep import CATEGORY_ID_TO_NAME, load_and_prepare_dataset
from src.pipeline import run_pipeline

DATA_PATH = "data/USDA_V2_merged.json"
PROPOSED_ROWS_PATH = Path("data/proposed_rows.csv")


@st.cache_data
def get_dataset():
    return load_and_prepare_dataset(DATA_PATH)


@st.cache_data
def load_review_queue():
    if PROPOSED_ROWS_PATH.exists():
        return pd.read_csv(PROPOSED_ROWS_PATH)
    return pd.DataFrame()


st.set_page_config(page_title="GymAura Nutrition Prototype", layout="wide")

st.title("GymAura Nutrition Prototype")
st.write(
    "Flow: text (any language) → Gemma/OpenRouter → food;value;category_id → matcher → nutrition"
)

dataset = get_dataset()

model = st.text_input("OpenRouter model", value="google/gemma-3-4b-it")
save_unmatched = st.checkbox("Save unmatched / low-confidence items to review queue", value=True)

example_sentences = [
    "I had chicken with rice",
    "J'ai mangé du poulet avec du riz",
    "أكلت حمص",
    "أكلت بيتزا بيبروني",
    "I ate dragon fruit pizza",
    "I had a caesar salad",
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

user_input = st.text_area(
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
            model=model,
            save_unmatched_candidates=save_unmatched,
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
                    else:
                        st.warning("Nutrition source: AI estimate")

                    if not item.get("matched"):
                        st.warning("This food was not found in the dataset and was added to the review queue.")

                    if item.get("needs_clarification"):
                        st.info("This item may need clarification or review.")

                    st.subheader("Nutrition")
                    st.json(item.get("nutrition"))

                    if item.get("fallback_nutrition_raw_output"):
                        st.subheader("Fallback Nutrition Raw Output")
                        st.code(item.get("fallback_nutrition_raw_output"), language="text")

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
            st.subheader("Usage")
            st.write(result["ai_usage"])

            cost = result["ai_usage"].get("cost")
            if cost is not None:
                st.write(f"Cost: ${cost:.6f}")

st.subheader("Dataset Review Queue")

if PROPOSED_ROWS_PATH.exists():
    review_df = load_review_queue()

    if review_df.empty:
        st.info("No items pending review.")
    else:
        st.write(f"{len(review_df)} item(s) pending review")
        st.dataframe(review_df, use_container_width=True)
else:
    st.info("No proposed_rows.csv file yet. It will be created automatically when an unmatched or low-confidence item is logged.")