## Overview

This project is a **nutrition extraction prototype** built for GymAura.

It explores using a **lightweight LLM (via OpenRouter, e.g. Gemma 4B)** as an alternative to larger models (OpenAI) for:

- food extraction  
- portion estimation  
- structured meal understanding  

The goal is to build a **scalable, cost-efficient system** while enabling **dataset growth from real user inputs**.

---

## 🚀 Key Features

### 1. LLM-based Food Extraction
- Uses OpenRouter (Gemma 4B or similar lightweight models)
- Handles **multilingual input** (Arabic, French, etc.)
- Outputs structured data in a robust format:


food_text;grams;category_id


---

### 2. No Hardcoded Portions
- Removed all rule-based mappings like:
  - "a bowl" = 100g  
  - "a bit" = 50g  
- The LLM **estimates quantities directly in grams**

---

### 3. Category-Aware Matching
- LLM predicts a **food category**
- Matching is done **within that category first**
- Prevents incorrect mappings like:
  - `pizza → pizza rolls`
  - `chicken → chicken spread`

---

### 4. Improved Matching & Ranking
Custom matcher combines:
- fuzzy matching (RapidFuzz)
- token overlap scoring
- category filtering
- penalty system for incorrect candidates

Design principle:

> Better to return no match than a wrong match

---

### 5. Dataset Growth System (Core Feature)

Unmatched or low-confidence items are automatically stored in:


data/proposed_rows.csv


This enables:
- continuous dataset enrichment  
- real-world data collection  
- human review loop  

Example:


pizza,200,19,"أكلت بيتزا بيبروني",0,pending_review


---

### 6. Smart Dish Handling

The system avoids hardcoding recipes and instead uses general rules:

- Simple food → keep as one item  
- Branded product → keep as one item  
- Dish/recipe:
  - Decompose **only if confident**
  - Otherwise keep as a single item  

Key principle:

> Incomplete decomposition is worse than no decomposition

---

### 7. Multilingual Support

- Input can be in **any language supported by the model**  
- Output is always normalized in **English**  

Tested with:
- Arabic 
- French 
- English 
- Russian
- Italian 
---

## 🧠 Pipeline


text
→ OpenRouter (LLM)
→ semicolon output
→ parser
→ category-aware matcher
→ nutrition calculation


---

## 📊 Example

**Input:**

أكلت بيتزا بيبروني


**LLM Output:**

pizza;200;19


**Behavior:**
- If matched → nutrition computed  
- If not matched → added to dataset growth queue  

---

## ⚙️ Tech Stack

- Python  
- Streamlit (UI)  
- OpenRouter API  
- RapidFuzz (matching)  
- Pandas  
- UV (package manager)  

---

## 🛠 Setup

### 1. Install UV

```bash
pip install uv
2. Create .env
OPENROUTER_API_KEY=your_api_key
OPENROUTER_MODEL=google/gemma-3-4b-it
3. Install dependencies
uv sync
4. Run the app
uv run streamlit run app.py
📁 Project Structure
src/
├── openrouter_extractor.py   # LLM interaction + prompting
├── semicolon_parser.py       # Parses LLM output
├── matcher.py                # Matching + ranking logic
├── nutrition.py              # Nutrition computation
├── dataset_growth.py         # Logs unmatched items
├── data_prep.py              # Dataset preparation
├── pipeline.py               # Full pipeline

data/
├── USDA_V2_merged.json       # Main dataset
├── proposed_rows.csv         # Dataset growth queue
⚠️ Current Limitations
Some complex dishes may:
be partially decomposed
or not match the dataset
Matching depends on dataset coverage
Lightweight models are less consistent than larger models
🔮 Next Steps
Improve ranking system (multi-signal scoring)
Build a review UI for proposed dataset rows
Improve dish decomposition reliability
Expand dataset automatically