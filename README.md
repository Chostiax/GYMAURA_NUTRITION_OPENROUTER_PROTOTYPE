# GymAura Nutrition PoC - OpenRouter Version

## Goal
This PoC tests a lightweight OpenRouter model as an alternative to OpenAI for food extraction.

## Pipeline
text
→ OpenRouter lightweight model
→ semicolon output
→ parser
→ matcher
→ nutrition

## Why semicolon output?
Smaller models are less reliable with strict JSON, so this version uses a simpler format:

banana;120;g
coffee;1;cup
egg;2;

## Setup

### 1. Install uv
### 2. Create .env
### 3. Run:
uv sync
uv run python main.py

Or:
uv run streamlit run app.py