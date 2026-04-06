from __future__ import annotations

from rapidfuzz import fuzz, process

from src.data_prep import normalize_food_text, CATEGORY_ID_TO_NAME


ALIASES = {
    "eggs": "egg",
    "fried eggs": "egg",
    "boiled eggs": "egg",
    "scrambled eggs": "egg",
    "white rice": "rice",
    "brown rice": "rice",
    "basmati rice": "rice",
    "jasmine rice": "rice",
    "cooked white rice": "rice",
    "black coffee": "coffee",
    "latte": "coffee",
    "cappuccino": "coffee",
    "almonds": "almond",
    "dates": "date",
    "cereals": "cereal",
    "toast": "bread",
    "fries": "fry",
    "greek yogurt": "yogurt",
    "yoghurt": "yogurt",
    "berries": "berry",
    "vegetables": "vegetable",
    "iced tea": "tea",
    "tomato salsa": "salsa",
    "new york cheesecake": "cheesecake",
    "ribeye": "steak",
    "ground beef": "beef",
}

# Strong negative hints for generic foods that often match the wrong processed item
NEGATIVE_HINTS = {
    "chicken": {"spread", "salad", "luncheon", "sausage"},
    "beef": {"spread", "sausage"},
    "turkey": {"spread", "sausage"},
    "pizza": {"roll", "rolls", "bite", "bites", "snack", "frozen"},
    "bread": {"stuffing", "crumb", "cracker"},
    "tea": {"mix", "powder", "ready"},
}

GENERIC_SINGLE_TOKEN_FOODS = {
    "pizza",
    "burger",
    "tea",
    "coffee",
    "bread",
    "rice",
    "pasta",
    "chicken",
    "beef",
    "milk",
    "juice",
    "salad",
    "cake",
    "cookie",
    "sandwich",
}


def apply_alias(food_text: str) -> str:
    normalized = normalize_food_text(food_text, drop_generic_words=False)
    return ALIASES.get(normalized, normalized)


def token_overlap_score(query: str, candidate: str) -> float:
    query_tokens = set(query.split())
    candidate_tokens = set(candidate.split())
    if not query_tokens:
        return 0.0
    overlap = len(query_tokens.intersection(candidate_tokens))
    return overlap / len(query_tokens)


def rank_token_candidate(query: str, candidate: str) -> tuple:
    query_tokens = query.split()
    candidate_tokens = candidate.split()
    length_penalty = abs(len(candidate_tokens) - len(query_tokens))
    starts_with_query = 1 if candidate.startswith(query) else 0
    first_token_match = 1 if query_tokens and candidate_tokens and candidate_tokens[0] == query_tokens[0] else 0
    exact_length = 1 if len(candidate_tokens) == len(query_tokens) else 0
    return (starts_with_query, first_token_match, exact_length, -length_penalty)


def apply_penalties(query: str, candidate: str, base_score: float) -> float:
    lowered = candidate.lower()
    query_tokens = query.split()
    candidate_tokens = lowered.split()

    # Strong penalty for clearly wrong processed variants
    for key, banned_words in NEGATIVE_HINTS.items():
        if key in query and any(word in candidate_tokens for word in banned_words):
            base_score -= 35.0

    # Important: generic one-word foods should not match long weird variants too easily
    if len(query_tokens) == 1 and query_tokens[0] in GENERIC_SINGLE_TOKEN_FOODS:
        extra_tokens = [tok for tok in candidate_tokens if tok != query_tokens[0]]
        if len(extra_tokens) >= 2:
            base_score -= 25.0

    return base_score


def is_safe_generic_match(query: str, candidate: str) -> bool:
    """
    Prevent generic one-word queries like 'pizza' from matching
    things like 'pizza rolls, frozen, unprepared'.
    """
    query_tokens = query.split()
    candidate_tokens = candidate.split()

    if len(query_tokens) != 1:
        return True

    query_token = query_tokens[0]

    if query_token not in GENERIC_SINGLE_TOKEN_FOODS:
        return True

    # Exact one-token match is always safe
    if candidate == query:
        return True

    # If candidate contains strong bad modifiers, reject
    banned = NEGATIVE_HINTS.get(query_token, set())
    if any(tok in banned for tok in candidate_tokens):
        return False

    # If candidate is much longer than query, reject for generic foods
    extra_tokens = [tok for tok in candidate_tokens if tok != query_token]
    if len(extra_tokens) >= 2:
        return False

    return True


def _build_result(row, query: str, matched: bool, match_type: str | None, score: float):
    return {
        "matched": matched,
        "match_type": match_type,
        "score": round(score, 1),
        "row_index": int(row.name) if row is not None else None,
        "description": row["english_description"] if row is not None else None,
        "normalized_query": query,
        "matched_category": row["food_category"] if row is not None else None,
        "matched_category_id": row["category_id"] if row is not None else None,
    }


def _match_in_subset(query: str, subset, fuzzy_threshold: int = 84, strict_generic: bool = False) -> dict:
    query_tokens = set(query.split())

    # 1) Exact clean match
    exact_matches = subset[subset["clean_description"] == query]
    if len(exact_matches) > 0:
        row = exact_matches.iloc[0]
        return _build_result(row, query, True, "exact", 100.0)

    # 2) Startswith match with strong length control
    startswith_candidates = subset[subset["clean_description"].str.startswith(query + " ", na=False)]
    if len(startswith_candidates) > 0:
        safe_rows = []
        for _, row in startswith_candidates.iterrows():
            candidate = row["clean_description"]
            if not strict_generic or is_safe_generic_match(query, candidate):
                safe_rows.append(row)

        if safe_rows:
            safe_rows = sorted(
                safe_rows,
                key=lambda r: rank_token_candidate(query, r["clean_description"]),
                reverse=True,
            )
            row = safe_rows[0]
            return _build_result(row, query, True, "startswith", 96.0)

    # 3) Token overlap
    best_idx = None
    best_score = -1.0
    best_rank = (-1, -1, -1, float("-inf"))

    for idx, candidate in subset["clean_description"].items():
        candidate_tokens = set(candidate.split())
        if query_tokens and query_tokens.issubset(candidate_tokens):
            if strict_generic and not is_safe_generic_match(query, candidate):
                continue

            overlap_score = token_overlap_score(query, candidate)
            rank = rank_token_candidate(query, candidate)
            scored = apply_penalties(query, candidate, overlap_score * 100.0)

            if scored > best_score or (scored == best_score and rank > best_rank):
                best_score = scored
                best_rank = rank
                best_idx = idx

    if best_idx is not None and best_score >= 85:
        row = subset.loc[best_idx]
        return _build_result(row, query, True, "token_overlap", best_score)

    # 4) Fuzzy fallback — but be strict for generic single-word foods
    choices = subset["clean_description"].tolist()
    best_match = process.extractOne(query, choices, scorer=fuzz.token_sort_ratio)

    if best_match is not None:
        matched_text, score, match_pos = best_match
        score = apply_penalties(query, matched_text, float(score))

        if strict_generic and not is_safe_generic_match(query, matched_text):
            score = 0.0

        min_threshold = 92 if strict_generic else fuzzy_threshold

        if score >= min_threshold:
            row = subset.iloc[match_pos]
            return _build_result(row, query, True, "fuzzy", score)

    return {
        "matched": False,
        "match_type": None,
        "score": 0.0,
        "row_index": None,
        "description": None,
        "normalized_query": query,
        "matched_category": None,
        "matched_category_id": None,
    }


def match_food_to_dataset(
    food_text: str,
    dataset,
    category_id: int | None = None,
    fuzzy_threshold: int = 84,
) -> dict:
    query = apply_alias(food_text)
    query_tokens = query.split()
    strict_generic = len(query_tokens) == 1 and query_tokens[0] in GENERIC_SINGLE_TOKEN_FOODS

    # 1) Category-first search
    if category_id is not None and category_id in CATEGORY_ID_TO_NAME:
        category_subset = dataset[dataset["category_id"] == category_id]

        if len(category_subset) > 0:
            category_match = _match_in_subset(
                query,
                category_subset,
                fuzzy_threshold=fuzzy_threshold,
                strict_generic=strict_generic,
            )
            if category_match["matched"]:
                category_match["search_scope"] = "category_first"
                return category_match

    # 2) Global fallback
    global_match = _match_in_subset(
        query,
        dataset,
        fuzzy_threshold=fuzzy_threshold,
        strict_generic=strict_generic,
    )

    # 3) If this is a generic one-item food and global match changes the category,
    # be conservative and reject the match
    if global_match["matched"] and strict_generic and category_id is not None:
        matched_category_id = global_match.get("matched_category_id")
        if matched_category_id is not None and matched_category_id != category_id:
            return {
                "matched": False,
                "match_type": None,
                "score": 0.0,
                "row_index": None,
                "description": None,
                "normalized_query": query,
                "matched_category": None,
                "matched_category_id": None,
                "search_scope": "global_rejected_category_mismatch",
            }

    global_match["search_scope"] = "global_fallback"
    return global_match