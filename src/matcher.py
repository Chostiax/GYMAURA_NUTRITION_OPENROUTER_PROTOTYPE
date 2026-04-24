from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from rapidfuzz import fuzz

from src.data_prep import CATEGORY_ID_TO_NAME, normalize_food_text


ALIASES = {
    "eggs": "egg",
    "fried eggs": "egg",
    "boiled eggs": "egg",
    "scrambled eggs": "egg",
    "yoghurt": "yogurt",
    "greek yoghurt": "greek yogurt",
    "iced tea": "tea",
    "fries": "fry",
    "berries": "berry",
    "vegetables": "vegetable",
    "cereals": "cereal",
    "apples": "apple",
    "bananas": "banana",
}


DERIVATIVE_FORM_TOKENS = {
    "dressing",
    "sauce",
    "spread",
    "dip",
    "seasoning",
    "powder",
    "flour",
    "extract",
    "syrup",
    "concentrate",
    "mix",
    "mixes",
    "topping",
    "stuffing",
    "crumb",
    "crumbs",
    "filling",
    "frosting",
}

PRODUCT_FORM_TOKENS = {
    "frozen",
    "instant",
    "canned",
    "commercial",
    "prepared",
    "ready",
}

SNACK_DERIVATIVE_TOKENS = {
    "chip",
    "chips",
    "cracker",
    "crackers",
    "bar",
    "bars",
    "bite",
    "bites",
    "roll",
    "rolls",
    "snack",
    "snacks",
}

GENERIC_HEAD_FOODS = {
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
    "wrap",
    "soup",
    "apple",
    "banana",
    "egg",
}

SEVERE_CATEGORY_CONFLICTS = {
    15: {10, 22, 23},
    19: {10, 22, 23},
    5: {4, 17, 18, 20},
}


@dataclass
class CandidateScore:
    idx: int
    final_score: float
    exact: bool
    startswith: bool
    word_boundary_phrase: bool
    token_coverage: float
    token_precision: float
    fuzzy_sort: float
    fuzzy_set: float
    category_bonus: float
    semantic_penalty: float
    length_penalty: float


def apply_alias(food_text: str) -> str:
    normalized = normalize_food_text(food_text, drop_generic_words=False)
    return ALIASES.get(normalized, normalized)


def _tokenize(text: str) -> list[str]:
    normalized = normalize_food_text(text, drop_generic_words=False)
    return [tok for tok in normalized.split() if tok]


def _contains_phrase_as_word_boundary(query: str, candidate: str) -> bool:
    query_tokens = query.split()
    candidate_tokens = candidate.split()

    if not query_tokens or len(query_tokens) > len(candidate_tokens):
        return False

    window = len(query_tokens)
    for i in range(len(candidate_tokens) - window + 1):
        if candidate_tokens[i : i + window] == query_tokens:
            return True

    return False


def _token_coverage(query_tokens: set[str], candidate_tokens: set[str]) -> float:
    if not query_tokens:
        return 0.0
    return len(query_tokens & candidate_tokens) / len(query_tokens)


def _token_precision(query_tokens: set[str], candidate_tokens: set[str]) -> float:
    if not candidate_tokens:
        return 0.0
    return len(query_tokens & candidate_tokens) / len(candidate_tokens)


def _semantic_form_penalty(query_tokens: set[str], candidate_tokens: set[str]) -> float:
    penalty = 0.0

    derivative_overlap = (DERIVATIVE_FORM_TOKENS & candidate_tokens) - query_tokens
    product_overlap = (PRODUCT_FORM_TOKENS & candidate_tokens) - query_tokens
    snack_overlap = (SNACK_DERIVATIVE_TOKENS & candidate_tokens) - query_tokens

    if derivative_overlap:
        penalty += 22.0
    if product_overlap:
        penalty += 8.0
    if snack_overlap:
        penalty += 14.0

    if len(query_tokens) == 1:
        only = next(iter(query_tokens))
        if only in GENERIC_HEAD_FOODS:
            extra_tokens = candidate_tokens - query_tokens
            if len(extra_tokens) >= 2:
                penalty += 10.0
            if len(extra_tokens) >= 4:
                penalty += 8.0

    return penalty


def _category_bonus_or_penalty(query_category_id: int | None, candidate_category_id: int | None) -> float:
    if query_category_id is None or candidate_category_id is None:
        return 0.0

    if query_category_id == candidate_category_id:
        return 8.0

    severe_conflicts = SEVERE_CATEGORY_CONFLICTS.get(query_category_id, set())
    if candidate_category_id in severe_conflicts:
        return -18.0

    return -4.0


def _length_penalty(query_tokens: list[str], candidate_tokens: list[str]) -> float:
    diff = abs(len(candidate_tokens) - len(query_tokens))

    if diff == 0:
        return 0.0
    if diff == 1:
        return 2.0
    if diff == 2:
        return 5.0

    return 9.0


def _score_candidate(
    query: str,
    query_tokens_list: list[str],
    query_category_id: int | None,
    candidate: str,
    candidate_category_id: int | None,
    idx: int,
) -> CandidateScore:
    query_tokens = set(query_tokens_list)
    candidate_tokens_list = candidate.split()
    candidate_tokens = set(candidate_tokens_list)

    exact = candidate == query
    startswith = candidate.startswith(query + " ")
    word_boundary_phrase = _contains_phrase_as_word_boundary(query, candidate)

    token_coverage = _token_coverage(query_tokens, candidate_tokens)
    token_precision = _token_precision(query_tokens, candidate_tokens)

    fuzzy_sort = float(fuzz.token_sort_ratio(query, candidate))
    fuzzy_set = float(fuzz.token_set_ratio(query, candidate))

    category_term = _category_bonus_or_penalty(query_category_id, candidate_category_id)
    semantic_penalty = _semantic_form_penalty(query_tokens, candidate_tokens)
    length_term = _length_penalty(query_tokens_list, candidate_tokens_list)

    if exact:
        score = 100.0 + category_term
    else:
        score = 0.0
        score += token_coverage * 42.0
        score += token_precision * 14.0
        score += fuzzy_sort * 0.22
        score += fuzzy_set * 0.16

        if word_boundary_phrase:
            score += 10.0
        elif startswith:
            score += 6.0

        score += category_term
        score -= semantic_penalty
        score -= length_term

    return CandidateScore(
        idx=idx,
        final_score=round(score, 4),
        exact=exact,
        startswith=startswith,
        word_boundary_phrase=word_boundary_phrase,
        token_coverage=token_coverage,
        token_precision=token_precision,
        fuzzy_sort=fuzzy_sort,
        fuzzy_set=fuzzy_set,
        category_bonus=category_term,
        semantic_penalty=semantic_penalty,
        length_penalty=length_term,
    )


def _accept_match(
    best: CandidateScore,
    second_best: CandidateScore | None,
    query_tokens: list[str],
) -> tuple[bool, str | None]:
    if best.exact:
        return True, None

    # Important fix:
    # If the score is very high and there is no severe derivative/product penalty,
    # accept it even if the top-2 margin is small.
    if best.final_score >= 92.0 and best.semantic_penalty < 22.0:
        return True, None

    if best.final_score < 72.0:
        return False, "rejected_low_score"

    if best.semantic_penalty >= 22.0 and best.final_score < 82.0:
        return False, "rejected_derivative_form"

    if best.token_coverage < 1.0 and len(query_tokens) >= 2:
        return False, "rejected_missing_query_tokens"

    if second_best is not None:
        margin = best.final_score - second_best.final_score

        if best.final_score < 82.0 and margin < 6.0:
            return False, "rejected_ambiguous_top2"

        if best.final_score >= 82.0 and margin < 3.0:
            return False, "rejected_small_margin"

    return True, None


def _build_result(
    row,
    query: str,
    matched: bool,
    match_type: str | None,
    score: float,
    search_scope: str | None,
    reject_reason: str | None = None,
):
    return {
        "matched": matched,
        "match_type": match_type,
        "score": round(score, 1),
        "row_index": int(row.name) if row is not None else None,
        "description": row["english_description"] if row is not None else None,
        "normalized_query": query,
        "matched_category": row["food_category"] if row is not None else None,
        "matched_category_id": row["category_id"] if row is not None else None,
        "search_scope": search_scope,
        "reject_reason": reject_reason,
    }


def _iter_scored_candidates(query: str, query_category_id: int | None, subset) -> Iterable[CandidateScore]:
    query_tokens = _tokenize(query)

    for idx, candidate in subset["clean_description"].items():
        candidate_category_id = subset.loc[idx, "category_id"]

        yield _score_candidate(
            query=query,
            query_tokens_list=query_tokens,
            query_category_id=query_category_id,
            candidate=candidate,
            candidate_category_id=candidate_category_id,
            idx=idx,
        )


def _match_in_subset(query: str, subset, query_category_id: int | None, scope_name: str) -> dict:
    if len(subset) == 0:
        return _build_result(
            row=None,
            query=query,
            matched=False,
            match_type=None,
            score=0.0,
            search_scope=scope_name,
            reject_reason="empty_subset",
        )

    query_tokens = _tokenize(query)

    scored = sorted(
        _iter_scored_candidates(
            query=query,
            query_category_id=query_category_id,
            subset=subset,
        ),
        key=lambda x: x.final_score,
        reverse=True,
    )

    if not scored:
        return _build_result(
            row=None,
            query=query,
            matched=False,
            match_type=None,
            score=0.0,
            search_scope=scope_name,
            reject_reason="no_candidates",
        )

    best = scored[0]
    second_best = scored[1] if len(scored) > 1 else None

    accept, reject_reason = _accept_match(best, second_best, query_tokens)

    if not accept:
        return _build_result(
            row=None,
            query=query,
            matched=False,
            match_type=None,
            score=best.final_score,
            search_scope=scope_name,
            reject_reason=reject_reason,
        )

    row = subset.loc[best.idx]

    if best.exact:
        match_type = "exact"
    elif best.word_boundary_phrase:
        match_type = "phrase"
    elif best.startswith:
        match_type = "startswith"
    elif best.token_coverage == 1.0:
        match_type = "token_cover"
    else:
        match_type = "fuzzy_ranked"

    return _build_result(
        row=row,
        query=query,
        matched=True,
        match_type=match_type,
        score=best.final_score,
        search_scope=scope_name,
        reject_reason=None,
    )


def match_food_to_dataset(
    food_text: str,
    dataset,
    category_id: int | None = None,
    fuzzy_threshold: int = 84,
) -> dict:
    del fuzzy_threshold

    query = apply_alias(food_text)

    if category_id is not None and category_id in CATEGORY_ID_TO_NAME:
        category_subset = dataset[dataset["category_id"] == category_id]

        if len(category_subset) > 0:
            category_match = _match_in_subset(
                query=query,
                subset=category_subset,
                query_category_id=category_id,
                scope_name="category_first",
            )

            if category_match["matched"]:
                return category_match

    return _match_in_subset(
        query=query,
        subset=dataset,
        query_category_id=category_id,
        scope_name="global_fallback",
    )