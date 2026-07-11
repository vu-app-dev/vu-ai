from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
import re


_STOPWORDS = {
    "a", "an", "and", "approach", "approaches", "are", "as", "at", "be",
    "between", "by", "can", "complex", "could", "do", "does", "for", "from",
    "give", "how", "if", "in", "is", "it", "me", "of", "on", "or", "please",
    "tell", "that", "the", "their", "this", "to", "what", "when", "where",
    "which", "with", "would", "you", "your",
}

_TOKEN_SYNONYMS = {
    "problem": "issue",
}


@dataclass(frozen=True)
class QuestionSimilarityScores:
    token_jaccard: float
    token_containment: float
    trigram_dice: float
    sequence_ratio: float
    combined: float


def normalize_question_text(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _stem_token(token: str) -> str:
    if len(token) > 5 and token.endswith("ing"):
        token = token[:-3]
        if len(token) > 3 and token[-1] == token[-2]:
            token = token[:-1]
    elif len(token) > 4 and token.endswith("ed"):
        token = token[:-2]
    elif len(token) > 4 and token.endswith("es"):
        token = token[:-2]
    elif len(token) > 3 and token.endswith("s"):
        token = token[:-1]
    return _TOKEN_SYNONYMS.get(token, token)


def _tokens(text: str) -> set[str]:
    return {
        _stem_token(token)
        for token in normalize_question_text(text).split()
        if token and token not in _STOPWORDS
    }


def _char_trigrams(text: str) -> set[str]:
    compact = normalize_question_text(text).replace(" ", "")
    if not compact:
        return set()
    if len(compact) <= 3:
        return {compact}
    return {compact[i : i + 3] for i in range(len(compact) - 2)}


def question_similarity(a: str, b: str) -> QuestionSimilarityScores:
    norm_a = normalize_question_text(a)
    norm_b = normalize_question_text(b)
    if not norm_a or not norm_b:
        return QuestionSimilarityScores(0.0, 0.0, 0.0, 0.0, 0.0)

    words_a = _tokens(norm_a)
    words_b = _tokens(norm_b)
    if words_a and words_b:
        intersection = len(words_a & words_b)
        union = len(words_a | words_b)
        token_jaccard = intersection / union if union else 0.0
        token_containment = intersection / min(len(words_a), len(words_b))
    else:
        token_jaccard = 0.0
        token_containment = 0.0

    grams_a = _char_trigrams(norm_a)
    grams_b = _char_trigrams(norm_b)
    if grams_a and grams_b:
        trigram_dice = (2 * len(grams_a & grams_b)) / (len(grams_a) + len(grams_b))
    else:
        trigram_dice = 0.0

    sequence_ratio = SequenceMatcher(None, norm_a, norm_b).ratio()
    combined = (
        (0.35 * token_containment)
        + (0.30 * trigram_dice)
        + (0.25 * sequence_ratio)
        + (0.10 * token_jaccard)
    )

    return QuestionSimilarityScores(
        token_jaccard=token_jaccard,
        token_containment=token_containment,
        trigram_dice=trigram_dice,
        sequence_ratio=sequence_ratio,
        combined=combined,
    )


def is_similar_question(a: str, b: str) -> bool:
    norm_a = normalize_question_text(a)
    norm_b = normalize_question_text(b)
    if not norm_a or not norm_b:
        return False
    if norm_a == norm_b:
        return True

    scores = question_similarity(norm_a, norm_b)
    return (
        scores.token_containment >= 0.82
        or scores.trigram_dice >= 0.78
        or scores.sequence_ratio >= 0.82
        or scores.combined >= 0.72
    )
