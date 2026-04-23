from __future__ import annotations

import re
from typing import Any

from typhon.utils.text import significant_terms


def _normalize_text(text: str) -> str:
    return " ".join(text.lower().split())


def split_sentences(text: str) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [sentence.strip() for sentence in sentences if sentence.strip()]


def rank_sentences(question: str, retrieval_texts: list[str], *, top_k: int = 2) -> list[str]:
    question_terms = set(significant_terms(question))
    candidates: list[tuple[float, str]] = []
    for text in retrieval_texts:
        for sentence in split_sentences(text):
            sentence_terms = set(significant_terms(sentence))
            overlap = len(question_terms.intersection(sentence_terms))
            numeric_signal = 1.0 if re.search(r"\d", sentence) else 0.0
            score = overlap + 0.25 * numeric_signal + 0.01 * len(sentence_terms)
            candidates.append((score, sentence))
    ranked = sorted(candidates, key=lambda item: item[0], reverse=True)
    selected: list[str] = []
    seen: set[str] = set()
    for _, sentence in ranked:
        normalized = _normalize_text(sentence)
        if normalized in seen:
            continue
        seen.add(normalized)
        selected.append(sentence)
        if len(selected) >= top_k:
            break
    return selected


def predict_answer(
    question: str,
    retrieval_texts: list[str],
    *,
    expected_answer_type: str,
) -> str:
    if not retrieval_texts:
        return ""
    top_k = 1 if expected_answer_type == "classification" else 2
    ranked_sentences = rank_sentences(question, retrieval_texts, top_k=top_k)
    if ranked_sentences:
        return " ".join(ranked_sentences)
    return retrieval_texts[0]


def _score_against_reference(predicted_answer: str, reference_answer: str) -> dict[str, Any]:
    predicted_terms = significant_terms(predicted_answer)
    reference_terms = significant_terms(reference_answer)
    predicted_set = set(predicted_terms)
    reference_set = set(reference_terms)
    overlap = predicted_set.intersection(reference_set)

    precision = len(overlap) / len(predicted_set) if predicted_set else 0.0
    recall = len(overlap) / len(reference_set) if reference_set else 0.0
    f1 = 0.0 if precision + recall == 0 else (2 * precision * recall) / (precision + recall)

    return {
        "exact_match": _normalize_text(predicted_answer) == _normalize_text(reference_answer),
        "token_precision": round(precision, 4),
        "token_recall": round(recall, 4),
        "token_f1": round(f1, 4),
    }


def score_prediction(
    predicted_answer: str,
    reference_answer: str | None,
    *,
    reference_answers: list[str] | tuple[str, ...] | None = None,
) -> dict[str, Any]:
    references = [item for item in (reference_answers or []) if item]
    if reference_answer and reference_answer not in references:
        references.insert(0, reference_answer)

    if not references:
        return {
            "has_reference": False,
            "exact_match": None,
            "token_precision": None,
            "token_recall": None,
            "token_f1": None,
            "matched_reference_answer": None,
        }

    scored = [
        {
            "reference_answer": candidate,
            **_score_against_reference(predicted_answer, candidate),
        }
        for candidate in references
    ]
    scored.sort(
        key=lambda item: (
            1 if item["exact_match"] else 0,
            float(item["token_f1"]),
            float(item["token_recall"]),
            float(item["token_precision"]),
        ),
        reverse=True,
    )
    best = scored[0]

    return {
        "has_reference": True,
        "exact_match": best["exact_match"],
        "token_precision": best["token_precision"],
        "token_recall": best["token_recall"],
        "token_f1": best["token_f1"],
        "matched_reference_answer": best["reference_answer"],
    }


def build_prediction_block(
    *,
    question: str,
    retrieval_texts: list[str],
    expected_answer_type: str,
    reference_answer: str | None,
    reference_answers: list[str] | tuple[str, ...] | None = None,
) -> dict[str, Any]:
    predicted_answer = predict_answer(
        question=question,
        retrieval_texts=retrieval_texts,
        expected_answer_type=expected_answer_type,
    )
    return {
        "predicted_answer": predicted_answer,
        "reference_answer": reference_answer,
        "reference_answers": list(reference_answers or []),
        "metrics": score_prediction(
            predicted_answer,
            reference_answer,
            reference_answers=reference_answers,
        ),
    }
