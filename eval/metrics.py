"""Retrieval metrics computed against the golden set.

Relevance proxy: a retrieved chunk counts as relevant if its text contains any of
the example's `relevant_substrings` (case-insensitive). This avoids pinning exact
chunk ids that shift when chunking changes, while still measuring whether the
right evidence was fetched. Unanswerable examples (no relevant_substrings) return
None so they are excluded from retrieval averages.
"""
from __future__ import annotations


def _chunk_is_relevant(text: str, relevant_substrings: list[str]) -> bool:
    t = text.lower()
    return any(s.lower() in t for s in relevant_substrings)


def recall_at_k(chunk_texts: list[str], relevant_substrings: list[str], k: int) -> float | None:
    if not relevant_substrings:
        return None
    topk = chunk_texts[:k]
    found = sum(
        1 for s in relevant_substrings if any(s.lower() in ct.lower() for ct in topk)
    )
    return found / len(relevant_substrings)


def precision_at_k(chunk_texts: list[str], relevant_substrings: list[str], k: int) -> float | None:
    if not relevant_substrings:
        return None
    topk = chunk_texts[:k]
    if not topk:
        return 0.0
    relevant = sum(1 for ct in topk if _chunk_is_relevant(ct, relevant_substrings))
    return relevant / len(topk)


def mrr(chunk_texts: list[str], relevant_substrings: list[str]) -> float | None:
    if not relevant_substrings:
        return None
    for i, ct in enumerate(chunk_texts):
        if _chunk_is_relevant(ct, relevant_substrings):
            return 1.0 / (i + 1)
    return 0.0


def hit_at_k(chunk_texts: list[str], relevant_substrings: list[str], k: int) -> float | None:
    if not relevant_substrings:
        return None
    return 1.0 if any(_chunk_is_relevant(ct, relevant_substrings) for ct in chunk_texts[:k]) else 0.0


def mean(values: list[float | None]) -> float:
    nums = [v for v in values if v is not None]
    return sum(nums) / len(nums) if nums else 0.0
