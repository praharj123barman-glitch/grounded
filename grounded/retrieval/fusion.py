"""Reciprocal Rank Fusion: combine several ranked id lists into one.

score(id) = sum over lists of 1 / (k + rank), rank starting at 1. A robust,
score-free way to merge dense and keyword rankings (used by the hybrid retriever).
"""
from __future__ import annotations


def reciprocal_rank_fusion(rankings: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, cid in enumerate(ranking):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: -x[1])
