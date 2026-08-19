"""Semantic cache: reuse a past answer when a new question is
near-identical in meaning, cutting cost and latency. Embeddings are assumed
normalised, so cosine similarity is a dot product.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SemanticCache:
    embedder: Any
    threshold: float = 0.92
    _items: list = field(default_factory=list)  # (vector, question, value)

    @staticmethod
    def _cosine(a: list[float], b: list[float]) -> float:
        return sum(x * y for x, y in zip(a, b))

    def get(self, question: str):
        if not self._items:
            return None
        qv = self.embedder.embed_query(question)
        best_val, best_sim = None, -1.0
        for vec, _q, val in self._items:
            sim = self._cosine(qv, vec)
            if sim > best_sim:
                best_sim, best_val = sim, val
        return best_val if best_sim >= self.threshold else None

    def put(self, question: str, value) -> None:
        self._items.append((self.embedder.embed_query(question), question, value))
