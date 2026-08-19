"""Rerankers: reorder retrieved candidates so the best evidence is on top.

Default is an LLM listwise reranker (no extra dependency). A cross-encoder
reranker is available for when sentence-transformers is installed.
"""
from __future__ import annotations

import re

from grounded.retrieval.types import RetrievedChunk


class LLMReranker:
    def __init__(self, client) -> None:
        self.client = client

    def rerank(self, query: str, chunks: list[RetrievedChunk], top_k: int | None = None):
        top_k = top_k or len(chunks)
        if len(chunks) <= 1:
            return chunks[:top_k]
        listing = "\n".join(f"{i}. {c.text[:220]}" for i, c in enumerate(chunks))
        system = (
            "Rank the passages from most to least relevant to the question. "
            "Return only the passage numbers, best first, comma separated."
        )
        try:
            out = self.client.complete(f"Question: {query}\n\nPassages:\n{listing}", system=system)
        except Exception:
            return chunks[:top_k]
        order = [int(x) for x in re.findall(r"\d+", out)]
        seen: set[int] = set()
        ranked: list[RetrievedChunk] = []
        for i in order:
            if 0 <= i < len(chunks) and i not in seen:
                seen.add(i)
                ranked.append(chunks[i])
        for i, c in enumerate(chunks):
            if i not in seen:
                ranked.append(c)
        return ranked[:top_k]


class CrossEncoderReranker:
    """Optional. Requires `pip install sentence-transformers` (pulls torch)."""

    def __init__(self, model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model)

    def rerank(self, query: str, chunks: list[RetrievedChunk], top_k: int | None = None):
        if not chunks:
            return chunks
        scores = self.model.predict([(query, c.text) for c in chunks])
        ranked = [c for _, c in sorted(zip(scores, chunks), key=lambda x: -x[0])]
        return ranked[: top_k or len(chunks)]
