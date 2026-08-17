"""Hybrid retriever: dense (embeddings) + sparse (BM25 keyword), fused with RRF.

Dense catches meaning, BM25 catches exact terms and numbers (important for
financial reports). RRF merges the two rankings without needing to normalise
their very different score scales.
"""
from __future__ import annotations

import re

from rank_bm25 import BM25Okapi

from grounded.config import settings
from grounded.retrieval.dense import DenseRetriever
from grounded.retrieval.fusion import reciprocal_rank_fusion
from grounded.retrieval.types import RetrievedChunk


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


class HybridRetriever:
    def __init__(self, embedder, vector_dir=None, collection=None, top_k=None) -> None:
        self.dense = DenseRetriever(embedder, vector_dir, collection, top_k)
        self.top_k = top_k or settings.top_k
        data = self.dense.col.get(include=["documents", "metadatas"])
        self.ids: list[str] = data["ids"]
        self.docs: list[str] = data["documents"]
        self.metas: list[dict] = data["metadatas"]
        self.bm25 = BM25Okapi([_tokens(d) for d in self.docs]) if self.docs else None

    def _chunk_from_store(self, i: int, score: float) -> RetrievedChunk:
        meta = self.metas[i] or {}
        return RetrievedChunk(
            id=self.ids[i], text=self.docs[i],
            source=meta.get("source", ""), page=int(meta.get("page", 0)), score=score,
        )

    def retrieve(self, query: str, k: int | None = None) -> list[RetrievedChunk]:
        k = k or self.top_k
        if not self.ids:
            return []
        pool = max(k * 3, 10)

        dense_hits = self.dense.retrieve(query, k=pool)
        dense_ranking = [c.id for c in dense_hits]
        idmap = {c.id: c for c in dense_hits}

        bm25_ranking: list[str] = []
        if self.bm25 is not None:
            scores = self.bm25.get_scores(_tokens(query))
            order = sorted(range(len(scores)), key=lambda i: -scores[i])[:pool]
            bm25_ranking = [self.ids[i] for i in order]
            for i in order:
                idmap.setdefault(self.ids[i], self._chunk_from_store(i, 0.0))

        fused = reciprocal_rank_fusion([dense_ranking, bm25_ranking])
        out: list[RetrievedChunk] = []
        for cid, score in fused[:k]:
            chunk = idmap.get(cid)
            if chunk is not None:
                chunk.score = score
                out.append(chunk)
        return out
