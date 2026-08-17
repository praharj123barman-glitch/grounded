"""Dense retriever over the Chroma collection built in Phase 1."""
from __future__ import annotations

import chromadb

from grounded.config import settings
from grounded.retrieval.types import RetrievedChunk


class DenseRetriever:
    def __init__(self, embedder, vector_dir=None, collection=None, top_k=None) -> None:
        self.embedder = embedder
        self.top_k = top_k or settings.top_k
        client = chromadb.PersistentClient(path=str(vector_dir or settings.vector_dir))
        self.col = client.get_or_create_collection(
            name=collection or settings.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def retrieve(self, query: str, k: int | None = None) -> list[RetrievedChunk]:
        k = k or self.top_k
        if self.col.count() == 0:
            return []
        qv = self.embedder.embed_query(query)
        res = self.col.query(
            query_embeddings=[qv],
            n_results=min(k, self.col.count()),
            include=["documents", "metadatas", "distances"],
        )
        ids, docs = res["ids"][0], res["documents"][0]
        metas, dists = res["metadatas"][0], res["distances"][0]
        out: list[RetrievedChunk] = []
        for i in range(len(ids)):
            out.append(
                RetrievedChunk(
                    id=ids[i],
                    text=docs[i],
                    source=(metas[i] or {}).get("source", ""),
                    page=int((metas[i] or {}).get("page", 0)),
                    score=1.0 - float(dists[i]),   # cosine distance -> similarity
                )
            )
        return out
