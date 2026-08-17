"""The RAG pipeline: optional query rewrite, retrieve, optional rerank, answer.

One class covers v1 (plain retrieve then answer) and v2 (rewrite plus rerank),
toggled by which components are passed in, so the eval harness can ablate each
piece by constructing the pipeline differently.
"""
from __future__ import annotations

from grounded.config import settings
from grounded.llm.schemas import GroundedAnswer
from grounded.rag.answerer import Answerer
from grounded.retrieval.types import RetrievedChunk


class RAGPipeline:
    def __init__(self, retriever, answerer: Answerer, rewriter=None, reranker=None, k=None) -> None:
        self.retriever = retriever
        self.answerer = answerer
        self.rewriter = rewriter
        self.reranker = reranker
        self.k = k or settings.top_k

    def retrieve(self, question: str, k: int | None = None) -> list[RetrievedChunk]:
        k = k or self.k
        queries = self.rewriter.rewrite(question) if self.rewriter else [question]
        pool: dict[str, RetrievedChunk] = {}
        for q in queries:
            for c in self.retriever.retrieve(q, k=k):
                # keep the best score seen for a chunk across sub-queries
                if c.id not in pool or c.score > pool[c.id].score:
                    pool[c.id] = c
        chunks = list(pool.values())
        if self.reranker:
            chunks = self.reranker.rerank(question, chunks, top_k=k)
        else:
            chunks = sorted(chunks, key=lambda c: -c.score)[:k]
        return chunks

    def run(self, question: str, k: int | None = None) -> tuple[GroundedAnswer, list[RetrievedChunk]]:
        chunks = self.retrieve(question, k=k)
        return self.answerer.answer(question, chunks), chunks


def build_default_pipeline() -> RAGPipeline:
    """Wire the production pipeline from config. Requires GOOGLE_API_KEY."""
    from grounded.ingestion.embeddings import Embedder
    from grounded.llm.client import LLMClient

    embedder = Embedder()
    client = LLMClient()

    if settings.use_hybrid:
        from grounded.retrieval.hybrid import HybridRetriever
        retriever = HybridRetriever(embedder)
    else:
        from grounded.retrieval.dense import DenseRetriever
        retriever = DenseRetriever(embedder)

    rewriter = None
    if settings.use_query_rewrite:
        from grounded.retrieval.query_rewrite import QueryRewriter
        rewriter = QueryRewriter(client)

    reranker = None
    if settings.use_rerank:
        from grounded.retrieval.rerank import LLMReranker
        reranker = LLMReranker(client)

    return RAGPipeline(retriever, Answerer(client), rewriter=rewriter, reranker=reranker)
