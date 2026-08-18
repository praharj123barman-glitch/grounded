"""FastAPI service for Grounded (Phase 7).

  uvicorn grounded.api.app:app --reload

The pipeline is built lazily on first /ask, so the app imports and /health works
without an API key (which is what CI and the test suite check).
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from functools import lru_cache

from fastapi import FastAPI
from pydantic import BaseModel

from grounded.config import settings
from grounded.guardrails.input_guard import scan


def _bootstrap_index() -> None:
    """On deploy, build the index from the sample corpus if it is empty."""
    import chromadb

    client = chromadb.PersistentClient(path=str(settings.vector_dir))
    col = client.get_or_create_collection(
        settings.collection_name, metadata={"hnsw:space": "cosine"}
    )
    if col.count() > 0 or not settings.google_api_key:
        return
    from grounded.ingestion.chunker import chunk_documents
    from grounded.ingestion.embeddings import Embedder
    from grounded.ingestion.indexer import build_index
    from grounded.ingestion.loader import load_dir

    chunks = chunk_documents(load_dir(settings.sample_dir))
    emb = Embedder()
    build_index(chunks, emb.embed_texts([c.text for c in chunks]))


@asynccontextmanager
async def lifespan(app: FastAPI):
    if settings.bootstrap_on_start:
        try:
            _bootstrap_index()
        except Exception:
            pass
    yield


app = FastAPI(title="Grounded", version="0.1.0", lifespan=lifespan)


class AskRequest(BaseModel):
    question: str
    k: int | None = None


@lru_cache(maxsize=1)
def _pipeline():
    from grounded.rag.pipeline import build_default_pipeline

    return build_default_pipeline()


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/ask")
def ask(req: AskRequest) -> dict:
    guard = scan(req.question)
    if guard["blocked"]:
        return {
            "answered": False,
            "answer": "This request was blocked by the input guardrail.",
            "citations": [],
            "confidence": 0.0,
            "blocked": True,
        }
    answer, chunks = _pipeline().run(req.question, k=req.k)
    return {**answer.model_dump(), "retrieved": len(chunks), "pii_flags": guard["pii"]}
