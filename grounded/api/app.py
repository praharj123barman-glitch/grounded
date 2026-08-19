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


def _answer(question: str, k: int | None) -> dict:
    guard = scan(question)
    if guard["blocked"]:
        return {
            "answered": False,
            "answer": "This request was blocked by the input guardrail.",
            "citations": [],
            "confidence": 0.0,
            "blocked": True,
        }
    answer, chunks = _pipeline().run(question, k=k)
    return {**answer.model_dump(), "retrieved": len(chunks), "pii_flags": guard["pii"]}


@app.get("/")
def root():
    """Serve the landing UI, or fall back to the API docs if it is missing."""
    from pathlib import Path

    from fastapi.responses import FileResponse, RedirectResponse

    index = Path(__file__).resolve().parent / "static" / "index.html"
    if index.exists():
        return FileResponse(str(index))
    return RedirectResponse(url="/docs")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/ask")
def ask_get(q: str, k: int | None = None) -> dict:
    """Browser-friendly: /ask?q=What+was+FY2025+revenue"""
    return _answer(q, k)


@app.post("/ask")
def ask_post(req: AskRequest) -> dict:
    return _answer(req.question, req.k)
