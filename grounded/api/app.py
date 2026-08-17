"""FastAPI service for Grounded (Phase 7).

  uvicorn grounded.api.app:app --reload

The pipeline is built lazily on first /ask, so the app imports and /health works
without an API key (which is what CI and the test suite check).
"""
from __future__ import annotations

from functools import lru_cache

from fastapi import FastAPI
from pydantic import BaseModel

from grounded.guardrails.input_guard import scan

app = FastAPI(title="Grounded", version="0.1.0")


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
