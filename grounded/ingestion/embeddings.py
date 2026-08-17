"""Gemini embeddings wrapper. Same interface will back both ingestion and query
time, so documents and questions live in the same vector space.
"""
from __future__ import annotations

from grounded.config import settings


class Embedder:
    def __init__(self) -> None:
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        self._emb = GoogleGenerativeAIEmbeddings(
            model=settings.embedding_model,
            google_api_key=settings.google_api_key,
        )

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return self._emb.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._emb.embed_query(text)
