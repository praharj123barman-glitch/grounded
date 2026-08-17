"""Persist chunks and their embeddings into a local, on-disk Chroma collection.

upsert (not add) so re-running ingestion is idempotent: the same chunk id
overwrites rather than duplicating.
"""
from __future__ import annotations

import chromadb

from grounded.config import settings
from grounded.ingestion.chunker import Chunk


def build_index(chunks: list[Chunk], embeddings: list[list[float]]) -> int:
    settings.vector_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(settings.vector_dir))
    col = client.get_or_create_collection(
        name=settings.collection_name,
        metadata={"hnsw:space": "cosine"},
    )
    col.upsert(
        ids=[c.id for c in chunks],
        documents=[c.text for c in chunks],
        embeddings=embeddings,
        metadatas=[
            {"source": c.source, "page": c.page or 0, "chunk_index": c.chunk_index}
            for c in chunks
        ],
    )
    return col.count()
