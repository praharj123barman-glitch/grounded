"""Configurable chunking. The strategy and size are config knobs (not constants)
specifically so Phase 4 can ablate them and show which choice wins with numbers.
"""
from __future__ import annotations

from dataclasses import dataclass

from grounded.config import settings
from grounded.ingestion.loader import Document


@dataclass
class Chunk:
    id: str
    text: str
    source: str
    page: int | None
    chunk_index: int


def _splitter():
    from langchain_text_splitters import (
        CharacterTextSplitter,
        RecursiveCharacterTextSplitter,
    )

    if settings.chunk_strategy == "fixed":
        return CharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            separator="\n",
        )
    return RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )


def chunk_documents(docs: list[Document]) -> list[Chunk]:
    splitter = _splitter()
    chunks: list[Chunk] = []
    for d in docs:
        for piece in splitter.split_text(d.text):
            if not piece.strip():
                continue
            idx = len(chunks)
            cid = f"{d.source}::p{d.page or 0}::{idx}"
            chunks.append(
                Chunk(id=cid, text=piece, source=d.source, page=d.page, chunk_index=idx)
            )
    return chunks
