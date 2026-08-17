"""Shared retrieval type. Every retriever returns a list of these."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RetrievedChunk:
    id: str
    text: str
    source: str
    page: int
    score: float
