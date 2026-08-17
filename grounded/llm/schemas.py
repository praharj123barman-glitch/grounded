"""Typed answer schema. Every answer Grounded returns is one of these, so the
citations, confidence, and refusal flag are guaranteed, not hoped for.
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class Citation(BaseModel):
    source: str = Field(description="File the evidence came from")
    chunk_id: str = Field(description="Id of the retrieved chunk that supports the claim")
    quote: str = Field(description="The exact sentence or sentences that support the answer")


class GroundedAnswer(BaseModel):
    answered: bool = Field(
        description="False when the retrieved context does not support an answer (a refusal)"
    )
    answer: str = Field(description="The answer, grounded only in the provided context")
    citations: list[Citation] = Field(default_factory=list)
    confidence: float = Field(
        ge=0.0, le=1.0, description="Self-estimated support from the context, 0 to 1"
    )
