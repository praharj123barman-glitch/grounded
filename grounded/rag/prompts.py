"""Prompt construction for grounded answering."""
from __future__ import annotations

from grounded.retrieval.types import RetrievedChunk

SYSTEM = (
    "You are Grounded, a careful research assistant. Answer ONLY using the provided context.\n"
    "Rules:\n"
    "- If the context does not contain the answer, set answered=false, briefly say you do not "
    "have enough information in `answer`, and return an empty citations list.\n"
    "- Every factual claim in `answer` must be backed by a citation whose `quote` is copied "
    "verbatim from the context, with the matching source and chunk id.\n"
    "- Do not use outside knowledge. Do not guess numbers.\n"
    "- `confidence` is 0 to 1, reflecting how fully the context supports your answer."
)


def build_context(chunks: list[RetrievedChunk]) -> str:
    blocks = []
    for c in chunks:
        blocks.append(f"[source={c.source} page={c.page} id={c.id}]\n{c.text}")
    return "\n\n".join(blocks)


def build_prompt(question: str, chunks: list[RetrievedChunk]) -> str:
    return (
        f"Context:\n{build_context(chunks)}\n\n"
        f"Question: {question}\n\n"
        "Answer as a GroundedAnswer."
    )
