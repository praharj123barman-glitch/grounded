"""Turn a question plus retrieved chunks into a typed, cited GroundedAnswer."""
from __future__ import annotations

from grounded.llm.schemas import GroundedAnswer
from grounded.rag.prompts import SYSTEM, build_prompt
from grounded.retrieval.types import RetrievedChunk


class Answerer:
    def __init__(self, client) -> None:
        self.client = client

    def answer(self, question: str, chunks: list[RetrievedChunk]) -> GroundedAnswer:
        # Guardrail: nothing retrieved means we refuse rather than hallucinate.
        if not chunks:
            return GroundedAnswer(
                answered=False,
                answer="I could not find relevant information in the documents.",
                citations=[],
                confidence=0.0,
            )
        prompt = build_prompt(question, chunks)
        return self.client.structured(prompt, GroundedAnswer, system=SYSTEM)
