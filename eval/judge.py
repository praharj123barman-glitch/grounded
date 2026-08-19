"""LLM-as-judge for generation quality.

Scores faithfulness (is every claim supported by the retrieved context) and
answer relevance (does the answer address the question). Needs a live LLM client.
"""
from __future__ import annotations

from pydantic import BaseModel, Field

_SYSTEM = (
    "You are a strict evaluator of a retrieval-augmented answer. Given the question, "
    "the retrieved context, and the answer, rate two things from 0 to 1: "
    "faithfulness (every claim in the answer is supported by the context; punish any "
    "unsupported or invented claim) and answer_relevance (the answer actually addresses "
    "the question). Be harsh."
)


class JudgeScores(BaseModel):
    faithfulness: float = Field(ge=0.0, le=1.0)
    answer_relevance: float = Field(ge=0.0, le=1.0)
    reason: str = ""


class Judge:
    def __init__(self, client) -> None:
        self.client = client

    def score(self, question: str, answer: str, context: str) -> JudgeScores:
        prompt = f"Question: {question}\n\nContext:\n{context}\n\nAnswer:\n{answer}"
        return self.client.structured(prompt, JudgeScores, system=_SYSTEM)
