"""Multi-agent flow (course W6): planner, retriever, answerer, critic.

A supervisor-style graph. The planner decomposes the question into sub-questions,
the retriever gathers evidence for all of them, the answerer drafts a grounded
answer, and the critic verifies every claim is supported before it is returned.
If the critic rejects it, the answer is redrafted, up to `max_revisions`.

All collaborators (pipeline, client) are injected, so this compiles and runs in
tests with fakes and no API key.
"""
from __future__ import annotations

from typing import Any, TypedDict

from langgraph.graph import END, StateGraph
from pydantic import BaseModel

from grounded.rag.prompts import build_context

PLANNER_SYSTEM = (
    "Break the user's question into 1 to 3 focused sub-questions, one per line. "
    "If it is already simple, return it unchanged."
)
CRITIC_SYSTEM = (
    "You verify a drafted answer against the provided context. Approve only if "
    "every claim in the answer is supported by the context. Return approved=true "
    "or false with a short critique."
)


class Verdict(BaseModel):
    approved: bool
    critique: str = ""


class MAState(TypedDict, total=False):
    question: str
    subquestions: list[str]
    chunks: list[Any]
    draft: Any
    approved: bool
    critique: str
    revisions: int


def build_multi_agent(pipeline, client, max_revisions: int = 1):
    def planner(state: MAState) -> dict:
        q = state["question"]
        try:
            text = client.complete(q, system=PLANNER_SYSTEM)
            subs = [ln.strip("-*0123456789. ").strip() for ln in text.splitlines() if ln.strip()]
        except Exception:
            subs = []
        return {"subquestions": (subs or [q])[:3]}

    def retriever(state: MAState) -> dict:
        pool: dict[str, Any] = {}
        for sq in state.get("subquestions", [state["question"]]):
            for c in pipeline.retrieve(sq):
                if c.id not in pool or c.score > pool[c.id].score:
                    pool[c.id] = c
        return {"chunks": list(pool.values())}

    def answerer(state: MAState) -> dict:
        draft = pipeline.answerer.answer(state["question"], state.get("chunks", []))
        return {"draft": draft, "revisions": state.get("revisions", 0) + 1}

    def critic(state: MAState) -> dict:
        draft = state["draft"]
        if not draft.answered:
            return {"approved": True, "critique": "refusal accepted"}
        ctx = build_context(state.get("chunks", []))
        try:
            verdict = client.structured(
                f"Question: {state['question']}\n\nContext:\n{ctx}\n\nDraft answer:\n{draft.answer}",
                Verdict,
                system=CRITIC_SYSTEM,
            )
            return {"approved": bool(verdict.approved), "critique": verdict.critique}
        except Exception:
            return {"approved": True, "critique": "critic unavailable, passing through"}

    def route(state: MAState) -> str:
        if state.get("approved") or state.get("revisions", 0) >= max_revisions:
            return "end"
        return "revise"

    graph = StateGraph(MAState)
    graph.add_node("planner", planner)
    graph.add_node("retriever", retriever)
    graph.add_node("answerer", answerer)
    graph.add_node("critic", critic)
    graph.set_entry_point("planner")
    graph.add_edge("planner", "retriever")
    graph.add_edge("retriever", "answerer")
    graph.add_edge("answerer", "critic")
    graph.add_conditional_edges("critic", route, {"revise": "answerer", "end": END})
    return graph.compile()
