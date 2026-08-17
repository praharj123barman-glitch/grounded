"""LangGraph agent that wraps the RAG pipeline (Phase 5).

A small state graph: retrieve, then answer. Built as a graph (not a straight
function) so it extends naturally to multi-hop retrieval, tool calls, and
conditional routing. The pipeline is injected, so this compiles and runs in
tests with a fake pipeline (no API key needed to compile).
"""
from __future__ import annotations

from typing import Any, TypedDict

from langgraph.graph import END, StateGraph


class AgentState(TypedDict, total=False):
    question: str
    chunks: list[Any]
    answer: Any


def build_agent(pipeline):
    def retrieve_node(state: AgentState) -> dict:
        return {"chunks": pipeline.retrieve(state["question"])}

    def answer_node(state: AgentState) -> dict:
        chunks = state.get("chunks", [])
        return {"answer": pipeline.answerer.answer(state["question"], chunks)}

    graph = StateGraph(AgentState)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("answer", answer_node)
    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "answer")
    graph.add_edge("answer", END)
    return graph.compile()
