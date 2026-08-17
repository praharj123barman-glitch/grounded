"""MCP server (course W6): expose Grounded's retrieval and answering as tools any
MCP client (Claude Desktop, IDEs, other agents) can call.

Run over stdio:
  python -m grounded.mcp_server.server

Tools:
  search_documents(query, k) -> list of cited chunks
  answer_question(question)  -> grounded answer with citations and confidence
"""
from __future__ import annotations

from functools import lru_cache

from fastmcp import FastMCP

from grounded.retrieval.types import RetrievedChunk

mcp = FastMCP("grounded")


def format_hits(hits: list[RetrievedChunk]) -> list[dict]:
    return [
        {"source": h.source, "page": h.page, "score": round(h.score, 4), "text": h.text}
        for h in hits
    ]


@lru_cache(maxsize=1)
def _pipeline():
    from grounded.rag.pipeline import build_default_pipeline

    return build_default_pipeline()


@mcp.tool()
def search_documents(query: str, k: int = 5) -> list[dict]:
    """Search the indexed annual reports; return the top-k chunks with source and page."""
    return format_hits(_pipeline().retrieve(query, k=k))


@mcp.tool()
def answer_question(question: str) -> dict:
    """Answer a question grounded only in the documents, with citations and a confidence score."""
    answer, chunks = _pipeline().run(question)
    return {**answer.model_dump(), "retrieved": len(chunks)}


if __name__ == "__main__":
    mcp.run()
