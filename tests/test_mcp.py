from grounded.mcp_server.server import format_hits, mcp
from grounded.retrieval.types import RetrievedChunk


def test_format_hits_shapes_citation():
    hits = [RetrievedChunk(id="1", text="revenue 48,200", source="report.md", page=2, score=0.912)]
    out = format_hits(hits)
    assert out[0]["source"] == "report.md"
    assert out[0]["page"] == 2
    assert "48,200" in out[0]["text"]


def test_mcp_server_exists():
    assert mcp is not None
