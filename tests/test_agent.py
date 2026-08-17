from grounded.agent.graph import build_agent
from grounded.llm.schemas import GroundedAnswer
from grounded.retrieval.types import RetrievedChunk


class _FakeAnswerer:
    def answer(self, question, chunks):
        return GroundedAnswer(
            answered=bool(chunks), answer="ok", citations=[], confidence=0.8
        )


class FakePipeline:
    def __init__(self):
        self.answerer = _FakeAnswerer()

    def retrieve(self, question, k=None):
        return [RetrievedChunk(id="1", text="revenue 48,200", source="s", page=1, score=1.0)]


def test_agent_compiles_and_runs():
    agent = build_agent(FakePipeline())
    out = agent.invoke({"question": "what was revenue?"})
    assert out["chunks"]
    assert out["answer"].answered is True
