from grounded.agent.multi_agent import build_multi_agent
from grounded.llm.schemas import GroundedAnswer
from grounded.retrieval.types import RetrievedChunk


class _Answerer:
    def answer(self, question, chunks):
        return GroundedAnswer(answered=bool(chunks), answer="ok", citations=[], confidence=0.9)


class FakePipeline:
    def __init__(self):
        self.answerer = _Answerer()

    def retrieve(self, question, k=None):
        return [RetrievedChunk(id="1", text="revenue 48,200", source="s", page=1, score=1.0)]


class FakeClient:
    approved_value = True

    def complete(self, prompt, system=None):
        return "What was revenue?\nHow much did it grow?"

    def structured(self, prompt, schema, system=None):
        return schema(approved=self.approved_value, critique="checked")


def test_multi_agent_approves_and_answers():
    agent = build_multi_agent(FakePipeline(), FakeClient())
    out = agent.invoke({"question": "revenue and growth?"})
    assert out["draft"].answered is True
    assert out["approved"] is True
    assert out["subquestions"]  # planner produced sub-questions


def test_multi_agent_stops_at_max_revisions():
    class Rejecter(FakeClient):
        approved_value = False

    agent = build_multi_agent(FakePipeline(), Rejecter(), max_revisions=2)
    out = agent.invoke({"question": "revenue?"})
    assert out["approved"] is False
    assert out["revisions"] <= 2   # the loop is bounded, no infinite revision
