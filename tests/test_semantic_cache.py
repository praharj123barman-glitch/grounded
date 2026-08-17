from grounded.cache.semantic_cache import SemanticCache
from tests.fakes import FakeEmbedder


def test_cache_returns_on_identical_query():
    cache = SemanticCache(FakeEmbedder(), threshold=0.9)
    cache.put("what was the revenue", "ANSWER")
    assert cache.get("what was the revenue") == "ANSWER"


def test_cache_misses_on_unrelated_query():
    cache = SemanticCache(FakeEmbedder(), threshold=0.99)
    cache.put("revenue figure for the year", "ANSWER")
    assert cache.get("what is the capital of France") is None
