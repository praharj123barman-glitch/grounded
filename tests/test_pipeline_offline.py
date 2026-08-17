import pytest

from grounded.config import settings
from grounded.ingestion.chunker import chunk_documents
from grounded.ingestion.indexer import build_index
from grounded.ingestion.loader import load_dir
from grounded.rag.answerer import Answerer
from grounded.rag.pipeline import RAGPipeline
from grounded.retrieval.dense import DenseRetriever
from tests.fakes import FakeEmbedder, FakeLLM


@pytest.fixture
def embedder(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "vector_dir", tmp_path / "chroma")
    monkeypatch.setattr(settings, "collection_name", "test")
    docs = load_dir(settings.sample_dir)
    chunks = chunk_documents(docs)
    emb = FakeEmbedder()
    build_index(chunks, emb.embed_texts([c.text for c in chunks]))
    return emb


def test_dense_retrieves_revenue_chunk(embedder):
    hits = DenseRetriever(embedder).retrieve("What was consolidated revenue in FY2025?", k=3)
    assert hits
    assert any("48,200" in h.text for h in hits)


def test_pipeline_answers_when_context_found(embedder):
    pipe = RAGPipeline(DenseRetriever(embedder), Answerer(FakeLLM()))
    answer, chunks = pipe.run("What was revenue?", k=3)
    assert chunks
    assert answer.answered is True


def test_pipeline_refuses_on_empty_index(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "vector_dir", tmp_path / "empty")
    monkeypatch.setattr(settings, "collection_name", "empty")
    pipe = RAGPipeline(DenseRetriever(FakeEmbedder()), Answerer(FakeLLM()))
    answer, chunks = pipe.run("anything at all", k=3)
    assert chunks == []
    assert answer.answered is False
