from grounded.config import settings
from grounded.ingestion.chunker import chunk_documents
from grounded.ingestion.indexer import build_index
from grounded.ingestion.loader import load_dir
from grounded.retrieval.hybrid import HybridRetriever
from tests.fakes import FakeEmbedder


def test_hybrid_finds_keyword_heavy_chunk(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "vector_dir", tmp_path / "chroma")
    monkeypatch.setattr(settings, "collection_name", "test")
    docs = load_dir(settings.sample_dir)
    chunks = chunk_documents(docs)
    emb = FakeEmbedder()
    build_index(chunks, emb.embed_texts([c.text for c in chunks]))

    hits = HybridRetriever(emb).retrieve("net debt to EBITDA ratio", k=3)
    assert hits
    assert any(("0.50" in h.text) or ("4,600" in h.text) for h in hits)
