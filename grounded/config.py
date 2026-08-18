"""Central configuration for Grounded, loaded from environment or a .env file.

Every tunable lives here so experiments (chunking, top_k, model choice, retrieval
strategy) are a single config change, not a code edit scattered across files.
"""
from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# Repository root: C:\Users\praha\grounded  (this file is grounded/grounded/config.py)
ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- LLM and embeddings ---
    google_api_key: str = ""
    llm_provider: str = "google"          # google | (openai / anthropic later)
    llm_model: str = "gemini-2.5-flash"
    judge_model: str = "gemini-2.5-flash-lite"   # separate model for the eval judge (own daily quota)
    embedding_model: str = "models/gemini-embedding-001"

    # --- Ingestion / chunking (ablatable knobs) ---
    chunk_strategy: str = "recursive"     # recursive | fixed
    chunk_size: int = 800
    chunk_overlap: int = 120

    # --- Vector store ---
    vector_dir: Path = ROOT / "data" / "chroma"
    collection_name: str = "grounded"

    # --- Paths and retrieval ---
    raw_dir: Path = ROOT / "data" / "raw"
    sample_dir: Path = ROOT / "data" / "sample"
    top_k: int = 5
    bootstrap_on_start: bool = False      # if true, the API ingests the sample corpus on startup (deploy)

    # --- Retrieval strategy (ablatable in Phase 4) ---
    use_hybrid: bool = True               # BM25 + dense with reciprocal rank fusion
    use_query_rewrite: bool = False       # expand the question into sub-queries
    use_rerank: bool = False              # reorder candidates with an LLM reranker

    # --- Observability (optional, no-op if unset) ---
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"


settings = Settings()
