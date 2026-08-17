"""Phase 1 ingestion pipeline: load, chunk, embed, index.

Usage:
  python -m grounded.ingest --dry-run             # load + chunk only (no API key needed)
  python -m grounded.ingest --source sample       # use data/sample (the shipped excerpt)
  python -m grounded.ingest                        # full run over data/raw (needs GOOGLE_API_KEY)
"""
from __future__ import annotations

import argparse
import logging
import time

from grounded.config import settings
from grounded.ingestion.chunker import chunk_documents
from grounded.ingestion.loader import load_dir

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("grounded.ingest")


def main() -> None:
    ap = argparse.ArgumentParser(description="Grounded ingestion pipeline")
    ap.add_argument("--dry-run", action="store_true",
                    help="load and chunk only, skip embeddings (no API key required)")
    ap.add_argument("--source", choices=["raw", "sample"], default="raw",
                    help="which folder to ingest from")
    ap.add_argument("--limit", type=int, default=0,
                    help="cap number of chunks (quick tests)")
    args = ap.parse_args()

    directory = settings.raw_dir if args.source == "raw" else settings.sample_dir
    log.info("Loading from %s", directory)
    docs = load_dir(directory)
    log.info("Loaded %d document pages", len(docs))
    if not docs:
        log.warning("No documents found. Drop PDFs into %s and run again.", directory)
        return

    chunks = chunk_documents(docs)
    if args.limit:
        chunks = chunks[: args.limit]
    avg = sum(len(c.text) for c in chunks) // max(len(chunks), 1)
    log.info(
        "Chunked into %d chunks (avg %d chars) [strategy=%s size=%d overlap=%d]",
        len(chunks), avg, settings.chunk_strategy, settings.chunk_size, settings.chunk_overlap,
    )

    if args.dry_run:
        preview = chunks[0].text[:280] if chunks else ""
        log.info("Dry run complete. First chunk (%s):\n%s", chunks[0].id, preview)
        return

    if not settings.google_api_key:
        log.error("GOOGLE_API_KEY is empty. Copy .env.example to .env, add your key, "
                  "then run again without --dry-run.")
        return

    from grounded.ingestion.embeddings import Embedder
    from grounded.ingestion.indexer import build_index

    emb = Embedder()
    t0 = time.time()
    vectors = emb.embed_texts([c.text for c in chunks])
    log.info("Embedded %d chunks in %.1fs", len(vectors), time.time() - t0)

    count = build_index(chunks, vectors)
    log.info("Index built. Collection '%s' now holds %d chunks at %s",
             settings.collection_name, count, settings.vector_dir)


if __name__ == "__main__":
    main()
