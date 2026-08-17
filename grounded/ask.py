"""Ask Grounded a question from the terminal (Phase 2).

  python -m grounded.ask "What was Meridian's FY2025 revenue?"
"""
from __future__ import annotations

import sys

from grounded.config import settings


def main() -> None:
    if len(sys.argv) < 2:
        print('Usage: python -m grounded.ask "your question"')
        return
    question = " ".join(sys.argv[1:])

    if not settings.google_api_key:
        print("GOOGLE_API_KEY is empty. Add it to .env, then run ingestion, then ask again.")
        return

    from grounded.rag.pipeline import build_default_pipeline

    pipe = build_default_pipeline()
    answer, chunks = pipe.run(question)

    print(f"\nAnswered: {answer.answered}   confidence: {answer.confidence:.2f}")
    print(f"Answer: {answer.answer}\n")
    for c in answer.citations:
        print(f"  cite [{c.source} p{getattr(c, 'page', '')}] {c.quote[:140]}")
    print(f"\n(retrieved {len(chunks)} chunks; "
          f"llm spend so far ${pipe.answerer.client.usage.cost:.5f})")


if __name__ == "__main__":
    main()
