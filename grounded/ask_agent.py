"""Run the multi-agent (planner, retrieve, answer, critic) over one question.

  python -m grounded.ask_agent "Compare Meridian's FY2025 revenue and profit growth."

Requires GOOGLE_API_KEY.
"""
from __future__ import annotations

import sys

from grounded.config import settings


def main() -> None:
    if len(sys.argv) < 2:
        print('Usage: python -m grounded.ask_agent "your question"')
        return
    question = " ".join(sys.argv[1:])
    if not settings.google_api_key:
        print("GOOGLE_API_KEY is empty. Add it to .env, ingest, then try again.")
        return

    from grounded.agent.multi_agent import build_multi_agent
    from grounded.llm.client import LLMClient
    from grounded.rag.pipeline import build_default_pipeline

    agent = build_multi_agent(build_default_pipeline(), LLMClient())
    out = agent.invoke({"question": question})
    draft = out["draft"]

    print(f"\nSub-questions: {out.get('subquestions')}")
    print(f"Critic approved: {out.get('approved')}   ({out.get('critique', '')})")
    print(f"Answer: {draft.answer}   [confidence {draft.confidence:.2f}]")
    for c in draft.citations:
        print(f"  cite [{c.source}] {c.quote[:120]}")


if __name__ == "__main__":
    main()
