"""Query rewriting: expand a question into a few diverse search queries.

Catches the case where the user's wording does not match the document's wording.
Falls back to the original question on any error, so it can never break retrieval.
"""
from __future__ import annotations

_SYSTEM = (
    "Rewrite the user's question into up to 3 diverse search queries that would "
    "retrieve relevant passages. Keep the original intent. Output one query per line, "
    "no numbering."
)


class QueryRewriter:
    def __init__(self, client) -> None:
        self.client = client

    def rewrite(self, question: str) -> list[str]:
        try:
            text = self.client.complete(question, system=_SYSTEM)
        except Exception:
            return [question]
        subs = [line.strip("-*0123456789. ").strip() for line in text.splitlines()]
        subs = [s for s in subs if s]
        queries = [question] + subs
        # de-duplicate, preserve order, cap at 4
        seen: set[str] = set()
        out: list[str] = []
        for q in queries:
            key = q.lower()
            if key not in seen:
                seen.add(key)
                out.append(q)
        return out[:4]
