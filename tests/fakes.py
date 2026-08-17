"""Offline fakes so the pipeline can be tested without any API key.

FakeEmbedder maps text to a deterministic normalised bag-of-hashed-tokens vector,
so texts that share words end up close in cosine space (good enough for retrieval
tests). FakeLLM returns a canned structured answer.
"""
from __future__ import annotations

import hashlib
import math
import re


class FakeEmbedder:
    dim = 256

    def _vec(self, text: str) -> list[float]:
        v = [0.0] * self.dim
        for tok in re.findall(r"[a-z0-9]+", text.lower()):
            idx = int(hashlib.md5(tok.encode()).hexdigest(), 16) % self.dim
            v[idx] += 1.0
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]


class _Usage:
    cost = 0.0


class FakeLLM:
    def __init__(self, answer=None, complete_return: str = "") -> None:
        self.usage = _Usage()
        self._answer = answer
        self._complete = complete_return

    def structured(self, prompt, schema, system=None):
        if self._answer is not None:
            return self._answer
        return schema(answered=True, answer="stub answer", citations=[], confidence=0.5)

    def complete(self, prompt, system=None) -> str:
        return self._complete
