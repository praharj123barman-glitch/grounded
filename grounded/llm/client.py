"""Provider-agnostic LLM client with token, cost, and latency logging.

Phase 0 wires Google Gemini. Adding OpenAI or Anthropic later is a new branch in
_build_chat; nothing else in the app changes. Cost logging is here from the first
call so every later experiment is measurable.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Type

from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_exponential

from grounded.config import settings
from grounded.llm.pricing import cost_usd

log = logging.getLogger("grounded.llm")


@dataclass
class Usage:
    """Running tally of spend for one client instance."""
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost: float = 0.0

    def add(self, model: str, input_tokens: int, output_tokens: int) -> None:
        self.calls += 1
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.cost += cost_usd(model, input_tokens, output_tokens)


class LLMClient:
    def __init__(self, model: str | None = None) -> None:
        self.model = model or settings.llm_model
        self.usage = Usage()
        self._chat = self._build_chat()

    def _build_chat(self):
        if settings.llm_provider == "google":
            from langchain_google_genai import ChatGoogleGenerativeAI

            return ChatGoogleGenerativeAI(
                model=self.model,
                google_api_key=settings.google_api_key,
                temperature=0,
            )
        raise ValueError(f"Unknown LLM provider: {settings.llm_provider}")

    def _record(self, resp, t0: float) -> None:
        meta = getattr(resp, "usage_metadata", None) or {}
        i = int(meta.get("input_tokens", 0) or 0)
        o = int(meta.get("output_tokens", 0) or 0)
        self.usage.add(self.model, i, o)
        log.info(
            "llm model=%s in=%d out=%d cost=$%.5f %.2fs",
            self.model, i, o, cost_usd(self.model, i, o), time.time() - t0,
        )

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def complete(self, prompt: str, system: str | None = None) -> str:
        messages = ([("system", system)] if system else []) + [("human", prompt)]
        t0 = time.time()
        resp = self._chat.invoke(messages)
        self._record(resp, t0)
        return resp.content

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def structured(self, prompt: str, schema: Type[BaseModel], system: str | None = None):
        """Return a validated pydantic object of type `schema`."""
        messages = ([("system", system)] if system else []) + [("human", prompt)]
        model = self._chat.with_structured_output(schema)
        return model.invoke(messages)
