"""Approximate token pricing (USD per 1,000,000 tokens) for internal cost logging.

These are order-of-magnitude figures used to track spend across experiments, not
a billing source of truth. Update as provider pricing changes.
"""
from __future__ import annotations

# model -> (input_per_million, output_per_million)
PRICING: dict[str, tuple[float, float]] = {
    "gemini-2.5-flash": (0.30, 2.50),
    "gemini-2.5-flash-lite": (0.10, 0.40),
    "gemini-2.0-flash": (0.10, 0.40),
    "gemini-1.5-flash": (0.075, 0.30),
    "models/text-embedding-004": (0.0, 0.0),   # free tier
}


def cost_usd(model: str, input_tokens: int, output_tokens: int = 0) -> float:
    inp, out = PRICING.get(model, (0.0, 0.0))
    return (input_tokens / 1_000_000) * inp + (output_tokens / 1_000_000) * out
