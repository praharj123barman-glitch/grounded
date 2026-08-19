"""Input guardrails (OWASP LLM Top 10).

Detects common prompt-injection phrasing and flags PII in the incoming question.
Injection blocks the request; PII is flagged (and could be redacted upstream).
Pattern-based and cheap, meant as a first line, not a complete defense.
"""
from __future__ import annotations

import re

_INJECTION = [
    r"ignore (all|any|the|your|previous|above) .*instructions",
    r"disregard .*(instructions|context|rules)",
    r"you are now",
    r"forget (all|everything|previous)",
    r"reveal .*(system prompt|instructions|prompt)",
    r"print .*(system prompt|your instructions)",
    r"act as .*(dan|jailbreak)",
]

_PII = {
    "email": r"[\w.+-]+@[\w-]+\.[\w.-]+",
    "phone": r"\b\d{10}\b",
    "pan": r"\b[A-Z]{5}\d{4}[A-Z]\b",
    "aadhaar": r"\b\d{4}\s?\d{4}\s?\d{4}\b",
}


def scan(text: str) -> dict:
    lowered = text.lower()
    injection = any(re.search(p, lowered) for p in _INJECTION)
    pii = [name for name, pat in _PII.items() if re.search(pat, text)]
    return {"injection": injection, "pii": pii, "blocked": injection}
