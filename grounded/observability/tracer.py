"""Thin observability shim (course W7.1).

If Langfuse keys are set it sends traces; otherwise it is a no-op, so nothing in
the app depends on Langfuse being installed or configured.
"""
from __future__ import annotations

import logging

from grounded.config import settings

log = logging.getLogger("grounded.trace")


class Tracer:
    def __init__(self) -> None:
        self.enabled = bool(settings.langfuse_public_key and settings.langfuse_secret_key)
        self._client = None
        if self.enabled:
            try:
                from langfuse import Langfuse

                self._client = Langfuse(
                    public_key=settings.langfuse_public_key,
                    secret_key=settings.langfuse_secret_key,
                    host=settings.langfuse_host,
                )
            except Exception:
                self.enabled = False

    def event(self, name: str, **data) -> None:
        if self.enabled and self._client is not None:
            try:
                self._client.event(name=name, metadata=data)
                return
            except Exception:
                pass
        log.debug("trace %s %s", name, data)


_tracer: Tracer | None = None


def get_tracer() -> Tracer:
    global _tracer
    if _tracer is None:
        _tracer = Tracer()
    return _tracer
