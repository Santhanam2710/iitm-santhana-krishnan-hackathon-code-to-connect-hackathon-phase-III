"""Adapter base class: polling loop, token-bucket rate limit, exponential backoff and a
circuit breaker per source (red-team item 14). Adapters implement ``fetch()`` and yield
canonical Docs; the runtime consumes ``stream()``.
"""
from __future__ import annotations

import asyncio
import random
import time
from dataclasses import dataclass, field
from typing import AsyncIterator

from trident.signals.schema import Doc
from trident.utils.common import get_logger, iso, utcnow


@dataclass
class SourceHealth:
    name: str
    state: str = "idle"            # idle | ok | degraded | open (circuit open) | disabled
    last_ok: str | None = None
    last_error: str | None = None
    consecutive_failures: int = 0
    docs: int = 0
    polls: int = 0
    lag_seconds: float | None = None
    note: str = ""

    def as_dict(self) -> dict:
        return self.__dict__.copy()


class TokenBucket:
    def __init__(self, rate_per_s: float, burst: int = 1):
        self.rate, self.capacity, self.tokens, self.t = rate_per_s, burst, float(burst), time.monotonic()

    async def take(self) -> None:
        while True:
            now = time.monotonic()
            self.tokens = min(self.capacity, self.tokens + (now - self.t) * self.rate)
            self.t = now
            if self.tokens >= 1:
                self.tokens -= 1
                return
            await asyncio.sleep((1 - self.tokens) / self.rate)


@dataclass
class Adapter:
    name: str
    poll_seconds: float = 60.0
    rate_per_s: float = 1.0
    failure_threshold: int = 3
    open_seconds: float = 300.0
    health: SourceHealth = field(init=False)
    enabled: bool = True

    def __post_init__(self):
        self.health = SourceHealth(self.name)
        self.bucket = TokenBucket(self.rate_per_s)
        self.log = get_logger(f"trident.ingest.{self.name}")
        self._seen: set[str] = set()

    async def fetch(self) -> list[Doc]:  # pragma: no cover - abstract
        raise NotImplementedError

    def _fresh(self, docs: list[Doc]) -> list[Doc]:
        out = []
        for d in docs:
            if d.doc_id not in self._seen:
                self._seen.add(d.doc_id)
                out.append(d)
        if len(self._seen) > 50000:
            self._seen = set(list(self._seen)[-20000:])
        return out

    async def stream(self) -> AsyncIterator[Doc]:
        backoff = self.poll_seconds
        while self.enabled:
            if self.health.state == "open":
                await asyncio.sleep(self.open_seconds)
                self.health.state = "degraded"  # half-open: try once
            await self.bucket.take()
            self.health.polls += 1
            try:
                docs = self._fresh(await self.fetch())
                self.health.state, self.health.last_ok = "ok", iso(utcnow())
                self.health.consecutive_failures = 0
                self.health.docs += len(docs)
                if docs:
                    from trident.utils.common import parse_ts

                    self.health.lag_seconds = round(min((utcnow() - parse_ts(d.ts_published)).total_seconds() for d in docs), 1)
                for d in docs:
                    yield d
                backoff = self.poll_seconds
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # network, parsing, rate limit
                self.health.consecutive_failures += 1
                self.health.last_error = f"{type(exc).__name__}: {str(exc)[:160]}"
                self.health.state = "open" if self.health.consecutive_failures >= self.failure_threshold else "degraded"
                self.log.warning("%s fetch failed (%s); state=%s", self.name, self.health.last_error, self.health.state)
                backoff = min(backoff * 2, 900)
            await asyncio.sleep(backoff * (0.9 + 0.2 * random.random()))
