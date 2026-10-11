"""Pluggable message bus.

``InMemoryBus`` (default) fans messages out to asyncio queues inside one process.
``RedisBus`` uses Redis Streams (XADD / XREAD) so several engine or module processes
can share the stream; enable with ``TRIDENT_BUS=redis`` and ``REDIS_URL``. Kafka would
slot in behind the same two methods, partitioned by ticker.
"""
from __future__ import annotations

import asyncio
import json
import os
from collections import defaultdict
from typing import Any, AsyncIterator


class InMemoryBus:
    def __init__(self, maxsize: int = 10000):
        self.subs: dict[str, list[asyncio.Queue]] = defaultdict(list)
        self.maxsize = maxsize
        self.dropped = 0

    async def publish(self, topic: str, message: dict[str, Any]) -> None:
        for q in list(self.subs[topic]):
            try:
                q.put_nowait(message)
            except asyncio.QueueFull:
                self.dropped += 1

    async def subscribe(self, topic: str) -> AsyncIterator[dict[str, Any]]:
        q: asyncio.Queue = asyncio.Queue(self.maxsize)
        self.subs[topic].append(q)
        try:
            while True:
                yield await q.get()
        finally:
            self.subs[topic].remove(q)


class RedisBus:  # pragma: no cover - requires a Redis server
    def __init__(self, url: str | None = None):
        import redis.asyncio as redis

        self.r = redis.from_url(url or os.environ.get("REDIS_URL", "redis://localhost:6379/0"))

    async def publish(self, topic: str, message: dict[str, Any]) -> None:
        await self.r.xadd(f"trident:{topic}", {"data": json.dumps(message, default=str)}, maxlen=100000, approximate=True)

    async def subscribe(self, topic: str) -> AsyncIterator[dict[str, Any]]:
        last = "$"
        while True:
            resp = await self.r.xread({f"trident:{topic}": last}, block=1000, count=100)
            for _, entries in resp or []:
                for eid, fields in entries:
                    last = eid
                    yield json.loads(fields[b"data"])


def make_bus():
    if os.environ.get("TRIDENT_BUS", "memory").lower() == "redis":
        return RedisBus()
    return InMemoryBus()
