"""Deterministic replay of historical windows (the demo's backbone).

A replay file is JSONL with one item per line:
  {"ts": ISO-8601, "text": str, "source_type": "newswire|social|filing|gdelt",
   "source_name": str, "url": optional, "is_synthetic": bool}
Items are emitted in timestamp order, compressed by ``speed`` (simulated seconds per
real second), and stamped with a ``replay_id`` so their signals are labeled on screen.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import AsyncIterator

from trident.config import DATA_DIR
from trident.pipeline.normalize import make_doc
from trident.signals.schema import Doc
from trident.utils.common import parse_ts, read_jsonl


def list_replays() -> list[str]:
    return sorted(p.stem for p in (DATA_DIR / "replay").glob("*.jsonl"))


def load_replay(name: str) -> list[dict]:
    p = Path(name)
    if not p.exists():
        p = DATA_DIR / "replay" / f"{name}.jsonl"
    rows = list(read_jsonl(p))
    rows.sort(key=lambda r: parse_ts(r["ts"]))
    return rows


def replay_docs(name: str) -> list[Doc]:
    return [make_doc(r["text"], r.get("source_type", "newswire"), r.get("source_name", "replay"), r["ts"], r.get("url"),
                     r.get("title"), bool(r.get("is_synthetic", False)), replay_id=name, meta=r.get("meta", {}))
            for r in load_replay(name)]


@dataclass
class ReplayController:
    name: str
    speed: float = 600.0           # 600x: one simulated hour in six seconds
    max_gap_s: float = 4.0
    state: dict = field(default_factory=lambda: {"running": False, "position": 0, "total": 0, "sim_time": None})

    async def stream(self) -> AsyncIterator[Doc]:
        docs = replay_docs(self.name)
        self.state.update(running=True, position=0, total=len(docs))
        prev = None
        try:
            for i, d in enumerate(docs):
                ts = parse_ts(d.ts_published)
                if prev is not None:
                    await asyncio.sleep(min(max((ts - prev).total_seconds() / self.speed, 0.0), self.max_gap_s))
                prev = ts
                self.state.update(position=i + 1, sim_time=d.ts_published)
                yield d
        finally:
            self.state["running"] = False
