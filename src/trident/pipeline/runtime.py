"""Runtime: wires sources -> queue -> batched engine -> store/bus -> Module A + Module B -> WebSocket hub.

Modes:
  replay  stream a historical window from data/replay (default; no network or keys needed)
  live    poll RSS, SEC EDGAR, GDELT, NewsAPI (if key) and Reddit
  idle    API only; feed it with POST /analyze or POST /replay/start
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import time
from collections import deque
from datetime import datetime
from typing import Any

import numpy as np

from trident.config import get_config
from trident.ingest.base import Adapter
from trident.ingest.replay import ReplayController, list_replays
from trident.modules.rebalancer.service import Rebalancer
from trident.modules.stress.service import StressService
from trident.pipeline.bus import make_bus
from trident.pipeline.engine import Engine
from trident.pipeline.normalize import make_doc
from trident.signals.schema import Doc
from trident.signals.store import SignalStore
from trident.utils.common import get_logger, iso, parse_ts, utcnow

log = get_logger("trident.runtime")


class Hub:
    """WebSocket fan-out."""

    def __init__(self):
        self.clients: set[Any] = set()

    async def broadcast(self, kind: str, data: Any) -> None:
        if not self.clients:
            return
        msg = json.dumps({"type": kind, "data": data}, default=str)
        dead = []
        for ws in list(self.clients):
            try:
                await ws.send_text(msg)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.clients.discard(ws)


def stress_summary(res: dict) -> dict:
    keep = ("run_id", "ts", "label", "event_class", "impact", "multiplier", "preset", "value_before", "value_after", "pnl", "pnl_pct",
            "cet1_before", "cet1_after", "rwa_before", "rwa_after", "var_before", "var_after", "es_before", "es_after", "minimum",
            "buffer_line", "by_class", "reverse", "trigger_signal", "trigger_text", "decision", "shock")
    out = {k: res.get(k) for k in keep}
    out["positions"] = [{k: p.get(k) for k in ("id", "name", "asset_class", "kind", "sector", "v0", "v1", "dv", "method", "pd", "pd_stressed",
                                                "lgd", "lgd_stressed", "dy_bp")} for p in res.get("positions", [])]
    return out


class Runtime:
    def __init__(self, mode: str = "replay", replay: str | None = None, speed: float = 600.0, reset: bool = True):
        self.cfg = get_config()
        self.mode = mode
        self.replay_name = replay or ("svb_2023-03" if "svb_2023-03" in list_replays() else (list_replays() or [None])[0])
        self.speed = speed
        self.engine = Engine()
        self.store = SignalStore()
        if reset:
            self.store.reset()
        self.bus = make_bus()
        self.rebalancer = Rebalancer()
        self.stress = StressService()
        self.hub = Hub()
        self.queue: asyncio.Queue[Doc] = asyncio.Queue(maxsize=10000)
        self.adapters: dict[str, Adapter] = {}
        self.replay: ReplayController | None = None
        self.tasks: list[asyncio.Task] = []
        self.replay_task: asyncio.Task | None = None
        self.recent: deque[dict] = deque(maxlen=300)
        self.rejected: deque[dict] = deque(maxlen=200)
        self.batch_ms: deque[float] = deque(maxlen=2000)
        self.started = iso(utcnow())
        self.sim_time: datetime | None = None
        eng = self.cfg.section("engine")
        self.batch_size = int(eng.get("batch_size", 16))
        self.batch_wait = float(eng.get("batch_wait_ms", 20)) / 1000.0

    # ------------------------------------------------------------------ lifecycle
    async def start(self) -> None:
        self.engine.warmup()
        self.tasks.append(asyncio.create_task(self._worker(), name="engine-worker"))
        if self.mode == "live":
            self._start_live()
        elif self.mode == "replay" and self.replay_name:
            await self.start_replay(self.replay_name, self.speed)
        log.info("runtime started in %s mode (sentiment=%s, events=%s)", self.mode, self.engine.sentiment.name, self.engine.events.name)

    async def stop(self) -> None:
        for t in self.tasks + ([self.replay_task] if self.replay_task else []):
            t.cancel()
        for t in self.tasks:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await t

    def _start_live(self) -> None:
        from trident.ingest.edgar import EdgarAdapter
        from trident.ingest.gdelt import GdeltDocAdapter
        from trident.ingest.newsapi import NewsApiAdapter
        from trident.ingest.rss import RSSAdapter
        from trident.ingest.social import RedditAdapter

        ing = self.cfg.section("ingest")
        adapters = [RSSAdapter("rss", poll_seconds=ing.get("rss_poll_seconds", 60)),
                    EdgarAdapter("sec-edgar", poll_seconds=ing.get("edgar_poll_seconds", 120), rate_per_s=0.5),
                    GdeltDocAdapter("gdelt", poll_seconds=ing.get("gdelt_poll_seconds", 900)),
                    NewsApiAdapter("newsapi", poll_seconds=ing.get("newsapi_poll_seconds", 1800)),
                    RedditAdapter("reddit", poll_seconds=120)]
        for a in adapters:
            self.adapters[a.name] = a
            if a.enabled:
                self.tasks.append(asyncio.create_task(self._pump(a.stream()), name=f"src-{a.name}"))

    async def _pump(self, stream) -> None:
        async for doc in stream:
            await self.queue.put(doc)

    async def start_replay(self, name: str, speed: float | None = None) -> dict:
        await self.stop_replay()
        self.replay_name = name
        self.replay = ReplayController(name, speed=speed or self.speed)
        self.replay_task = asyncio.create_task(self._pump(self.replay.stream()), name=f"replay-{name}")
        return {"replay": name, "speed": self.replay.speed}

    async def stop_replay(self) -> None:
        if self.replay_task and not self.replay_task.done():
            self.replay_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self.replay_task
        self.replay_task = None

    async def inject_synthetic(self, n: int = 12, scenario: str = "credit") -> int:
        """Arm a synthetic burst (tagged is_synthetic) at the current simulated time."""
        from trident.ingest.synthetic import TemplateGenerator

        gen = TemplateGenerator(seed=int(time.time()) % 10000)
        start = self.sim_time or utcnow()
        rows = [r for r in gen.generate(n * 4, start=start, hours=1.0, burst=True)]
        want = {"credit": "Credit Event", "geo": "Geopolitical", "macro": "Macroeconomic"}.get(scenario)
        rows = [r for r in rows if want is None or r["label_event"] == want][:n] or rows[:n]
        for r in rows:
            await self.queue.put(make_doc(r["text"], r["source_type"], r["source_name"], r["ts"], is_synthetic=True))
        return len(rows)

    # ------------------------------------------------------------------ processing
    async def _worker(self) -> None:
        while True:
            first = await self.queue.get()
            batch = [first]
            deadline = time.perf_counter() + self.batch_wait
            while len(batch) < self.batch_size:
                timeout = deadline - time.perf_counter()
                if timeout <= 0:
                    break
                try:
                    batch.append(await asyncio.wait_for(self.queue.get(), timeout))
                except asyncio.TimeoutError:
                    break
            try:
                await self.process(batch)
            except Exception as exc:  # keep the worker alive
                log.exception("batch failed: %s", exc)

    async def process(self, docs: list[Doc], publish: bool = True) -> list[dict]:
        t0 = time.perf_counter()
        outs = await asyncio.to_thread(self.engine.process_batch, docs) if not self.engine.sentiment.is_lexicon else self.engine.process_batch(docs)
        self.batch_ms.append((time.perf_counter() - t0) * 1e3)
        emitted: list[dict] = []
        for out in outs:
            if out.dropped or out.rejected_entities:
                self.rejected.append({"ts": out.doc.ts_published, "text": out.doc.text[:200], "dropped": out.dropped,
                                      "rejected_entities": out.rejected_entities, "source": out.doc.source_name})
            if not out.signals:
                continue
            if publish:
                self.store.add(out.signals)
            for s in out.signals:
                d = s.model_dump()
                emitted.append(d)
                if not publish:
                    continue
                self.recent.append(d)
                ts = parse_ts(s.ts_published)
                self.sim_time = ts if not self.sim_time or ts > self.sim_time else self.sim_time
                self.rebalancer.on_signal(d)
                await self.bus.publish("signals", d)
                await self.hub.broadcast("signal", d)
                res = self.stress.on_signal(d)
                dec = self.stress.decisions[-1]
                await self.hub.broadcast("decision", dec)
                if res:
                    summ = stress_summary(res)
                    self.store.add_stress_run(res["run_id"], res["ts"], res["trigger_signal"], summ)
                    await self.hub.broadcast("stress", summ)
        if publish and emitted and self.sim_time:
            rec = self.rebalancer.rebalance(self.sim_time)
            await self.hub.broadcast("weights", rec)
        if publish:
            await self.hub.broadcast("rejected", list(self.rejected)[-5:])
        return emitted

    async def analyze(self, text: str, source_type: str = "newswire", publish: bool = False) -> dict:
        """Engine Lab endpoint. Uses a sibling engine (shared models, fresh dedup/novelty) unless publish=True."""
        doc = make_doc(text, source_type, "analyze:manual", self.sim_time or utcnow(), replay_id="analyze" if self.sim_time else None)
        t0 = time.perf_counter()
        if publish:
            sigs = await self.process([doc], publish=True)
            return {"signals": sigs, "rejected_entities": [], "dropped": None, "elapsed_ms": round((time.perf_counter() - t0) * 1e3, 3)}
        out = _fresh_engine(self.engine).process_batch([doc])[0]
        return {"signals": [s.model_dump() for s in out.signals], "rejected_entities": out.rejected_entities, "dropped": out.dropped,
                "timings_ms": {k: round(v, 3) for k, v in out.timings_ms.items()}, "elapsed_ms": round((time.perf_counter() - t0) * 1e3, 3)}

    def reset_state(self) -> None:
        """Clear signals and module state (used when a new replay starts)."""
        self.store.reset()
        self.engine = Engine(sentiment=self.engine.sentiment, events=self.engine.events, impact=self.engine.impact, resolver=self.engine.resolver)
        self.rebalancer = Rebalancer()
        self.stress = StressService()
        self.recent.clear()
        self.rejected.clear()
        self.sim_time = None

    # ------------------------------------------------------------------ status
    def status(self) -> dict:
        lat = np.array(self.engine.latencies) if self.engine.latencies else np.array([0.0])
        return {
            "mode": self.mode, "started": self.started, "sim_time": iso(self.sim_time) if self.sim_time else None,
            "replay": {"name": self.replay_name, **(self.replay.state if self.replay else {})},
            "models": self.engine.model_versions, "counters": self.engine.counters, "signals_stored": self.store.count(),
            "latency_ms": {"p50": float(np.percentile(lat, 50)), "p95": float(np.percentile(lat, 95)), "p99": float(np.percentile(lat, 99)),
                           "n": int(len(self.engine.latencies))},
            "sources": {k: a.health.as_dict() for k, a in self.adapters.items()},
            "queue_depth": self.queue.qsize(), "replays": list_replays(),
        }


_ANALYZE_ENGINE: Engine | None = None


def _fresh_engine(base: Engine) -> Engine:
    """A sibling engine sharing models but with its own dedup/novelty state (for the Analyze tab)."""
    global _ANALYZE_ENGINE
    if _ANALYZE_ENGINE is None:
        _ANALYZE_ENGINE = Engine(sentiment=base.sentiment, events=base.events, impact=base.impact, resolver=base.resolver)
    from trident.pipeline.dedup import Deduper
    from trident.pipeline.novelty import NoveltyTracker

    _ANALYZE_ENGINE.dedup = Deduper()
    _ANALYZE_ENGINE.novelty = NoveltyTracker()
    return _ANALYZE_ENGINE
