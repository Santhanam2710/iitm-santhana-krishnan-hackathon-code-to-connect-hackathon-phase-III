"""FastAPI service: REST + WebSocket + SSE, and the static dashboard at ``/``.

Run:  python -m trident demo            (replay mode, no keys)
      python -m trident serve --live    (live sources)
OpenAPI docs: http://localhost:8000/docs
"""
from __future__ import annotations

import asyncio
import json
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal, Optional

from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from trident import __version__
from trident.config import get_config
from trident.pipeline.runtime import Runtime, stress_summary

STATIC = Path(__file__).resolve().parents[1] / "dashboard" / "static"
RT: Runtime | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global RT
    RT = Runtime(mode=os.environ.get("TRIDENT_MODE", "replay"), replay=os.environ.get("TRIDENT_REPLAY") or None,
                 speed=float(os.environ.get("TRIDENT_SPEED", "600")))
    await RT.start()
    yield
    await RT.stop()


app = FastAPI(title="TRIDENT Risk Engine API", version=__version__, lifespan=lifespan,
              description="Structured risk signals from unstructured text, plus a tactical rebalancer and a wholesale stress tester.")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


def rt() -> Runtime:
    if RT is None:
        raise HTTPException(503, "runtime not started")
    return RT


class AnalyzeIn(BaseModel):
    text: str = Field(min_length=3, max_length=2000)
    source_type: Literal["newswire", "filing", "social", "gdelt"] = "newswire"
    publish: bool = False


class StressIn(BaseModel):
    event_class: Optional[str] = "Credit Event"
    impact: float = Field(8, ge=1, le=10)
    preset: Optional[str] = None
    origin_sector: Optional[str] = None
    custom_shock: Optional[dict[str, Any]] = None


class ReplayIn(BaseModel):
    name: str
    speed: float = Field(600, gt=0, le=100000)


class RebalancerParams(BaseModel):
    lam: Optional[float] = Field(None, alias="lambda", ge=0, le=5)
    w_max: Optional[float] = Field(None, ge=0.07, le=0.5)
    band: Optional[float] = Field(None, ge=0, le=0.05)
    turnover_max: Optional[float] = Field(None, ge=0, le=1)
    method: Optional[Literal["tilt", "optimizer"]] = None


@app.get("/healthz")
def healthz() -> dict:
    return {"ok": True, "version": __version__}


@app.get("/api/status")
def status() -> dict:
    return rt().status()


@app.get("/metrics", response_class=PlainTextResponse)
def metrics() -> str:
    s = rt().status()
    lines = [f'trident_signals_total {s["counters"]["signals"]}', f'trident_docs_total {s["counters"]["docs"]}',
             f'trident_dropped_duplicate_total {s["counters"]["dropped_duplicate"]}',
             f'trident_engine_latency_ms{{quantile="0.5"}} {s["latency_ms"]["p50"]:.4f}',
             f'trident_engine_latency_ms{{quantile="0.95"}} {s["latency_ms"]["p95"]:.4f}', f'trident_queue_depth {s["queue_depth"]}']
    return "\n".join(lines) + "\n"


@app.get("/api/signals")
def signals(ticker: Optional[str] = None, since: Optional[str] = None, until: Optional[str] = None, event: Optional[str] = None,
            min_impact: Optional[int] = Query(None, ge=1, le=10), include_synthetic: bool = True, compact: bool = False,
            limit: int = Query(200, ge=1, le=5000)) -> list[dict]:
    rows = rt().store.query(ticker, since, until, event, min_impact, include_synthetic, limit)
    if compact:
        return [{"signal_id": r["signal_id"], "ticker": r["entity"]["ticker"], "ts": r["ts_published"], "sentiment_score": r["sentiment_score"],
                 "event_classification": r["event_classification"]["label"], "impact_score": r["impact_score"]} for r in rows]
    return rows


@app.get("/api/signals/{signal_id}")
def signal(signal_id: str) -> dict:
    s = rt().store.get(signal_id)
    if not s:
        raise HTTPException(404, "signal not found")
    return s


@app.post("/api/analyze")
async def analyze(body: AnalyzeIn) -> dict:
    return await rt().analyze(body.text, body.source_type, body.publish)


@app.get("/api/rebalancer")
def rebalancer() -> dict:
    return rt().rebalancer.snapshot()


@app.get("/api/rebalancer/history")
def rebalancer_history(limit: int = Query(300, ge=1, le=2000)) -> list[dict]:
    h = list(rt().rebalancer.history)[-limit:]
    return [{k: v for k, v in r.items() if k != "scores" or i == len(h) - 1} for i, r in enumerate(h)]


@app.post("/api/rebalancer/params")
def rebalancer_params(p: RebalancerParams) -> dict:
    r = rt().rebalancer
    r.set_params(**{"lambda": p.lam, "w_max": p.w_max, "band": p.band, "turnover_max": p.turnover_max, "method": p.method})
    if rt().sim_time:
        r.rebalance(rt().sim_time)
    return r.snapshot()


@app.get("/api/stress/latest")
def stress_latest() -> dict:
    h = rt().stress.history
    return stress_summary(h[-1]) if h else {}


@app.get("/api/stress/history")
def stress_history() -> list[dict]:
    return [{k: r.get(k) for k in ("run_id", "ts", "label", "pnl", "cet1_before", "cet1_after", "event_class", "impact")} for r in rt().stress.history]


@app.get("/api/stress/decisions")
def stress_decisions(limit: int = 50) -> list[dict]:
    return list(rt().stress.decisions)[-limit:]


@app.post("/api/stress/run")
def stress_run(body: StressIn) -> dict:
    svc = rt().stress
    if body.custom_shock:
        res = svc.engine.run(body.custom_shock, "custom")
        res["reverse"] = svc.engine.reverse_stress(body.custom_shock)
        return stress_summary(res)
    trigger = {"entity": {"sector": body.origin_sector}, "ts_published": None, "signal_id": None} if body.origin_sector else None
    if body.preset:
        res = svc.run_for(body.event_class or "Credit Event", 10, preset=body.preset)
    else:
        res = svc.run_for(body.event_class or "Credit Event", body.impact, trigger=None)
        if trigger:
            shock, _ = svc.engine.scenario(body.event_class, body.impact)
            res = svc.engine.run(svc.engine.contagion(shock, body.origin_sector), f"{body.event_class}, impact {body.impact:g}, contagion from {body.origin_sector}")
            res["reverse"] = svc.engine.reverse_stress(get_config().scenarios["scenarios"][body.event_class])
    return stress_summary(res)


@app.get("/api/stress/presets")
def stress_presets() -> dict:
    sc = get_config().scenarios
    return {"presets": {k: v["label"] for k, v in sc["presets"].items()}, "classes": list(sc["scenarios"]), "trigger": sc["trigger"]}


@app.get("/api/book")
def book() -> list[dict]:
    return [p.__dict__ for p in rt().stress.engine.book]


@app.post("/api/replay/start")
async def replay_start(body: ReplayIn) -> dict:
    r = rt()
    await r.stop_replay()
    r.reset_state()
    return await r.start_replay(body.name, body.speed)


@app.post("/api/replay/stop")
async def replay_stop() -> dict:
    await rt().stop_replay()
    return {"stopped": True}


@app.post("/api/synthetic/inject")
async def synthetic_inject(scenario: Literal["credit", "geo", "macro"] = "credit", n: int = Query(12, ge=1, le=200)) -> dict:
    return {"injected": await rt().inject_synthetic(n, scenario), "flag": "is_synthetic=true"}


@app.get("/api/rejected")
def rejected() -> list[dict]:
    return list(rt().rejected)[-50:]


@app.get("/api/recent")
def recent(limit: int = 100) -> list[dict]:
    return list(rt().recent)[-limit:]


@app.get("/api/results")
def results() -> dict:
    p = Path(get_config().settings.get("results_path", "docs/results/results.json"))
    from trident.config import resolve

    f = resolve(p)
    return json.loads(f.read_text()) if f.exists() else {}


@app.websocket("/ws")
async def ws(websocket: WebSocket) -> None:
    await websocket.accept()
    r = rt()
    r.hub.clients.add(websocket)
    try:
        await websocket.send_text(json.dumps({"type": "hello", "data": {"status": r.status(), "recent": list(r.recent)[-60:],
                                                                         "weights": r.rebalancer.history[-1] if r.rebalancer.history else None,
                                                                         "stress": stress_summary(r.stress.history[-1]) if r.stress.history else None}}, default=str))
        while True:
            await websocket.receive_text()  # keepalive / ignore client messages
    except WebSocketDisconnect:
        pass
    finally:
        r.hub.clients.discard(websocket)


@app.get("/api/stream")
async def sse() -> StreamingResponse:
    """Server-Sent Events alternative to the WebSocket for simple consumers (curl -N)."""
    r = rt()

    async def gen():
        async for msg in r.bus.subscribe("signals"):
            yield f"data: {json.dumps({k: msg[k] for k in ('signal_id', 'ts_published', 'sentiment_score', 'impact_score')} | {'ticker': msg['entity']['ticker'], 'event': msg['event_classification']['label']})}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (STATIC / "index.html").read_text(encoding="utf-8")


app.mount("/static", StaticFiles(directory=STATIC), name="static")
