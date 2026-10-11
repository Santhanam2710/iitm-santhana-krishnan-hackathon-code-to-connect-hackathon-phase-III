"""Latency and throughput benchmark of the engine (ingest -> signal), excluding source lag.

  python -m trident bench --n 5000
Reports per-document engine latency percentiles (from signal latency_ms) and batch throughput,
plus Module A rebalance and Module B full revaluation timings.
"""
from __future__ import annotations

import platform
import random
import time

import numpy as np

from trident.ingest.synthetic import TemplateGenerator
from trident.modules.rebalancer.service import Rebalancer
from trident.modules.stress.service import StressService
from trident.pipeline.engine import Engine
from trident.pipeline.normalize import make_doc
from trident.utils.common import parse_ts


def run(n: int = 3000, batch: int = 16) -> dict:
    gen = TemplateGenerator(seed=7)
    rows = gen.generate(n, hours=72)
    docs = [make_doc(r["text"], r["source_type"], f"bench:{i % 7}", r["ts"]) for i, r in enumerate(rows)]
    eng = Engine()
    eng.warmup()
    t0 = time.perf_counter()
    sigs = []
    for i in range(0, len(docs), batch):
        for o in eng.process_batch(docs[i : i + batch]):
            sigs.extend(o.signals)
    wall = time.perf_counter() - t0
    lat = np.array([s.latency_ms["engine_total"] for s in sigs])
    reb, st = Rebalancer(), StressService()
    tA, tB = [], []
    for s in sigs[:500]:
        d = s.model_dump()
        reb.on_signal(d)
        t = time.perf_counter()
        reb.rebalance(parse_ts(d["ts_published"]))
        tA.append((time.perf_counter() - t) * 1e3)
    for _ in range(50):
        t = time.perf_counter()
        st.engine.run(st.engine.scenario("Credit Event", random.choice([8, 9, 10]))[0])
        tB.append((time.perf_counter() - t) * 1e3)
    return {
        "documents": n, "signals": len(sigs), "batch_size": batch, "backend": eng.model_versions,
        "engine_ms": {"p50": float(np.percentile(lat, 50)), "p95": float(np.percentile(lat, 95)), "p99": float(np.percentile(lat, 99)), "mean": float(lat.mean())},
        "throughput_docs_per_s": n / wall,
        "module_a_rebalance_ms": {"p50": float(np.percentile(tA, 50)), "p95": float(np.percentile(tA, 95))},
        "module_b_full_revaluation_ms": {"p50": float(np.percentile(tB, 50)), "p95": float(np.percentile(tB, 95))},
        "machine": f"{platform.processor() or platform.machine()} · {platform.system()} · Python {platform.python_version()}",
    }
