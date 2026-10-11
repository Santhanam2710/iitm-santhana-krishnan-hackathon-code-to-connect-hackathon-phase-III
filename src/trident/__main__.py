"""TRIDENT command line.

  python -m trident demo [--replay svb_2023-03] [--speed 600] [--port 8000]
  python -m trident serve [--live | --idle]
  python -m trident analyze "JPMorgan downgraded two notches as CRE losses mount"
  python -m trident replay svb_2023-03 [--out data/runtime/svb_signals.jsonl]
  python -m trident signals data/synthetic/market_news.jsonl --out data/runtime/signals_hist.jsonl
  python -m trident stress --event "Credit Event" --impact 9 | --preset covid_2020
  python -m trident backtest --signals ... --prices ...
  python -m trident bench [--n 5000]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

from trident.config import resolve


def _engine_over(items: list[dict], replay_id: str | None = None):
    from trident.pipeline.engine import Engine
    from trident.pipeline.normalize import make_doc

    eng = Engine()
    docs = [make_doc(r["text"], r.get("source_type", "newswire"), r.get("source_name", "file"), r.get("ts"), r.get("url"),
                     is_synthetic=bool(r.get("is_synthetic", False)), replay_id=replay_id,
                     meta={"ticker_hint": r.get("ticker_hint")} if r.get("ticker_hint") else None) for r in items]
    docs.sort(key=lambda d: d.ts_published)
    outs = []
    bs = 64
    for i in range(0, len(docs), bs):
        outs.extend(eng.process_batch(docs[i : i + bs]))
    return eng, outs


def cmd_demo(a: argparse.Namespace, mode: str = "replay") -> None:
    import uvicorn

    os.environ["TRIDENT_MODE"] = mode
    if getattr(a, "replay", None):
        os.environ["TRIDENT_REPLAY"] = a.replay
    os.environ["TRIDENT_SPEED"] = str(getattr(a, "speed", 600))
    print(f"\n  TRIDENT dashboard: http://localhost:{a.port}   (API docs: http://localhost:{a.port}/docs)\n")
    uvicorn.run("trident.api.app:app", host=a.host, port=a.port, log_level="warning")


def cmd_serve(a: argparse.Namespace) -> None:
    cmd_demo(a, "live" if a.live else ("idle" if a.idle else "replay"))


def cmd_analyze(a: argparse.Namespace) -> None:
    from trident.pipeline.engine import Engine
    from trident.pipeline.normalize import make_doc

    out = Engine().process(make_doc(a.text, a.source, "cli"))
    print(json.dumps({"dropped": out.dropped, "rejected_entities": out.rejected_entities,
                      "signals": [json.loads(s.model_dump_json()) for s in out.signals]}, indent=2))


def cmd_replay(a: argparse.Namespace) -> None:
    from trident.ingest.replay import load_replay
    from trident.modules.rebalancer.service import Rebalancer
    from trident.modules.stress.service import StressService
    from trident.utils.common import parse_ts, write_jsonl

    rows = load_replay(a.name)
    t0 = time.perf_counter()
    eng, outs = _engine_over(rows, replay_id=a.name)
    reb, st = Rebalancer(), StressService()
    sigs = [s.model_dump() for o in outs for s in o.signals]
    runs = []
    for s in sigs:
        reb.on_signal(s)
        res = st.on_signal(s)
        reb.rebalance(parse_ts(s["ts_published"]))
        if res:
            runs.append(res)
    elapsed = time.perf_counter() - t0
    if a.out:
        write_jsonl(resolve(a.out), sigs)
    print(f"replay {a.name}: {len(rows)} docs -> {len(sigs)} signals in {elapsed:.2f}s; dropped={Counter(o.dropped for o in outs if o.dropped)}")
    for d in st.decisions:
        if d["state"] in ("held", "triggered", "cooldown"):
            print(f"  {d['ts']}  {d['state']:9s} {d['label']:14s} I={d['impact']}  {d['ticker']:12s} {d['reason']}")
    for r in runs:
        print(f"  STRESS {r['ts']}: {r['label']}  P&L {r['pnl']:+.1f}m  CET1 {r['cet1_before']:.2%} -> {r['cet1_after']:.2%}")
    last = reb.history[-1]
    movers = sorted(last["weights"], key=lambda t: last["weights"][t] - last["benchmark"][t])
    print("  most underweight:", [(t, round((last['weights'][t] - last['benchmark'][t]) * 100, 2)) for t in movers[:3]])


def cmd_signals(a: argparse.Namespace) -> None:
    from trident.utils.common import read_jsonl, write_jsonl

    items = list(read_jsonl(resolve(a.input)))
    t0 = time.perf_counter()
    eng, outs = _engine_over(items)
    sigs = [s.model_dump() for o in outs for s in o.signals]
    write_jsonl(resolve(a.out), sigs)
    print(f"{len(items)} items -> {len(sigs)} signals in {time.perf_counter() - t0:.1f}s -> {a.out}")


def cmd_stress(a: argparse.Namespace) -> None:
    from trident.modules.stress.service import StressService

    s = StressService()
    r = s.run_for(a.event, a.impact, preset=a.preset)
    print(f"{r['label']}: P&L {r['pnl']:+.1f}m ({r['pnl_pct']:+.2%}); CET1 {r['cet1_before']:.2%} -> {r['cet1_after']:.2%}; "
          f"VaR99/10d {r['var_before']:.1f} -> {r['var_after']:.1f}m; reverse stress {r['reverse']}")
    for k, v in r["by_class"].items():
        print(f"  {k:12s} {v['dv']:+8.1f}m")


def cmd_backtest(a: argparse.Namespace) -> None:
    sys.argv = ["backtest", "--signals", a.signals, "--prices", a.prices, "--out", a.out]
    from trident.modules.rebalancer.backtest import main

    main()


def cmd_bench(a: argparse.Namespace) -> None:
    from trident.eval.latency_bench import run

    print(json.dumps(run(a.n), indent=2))


def main() -> None:
    ap = argparse.ArgumentParser(prog="trident", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("demo", "serve"):
        p = sub.add_parser(name)
        p.add_argument("--replay", default=None)
        p.add_argument("--speed", type=float, default=600)
        p.add_argument("--host", default="0.0.0.0")
        p.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8000)))
        if name == "serve":
            p.add_argument("--live", action="store_true")
            p.add_argument("--idle", action="store_true")
    p = sub.add_parser("analyze")
    p.add_argument("text")
    p.add_argument("--source", default="newswire", choices=["newswire", "filing", "social", "gdelt"])
    p = sub.add_parser("replay")
    p.add_argument("name")
    p.add_argument("--out", default=None)
    p = sub.add_parser("signals")
    p.add_argument("input")
    p.add_argument("--out", required=True)
    p = sub.add_parser("stress")
    p.add_argument("--event", default="Credit Event")
    p.add_argument("--impact", type=float, default=9)
    p.add_argument("--preset", default=None)
    p = sub.add_parser("backtest")
    p.add_argument("--signals", required=True)
    p.add_argument("--prices", required=True)
    p.add_argument("--out", default="docs/results/backtest.json")
    p = sub.add_parser("bench")
    p.add_argument("--n", type=int, default=3000)
    a = ap.parse_args()
    {"demo": cmd_demo, "serve": cmd_serve, "analyze": cmd_analyze, "replay": cmd_replay, "signals": cmd_signals,
     "stress": cmd_stress, "backtest": cmd_backtest, "bench": cmd_bench}[a.cmd](a)


if __name__ == "__main__":
    main()
