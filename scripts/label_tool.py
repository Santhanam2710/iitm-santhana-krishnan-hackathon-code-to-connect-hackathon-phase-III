"""Label REAL headlines fast, with the engine's prediction as the default (one keypress to accept).

    python scripts/label_tool.py                # uses data/raw/{rss,gdelt,newsapi}/*.jsonl
    python scripts/label_tool.py --target 300

Keys per item:  Enter = accept suggestion   1-9 = event class   p/n/u = polarity   s = skip   q = quit
Saves to data/gold/headlines_gold_real.jsonl (resumable). This becomes the 'gold_real' evaluation set
used by nlp_eval, train_events and the champion/challenger registry. Never train on it.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from trident.nlp.entities import EntityResolver  # noqa: E402
from trident.nlp.events import EventClassifier  # noqa: E402
from trident.nlp.sentiment import SentimentScorer  # noqa: E402
from trident.signals.schema import EVENT_LABELS  # noqa: E402

OUT = ROOT / "data" / "gold" / "headlines_gold_real.jsonl"


def candidates() -> list[dict]:
    rows = []
    for f in sorted((ROOT / "data" / "raw").glob("*/*.jsonl")):
        if "gdelt/events" in str(f):
            continue
        for line in f.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            if r.get("text") and len(r["text"]) > 25:
                rows.append({"text": r["text"], "source": f.parent.name, "url": r.get("url")})
    seen, out = set(), []
    for r in rows:
        k = r["text"].lower()[:80]
        if k not in seen:
            seen.add(k)
            out.append(r)
    random.Random(7).shuffle(out)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=300)
    a = ap.parse_args()
    done = {json.loads(l)["text"] for l in OUT.read_text().splitlines()} if OUT.exists() else set()
    cands = [c for c in candidates() if c["text"] not in done]
    if not cands:
        print("No candidates. Run: python scripts/fetch_data.py --rss --gdelt")
        return
    ev, sent, res = EventClassifier(), SentimentScorer(), EntityResolver()
    print(__doc__)
    print("Classes: " + "  ".join(f"{i + 1}={c}" for i, c in enumerate(EVENT_LABELS)))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "a", encoding="utf-8") as fh:
        for c in cands:
            if len(done) >= a.target:
                break
            e = ev.classify(c["text"]).label
            s = sent.score(c["text"]).score
            pol = "pos" if s > 0.15 else ("neg" if s < -0.15 else "neu")
            tick = [m.ticker for m in res.resolve(c["text"]).matches]
            print(f"\n[{len(done) + 1}/{a.target}] {c['text']}\n   suggestion: {e} | {pol} | {tick}")
            k = input("   > ").strip().lower()
            if k == "q":
                break
            if k == "s":
                continue
            for ch in k:
                if ch.isdigit() and 1 <= int(ch) <= len(EVENT_LABELS):
                    e = EVENT_LABELS[int(ch) - 1]
                elif ch in "pnu":
                    pol = {"p": "pos", "n": "neg", "u": "neu"}[ch]
            fh.write(json.dumps({"text": c["text"], "event": e, "polarity": pol, "tickers": tick, "source": c["source"], "url": c["url"],
                                 "provenance": "real-labeled"}) + "\n")
            fh.flush()
            done.add(c["text"])
    print(f"\n{len(done)} labeled -> {OUT}")


if __name__ == "__main__":
    main()
