"""Behavioral (CheckList-style) tests for the engine: minimum functionality, invariance and
directional expectations. Each test returns pass counts; the suite also runs in pytest.

  python -m trident.eval.checklist
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Callable

from trident.nlp.entities import EntityResolver
from trident.nlp.events import EventClassifier
from trident.nlp.sentiment import SentimentScorer

COMPANIES = ["Apple", "Microsoft", "Nvidia", "Tesla", "Boeing", "Pfizer", "Walmart", "JPMorgan"]


@dataclass
class Test:
    name: str
    kind: str  # MFT | INV | DIR
    cases: list
    check: Callable


def build(scorer: SentimentScorer, events: EventClassifier, resolver: EntityResolver) -> list[Test]:
    s = lambda t: scorer.score(t).score
    return [
        Test("negation flips sign", "DIR", [(f"{c} beats expectations", f"{c} did not beat expectations") for c in COMPANIES],
             lambda a, b: s(a) > 0.1 and s(b) < s(a) - 0.2),
        Test("delta phrase is positive", "MFT", [f"{c} losses narrowed in the quarter" for c in COMPANIES], lambda t: s(t) > 0),
        Test("entity swap keeps sentiment", "INV", [(f"{a} shares plunge after profit warning", f"{b} shares plunge after profit warning") for a, b in zip(COMPANIES, COMPANIES[1:])],
             lambda a, b: abs(s(a) - s(b)) < 0.05),
        Test("entity swap keeps event", "INV", [(f"{a} agrees to acquire rival in $5 billion deal", f"{b} agrees to acquire rival in $5 billion deal") for a, b in zip(COMPANIES, COMPANIES[1:])],
             lambda a, b: events.classify(a).label == events.classify(b).label),
        Test("downgrade is a credit event", "MFT", [f"{c} downgraded to junk by rating agency" for c in COMPANIES], lambda t: events.classify(t).label == "Credit Event"),
        Test("sanctions are geopolitical", "MFT", ["US imposes sanctions on Iran oil exports", "EU expands sanctions over invasion", "New tariffs announced on Chinese imports"],
             lambda t: events.classify(t).label == "Geopolitical"),
        Test("intensity raises magnitude", "DIR", [(f"{c} shares fall", f"{c} shares plunge in record collapse") for c in COMPANIES], lambda a, b: s(b) < s(a)),
        Test("fruit is not Apple Inc.", "MFT", ["I ate an apple pie", "apple juice prices at my grocery", "Apple crumble recipe for the weekend"],
             lambda t: "AAPL" not in {m.ticker for m in resolver.resolve(t).matches}),
        Test("cashtag resolves", "MFT", ["$NVDA ripping today", "$TSLA earnings tonight", "$JPM breaking out"], lambda t: len(resolver.resolve(t).matches) == 1),
        Test("lower-case social names resolve", "MFT", ["credit suisse cds going vertical", "nvidia shares halted lol", "tesla stock crash incoming"],
             lambda t: len(resolver.resolve(t).matches) == 1),
    ]


def run() -> dict:
    tests = build(SentimentScorer(), EventClassifier(), EntityResolver())
    out = {}
    total = passed = 0
    for t in tests:
        ok = 0
        fails = []
        for c in t.cases:
            r = t.check(*c) if isinstance(c, tuple) else t.check(c)
            ok += bool(r)
            if not r:
                fails.append(c)
        out[t.name] = {"kind": t.kind, "passed": ok, "total": len(t.cases), "failures": fails[:3]}
        total += len(t.cases)
        passed += ok
    out["_summary"] = {"passed": passed, "total": total, "rate": passed / total}
    return out


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
