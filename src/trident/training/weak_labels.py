"""Weak supervision for the event classifier (Snorkel-style, no extra dependency).

Labeling functions (LFs) vote on each unlabeled headline; votes are combined with LF
accuracy weights estimated on the gold set (or uniform if no gold). Sources of signal:
  * taxonomy keywords (configs/taxonomy.yaml), one LF per class;
  * SEC 8-K Item codes (filings carry ``meta.item_labels``);
  * GDELT CAMEO roots (``weak_label`` from gdelt_events_latest);
  * regex LFs for patterns keywords miss (e.g. "agrees to buy", "per cent").
Output: data/synthetic/weak_labeled.jsonl with text, label, confidence, votes.

  python -m trident.training.weak_labels [--min-conf 0.5]
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from typing import Callable

from trident.config import DATA_DIR
from trident.nlp.events import RulesBackend
from trident.training import datasets as ds

ABSTAIN = None
LF = Callable[[str], str | None]


def keyword_lfs() -> list[tuple[str, LF]]:
    rb = RulesBackend()
    out = []
    for cls, pat in rb.patterns.items():
        if pat is None:
            continue
        out.append((f"kw:{cls}", (lambda p, c: (lambda t: c if p.search(t) else ABSTAIN))(pat, cls)))
    return out


REGEX_LFS: list[tuple[str, re.Pattern, str]] = [
    ("re:buy_deal", re.compile(r"\b(agrees?|agreed|plans?|offers?) to (buy|acquire|purchase)\b", re.I), "Merger/Acquisition"),
    ("re:rating_action", re.compile(r"\b(cut|cuts|lowers?|slashes?) .{0,30}\b(rating|outlook)\b", re.I), "Credit Event"),
    ("re:central_bank", re.compile(r"\b(fed|ecb|boe|boj|rbi|central bank)\b.{0,40}\b(rates?|hike|cut|policy)\b", re.I), "Macroeconomic"),
    ("re:quarter_results", re.compile(r"\b(q[1-4]|quarter(ly)?|fiscal)\b.{0,40}\b(profit|revenue|eps|earnings|sales)\b", re.I), "Earnings"),
    ("re:legal", re.compile(r"\b(sues?|sued|lawsuit|probe|fined?|settle[sd]?)\b", re.I), "Regulatory/Legal"),
    ("re:launch", re.compile(r"\b(launch(es|ed)?|unveil(s|ed)?|introduc(es|ed))\b", re.I), "Product Launch"),
    ("re:conflict", re.compile(r"\b(war|missiles?|troops|sanctions?|tariffs?)\b", re.I), "Geopolitical"),
    ("re:ops", re.compile(r"\b(outage|breach|hack(ed)?|strike|recall(s|ed)?|explosion|fire at)\b", re.I), "Operational"),
]


def all_lfs() -> list[tuple[str, LF]]:
    lfs = keyword_lfs()
    for name, pat, cls in REGEX_LFS:
        lfs.append((name, (lambda p, c: (lambda t: c if p.search(t) else ABSTAIN))(pat, cls)))
    return lfs


def lf_accuracies(lfs: list[tuple[str, LF]]) -> dict[str, float]:
    gold = ds.gold_real("event")
    if gold is None:
        gold = ds.gold_v0("event")
    acc = {}
    for name, f in lfs:
        hits = [(f(t), y) for t, y in zip(gold["text"], gold["label"])]
        fired = [(p, y) for p, y in hits if p is not None]
        acc[name] = (sum(p == y for p, y in fired) + 1) / (len(fired) + 2)  # Laplace-smoothed
    return acc


def label(texts: list[str], min_conf: float = 0.5) -> list[dict]:
    lfs = all_lfs()
    acc = lf_accuracies(lfs)
    out = []
    for t in texts:
        votes: dict[str, float] = defaultdict(float)
        fired = []
        for name, f in lfs:
            v = f(t)
            if v is not None:
                w = max(acc[name] - 0.2, 0.05)
                votes[v] += w
                fired.append(name)
        if not votes:
            out.append({"text": t, "label": "Other", "confidence": 0.5, "votes": []})
            continue
        lab, score = max(votes.items(), key=lambda kv: kv[1])
        conf = score / sum(votes.values())
        if conf >= min_conf:
            out.append({"text": t, "label": lab, "confidence": round(conf, 3), "votes": fired})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-conf", type=float, default=0.5)
    a = ap.parse_args()
    texts = ds.unlabeled_corpus()
    rows = label(texts, a.min_conf)
    f = DATA_DIR / "synthetic" / "weak_labeled.jsonl"
    with open(f, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print(f"weak-labeled {len(rows)}/{len(texts)} headlines -> {f}", Counter(r["label"] for r in rows).most_common())


if __name__ == "__main__":
    main()
