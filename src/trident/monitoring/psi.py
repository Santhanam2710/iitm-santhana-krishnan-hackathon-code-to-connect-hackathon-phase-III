"""Drift monitoring (red-team item 15): population stability index of score distributions.

PSI = sum (a_i - e_i) ln(a_i / e_i) over bins. Rule of thumb: < 0.1 stable, 0.1-0.25 watch,
> 0.25 investigate (retrain or recalibrate). Compare a reference window of signals with a
recent one, for sentiment, impact and the event-class mix.

  python -m trident.monitoring.psi --ref data/runtime/signals_hist.jsonl --cur data/runtime/signals.jsonl
"""
from __future__ import annotations

import argparse
import json
from collections import Counter

import numpy as np

from trident.config import resolve
from trident.utils.common import read_jsonl


def psi(expected: np.ndarray, actual: np.ndarray, bins: np.ndarray) -> float:
    e = np.histogram(expected, bins)[0] / max(len(expected), 1)
    a = np.histogram(actual, bins)[0] / max(len(actual), 1)
    e, a = np.clip(e, 1e-4, None), np.clip(a, 1e-4, None)
    return float(np.sum((a - e) * np.log(a / e)))


def psi_categorical(expected: list[str], actual: list[str]) -> float:
    cats = sorted(set(expected) | set(actual))
    ce, ca = Counter(expected), Counter(actual)
    e = np.clip(np.array([ce[c] / len(expected) for c in cats]), 1e-4, None)
    a = np.clip(np.array([ca[c] / len(actual) for c in cats]), 1e-4, None)
    return float(np.sum((a - e) * np.log(a / e)))


def report(ref: list[dict], cur: list[dict]) -> dict:
    out = {
        "sentiment_psi": psi(np.array([s["sentiment_score"] for s in ref]), np.array([s["sentiment_score"] for s in cur]), np.linspace(-1, 1, 11)),
        "impact_psi": psi(np.array([s["impact_score"] for s in ref]), np.array([s["impact_score"] for s in cur]), np.arange(0.5, 11)),
        "event_mix_psi": psi_categorical([s["event_classification"]["label"] for s in ref], [s["event_classification"]["label"] for s in cur]),
    }
    worst = max(out.values())
    out["status"] = "stable" if worst < 0.1 else ("watch" if worst < 0.25 else "investigate")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--cur", required=True)
    a = ap.parse_args()
    print(json.dumps(report(list(read_jsonl(resolve(a.ref))), list(read_jsonl(resolve(a.cur)))), indent=2))


if __name__ == "__main__":
    main()
