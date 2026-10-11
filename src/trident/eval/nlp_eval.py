"""NLP evaluation against named baselines.

Sentiment: TRIDENT lexicon, VADER (general-purpose baseline), TextBlob-style naive word
count, the sklearn model and the ensemble, on every available labeled set.
Events: keyword rules vs the trained model vs the hybrid.
Entities: ticker-level precision/recall on the gold set, including disambiguation traps.

  python -m trident.eval.nlp_eval --out docs/results/nlp_eval.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score

from trident.config import resolve
from trident.nlp.entities import EntityResolver
from trident.nlp.events import EventClassifier
from trident.nlp.lexicon import score_span
from trident.nlp.sentiment import LexiconBackend, SentimentScorer, load_backend
from trident.signals.schema import EVENT_LABELS
from trident.training import datasets as ds

LAB3 = ["negative", "neutral", "positive"]


def to_label(s: float, band: float = 0.15) -> str:
    return "positive" if s > band else ("negative" if s < -band else "neutral")


def vader_scores(texts: list[str]) -> list[float] | None:
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    except Exception:
        try:
            from nltk.sentiment import SentimentIntensityAnalyzer  # type: ignore
        except Exception:
            return None
    sia = SentimentIntensityAnalyzer()
    return [sia.polarity_scores(t)["compound"] for t in texts]


NAIVE_POS = {"good", "great", "up", "gain", "gains", "rise", "rises", "growth", "strong", "best", "win", "wins", "record", "high"}
NAIVE_NEG = {"bad", "down", "loss", "losses", "fall", "falls", "weak", "worst", "drop", "drops", "low", "cut", "cuts", "risk"}


def naive_scores(texts: list[str]) -> list[float]:
    out = []
    for t in texts:
        w = t.lower().split()
        p, n = sum(x in NAIVE_POS for x in w), sum(x in NAIVE_NEG for x in w)
        out.append((p - n) / max(p + n, 1))
    return out


def f1(y, p) -> float:
    return float(f1_score(y, p, average="macro", labels=LAB3, zero_division=0))


def eval_sentiment(df: pd.DataFrame) -> dict:
    texts, y = df["text"].tolist(), df["label"].tolist()
    res: dict = {"n": len(df)}
    lex = [score_span(t).score for t in texts]
    res["trident_lexicon"] = f1(y, [to_label(s) for s in lex])
    v = vader_scores(texts)
    if v is not None:
        res["vader_baseline"] = f1(y, [to_label(s, 0.05) for s in v])
    res["naive_wordcount_baseline"] = f1(y, [to_label(s, 0.0) for s in naive_scores(texts)])
    backend = load_backend("auto")
    if not isinstance(backend, LexiconBackend):
        p = backend.predict(texts)
        res[f"model:{backend.name}"] = f1(y, [LAB3[i] for i in p.argmax(1)])
        ens = SentimentScorer(backend).score_many(texts)
        res["trident_ensemble"] = f1(y, [to_label(r.score) for r in ens])
        best = res["trident_ensemble"]
    else:
        best = res["trident_lexicon"]
    res["trident_best"] = best
    if "hard" in df:
        hard = df[df["hard"].astype(str).str.len() > 0]
        if len(hard):
            res["hard_cases_n"] = int(len(hard))
            res["hard_cases_lexicon_acc"] = float(np.mean([to_label(score_span(t).score) == l for t, l in zip(hard["text"], hard["label"])]))
            if v is not None:
                vh = vader_scores(hard["text"].tolist())
                res["hard_cases_vader_acc"] = float(np.mean([to_label(s, 0.05) == l for s, l in zip(vh, hard["label"])]))
    return res


def eval_events(df: pd.DataFrame) -> dict:
    rules = EventClassifier(backend="rules")
    hybrid = EventClassifier(backend="auto")
    y = df["label"].tolist()
    pr = [r.label for r in rules.classify_many(df["text"].tolist())]
    ph = [r.label for r in hybrid.classify_many(df["text"].tolist())]
    labels = [l for l in EVENT_LABELS if l in set(y)]
    cm = confusion_matrix(y, ph, labels=labels)
    return {"n": len(df), "rules_macro_f1": float(f1_score(y, pr, average="macro", labels=labels, zero_division=0)),
            f"{hybrid.name}_macro_f1": float(f1_score(y, ph, average="macro", labels=labels, zero_division=0)),
            "best_macro_f1": float(max(f1_score(y, pr, average="macro", labels=labels, zero_division=0), f1_score(y, ph, average="macro", labels=labels, zero_division=0))),
            "accuracy": float(np.mean(np.array(ph) == np.array(y))), "labels": labels, "confusion": cm.tolist(), "model": hybrid.name}


def eval_entities() -> dict:
    rows = ds.gold_v0("raw").to_dict(orient="records")
    res = EntityResolver()
    tp = fp = fn = 0
    trap_ok = trap_n = 0
    for r in rows:
        want = {t for t in r["tickers"] if not t.startswith(("MARKET", "SECTOR:"))}
        got = {m.ticker for m in res.resolve(r["text"]).matches}
        tp += len(want & got)
        fp += len(got - want)
        fn += len(want - got)
        if r["hard"] == "ambiguous-entity":
            trap_n += 1
            trap_ok += int(not got)
    p = tp / max(tp + fp, 1)
    rc = tp / max(tp + fn, 1)
    return {"precision": p, "recall": rc, "f1": 2 * p * rc / max(p + rc, 1e-9), "ambiguity_traps_correct": f"{trap_ok}/{trap_n}", "tp": tp, "fp": fp, "fn": fn}


def run() -> dict:
    out: dict = {"sentiment": {}, "events": {}, "entities": eval_entities(), "provenance": {}}
    sets = {"gold_v0": ds.gold_v0("sentiment")}
    for name, loader in (("phrasebank_allagree", lambda: ds.phrasebank("AllAgree")), ("twitter_sentiment_valid", lambda: ds.twitter_sentiment("valid")),
                         ("gold_real", lambda: ds.gold_real("sentiment"))):
        d = loader()
        if d is not None:
            sets[name] = d
    for name, d in sets.items():
        out["sentiment"][name] = eval_sentiment(d)
    ev_sets = {"gold_v0": ds.gold_v0("event")}
    for name, loader in (("twitter_topic_valid", lambda: ds.twitter_topic("valid")), ("gold_real", lambda: ds.gold_real("event"))):
        d = loader()
        if d is not None:
            ev_sets[name] = d
    for name, d in ev_sets.items():
        out["events"][name] = eval_events(d)
    out["provenance"] = {k: ("authored fictional headlines (regression/smoke benchmark)" if k == "gold_v0" else "real labeled data") for k in set(sets) | set(ev_sets)}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/results/nlp_eval.json")
    a = ap.parse_args()
    res = run()
    Path(resolve(a.out)).parent.mkdir(parents=True, exist_ok=True)
    Path(resolve(a.out)).write_text(json.dumps(res, indent=2))
    print(json.dumps({k: v for k, v in res.items() if k != "events"}, indent=2))
    for k, v in res["events"].items():
        print("events", k, {kk: vv for kk, vv in v.items() if kk not in ("confusion", "labels")})


if __name__ == "__main__":
    main()
