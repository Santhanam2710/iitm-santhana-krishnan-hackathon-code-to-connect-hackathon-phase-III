"""Train the sklearn event classifier and evaluate it.

Training data: weak labels (training/weak_labels.py) + twitter-financial-news-topic mapped to
our taxonomy (if fetched) + synthetic headlines capped at --syn-share with prior-correcting
weights. The authored gold_v0 set and your real labeled set (gold_real) are held out and
never used for training.

  python -m trident.training.train_events [--syn-share 0.3]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, f1_score

from trident.config import DATA_DIR, MODELS_DIR, resolve
from trident.nlp.events import EventClassifier, RulesBackend
from trident.signals.schema import EVENT_LABELS
from trident.training import datasets as ds
from trident.training.train_sentiment import build_pipeline, mix_with_synthetic


def load_training(syn_share: float) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    parts = []
    weak = DATA_DIR / "synthetic" / "weak_labeled.jsonl"
    if weak.exists():
        w = pd.read_json(weak, lines=True)
        parts.append(pd.DataFrame({"text": w["text"], "label": w["label"], "source": "weak_labels"}))
    tt = ds.twitter_topic("train")
    if tt is not None:
        parts.append(tt)
    real = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["text", "label", "source"])
    real = real[real["label"].isin(EVENT_LABELS)]
    syn = ds.synthetic("event")
    if len(real) == 0:
        return syn, np.ones(len(syn)), ["synthetic"]
    mix, w = mix_with_synthetic(real, syn, syn_share)
    return mix, w, sorted(mix["source"].unique())


def evaluate(predict, df: pd.DataFrame) -> dict:
    pred = predict(list(df["text"]))
    return {"n": int(len(df)), "macro_f1": float(f1_score(df["label"], pred, average="macro", labels=list(EVENT_LABELS), zero_division=0)),
            "accuracy": float(np.mean(np.array(pred) == df["label"].to_numpy())),
            "report": classification_report(df["label"], pred, output_dict=True, zero_division=0)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--syn-share", type=float, default=0.3)
    ap.add_argument("--out", default=str(MODELS_DIR / "candidates" / "events_sklearn.joblib"))
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--metrics", default="docs/results/train_events.json")
    a = ap.parse_args()
    train_df, w, sources = load_training(a.syn_share)
    pipe = build_pipeline()
    pipe.fit(train_df["text"], train_df["label"], clf__sample_weight=w)
    Path(resolve(a.out)).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, resolve(a.out))

    rules = RulesBackend()
    rules_only = lambda xs: [EVENT_LABELS[int(np.argmax(rules.predict_one(x)[0]))] for x in xs]
    model_only = lambda xs: list(pipe.predict(xs))
    def hybrid_pred(xs):
        rp = np.vstack([rules.predict_one(x)[0] for x in xs])
        mp = pipe.predict_proba(xs)
        cls = list(pipe.classes_)
        full = np.zeros_like(rp)
        for j, c in enumerate(EVENT_LABELS):
            if c in cls:
                full[:, j] = mp[:, cls.index(c)]
        return [EVENT_LABELS[i] for i in (0.5 * rp + 0.5 * full).argmax(1)]
    out = {"train_rows": int(len(train_df)), "train_sources": sources, "synthetic_share_cap": a.syn_share, "eval": {}}
    sets = {"gold_v0 (authored)": ds.gold_v0("event")}
    if ds.gold_real("event") is not None:
        sets["gold_real (your labels)"] = ds.gold_real("event")
    if ds.twitter_topic("valid") is not None:
        sets["twitter_topic_valid (mapped)"] = ds.twitter_topic("valid")
    for name, df in sets.items():
        out["eval"][name] = {"rules": evaluate(rules_only, df), "sklearn": evaluate(model_only, df), "hybrid": evaluate(hybrid_pred, df)}
    from trident.training.registry import decide

    ev_name = next((n for n in out["eval"] if "real" in n or "twitter" in n), "gold_v0 (authored)")
    r = out["eval"][ev_name]
    out["registry"] = decide("events_sklearn", Path(resolve(a.out)), MODELS_DIR / "events_sklearn.joblib", r["hybrid"]["macro_f1"], "rules",
                             r["rules"]["macro_f1"], ev_name, a.force)
    print("registry:", out["registry"]["decision"], out["registry"]["candidate_score"], "vs", out["registry"]["bar"], out["registry"]["bar_score"])
    Path(resolve(a.metrics)).write_text(json.dumps(out, indent=2))
    for name, r in out["eval"].items():
        print(f"{name}: rules F1={r['rules']['macro_f1']:.3f}  sklearn F1={r['sklearn']['macro_f1']:.3f}  hybrid F1={r['hybrid']['macro_f1']:.3f}")


if __name__ == "__main__":
    main()
