"""Train the sklearn sentiment backend (TF-IDF word+char n-grams -> logistic regression).

Training data, in order of preference: Financial PhraseBank + twitter-financial-news-sentiment
(real, if fetched), plus synthetic augmentation capped at --syn-share (eq. D3) with prior-
correcting weights (eq. D4). Evaluation always uses REAL held-out data when available, and
also reports the authored gold_v0 set as a regression check.

  python -m trident.training.train_sentiment [--syn-share 0.3] [--out models/sentiment_sklearn.joblib]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline

from trident.config import MODELS_DIR, resolve
from trident.training import datasets as ds

LABELS = ["negative", "neutral", "positive"]


def build_pipeline() -> Pipeline:
    feats = FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True, lowercase=True)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True)),
    ])
    return Pipeline([("features", feats), ("clf", LogisticRegression(max_iter=3000, C=4.0, class_weight="balanced"))])


def mix_with_synthetic(real: pd.DataFrame, syn: pd.DataFrame, share: float, seed: int = 0) -> tuple[pd.DataFrame, np.ndarray]:
    """Cap synthetic share at ``share`` and reweight so the class prior matches the real data (D3, D4)."""
    if share <= 0 or syn.empty:
        return real, np.ones(len(real))
    n_syn = int(len(real) * share / (1 - share)) if len(real) else len(syn)
    syn = syn.sample(min(n_syn, len(syn)), random_state=seed)
    mix = pd.concat([real, syn], ignore_index=True)
    p_real = real["label"].value_counts(normalize=True) if len(real) else mix["label"].value_counts(normalize=True)
    p_mix = mix["label"].value_counts(normalize=True)
    w = mix["label"].map(lambda l: p_real.get(l, 1e-3) / p_mix[l]).to_numpy()
    return mix, w


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--syn-share", type=float, default=0.3)
    ap.add_argument("--out", default=str(MODELS_DIR / "candidates" / "sentiment_sklearn.joblib"))
    ap.add_argument("--force", action="store_true", help="promote even if it does not beat the champion")
    ap.add_argument("--metrics", default="docs/results/train_sentiment.json")
    a = ap.parse_args()

    real_parts = [d for d in (ds.phrasebank("AllAgree"), ds.twitter_sentiment("train")) if d is not None]
    real = pd.concat(real_parts, ignore_index=True) if real_parts else pd.DataFrame(columns=["text", "label", "source"])
    real = real[real["label"].isin(LABELS)]
    syn = ds.synthetic("sentiment")
    if len(real):
        tr, te = train_test_split(real, test_size=0.2, random_state=42, stratify=real["label"])
        eval_name = "+".join(sorted(real["source"].unique())) + " (20% held out)"
    else:
        tr, te = real, None
        eval_name = None
    train_df, w = mix_with_synthetic(tr, syn, a.syn_share) if len(tr) else (syn, np.ones(len(syn)))
    pipe = build_pipeline()
    pipe.fit(train_df["text"], train_df["label"], clf__sample_weight=w)
    Path(resolve(a.out)).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, resolve(a.out))

    out: dict = {"train_rows": int(len(train_df)), "real_rows": int(len(tr)), "synthetic_share_cap": a.syn_share,
                 "train_sources": sorted(train_df["source"].unique().tolist())}
    if te is not None:
        pred = pipe.predict(te["text"])
        out["real_heldout"] = {"name": eval_name, "n": int(len(te)), "macro_f1": float(f1_score(te["label"], pred, average="macro")),
                               "report": classification_report(te["label"], pred, output_dict=True, zero_division=0)}
    g = ds.gold_v0("sentiment")
    pred = pipe.predict(g["text"])
    out["gold_v0"] = {"n": int(len(g)), "macro_f1": float(f1_score(g["label"], pred, average="macro"))}
    # champion/challenger: compare with the lexicon on the same evaluation set
    from trident.eval.nlp_eval import to_label
    from trident.nlp.lexicon import score_span
    from trident.training.registry import decide

    ev_df, ev_name = (te, eval_name) if te is not None else (g, "gold_v0 (authored)")
    cand = float(f1_score(ev_df["label"], pipe.predict(ev_df["text"]), average="macro"))
    base = float(f1_score(ev_df["label"], [to_label(score_span(t).score) for t in ev_df["text"]], average="macro"))
    out["registry"] = decide("sentiment_sklearn", Path(resolve(a.out)), MODELS_DIR / "sentiment_sklearn.joblib", cand, "lexicon", base, ev_name, a.force)
    Path(resolve(a.metrics)).parent.mkdir(parents=True, exist_ok=True)
    Path(resolve(a.metrics)).write_text(json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out.items() if k != "real_heldout"} | ({"real_heldout_f1": out["real_heldout"]["macro_f1"]} if te is not None else {}), indent=2))


if __name__ == "__main__":
    main()
