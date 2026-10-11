"""Adversarial validation (eq. D5): can a classifier tell real text from synthetic?

AUC near 0.5 means the synthetic generator has no obvious style tell. AUC well above 0.65
means it does; the top features show what to fix. "Real" is the best available real corpus
(PhraseBank / tweets / your gold_real / replay text), "synthetic" is data/synthetic.

  python -m trident.eval.adversarial
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_auc_score

from trident.training import datasets as ds


def run(max_n: int = 3000) -> dict:
    real_parts = [d for d in (ds.phrasebank("AllAgree"), ds.twitter_sentiment("train"), ds.gold_real("event")) if d is not None]
    if real_parts:
        real = pd.concat(real_parts)["text"].tolist()
        real_name = "real labeled corpora"
    else:
        real = ds.gold_v0("event")["text"].tolist() + ds.unlabeled_corpus()
        real_name = "authored gold_v0 + replay text (no real corpus fetched)"
    syn = ds.synthetic("event")["text"].tolist()
    rng = np.random.default_rng(0)
    n = min(len(real), len(syn), max_n)
    real = list(rng.choice(real, n, replace=False))
    syn = list(rng.choice(syn, n, replace=False))
    X, y = real + syn, np.array([0] * n + [1] * n)
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2)
    Z = vec.fit_transform(X)
    clf = LogisticRegression(max_iter=2000)
    p = cross_val_predict(clf, Z, y, cv=5, method="predict_proba")[:, 1]
    auc = float(roc_auc_score(y, p))
    clf.fit(Z, y)
    names = np.array(vec.get_feature_names_out())
    order = np.argsort(clf.coef_[0])
    return {"n_each": n, "real_source": real_name, "auc": auc, "target": "<= 0.65", "passes": auc <= 0.65,
            "synthetic_tells": names[order[-10:]][::-1].tolist(), "real_tells": names[order[:10]].tolist()}


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
