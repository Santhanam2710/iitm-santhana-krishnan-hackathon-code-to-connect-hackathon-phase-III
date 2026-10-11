"""Calibrate the impact score against realized absolute abnormal returns (eq. E3, E4).

For each signal on an index name: day 0 = the session whose close-to-close return first
contains the publication time (published before 21:00 UTC -> same day, otherwise next day).
AR = r_i - (alpha + beta r_m) with alpha, beta from a 120-day window ending 10 days before;
the market return is the equal-weighted universe. Target = decile of |CAR[0,+1]| (1..10).
An ordinal model is fit on the earlier years and tested on the last year.

  python -m trident.training.train_impact --signals data/runtime/signals_hist.jsonl \
        --prices data/market/prices_sp15.csv [--test-from 2025-01-01]
"""
from __future__ import annotations

import argparse
import json
from datetime import time as dtime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from trident.config import MODELS_DIR, get_config, resolve
from trident.nlp.impact import FEATURES, OrdinalLogit, build_features
from trident.utils.common import parse_ts, read_jsonl


def event_study(signals: list[dict], prices: pd.DataFrame, est: int = 120, gap: int = 10) -> pd.DataFrame:
    rets = prices.pct_change().dropna(how="all")
    mkt = rets.mean(axis=1)
    idx = rets.index
    rows = []
    for s in signals:
        t = s["entity"]["ticker"]
        if t not in rets.columns:
            continue
        ts = parse_ts(s["ts_published"])
        d = pd.Timestamp(ts.date())
        if ts.time() >= dtime(21, 0):
            d += pd.Timedelta(days=1)
        pos = idx.searchsorted(d)
        if pos + 1 >= len(idx) or pos - gap - est < 0:
            continue
        win = slice(pos - gap - est, pos - gap)
        y, x = rets[t].iloc[win].to_numpy(), mkt.iloc[win].to_numpy()
        ok = np.isfinite(y) & np.isfinite(x)
        if ok.sum() < 60:
            continue
        beta, alpha = np.polyfit(x[ok], y[ok], 1)
        ar = rets[t].iloc[pos : pos + 2].to_numpy() - (alpha + beta * mkt.iloc[pos : pos + 2].to_numpy())
        rows.append({"signal_id": s["signal_id"], "ticker": t, "date": idx[pos], "car": float(ar.sum()), "abs_car": float(abs(ar.sum())),
                     "impact": s["impact_score"], "sentiment": s["sentiment_score"], "label": s["event_classification"]["label"],
                     "text": s.get("text", ""), "novelty": s["source"].get("novelty", 1.0), "reliability": s["source"]["reliability"]})
    return pd.DataFrame(rows)


def features_for(df: pd.DataFrame) -> np.ndarray:
    feats = [build_features(r.text, r.label, abs(r.sentiment), r.novelty, 0.0, r.reliability, False, 1).vector() for r in df.itertuples()]
    return np.vstack(feats)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--signals", required=True)
    ap.add_argument("--prices", required=True)
    ap.add_argument("--test-from", default="2025-01-01")
    ap.add_argument("--out", default=str(MODELS_DIR / "impact_model.json"))
    ap.add_argument("--metrics", default="docs/results/impact_calibration.json")
    ap.add_argument("--no-save", action="store_true", help="report only; do not install the model")
    a = ap.parse_args()
    prices = pd.read_csv(resolve(a.prices), parse_dates=["date"]).set_index("date").sort_index()
    sigs = [s for s in read_jsonl(resolve(a.signals)) if s["entity"].get("in_index")]
    es = event_study(sigs, prices)
    if len(es) < 50:
        raise SystemExit(f"only {len(es)} usable events; need >= 50")
    es["decile"] = pd.qcut(es["abs_car"].rank(method="first"), 10, labels=False) + 1
    X = features_for(es)
    test = (es["date"] >= pd.Timestamp(a.test_from)).to_numpy()
    model = OrdinalLogit(10).fit(X[~test], es["decile"].to_numpy()[~test])
    pred = model.predict_score(X)
    es["impact_calibrated"] = np.clip(np.round(pred), 1, 10)
    rho_def = spearmanr(es.loc[test, "impact"], es.loc[test, "abs_car"]).statistic
    rho_cal = spearmanr(pred[test], es.loc[test, "abs_car"]).statistic
    by_imp = es[test].groupby("impact_calibrated")["abs_car"].mean().mul(1e4).round(1).to_dict()
    res = {"events": int(len(es)), "train_events": int((~test).sum()), "test_events": int(test.sum()), "test_from": a.test_from,
           "spearman_default_impact_vs_abs_car": float(rho_def), "spearman_calibrated_vs_abs_car": float(rho_cal),
           "mean_abs_car_bps_by_calibrated_impact": {str(int(k)): v for k, v in by_imp.items()},
           "beta": dict(zip(FEATURES, np.round(model.beta, 4).tolist()))}
    if not a.no_save:
        Path(resolve(a.out)).write_text(json.dumps(model.to_json(), indent=2))
    Path(resolve(a.metrics)).parent.mkdir(parents=True, exist_ok=True)
    Path(resolve(a.metrics)).write_text(json.dumps(res, indent=2))
    es.to_csv(resolve(a.metrics).with_suffix(".csv"), index=False)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
