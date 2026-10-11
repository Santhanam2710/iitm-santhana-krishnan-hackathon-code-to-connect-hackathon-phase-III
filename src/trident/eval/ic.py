"""Information coefficient: Spearman correlation between the per-ticker aggregated
sentiment at the close of day d-1 and the abnormal return on day d (point-in-time).

  python -m trident.eval.ic --signals data/runtime/signals_hist.jsonl --prices data/market/prices_sp15.csv
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, time, timezone

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from trident.config import get_config, resolve
from trident.modules.rebalancer.aggregate import SentimentBook
from trident.utils.common import parse_ts, read_jsonl


def daily_ic(signals: list[dict], prices: pd.DataFrame) -> dict:
    tickers = [e.ticker for e in get_config().index if e.ticker in prices.columns]
    rets = prices[tickers].pct_change().dropna(how="all")
    ar = rets.sub(rets.mean(axis=1), axis=0)  # market-adjusted
    sigs = sorted(signals, key=lambda s: s["ts_published"])
    book = SentimentBook(contagion=0.0)
    book.index = tickers
    i = 0
    ics, prev = [], prices.index[0]
    for day in rets.index:
        cutoff = datetime.combine(prev.date(), time(21, 0), tzinfo=timezone.utc)
        prev = day
        while i < len(sigs) and parse_ts(sigs[i]["ts_published"]) < cutoff:
            s = sigs[i]
            book.add(s["entity"]["ticker"], parse_ts(s["ts_published"]), float(s["sentiment_score"]), float(s["sentiment_confidence"]),
                     float(s["source"]["reliability"]))
            i += 1
        sc = book.scores(cutoff)
        x = np.array([sc[t]["own"] for t in tickers])
        if np.count_nonzero(np.abs(x) > 1e-4) < 3:
            continue
        r = spearmanr(x, ar.loc[day, tickers].to_numpy()).statistic
        if np.isfinite(r):
            ics.append(r)
    ics = np.array(ics)
    return {"days": int(len(ics)), "mean_ic": float(ics.mean()) if len(ics) else None,
            "t_stat": float(ics.mean() / (ics.std(ddof=1) / np.sqrt(len(ics)))) if len(ics) > 2 else None,
            "hit_rate": float((ics > 0).mean()) if len(ics) else None}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--signals", required=True)
    ap.add_argument("--prices", required=True)
    a = ap.parse_args()
    prices = pd.read_csv(resolve(a.prices), parse_dates=["date"]).set_index("date").sort_index()
    print(json.dumps(daily_ic(list(read_jsonl(resolve(a.signals))), prices), indent=2))


if __name__ == "__main__":
    main()
