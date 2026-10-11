"""Generate every synthetic dataset the offline pipeline needs. All outputs are tagged.

Outputs
  data/synthetic/synthetic_news.jsonl      template headlines/tweets with labels (training augmentation, D3)
  data/synthetic/README.md                 generator, seeds, prompt template (reproducibility)
  data/samples/tweets_sample.jsonl         small tweet stream for the social replay adapter
  data/portfolio/transactions_sample.csv   transaction records in the shape of the Kaggle
                                           "Financial Transactions" dataset (synthetic; replace
                                           with the real file via scripts/fetch_data.py)
  data/market/prices_synthetic.csv         KNOWN-ANSWER market: daily prices for the 15 names
  data/synthetic/market_news.jsonl         dated synthetic news whose events drive those prices

Known-answer design (used to validate the pipeline, never reported as performance):
  each event is company-specific (credit, earnings, M&A, product, legal, operational), has a
  polarity and severity; the market prices it on the publication day
  (unforecastable under our point-in-time rule) and adds a small post-event drift on the
  next day (forecastable). A correct pipeline should recover (a) a positive relation between
  impact score and |CAR| and (b) a small positive information ratio from the drift only.
"""
from __future__ import annotations

import csv
import json
import math
import random
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from trident.config import get_config  # noqa: E402
from trident.ingest.synthetic import PROMPT, TemplateGenerator  # noqa: E402

SEED = 20261011
MCC = [  # (merchant category, sector, relative risk)
    ("Grocery Stores", "Consumer", 0.8), ("Restaurants", "Consumer", 1.2), ("Fuel Stations", "Energy", 1.0),
    ("Utilities - Electric", "Energy", 0.6), ("Real Estate Rent", "Real Estate", 1.4), ("Construction Supplies", "Real Estate", 1.6),
    ("Computer Software", "Technology", 0.7), ("Telecom Services", "Technology", 0.8), ("Freight and Shipping", "Industrials", 1.3),
    ("Machinery", "Industrials", 1.1), ("Medical Services", "Healthcare", 0.6), ("Pharmacies", "Healthcare", 0.7),
    ("Airlines", "Industrials", 1.8), ("Hotels", "Consumer", 1.5), ("Financial Services", "Banks", 1.0),
]


def transactions(n: int = 5000) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    weights = np.array([14, 10, 9, 6, 12, 4, 6, 6, 5, 3, 7, 6, 3, 4, 5], float)
    idx = rng.choice(len(MCC), size=n, p=weights / weights.sum())
    amt = np.round(np.exp(rng.normal(3.6, 1.1, size=n)), 2)
    start = date(2025, 1, 1)
    rows = []
    for i in range(n):
        cat, sector, risk = MCC[idx[i]]
        rows.append({"transaction_id": f"T{i:06d}", "date": (start + timedelta(days=int(rng.integers(0, 365)))).isoformat(),
                     "client_id": f"C{int(rng.integers(0, 900)):04d}", "amount": float(amt[i]), "merchant_category": cat,
                     "use_chip": rng.choice(["Chip", "Swipe", "Online"]), "is_fraud": int(rng.random() < 0.004 * risk),
                     "is_synthetic": 1})
    return pd.DataFrame(rows)


def known_answer_market(start: str = "2019-01-02", end: str = "2025-12-31", seed: int = SEED) -> tuple[pd.DataFrame, list[dict]]:
    cfg = get_config()
    rng = np.random.default_rng(seed + 1)
    gen = TemplateGenerator(seed=seed + 2)
    days = pd.bdate_range(start, end)
    names = cfg.index
    tick = [e.ticker for e in names]
    sectors = sorted({e.sector for e in names})
    vol = np.array([e.vol for e in names]) / math.sqrt(252)
    mkt = rng.normal(0.0003, 0.011, len(days))
    sec_f = {s: rng.normal(0, 0.006, len(days)) for s in sectors}
    beta = rng.uniform(0.8, 1.3, len(names))
    rets = np.zeros((len(days), len(names)))
    for j, e in enumerate(names):
        idio = rng.normal(0, max(vol[j] * 0.7, 0.006), len(days))
        rets[:, j] = beta[j] * mkt + sec_f[e.sector] + idio
    news: list[dict] = []
    p_event = 0.035
    for d_i, day in enumerate(days[:-2]):
        for j, e in enumerate(names):
            if rng.random() > p_event:
                continue
            cell = gen.cell()
            while cell.event_class in ("Geopolitical", "Macroeconomic"):  # market-wide templates name no company
                cell = gen.cell()
            cell = type(cell)(e.ticker, e.name, e.sector, cell.event_class, cell.polarity, cell.severity, cell.register, "none")
            text, pol = gen.text(cell)
            if cell.register == "tweet" and f"${e.ticker}" not in text:
                text = f"${e.ticker} " + text
            sign = 1 if pol == "pos" else (-1 if pol == "neg" else 0)
            sev = cell.severity
            base = {"Credit Event": 1.3, "Geopolitical": 1.0, "Macroeconomic": 0.9, "Regulatory/Legal": 1.0, "Operational": 0.9,
                    "Earnings": 1.1, "Merger/Acquisition": 0.8, "Product Launch": 0.5}.get(cell.event_class, 0.6)
            shock = sign * base * (0.004 + 0.0045 * sev) * rng.uniform(0.6, 1.4)
            hour = int(rng.integers(0, 21))  # published before the close of `day`
            ts = datetime(day.year, day.month, day.day, hour, int(rng.integers(0, 60)), tzinfo=timezone.utc)
            rets[d_i, j] += shock               # priced on the day of publication (not tradable)
            rets[d_i + 1, j] += 0.30 * shock    # post-event drift next day (tradable)
            news.append({"ts": ts.isoformat().replace("+00:00", "Z"), "text": text, "source_type": "newswire" if cell.register != "tweet" else "social",
                         "source_name": f"synthetic:{cell.register}", "is_synthetic": True, "ticker": e.ticker, "label_event": cell.event_class,
                         "label_polarity": pol, "intended_severity": sev, "planted_shock": round(shock, 6)})
    prices = 100 * np.cumprod(1 + rets, axis=0)
    df = pd.DataFrame(prices, index=days, columns=tick)
    df.index.name = "date"
    return df.round(4), news


def main() -> None:
    random.seed(SEED)
    (ROOT / "data" / "synthetic").mkdir(parents=True, exist_ok=True)
    (ROOT / "data" / "samples").mkdir(parents=True, exist_ok=True)
    (ROOT / "data" / "market").mkdir(parents=True, exist_ok=True)
    (ROOT / "data" / "portfolio").mkdir(parents=True, exist_ok=True)

    gen = TemplateGenerator(seed=SEED)
    rows = gen.generate(3000, hours=24 * 30)
    with open(ROOT / "data/synthetic/synthetic_news.jsonl", "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    tweets = [r for r in TemplateGenerator(seed=SEED + 9).generate(900, hours=72) if r["source_type"] == "social"][:300]
    with open(ROOT / "data/samples/tweets_sample.jsonl", "w", encoding="utf-8") as fh:
        for r in tweets:
            fh.write(json.dumps({"ts": r["ts"], "text": r["text"], "ticker": r["ticker"], "is_synthetic": True}) + "\n")

    transactions().to_csv(ROOT / "data/portfolio/transactions_sample.csv", index=False)

    prices, news = known_answer_market()
    prices.to_csv(ROOT / "data/market/prices_synthetic.csv")
    with open(ROOT / "data/synthetic/market_news.jsonl", "w", encoding="utf-8") as fh:
        for r in news:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    (ROOT / "data/synthetic/README.md").write_text(f"""# Synthetic data

Everything in this folder is generated by `scripts/make_synthetic_data.py` (seed {SEED}) and
`src/trident/ingest/synthetic.py`. Every record carries `is_synthetic: true` and is excluded
from all reported evaluation metrics.

| File | Rows | Purpose |
|---|---|---|
| synthetic_news.jsonl | {len(rows)} | Training augmentation for the event and sentiment models (capped share, reweighted; eq. D3/D4) |
| market_news.jsonl | {len(news)} | Known-answer test: dated events that drive `data/market/prices_synthetic.csv` |

Hard cases are injected with probability 0.35 (negation, sarcasm, multi-entity, delta-phrase,
retrospective, rumor-then-denial). Severity 7-10 is oversampled to 25% (eq. D2). Arrival times
follow a Hawkes process (eq. D1) when `burst=True`.

## LLM generator prompt (`LLMGenerator`)

```
{PROMPT}
```
""", encoding="utf-8")
    print(f"synthetic_news={len(rows)} tweets={len(tweets)} market_days={len(prices)} market_news={len(news)}")


if __name__ == "__main__":
    main()
