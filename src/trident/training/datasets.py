"""Dataset loaders. Each returns a pandas DataFrame with ``text`` and ``label`` columns plus
a ``source`` column, or ``None`` if the dataset is not present locally.

Real datasets are fetched by scripts/fetch_data.py into data/raw/:
  phrasebank/        Financial PhraseBank (Malo et al. 2014; CC BY-NC-SA 3.0)
                     - Sentences_AllAgree.txt / Sentences_75Agree.txt (original "sentence@label" format), or
                     - all-data.csv (Kaggle "Sentiment Analysis for Financial News": label,text, latin-1)
  twitter_sentiment/ zeroshot/twitter-financial-news-sentiment  (train.csv, valid.csv; label 0 bearish, 1 bullish, 2 neutral)
  twitter_topic/     zeroshot/twitter-financial-news-topic      (train.csv, valid.csv; 20 topic ids, mapped below)
Authored and synthetic sets ship with the repo (data/gold, data/synthetic).
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from trident.config import DATA_DIR

RAW = DATA_DIR / "raw"

# Topic ids of zeroshot/twitter-financial-news-topic -> TRIDENT taxonomy.
# Verify against the dataset card before reporting (ids listed there as of 2023-2024).
TOPIC_TO_EVENT = {
    0: "Other",               # Analyst Update
    1: "Macroeconomic",       # Fed | Central Banks
    2: "Product Launch",      # Company | Product News
    3: "Credit Event",        # Treasuries | Corporate Debt
    4: "Other",               # Dividend
    5: "Earnings",            # Earnings
    6: "Other",               # Energy | Oil
    7: "Other",               # Financials
    8: "Other",               # Currencies
    9: "Other",               # General News | Opinion
    10: "Other",              # Gold | Metals | Materials
    11: "Other",              # IPO
    12: "Regulatory/Legal",   # Legal | Regulation
    13: "Merger/Acquisition", # M&A | Investments
    14: "Macroeconomic",      # Macro
    15: "Other",              # Markets
    16: "Geopolitical",       # Politics
    17: "Other",              # Personnel Change
    18: "Other",              # Stock Commentary
    19: "Other",              # Stock Movement
}
TW_SENT = {0: "negative", 1: "positive", 2: "neutral"}
POL = {"pos": "positive", "neg": "negative", "neu": "neutral"}


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def phrasebank(agree: str = "AllAgree") -> pd.DataFrame | None:
    d = RAW / "phrasebank"
    txt = d / f"Sentences_{agree}.txt"
    if txt.exists():
        rows = []
        for line in txt.read_text(encoding="latin-1").splitlines():
            if "@" in line:
                t, l = line.rsplit("@", 1)
                rows.append({"text": t.strip(), "label": l.strip()})
        return pd.DataFrame(rows).assign(source=f"phrasebank_{agree}")
    csv = d / "all-data.csv"
    if csv.exists():
        df = pd.read_csv(csv, encoding="latin-1", header=None, names=["label", "text"])
        return df.assign(source="phrasebank_kaggle")
    parquet = d / f"{agree}.parquet"  # written by fetch_data.py from Hugging Face
    if parquet.exists():
        return pd.read_parquet(parquet).assign(source=f"phrasebank_{agree}")
    return None


def twitter_sentiment(split: str = "train") -> pd.DataFrame | None:
    f = RAW / "twitter_sentiment" / f"{split}.csv"
    if not f.exists():
        return None
    df = pd.read_csv(f)
    return pd.DataFrame({"text": df["text"], "label": df["label"].map(TW_SENT), "source": f"twitter_sentiment_{split}"})


def twitter_topic(split: str = "train") -> pd.DataFrame | None:
    f = RAW / "twitter_topic" / f"{split}.csv"
    if not f.exists():
        return None
    df = pd.read_csv(f)
    return pd.DataFrame({"text": df["text"], "label": df["label"].map(TOPIC_TO_EVENT), "source": f"twitter_topic_{split}"})


def gold_v0(task: str = "event") -> pd.DataFrame:
    rows = _jsonl(DATA_DIR / "gold" / "headlines_gold_v0.jsonl")
    if task == "event":
        return pd.DataFrame({"text": [r["text"] for r in rows], "label": [r["event"] for r in rows], "source": "gold_v0",
                             "hard": [r["hard"] for r in rows]})
    if task == "sentiment":
        return pd.DataFrame({"text": [r["text"] for r in rows], "label": [POL[r["polarity"]] for r in rows], "source": "gold_v0",
                             "hard": [r["hard"] for r in rows]})
    return pd.DataFrame(rows)


def gold_real(task: str = "event") -> pd.DataFrame | None:
    """Real headlines you labeled with scripts/label_tool.py (data/gold/headlines_gold_real.jsonl)."""
    f = DATA_DIR / "gold" / "headlines_gold_real.jsonl"
    if not f.exists():
        return None
    rows = [r for r in _jsonl(f) if r.get("event")]
    if task == "event":
        return pd.DataFrame({"text": [r["text"] for r in rows], "label": [r["event"] for r in rows], "source": "gold_real"})
    return pd.DataFrame({"text": [r["text"] for r in rows], "label": [POL.get(r.get("polarity", "neu"), "neutral") for r in rows], "source": "gold_real"})


def synthetic(task: str = "event") -> pd.DataFrame:
    rows = _jsonl(DATA_DIR / "synthetic" / "synthetic_news.jsonl")
    lab = [r["label_event"] for r in rows] if task == "event" else [POL[r["label_polarity"]] for r in rows]
    return pd.DataFrame({"text": [r["text"] for r in rows], "label": lab, "source": "synthetic"})


def unlabeled_corpus() -> list[str]:
    """Headlines for weak labeling: GDELT pulls, RSS captures and replay text, whichever exist."""
    texts: list[str] = []
    for f in list((RAW / "gdelt").glob("*.jsonl")) + list((RAW / "rss").glob("*.jsonl")) + list((DATA_DIR / "replay").glob("*.jsonl")):
        texts += [r.get("text") or r.get("title", "") for r in _jsonl(f)]
    return [t for t in texts if t and len(t) > 15]
