"""Fetch the REAL public datasets on your own machine (the build sandbox could not reach them).

    python scripts/fetch_data.py --all            # everything that needs no account
    python scripts/fetch_data.py --kaggle         # Kaggle datasets (needs ~/.kaggle/kaggle.json)
    python scripts/fetch_data.py --sentiment --prices --fred --gdelt --rss --spacy --lexicons

What lands where (all under data/raw/ unless noted; large files are gitignored):
  --sentiment  phrasebank/AllAgree.parquet           Financial PhraseBank via Hugging Face (CC BY-NC-SA 3.0)
               twitter_sentiment/{train,valid}.csv   zeroshot/twitter-financial-news-sentiment (MIT)
               twitter_topic/{train,valid}.csv       zeroshot/twitter-financial-news-topic (MIT)
  --prices     data/market/prices_sp15.csv           yfinance daily closes, 15 index names, 2009-today
  --fred       data/market/fred.csv                  DGS2, DGS10, IG OAS, HY OAS, VIX (for shock calibration)
  --gdelt      gdelt/doc_<date>.jsonl, gdelt/events_latest.jsonl   real headlines + CAMEO weak labels
  --rss        rss/capture_<ts>.jsonl                one pass over configured RSS feeds (for labeling)
  --newsapi    newsapi/backfill.jsonl                needs NEWSAPI_KEY (free plan: 24 h delayed, dev use only)
  --kaggle     news_hist.jsonl                       headlines with dates/tickers for the backtest + IC
               tweets_kaggle.csv                     stock tweets for the social replay
               transactions.csv                      Financial Transactions dataset for Module B's book
  --spacy      en_core_web_sm model (enables ORG verification in entity resolution)
  --lexicons   data/lexicons/lm_{positive,negative}.txt   Loughran-McDonald word lists
Each step is independent and prints what it did or why it was skipped.
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import io
import json
import os
import subprocess
import sys
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
RAW = ROOT / "data" / "raw"
MARKET = ROOT / "data" / "market"
HF = "https://huggingface.co/datasets"

TICKERS = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "JPM", "BAC", "XOM", "JNJ", "PFE", "TSLA", "BA", "WMT", "KO"]
KAGGLE = {
    # slug -> what it is (verify the slug on kaggle.com if a download fails; datasets get renamed)
    "ankurzing/sentiment-analysis-for-financial-news": "Financial PhraseBank (Kaggle copy) -> phrasebank/all-data.csv",
    "miguelaenlle/massive-stock-news-analysis-db-for-nlpbacktests": "Daily headlines with tickers (2009-2020) -> news_hist.jsonl",
    "equinxx/stock-tweets-for-sentiment-analysis-and-prediction": "Stock tweets 2021-2022 -> tweets_kaggle.csv",
    "computingvictor/transactions-fraud-datasets": "Financial Transactions dataset -> transactions.csv",
}


def ok(msg: str) -> None:
    print(f"  [ok] {msg}")


def skip(msg: str) -> None:
    print(f"  [skip] {msg}")


def http_get(url: str, timeout: int = 60) -> bytes:
    import httpx

    r = httpx.get(url, timeout=timeout, follow_redirects=True, headers={"User-Agent": "TRIDENT research prototype"})
    r.raise_for_status()
    return r.content


def fetch_sentiment() -> None:
    print("sentiment datasets")
    for repo, files, sub in (("zeroshot/twitter-financial-news-sentiment", {"sent_train.csv": "train.csv", "sent_valid.csv": "valid.csv"}, "twitter_sentiment"),
                             ("zeroshot/twitter-financial-news-topic", {"topic_train.csv": "train.csv", "topic_valid.csv": "valid.csv"}, "twitter_topic")):
        d = RAW / sub
        d.mkdir(parents=True, exist_ok=True)
        for src, dst in files.items():
            try:
                (d / dst).write_bytes(http_get(f"{HF}/{repo}/resolve/main/{src}"))
                ok(f"{repo}/{src} -> {d / dst}")
            except Exception as exc:
                skip(f"{repo}/{src}: {exc}")
    d = RAW / "phrasebank"
    d.mkdir(parents=True, exist_ok=True)
    try:
        from datasets import load_dataset

        for cfg, name in (("sentences_allagree", "AllAgree"), ("sentences_75agree", "75Agree")):
            ds = load_dataset("takala/financial_phrasebank", cfg, trust_remote_code=True)["train"]
            names = ds.features["label"].names
            import pandas as pd

            pd.DataFrame({"text": ds["sentence"], "label": [names[i] for i in ds["label"]]}).to_parquet(d / f"{name}.parquet")
            ok(f"Financial PhraseBank {cfg} ({len(ds)} sentences)")
    except Exception as exc:
        skip(f"PhraseBank via Hugging Face failed ({exc}). Alternative: --kaggle (ankurzing/sentiment-analysis-for-financial-news) "
             "or download FinancialPhraseBank-v1.0.zip and put Sentences_AllAgree.txt in data/raw/phrasebank/")


def fetch_prices(start: str = "2009-01-01") -> None:
    print("prices (yfinance)")
    try:
        import yfinance as yf
    except ImportError:
        return skip("pip install yfinance")
    df = yf.download(TICKERS, start=start, auto_adjust=True, progress=False)["Close"]
    df = df.dropna(how="all")
    df.index.name = "date"
    MARKET.mkdir(parents=True, exist_ok=True)
    df.round(4).to_csv(MARKET / "prices_sp15.csv")
    ok(f"{df.shape[0]} days x {df.shape[1]} tickers -> data/market/prices_sp15.csv")


def fetch_fred() -> None:
    print("FRED series")
    import pandas as pd

    series = {"DGS2": "ust2y", "DGS10": "ust10y", "BAMLC0A0CM": "ig_oas", "BAMLH0A0HYM2": "hy_oas", "VIXCLS": "vix", "DEXUSEU": "eurusd"}
    frames = []
    for sid, name in series.items():
        try:
            s = pd.read_csv(io.BytesIO(http_get(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}")), na_values=".")
            s.columns = ["date", name]
            frames.append(s.set_index("date"))
            ok(sid)
        except Exception as exc:
            skip(f"{sid}: {exc}")
    if frames:
        out = pd.concat(frames, axis=1)
        out.to_csv(MARKET / "fred.csv")
        ok("data/market/fred.csv  (use scripts/calibrate_shocks.py to compare presets with history)")


def fetch_gdelt(days: int = 3) -> None:
    print("GDELT")
    from trident.ingest.gdelt import gdelt_doc_search, gdelt_events_latest

    d = RAW / "gdelt"
    d.mkdir(parents=True, exist_ok=True)
    q = '(stocks OR shares OR bank OR "central bank" OR tariff OR sanctions OR downgrade OR earnings OR acquisition) sourcelang:english'
    rows = []
    end = datetime.now(timezone.utc)
    for k in range(days * 2):
        t1 = end - timedelta(hours=12 * k)
        t0 = t1 - timedelta(hours=12)
        try:
            arts = asyncio.run(gdelt_doc_search(q, t0, t1, maxrecords=250))
        except Exception as exc:
            skip(f"GDELT DOC API: {exc}")
            break
        for a in arts:
            sd = a.get("seendate", "")
            ts = datetime.strptime(sd, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc).isoformat() if sd else None
            rows.append({"ts": ts, "text": a.get("title", ""), "url": a.get("url"), "source_type": "gdelt", "source_name": f"gdelt:{a.get('domain')}"})
    f = d / f"doc_{end:%Y%m%d}.jsonl"
    with open(f, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    ok(f"{len(rows)} real headlines -> {f}")
    try:
        ev = asyncio.run(gdelt_events_latest())
        with open(d / "events_latest.jsonl", "w", encoding="utf-8") as fh:
            for r in ev:
                fh.write(json.dumps(r) + "\n")
        ok(f"{len(ev)} GDELT events with CAMEO weak labels -> {d / 'events_latest.jsonl'}")
    except Exception as exc:
        skip(f"GDELT events file: {exc}")


def fetch_rss() -> None:
    print("RSS capture")
    from trident.ingest.rss import RSSAdapter

    a = RSSAdapter("rss")
    try:
        docs = asyncio.run(a.fetch())
    except Exception as exc:
        return skip(str(exc))
    d = RAW / "rss"
    d.mkdir(parents=True, exist_ok=True)
    f = d / f"capture_{datetime.now():%Y%m%d_%H%M}.jsonl"
    with open(f, "w", encoding="utf-8") as fh:
        for doc in docs:
            fh.write(json.dumps({"ts": doc.ts_published, "text": doc.text, "url": doc.url, "source_type": "newswire", "source_name": doc.source_name}) + "\n")
    ok(f"{len(docs)} headlines -> {f}")


def fetch_newsapi() -> None:
    print("NewsAPI backfill")
    if not os.environ.get("NEWSAPI_KEY"):
        return skip("set NEWSAPI_KEY (free Developer plan: 100 requests/day, 24 h delay, development use only)")
    from trident.ingest.newsapi import NewsApiAdapter

    docs = asyncio.run(NewsApiAdapter("newsapi").fetch())
    d = RAW / "newsapi"
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "backfill.jsonl", "w", encoding="utf-8") as fh:
        for doc in docs:
            fh.write(json.dumps({"ts": doc.ts_published, "text": doc.text, "url": doc.url, "source_type": "newswire", "source_name": doc.source_name}) + "\n")
    ok(f"{len(docs)} articles -> {d / 'backfill.jsonl'}")


def fetch_kaggle() -> None:
    print("Kaggle datasets")
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi

        api = KaggleApi()
        api.authenticate()
    except Exception as exc:
        return skip(f"Kaggle API not configured ({exc}). pip install kaggle and place kaggle.json in ~/.kaggle/")
    tmp = RAW / "kaggle"
    tmp.mkdir(parents=True, exist_ok=True)
    for slug, desc in KAGGLE.items():
        try:
            api.dataset_download_files(slug, path=str(tmp / slug.split("/")[1]), unzip=True, quiet=True)
            ok(f"{slug}: {desc}")
        except Exception as exc:
            skip(f"{slug}: {exc}")
    convert_kaggle(tmp)


def convert_kaggle(tmp: Path) -> None:
    import pandas as pd

    from trident.config import get_config

    pb = next(iter((tmp / "sentiment-analysis-for-financial-news").glob("*.csv")), None)
    if pb:
        (RAW / "phrasebank").mkdir(parents=True, exist_ok=True)
        (RAW / "phrasebank" / "all-data.csv").write_bytes(pb.read_bytes())
        ok("phrasebank/all-data.csv")
    news_dir = tmp / "massive-stock-news-analysis-db-for-nlpbacktests"
    files = list(news_dir.glob("*.csv")) if news_dir.exists() else []
    if files:
        want = {e.ticker for e in get_config().index}
        rows = 0
        with open(RAW / "news_hist.jsonl", "w", encoding="utf-8") as out:
            for f in files:
                for chunk in pd.read_csv(f, chunksize=200000):
                    cols = {c.lower(): c for c in chunk.columns}
                    tcol, hcol, dcol = cols.get("stock"), cols.get("title") or cols.get("headline"), cols.get("date")
                    if not (tcol and hcol and dcol):
                        continue
                    chunk = chunk[chunk[tcol].isin(want)]
                    for t, h, d in zip(chunk[tcol], chunk[hcol], chunk[dcol]):
                        out.write(json.dumps({"ts": pd.Timestamp(d, tz="UTC").isoformat() if pd.Timestamp(d).tzinfo is None else pd.Timestamp(d).isoformat(),
                                              "text": str(h), "ticker_hint": t, "source_type": "newswire", "source_name": "kaggle:news"}) + "\n")
                        rows += 1
        ok(f"news_hist.jsonl ({rows} headlines on index names) -> run python scripts/run_all_results.py for real IC/backtest")
    tw = next(iter((tmp / "stock-tweets-for-sentiment-analysis-and-prediction").glob("*tweets*.csv")), None)
    if tw:
        (RAW / "tweets_kaggle.csv").write_bytes(tw.read_bytes())
        ok("tweets_kaggle.csv (stream it with SocialReplayAdapter(path='data/raw/tweets_kaggle.csv'))")
    tx_dir = tmp / "transactions-fraud-datasets"
    tx = next(iter(tx_dir.glob("transactions*.csv")), None) if tx_dir.exists() else None
    if tx:
        df = pd.read_csv(tx, nrows=200000)
        mcc = next(iter(tx_dir.glob("*mcc*.json")), None)
        if mcc is not None and "mcc" in df.columns:
            codes = json.loads(mcc.read_text())
            df["merchant_category"] = df["mcc"].astype(str).map(codes).fillna("Other")
        df.to_csv(RAW / "transactions.csv", index=False)
        ok("transactions.csv -> run python scripts/build_portfolio.py to rebuild the wholesale book from real data")


def fetch_spacy() -> None:
    print("spaCy model")
    r = subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"], capture_output=True, text=True)
    ok("en_core_web_sm installed") if r.returncode == 0 else skip(r.stderr[-300:])


def fetch_lexicons() -> None:
    print("Loughran-McDonald word lists")
    from trident.nlp.lexicon import lm_words

    pos, neg = lm_words()
    if not pos:
        return skip("pip install pysentiment2 (bundles the LM dictionary), or download the master dictionary from the authors' site")
    d = ROOT / "data" / "lexicons"
    d.mkdir(parents=True, exist_ok=True)
    (d / "lm_positive.txt").write_text("\n".join(sorted(pos)))
    (d / "lm_negative.txt").write_text("\n".join(sorted(neg)))
    ok(f"{len(pos)} positive / {len(neg)} negative words")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for flag in ("all", "sentiment", "prices", "fred", "gdelt", "rss", "newsapi", "kaggle", "spacy", "lexicons"):
        ap.add_argument(f"--{flag}", action="store_true")
    a = ap.parse_args()
    steps = [("sentiment", fetch_sentiment), ("prices", fetch_prices), ("fred", fetch_fred), ("gdelt", fetch_gdelt), ("rss", fetch_rss),
             ("newsapi", fetch_newsapi), ("spacy", fetch_spacy), ("lexicons", fetch_lexicons), ("kaggle", fetch_kaggle)]
    for name, fn in steps:
        if getattr(a, name) or (a.all and name != "kaggle"):
            try:
                fn()
            except Exception as exc:
                skip(f"{name} failed: {exc}")
    if not any(vars(a).values()):
        ap.print_help()


if __name__ == "__main__":
    main()
