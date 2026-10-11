"""Social sources.

* ``SocialReplayAdapter`` streams a tweets CSV/JSONL (e.g. the Kaggle stock-tweets
  dataset) in timestamp order at k-times speed. Reading X's API is a paid product, so
  replay replaces live X ingestion.
* ``RedditAdapter`` polls public subreddit JSON listings (no key needed, low volume).
"""
from __future__ import annotations

import asyncio
import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import AsyncIterator

import httpx

from trident.config import get_config, resolve
from trident.ingest.base import Adapter
from trident.pipeline.normalize import make_doc
from trident.signals.schema import Doc
from trident.utils.common import parse_ts


def load_social_rows(path: str | Path) -> list[dict]:
    p = resolve(path)
    rows: list[dict] = []
    if p.suffix == ".jsonl":
        rows = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    else:
        with open(p, newline="", encoding="utf-8", errors="ignore") as fh:
            for r in csv.DictReader(fh):
                text = r.get("Tweet") or r.get("text") or r.get("tweet") or ""
                ts = r.get("Date") or r.get("date") or r.get("created_at")
                rows.append({"text": text, "ts": ts, "ticker": r.get("Stock Name") or r.get("ticker")})
    rows = [r for r in rows if r.get("text") and r.get("ts")]
    rows.sort(key=lambda r: parse_ts(r["ts"]))
    return rows


@dataclass
class SocialReplayAdapter(Adapter):
    path: str = "data/samples/tweets_sample.jsonl"
    speed: float = 60.0

    async def stream(self) -> AsyncIterator[Doc]:  # type: ignore[override]
        rows = load_social_rows(self.path)
        self.health.state = "ok"
        prev = None
        for r in rows:
            ts = parse_ts(r["ts"])
            if prev is not None:
                await asyncio.sleep(min(max((ts - prev).total_seconds() / self.speed, 0), 5))
            prev = ts
            self.health.docs += 1
            yield make_doc(r["text"], "social", "social:replay", ts, meta={"ticker_hint": r.get("ticker")})


@dataclass
class RedditAdapter(Adapter):
    subreddits: list[str] = field(default_factory=lambda: ["stocks", "investing", "wallstreetbets"])

    async def fetch(self) -> list[Doc]:
        ua = get_config().section("ingest").get("user_agent")
        docs = []
        async with httpx.AsyncClient(timeout=15, headers={"User-Agent": ua}) as client:
            for sub in self.subreddits:
                r = await client.get(f"https://www.reddit.com/r/{sub}/new.json", params={"limit": 50})
                if r.status_code != 200:
                    continue
                for c in r.json().get("data", {}).get("children", []):
                    d = c.get("data", {})
                    docs.append(make_doc(d.get("selftext", "")[:280], "social", f"reddit:{sub}", d.get("created_utc"),
                                         "https://reddit.com" + d.get("permalink", ""), d.get("title")))
        return docs
