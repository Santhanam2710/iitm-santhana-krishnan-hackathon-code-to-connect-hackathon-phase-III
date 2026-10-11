"""NewsAPI adapter (backfill only).

The free Developer plan allows 100 requests/day, delays articles by 24 hours, covers one
month of history and is licensed for development use only. TRIDENT therefore treats it
as a backfill/enrichment source and never as the basis of a real-time claim.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import httpx

from trident.ingest.base import Adapter
from trident.pipeline.normalize import make_doc
from trident.signals.schema import Doc

URL = "https://newsapi.org/v2/everything"


@dataclass
class NewsApiAdapter(Adapter):
    query: str = field(default='(JPMorgan OR "Bank of America" OR Nvidia OR Apple OR Microsoft OR Tesla OR Boeing OR Pfizer) AND (shares OR stock)')

    def __post_init__(self):
        super().__post_init__()
        self.key = os.environ.get("NEWSAPI_KEY", "")
        if not self.key:
            self.enabled = False
            self.health.state = "disabled"
            self.health.note = "set NEWSAPI_KEY to enable"

    async def fetch(self) -> list[Doc]:
        params = {"q": self.query, "language": "en", "sortBy": "publishedAt", "pageSize": 100, "apiKey": self.key}
        async with httpx.AsyncClient(timeout=20) as client:
            r = await client.get(URL, params=params)
            r.raise_for_status()
        docs = []
        for a in r.json().get("articles", []):
            src = (a.get("source") or {}).get("name", "newsapi")
            docs.append(make_doc((a.get("description") or "")[:280], "newswire", f"newsapi:{src}", a.get("publishedAt"),
                                 a.get("url"), a.get("title"), meta={"delayed_free_tier": True}))
        return docs
