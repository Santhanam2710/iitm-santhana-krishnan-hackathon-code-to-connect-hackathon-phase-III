"""RSS newswire adapter with conditional GET (ETag / Last-Modified)."""
from __future__ import annotations

from dataclasses import dataclass, field

import feedparser
import httpx

from trident.config import get_config
from trident.ingest.base import Adapter
from trident.pipeline.normalize import make_doc
from trident.signals.schema import Doc


@dataclass
class RSSAdapter(Adapter):
    feeds: list[str] = field(default_factory=list)

    def __post_init__(self):
        super().__post_init__()
        cfg = get_config().section("ingest")
        self.feeds = self.feeds or list(cfg.get("rss_feeds", []))
        self.ua = cfg.get("user_agent", "TRIDENT research prototype")
        self.etags: dict[str, dict[str, str]] = {}

    async def fetch(self) -> list[Doc]:
        docs: list[Doc] = []
        errors = 0
        async with httpx.AsyncClient(timeout=15, follow_redirects=True, headers={"User-Agent": self.ua}) as client:
            for url in self.feeds:
                headers = {}
                cache = self.etags.get(url, {})
                if "etag" in cache:
                    headers["If-None-Match"] = cache["etag"]
                if "modified" in cache:
                    headers["If-Modified-Since"] = cache["modified"]
                try:
                    r = await client.get(url, headers=headers)
                except httpx.HTTPError:
                    errors += 1
                    continue
                if r.status_code == 304:
                    continue
                if r.status_code >= 400:
                    errors += 1
                    continue
                self.etags[url] = {k: v for k, v in (("etag", r.headers.get("ETag")), ("modified", r.headers.get("Last-Modified"))) if v}
                feed = feedparser.parse(r.content)
                host = httpx.URL(url).host
                for e in feed.entries[:50]:
                    title = e.get("title", "")
                    summary = e.get("summary", "")
                    published = e.get("published") or e.get("updated")
                    # Store headline + short summary only (licensing guardrail).
                    docs.append(make_doc(summary[:280], "newswire", f"rss:{host}", published, e.get("link"), title))
        if errors == len(self.feeds) and self.feeds:
            raise RuntimeError("all RSS feeds failed")
        return docs
