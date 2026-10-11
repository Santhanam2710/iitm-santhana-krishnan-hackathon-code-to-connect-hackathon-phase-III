"""SEC EDGAR current 8-K filings (Atom). Item codes become gold event labels.

SEC requires a descriptive User-Agent with contact details and limits request rates;
set ``ingest.user_agent`` in configs/settings.yaml before enabling live mode.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import feedparser
import httpx

from trident.config import get_config
from trident.ingest.base import Adapter
from trident.pipeline.normalize import make_doc
from trident.signals.schema import Doc

FEED = "https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent&type=8-K&company=&dateb=&owner=include&count=100&output=atom"

# 8-K Item -> taxonomy class (weak/gold label for training and evaluation).
ITEM_TO_CLASS = {
    "1.01": "Merger/Acquisition", "2.01": "Merger/Acquisition", "1.03": "Credit Event", "2.04": "Credit Event",
    "2.06": "Credit Event", "3.01": "Regulatory/Legal", "2.02": "Earnings", "4.02": "Regulatory/Legal",
    "5.02": "Other", "8.01": "Other", "7.01": "Other", "1.05": "Operational", "2.05": "Operational",
}
ITEM_RE = re.compile(r"Item\s+(\d\.\d\d)")


@dataclass
class EdgarAdapter(Adapter):
    def __post_init__(self):
        super().__post_init__()
        self.ua = get_config().section("ingest").get("user_agent")

    async def fetch(self) -> list[Doc]:
        async with httpx.AsyncClient(timeout=20, headers={"User-Agent": self.ua, "Accept-Encoding": "gzip"}) as client:
            r = await client.get(FEED)
            r.raise_for_status()
        feed = feedparser.parse(r.content)
        docs = []
        for e in feed.entries:
            title = e.get("title", "")  # e.g. "8-K - Example Corp (0000123456) (Filer)"
            summary = re.sub(r"<[^>]+>", " ", e.get("summary", ""))
            items = ITEM_RE.findall(summary)
            company = re.sub(r"^8-K\s*-\s*", "", title).split("(")[0].strip()
            labels = sorted({ITEM_TO_CLASS.get(i, "Other") for i in items})
            text = f"{company} files 8-K reporting " + ", ".join(f"Item {i}" for i in items) if items else f"{company} files 8-K"
            docs.append(make_doc(text, "filing", "sec:edgar-8k", e.get("updated"), e.get("link"), None,
                                 meta={"items": items, "item_labels": labels, "company": company}))
        return docs
