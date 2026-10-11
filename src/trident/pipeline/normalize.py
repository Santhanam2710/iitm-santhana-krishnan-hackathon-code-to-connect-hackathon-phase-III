"""Normalize raw text into the canonical Doc: strip HTML, fix whitespace, detect language."""
from __future__ import annotations

import html
import re
from typing import Any

from trident.signals.schema import Doc
from trident.utils.common import iso, parse_ts, short_hash, utcnow

TAG = re.compile(r"<[^>]+>")
WS = re.compile(r"\s+")
URLS = re.compile(r"https?://\S+")
FOREIGN_STOP = re.compile(r"\b(le|la|les|des|du|une|est|et|der|die|das|und|ist|von|mit|el|los|las|del|que|por|con|il|di|della|per|che|une|nach|auf|sur|dans)\b")
EN_STOP = re.compile(r"\b(the|a|an|of|to|in|and|on|for|as|after|is|its|with|at|by|from|more|than|despite|if|you|have|there|why|this|that|are|be|has|will|new|says|said|over|amid|into|not|no|up|down|out|it|was|were|but|or|what|how|who|all|just|again)\b")


def clean_text(text: str) -> str:
    text = html.unescape(TAG.sub(" ", text or ""))
    text = URLS.sub("", text)
    text = text.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    return WS.sub(" ", text).strip()


def looks_english(text: str) -> bool:
    if not text:
        return False
    ascii_ratio = sum(1 for c in text if ord(c) < 128) / len(text)
    if ascii_ratio <= 0.85:
        return False
    low = text.lower()
    english, foreign = len(EN_STOP.findall(low)), len(FOREIGN_STOP.findall(low))
    return not (foreign >= 2 and foreign > english)


def make_doc(text: str, source_type: str, source_name: str, ts_published: Any = None, url: str | None = None,
             title: str | None = None, is_synthetic: bool = False, replay_id: str | None = None,
             meta: dict | None = None) -> Doc:
    body = clean_text(text)
    title_c = clean_text(title) if title else None
    if title_c and title_c not in body:
        body = f"{title_c}. {body}" if body else title_c
    published = parse_ts(ts_published) if ts_published else utcnow()
    return Doc(
        doc_id=f"doc_{short_hash(source_name + '|' + body + '|' + iso(published), 16)}",
        text=body[:2000],
        title=title_c,
        url=url,
        source_type=source_type,  # type: ignore[arg-type]
        source_name=source_name,
        ts_published=iso(published),
        ts_ingested=iso(utcnow()),
        lang="en" if looks_english(body) else "other",
        is_synthetic=is_synthetic,
        replay_id=replay_id,
        meta=meta or {},
    )
