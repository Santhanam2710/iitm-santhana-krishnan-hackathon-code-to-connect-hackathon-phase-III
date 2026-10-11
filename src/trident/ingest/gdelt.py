"""GDELT adapters.

* ``GdeltDocAdapter``: GDELT DOC 2.0 API article search (headlines + URLs), polled every
  15 minutes. Also used offline by scripts/build_replay.py to pull real headlines for a
  historical window (the API accepts start/end datetimes).
* ``gdelt_events_latest``: download the latest 15-minute GDELT 2.0 events file; CAMEO root
  codes become weak labels for the event classifier (training/weak_labels.py).
"""
from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass, field
from datetime import datetime

import httpx

from trident.ingest.base import Adapter
from trident.pipeline.normalize import make_doc
from trident.signals.schema import Doc

DOC_API = "https://api.gdeltproject.org/api/v2/doc/doc"
LASTUPDATE = "http://data.gdeltproject.org/gdeltv2/lastupdate.txt"

# CAMEO root code -> taxonomy class (weak labels).
CAMEO_TO_CLASS = {
    "13": "Geopolitical", "14": "Geopolitical", "15": "Geopolitical", "16": "Geopolitical", "17": "Geopolitical",
    "18": "Geopolitical", "19": "Geopolitical", "20": "Geopolitical", "12": "Geopolitical", "11": "Geopolitical",
}


def _fmt(dt: datetime) -> str:
    return dt.strftime("%Y%m%d%H%M%S")


async def gdelt_doc_search(query: str, start: datetime | None = None, end: datetime | None = None, maxrecords: int = 250,
                           timespan: str | None = "15min") -> list[dict]:
    params = {"query": query, "mode": "ArtList", "format": "json", "maxrecords": str(maxrecords), "sort": "DateAsc"}
    if start and end:
        params.update({"startdatetime": _fmt(start), "enddatetime": _fmt(end)})
    elif timespan:
        params["timespan"] = timespan
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.get(DOC_API, params=params)
        r.raise_for_status()
        if not r.text.strip().startswith("{"):
            return []
        return r.json().get("articles", [])


@dataclass
class GdeltDocAdapter(Adapter):
    query: str = field(default='(sanctions OR tariff OR "central bank" OR inflation OR default OR downgrade OR bankruptcy) sourcelang:english')

    async def fetch(self) -> list[Doc]:
        arts = await gdelt_doc_search(self.query, timespan="15min")
        docs = []
        for a in arts:
            ts = a.get("seendate")  # 20230309T134500Z
            if ts and "T" in ts:
                ts = datetime.strptime(ts, "%Y%m%dT%H%M%SZ").isoformat() + "Z"
            docs.append(make_doc("", "gdelt", f"gdelt:{a.get('domain', 'web')}", ts, a.get("url"), a.get("title", "")))
        return docs


async def gdelt_events_latest() -> list[dict]:
    """Return rows of the latest GDELT 2.0 export file with CAMEO weak labels."""
    async with httpx.AsyncClient(timeout=60) as client:
        idx = (await client.get(LASTUPDATE)).text.split()
        export_url = next(u for u in idx if u.endswith(".export.CSV.zip"))
        z = zipfile.ZipFile(io.BytesIO((await client.get(export_url)).content))
    rows = []
    with z.open(z.namelist()[0]) as fh:
        for line in io.TextIOWrapper(fh, encoding="utf-8", errors="ignore"):
            f = line.rstrip("\n").split("\t")
            if len(f) < 61:
                continue
            root = f[28]
            rows.append({"date": f[1], "actor1": f[6], "actor2": f[16], "event_root": root, "goldstein": f[30],
                         "tone": f[34], "url": f[60], "weak_label": CAMEO_TO_CLASS.get(root)})
    return rows
