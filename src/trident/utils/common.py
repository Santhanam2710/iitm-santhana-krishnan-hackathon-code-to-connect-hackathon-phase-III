"""Small shared helpers: time, ids, hashing, logging, JSONL IO."""
from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import string
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

_LOG_CONFIGURED = False


def get_logger(name: str) -> logging.Logger:
    global _LOG_CONFIGURED
    if not _LOG_CONFIGURED:
        logging.basicConfig(
            level=os.environ.get("TRIDENT_LOG_LEVEL", "INFO"),
            format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        )
        _LOG_CONFIGURED = True
    return logging.getLogger(name)


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def iso(dt: datetime | None = None) -> str:
    dt = dt or utcnow()
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def parse_ts(value: Any) -> datetime:
    """Parse ISO strings, epoch seconds, or datetimes into an aware UTC datetime."""
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(float(value), tz=timezone.utc)
    s = str(value).strip().replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        from email.utils import parsedate_to_datetime

        dt = parsedate_to_datetime(str(value))
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


_ALPHABET = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"  # Crockford base32


def ulid() -> str:
    """Time-sortable unique id (ULID-like, 26 chars)."""
    ms = int(time.time() * 1000)
    t = ""
    for _ in range(10):
        t = _ALPHABET[ms % 32] + t
        ms //= 32
    r = "".join(random.choice(_ALPHABET) for _ in range(16))
    return t + r


def short_hash(text: str, n: int = 12) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:n]


def file_hash(path: str | Path, n: int = 12) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()[:n]


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line and not line.startswith("//"):
                yield json.loads(line)


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]], append: bool = False) -> int:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "a" if append else "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
            n += 1
    return n


def rand_id(prefix: str = "", n: int = 8) -> str:
    return prefix + "".join(random.choices(string.ascii_lowercase + string.digits, k=n))
