"""Retrospective-article detection (red-team item 11).

A story such as "15 years after Lehman, banks are safer" mentions a credit event but
reports no new one. Such items keep their sentiment but have impact capped so they
cannot trigger a stress test.
"""
from __future__ import annotations

import re
from datetime import datetime

RETRO = re.compile(
    r"\b((\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|fifteen|twenty|twenty-five|thirty|fifty)\s+years?\s+(after|since|ago|on)|anniversary|looking back|in hindsight|remember(s|ing)? when|"
    r"lessons from|a decade after|retrospective|history of)\b",
    re.I,
)
YEAR = re.compile(r"\b(19[5-9]\d|20[0-4]\d)\b")


def is_retrospective(text: str, published: datetime) -> tuple[bool, str | None]:
    if RETRO.search(text):
        return True, "retrospective phrasing"
    years = [int(y) for y in YEAR.findall(text)]
    if years and max(years) <= published.year - 2 and not re.search(r"\b(expects?|forecasts?|by|until|through)\s+(19|20)\d\d", text, re.I):
        return True, f"refers to past year {max(years)}"
    return False, None
