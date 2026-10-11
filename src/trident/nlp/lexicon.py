"""Finance sentiment lexicon with negation, delta-phrases and Loughran-McDonald words.

This is the always-available sentiment backend and the second opinion in the
ensemble (eq. E1). Curated phrases carry the strongest weights; single words from the
Loughran-McDonald (LM) master dictionary add weaker evidence when the optional
``pysentiment2`` package is installed (or ``data/lexicons/lm_*.txt`` exist).
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from trident.config import DATA_DIR

# Curated phrases. Longer phrases win over their sub-phrases.
PHRASES: dict[str, float] = {
    # positive
    "beat": 0.6, "beats": 0.6, "beats expectations": 0.7, "tops estimates": 0.7, "above expectations": 0.6,
    "surge": 0.7, "surges": 0.7, "surged": 0.7, "soar": 0.8, "soars": 0.8, "soared": 0.8,
    "jump": 0.6, "jumps": 0.6, "jumped": 0.6, "rally": 0.6, "rallies": 0.6, "rallied": 0.6,
    "upgrade": 0.7, "upgraded": 0.7, "upgrades": 0.7, "raises guidance": 0.8, "raised guidance": 0.8,
    "raises forecast": 0.7, "raised its forecast": 0.7, "outlook raised": 0.7, "record high": 0.6,
    "record revenue": 0.7, "record profit": 0.7, "strong": 0.5, "strong demand": 0.6, "growth": 0.35,
    "gains": 0.5, "gain": 0.4, "approval": 0.55, "approved": 0.55, "wins": 0.5, "won": 0.4,
    "outperform": 0.6, "bullish": 0.7, "rebound": 0.5, "rebounds": 0.5, "recovers": 0.45,
    "losses narrowed": 0.5, "narrowed losses": 0.5, "narrower loss": 0.5, "smaller-than-expected loss": 0.5,
    "rate cut": 0.4, "rate cuts": 0.4, "cuts rates": 0.4, "buyback": 0.5, "share repurchase": 0.5,
    "dividend increase": 0.5, "raises dividend": 0.5, "partnership": 0.3, "unveils": 0.25, "launches": 0.25,
    "expands": 0.35, "profit rises": 0.6, "profit jumps": 0.7, "boost": 0.4, "boosts": 0.4,
    "stabilize": 0.3, "stabilizes": 0.3, "backstop": 0.35, "deposits are safe": 0.5, "ceasefire": 0.4,
    "eases": 0.3, "cooling inflation": 0.5, "soft landing": 0.5, "rescue deal": 0.2,
    # negative
    "miss": -0.6, "misses": -0.6, "missed": -0.6, "sales miss": -0.65, "below expectations": -0.6,
    "plunge": -0.8, "plunges": -0.8, "plunged": -0.8, "crash": -0.9, "crashes": -0.9,
    "slump": -0.7, "slumps": -0.7, "falls": -0.45, "fell": -0.45, "drops": -0.45, "dropped": -0.45,
    "tumble": -0.7, "tumbles": -0.7, "tumbled": -0.7, "sinks": -0.6, "sank": -0.6, "slides": -0.45,
    "downgrade": -0.7, "downgraded": -0.7, "downgrades": -0.7, "default": -0.9, "defaults": -0.9,
    "defaulted": -0.9, "bankruptcy": -1.0, "chapter 11": -1.0, "lawsuit": -0.5, "sued": -0.5,
    "probe": -0.5, "investigation": -0.45, "fined": -0.5, "recall": -0.6, "recalls": -0.6,
    "sanctions": -0.6, "war": -0.7, "invasion": -0.8, "missile strikes": -0.7, "airstrikes": -0.7,
    "strikes": -0.4, "losses mount": -0.6, "losses": -0.45, "loss": -0.4, "layoffs": -0.6,
    "job cuts": -0.6, "profit warning": -0.8, "cuts guidance": -0.8, "lowers guidance": -0.8,
    "cut its forecast": -0.7, "weak": -0.5, "weaker": -0.45, "bearish": -0.7, "outage": -0.6,
    "breach": -0.7, "data breach": -0.75, "cyberattack": -0.8, "fraud": -0.9, "bank run": -0.9,
    "deposit outflows": -0.7, "deposit flight": -0.75, "insolvency": -0.9, "insolvent": -0.9,
    "rate hike": -0.4, "rate hikes": -0.4, "recession": -0.7, "sell-off": -0.6, "selloff": -0.6,
    "not looking good": -0.6, "spikes": -0.3, "surprises to upside": -0.35, "hotter-than-expected": -0.45,
    "halts": -0.5, "halted": -0.5, "grounded": -0.6, "shortage": -0.4, "tariffs": -0.45,
    "crisis": -0.7, "collapse": -0.9, "collapses": -0.9, "collapsed": -0.9, "fails": -0.6,
    "failure": -0.7, "seized": -0.8, "receivership": -0.9, "unrealized losses": -0.6,
    "panic": -0.7, "contagion": -0.7, "turmoil": -0.6, "fears": -0.4, "warns": -0.45,
    "plummet": -0.8, "plummets": -0.8, "writedown": -0.6, "write-down": -0.6, "impairment": -0.5,
    "defect": -0.5, "outflows": -0.5, "lockdown": -0.6, "lockdowns": -0.6, "pandemic": -0.5,
    "emergency": -0.4, "worst day": -0.6, "worst": -0.4, "withdraw deposits": -0.6, "withdrawals": -0.3, "pull deposits": -0.6, "pull funds": -0.5, "falter": -0.4, "shore up capital": -0.4, "volatility": -0.25, "uncertainty": -0.25, "stagflation": -0.6,
}

NEGATORS = re.compile(
    r"\b(not|no|never|without|fails? to|failed to|didn't|doesn't|isn't|wasn't|aren't|won't|hardly|denies|denied)\s+(\w+\s+){0,2}$",
    re.I,
)
INTENSIFIERS = re.compile(r"\b(sharply|massive|sweeping|record|unprecedented|severe|deep|steep|huge)\b", re.I)
LM_WEIGHT = 0.22
WORD = re.compile(r"[A-Za-z][A-Za-z'\-]+")


def _compile(phrases: list[str]) -> re.Pattern[str]:
    alts = "|".join(re.escape(p) for p in sorted(phrases, key=len, reverse=True))
    return re.compile(r"(?<![A-Za-z0-9])(" + alts + r")(?![A-Za-z0-9])", re.I)


_PHRASE_RE = _compile(list(PHRASES))


@lru_cache(maxsize=1)
def lm_words() -> tuple[frozenset[str], frozenset[str]]:
    """Loughran-McDonald positive/negative word sets (upper case). Empty if unavailable."""
    pos_f, neg_f = DATA_DIR / "lexicons" / "lm_positive.txt", DATA_DIR / "lexicons" / "lm_negative.txt"
    if pos_f.exists() and neg_f.exists():
        pos = frozenset(w.strip().upper() for w in pos_f.read_text().splitlines() if w.strip())
        neg = frozenset(w.strip().upper() for w in neg_f.read_text().splitlines() if w.strip())
        return pos, neg
    try:
        import pandas as pd
        import pysentiment2

        p = Path(pysentiment2.__file__).parent / "static" / "LM.csv"
        df = pd.read_csv(p, usecols=["Word", "Positive", "Negative"])
        pos = frozenset(df.loc[df["Positive"] > 0, "Word"].astype(str).str.upper())
        neg = frozenset(df.loc[df["Negative"] > 0, "Word"].astype(str).str.upper())
        return pos, neg
    except Exception:
        return frozenset(), frozenset()


@dataclass
class Hit:
    start: int
    end: int
    text: str
    weight: float
    source: str  # "phrase" | "lm"


@dataclass
class LexResult:
    score: float
    confidence: float
    pos: float
    neg: float
    hits: list[Hit] = field(default_factory=list)


def _negated(text: str, start: int) -> bool:
    return bool(NEGATORS.search(text[max(0, start - 28) : start]))


def find_hits(text: str, use_lm: bool = True) -> list[Hit]:
    hits: list[Hit] = []
    covered: list[tuple[int, int]] = []
    for m in _PHRASE_RE.finditer(text):
        w = PHRASES[m.group(1).lower()]
        if _negated(text, m.start()):
            w *= -0.7
        hits.append(Hit(m.start(), m.end(), m.group(0), w, "phrase"))
        covered.append((m.start(), m.end()))
    if use_lm:
        pos, neg = lm_words()
        if pos or neg:
            for m in WORD.finditer(text):
                if any(a <= m.start() < b for a, b in covered):
                    continue
                u = m.group(0).upper()
                w = LM_WEIGHT if u in pos else (-LM_WEIGHT if u in neg else 0.0)
                if w:
                    if _negated(text, m.start()):
                        w *= -0.7
                    hits.append(Hit(m.start(), m.end(), m.group(0), w, "lm"))
    hits.sort(key=lambda h: h.start)
    return hits


def score_span(text: str, use_lm: bool = True) -> LexResult:
    hits = find_hits(text, use_lm)
    pos = sum(h.weight for h in hits if h.weight > 0)
    neg = sum(h.weight for h in hits if h.weight < 0)
    raw = pos + neg
    boost = 1.0 + 0.15 * min(len(INTENSIFIERS.findall(text)), 2)
    s = math.tanh(raw * boost / 1.1)
    conflict = (min(pos, -neg) / max(pos, -neg)) if (pos > 0 and neg < 0) else 0.0
    n = len([h for h in hits if h.source == "phrase"]) + 0.5 * len([h for h in hits if h.source == "lm"])
    conf = (0.45 + 0.14 * n) * (1.0 - 0.5 * conflict)
    conf = max(0.2, min(0.97, conf)) if hits else 0.3
    return LexResult(score=s, confidence=conf, pos=pos, neg=neg, hits=hits)
