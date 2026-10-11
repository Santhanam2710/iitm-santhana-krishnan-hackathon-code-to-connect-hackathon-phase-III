"""Synthetic news and social generator (Part IV of the blueprint).

Two generators share one scenario grid (entity x class x polarity x severity x register):
  * ``TemplateGenerator``: deterministic templates with a noise model (typos, slang,
    cashtags, emoji, negation, multi-entity, delta phrases, retrospective dates,
    rumor-then-denial). Needs no network; used for tests and the default demo.
  * ``LLMGenerator``: schema-constrained prompts to any OpenAI-compatible endpoint.
Arrival times follow a Hawkes self-exciting process (eq. D1) for crisis bursts.
Every record carries is_synthetic=True, the generator name and the seed.
"""
from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx

from trident.config import get_config

TEMPLATES: dict[str, dict[str, list[str]]] = {
    "Credit Event": {
        "neg": ["{name} downgraded {n} notches to {grade} as {sector_l} losses mount",
                "{name} shares plunge {pct}% after report of deposit outflows and emergency capital raise",
                "Rating agency cuts {name} outlook to negative on refinancing risk",
                "{name} misses coupon payment, bondholders hire restructuring advisers",
                "{name} faces liquidity crisis as funding costs spike {pct}%"],
        "pos": ["{name} upgraded to {grade} as balance sheet strengthens",
                "{name} completes refinancing, removing near-term default risk"],
    },
    "Geopolitical": {
        "neg": ["{country} imposes sweeping sanctions on {country2}; oil jumps {pct}%",
                "Missile strikes near {strait} disrupt shipping; insurers pull cover",
                "New tariffs of {pct}% on {country2} imports announced, retaliation expected"],
        "pos": ["Ceasefire agreed between {country} and {country2}, easing supply fears"],
    },
    "Macroeconomic": {
        "neg": ["CPI surprises to upside at {pct}%, traders price more Fed rate hikes",
                "Payrolls collapse by {k}k as recession fears grow",
                "Treasury yields spike {bp} basis points after hawkish Fed minutes"],
        "pos": ["Inflation cools to {pct}%, raising hopes of a rate cut",
                "GDP growth beats expectations at {pct}% annualized"],
    },
    "Earnings": {
        "neg": ["{name} misses estimates and cuts guidance; shares fall {pct}%",
                "{name} quarterly revenue below expectations as demand weakens"],
        "pos": ["{name} beats expectations, raises guidance; shares jump {pct}%",
                "{name} posts record revenue on strong demand"],
    },
    "Merger/Acquisition": {
        "pos": ["{name} agrees to acquire a rival for ${bn} billion in all-cash deal",
                "{name} announces merger talks; shares rally {pct}%"],
        "neg": ["{name} takeover bid collapses after regulators object"],
    },
    "Product Launch": {
        "pos": ["{name} unveils next-gen product line to strong early demand",
                "{name} launches new platform ahead of schedule"],
        "neg": ["{name} delays launch of flagship product after defects found"],
    },
    "Regulatory/Legal": {
        "neg": ["{name} hit with antitrust lawsuit by regulators",
                "{name} fined ${bn} billion after probe into sales practices",
                "Regulator opens investigation into {name} safety record"],
        "pos": ["Court dismisses lawsuit against {name}"],
    },
    "Operational": {
        "neg": ["{name} suffers global outage affecting millions of users",
                "Cyberattack forces {name} to halt production at several plants",
                "Workers strike at {name} plants, halting output"],
        "pos": ["{name} restores services after brief outage"],
    },
}
HARD_CASES = ("negation", "sarcasm", "multi-entity", "delta-phrase", "retrospective", "rumor-denial", "none")
COUNTRIES = ["the US", "the EU", "China", "Russia", "Iran", "Japan", "India"]
STRAITS = ["the Strait of Hormuz", "the Red Sea", "the Taiwan Strait"]
SLANG = [" lol", " 🚀", " 📉", "", "", "", " !!", ""]


@dataclass
class ScenarioCell:
    ticker: str
    name: str
    sector: str
    event_class: str
    polarity: str
    severity: int
    register: str
    hard_case: str


def hawkes_times(start: datetime, hours: float, mu: float, alpha: float, beta: float, rng: random.Random) -> list[datetime]:
    """Ogata thinning for lambda(t) = mu + sum alpha*exp(-beta (t - t_i)); t in hours."""
    assert alpha < beta, "branching ratio alpha/beta must be < 1"
    t, events = 0.0, []
    while t < hours:
        lam_bar = mu + sum(alpha * math.exp(-beta * (t - ti)) for ti in events[-200:])
        t += rng.expovariate(lam_bar)
        lam_t = mu + sum(alpha * math.exp(-beta * (t - ti)) for ti in events[-200:])
        if t < hours and rng.random() <= lam_t / lam_bar:
            events.append(t)
    return [start + timedelta(hours=x) for x in events]


class TemplateGenerator:
    name = "template-v1"

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.rng = random.Random(seed)
        cfg = get_config()
        self.entities = [e for e in cfg.entities]

    def short_name(self, e) -> str:
        """Headlines use short brand names ("Apple"), not legal names ("Apple Inc.")."""
        clean = [a for a in e.aliases if " " not in a or len(a) < 18]
        return clean[0] if clean and self.rng.random() < 0.85 else e.name.replace(" Inc.", "").replace(" Group", "")

    def _fill(self, tpl: str, e) -> str:
        r = self.rng
        c1, c2 = r.sample(COUNTRIES, 2)
        return tpl.format(name=self.short_name(e), n=r.choice(["one", "two", "three"]), grade=r.choice(["BBB-", "BB+", "BB", "A-"]),
                          sector_l=r.choice(["commercial real estate", "consumer credit", "energy", "leveraged"]),
                          pct=r.choice([2, 3, 4, 6, 8, 9, 12, 15, 22, 35]), country=c1, country2=c2, strait=r.choice(STRAITS),
                          k=r.choice([50, 80, 120, 200]), bp=r.choice([15, 25, 40]), bn=r.choice([2, 5, 12, 18, 40]))

    def cell(self) -> ScenarioCell:
        r = self.rng
        e = r.choice(self.entities)
        cls = r.choice(list(TEMPLATES))
        pol = r.choice(list(TEMPLATES[cls]))
        sev = r.randint(7, 10) if r.random() < 0.25 else r.randint(1, 7)  # eq. D2: tail oversampling
        reg = r.choices(["newswire", "tweet", "filing"], [0.6, 0.35, 0.05])[0]
        hard = r.choice(HARD_CASES) if r.random() < 0.35 else "none"
        return ScenarioCell(e.ticker, e.name, e.sector, cls, pol, sev, reg, hard)

    def text(self, c: ScenarioCell) -> tuple[str, str]:
        r = self.rng
        e = next(x for x in self.entities if x.ticker == c.ticker)
        t = self._fill(r.choice(TEMPLATES[c.event_class][c.polarity]), e)
        nm = self.short_name(e)
        pol = c.polarity
        if c.hard_case == "negation" and c.polarity == "pos":
            t = f"{nm} did not " + r.choice(["beat expectations", "win approval", "see strong demand"]) + " this quarter"
            pol = "neg"
        elif c.hard_case == "delta-phrase":
            t = f"{nm} losses narrowed sharply in the quarter as costs fell"
            pol = "pos"
        elif c.hard_case == "multi-entity":
            other = r.choice([x for x in self.entities if x.ticker != c.ticker])
            t = f"{nm} beats {self.short_name(other)} for a major contract while sector peers slump"
            pol = "pos"
        elif c.hard_case == "retrospective":
            t = f"Ten years after the {r.choice(['2008', '2012', '2015'])} crisis, {nm} reflects on lessons learned"
            pol = "neu"
        elif c.hard_case == "rumor-denial":
            t = f"{nm} denies report of bankruptcy filing, says liquidity is strong"
            pol = "pos"
        elif c.hard_case == "sarcasm" and c.register == "tweet":
            t = f"great, ${e.ticker} 'record quarter' and the stock is down {r.choice([5, 8, 11])}% again 🙃"
            pol = "neg"
        if c.register == "tweet":
            t = (f"${e.ticker} " if r.random() < 0.6 else "") + t.lower() + r.choice(SLANG)
            if r.random() < 0.15:  # typo noise
                i = r.randrange(1, max(2, len(t) - 1))
                t = t[:i] + t[i + 1 :]
        elif c.register == "filing":
            t = f"{nm} 8-K: " + t
        return t, pol

    def generate(self, n: int, start: datetime | None = None, hours: float = 48.0, burst: bool = False) -> list[dict[str, Any]]:
        start = start or datetime(2026, 3, 2, 13, 0, tzinfo=timezone.utc)
        times = hawkes_times(start, hours, mu=n / hours * (0.4 if burst else 1.0), alpha=0.8 if burst else 0.0001,
                             beta=1.2 if burst else 1.0, rng=self.rng) if n else []
        while len(times) < n:
            times.append(start + timedelta(hours=self.rng.uniform(0, hours)))
        times = sorted(times)[:n]
        rows = []
        for ts in times:
            c = self.cell()
            text, pol = self.text(c)
            label = c.event_class if c.hard_case not in ("retrospective",) else "Other"
            rows.append({"ts": ts.isoformat().replace("+00:00", "Z"), "text": text,
                         "source_type": "social" if c.register == "tweet" else ("filing" if c.register == "filing" else "newswire"),
                         "source_name": f"synthetic:{c.register}", "is_synthetic": True,
                         "label_event": label, "label_polarity": pol, "intended_severity": c.severity,
                         "ticker": c.ticker, "hard_case": c.hard_case, "generator": self.name, "seed": self.seed})
        return rows


PROMPT = """You write realistic financial text for model testing.
Return JSON only: {{"text": str, "label_event": str, "label_polarity": "pos"|"neg"|"neu", "intended_severity": int}}
Entity: {name} ({ticker}), sector {sector}
Event class: {event_class}   Polarity: {polarity}   Severity (1-10): {severity}
Register: {register}   # newswire | tweet | 8-K excerpt
Constraints: under 35 words; no real journalist names; plausible invented figures;
if register=tweet use informal style, may include ${ticker}. Hard case: {hard_case}"""


class LLMGenerator(TemplateGenerator):  # pragma: no cover - needs an LLM endpoint
    name = "llm-v1"

    def __init__(self, seed: int = 42, base_url: str | None = None, model: str | None = None):
        super().__init__(seed)
        eng = get_config().section("engine")
        self.base_url = (base_url or eng.get("llm_base_url")).rstrip("/")
        self.model = model or eng.get("llm_model")

    def text(self, c: ScenarioCell) -> tuple[str, str]:
        body = {"model": self.model, "temperature": 0.9, "response_format": {"type": "json_object"},
                "messages": [{"role": "user", "content": PROMPT.format(**c.__dict__, name=c.name)}]}
        try:
            r = httpx.post(self.base_url + "/chat/completions", json=body, timeout=60)
            data = json.loads(r.json()["choices"][0]["message"]["content"])
            return str(data["text"]), str(data.get("label_polarity", c.polarity))
        except Exception:
            return super().text(c)
