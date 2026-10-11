"""Entity resolution: text -> tickers.

Order of evidence: cashtags ($AAPL) > exchange tags (NASDAQ: AAPL) > long standalone
tickers > alias trie (brands, products, executives) > fuzzy match. Ambiguous aliases
(Apple, Meta, Coke) need a financial context window, and optionally a spaCy ORG tag.
Every rejection is returned with a reason so it can be logged and shown in the demo.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache

from trident.config import Config, Entity, get_config
from trident.utils.common import get_logger

log = get_logger("trident.entities")

FIN_CONTEXT = re.compile(
    r"\b(shares?|stocks?|earnings|revenue|iphone|ceo|investors?|nasdaq|nyse|market|sales|analysts?|"
    r"guidance|quarter|deal|billion|million|launch(?:es)?|unveils?|downgrade[sd]?|upgrade[sd]?|antitrust|"
    r"lawsuit|acquire[sd]?|acquisition|premarket|regulators?|profit|inc\.?|corp\.?|platforms|valuation|"
    r"ipo|bonds?|debt|dividend|buyback|layoffs|ai|chips?|ads?|users|cds|default|bankrupt\w*|crash\w*|plunge\w*|halted|output|production|plants?|workers|outage|services|outlook|funding|liquidity|refinancing|merger|talks|recalls?)\b|\$[A-Z]{1,5}",
    re.I,
)
CASHTAG = re.compile(r"\$([A-Z]{1,5})\b")
CI_EXCLUDE = {"Amazon", "Azure", "Apple", "Signature Bank", "Blackwell"}
EXCHANGE = re.compile(r"\b(?:NYSE|NASDAQ|Nasdaq|NYSEARCA)\s*:\s*([A-Z]{1,5})\b")


@dataclass
class EntityMatch:
    ticker: str
    name: str
    match: str
    confidence: float
    spans: list[tuple[int, int]] = field(default_factory=list)
    in_index: bool = False
    sector: str = "Other"


@dataclass
class Resolution:
    matches: list[EntityMatch]
    rejected: list[dict]


class _Spacy:
    nlp = None
    tried = False

    @classmethod
    def get(cls, mode: str):
        if cls.tried:
            return cls.nlp
        cls.tried = True
        if str(mode).lower() == "false":
            return None
        try:
            import spacy

            cls.nlp = spacy.load("en_core_web_sm", disable=["lemmatizer"])
            log.info("spaCy en_core_web_sm loaded for ORG verification")
        except Exception as exc:  # pragma: no cover - optional dependency
            if str(mode).lower() == "true":
                log.warning("spaCy requested but unavailable: %s", exc)
            cls.nlp = None
        return cls.nlp


class EntityResolver:
    def __init__(self, cfg: Config | None = None, use_spacy: str | None = None, fuzzy: bool = True):
        self.cfg = cfg or get_config()
        self.by_ticker: dict[str, Entity] = {e.ticker: e for e in self.cfg.entities}
        self.alias_owner: dict[str, Entity] = {}
        for e in self.cfg.entities:
            for a in (*e.aliases, e.name):
                self.alias_owner.setdefault(a, e)
        alts = "|".join(re.escape(a) for a in sorted(self.alias_owner, key=len, reverse=True))
        # Case-sensitive: proper nouns only. "apple" (fruit) never matches "Apple".
        self.alias_re = re.compile(r"(?<![A-Za-z0-9])(" + alts + r")(?![A-Za-z0-9])")
        ci = [a for a, e in self.alias_owner.items() if len(a) >= 5 and a not in e.ambiguous and a not in CI_EXCLUDE]
        self.alias_ci_re = re.compile(r"(?<![A-Za-z0-9])(" + "|".join(re.escape(a) for a in sorted(ci, key=len, reverse=True)) + r")(?![A-Za-z0-9])", re.I)
        self.alias_ci_map = {a.lower(): self.alias_owner[a] for a in ci}
        self.long_tickers = {t for t in self.by_ticker if len(t) >= 4 and t not in {"META"}}
        self.ticker_re = re.compile(r"(?<![$A-Za-z0-9])(" + "|".join(sorted(self.long_tickers, key=len, reverse=True)) + r")(?![A-Za-z0-9])")
        self.use_spacy = use_spacy if use_spacy is not None else str(self.cfg.section("engine").get("use_spacy", "auto"))
        self.fuzzy = fuzzy
        self._fuzzy_choices = [a for a in self.alias_owner if len(a) >= 5]

    def _add(self, out: dict[str, EntityMatch], e: Entity, how: str, conf: float, span: tuple[int, int]):
        if e.ticker in out:
            m = out[e.ticker]
            m.spans.append(span)
            if conf > m.confidence:
                m.confidence, m.match = conf, how
            return
        out[e.ticker] = EntityMatch(e.ticker, e.name, how, conf, [span], e.in_index, e.sector)

    def _spacy_orgs(self, text: str) -> list[tuple[int, int]] | None:
        nlp = _Spacy.get(self.use_spacy)
        if nlp is None:
            return None
        doc = nlp(text)
        return [(ent.start_char, ent.end_char) for ent in doc.ents if ent.label_ in {"ORG", "PRODUCT", "PERSON", "GPE"}]

    def resolve(self, text: str) -> Resolution:
        out: dict[str, EntityMatch] = {}
        rejected: list[dict] = []
        for m in CASHTAG.finditer(text):
            e = self.by_ticker.get(m.group(1))
            if e:
                self._add(out, e, "cashtag", 0.99, m.span())
        for m in EXCHANGE.finditer(text):
            e = self.by_ticker.get(m.group(1))
            if e:
                self._add(out, e, "exchange", 0.98, m.span())
        for m in self.ticker_re.finditer(text):
            self._add(out, self.by_ticker[m.group(1)], "ticker", 0.95, m.span())

        orgs = None
        for m in self.alias_re.finditer(text):
            alias = m.group(1)
            e = self.alias_owner[alias]
            if alias in e.ambiguous:
                rest = text[: m.start()] + text[m.end() :]
                has_ctx = bool(FIN_CONTEXT.search(rest)) or e.ticker in out
                if orgs is None:
                    orgs = self._spacy_orgs(text)
                spacy_ok = True if orgs is None else any(a <= m.start() < b for a, b in orgs)
                if not has_ctx or not spacy_ok:
                    rejected.append({"alias": alias, "ticker": e.ticker,
                                     "reason": "ambiguous common noun without financial context" if not has_ctx else "spaCy did not tag span as an organization"})
                    continue
                self._add(out, e, "alias-ambiguous", 0.86, m.span())
            else:
                self._add(out, e, "alias", 0.94, m.span())

        # lower-case mentions in social posts ("credit suisse", "nvidia"): multi-word aliases always,
        # single-word aliases only with financial context
        for m in self.alias_ci_re.finditer(text):
            if m.group(1)[0].isupper():
                continue
            e = self.alias_ci_map.get(m.group(1).lower())
            if e is None or e.ticker in out:
                continue
            if " " in m.group(1) or FIN_CONTEXT.search(text):
                self._add(out, e, "alias-lowercase", 0.8, m.span())

        if self.fuzzy and not out:
            out.update(self._fuzzy(text))
        if re.search(r"\bapple\b", text) and "AAPL" not in out and not any(r["ticker"] == "AAPL" for r in rejected):
            rejected.append({"alias": "apple", "ticker": "AAPL", "reason": "lower-case common noun, not an organization"})
        return Resolution(sorted(out.values(), key=lambda m: m.spans[0][0]), rejected)

    def _fuzzy(self, text: str) -> dict[str, EntityMatch]:
        try:
            from rapidfuzz import fuzz, process
        except Exception:  # pragma: no cover
            return {}
        out: dict[str, EntityMatch] = {}
        for m in re.finditer(r"\b[A-Z][a-zA-Z&\.]{4,}(?:\s+[A-Z][a-zA-Z&\.]+)?", text):
            cand = m.group(0)
            best = process.extractOne(cand, self._fuzzy_choices, scorer=fuzz.ratio, score_cutoff=88)
            if best:
                e = self.alias_owner[best[0]]
                if best[0] in e.ambiguous:
                    continue
                self._add(out, e, "fuzzy", round(0.6 + 0.3 * (best[1] - 88) / 12, 3), m.span())
        return out


@lru_cache(maxsize=1)
def default_resolver() -> EntityResolver:
    return EntityResolver()
