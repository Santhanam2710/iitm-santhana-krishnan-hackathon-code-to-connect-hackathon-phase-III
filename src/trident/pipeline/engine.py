"""The AI/NLP Risk Engine: Doc -> list[Signal].

Stages (timed individually): normalize/dedup -> (1) entity resolution -> (2) target-aware
sentiment -> (3) event classification -> temporal check -> (4) impact -> (5) template
rationale -> Signal. Model calls are batched in ``process_batch`` for throughput.
"""
from __future__ import annotations

import re
import time
from collections import deque
from dataclasses import dataclass, field

from trident import __version__
from trident.config import get_config
from trident.nlp.entities import EntityResolver
from trident.nlp.events import EventClassifier
from trident.nlp.impact import ImpactScorer, build_features
from trident.nlp.lexicon import find_hits
from trident.nlp.rationale import template
from trident.nlp.sentiment import SentimentScorer
from trident.nlp.temporal import is_retrospective
from trident.pipeline.dedup import Deduper
from trident.pipeline.novelty import NoveltyTracker, VelocityTracker
from trident.signals.schema import (Doc, EntityInfo, EventInfo, Evidence, Lineage, Rationale, Signal,
                                    SourceInfo)
from trident.utils.common import iso, parse_ts, short_hash, ulid, utcnow

SECTOR_WORDS = {
    "Banks": re.compile(r"\b(banks?|banking|lenders?|regional banks?|bank stocks|depositors?|deposits)\b", re.I),
    "Energy": re.compile(r"\b(oil|crude|energy stocks|opec|natural gas|brent)\b", re.I),
    "Technology": re.compile(r"\b(tech stocks|chipmakers?|semiconductors?|big tech)\b", re.I),
    "Industrials": re.compile(r"\b(airlines?|automakers?|aerospace|carmakers?)\b", re.I),
}
MARKET_CLASSES = {"Geopolitical", "Macroeconomic"}


def _spacy_loaded() -> bool:
    from trident.nlp.entities import _Spacy

    return _Spacy.nlp is not None


@dataclass
class EngineOutput:
    doc: Doc
    signals: list[Signal] = field(default_factory=list)
    dropped: str | None = None
    rejected_entities: list[dict] = field(default_factory=list)
    timings_ms: dict[str, float] = field(default_factory=dict)


class Engine:
    def __init__(self, sentiment: SentimentScorer | None = None, events: EventClassifier | None = None,
                 impact: ImpactScorer | None = None, resolver: EntityResolver | None = None):
        cfg = get_config()
        eng = cfg.section("engine")
        self.cfg = cfg
        self.resolver = resolver or EntityResolver(cfg)
        self.sentiment = sentiment or SentimentScorer()
        self.events = events or EventClassifier()
        self.impact = impact or ImpactScorer()
        self.dedup = Deduper(threshold=float(eng.get("dedup_threshold", 0.8)))
        self.novelty = NoveltyTracker(window_hours=float(eng.get("novelty_window_hours", 24)))
        self.velocity = VelocityTracker()
        self.reliability = {k: float(v) for k, v in cfg.section("source_reliability").items()}
        self.latencies: deque[float] = deque(maxlen=5000)
        self.last_batch_ms = 0.0
        self.counters = {"docs": 0, "signals": 0, "dropped_duplicate": 0, "dropped_language": 0,
                         "unrouted": 0, "rejected_entities": 0}

    def warmup(self) -> None:
        """Load lazy resources (lexicons, regexes, vectorizers, models) before the first real document."""
        texts = ["JPMorgan downgraded as losses mount", "Fed signals rate hikes as inflation rises", "$AAPL beats estimates"]
        for t in texts:
            self.resolver.resolve(t)
            find_hits(t)
        self.sentiment.score_many(texts)
        self.events.classify_many(texts)
        self.novelty._vec(texts[0])

    @property
    def model_versions(self) -> dict[str, str]:
        return {"sentiment": self.sentiment.name, "event": self.events.name, "impact": self.impact.name,
                "ner": "alias-trie+cashtag" + ("+spacy" if _spacy_loaded() else "")}

    def process(self, doc: Doc) -> EngineOutput:
        return self.process_batch([doc])[0]

    def process_batch(self, docs: list[Doc]) -> list[EngineOutput]:
        t_batch = time.perf_counter()
        outs: list[EngineOutput] = []
        live: list[tuple[EngineOutput, object, object, object]] = []
        for doc in docs:
            self.counters["docs"] += 1
            out = EngineOutput(doc)
            t0 = time.perf_counter()
            if doc.lang != "en":
                out.dropped = "non-English"
                self.counters["dropped_language"] += 1
                outs.append(out)
                continue
            ts = parse_ts(doc.ts_published)
            dd = self.dedup.check(doc.doc_id, doc.text, doc.source_name, ts.timestamp())
            if dd.is_duplicate:
                out.dropped = f"duplicate of cluster {dd.cluster_id}"
                self.counters["dropped_duplicate"] += 1
                outs.append(out)
                continue
            novelty = 0.15 if dd.is_syndicated else self.novelty.score(doc.text, ts)
            t1 = time.perf_counter()
            res = self.resolver.resolve(doc.text)
            out.rejected_entities = res.rejected
            self.counters["rejected_entities"] += len(res.rejected)
            t2 = time.perf_counter()
            out.timings_ms.update({"normalize_dedup": (t1 - t0) * 1e3, "entities": (t2 - t1) * 1e3})
            live.append((out, dd, res, novelty))
            outs.append(out)

        texts = [o.doc.text for o, *_ in live]
        t3 = time.perf_counter()
        sents = self.sentiment.score_many(texts)
        t4 = time.perf_counter()
        evs = self.events.classify_many(texts)
        t5 = time.perf_counter()
        n = max(len(live), 1)
        per_sent, per_ev = (t4 - t3) * 1e3 / n, (t5 - t4) * 1e3 / n

        for (out, dd, res, novelty), sr, ev in zip(live, sents, evs):
            t6 = time.perf_counter()
            doc = out.doc
            ts = parse_ts(doc.ts_published)
            targets = [(m.ticker, m.spans) for m in res.matches]
            per_target = self.sentiment.score_targets(doc.text, targets, sr) if targets else {}
            infos = {m.ticker: EntityInfo(ticker=m.ticker, name=m.name, match=m.match, confidence=m.confidence,
                                          in_index=m.in_index, sector=m.sector) for m in res.matches}
            market_wide = False
            hint = doc.meta.get("ticker_hint")
            if not infos and hint and self.cfg.entity(str(hint)):
                e = self.cfg.entity(str(hint))
                infos[e.ticker] = EntityInfo(ticker=e.ticker, name=e.name, match="source-tag", confidence=0.9, in_index=e.in_index, sector=e.sector)
                per_target[e.ticker] = sr.score
            if not infos:
                sector = next((s for s, pat in SECTOR_WORDS.items() if pat.search(doc.text)), None)
                if ev.label in MARKET_CLASSES:
                    market_wide = True
                    infos["MARKET"] = EntityInfo(ticker="MARKET", name="Index-level", match="scope", confidence=0.9, sector="Market")
                    per_target["MARKET"] = sr.score
                if sector and ev.label != "Other":
                    key = f"SECTOR:{sector}"
                    infos[key] = EntityInfo(ticker=key, name=f"{sector} sector", match="scope", confidence=0.8, sector=sector)
                    per_target[key] = sr.score
                    market_wide = ev.label in MARKET_CLASSES
                elif ev.label in {"Credit Event"}:
                    infos["MARKET"] = EntityInfo(ticker="MARKET", name="Index-level", match="scope", confidence=0.7, sector="Market")
                    per_target["MARKET"] = sr.score
            if not infos:
                out.dropped = "no resolvable entity and not market-wide"
                self.counters["unrouted"] += 1
                continue
            retro, retro_reason = is_retrospective(doc.text, ts)
            rel = self.reliability.get(doc.source_type, 0.8)
            hits = find_hits(doc.text)
            evidence = [Evidence(span=h.text, start=h.start, end=h.end, polarity="pos" if h.weight > 0 else "neg")
                        for h in hits if h.source == "phrase"][:6]
            evidence += [Evidence(span=doc.text[a:b], start=a, end=b, polarity="event") for a, b, c in ev.spans if c == ev.label][:3]
            flags = []
            if doc.is_synthetic:
                flags.append("synthetic")
            if dd.is_syndicated:
                flags.append(f"syndicated x{dd.cluster_sources}")
            if retro:
                flags.append(f"retrospective ({retro_reason})")
            if sr.disagreement > 1.0:
                flags.append("model-lexicon disagreement")
            t7 = time.perf_counter()
            for ticker, info in infos.items():
                s_t = float(per_target.get(ticker, sr.score))
                vz = self.velocity.update(ticker, ts)
                feats = build_features(doc.text, ev.label, abs(s_t), novelty, vz, rel, market_wide, len(infos))
                impact, raw = self.impact.score(feats)
                if retro:
                    impact = min(impact, 4)
                if ev.label == "Other":
                    impact = min(impact, 3)
                rationale = template(ev.label, ev.confidence, s_t, [e.span for e in evidence if e.polarity in ("pos", "neg")],
                                     impact, feats.base_severity, feats.magnitude, market_wide, doc.source_type, flags)
                now = utcnow()
                lat_engine = (t7 - t6) * 1e3 + out.timings_ms.get("normalize_dedup", 0) + out.timings_ms.get("entities", 0) + per_sent + per_ev
                sig = Signal(
                    signal_id="sig_" + ulid(),
                    ts_published=doc.ts_published,
                    ts_ingested=doc.ts_ingested,
                    ts_emitted=iso(now),
                    source=SourceInfo(type=doc.source_type, name=doc.source_name, reliability=rel, url=doc.url,
                                      cluster_id=dd.cluster_id, novelty=novelty),
                    entity=info,
                    sentiment_score=round(s_t, 4),
                    sentiment_confidence=round(sr.confidence, 3),
                    event_classification=EventInfo(label=ev.label, confidence=round(ev.confidence, 3), secondary=ev.secondary,
                                                   subtype=ev.subtype, probabilities=ev.probs),
                    impact_score=impact,
                    evidence=evidence,
                    rationale=Rationale(text=rationale, mode="template"),
                    lineage=Lineage(engine_version=__version__, models=self.model_versions, is_synthetic=doc.is_synthetic,
                                    replay_id=doc.replay_id, doc_id=doc.doc_id, doc_hash=short_hash(doc.text)),
                    latency_ms={"engine_total": round(lat_engine, 3), "sentiment": round(per_sent, 3), "event": round(per_ev, 3),
                                "source_lag_s": round(max(0.0, (parse_ts(doc.ts_ingested) - ts).total_seconds()), 1) if not doc.replay_id else 0.0},
                    flags=flags,
                    text=doc.text,
                )
                out.signals.append(sig)
                self.latencies.append(lat_engine)
                self.counters["signals"] += 1
            out.timings_ms.update({"sentiment": per_sent, "event": per_ev, "impact_build": (time.perf_counter() - t7) * 1e3})
        self.last_batch_ms = (time.perf_counter() - t_batch) * 1e3
        return outs
