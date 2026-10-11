"""Engine and pipeline tests: schema, dedup, routing, replay integration."""
from trident.ingest.replay import load_replay, replay_docs
from trident.ingest.synthetic import TemplateGenerator, hawkes_times
from trident.pipeline.dedup import Deduper
from trident.pipeline.engine import Engine
from trident.pipeline.normalize import looks_english, make_doc
from trident.signals.schema import Signal


def test_signal_schema_and_required_fields():
    out = Engine().process(make_doc("JPMorgan downgraded two notches as commercial real estate losses mount", "newswire", "t"))
    assert out.signals
    s = out.signals[0]
    Signal.model_validate(s.model_dump())
    assert -1 <= s.sentiment_score <= 1 and 1 <= s.impact_score <= 10
    assert s.event_classification.label == "Credit Event"
    assert s.lineage.models["sentiment"] and s.evidence


def test_same_source_duplicate_dropped_cross_source_kept():
    eng = Engine()
    a = make_doc("Bank X shares plunge after deposit outflows hit record levels today", "newswire", "a", "2024-01-01T10:00:00Z")
    b = make_doc("Bank X shares plunge after deposit outflows hit record levels today", "newswire", "a", "2024-01-01T10:05:00Z")
    c = make_doc("Bank X shares plunge after deposit outflows hit record levels today", "newswire", "b", "2024-01-01T10:06:00Z")
    outs = eng.process_batch([a, b, c])
    assert outs[1].dropped and outs[1].dropped.startswith("duplicate")
    assert outs[2].dropped is None


def test_dedup_window():
    d = Deduper(threshold=0.7, window_hours=48)
    t = "Regional bank shares plunge after deposit outflows accelerate across lenders"
    d.check("1", t, "a", 0)
    assert not d.check("2", t, "a", 10 * 86400).is_duplicate  # ten days later is a new story


def test_market_and_sector_routing():
    eng = Engine()
    m = eng.process(make_doc("Fed signals two more rate hikes as CPI surprises to upside", "newswire", "x"))
    assert m.signals[0].entity.ticker == "MARKET"
    s = eng.process(make_doc("Regional bank stocks plunge as deposit flight spreads across lenders", "newswire", "y"))
    assert any(x.entity.ticker == "SECTOR:Banks" for x in s.signals)


def test_unrouted_and_language():
    eng = Engine()
    assert eng.process(make_doc("Just baked an apple pie", "social", "z")).dropped
    assert not looks_english("Les marchés reculent après la décision de la banque centrale européenne à Francfort")


def test_replay_files_ordered_and_tagged():
    rows = load_replay("svb_2023-03")
    assert rows == sorted(rows, key=lambda r: r["ts"])
    assert all("provenance" in r for r in rows)
    docs = replay_docs("svb_2023-03")
    assert all(d.replay_id == "svb_2023-03" for d in docs)


def test_synthetic_generator_tags_and_hawkes():
    import random
    from datetime import datetime, timezone

    rows = TemplateGenerator(seed=1).generate(50)
    assert all(r["is_synthetic"] and r["generator"] for r in rows)
    times = hawkes_times(datetime(2024, 1, 1, tzinfo=timezone.utc), 24, 0.5, 0.8, 1.2, random.Random(0))
    assert times == sorted(times)
