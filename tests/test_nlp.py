"""Unit and behavioral tests for the NLP stages."""
import pytest

from trident.eval.checklist import run as checklist
from trident.nlp.entities import EntityResolver
from trident.nlp.events import EventClassifier
from trident.nlp.impact import ImpactScorer, OrdinalLogit, build_features, magnitude_cues
from trident.nlp.lexicon import score_span
from trident.nlp.sentiment import SentimentScorer
from trident.nlp.temporal import is_retrospective
from trident.utils.common import parse_ts

R = EntityResolver(use_spacy="false")
S = SentimentScorer()
E = EventClassifier(backend="rules")


@pytest.mark.parametrize("text,ticker", [
    ("JPMorgan downgraded two notches", "JPM"), ("$NVDA ripping", "NVDA"), ("(NASDAQ: AAPL) shares fell", "AAPL"),
    ("Silicon Valley Bank seized by regulators", "SIVB"), ("credit suisse cds going vertical", "CS"), ("Meta shares jump after earnings", "META"),
])
def test_entity_resolves(text, ticker):
    assert ticker in {m.ticker for m in R.resolve(text).matches}


@pytest.mark.parametrize("text", ["I ate an apple pie", "Meta-analysis shows coffee helps", "grabbed a coke after the game"])
def test_ambiguous_entities_rejected(text):
    res = R.resolve(text)
    assert not res.matches


def test_sentiment_range_and_direction():
    pos, neg = S.score("Nvidia beats expectations and raises guidance"), S.score("Boeing shares plunge after profit warning")
    assert -1 <= neg.score < 0 < pos.score <= 1
    assert 0 <= pos.confidence <= 1


def test_negation_and_delta():
    assert S.score("Apple did not beat expectations").score < 0
    assert S.score("Pfizer losses narrowed in the quarter").score > 0


def test_target_dependent_scores():
    text = "Microsoft beats Google to acquire gaming studio while Amazon slumps"
    targets = [(m.ticker, m.spans) for m in R.resolve(text).matches]
    sc = S.score_targets(text, targets)
    assert sc["MSFT"] > 0 > sc["GOOGL"]
    assert sc["AMZN"] < 0


@pytest.mark.parametrize("text,label", [
    ("JPMorgan downgraded to junk", "Credit Event"), ("US imposes sanctions after missile strikes", "Geopolitical"),
    ("Fed signals more rate hikes as CPI rises", "Macroeconomic"), ("Microsoft agrees to acquire rival for $5 billion", "Merger/Acquisition"),
    ("Tesla misses estimates and cuts guidance", "Earnings"), ("Google hit with antitrust lawsuit", "Regulatory/Legal"),
    ("Microsoft suffers global outage", "Operational"), ("Nvidia unveils next-gen GPUs", "Product Launch"),
    ("The weather was lovely today", "Other"),
])
def test_event_classes(text, label):
    assert E.classify(text).label == label


def test_credit_subtype():
    assert E.classify("JPMorgan downgraded two notches").subtype == "downgrade"


def test_impact_bounds_and_monotonic_severity():
    imp = ImpactScorer(model_path=__import__("pathlib").Path("/nonexistent"))
    lo = imp.score(build_features("Apple launches a new color", "Product Launch", 0.2, 1.0, 0, 1.0, False, 1))[0]
    hi = imp.score(build_features("Sweeping sanctions; oil spikes 9%", "Geopolitical", 0.9, 1.0, 0, 1.0, True, 0))[0]
    assert 1 <= lo < hi <= 10


def test_magnitude_cues_counts_largest_amount_once():
    assert magnitude_cues("$21 billion, $1.8 billion and $2.25 billion") == magnitude_cues("$21 billion")


def test_retrospective_detection():
    assert is_retrospective("Fifteen years after Lehman, banks reflect", parse_ts("2023-03-11T00:00:00Z"))[0]
    assert not is_retrospective("Bank shares plunge today", parse_ts("2023-03-11T00:00:00Z"))[0]


def test_ordinal_logit_learns_order():
    import numpy as np

    rng = np.random.default_rng(0)
    x = rng.normal(size=(600, 2))
    y = np.clip(np.digitize(x[:, 0] + 0.3 * rng.normal(size=600), np.quantile(x[:, 0], np.linspace(0.1, 0.9, 9))) + 1, 1, 10)
    m = OrdinalLogit(10).fit(x, y)
    pred = m.predict_score(x)
    assert np.corrcoef(pred, y)[0, 1] > 0.8
    m2 = OrdinalLogit.from_json(m.to_json())
    assert np.allclose(m2.predict_score(x[:5]), pred[:5])


def test_checklist_suite_passes():
    assert checklist()["_summary"]["rate"] >= 0.95


def test_lexicon_hits_have_spans():
    r = score_span("Shares plunge after downgrade")
    assert all(0 <= h.start < h.end for h in r.hits)
