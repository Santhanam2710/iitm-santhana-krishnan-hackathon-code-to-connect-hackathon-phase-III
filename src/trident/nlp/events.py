"""Event classification into the 8-class taxonomy plus "Other" (abstain).

Backends (``engine.event_backend``):
  rules     keyword evidence from configs/taxonomy.yaml with class priors and softmax
  sklearn   TF-IDF + logistic regression trained on weak labels + gold (train_events.py)
  zeroshot  NLI zero-shot (facebook/bart-large-mnli); slow, used for bootstrapping labels
  deberta   fine-tuned DeBERTa-v3-small (finetune_deberta_events.py)
  auto      deberta (if trained) > sklearn+rules hybrid (if trained) > rules
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Sequence

import numpy as np

from trident.config import MODELS_DIR, get_config
from trident.signals.schema import EVENT_LABELS
from trident.utils.common import get_logger

log = get_logger("trident.events")


def _compile(words: list[str]) -> re.Pattern[str] | None:
    if not words:
        return None
    alts = "|".join(re.escape(w) for w in sorted(words, key=len, reverse=True))
    return re.compile(r"(?<![A-Za-z0-9])(" + alts + r")(?![A-Za-z0-9])", re.I)


@dataclass
class EventResult:
    label: str
    confidence: float
    probs: dict[str, float]
    secondary: str | None = None
    subtype: str | None = None
    spans: list[tuple[int, int, str]] = field(default_factory=list)  # (start, end, class)


class RulesBackend:
    name = "rules-v1"

    def __init__(self):
        tax = get_config().taxonomy
        self.classes = tax["classes"]
        self.patterns = {c: _compile(v.get("keywords", [])) for c, v in self.classes.items()}
        self.priors = {c: float(v.get("prior", 1.0)) for c, v in self.classes.items()}
        self.temperature = 0.45

    def evidence(self, text: str) -> tuple[dict[str, float], list[tuple[int, int, str]]]:
        scores: dict[str, float] = {}
        spans: list[tuple[int, int, str]] = []
        for c, pat in self.patterns.items():
            if pat is None:
                continue
            terms = set()
            for m in pat.finditer(text):
                terms.add(m.group(1).lower())
                spans.append((m.start(), m.end(), c))
            scores[c] = len(terms) * self.priors[c]
        return scores, spans

    def predict_one(self, text: str) -> tuple[np.ndarray, list[tuple[int, int, str]]]:
        scores, spans = self.evidence(text)
        scores["Other"] = 0.35
        logits = np.array([scores.get(c, 0.0) / self.temperature for c in EVENT_LABELS])
        e = np.exp(logits - logits.max())
        return e / e.sum(), spans

    def predict(self, texts: Sequence[str]) -> np.ndarray:
        return np.vstack([self.predict_one(t)[0] for t in texts]) if texts else np.zeros((0, len(EVENT_LABELS)))


class SklearnEventBackend:
    def __init__(self, path):
        import joblib

        self.pipe = joblib.load(path)
        cls = list(self.pipe.classes_)
        self.idx = [cls.index(c) if c in cls else None for c in EVENT_LABELS]
        self.name = "sklearn-events"

    def predict(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, len(EVENT_LABELS)))
        p = self.pipe.predict_proba(list(texts))
        out = np.zeros((len(texts), len(EVENT_LABELS)))
        for j, i in enumerate(self.idx):
            if i is not None:
                out[:, j] = p[:, i]
        return out / out.sum(axis=1, keepdims=True)


class HFSequenceBackend:  # pragma: no cover - requires torch + transformers
    def __init__(self, model_dir: str):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir).eval()
        id2 = {int(k): v for k, v in self.model.config.id2label.items()}
        self.idx = [next((i for i, n in id2.items() if n == c), None) for c in EVENT_LABELS]
        self.name = "deberta-v3-small-ft"

    def predict(self, texts: Sequence[str]) -> np.ndarray:
        with self.torch.no_grad():
            enc = self.tok(list(texts), padding=True, truncation=True, max_length=96, return_tensors="pt")
            p = self.torch.softmax(self.model(**enc).logits, dim=-1).numpy()
        out = np.zeros((len(texts), len(EVENT_LABELS)))
        for j, i in enumerate(self.idx):
            if i is not None:
                out[:, j] = p[:, i]
        return out


class ZeroShotBackend:  # pragma: no cover - requires torch + transformers
    def __init__(self, model_id: str = "facebook/bart-large-mnli"):
        from transformers import pipeline

        self.pipe = pipeline("zero-shot-classification", model=model_id)
        self.labels = [c for c in EVENT_LABELS if c != "Other"]
        self.name = f"zeroshot@{model_id.split('/')[-1]}"

    def predict(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), len(EVENT_LABELS)))
        res = self.pipe(list(texts), candidate_labels=self.labels, hypothesis_template="This news is about {}.", multi_label=False)
        res = res if isinstance(res, list) else [res]
        for i, r in enumerate(res):
            for lab, sc in zip(r["labels"], r["scores"]):
                out[i, EVENT_LABELS.index(lab)] = sc
            top = max(r["scores"])
            out[i, EVENT_LABELS.index("Other")] = max(0.0, 0.5 - top)
        return out / out.sum(axis=1, keepdims=True)


class EventClassifier:
    def __init__(self, backend: str | None = None):
        cfg = get_config()
        self.rules = RulesBackend()
        self.abstain = float(cfg.taxonomy.get("abstain_threshold", 0.45))
        self.subtypes = {k: _compile(v) for k, v in cfg.taxonomy["classes"]["Credit Event"].get("subtypes", {}).items()}
        name = (backend or cfg.section("engine").get("event_backend", "auto")).lower()
        self.model = None
        deb = MODELS_DIR / "events-deberta"
        sk = MODELS_DIR / "events_sklearn.joblib"
        try:
            if name == "deberta" or (name == "auto" and (deb / "config.json").exists()):
                self.model = HFSequenceBackend(str(deb))
            elif name == "zeroshot":
                self.model = ZeroShotBackend()
            elif name == "sklearn" or (name == "auto" and sk.exists()):
                self.model = SklearnEventBackend(sk)
        except Exception as exc:
            log.warning("event backend %s unavailable (%s); using rules", name, exc)
            self.model = None

    @property
    def name(self) -> str:
        return f"{self.model.name}+rules" if self.model else self.rules.name

    def classify_many(self, texts: Sequence[str]) -> list[EventResult]:
        model_p = self.model.predict(texts) if self.model is not None and texts else None
        out = []
        for i, t in enumerate(texts):
            rp, spans = self.rules.predict_one(t)
            p = rp if model_p is None else 0.5 * rp + 0.5 * model_p[i]
            order = np.argsort(-p)
            top, second = EVENT_LABELS[order[0]], EVENT_LABELS[order[1]]
            conf = float(p[order[0]])
            evidence_strength = sum(1 for s in spans if s[2] == top)
            if top != "Other" and conf < self.abstain and evidence_strength == 0:
                top, second = "Other", top
            subtype = None
            if top == "Credit Event":
                for name, pat in self.subtypes.items():
                    if pat and pat.search(t):
                        subtype = name
                        break
            out.append(EventResult(top, conf, {c: round(float(v), 4) for c, v in zip(EVENT_LABELS, p)},
                                   second if second != top else None, subtype, spans))
        return out

    def classify(self, text: str) -> EventResult:
        return self.classify_many([text])[0]


@lru_cache(maxsize=1)
def default_classifier() -> EventClassifier:
    return EventClassifier()
