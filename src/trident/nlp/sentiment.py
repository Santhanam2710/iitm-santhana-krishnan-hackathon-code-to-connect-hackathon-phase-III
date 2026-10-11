"""Sentiment scoring with pluggable backends and target-dependent scores.

Backends (``engine.sentiment_backend``):
  lexicon       finance phrase lexicon + LM words (always available, no training)
  sklearn       TF-IDF + logistic regression trained by training/train_sentiment.py
  finbert       Hugging Face FinBERT (ProsusAI/finbert or a fine-tuned local copy)
  finbert-onnx  FinBERT exported to ONNX INT8 by training/export_onnx.py
  auto          finbert-onnx > finbert (local fine-tuned dir) > sklearn > lexicon

The model score is blended with the lexicon (eq. E1). Target-dependent scores are
computed per entity from the clause that mentions it, with a rule for competitive
verbs ("X beats Y": X positive, Y negative).
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Protocol, Sequence

import numpy as np

from trident.config import MODELS_DIR, get_config
from trident.nlp import lexicon
from trident.utils.common import get_logger

log = get_logger("trident.sentiment")
LABELS = ("negative", "neutral", "positive")

DELTA_POS = re.compile(r"\b(losses? narrowed|narrower loss|smaller[- ]than[- ]expected loss|less bad than feared|beat (lowered|reduced) estimates)\b", re.I)
DELTA_NEG = re.compile(r"\b(profit (fell|slid|dropped)|gains? (faded|evaporated)|growth slowed|missed (lowered|reduced) estimates)\b", re.I)
COMPETE = re.compile(r"\b(beats|beat|outpaces|outsells|overtakes|edges out|wins .{0,30}? over|takes (market )?share from|outbids|tops)\b", re.I)
CLAUSE_SPLIT = re.compile(r";|\bbut\b|\bwhile\b|\bwhereas\b|\bhowever\b|\bas\b(?=\s+\w+\s+(falls|drops|slumps|rises|jumps))", re.I)


class Backend(Protocol):
    name: str

    def predict(self, texts: Sequence[str]) -> np.ndarray:  # (n, 3) probs in LABELS order
        ...


def _lex_probs(s: float, conf: float) -> np.ndarray:
    p_neu = max(0.04, min(0.9, 1 - abs(s))) * (1 - conf * 0.4)
    rest = 1 - p_neu
    return np.array([rest * (1 - s) / 2, p_neu, rest * (1 + s) / 2])


class LexiconBackend:
    name = "lexicon-v1"

    def predict(self, texts: Sequence[str]) -> np.ndarray:
        out = []
        for t in texts:
            r = lexicon.score_span(t)
            out.append(_lex_probs(r.score, r.confidence))
        return np.vstack(out) if out else np.zeros((0, 3))


class SklearnBackend:
    def __init__(self, path: Path):
        import joblib

        self.pipe = joblib.load(path)
        self.order = [list(self.pipe.classes_).index(l) for l in LABELS]
        self.name = f"sklearn-tfidf-lr@{path.stat().st_mtime_ns % 10**6:06d}"

    def predict(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 3))
        return self.pipe.predict_proba(list(texts))[:, self.order]


def _label_order(id2label: dict) -> list[int]:
    names = {int(k): str(v).lower() for k, v in id2label.items()}
    idx = {}
    for i, n in names.items():
        for l in LABELS:
            if n.startswith(l[:3]):
                idx[l] = i
    return [idx[l] for l in LABELS]


class FinbertBackend:  # pragma: no cover - requires torch + transformers
    def __init__(self, model_id: str):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_id).eval()
        self.order = _label_order(self.model.config.id2label)
        self.name = f"finbert@{Path(model_id).name}"

    def predict(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 3))
        with self.torch.no_grad():
            enc = self.tok(list(texts), padding=True, truncation=True, max_length=96, return_tensors="pt")
            logits = self.model(**enc).logits
            p = self.torch.softmax(logits, dim=-1).numpy()
        return p[:, self.order]


class OnnxFinbertBackend:  # pragma: no cover - requires onnxruntime + exported model
    def __init__(self, model_dir: Path):
        import json

        import onnxruntime as ort
        from transformers import AutoTokenizer

        onnx_file = model_dir / "model_quant.onnx"
        if not onnx_file.exists():
            onnx_file = model_dir / "model.onnx"
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(str(onnx_file), so, providers=["CPUExecutionProvider"])
        self.tok = AutoTokenizer.from_pretrained(model_dir)
        cfg = json.loads((model_dir / "config.json").read_text())
        self.order = _label_order(cfg["id2label"])
        self.inputs = {i.name for i in self.sess.get_inputs()}
        self.name = f"finbert-onnx-int8@{onnx_file.stat().st_size % 10**6:06d}"

    def predict(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 3))
        enc = self.tok(list(texts), padding=True, truncation=True, max_length=96, return_tensors="np")
        feed = {k: v.astype(np.int64) for k, v in enc.items() if k in self.inputs}
        logits = self.sess.run(None, feed)[0]
        e = np.exp(logits - logits.max(axis=1, keepdims=True))
        p = e / e.sum(axis=1, keepdims=True)
        return p[:, self.order]


def load_backend(name: str | None = None) -> Backend:
    cfg = get_config().section("engine")
    name = (name or cfg.get("sentiment_backend", "auto")).lower()
    onnx_dir = MODELS_DIR / "finbert-onnx"
    ft_dir = MODELS_DIR / "finbert-ft"
    sk = MODELS_DIR / "sentiment_sklearn.joblib"
    try:
        if name in ("finbert-onnx", "onnx") or (name == "auto" and (onnx_dir / "config.json").exists()):
            return OnnxFinbertBackend(onnx_dir)
        if name == "finbert" or (name == "auto" and (ft_dir / "config.json").exists()):
            return FinbertBackend(str(ft_dir) if (ft_dir / "config.json").exists() else "ProsusAI/finbert")
        if name == "sklearn" or (name == "auto" and sk.exists()):
            return SklearnBackend(sk)
    except Exception as exc:
        log.warning("sentiment backend %s unavailable (%s); falling back to lexicon", name, exc)
    return LexiconBackend()


@dataclass
class SentimentResult:
    score: float
    confidence: float
    probs: tuple[float, float, float]
    model_score: float
    lexicon_score: float
    disagreement: float


def _entropy_conf(p: np.ndarray) -> float:
    p = np.clip(p, 1e-9, 1)
    return float(1 - (-(p * np.log(p)).sum()) / math.log(3))


class SentimentScorer:
    def __init__(self, backend: Backend | None = None, alpha: float | None = None):
        self.backend = backend or load_backend()
        self.alpha = alpha if alpha is not None else float(get_config().section("engine").get("ensemble_alpha", 0.8))
        self.is_lexicon = isinstance(self.backend, LexiconBackend)

    @property
    def name(self) -> str:
        return self.backend.name

    def score_many(self, texts: Sequence[str]) -> list[SentimentResult]:
        probs = self.backend.predict(texts)
        out = []
        for t, p in zip(texts, probs):
            lex = lexicon.score_span(t)
            s_model = float(p[2] - p[0])
            if DELTA_POS.search(t):
                s_model = max(s_model, 0.35)
            if DELTA_NEG.search(t):
                s_model = min(s_model, -0.35)
            if self.is_lexicon:
                s, conf, dis = lex.score, lex.confidence, 0.0
            else:
                s = self.alpha * s_model + (1 - self.alpha) * lex.score
                dis = abs(s_model - lex.score) if lex.hits else 0.0
                conf = _entropy_conf(p) * (1 - 0.5 * min(dis, 1.0))
            out.append(SentimentResult(float(np.clip(s, -1, 1)), float(np.clip(conf, 0, 1)), tuple(map(float, p)), s_model, lex.score, dis))
        return out

    def score(self, text: str) -> SentimentResult:
        return self.score_many([text])[0]

    def score_targets(self, text: str, targets: list[tuple[str, list[tuple[int, int]]]], base: SentimentResult | None = None) -> dict[str, float]:
        """Per-target scores. ``targets`` is [(ticker, spans)]. Falls back to the sentence score."""
        base = base or self.score(text)
        if len(targets) <= 1:
            return {t: base.score for t, _ in targets}
        result = {t: base.score for t, _ in targets}
        # 1) competitive verbs: subject before the verb gets +, object after gets -
        first = {t: spans[0][0] for t, spans in targets}
        assigned: set[str] = set()
        for m in COMPETE.finditer(text):
            before = [t for t, p in first.items() if p < m.start()]
            after = [t for t, p in first.items() if p > m.end()]
            if before and after:
                mag = max(abs(base.score), 0.4)
                subj = max(before, key=lambda t: first[t])
                obj = min(after, key=lambda t: first[t])
                result[subj] = mag
                result[obj] = -mag
                assigned = {subj, obj}
                break
        else:
            assigned = set()
        # 2) clause-level assignment
        bounds = [0] + [m.start() for m in CLAUSE_SPLIT.finditer(text)] + [len(text)]
        clauses = [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)]
        if len(clauses) > 1:
            cache: dict[tuple[int, int], float] = {}
            for t, spans in targets:
                if t in assigned:
                    continue
                pos = spans[0][0]
                for a, b in clauses:
                    if a <= pos < b:
                        if (a, b) not in cache:
                            clause = text[a:b]
                            r = self.score(clause)
                            cache[(a, b)] = r.score if abs(r.score) > 0.05 else base.score * 0.7
                        result[t] = cache[(a, b)]
                        break
        return result


@lru_cache(maxsize=1)
def default_scorer() -> SentimentScorer:
    return SentimentScorer()
