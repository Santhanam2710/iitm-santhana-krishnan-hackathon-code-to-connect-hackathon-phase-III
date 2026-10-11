"""Novelty and burst velocity.

Novelty = 1 - max cosine similarity to documents seen in the last window (hashed TF-IDF
style vectors; swap in MiniLM embeddings with ``TRIDENT_NOVELTY=minilm`` when
sentence-transformers is installed). Velocity is a burst z-score of mentions per target
in the last 30 minutes against the trailing 6-hour rate.
"""
from __future__ import annotations

import math
import os
from collections import defaultdict, deque
from datetime import datetime, timedelta

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import HashingVectorizer


class NoveltyTracker:
    """Chunked sparse matrix of recent document vectors; one mat-vec per chunk per query."""

    CHUNK = 256

    def __init__(self, window_hours: float = 24, max_docs: int = 20000):
        self.window = timedelta(hours=window_hours)
        self.max_docs = max_docs
        self.chunks: deque[tuple[datetime, object]] = deque()  # (latest ts in chunk, matrix)
        self.pending: list = []
        self.pending_ts: list[datetime] = []
        self.encoder = None
        if os.environ.get("TRIDENT_NOVELTY", "").lower() == "minilm":  # pragma: no cover - optional
            try:
                from sentence_transformers import SentenceTransformer

                self.encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
            except Exception:
                self.encoder = None
        self.hv = HashingVectorizer(n_features=2**18, ngram_range=(1, 2), alternate_sign=False, norm="l2", stop_words="english")

    def _vec(self, text: str):
        if self.encoder is not None:  # pragma: no cover
            return sparse.csr_matrix(self.encoder.encode([text], normalize_embeddings=True))
        return self.hv.transform([text])

    def score(self, text: str, ts: datetime) -> float:
        v = self._vec(text)
        while self.chunks and (ts - self.chunks[0][0] > self.window or len(self.chunks) * self.CHUNK > self.max_docs):
            self.chunks.popleft()
        best = 0.0
        for _, m in self.chunks:
            r = m @ v.T
            if r.nnz:
                best = max(best, float(r.max()))
        if self.pending:
            r = sparse.vstack(self.pending) @ v.T
            if r.nnz:
                best = max(best, float(r.max()))
        self.pending.append(v)
        self.pending_ts.append(ts)
        if len(self.pending) >= self.CHUNK:
            self.chunks.append((max(self.pending_ts), sparse.vstack(self.pending).tocsr()))
            self.pending, self.pending_ts = [], []
        return round(max(0.0, 1.0 - min(best, 1.0)), 4)


class VelocityTracker:
    def __init__(self, short_min: int = 30, long_hours: int = 6):
        self.short = timedelta(minutes=short_min)
        self.long = timedelta(hours=long_hours)
        self.ratio = self.long / self.short
        self.events: dict[str, deque[datetime]] = defaultdict(deque)

    def update(self, key: str, ts: datetime) -> float:
        q = self.events[key]
        q.append(ts)
        while q and ts - q[0] > self.long:
            q.popleft()
        recent = sum(1 for t in q if ts - t <= self.short)
        baseline = max((len(q) - recent) / max(self.ratio - 1, 1), 0.0)
        return round((recent - baseline) / math.sqrt(baseline + 1.0), 3)
