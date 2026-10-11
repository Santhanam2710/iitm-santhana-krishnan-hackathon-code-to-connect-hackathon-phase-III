"""Near-duplicate clustering with MinHash + LSH (red-team item 6).

Wire stories are republished by many outlets. Items whose estimated Jaccard similarity
on word 3-shingles exceeds the threshold join an existing cluster. A repeat from the
*same* source is dropped; a repeat from a *different* source is kept (it counts as an
independent confirmation) but gets low novelty, so it adds little impact.
"""
from __future__ import annotations

import re
import zlib
from collections import OrderedDict
from dataclasses import dataclass

import numpy as np

_MERSENNE = (1 << 61) - 1
_MAX = (1 << 32) - 1


def shingles(text: str, k: int = 3) -> set[int]:
    words = re.findall(r"[a-z0-9$%]+", text.lower())
    if len(words) < k:
        return {zlib.crc32(" ".join(words).encode())} if words else set()
    return {zlib.crc32(" ".join(words[i : i + k]).encode()) for i in range(len(words) - k + 1)}


class MinHasher:
    def __init__(self, num_perm: int = 64, seed: int = 7):
        rng = np.random.default_rng(seed)
        self.a = rng.integers(1, _MERSENNE, size=num_perm, dtype=np.uint64)
        self.b = rng.integers(0, _MERSENNE, size=num_perm, dtype=np.uint64)
        self.num_perm = num_perm

    def signature(self, sh: set[int]) -> np.ndarray:
        if not sh:
            return np.full(self.num_perm, _MAX, dtype=np.uint64)
        x = np.fromiter(sh, dtype=np.uint64)[:, None]
        hv = (x * self.a + self.b) % np.uint64(_MERSENNE) & np.uint64(_MAX)
        return hv.min(axis=0)


@dataclass
class DedupResult:
    cluster_id: str
    is_duplicate: bool          # same source repeated: drop
    is_syndicated: bool         # different source, same story: keep, low novelty
    similarity: float
    cluster_sources: int


class Deduper:
    def __init__(self, threshold: float = 0.8, num_perm: int = 64, bands: int = 16, capacity: int = 20000, window_hours: float = 48.0):
        self.h = MinHasher(num_perm)
        self.threshold = threshold
        self.bands = bands
        self.rows = num_perm // bands
        self.buckets: dict[tuple[int, int], list[str]] = {}
        self.sigs: OrderedDict[str, np.ndarray] = OrderedDict()
        self.cluster_of: dict[str, str] = {}
        self.sources: dict[str, set[str]] = {}
        self.capacity = capacity
        self.window_s = window_hours * 3600.0
        self.ts: dict[str, float] = {}

    def _keys(self, sig: np.ndarray):
        for b in range(self.bands):
            yield (b, hash(sig[b * self.rows : (b + 1) * self.rows].tobytes()))

    def check(self, doc_id: str, text: str, source_name: str, ts: float | None = None) -> DedupResult:
        """``ts`` is epoch seconds of publication; stories older than the window never match."""
        sig = self.h.signature(shingles(text))
        cands: set[str] = set()
        for key in self._keys(sig):
            cands.update(self.buckets.get(key, ()))
        if ts is not None:
            cands = {c for c in cands if abs(ts - self.ts.get(c, ts)) <= self.window_s}
        best, best_sim = None, 0.0
        for c in cands:
            sim = float(np.mean(self.sigs[c] == sig))
            if sim > best_sim:
                best, best_sim = c, sim
        if best is not None and best_sim >= self.threshold:
            cid = self.cluster_of[best]
            srcs = self.sources.setdefault(cid, set())
            dup = source_name in srcs
            srcs.add(source_name)
            self._store(doc_id, sig, cid, ts)
            return DedupResult(cid, dup, not dup, best_sim, len(srcs))
        cid = "c_" + doc_id[-10:]
        self.sources[cid] = {source_name}
        self._store(doc_id, sig, cid, ts)
        return DedupResult(cid, False, False, best_sim, 1)

    def _store(self, doc_id: str, sig: np.ndarray, cid: str, ts: float | None = None) -> None:
        self.sigs[doc_id] = sig
        if ts is not None:
            self.ts[doc_id] = ts
        self.cluster_of[doc_id] = cid
        for key in self._keys(sig):
            self.buckets.setdefault(key, []).append(doc_id)
        while len(self.sigs) > self.capacity:
            old, osig = self.sigs.popitem(last=False)
            self.cluster_of.pop(old, None)
            self.ts.pop(old, None)
            for key in self._keys(osig):
                lst = self.buckets.get(key)
                if lst and old in lst:
                    lst.remove(old)
