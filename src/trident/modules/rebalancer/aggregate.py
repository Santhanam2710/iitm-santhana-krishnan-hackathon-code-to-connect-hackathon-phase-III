"""Per-ticker sentiment aggregation (eq. E2) with sector and market spillover.

S_i(t) = Sbar_i(t) * (1 - exp(-n_eff / n0)), where Sbar is the decay- and
reliability-weighted mean of signal sentiments. Sector-level signals (``SECTOR:Banks``)
spill into index members through the sector link matrix in configs/universe.yaml (moonshot:
contagion). Market-level signals (``MARKET``) are loaded by relative volatility, so a risk-off
headline rotates weight from high-volatility names to defensive ones.
"""
from __future__ import annotations

import math
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime

from trident.config import get_config


@dataclass
class Obs:
    ts: datetime
    s: float
    c: float
    w: float  # source reliability x novelty floor


class SentimentBook:
    def __init__(self, tau_minutes: float | None = None, n0: float | None = None, contagion: float | None = None):
        cfg = get_config()
        agg = cfg.section("aggregation")
        self.tau = float(tau_minutes or agg.get("tau_minutes", 360)) * 60.0
        self.n0 = float(n0 or agg.get("n0", 1.5))
        self.contagion = float(contagion if contagion is not None else cfg.section("rebalancer").get("contagion_weight", 0.35))
        self.obs: dict[str, deque[Obs]] = defaultdict(lambda: deque(maxlen=500))
        self.index = [e.ticker for e in cfg.index]
        self.sector_of = {e.ticker: e.sector for e in cfg.entities}
        self.links = cfg.universe.get("sector_links", {})
        vols = {e.ticker: e.vol for e in cfg.index}
        mean_vol = sum(vols.values()) / max(len(vols), 1)
        # market-wide news rotates the index by risk: high-vol names absorb more of a risk-off signal
        self.market_loading = {t: v / mean_vol - 1.0 for t, v in vols.items()}

    def add(self, key: str, ts: datetime, s: float, c: float, reliability: float, novelty: float = 1.0) -> None:
        self.obs[key].append(Obs(ts, s, c, reliability * max(novelty, 0.25)))

    def _own(self, key: str, now: datetime) -> tuple[float, float]:
        num = den = 0.0
        for o in self.obs.get(key, ()):
            dt = (now - o.ts).total_seconds()
            if dt < 0:
                continue  # point-in-time: ignore the future
            k = o.w * o.c * math.exp(-dt / self.tau)
            num += k * o.s
            den += k
        if den < 1e-9:
            return 0.0, 0.0
        return (num / den) * (1 - math.exp(-den / self.n0)), den

    def scores(self, now: datetime) -> dict[str, dict[str, float]]:
        sector_scores: dict[str, float] = {}
        for key in self.obs:
            if key.startswith("SECTOR:"):
                sector_scores[key.split(":", 1)[1]] = self._own(key, now)[0]
        # watchlist names feed their sector, e.g. SVB -> Banks
        watch_by_sector: dict[str, list[float]] = defaultdict(list)
        for key in self.obs:
            if key not in self.index and not key.startswith(("SECTOR:", "MARKET")):
                sec = self.sector_of.get(key)
                if sec:
                    watch_by_sector[sec].append(self._own(key, now)[0])
        for sec, vals in watch_by_sector.items():
            sector_scores[sec] = sector_scores.get(sec, 0.0) + sum(vals) / len(vals)
        market = self._own("MARKET", now)[0]
        out = {}
        for t in self.index:
            own, n_eff = self._own(t, now)
            sec = self.sector_of.get(t, "Other")
            spill = 0.0
            for src_sec, val in sector_scores.items():
                link = self.links.get(src_sec, {}).get(sec, 1.0 if src_sec == sec else 0.0)
                spill += link * val
            mkt = market * self.market_loading.get(t, 0.0)
            total = own + self.contagion * spill + mkt
            out[t] = {"own": round(own, 4), "spill": round(self.contagion * spill, 4), "market": round(mkt, 4),
                      "S": round(max(-1.0, min(1.0, total)), 4), "n_eff": round(n_eff, 3)}
        return out
