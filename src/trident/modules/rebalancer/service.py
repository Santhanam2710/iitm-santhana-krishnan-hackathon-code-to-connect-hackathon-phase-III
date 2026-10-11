"""Live Module A: subscribes to signals, maintains the sentiment book and weights."""
from __future__ import annotations

from collections import deque
from datetime import datetime, timedelta
from typing import Any

import numpy as np

from trident.config import get_config
from trident.modules.rebalancer.aggregate import SentimentBook
from trident.modules.rebalancer.optimizer import optimize
from trident.modules.rebalancer.tilt import apply_band_and_turnover, tilt_target, trade_cost
from trident.utils.common import iso, parse_ts


class Rebalancer:
    def __init__(self, params: dict[str, Any] | None = None):
        cfg = get_config()
        self.p = {**cfg.section("rebalancer"), **(params or {})}
        self.tickers = [e.ticker for e in cfg.index]
        caps = np.array([e.cap_weight for e in cfg.index], float)
        self.vol = np.array([e.vol for e in cfg.index], float)
        self.wb = self._bench(caps)
        self.w = self.wb.copy()
        self.book = SentimentBook()
        self.history: deque[dict] = deque(maxlen=2000)
        self.cost_total = 0.0
        self.suppressed_total = 0
        self.trades_total = 0
        self.last_rebalance: datetime | None = None
        self.cov = np.diag((self.vol / np.sqrt(252)) ** 2)

    def _bench(self, caps: np.ndarray) -> np.ndarray:
        from trident.modules.rebalancer.tilt import project_caps

        return project_caps(caps / caps.sum(), float(self.p["w_min"]), float(self.p["w_max"]))

    def set_params(self, **kw) -> None:
        self.p.update({k: v for k, v in kw.items() if v is not None})
        caps = np.array([e.cap_weight for e in get_config().index], float)
        self.wb = self._bench(caps)

    def on_signal(self, sig: dict) -> None:
        key = sig["entity"]["ticker"]
        self.book.add(key, parse_ts(sig["ts_published"]), float(sig["sentiment_score"]), float(sig["sentiment_confidence"]),
                      float(sig["source"]["reliability"]), float(sig["source"].get("novelty", 1.0)))

    def rebalance(self, now: datetime) -> dict:
        p = self.p
        scores = self.book.scores(now)
        S = np.array([scores[t]["S"] for t in self.tickers])
        lo, hi = float(p["w_min"]), float(p["w_max"])
        if p.get("method") == "optimizer":
            z = S * (self.vol.mean() / self.vol)
            target, how = optimize(self.wb, self.w, z, self.cov, float(p["lambda"]) * 0.01, float(p["gamma"]),
                                   float(p["kappa_bps"]) * 1e-4, lo, hi, float(p["turnover_max"]), self.vol)
        else:
            target, how = tilt_target(self.wb, S, self.vol, float(p["lambda"]), lo, hi), "tilt"
        min_hold = timedelta(minutes=float(p.get("min_hold_minutes", 0)))
        if self.last_rebalance and now - self.last_rebalance < min_hold:
            new, suppressed = self.w.copy(), int(np.sum(np.abs(target - self.w) > 1e-6))
        else:
            new, suppressed = apply_band_and_turnover(self.w, target, float(p["band"]), float(p["turnover_max"]), lo, hi)
        cost = trade_cost(self.w, new, float(p["kappa_bps"]), float(p.get("impact_eta", 0)), self.vol, np.full(len(new), 0.01))
        traded = int(np.sum(np.abs(new - self.w) > 1e-6))
        if traded:
            self.last_rebalance = now
        turnover = 0.5 * float(np.abs(new - self.w).sum())
        self.cost_total += cost
        self.suppressed_total += suppressed
        self.trades_total += traded
        self.w = new
        rec = {"ts": iso(now), "method": how, "weights": dict(zip(self.tickers, np.round(new, 6).tolist())),
               "benchmark": dict(zip(self.tickers, np.round(self.wb, 6).tolist())), "scores": scores,
               "turnover": round(turnover, 6), "cost_bps": round(cost * 1e4, 4), "suppressed": suppressed, "traded": traded,
               "active_share": round(0.5 * float(np.abs(new - self.wb).sum()), 6)}
        self.history.append(rec)
        return rec

    def snapshot(self) -> dict:
        last = self.history[-1] if self.history else None
        return {"tickers": self.tickers, "last": last, "cost_total_bps": round(self.cost_total * 1e4, 3),
                "suppressed_total": self.suppressed_total, "trades_total": self.trades_total, "params": self.p}
