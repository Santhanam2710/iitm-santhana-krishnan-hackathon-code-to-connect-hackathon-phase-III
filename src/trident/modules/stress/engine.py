"""Stress-test engine: shock construction, capital, VaR, reverse stress and contagion."""
from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.stats import norm

from trident.config import get_config
from trident.modules.stress.book import Position, load_book
from trident.modules.stress.revalue import ZERO, revalue


class StressEngine:
    def __init__(self, book: list[Position] | None = None):
        cfg = get_config().scenarios
        self.cfg = cfg
        self.book = book or load_book()
        cap = cfg["capital"]
        self.rho, self.ttc = float(cap["vasicek_rho"]), float(cap["ttc_passthrough"])
        self.mkt_rwa0, self.mkt_scale = float(cap["market_rwa"]), float(cap["market_rwa_vol_scale"])
        base = revalue(self.book, ZERO, self.rho, self.ttc)
        self.rwa0 = self._rwa(base, ZERO, stressed=False)
        self.cet1_0 = float(cap["cet1_ratio_start"]) * self.rwa0
        self.gamma = float(cfg.get("severity_gamma", 1.5))
        v = cfg["var"]
        self.factors = v["factors"]
        self.daily_vol = np.array(v["daily_vol"], float)
        self.corr = np.array(v["corr"], float)
        self.z_alpha = float(norm.ppf(v["confidence"]))
        self.h = float(v["horizon_days"])
        self.sens = self._sensitivities()

    # --- shocks -----------------------------------------------------------------
    def scenario(self, event_class: str, impact: float) -> tuple[dict[str, Any], float]:
        full = self.cfg["scenarios"].get(event_class)
        if full is None:
            return dict(ZERO), 0.0
        mult = (max(0.0, min(impact, 10.0)) / 10.0) ** self.gamma
        return self.scale(full, mult), mult

    @staticmethod
    def scale(full: dict[str, Any], mult: float) -> dict[str, Any]:
        return {k: (v if k == "beta_adj" else float(v) * mult) for k, v in full.items()}

    def contagion(self, shock: dict[str, Any], origin_sector: str | None, strength: float = 0.5) -> dict[str, Any]:
        """Moonshot: raise loan sector loadings for sectors linked to the event's origin sector."""
        if not origin_sector:
            return shock
        links = get_config().universe.get("sector_links", {}).get(origin_sector, {})
        adj = dict(shock.get("beta_adj", {}))
        for p in self.book:
            if p.kind == "loan":
                link = links.get(p.sector, 0.0)
                if link:
                    adj[p.id] = adj.get(p.id, 1.0) * (1 + strength * link)
        return {**shock, "beta_adj": adj}

    # --- capital ----------------------------------------------------------------
    def _rwa(self, rows: list[dict], shock: dict[str, Any], stressed: bool = True) -> float:
        credit = sum(r["rwa1" if stressed else "rwa0"] for r in rows if r["kind"] == "loan")
        mkt = self.mkt_rwa0 * (1 + (max(0.0, float(shock.get("vol", 0))) / self.mkt_scale if stressed else 0.0))
        return credit + mkt

    # --- VaR ----------------------------------------------------------------------
    def _value(self, shock: dict[str, Any]) -> float:
        return sum(r["v1"] for r in revalue(self.book, shock, self.rho, self.ttc))

    def _sensitivities(self) -> np.ndarray:
        v0 = self._value(ZERO)
        return np.array([self._value({**ZERO, f: 1.0}) - v0 for f in self.factors])

    def var(self, vol_shock: float = 0.0, phi: float | None = None) -> dict[str, float]:
        m = max(0.0, vol_shock)
        vols = self.daily_vol * (1 + m / 20)
        phi = min(0.6, 0.02 * m) if phi is None else phi
        R = self.corr.copy()
        off = ~np.eye(len(R), dtype=bool)
        R[off] = R[off] + phi * (np.sign(R[off]) - R[off])
        cov = np.outer(vols, vols) * R
        sd = math.sqrt(max(float(self.sens @ cov @ self.sens), 0.0))
        var = self.z_alpha * math.sqrt(self.h) * sd
        es = sd * math.sqrt(self.h) * float(norm.pdf(self.z_alpha)) / (1 - norm.cdf(self.z_alpha))
        return {"var": var, "es": es, "phi": phi}

    # --- run ----------------------------------------------------------------------
    def run(self, shock: dict[str, Any], label: str = "custom") -> dict[str, Any]:
        rows = revalue(self.book, shock, self.rho, self.ttc)
        v0 = sum(r["v0"] for r in rows)
        dv = sum(r["dv"] for r in rows)
        rwa1 = self._rwa(rows, shock)
        by_class: dict[str, dict[str, float]] = {}
        for r in rows:
            c = by_class.setdefault(r["asset_class"], {"v0": 0.0, "v1": 0.0, "dv": 0.0})
            c["v0"] += r["v0"]
            c["v1"] += r["v1"]
            c["dv"] += r["dv"]
        var0, var1 = self.var(0.0), self.var(float(shock.get("vol", 0.0)))
        cap = self.cfg["capital"]
        return {
            "label": label, "shock": shock,
            "value_before": v0, "value_after": v0 + dv, "pnl": dv, "pnl_pct": dv / v0 if v0 else 0.0,
            "cet1_before": self.cet1_0 / self.rwa0, "cet1_after": (self.cet1_0 + dv) / rwa1,
            "cet1_capital_before": self.cet1_0, "cet1_capital_after": self.cet1_0 + dv,
            "rwa_before": self.rwa0, "rwa_after": rwa1, "minimum": cap["minimum"], "buffer_line": cap["buffer_line"],
            "var_before": var0["var"], "var_after": var1["var"], "es_before": var0["es"], "es_after": var1["es"],
            "by_class": by_class, "positions": rows,
        }

    def reverse_stress(self, base_shock: dict[str, Any], line: float | None = None, max_mult: float = 3.0) -> dict[str, Any]:
        """Smallest multiple of ``base_shock`` that takes CET1 below ``line`` (default: buffer line)."""
        line = line if line is not None else float(self.cfg["capital"]["buffer_line"])

        def cet1(m: float) -> float:
            r = self.run(self.scale(base_shock, m))
            return r["cet1_after"]

        if cet1(max_mult) >= line:
            return {"breach": False, "multiple": None, "line": line, "implied_impact": None}
        lo, hi = 0.0, max_mult
        for _ in range(40):
            mid = (lo + hi) / 2
            if cet1(mid) < line:
                hi = mid
            else:
                lo = mid
        implied = 10 * hi ** (1 / self.gamma) if hi <= 1 else None
        return {"breach": True, "multiple": hi, "line": line, "implied_impact": implied}
