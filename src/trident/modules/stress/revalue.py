"""Position revaluation under a factor shock (eq. B3, B4).

Shock keys: eq (%), rates (bp), ig / hy (bp), fx (% EUR/USD), vol (vol points),
Z (systematic credit factor, sigma), lgd_add (absolute), beta_adj ({position_id: mult}).
Options are fully revalued with Black-Scholes (edge tier); other derivatives use Greeks.
"""
from __future__ import annotations

import math
from dataclasses import asdict
from typing import Any

from scipy.stats import norm

from trident.modules.stress.book import Position
from trident.modules.stress.credit import irb_rwa, vasicek_pd

ZERO: dict[str, Any] = {"eq": 0.0, "rates": 0.0, "ig": 0.0, "hy": 0.0, "fx": 0.0, "vol": 0.0, "Z": 0.0, "lgd_add": 0.0, "beta_adj": {}}


def bs_price(S: float, K: float, T: float, sigma: float, r: float = 0.04, kind: str = "put") -> float:
    if T <= 0 or sigma <= 0:
        return max(0.0, (K - S) if kind == "put" else (S - K))
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    if kind == "put":
        return K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)


def revalue(book: list[Position], shock: dict[str, Any], rho: float = 0.12, ttc: float = 0.35) -> list[dict[str, Any]]:
    x = {**ZERO, **shock}
    x["eq"] = max(float(x["eq"]), -95.0)  # prices cannot go below zero under extreme reverse-stress multiples
    out = []
    for p in book:
        row = asdict(p)
        detail: dict[str, Any] = {}
        if p.kind == "loan":
            beta = p.beta * float(x["beta_adj"].get(p.id, 1.0)) * float(x["beta_adj"].get(p.sector, 1.0))
            pds = vasicek_pd(p.pd, float(x["Z"]), rho, beta)
            lgds = min(0.95, p.lgd + float(x["lgd_add"]))
            v0 = p.ead * (1 - p.pd * p.lgd)
            v1 = p.ead * (1 - pds * lgds)
            pd_rwa, lgd_rwa = p.pd + ttc * (pds - p.pd), p.lgd + ttc * (lgds - p.lgd)
            detail = {"pd_stressed": pds, "lgd_stressed": lgds, "rwa0": irb_rwa(p.pd, p.lgd, p.ead), "rwa1": irb_rwa(pd_rwa, lgd_rwa, p.ead),
                      "method": "Vasicek PD migration, delta EL"}
        elif p.kind == "bond":
            sp = float(x["ig"]) if p.spread == "ig" else (float(x["hy"]) if p.spread == "hy" else 0.0)
            dy = (float(x["rates"]) + sp) / 1e4
            v0 = p.mv
            v1 = p.mv * (1 - p.dur * dy + 0.5 * p.cvx * dy * dy)
            detail = {"dy_bp": dy * 1e4, "method": "duration + convexity"}
        elif p.kind == "irs":
            v0, v1 = p.mv, p.mv + p.dv01 * float(x["rates"])
            detail = {"method": "DV01"}
        elif p.kind == "eqd":
            e = float(x["eq"])
            v0 = p.mv
            v1 = p.mv + p.delta * e / 100 + 0.5 * p.gamma * e * e + p.vega * float(x["vol"])
            detail = {"method": "delta-gamma-vega"}
        elif p.kind == "option":
            S1 = p.S0 * (1 + float(x["eq"]) / 100)
            sig1 = max(0.01, p.sigma + float(x["vol"]) / 100)
            unit0 = bs_price(p.S0, p.K, p.T, p.sigma, kind=p.option_type or "put")
            unit1 = bs_price(S1, p.K, p.T, sig1, kind=p.option_type or "put")
            scale = p.qty / p.S0  # qty is $m of index notional
            v0, v1 = unit0 * scale, unit1 * scale
            detail = {"method": "Black-Scholes full revaluation", "S1": S1, "sigma1": sig1}
        elif p.kind == "cds":
            v0, v1 = p.mv, p.mv + p.cs01 * float(x["hy"])
            detail = {"method": "CS01"}
        elif p.kind == "eq":
            v0, v1 = p.mv, p.mv * (1 + p.beta * float(x["eq"]) / 100)
            detail = {"method": "beta"}
        elif p.kind == "fx":
            v0, v1 = p.mv, p.mv + p.notional * float(x["fx"]) / 100
            detail = {"method": "FX delta"}
        else:
            v0 = v1 = p.mv
        v0, v1 = float(v0), float(v1)
        row.update({"v0": v0, "v1": v1, "dv": v1 - v0, **detail})
        out.append(row)
    return out
