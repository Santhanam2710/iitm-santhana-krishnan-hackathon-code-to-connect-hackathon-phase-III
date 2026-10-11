"""Credit risk mathematics: Vasicek conditional PD (eq. B3) and Basel IRB capital (eq. B5)."""
from __future__ import annotations

import math

from scipy.stats import norm

G999 = float(norm.ppf(0.999))


def vasicek_pd(pd: float, Z: float, rho: float = 0.12, beta: float = 1.0) -> float:
    """PD conditional on the systematic factor: Phi((Phi^-1(PD) - sqrt(rho) beta Z) / sqrt(1 - rho)).
    Z < 0 is the adverse draw; beta is the sector loading."""
    if Z == 0:
        return pd
    pd = min(max(pd, 1e-6), 0.999)
    return float(norm.cdf((norm.ppf(pd) - math.sqrt(rho) * beta * Z) / math.sqrt(1 - rho)))


def basel_corr(pd: float) -> float:
    """Basel corporate asset correlation, between 0.12 (high PD) and 0.24 (low PD)."""
    a = (1 - math.exp(-50 * pd)) / (1 - math.exp(-50))
    return 0.12 * a + 0.24 * (1 - a)


def irb_k(pd: float, lgd: float) -> float:
    """IRB capital requirement per unit EAD (maturity adjustment omitted, stated as a limitation)."""
    pd = min(max(pd, 0.0003), 0.999)
    R = basel_corr(pd)
    return float(lgd * (norm.cdf((norm.ppf(pd) + math.sqrt(R) * G999) / math.sqrt(1 - R)) - pd))


def irb_rwa(pd: float, lgd: float, ead: float) -> float:
    return 12.5 * irb_k(pd, lgd) * ead
