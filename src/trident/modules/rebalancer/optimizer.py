"""Edge tier: mean-variance rebalancer (eq. A4) with a Ledoit-Wolf shrunk covariance.

max_w  lambda z'w - gamma/2 (w - wb)' Sigma (w - wb) - kappa ||w - w_prev||_1
s.t.   sum w = 1, w_min <= w <= w_max, 0.5 ||w - w_prev||_1 <= TO_max
Falls back to the closed-form tilt if cvxpy is not installed or the solver fails.
"""
from __future__ import annotations

import numpy as np

from trident.modules.rebalancer.tilt import tilt_target


def ledoit_wolf_cov(returns: np.ndarray) -> np.ndarray:
    from sklearn.covariance import LedoitWolf

    return LedoitWolf().fit(returns).covariance_


def optimize(wb: np.ndarray, w_prev: np.ndarray, z: np.ndarray, cov: np.ndarray, lam: float, gamma: float, kappa: float,
             lo: float, hi: float, to_max: float, vol: np.ndarray | None = None) -> tuple[np.ndarray, str]:
    try:
        import cvxpy as cp

        n = len(wb)
        w = cp.Variable(n)
        active = w - wb
        obj = lam * z @ w - 0.5 * gamma * cp.quad_form(active, cp.psd_wrap(cov)) - kappa * cp.norm1(w - w_prev)
        cons = [cp.sum(w) == 1, w >= lo, w <= hi, 0.5 * cp.norm1(w - w_prev) <= to_max]
        prob = cp.Problem(cp.Maximize(obj), cons)
        prob.solve(solver=cp.CLARABEL if "CLARABEL" in cp.installed_solvers() else None)
        if w.value is None or prob.status not in ("optimal", "optimal_inaccurate"):
            raise RuntimeError(prob.status)
        x = np.clip(np.asarray(w.value).ravel(), lo, hi)
        return x / x.sum(), "cvxpy"
    except Exception:
        v = vol if vol is not None else np.sqrt(np.diag(cov)) * np.sqrt(252)
        return tilt_target(wb, z, v, lam, lo, hi), "tilt-fallback"
