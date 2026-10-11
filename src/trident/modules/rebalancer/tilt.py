"""Weight construction: vol-adjusted multiplicative tilt (A1, A2), projection onto the
constraint set (A3), no-trade band and turnover cap, plus a transaction cost model."""
from __future__ import annotations

import numpy as np


def vol_adjusted(S: np.ndarray, vol: np.ndarray) -> np.ndarray:
    """z_i = S_i * mean(sigma) / sigma_i   (eq. A1)."""
    return S * (vol.mean() / vol)


def project_caps(w: np.ndarray, lo: float, hi: float, iters: int = 100) -> np.ndarray:
    """Project onto {sum w = 1, lo <= w <= hi} by clipping and proportional redistribution."""
    n = len(w)
    if lo * n > 1 + 1e-12 or hi * n < 1 - 1e-12:
        raise ValueError("infeasible bounds")
    x = np.asarray(w, float).copy()
    x = x / x.sum()
    for _ in range(iters):
        x = np.clip(x, lo, hi)
        fixed = (x <= lo + 1e-12) | (x >= hi - 1e-12)
        free_sum = x[~fixed].sum()
        target = 1.0 - x[fixed].sum()
        if free_sum <= 0:
            break
        f = target / free_sum
        if abs(f - 1) < 1e-12:
            break
        x[~fixed] *= f
    # final exact normalization among free names
    x = np.clip(x, lo, hi)
    gap = 1 - x.sum()
    if abs(gap) > 1e-10:
        room = (hi - x) if gap > 0 else (x - lo)
        if room.sum() > 0:
            x += gap * room / room.sum()
    return x


def tilt_target(wb: np.ndarray, S: np.ndarray, vol: np.ndarray, lam: float, lo: float, hi: float) -> np.ndarray:
    """w* = Pi_W( wb * exp(lambda z) / sum )   (eq. A2)."""
    z = vol_adjusted(S, vol)
    raw = wb * np.exp(lam * z)
    return project_caps(raw / raw.sum(), lo, hi)


def apply_band_and_turnover(w_prev: np.ndarray, target: np.ndarray, band: float, to_max: float, lo: float, hi: float) -> tuple[np.ndarray, int]:
    """No-trade band, then scale the trade so one-way turnover <= to_max (eq. A3)."""
    delta = target - w_prev
    suppressed = int(np.sum((np.abs(delta) < band) & (np.abs(delta) > 1e-6)))
    delta = np.where(np.abs(delta) < band, 0.0, delta)
    w = w_prev + delta
    w = project_caps(np.clip(w, 1e-9, None), lo, hi)
    turnover = 0.5 * np.abs(w - w_prev).sum()
    if turnover > to_max > 0:
        w = w_prev + (w - w_prev) * (to_max / turnover)
        w = project_caps(w, lo, hi)
    return w, suppressed


def trade_cost(w_prev: np.ndarray, w_new: np.ndarray, kappa_bps: float, eta: float = 0.0, vol: np.ndarray | None = None,
               participation: np.ndarray | None = None) -> float:
    """Cost as a fraction of NAV: kappa * sum|dw| + eta * sum |dw| sigma_d sqrt(Q/ADV)."""
    dw = np.abs(w_new - w_prev)
    cost = kappa_bps * 1e-4 * dw.sum()
    if eta and vol is not None and participation is not None:
        daily_vol = vol / np.sqrt(252)
        cost += eta * float(np.sum(dw * daily_vol * np.sqrt(np.clip(participation, 0, None))))
    return float(cost)
