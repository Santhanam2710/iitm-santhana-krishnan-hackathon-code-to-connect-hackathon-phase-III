"""Module A (property-based) and Module B (risk mathematics) tests."""
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from trident.modules.rebalancer.aggregate import SentimentBook
from trident.modules.rebalancer.backtest import metrics
from trident.modules.rebalancer.tilt import apply_band_and_turnover, project_caps, tilt_target
from trident.modules.stress.credit import basel_corr, irb_k, vasicek_pd
from trident.modules.stress.revalue import ZERO, bs_price
from trident.modules.stress.service import StressService, TriggerGate

N = 15
WB = project_caps(np.linspace(1, 3, N) / np.linspace(1, 3, N).sum(), 0.005, 0.15)
VOL = np.linspace(0.15, 0.6, N)


@settings(max_examples=200, deadline=None)
@given(st.lists(st.floats(-1, 1), min_size=N, max_size=N), st.floats(0, 3))
def test_weights_are_valid(S, lam):
    w = tilt_target(WB, np.array(S), VOL, lam, 0.005, 0.15)
    assert abs(w.sum() - 1) < 1e-8
    assert (w >= 0.005 - 1e-9).all() and (w <= 0.15 + 1e-9).all()


@settings(max_examples=100, deadline=None)
@given(st.lists(st.floats(-0.9, 0.9), min_size=N, max_size=N), st.integers(0, N - 1), st.floats(0.05, 0.5))
def test_more_sentiment_never_lowers_own_weight(S, i, bump):
    S = np.array(S)
    w0 = tilt_target(WB, S, VOL, 1.5, 0.005, 0.15)
    S2 = S.copy()
    S2[i] = min(1.0, S2[i] + bump)
    w1 = tilt_target(WB, S2, VOL, 1.5, 0.005, 0.15)
    assert w1[i] >= w0[i] - 1e-9


@settings(max_examples=100, deadline=None)
@given(st.lists(st.floats(-1, 1), min_size=N, max_size=N), st.floats(0.01, 0.3))
def test_turnover_cap_respected(S, cap):
    target = tilt_target(WB, np.array(S), VOL, 3.0, 0.005, 0.15)
    w, _ = apply_band_and_turnover(WB, target, 0.0, cap, 0.005, 0.15)
    assert 0.5 * np.abs(w - WB).sum() <= cap + 1e-6


def test_band_suppresses_small_trades():
    target = WB + np.r_[0.001, -0.001, np.zeros(N - 2)]
    w, suppressed = apply_band_and_turnover(WB, target, 0.002, 1.0, 0.005, 0.15)
    assert np.allclose(w, WB) and suppressed == 2


def test_sentiment_book_point_in_time_and_decay():
    b = SentimentBook(tau_minutes=60, n0=1.0, contagion=0.0)
    t0 = datetime(2024, 1, 1, 12, tzinfo=timezone.utc)
    b.add("JPM", t0, -0.8, 0.9, 1.0)
    assert b.scores(t0 - timedelta(minutes=1))["JPM"]["S"] == 0  # future news is invisible
    s1 = b.scores(t0 + timedelta(minutes=1))["JPM"]["S"]
    s2 = b.scores(t0 + timedelta(hours=10))["JPM"]["S"]
    assert s1 < 0 and abs(s2) <= abs(s1)


def test_contagion_spills_bank_news_into_bank_names():
    b = SentimentBook(contagion=0.5)
    t0 = datetime(2024, 1, 1, tzinfo=timezone.utc)
    b.add("SIVB", t0, -0.9, 0.9, 1.0)
    sc = b.scores(t0 + timedelta(minutes=5))
    assert sc["JPM"]["spill"] < 0 and sc["JPM"]["spill"] < sc["MSFT"]["spill"]


def test_metrics_sane():
    import pandas as pd

    r = pd.Series(np.random.default_rng(0).normal(0.0005, 0.01, 500))
    m = metrics(r, r * 0.9)
    assert 0 <= m["max_drawdown"] <= 1 and np.isfinite(m["sharpe"])


# ---------------------------------------------------------------- Module B
def test_vasicek_monotone_in_Z():
    assert vasicek_pd(0.02, -2) > vasicek_pd(0.02, -1) > vasicek_pd(0.02, 0) == 0.02 > vasicek_pd(0.02, 1)


def test_basel_correlation_bounds_and_capital_monotone():
    assert 0.12 <= basel_corr(0.2) < basel_corr(0.001) <= 0.24
    assert irb_k(0.05, 0.45) > irb_k(0.01, 0.45) > 0


def test_black_scholes_put_call_parity():
    S, K, T, s, r = 5000, 4800, 0.5, 0.2, 0.04
    c, p = bs_price(S, K, T, s, r, "call"), bs_price(S, K, T, s, r, "put")
    assert abs((c - p) - (S - K * np.exp(-r * T))) < 1e-6


def test_stress_signs_and_capital():
    svc = StressService()
    base = svc.engine.run(ZERO)
    assert abs(base["pnl"]) < 1e-9 and abs(base["cet1_after"] - base["cet1_before"]) < 1e-12
    r = svc.run_for("Credit Event", 9)
    assert r["pnl"] < 0 and r["cet1_after"] < r["cet1_before"] and r["var_after"] > r["var_before"]
    hedges = {p["id"]: p["dv"] for p in r["positions"]}
    assert hedges["D3"] > 0  # CDS protection gains when spreads widen
    assert svc.run_for("Credit Event", 10)["pnl"] < svc.run_for("Credit Event", 7)["pnl"]  # convex in severity


def test_reverse_stress_finds_breach():
    svc = StressService()
    from trident.config import get_config

    rv = svc.engine.reverse_stress(get_config().scenarios["scenarios"]["Credit Event"])
    assert rv["breach"] and 0 < rv["multiple"] <= 3
    assert svc.engine.run(svc.engine.scale(get_config().scenarios["scenarios"]["Credit Event"], rv["multiple"] * 1.01))["cet1_after"] < 0.07


def _sig(ts, src, impact=8, label="Credit Event", flags=()):
    return {"signal_id": f"s{ts}{src}", "ts_published": ts, "source": {"name": src, "reliability": 1.0}, "impact_score": impact,
            "event_classification": {"label": label, "confidence": 0.9}, "entity": {"ticker": "X", "sector": "Banks"}, "flags": list(flags)}


def test_gate_requires_two_sources_or_high_impact():
    g = TriggerGate()
    assert g.evaluate(_sig("2024-01-01T10:00:00Z", "a"))["state"] == "held"
    assert g.evaluate(_sig("2024-01-01T10:10:00Z", "b"))["state"] == "triggered"
    g2 = TriggerGate()
    assert g2.evaluate(_sig("2024-01-01T10:00:00Z", "a", impact=9))["state"] == "triggered"
    assert TriggerGate().evaluate(_sig("2024-01-01T10:00:00Z", "a", impact=6))["state"] == "monitored"
    assert TriggerGate().evaluate(_sig("2024-01-01T10:00:00Z", "a", label="Earnings", impact=10))["state"] == "monitored"
    assert TriggerGate().evaluate(_sig("2024-01-01T10:00:00Z", "a", impact=10, flags=["retrospective (x)"]))["state"] == "monitored"


def test_gate_cooldown_and_escalation():
    g = TriggerGate()
    g.evaluate(_sig("2024-01-01T10:00:00Z", "a", impact=9))
    assert g.evaluate(_sig("2024-01-01T11:00:00Z", "c", impact=9))["state"] in ("cooldown", "triggered")
    d = g.evaluate(_sig("2024-01-01T11:05:00Z", "d", impact=10))
    assert d["state"] == "triggered" and "escalation" in d["reason"]
