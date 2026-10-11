"""Walk-forward daily backtest of Module A against cap-weight and equal-weight portfolios.

Point-in-time rules (red-team item 10): returns are close-to-close, so the weights held
over day d's return (close d-1 -> close d) may use only signals published before the
close of day d-1 (21:00 UTC). News that lands overnight is used from day d+1. This is
conservative (it forgoes the overnight move) and cannot leak future information.
Costs are charged on every rebalance.

Usage:
  python -m trident.modules.rebalancer.backtest --signals data/runtime/signals_hist.jsonl \
      --prices data/market/prices_sp15.csv --out docs/results/backtest.json
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, time, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from trident.config import get_config, resolve
from trident.modules.rebalancer.aggregate import SentimentBook
from trident.modules.rebalancer.tilt import apply_band_and_turnover, project_caps, tilt_target, trade_cost
from trident.utils.common import parse_ts, read_jsonl


def metrics(r: pd.Series, bench: pd.Series | None = None, rf: float = 0.0) -> dict:
    r = r.dropna()
    ann = np.sqrt(252)
    mu, sd = r.mean(), r.std(ddof=1)
    downside = r[r < 0].std(ddof=1)
    eq = (1 + r).cumprod()
    mdd = float((1 - eq / eq.cummax()).max())
    out = {"ann_return": float((1 + mu) ** 252 - 1), "ann_vol": float(sd * ann), "sharpe": float((mu - rf / 252) / sd * ann) if sd > 0 else 0.0,
           "sortino": float(mu / downside * ann) if downside and downside > 0 else 0.0, "max_drawdown": mdd, "days": int(len(r))}
    if bench is not None:
        a = (r - bench.reindex(r.index)).dropna()
        te = a.std(ddof=1)
        out["information_ratio"] = float(a.mean() / te * ann) if te > 0 else 0.0
        out["tracking_error"] = float(te * ann)
        out["active_return"] = float(a.mean() * 252)
    return out


def load_prices(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(resolve(path), parse_dates=["date"]).set_index("date").sort_index()
    return df


def run_backtest(signals: list[dict], prices: pd.DataFrame, params: dict | None = None) -> dict:
    cfg = get_config()
    p = {**cfg.section("rebalancer"), **(params or {})}
    tickers = [e.ticker for e in cfg.index if e.ticker in prices.columns]
    caps = np.array([cfg.entity(t).cap_weight for t in tickers])
    lo, hi = float(p["w_min"]), float(p["w_max"])
    wb = project_caps(caps / caps.sum(), lo, hi)
    rets = prices[tickers].pct_change().dropna(how="all")
    vol_hist = rets.ewm(span=60, min_periods=20).std() * np.sqrt(252)
    prior_vol = np.array([cfg.entity(t).vol for t in tickers])

    sigs = sorted(signals, key=lambda s: s["ts_published"])
    book = SentimentBook()
    book.index = tickers
    i_sig = 0
    w = wb.copy()
    rows = []
    prev_day = prices.index[0]
    for day in rets.index:
        cutoff = datetime.combine(prev_day.date(), time(21, 0), tzinfo=timezone.utc)
        prev_day = day
        while i_sig < len(sigs) and parse_ts(sigs[i_sig]["ts_published"]) < cutoff:
            s = sigs[i_sig]
            book.add(s["entity"]["ticker"], parse_ts(s["ts_published"]), float(s["sentiment_score"]), float(s["sentiment_confidence"]),
                     float(s["source"]["reliability"]), float(s["source"].get("novelty", 1.0)))
            i_sig += 1
        sc = book.scores(cutoff)
        S = np.array([sc[t]["S"] for t in tickers])
        v = vol_hist.loc[day].to_numpy() if day in vol_hist.index else prior_vol
        v = np.where(np.isfinite(v) & (v > 0), v, prior_vol)
        target = tilt_target(wb, S, v, float(p["lambda"]), lo, hi)
        new, suppressed = apply_band_and_turnover(w, target, float(p["band"]), float(p["turnover_max"]), lo, hi)
        cost = trade_cost(w, new, float(p["kappa_bps"]), float(p.get("impact_eta", 0)), v, np.full(len(new), 0.01))
        turnover = 0.5 * float(np.abs(new - w).sum())
        w = new
        r = rets.loc[day, tickers].fillna(0).to_numpy()
        rows.append({"date": day, "strategy": float(w @ r) - cost, "strategy_gross": float(w @ r), "cap_weight": float(wb @ r),
                     "equal_weight": float(r.mean()), "turnover": turnover, "cost": cost,
                     "suppressed": suppressed, **{f"w_{t}": float(x) for t, x in zip(tickers, w)}})
        # drift weights with returns (buy-and-hold between rebalances)
        w = w * (1 + r)
        w = w / w.sum()
    df = pd.DataFrame(rows).set_index("date")
    res = {
        "params": {k: p[k] for k in ("lambda", "w_min", "w_max", "band", "turnover_max", "kappa_bps")},
        "strategy": metrics(df["strategy"], df["cap_weight"]),
        "strategy_gross": metrics(df["strategy_gross"], df["cap_weight"]),
        "cap_weight": metrics(df["cap_weight"]),
        "equal_weight": metrics(df["equal_weight"], df["cap_weight"]),
        "avg_daily_turnover": float(df["turnover"].mean()),
        "total_cost_bps": float(df["cost"].sum() * 1e4),
        "signals_used": i_sig,
        "period": [str(df.index[0].date()), str(df.index[-1].date())] if len(df) else None,
    }
    return {"summary": res, "daily": df}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--signals", required=True)
    ap.add_argument("--prices", required=True)
    ap.add_argument("--out", default="docs/results/backtest.json")
    ap.add_argument("--lambda", dest="lam", type=float, default=None)
    a = ap.parse_args()
    out = run_backtest(list(read_jsonl(resolve(a.signals))), load_prices(a.prices), {"lambda": a.lam} if a.lam else None)
    Path(resolve(a.out)).parent.mkdir(parents=True, exist_ok=True)
    Path(resolve(a.out)).write_text(json.dumps(out["summary"], indent=2))
    out["daily"].to_csv(resolve(a.out).with_suffix(".csv"))
    print(json.dumps(out["summary"], indent=2))


if __name__ == "__main__":
    main()


def walk_forward_select(signals: list[dict], prices: pd.DataFrame, split: str = "2024-01-01",
                        grid: dict | None = None) -> dict:
    """Choose (lambda, band, turnover_max) by net information ratio on data before ``split``,
    then report the chosen configuration on data from ``split`` onward (out of sample)."""
    grid = grid or {"lambda": [0.75, 1.5, 3.0], "band": [0.002, 0.005, 0.01], "turnover_max": [0.05, 0.2]}
    train_px = prices[prices.index < split]
    test_px = prices[prices.index >= pd.Timestamp(split) - pd.Timedelta(days=120)]  # warm-up for EWMA vol
    rows = []
    for lam in grid["lambda"]:
        for band in grid["band"]:
            for to in grid["turnover_max"]:
                p = {"lambda": lam, "band": band, "turnover_max": to}
                s = run_backtest(signals, train_px, p)["summary"]
                rows.append({**p, "train_net_ir": s["strategy"].get("information_ratio", 0.0), "train_gross_ir": s["strategy_gross"].get("information_ratio", 0.0),
                             "train_cost_bps": s["total_cost_bps"]})
    best = max(rows, key=lambda r: r["train_net_ir"])
    chosen = {k: best[k] for k in ("lambda", "band", "turnover_max")}
    test = run_backtest(signals, test_px, chosen)
    daily = test["daily"][test["daily"].index >= split]
    summ = {"strategy": metrics(daily["strategy"], daily["cap_weight"]), "strategy_gross": metrics(daily["strategy_gross"], daily["cap_weight"]),
            "cap_weight": metrics(daily["cap_weight"]), "equal_weight": metrics(daily["equal_weight"], daily["cap_weight"]),
            "total_cost_bps": float(daily["cost"].sum() * 1e4), "avg_daily_turnover": float(daily["turnover"].mean()),
            "period": [str(daily.index[0].date()), str(daily.index[-1].date())]}
    return {"grid": rows, "chosen": chosen, "split": split, "test": summ, "test_daily": daily}
