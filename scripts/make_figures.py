"""Render the figures used by the README and the slide deck from docs/results/*.

    python scripts/make_figures.py
Outputs PNGs in docs/figures/. Colour grammar matches the dashboard: green positive, red
negative, amber = engine attention, ink for totals.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "docs" / "results"
FIG = ROOT / "docs" / "figures"
INK, MUTED, LINE, ACCENT, BLUE, POS, NEG, WARN = "#0d1626", "#556275", "#d6dce5", "#b4540b", "#1f4f8f", "#0d7a5c", "#b3261e", "#9a6700"

plt.rcParams.update({
    "font.family": ["IBM Plex Sans", "Inter", "DejaVu Sans"], "font.size": 11, "axes.edgecolor": LINE, "axes.labelcolor": MUTED,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": LINE, "grid.linewidth": 0.6, "axes.axisbelow": True, "axes.titleweight": "bold", "axes.titlesize": 13,
    "axes.titlecolor": INK, "figure.dpi": 150, "savefig.bbox": "tight", "savefig.facecolor": "white",
})


def save(fig, name: str) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / name)
    plt.close(fig)
    print("wrote", FIG / name)


def sentiment_baselines(r: dict) -> None:
    sets = r["nlp"]["sentiment"]
    names = list(sets)
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    vals = [("naive word count", "naive_wordcount_baseline", LINE), ("VADER (general)", "vader_baseline", MUTED), ("TRIDENT", "trident_best", ACCENT)]
    w = 0.26
    for i, (lab, key, col) in enumerate(vals):
        v = [sets[n].get(key, np.nan) for n in names]
        bars = ax.bar(x + (i - 1) * w, v, w, label=lab, color=col)
        for b, val in zip(bars, v):
            if np.isfinite(val):
                ax.text(b.get_x() + b.get_width() / 2, val + 0.015, f"{val:.2f}", ha="center", fontsize=9, color=INK)
    hard = [n for n in names if "hard_cases_lexicon_acc" in sets[n]]
    ax.set_xticks(x, [n.replace("_", " ") for n in names])
    ax.set_ylim(0, 1)
    ax.set_ylabel("macro-F1")
    ax.set_title("Sentiment: finance-aware engine vs general-purpose baselines", loc="left")
    ax.legend(frameon=False, ncol=3, loc="upper left", fontsize=9)
    ax.set_ylim(0, 1.08)
    if hard:
        n = hard[0]
        ax.text(0, -0.2, f"Hard cases (negation, sarcasm, delta phrases): TRIDENT {sets[n]['hard_cases_lexicon_acc']:.0%} vs VADER {sets[n].get('hard_cases_vader_acc', float('nan')):.0%} accuracy. "
                f"gold v0 = authored set; re-run on real data for submission.", transform=ax.transAxes, ha="left", fontsize=8, color=MUTED)
    save(fig, "fig_sentiment_baselines.png")


def latency(r: dict) -> None:
    lat = r["latency"]
    fig, ax = plt.subplots(figsize=(7.2, 2.6))
    stages = [("Engine p50", lat["engine_ms"]["p50"], BLUE), ("Engine p95", lat["engine_ms"]["p95"], BLUE), ("Engine p99", lat["engine_ms"]["p99"], BLUE),
              ("Module A rebalance p50", lat["module_a_rebalance_ms"]["p50"], POS), ("Module B full revaluation p50", lat["module_b_full_revaluation_ms"]["p50"], ACCENT)]
    y = np.arange(len(stages))[::-1]
    ax.barh(y, [s[1] for s in stages], color=[s[2] for s in stages], height=0.6)
    for yy, s in zip(y, stages):
        ax.text(s[1] + 0.05, yy, f"{s[1]:.2f} ms", va="center", fontsize=9.5, color=INK)
    ax.set_yticks(y, [s[0] for s in stages])
    ax.set_xlabel("milliseconds per item (CPU, no GPU)")
    ax.set_title(f"Latency: {lat['throughput_docs_per_s']:.0f} documents/s on one CPU process", loc="left")
    ax.set_xlim(0, max(s[1] for s in stages) * 1.35)
    save(fig, "fig_latency.png")


def impact(r: dict) -> None:
    f = RES / "impact_calibration_synthetic.csv"
    if not f.exists():
        return
    df = pd.read_csv(f)
    g = df.groupby("impact")["abs_car"].agg(["mean", "count"]).reset_index()
    g = g[g["count"] >= 5]
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    ax.bar(g["impact"].astype(int).astype(str), g["mean"] * 1e4, color=ACCENT)
    for i, (m, c) in enumerate(zip(g["mean"], g["count"])):
        ax.text(i, m * 1e4 + 8, f"{m * 1e4:.0f}", ha="center", fontsize=9, color=INK)
        ax.text(i, 6, f"n={c}", ha="center", fontsize=7.5, color="white")
    ax.set_xlabel("TRIDENT impact score")
    ax.set_ylabel("mean |CAR[0,+1]|, bps")
    rho = r["impact_synthetic"]["spearman_default_impact_vs_abs_car"]
    ax.set_title(f"Impact score tracks realized move size (Spearman {rho:.2f}, out of sample)", loc="left", fontsize=11.5)
    ax.text(0, -0.24, "All events, known-answer synthetic market: validates the pipeline, not a performance claim.", transform=ax.transAxes, fontsize=8, color=MUTED)
    save(fig, "fig_impact_calibration.png")


def backtest(r: dict) -> None:
    f = RES / "backtest_synthetic_test_daily.csv"
    if not f.exists():
        return
    d = pd.read_csv(f, parse_dates=["date"]).set_index("date")
    fig, ax = plt.subplots(figsize=(7.2, 3.3))
    for col, lab, c, ls in (("strategy_gross", "TRIDENT gross", ACCENT, "--"), ("strategy", "TRIDENT net of costs", ACCENT, "-")):
        ax.plot(d.index, ((1 + d[col]).cumprod() / (1 + d["cap_weight"]).cumprod() - 1) * 100, color=c, ls=ls, lw=1.8, label=lab)
    ax.plot(d.index, ((1 + d["equal_weight"]).cumprod() / (1 + d["cap_weight"]).cumprod() - 1) * 100, color=MUTED, lw=1.2, label="equal weight")
    ax.axhline(0, color=INK, lw=0.8)
    ax.set_ylabel("cumulative return vs cap weight, %")
    bt = r["backtest_synthetic"]
    ax.set_title(f"Module A out of sample: net IR {bt['strategy']['information_ratio']:.2f}, gross IR {bt['strategy_gross']['information_ratio']:.2f}", loc="left")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    import matplotlib.dates as mdates

    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 7]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.text(0, -0.2, f"Parameters chosen on 2019-2023 only: {bt['chosen']}. Known-answer synthetic market.", transform=ax.transAxes, fontsize=8, color=MUTED)
    save(fig, "fig_backtest.png")


def stress(r: dict) -> None:
    by = r["stress"]["credit9_by_class"]
    cls = list(by)
    vals = [by[c] for c in cls]
    fig, ax = plt.subplots(figsize=(6.6, 3.4))
    run = 0.0
    for i, v in enumerate(vals):
        ax.bar(i, v, bottom=run, color=POS if v >= 0 else NEG, width=0.6)
        ax.text(i, run + v + (2 if v >= 0 else -2), f"{v:+.1f}", ha="center", va="bottom" if v >= 0 else "top", fontsize=9, color=INK)
        run += v
    ax.bar(len(vals), run, color=INK, width=0.6)
    ax.text(len(vals), run - 2, f"{run:+.1f}", ha="center", va="top", fontsize=9.5, color=INK, fontweight="bold")
    ax.set_xticks(range(len(vals) + 1), cls + ["Total"])
    ax.axhline(0, color=INK, lw=0.8)
    ax.set_ylabel("$m")
    s9 = r["stress"]["credit9"]
    ax.set_title(f"Credit Event, impact 9: P&L {s9['pnl']:+.0f}m, CET1 {s9['cet1_before']:.1%} → {s9['cet1_after']:.1%}", loc="left", fontsize=11.5)
    save(fig, "fig_stress_waterfall.png")

    t = pd.DataFrame(r["stress"]["table"])
    t = t[(t["class"] == "preset") | (t["impact"].isin([8, 10]))]
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    y = np.arange(len(t))[::-1]
    cols = [NEG if v < 0.07 else (WARN if v < 0.09 else BLUE) for v in t["cet1_after"]]
    ax.barh(y, t["cet1_after"] * 100, color=cols, height=0.6)
    ax.axvline(4.5, color=NEG, ls="--", lw=1)
    ax.axvline(7.0, color=WARN, ls="--", lw=1)
    ax.axvline(t["cet1_before"].iloc[0] * 100, color=MUTED, ls=":", lw=1)
    ax.text(4.5, len(t) - 0.3, "4.5% min", color=NEG, fontsize=8, ha="center")
    ax.text(7.0, len(t) - 0.3, "7% buffer", color=WARN, fontsize=8, ha="center")
    for yy, v in zip(y, t["cet1_after"]):
        ax.text(v * 100 + 0.1, yy, f"{v:.1%}", va="center", fontsize=8.5)
    ax.set_yticks(y, [s.replace(" (approximate moves)", "*") for s in t["scenario"]], fontsize=8.5)
    ax.set_xlabel("CET1 ratio after stress, %  (start 13.0%)")
    ax.set_title("Capital impact across scenarios", loc="left")
    save(fig, "fig_cet1_scenarios.png")


def svb(r: dict) -> None:
    f = RES / "replay_svb_2023-03_decisions.csv"
    s = RES / "replay_svb_2023-03_signals.jsonl"
    if not f.exists():
        return
    dec = pd.read_csv(f, parse_dates=["ts"])
    sig = pd.read_json(s, lines=True)
    sig["ts"] = pd.to_datetime(sig["ts_published"])
    sig["sent"] = sig["sentiment_score"]
    fig, ax = plt.subplots(figsize=(7.6, 3.0))
    ax.scatter(sig["ts"], sig["impact_score"], c=[NEG if v < -0.1 else (POS if v > 0.1 else MUTED) for v in sig["sent"]], s=22, alpha=0.8, label="signals (color = sentiment)")
    held = dec[dec["state"] == "held"]
    trig = dec[dec["state"] == "triggered"]
    ax.scatter(held["ts"], held["impact"] + 0.25, marker="v", color=WARN, s=50, label="held at gate (1 source)")
    ax.scatter(trig["ts"], trig["impact"] + 0.45, marker="*", color=NEG, s=140, label="stress test triggered")
    ax.axhline(7, color=MUTED, ls="--", lw=0.8)
    ax.set_ylim(0, 11)
    ax.set_ylabel("impact score")
    ax.set_title("SVB replay, 8-19 March 2023: signals, gate decisions, triggers", loc="left")
    ax.legend(frameon=False, fontsize=8, loc="lower right", ncol=3)
    fig.autofmt_xdate()
    save(fig, "fig_svb_timeline.png")


def main() -> None:
    r = json.loads((RES / "results.json").read_text())
    sentiment_baselines(r)
    latency(r)
    impact(r)
    backtest(r)
    stress(r)
    svb(r)


if __name__ == "__main__":
    main()
