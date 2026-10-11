"""Regenerate every number in the deck and README from code. One command:

    python scripts/run_all_results.py            # ~1-3 minutes on a laptop
    python scripts/run_all_results.py --skip-train

Writes docs/results/results.json (machine-readable, served at /api/results), docs/results/RESULTS.md
(human-readable, with provenance for every number) and the CSVs used by scripts/make_figures.py.

Data provenance rules applied automatically:
  * real datasets are used whenever scripts/fetch_data.py has placed them in data/raw/;
  * otherwise the authored gold set and the synthetic known-answer market are used, and every
    affected number is labeled as such in RESULTS.md and on the slides.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from trident.config import DATA_DIR, MODELS_DIR  # noqa: E402
from trident.utils.common import iso, read_jsonl, write_jsonl  # noqa: E402

OUT = ROOT / "docs" / "results"


def sh(*args: str) -> None:
    print("$", " ".join(args), flush=True)
    subprocess.run([sys.executable, *args], cwd=ROOT, check=True, env={**__import__("os").environ, "PYTHONPATH": str(ROOT / "src")})


def ensure_data() -> None:
    if not (DATA_DIR / "replay" / "svb_2023-03.jsonl").exists():
        sh("scripts/build_replay.py")
    if not (DATA_DIR / "gold" / "headlines_gold_v0.jsonl").exists():
        sh("scripts/build_gold_v0.py")
    if not (DATA_DIR / "synthetic" / "market_news.jsonl").exists():
        sh("scripts/make_synthetic_data.py")
    if not (DATA_DIR / "portfolio" / "wholesale_book.csv").exists():
        sh("scripts/build_portfolio.py")


def replay_summary(name: str) -> dict:
    from trident.ingest.replay import load_replay
    from trident.modules.rebalancer.service import Rebalancer
    from trident.modules.stress.service import StressService
    from trident.pipeline.engine import Engine
    from trident.pipeline.normalize import make_doc
    from trident.utils.common import parse_ts

    rows = load_replay(name)
    eng, reb, st = Engine(), Rebalancer(), StressService()
    docs = [make_doc(r["text"], r["source_type"], r["source_name"], r["ts"], is_synthetic=r.get("is_synthetic", False), replay_id=name) for r in rows]
    outs = eng.process_batch(docs)
    sigs = [s.model_dump() for o in outs for s in o.signals]
    runs, weights = [], []
    for s in sigs:
        reb.on_signal(s)
        r = st.on_signal(s)
        rec = reb.rebalance(parse_ts(s["ts_published"]))
        weights.append({"ts": rec["ts"], **{t: rec["weights"][t] - rec["benchmark"][t] for t in rec["weights"]}})
        if r:
            runs.append(r)
    pd.DataFrame(weights).to_csv(OUT / f"replay_{name}_active_weights.csv", index=False)
    dec = list(st.decisions)
    pd.DataFrame(dec).to_csv(OUT / f"replay_{name}_decisions.csv", index=False)
    write_jsonl(OUT / f"replay_{name}_signals.jsonl", sigs)
    first_doc = rows[0]["ts"]
    first_trig = next((d for d in dec if d["state"] == "triggered"), None)
    held = sum(d["state"] == "held" for d in dec)
    dropped = {}
    for o in outs:
        if o.dropped:
            k = o.dropped.split(" of ")[0]
            dropped[k] = dropped.get(k, 0) + 1
    rejected = [r for o in outs for r in o.rejected_entities]
    first = runs[0] if runs else None
    return {"docs": len(rows), "signals": len(sigs), "dropped": dropped, "entity_rejections": len(rejected),
            "first_document": first_doc, "first_trigger": first_trig, "held_at_gate": held,
            "triggers": sum(d["state"] == "triggered" for d in dec),
            "first_stress": {k: first[k] for k in ("label", "pnl", "cet1_before", "cet1_after", "var_before", "var_after")} if first else None,
            "final_active_weights_pp": {k: round(v * 100, 3) for k, v in weights[-1].items() if k != "ts"} if weights else {}}


def stress_table() -> dict:
    from trident.modules.stress.service import StressService

    s = StressService()
    rows = []
    for cls in ("Credit Event", "Geopolitical", "Macroeconomic"):
        for I in (7, 8, 9, 10):
            r = s.run_for(cls, I)
            rows.append({"scenario": f"{cls} I={I}", "class": cls, "impact": I, "pnl": r["pnl"], "pnl_pct": r["pnl_pct"], "cet1_before": r["cet1_before"],
                         "cet1_after": r["cet1_after"], "var_after": r["var_after"], "es_after": r["es_after"],
                         **{f"dv_{k}": v["dv"] for k, v in r["by_class"].items()}})
    for p in ("covid_2020", "svb_2023", "rates_2022"):
        r = s.run_for("", 10, preset=p)
        rows.append({"scenario": r["label"], "class": "preset", "impact": None, "pnl": r["pnl"], "pnl_pct": r["pnl_pct"], "cet1_before": r["cet1_before"],
                     "cet1_after": r["cet1_after"], "var_after": r["var_after"], "es_after": r["es_after"], "reverse_multiple": r["reverse"]["multiple"],
                     **{f"dv_{k}": v["dv"] for k, v in r["by_class"].items()}})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "stress_scenarios.csv", index=False)
    base = s.run_for("Credit Event", 9)
    pd.DataFrame(base["positions"]).to_csv(OUT / "stress_credit9_positions.csv", index=False)
    contagion = s.engine.run(s.engine.contagion(s.engine.scenario("Credit Event", 9)[0], "Banks"), "Credit 9 + Banks contagion")
    from trident.config import get_config

    rev = s.engine.reverse_stress(get_config().scenarios["scenarios"]["Credit Event"])
    return {"credit9": {k: base[k] for k in ("pnl", "pnl_pct", "cet1_before", "cet1_after", "var_before", "var_after", "es_before", "es_after", "rwa_before", "rwa_after")},
            "credit9_by_class": {k: v["dv"] for k, v in base["by_class"].items()},
            "credit9_with_bank_contagion": {"pnl": contagion["pnl"], "cet1_after": contagion["cet1_after"]},
            "reverse_credit": rev, "var_unstressed": base["var_before"], "table": rows}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--robust", action="store_true", help="re-run the 5-seed robustness check (~1 min)")
    a = ap.parse_args()
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    ensure_data()
    if not a.skip_train:
        sh("-m", "trident.training.weak_labels")
        sh("-m", "trident.training.train_sentiment")
        sh("-m", "trident.training.train_events")

    from trident.eval import adversarial, checklist, latency_bench, nlp_eval
    from trident.eval.ic import daily_ic
    from trident.modules.rebalancer.backtest import load_prices, walk_forward_select

    res: dict = {"generated": iso(), "provenance": {}}
    res["nlp"] = nlp_eval.run()
    res["checklist"] = checklist.run()["_summary"]
    res["checklist_detail"] = {k: {kk: vv for kk, vv in v.items() if kk != "failures"} for k, v in checklist.run().items() if not k.startswith("_")}
    res["adversarial"] = adversarial.run()
    res["latency"] = latency_bench.run(3000)
    res["registry"] = json.loads((MODELS_DIR / "registry.json").read_text())["log"][-4:] if (MODELS_DIR / "registry.json").exists() else []

    # market: real if fetched, else synthetic known-answer
    real_px = DATA_DIR / "market" / "prices_sp15.csv"
    real_news = DATA_DIR / "raw" / "news_hist.jsonl"
    sh("-m", "trident", "signals", "data/synthetic/market_news.jsonl", "--out", "data/runtime/signals_hist.jsonl")
    syn_sigs = list(read_jsonl(ROOT / "data/runtime/signals_hist.jsonl"))
    syn_px = load_prices("data/market/prices_synthetic.csv")
    sh("-m", "trident.training.train_impact", "--signals", "data/runtime/signals_hist.jsonl", "--prices", "data/market/prices_synthetic.csv",
       "--no-save", "--metrics", "docs/results/impact_calibration_synthetic.json")
    res["impact_synthetic"] = json.loads((OUT / "impact_calibration_synthetic.json").read_text())
    res["ic_synthetic"] = daily_ic(syn_sigs, syn_px)
    wf = walk_forward_select(syn_sigs, syn_px)
    wf["test_daily"].to_csv(OUT / "backtest_synthetic_test_daily.csv")
    pd.DataFrame(wf["grid"]).to_csv(OUT / "backtest_synthetic_grid.csv", index=False)
    res["backtest_synthetic"] = {**wf["test"], "chosen": wf["chosen"], "split": wf["split"]}
    res["provenance"]["market"] = "synthetic known-answer market (data/market/prices_synthetic.csv)"
    if real_px.exists() and real_news.exists():
        sh("-m", "trident", "signals", "data/raw/news_hist.jsonl", "--out", "data/runtime/signals_real.jsonl")
        real_sigs = list(read_jsonl(ROOT / "data/runtime/signals_real.jsonl"))
        px = load_prices(real_px)
        lo, hi = min(s["ts_published"] for s in real_sigs)[:10], max(s["ts_published"] for s in real_sigs)[:10]
        px = px.loc[lo:hi]
        split = str(px.index[int(len(px) * 0.7)].date())  # first 70% selects parameters, last 30% is out of sample
        res["ic_real"] = daily_ic(real_sigs, px)
        wfr = walk_forward_select(real_sigs, px, split=split)
        wfr["test_daily"].to_csv(OUT / "backtest_real_test_daily.csv")
        res["backtest_real"] = {**wfr["test"], "chosen": wfr["chosen"], "split": wfr["split"]}
        try:
            sh("-m", "trident.training.train_impact", "--signals", "data/runtime/signals_real.jsonl", "--prices", str(real_px), "--test-from", split)
            res["impact_real"] = json.loads((OUT / "impact_calibration.json").read_text())
        except subprocess.CalledProcessError:
            res["impact_real"] = None
        res["provenance"]["market"] = "real prices (yfinance) + real headlines (data/raw/news_hist.jsonl)"

    if a.robust or not (OUT / "robustness.json").exists():
        sh("scripts/robustness.py", "--seeds", "5")
    res["robustness"] = json.loads((OUT / "robustness.json").read_text())["summary"]
    res["stress"] = stress_table()
    res["replay_svb"] = replay_summary("svb_2023-03")
    res["replay_covid"] = replay_summary("covid_2020-03")
    g = res["nlp"]["sentiment"]
    best_sent = max(g, key=lambda k: 0 if k == "gold_v0" else 1)
    res["headline"] = {
        "engine_latency_ms_p50": round(res["latency"]["engine_ms"]["p50"], 2), "engine_latency_ms_p95": round(res["latency"]["engine_ms"]["p95"], 2),
        "throughput_docs_per_s": round(res["latency"]["throughput_docs_per_s"]),
        "sentiment_macro_f1": round(g[best_sent]["trident_best"], 3), "sentiment_vader_f1": round(g[best_sent].get("vader_baseline", float("nan")), 3),
        "sentiment_eval_set": best_sent, "event_macro_f1": round(res["nlp"]["events"][max(res["nlp"]["events"], key=lambda k: 0 if k == "gold_v0" else 1)]["best_macro_f1"], 3),
        "entity_precision": round(res["nlp"]["entities"]["precision"], 3), "entity_recall": round(res["nlp"]["entities"]["recall"], 3),
        "checklist_pass": f"{res['checklist']['passed']}/{res['checklist']['total']}",
        "impact_spearman_oos": round(res["impact_synthetic"]["spearman_default_impact_vs_abs_car"], 3),
        "backtest_oos_net_ir": round(res["backtest_synthetic"]["strategy"]["information_ratio"], 2),
        "backtest_oos_gross_ir": round(res["backtest_synthetic"]["strategy_gross"]["information_ratio"], 2),
        "credit9_cet1": [round(res["stress"]["credit9"]["cet1_before"] * 100, 2), round(res["stress"]["credit9"]["cet1_after"] * 100, 2)],
        "credit9_pnl_musd": round(res["stress"]["credit9"]["pnl"], 1),
        "svb_first_trigger": res["replay_svb"]["first_trigger"]["ts"] if res["replay_svb"]["first_trigger"] else None,
        "adversarial_auc": round(res["adversarial"]["auc"], 3),
    }
    res["elapsed_s"] = round(time.time() - t0, 1)
    (OUT / "results.json").write_text(json.dumps(res, indent=2, default=str))
    write_markdown(res)
    sh("scripts/make_figures.py")
    print(json.dumps(res["headline"], indent=2))


def write_markdown(r: dict) -> None:
    h = r["headline"]
    s9 = r["stress"]["credit9"]
    bt = r["backtest_synthetic"]
    lines = [
        "# TRIDENT results", "", f"Generated {r['generated']} by `python scripts/run_all_results.py` in {r['elapsed_s']} s. Every number below is produced by code in this repository.", "",
        "## Provenance", "",
        "| Number | Data | Status |", "|---|---|---|",
        f"| NLP metrics | {', '.join(f'{k}: {v}' for k, v in r['nlp']['provenance'].items())} | {'REAL' if any('real' in v for v in r['nlp']['provenance'].values()) else 'AUTHORED SET ONLY: replace by running fetch_data.py + label_tool.py'} |",
        f"| Impact calibration, IC, backtest | {r['provenance']['market']} | {'REAL' if 'real' in r['provenance']['market'] else 'KNOWN-ANSWER VALIDATION, not a performance claim'} |",
        "| Latency | measured on this machine, lexicon/rules backends | REAL measurement |",
        "| Stress tests | synthetic wholesale book (data/portfolio) under configured scenarios | model output |",
        "| SVB / COVID replays | reconstructed public timelines + tagged synthetic posts | demo / integration test |", "",
        "## Engine", "",
        f"* Latency per document: p50 **{h['engine_latency_ms_p50']} ms**, p95 **{h['engine_latency_ms_p95']} ms**; throughput {h['throughput_docs_per_s']} docs/s ({r['latency']['machine']}).",
        f"* Module A rebalance p50 {r['latency']['module_a_rebalance_ms']['p50']:.2f} ms; Module B full revaluation p50 {r['latency']['module_b_full_revaluation_ms']['p50']:.2f} ms.",
        "", "### Sentiment (macro-F1, 3 classes)", "", "| Set | n | TRIDENT | VADER | naive word count |", "|---|---|---|---|---|"]
    for k, v in r["nlp"]["sentiment"].items():
        lines.append(f"| {k} | {v['n']} | {v['trident_best']:.3f} | {v.get('vader_baseline', float('nan')):.3f} | {v['naive_wordcount_baseline']:.3f} |")
    lines += ["", "### Events (macro-F1, 9 labels)", "", "| Set | n | Rules | Best | Model |", "|---|---|---|---|---|"]
    for k, v in r["nlp"]["events"].items():
        lines.append(f"| {k} | {v['n']} | {v['rules_macro_f1']:.3f} | {v['best_macro_f1']:.3f} | {v['model']} |")
    e = r["nlp"]["entities"]
    lines += ["", f"Entities: precision {e['precision']:.3f}, recall {e['recall']:.3f}, ambiguity traps {e['ambiguity_traps_correct']}. Behavioral tests: {h['checklist_pass']} passed.",
              "", "Caution: gold_v0 was written by the same author as the taxonomy keywords and lexicon, so its event and sentiment scores are optimistic. "
              "Report the real-data rows once fetched.", "",
              "### Model registry (champion/challenger)", ""]
    for x in r["registry"]:
        lines.append(f"* {x['ts']} {x['kind']}: candidate {x['candidate_score']} vs {x['bar']} {x['bar_score']} on {x['eval']} -> **{x['decision']}**")
    adv = r["adversarial"]
    lines += ["", "### Synthetic data realism", "", f"Adversarial AUC real vs synthetic = **{adv['auc']:.3f}** (target <= 0.65; passes: {adv['passes']}). "
              f"Top synthetic tells: {', '.join(adv['synthetic_tells'][:6])}. Template data is easy to detect, which is why it is capped at 30% of training data and never used for evaluation.",
              "", "## Impact calibration and Module A", "",
              f"* Spearman(impact, |CAR[0,+1]|) out of sample: **{h['impact_spearman_oos']}** ({r['impact_synthetic']['test_events']} test events). "
              f"Mean |CAR| by calibrated impact (bps): {r['impact_synthetic']['mean_abs_car_bps_by_calibrated_impact']}.",
              f"* IC (sentiment at close d-1 vs market-adjusted return d): mean {r['ic_synthetic']['mean_ic']:.3f}, t {r['ic_synthetic']['t_stat']:.2f}, days {r['ic_synthetic']['days']}.",
              f"* Walk-forward: parameters chosen on data before {bt['split']} = {bt['chosen']}; out-of-sample {bt['period'][0]} to {bt['period'][1]}: "
              f"net IR **{bt['strategy']['information_ratio']:.2f}**, gross IR {bt['strategy_gross']['information_ratio']:.2f}, cost {bt['total_cost_bps']:.0f} bps, "
              f"Sharpe {bt['strategy']['sharpe']:.2f} vs cap-weight {bt['cap_weight']['sharpe']:.2f}, max drawdown {bt['strategy']['max_drawdown']:.1%}.",
              f"* Robustness over 5 seeds (known-answer): impact Spearman {r['robustness']['impact_spearman_oos']['mean']:.2f} ± {r['robustness']['impact_spearman_oos']['sd']:.2f}, "
              f"IC {r['robustness']['ic']['mean']:.3f} ± {r['robustness']['ic']['sd']:.3f}, net IR {r['robustness']['net_ir_oos']['mean']:.2f} ± {r['robustness']['net_ir_oos']['sd']:.2f}, "
              f"gross IR {r['robustness']['gross_ir_oos']['mean']:.2f} ± {r['robustness']['gross_ir_oos']['sd']:.2f}. The planted signal is recovered every time; whether it pays after costs varies by draw.",
              "", "## Module B", "",
              f"* Credit Event, impact 9: P&L **{s9['pnl']:+.1f} $m** ({s9['pnl_pct']:+.2%}), CET1 **{s9['cet1_before']:.2%} -> {s9['cet1_after']:.2%}**, "
              f"RWA {s9['rwa_before']:.0f} -> {s9['rwa_after']:.0f} $m, 99% 10-day VaR {s9['var_before']:.1f} -> {s9['var_after']:.1f} $m.",
              f"* With Banks-sector contagion: P&L {r['stress']['credit9_with_bank_contagion']['pnl']:+.1f} $m, CET1 {r['stress']['credit9_with_bank_contagion']['cet1_after']:.2%}.",
              f"* Reverse stress (Credit Event): CET1 breaches 7% at {r['stress']['reverse_credit']['multiple']:.2f}x the full scenario." if r['stress']['reverse_credit']['breach'] else "* Reverse stress: no breach at 3x.",
              "", "| Scenario | P&L $m | CET1 after | VaR after $m |", "|---|---|---|---|"]
    for row in r["stress"]["table"]:
        lines.append(f"| {row['scenario']} | {row['pnl']:+.1f} | {row['cet1_after']:.2%} | {row['var_after']:.1f} |")
    sv = r["replay_svb"]
    lines += ["", "## SVB replay (8-19 March 2023)", "",
              f"{sv['docs']} items -> {sv['signals']} signals; dropped {sv['dropped']}; entity rejections {sv['entity_rejections']}; held at gate {sv['held_at_gate']}; triggers {sv['triggers']}.",
              f"First item {sv['first_document']}; first stress trigger {sv['first_trigger']['ts'] if sv['first_trigger'] else 'none'} ({sv['first_trigger']['reason'] if sv['first_trigger'] else ''}).", ""]
    (OUT / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
