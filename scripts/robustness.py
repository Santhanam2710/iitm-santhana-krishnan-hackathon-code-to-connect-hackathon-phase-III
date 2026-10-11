"""Known-answer robustness: rebuild the synthetic market with several seeds and check that the
pipeline recovers the planted structure every time (not just on one lucky draw).

    python scripts/robustness.py --seeds 5
Writes docs/results/robustness.json: per-seed and mean/sd of impact Spearman (out of sample),
walk-forward net and gross IR, and IC.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from make_synthetic_data import known_answer_market  # noqa: E402
from trident.eval.ic import daily_ic  # noqa: E402
from trident.modules.rebalancer.backtest import walk_forward_select  # noqa: E402
from trident.pipeline.engine import Engine  # noqa: E402
from trident.pipeline.normalize import make_doc  # noqa: E402
from trident.training.train_impact import event_study  # noqa: E402


def one(seed: int) -> dict:
    prices, news = known_answer_market(seed=seed)
    eng = Engine()
    docs = sorted((make_doc(r["text"], r["source_type"], r["source_name"], r["ts"], is_synthetic=True) for r in news), key=lambda d: d.ts_published)
    sigs = []
    for i in range(0, len(docs), 64):
        for o in eng.process_batch(docs[i : i + 64]):
            sigs.extend(s.model_dump() for s in o.signals)
    es = event_study([s for s in sigs if s["entity"].get("in_index")], prices)
    test = es["date"] >= pd.Timestamp("2025-01-01")
    rho = float(spearmanr(es.loc[test, "impact"], es.loc[test, "abs_car"]).statistic)
    wf = walk_forward_select(sigs, prices, grid={"lambda": [0.75, 1.5], "band": [0.002, 0.005, 0.01], "turnover_max": [0.2]})
    ic = daily_ic(sigs, prices)
    return {"seed": seed, "events": int(len(es)), "impact_spearman_oos": rho, "net_ir_oos": wf["test"]["strategy"]["information_ratio"],
            "gross_ir_oos": wf["test"]["strategy_gross"]["information_ratio"], "ic": ic["mean_ic"], "chosen": wf["chosen"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    a = ap.parse_args()
    rows = [one(1000 + 17 * k) for k in range(a.seeds)]
    for r in rows:
        print(r)
    summ = {k: {"mean": float(np.mean([r[k] for r in rows])), "sd": float(np.std([r[k] for r in rows], ddof=1)), "min": float(np.min([r[k] for r in rows]))}
            for k in ("impact_spearman_oos", "net_ir_oos", "gross_ir_oos", "ic")}
    out = {"seeds": a.seeds, "summary": summ, "runs": rows,
           "note": "Known-answer synthetic markets: tests that the pipeline recovers planted structure. Not a performance claim."}
    (ROOT / "docs" / "results" / "robustness.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(summ, indent=2))


if __name__ == "__main__":
    main()
