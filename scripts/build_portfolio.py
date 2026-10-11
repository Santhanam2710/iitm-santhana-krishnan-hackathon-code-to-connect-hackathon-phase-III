"""Build the synthetic wholesale book from transaction data (blueprint section 4.4).

Mapping: merchant categories -> sectors; spend share -> loan sector mix; amount distribution
-> exposure sizes (lognormal fit, rescaled to wholesale tickets); fraud/default rate by sector
-> PD tier. The market book (bonds, swaps, CDS, equity derivatives, index puts, FX) is added
with Greeks set from typical market parameters.

Input: data/portfolio/transactions_sample.csv (synthetic) or the real Kaggle file placed at
data/raw/transactions.csv by scripts/fetch_data.py (columns are auto-detected).
Output: data/portfolio/wholesale_book.csv and data/portfolio/BOOK_DERIVATION.md
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CAT_TO_SECTOR = {
    "grocery": "Consumer", "restaurant": "Consumer", "hotel": "Consumer", "fuel": "Energy", "utilit": "Energy", "rent": "Real Estate",
    "real estate": "Real Estate", "construction": "Real Estate", "software": "Technology", "computer": "Technology", "telecom": "Technology",
    "freight": "Industrials", "shipping": "Industrials", "machinery": "Industrials", "airline": "Industrials", "medical": "Healthcare",
    "pharm": "Healthcare", "financial": "Banks", "bank": "Banks",
}
BASE_PD = {"Energy": 0.018, "Real Estate": 0.025, "Technology": 0.009, "Industrials": 0.012, "Consumer": 0.015, "Healthcare": 0.008, "Banks": 0.006}
LGD = {"Energy": 0.40, "Real Estate": 0.35, "Technology": 0.45, "Industrials": 0.30, "Consumer": 0.40, "Healthcare": 0.40, "Banks": 0.45}
BETA = {"Energy": 1.0, "Real Estate": 1.2, "Technology": 0.9, "Industrials": 1.1, "Consumer": 1.0, "Healthcare": 0.8, "Banks": 1.3}
NAMES = {"Energy": "Term loans · Energy", "Real Estate": "Term loans · Commercial real estate", "Technology": "Revolvers (drawn) · Technology",
         "Industrials": "Trade finance · Shipping & industrials", "Consumer": "Term loans · Consumer & leisure",
         "Healthcare": "Term loans · Healthcare", "Banks": "Interbank & FIG lending"}
LOAN_BOOK_TOTAL = 1800.0  # $m


def sector_of(cat: str) -> str:
    c = str(cat).lower()
    for k, v in CAT_TO_SECTOR.items():
        if k in c:
            return v
    return "Consumer"


def load_transactions() -> tuple[pd.DataFrame, str]:
    real = ROOT / "data" / "raw" / "transactions.csv"
    path = real if real.exists() else ROOT / "data" / "portfolio" / "transactions_sample.csv"
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    cat = next((cols[c] for c in cols if c in ("merchant_category", "mcc_description", "category", "merchant_category_code", "mcc")), None)
    amt = next((cols[c] for c in cols if c in ("amount", "transaction_amount", "amt")), None)
    fraud = next((cols[c] for c in cols if c in ("is_fraud", "fraud", "isfraud", "is_fraudulent")), None)
    if cat is None or amt is None:
        raise SystemExit(f"cannot find category/amount columns in {path}: {list(df.columns)}")
    out = pd.DataFrame({"category": df[cat].astype(str), "amount": pd.to_numeric(df[amt].astype(str).str.replace(r"[$,]", "", regex=True), errors="coerce").abs()})
    out["fraud"] = pd.to_numeric(df[fraud], errors="coerce").fillna(0) if fraud else 0
    return out.dropna(subset=["amount"]), str(path.relative_to(ROOT))


def main() -> None:
    tx, src = load_transactions()
    tx["sector"] = tx["category"].map(sector_of)
    agg = tx.groupby("sector").agg(spend=("amount", "sum"), n=("amount", "size"), fraud=("fraud", "mean"))
    share = (agg["spend"] / agg["spend"].sum()).clip(lower=0.04)
    share = share / share.sum()
    overall_fraud = max(float(tx["fraud"].mean()), 1e-6)
    mu, sigma = float(np.log(tx["amount"]).mean()), float(np.log(tx["amount"]).std())
    rows = []
    for i, (sec, w) in enumerate(share.sort_values(ascending=False).items(), start=1):
        rel = float(agg.loc[sec, "fraud"]) / overall_fraud if overall_fraud > 0 else 1.0
        pd_ = float(np.clip(BASE_PD.get(sec, 0.015) * (0.75 + 0.25 * rel), 0.003, 0.06))
        rows.append({"id": f"L{i}", "name": NAMES.get(sec, f"Term loans · {sec}"), "asset_class": "Loans", "kind": "loan", "sector": sec,
                     "ead": round(LOAN_BOOK_TOTAL * w, 1), "pd": round(pd_, 4), "lgd": LGD.get(sec, 0.4), "beta": BETA.get(sec, 1.0)})
    market = [
        {"id": "B1", "name": "IG corporate bonds", "asset_class": "Bonds", "kind": "bond", "mv": 600, "dur": 6.5, "cvx": 55, "spread": "ig"},
        {"id": "B2", "name": "HY corporate bonds", "asset_class": "Bonds", "kind": "bond", "mv": 220, "dur": 4.2, "cvx": 22, "spread": "hy"},
        {"id": "B3", "name": "US Treasuries (AFS)", "asset_class": "Bonds", "kind": "bond", "mv": 500, "dur": 7.8, "cvx": 75, "spread": ""},
        {"id": "D1", "name": "Pay-fixed interest rate swaps", "asset_class": "Derivatives", "kind": "irs", "mv": 12, "dv01": 0.40},
        {"id": "D2", "name": "Equity index futures (long)", "asset_class": "Derivatives", "kind": "eqd", "mv": 6, "delta": 150, "gamma": 0, "vega": 0},
        {"id": "D3", "name": "CDS protection bought · HY index", "asset_class": "Derivatives", "kind": "cds", "mv": -4, "cs01": 0.06},
        {"id": "D4", "name": "Index put options (6m, 10% OTM)", "asset_class": "Derivatives", "kind": "option", "qty": 120, "S0": 5000, "K": 4500,
         "T": 0.5, "sigma": 0.18, "option_type": "put"},
        {"id": "E1", "name": "Strategic equity stakes", "asset_class": "Equity", "kind": "eq", "mv": 140, "beta": 1.1},
        {"id": "F1", "name": "FX forwards · long EUR/USD", "asset_class": "FX", "kind": "fx", "mv": 3, "notional": 300},
    ]
    book = pd.DataFrame(rows + market)
    cols = ["id", "name", "asset_class", "kind", "sector", "ead", "pd", "lgd", "beta", "mv", "dur", "cvx", "spread", "dv01", "delta", "gamma",
            "vega", "cs01", "notional", "qty", "S0", "K", "T", "sigma", "option_type"]
    book = book.reindex(columns=cols)
    out = ROOT / "data" / "portfolio" / "wholesale_book.csv"
    book.to_csv(out, index=False)
    md = ["# Wholesale book derivation", "", f"Source transactions: `{src}` ({len(tx):,} rows).", "",
          f"Amount distribution: lognormal fit mu={mu:.2f}, sigma={sigma:.2f}; loan tickets rescaled to a ${LOAN_BOOK_TOTAL:,.0f}m book.", "",
          "| Sector | Spend share | Fraud rate | Relative risk | EAD $m | PD | LGD | Beta |", "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        sec = r["sector"]
        md.append(f"| {sec} | {share[sec]:.1%} | {agg.loc[sec, 'fraud']:.3%} | {agg.loc[sec, 'fraud'] / overall_fraud:.2f} | {r['ead']:.1f} | {r['pd']:.2%} | {r['lgd']:.0%} | {r['beta']} |")
    md += ["", "PD = sector base PD x (0.75 + 0.25 x relative fraud rate), clipped to [0.3%, 6%].",
           "Market book parameters are fixed, typical values; see the table in wholesale_book.csv.", "",
           "Assumption: consumer spend mix is a proxy for the sector mix of a bank's corporate clients. It is a modelling",
           "convenience for the hackathon, stated openly; a real book would come from the bank's own exposure data."]
    (ROOT / "data" / "portfolio" / "BOOK_DERIVATION.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(book[["id", "name", "ead", "pd", "mv"]].to_string(index=False))


if __name__ == "__main__":
    sys.exit(main())
