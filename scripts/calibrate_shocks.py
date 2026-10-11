"""Compare configured stress presets with actual historical moves (needs data/market/fred.csv and
data/market/prices_sp15.csv from scripts/fetch_data.py --fred --prices).

    python scripts/calibrate_shocks.py
Prints, for each window, the realized change in the 10y yield, IG/HY OAS, VIX, EUR/USD and an
equal-weighted index of the 15 names, next to the preset in configs/scenarios.yaml.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
WINDOWS = {"covid_2020": ("2020-02-19", "2020-03-23"), "svb_2023": ("2023-03-08", "2023-03-17"), "rates_2022": ("2022-06-09", "2022-06-16")}


def main() -> None:
    fred_f, px_f = ROOT / "data/market/fred.csv", ROOT / "data/market/prices_sp15.csv"
    if not fred_f.exists():
        sys.exit("run: python scripts/fetch_data.py --fred --prices")
    fred = pd.read_csv(fred_f, parse_dates=["date"]).set_index("date").ffill()
    px = pd.read_csv(px_f, parse_dates=["date"]).set_index("date") if px_f.exists() else None
    presets = yaml.safe_load((ROOT / "configs/scenarios.yaml").read_text())["presets"]
    for name, (a, b) in WINDOWS.items():
        f0, f1 = fred.loc[:a].iloc[-1], fred.loc[:b].iloc[-1]
        real = {"rates": (f1["ust10y"] - f0["ust10y"]) * 100, "ig": (f1["ig_oas"] - f0["ig_oas"]) * 100, "hy": (f1["hy_oas"] - f0["hy_oas"]) * 100,
                "vol": f1["vix"] - f0["vix"], "fx": (f1["eurusd"] / f0["eurusd"] - 1) * 100}
        if px is not None:
            ew = px.pct_change().mean(axis=1).loc[a:b]
            real["eq"] = ((1 + ew).prod() - 1) * 100
        p = presets[name]["shock"]
        print(f"\n{name} {a} -> {b}")
        for k, v in real.items():
            print(f"  {k:6s} realized {v:+8.1f}   preset {p.get(k, 0):+8.1f}")


if __name__ == "__main__":
    main()
