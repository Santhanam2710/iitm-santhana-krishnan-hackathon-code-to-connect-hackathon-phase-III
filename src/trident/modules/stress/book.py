"""Synthetic wholesale banking book (data/portfolio/wholesale_book.csv)."""
from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path

import pandas as pd

from trident.config import DATA_DIR


@dataclass
class Position:
    id: str
    name: str
    asset_class: str          # Loans | Bonds | Derivatives | Equity | FX
    kind: str                 # loan | bond | irs | eqd | option | cds | eq | fx
    sector: str = ""
    ead: float = 0.0
    pd: float = 0.0
    lgd: float = 0.0
    beta: float = 1.0
    mv: float = 0.0
    dur: float = 0.0
    cvx: float = 0.0
    spread: str = ""          # ig | hy | "" (sovereign)
    dv01: float = 0.0
    delta: float = 0.0
    gamma: float = 0.0
    vega: float = 0.0
    cs01: float = 0.0
    notional: float = 0.0
    qty: float = 0.0          # option contracts in $m of index notional
    S0: float = 0.0
    K: float = 0.0
    T: float = 0.0
    sigma: float = 0.0
    option_type: str = ""     # put | call


def load_book(path: str | Path | None = None) -> list[Position]:
    p = Path(path) if path else DATA_DIR / "portfolio" / "wholesale_book.csv"
    df = pd.read_csv(p).fillna({"sector": "", "spread": "", "option_type": ""}).fillna(0)
    names = {f.name for f in fields(Position)}
    out = []
    for row in df.to_dict(orient="records"):
        out.append(Position(**{k: v for k, v in row.items() if k in names}))
    return out
