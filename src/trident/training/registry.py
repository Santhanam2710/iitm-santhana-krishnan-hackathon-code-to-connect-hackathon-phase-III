"""Champion/challenger model registry (model-risk-management practice).

A newly trained model is a *challenger*. It replaces the *champion* (the file the engine
loads) only if it scores higher on the same evaluation set, preferring real held-out data.
Every decision is appended to models/registry.json, so the deck can show the audit trail.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from trident.config import MODELS_DIR
from trident.utils.common import iso

REGISTRY = MODELS_DIR / "registry.json"


def _load() -> dict:
    return json.loads(REGISTRY.read_text()) if REGISTRY.exists() else {"champions": {}, "log": []}


def decide(kind: str, candidate: Path, champion_path: Path, cand_score: float, baseline_name: str, baseline_score: float,
           eval_name: str, force: bool = False) -> dict:
    reg = _load()
    champ = reg["champions"].get(kind)
    bar = baseline_score
    bar_name = baseline_name
    if champ and champ.get("eval") == eval_name and champ.get("score", -1) > bar and champion_path.exists():
        bar, bar_name = champ["score"], f"current champion ({champ['path']})"
    promote = force or cand_score > bar
    if promote:
        champion_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(candidate, champion_path)
        reg["champions"][kind] = {"path": str(champion_path.relative_to(MODELS_DIR.parent)), "score": cand_score, "eval": eval_name, "ts": iso()}
    entry = {"ts": iso(), "kind": kind, "candidate": str(candidate.relative_to(MODELS_DIR.parent)), "candidate_score": round(cand_score, 4),
             "bar": bar_name, "bar_score": round(bar, 4), "eval": eval_name, "decision": "promoted" if promote else "rejected", "forced": force}
    reg["log"].append(entry)
    REGISTRY.parent.mkdir(parents=True, exist_ok=True)
    REGISTRY.write_text(json.dumps(reg, indent=2))
    return entry
