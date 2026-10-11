"""Live Module B: trigger gate (eq. B1) + stress runs on qualifying signals."""
from __future__ import annotations

from collections import deque
from datetime import datetime, timedelta
from typing import Any

from trident.config import get_config
from trident.modules.stress.engine import StressEngine
from trident.utils.common import iso, parse_ts, ulid


class TriggerGate:
    """Fires when class in stress set, impact > I*, confidence >= min, and either
    >= ``confirm_sources`` distinct sources reported the class within the window,
    or impact >= ``single_source_override``. A cooldown prevents repeated triggers."""

    def __init__(self, cfg: dict[str, Any] | None = None):
        self.c = cfg or get_config().scenarios["trigger"]
        self.recent: deque[tuple[datetime, str, str, int]] = deque(maxlen=5000)  # (ts, class, source, impact)
        self.last_fire: dict[str, tuple[datetime, float]] = {}

    def evaluate(self, sig: dict[str, Any]) -> dict[str, Any]:
        c = self.c
        label = sig["event_classification"]["label"]
        impact = int(sig["impact_score"])
        conf = float(sig["event_classification"]["confidence"])
        ts = parse_ts(sig["ts_published"])
        src = sig["source"]["name"]
        retro = any(f.startswith("retrospective") for f in sig.get("flags", []))
        if label in c["classes"] and impact > c["impact_threshold"] and not retro:
            self.recent.append((ts, label, src, impact))
        window = timedelta(minutes=c["confirm_window_min"])
        sources = {s for t, l, s, i in self.recent if l == label and timedelta(0) <= ts - t <= window}
        state, reason = "monitored", ""
        if label not in c["classes"]:
            reason = f"{label} is not a stress class"
        elif retro:
            reason = "retrospective article"
        elif impact <= c["impact_threshold"]:
            reason = f"impact {impact} is not above {c['impact_threshold']}"
        elif conf < c["min_confidence"]:
            reason = f"event confidence {conf:.2f} below {c['min_confidence']}"
        elif len(sources) < c["confirm_sources"] and impact < c["single_source_override"]:
            state, reason = "held", f"{len(sources)} source(s); needs {c['confirm_sources']} within {c['confirm_window_min']} min or impact >= {c['single_source_override']}"
        else:
            eff = min(10.0, impact + 0.25 * max(len(sources) - 1, 0))
            last = self.last_fire.get(label)
            if last and ts - last[0] < timedelta(minutes=c["cooldown_min"]) and eff <= last[1]:
                state, reason = "cooldown", f"{label} stress already run at {iso(last[0])} with effective impact {last[1]:g}"
            else:
                state = "triggered"
                reason = (f"{len(sources)} independent source(s)" if len(sources) >= c["confirm_sources"] else f"single source with impact {impact}")
                if last and ts - last[0] < timedelta(minutes=c["cooldown_min"]):
                    reason += f"; escalation {last[1]:g} -> {eff:g}"
                self.last_fire[label] = (ts, eff)
            return {"state": state, "reason": reason, "sources": sorted(sources), "label": label, "impact": impact, "effective_impact": eff}
        return {"state": state, "reason": reason, "sources": sorted(sources), "label": label, "impact": impact}


class StressService:
    def __init__(self):
        self.engine = StressEngine()
        self.gate = TriggerGate()
        self.history: deque[dict] = deque(maxlen=200)
        self.decisions: deque[dict] = deque(maxlen=500)
        self.contagion = True

    def on_signal(self, sig: dict[str, Any]) -> dict[str, Any] | None:
        d = self.gate.evaluate(sig)
        d.update({"signal_id": sig["signal_id"], "ts": sig["ts_published"], "ticker": sig["entity"]["ticker"]})
        self.decisions.append(d)
        if d["state"] != "triggered":
            return None
        return self.run_for(sig["event_classification"]["label"], float(d.get("effective_impact", sig["impact_score"])), trigger=sig, decision=d)

    def run_for(self, event_class: str, impact: float, trigger: dict | None = None, decision: dict | None = None,
                preset: str | None = None) -> dict[str, Any]:
        if preset:
            pr = get_config().scenarios["presets"][preset]
            shock, mult, label = dict(pr["shock"]), 1.0, pr["label"]
            event_class = pr["event_class"]
        else:
            shock, mult = self.engine.scenario(event_class, impact)
            label = f"{event_class}, impact {impact:g}"
        sector = (trigger or {}).get("entity", {}).get("sector")
        if self.contagion and sector and sector not in ("Market",):
            shock = self.engine.contagion(shock, sector)
        res = self.engine.run(shock, label)
        base = get_config().scenarios["presets"][preset]["shock"] if preset else get_config().scenarios["scenarios"].get(event_class, shock)
        res["reverse"] = self.engine.reverse_stress(base)
        if preset:
            res["reverse"]["implied_impact"] = None
        res.update({"run_id": "run_" + ulid(), "ts": trigger["ts_published"] if trigger else iso(),
                    "event_class": event_class, "impact": impact, "multiplier": mult, "preset": preset,
                    "trigger_signal": trigger["signal_id"] if trigger else None,
                    "trigger_text": trigger.get("text") if trigger else None, "decision": decision})
        self.history.append(res)
        return res
