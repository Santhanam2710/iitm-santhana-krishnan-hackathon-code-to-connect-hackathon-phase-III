"""Rationale generation: deterministic template, or an LLM under strict guards.

The LLM path calls any OpenAI-compatible chat endpoint (Ollama by default) and is
used off the critical path. Guards (red-team item 5):
  1. JSON response parsed against a fixed shape;
  2. the ticker must be one the engine already resolved;
  3. every quoted evidence span must appear verbatim in the source text;
  4. temperature 0.
Any failed guard discards the LLM output and keeps the template.
"""
from __future__ import annotations

import json
from typing import Any

import httpx

from trident.config import get_config
from trident.utils.common import get_logger

log = get_logger("trident.rationale")


def template(label: str, event_conf: float, score: float, evidence: list[str], impact: int, base: float,
             magnitude: float, market_wide: bool, source_type: str, flags: list[str]) -> str:
    if label == "Other" and not evidence:
        return "No market-relevant event detected; published with low impact for monitoring only."
    ev = " and ".join(f"“{e}”" for e in evidence[:2]) if evidence else "no strong polarity terms"
    parts = [
        f"{label} ({event_conf:.0%} confidence).",
        f"Sentiment {score:+.2f} driven by {ev}.",
        f"Impact {impact}/10 from base severity {base:g}, |s| {abs(score):.2f}, magnitude cues {magnitude:.1f}"
        + (", market-wide scope" if market_wide else "")
        + f", {source_type} source.",
    ]
    if flags:
        parts.append("Flags: " + ", ".join(flags) + ".")
    return " ".join(parts)


PROMPT = """You explain financial risk signals. Use only the text provided.
Text: {text}
Computed fields: ticker={ticker}, event={label}, sentiment={score:+.2f}, impact={impact}/10.
Return JSON only: {{"rationale": "<= 45 words", "evidence": ["exact phrase copied from the text", ...]}}"""


async def llm_rationale(text: str, ticker: str, label: str, score: float, impact: int, allowed: set[str]) -> str | None:
    cfg = get_config().section("engine")
    payload: dict[str, Any] = {
        "model": cfg.get("llm_model", "llama3.1:8b"),
        "temperature": 0,
        "messages": [{"role": "user", "content": PROMPT.format(text=text, ticker=ticker, label=label, score=score, impact=impact)}],
        "response_format": {"type": "json_object"},
    }
    try:
        async with httpx.AsyncClient(timeout=20) as client:
            r = await client.post(cfg.get("llm_base_url", "http://localhost:11434/v1").rstrip("/") + "/chat/completions", json=payload)
            r.raise_for_status()
            content = r.json()["choices"][0]["message"]["content"]
        data = json.loads(content)
        rationale = str(data["rationale"]).strip()
        evidence = [str(e) for e in data.get("evidence", [])]
    except Exception as exc:
        log.debug("LLM rationale unavailable: %s", exc)
        return None
    if ticker not in allowed:
        return None
    if not evidence or any(e not in text for e in evidence):
        log.info("LLM rationale rejected: evidence not verbatim in source")
        return None
    if len(rationale.split()) > 60:
        return None
    return rationale
