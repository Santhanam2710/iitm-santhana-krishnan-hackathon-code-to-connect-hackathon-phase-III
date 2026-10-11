"""Pydantic models: the canonical document and the signal contract.

``Doc`` is what every ingestion adapter produces. ``Signal`` is what the engine emits
and what the API, the modules and the JSONL sink consume. The three fields the brief
requires are ``sentiment_score``, ``event_classification`` and ``impact_score``.
"""
from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, field_validator

SourceType = Literal["newswire", "filing", "gdelt", "social", "synthetic"]

EVENT_LABELS = (
    "Geopolitical",
    "Macroeconomic",
    "Credit Event",
    "Merger/Acquisition",
    "Product Launch",
    "Earnings",
    "Regulatory/Legal",
    "Operational",
    "Other",
)


class Doc(BaseModel):
    """Canonical document produced by every adapter."""

    doc_id: str
    text: str
    title: Optional[str] = None
    url: Optional[str] = None
    source_type: SourceType = "newswire"
    source_name: str = "unknown"
    ts_published: str
    ts_ingested: str
    lang: str = "en"
    is_synthetic: bool = False
    replay_id: Optional[str] = None
    meta: dict[str, Any] = Field(default_factory=dict)


class SourceInfo(BaseModel):
    type: SourceType
    name: str
    reliability: float
    url: Optional[str] = None
    cluster_id: Optional[str] = None
    novelty: float = 1.0


class EntityInfo(BaseModel):
    ticker: str
    name: str
    match: str
    confidence: float
    in_index: bool = False
    sector: Optional[str] = None


class EventInfo(BaseModel):
    label: str
    confidence: float
    secondary: Optional[str] = None
    subtype: Optional[str] = None
    probabilities: dict[str, float] = Field(default_factory=dict)

    @field_validator("label")
    @classmethod
    def _known(cls, v: str) -> str:
        if v not in EVENT_LABELS:
            raise ValueError(f"unknown event label {v}")
        return v


class Evidence(BaseModel):
    span: str
    start: int
    end: int
    polarity: Literal["pos", "neg", "event", "entity"]


class Rationale(BaseModel):
    text: str
    mode: Literal["template", "llm"] = "template"


class Lineage(BaseModel):
    engine_version: str
    models: dict[str, str]
    is_synthetic: bool = False
    replay_id: Optional[str] = None
    doc_id: str
    doc_hash: str


class Signal(BaseModel):
    signal_id: str
    ts_published: str
    ts_ingested: str
    ts_emitted: str
    source: SourceInfo
    entity: EntityInfo
    sentiment_score: float = Field(ge=-1.0, le=1.0)
    sentiment_confidence: float = Field(ge=0.0, le=1.0)
    event_classification: EventInfo
    impact_score: int = Field(ge=1, le=10)
    evidence: list[Evidence] = Field(default_factory=list)
    rationale: Rationale
    lineage: Lineage
    latency_ms: dict[str, float] = Field(default_factory=dict)
    flags: list[str] = Field(default_factory=list)
    text: str = ""

    def compact(self) -> dict[str, Any]:
        """The three required fields plus identifiers, for lightweight consumers."""
        return {
            "signal_id": self.signal_id,
            "ticker": self.entity.ticker,
            "ts": self.ts_published,
            "sentiment_score": self.sentiment_score,
            "event_classification": self.event_classification.label,
            "impact_score": self.impact_score,
        }
