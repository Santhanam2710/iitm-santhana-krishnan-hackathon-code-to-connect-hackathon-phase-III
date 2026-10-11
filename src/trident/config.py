"""Configuration loading.

All YAML files live in ``configs/``. Any settings value can be overridden with an
environment variable named ``TRIDENT_<SECTION>_<KEY>`` (upper case), which is how
Docker and CI switch backends without editing files.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(os.environ.get("TRIDENT_ROOT", Path(__file__).resolve().parents[2]))
CONFIG_DIR = ROOT / "configs"
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
DOCS_DIR = ROOT / "docs"

try:  # optional .env support
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except Exception:  # pragma: no cover
    pass


def _load(name: str) -> dict[str, Any]:
    with open(CONFIG_DIR / name, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _coerce(old: Any, raw: str) -> Any:
    if isinstance(old, bool):
        return raw.lower() in {"1", "true", "yes", "on"}
    if isinstance(old, int) and not isinstance(old, bool):
        try:
            return int(raw)
        except ValueError:
            return raw
    if isinstance(old, float):
        try:
            return float(raw)
        except ValueError:
            return raw
    return raw


def _apply_env(settings: dict[str, Any]) -> dict[str, Any]:
    for section, values in settings.items():
        if not isinstance(values, dict):
            continue
        for key, old in list(values.items()):
            env = f"TRIDENT_{section}_{key}".upper()
            if env in os.environ:
                values[key] = _coerce(old, os.environ[env])
    return settings


@dataclass(frozen=True)
class Entity:
    ticker: str
    name: str
    sector: str
    aliases: tuple[str, ...]
    ambiguous: tuple[str, ...] = ()
    in_index: bool = False
    cap_weight: float = 0.0
    vol: float = 0.3


@dataclass
class Config:
    settings: dict[str, Any]
    universe: dict[str, Any]
    taxonomy: dict[str, Any]
    scenarios: dict[str, Any]
    entities: list[Entity] = field(default_factory=list)

    @property
    def index(self) -> list[Entity]:
        return [e for e in self.entities if e.in_index]

    def entity(self, ticker: str) -> Entity | None:
        for e in self.entities:
            if e.ticker == ticker:
                return e
        return None

    def section(self, name: str) -> dict[str, Any]:
        return self.settings.get(name, {})


def _build_entities(universe: dict[str, Any]) -> list[Entity]:
    out: list[Entity] = []
    for group, in_index in (("index", True), ("watchlist", False)):
        for row in universe.get(group, []):
            out.append(
                Entity(
                    ticker=row["ticker"],
                    name=row["name"],
                    sector=row.get("sector", "Other"),
                    aliases=tuple(row.get("aliases", [])),
                    ambiguous=tuple(row.get("ambiguous", [])),
                    in_index=in_index,
                    cap_weight=float(row.get("cap_weight", 0.0)),
                    vol=float(row.get("vol", 0.3)),
                )
            )
    return out


@lru_cache(maxsize=1)
def get_config() -> Config:
    settings = _apply_env(_load("settings.yaml"))
    universe = _load("universe.yaml")
    cfg = Config(
        settings=settings,
        universe=universe,
        taxonomy=_load("taxonomy.yaml"),
        scenarios=_load("scenarios.yaml"),
        entities=_build_entities(universe),
    )
    return cfg


def resolve(path: str | Path) -> Path:
    """Resolve a repository-relative path."""
    p = Path(path)
    return p if p.is_absolute() else ROOT / p
