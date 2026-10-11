"""Impact score (1-10), eq. E3, with an ordinal (proportional-odds) model for calibration.

Default mode uses the illustrative coefficients below. After running
``python -m trident.training.train_impact`` the fitted ordinal model in
``models/impact_model.json`` is used instead: it maps the same features to the decile
of realized |CAR[0,+1]|, so the score has an empirical meaning.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from trident.config import MODELS_DIR, get_config

FEATURES = ("base_severity", "abs_sentiment", "novelty", "log_velocity", "reliability", "magnitude", "breadth")
DEFAULT_BETA = {
    "intercept": -3.8,
    "base_severity": 0.42,
    "abs_sentiment": 1.1,
    "novelty": 0.4,
    "log_velocity": 0.35,
    "reliability": 0.8,
    "magnitude": 0.35,
    "breadth": 0.6,
}
PCT = re.compile(r"(\d+(?:\.\d+)?)\s?(?:%|percent|per cent)", re.I)
MONEY = re.compile(r"\$\s?(\d+(?:\.\d+)?)\s?(billion|bn|trillion|tn|million|m)\b", re.I)
CUES = re.compile(r"\b(record|sweeping|massive|collapse[sd]?|plunge[sd]?|unprecedented|emergency|historic|largest|worst|biggest)\b", re.I)


@dataclass
class ImpactFeatures:
    base_severity: float
    abs_sentiment: float
    novelty: float
    log_velocity: float
    reliability: float
    magnitude: float
    breadth: float

    def vector(self) -> np.ndarray:
        return np.array([getattr(self, f) for f in FEATURES], dtype=float)


def magnitude_cues(text: str) -> float:
    m = 0.0
    pct = [float(x) for x in PCT.findall(text)]
    if pct:
        m += min(max(pct) / 5.0, 2.0)
    money = 0.0
    for amount, unit in MONEY.findall(text):
        u = unit.lower()
        if u in ("billion", "bn"):
            money = max(money, min(0.4 + float(amount) / 50.0, 1.2))  # the largest amount counts, not the number of amounts
        elif u in ("trillion", "tn"):
            money = 1.5
    m += money
    m += min(len(CUES.findall(text)) * 0.5, 1.5)
    return round(m, 3)


def build_features(text: str, label: str, abs_sentiment: float, novelty: float, velocity_z: float,
                   reliability: float, market_wide: bool, n_entities: int) -> ImpactFeatures:
    base = float(get_config().taxonomy["classes"].get(label, {}).get("base_severity", 2.0))
    breadth = (1.0 if market_wide else 0.0) + (0.3 if n_entities > 1 else 0.0)
    return ImpactFeatures(base, abs_sentiment, novelty, math.log1p(max(velocity_z, 0.0)), reliability,
                          magnitude_cues(text), breadth)


class OrdinalLogit:
    """Proportional-odds model: P(y <= k) = sigmoid(theta_k - x.beta), k = 1..K-1."""

    def __init__(self, n_classes: int = 10, l2: float = 1e-2):
        self.K = n_classes
        self.l2 = l2
        self.beta: np.ndarray | None = None
        self.theta: np.ndarray | None = None
        self.mu: np.ndarray | None = None
        self.sd: np.ndarray | None = None

    @staticmethod
    def _sig(z):
        return 1.0 / (1.0 + np.exp(-z))

    def _unpack(self, w, d):
        beta = w[:d]
        raw = w[d:]
        theta = np.cumsum(np.concatenate([[raw[0]], np.exp(raw[1:])]))
        return beta, theta

    def fit(self, X: np.ndarray, y: np.ndarray) -> "OrdinalLogit":
        X = np.asarray(X, float)
        y = np.asarray(y, int)
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-9
        Z = (X - self.mu) / self.sd
        n, d = Z.shape
        w0 = np.concatenate([np.zeros(d), [-2.0], np.full(self.K - 2, math.log(0.5))])

        def nll(w):
            beta, theta = self._unpack(w, d)
            eta = Z @ beta
            cum = self._sig(theta[None, :] - eta[:, None])  # n x K-1
            cum = np.hstack([np.zeros((n, 1)), cum, np.ones((n, 1))])
            p = cum[np.arange(n), y] - cum[np.arange(n), y - 1]
            return -np.log(np.clip(p, 1e-12, 1)).sum() + self.l2 * (beta**2).sum()

        res = minimize(nll, w0, method="L-BFGS-B")
        self.beta, self.theta = self._unpack(res.x, d)
        return self

    def proba(self, X: np.ndarray) -> np.ndarray:
        Z = (np.atleast_2d(X) - self.mu) / self.sd
        eta = Z @ self.beta
        cum = self._sig(self.theta[None, :] - eta[:, None])
        cum = np.hstack([np.zeros((len(Z), 1)), cum, np.ones((len(Z), 1))])
        return np.diff(cum, axis=1)

    def predict_score(self, X: np.ndarray) -> np.ndarray:
        p = self.proba(X)
        return p @ np.arange(1, self.K + 1)

    def to_json(self) -> dict:
        return {"K": self.K, "beta": self.beta.tolist(), "theta": self.theta.tolist(),
                "mu": self.mu.tolist(), "sd": self.sd.tolist(), "features": list(FEATURES)}

    @classmethod
    def from_json(cls, d: dict) -> "OrdinalLogit":
        m = cls(d["K"])
        m.beta, m.theta = np.array(d["beta"]), np.array(d["theta"])
        m.mu, m.sd = np.array(d["mu"]), np.array(d["sd"])
        return m


class ImpactScorer:
    def __init__(self, model_path: Path | None = None):
        path = model_path or (MODELS_DIR / "impact_model.json")
        self.model: OrdinalLogit | None = None
        if Path(path).exists():
            self.model = OrdinalLogit.from_json(json.loads(Path(path).read_text()))
        self.name = "impact-ordinal-calibrated" if self.model else "impact-default-coefficients"

    def raw(self, f: ImpactFeatures) -> float:
        if self.model is not None:
            return float(self.model.predict_score(f.vector()[None, :])[0])
        z = DEFAULT_BETA["intercept"] + sum(DEFAULT_BETA[k] * v for k, v in asdict(f).items())
        return 1 + 9 / (1 + math.exp(-z))

    def score(self, f: ImpactFeatures) -> tuple[int, float]:
        r = self.raw(f)
        return int(min(10, max(1, round(r)))), r


@lru_cache(maxsize=1)
def default_impact() -> ImpactScorer:
    return ImpactScorer()
