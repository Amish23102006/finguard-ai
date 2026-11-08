"""Versioned model bundle: pipeline + training statistics used for OOD and imputation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class FeatureStats:
    """Per-feature training distribution summary."""

    mean: float
    std: float
    median: float
    lo: float  # 0.5th percentile
    hi: float  # 99.5th percentile


@dataclass
class ModelBundle:
    """Everything an agent needs to score, explain and self-check one model."""

    name: str
    version: str
    features: list[str]
    pipeline: Pipeline
    stats: dict[str, FeatureStats]
    metrics: dict[str, Any] = field(default_factory=dict)
    seed: int = 42

    def impute(self, row: dict[str, float | None]) -> tuple[np.ndarray, list[str]]:
        """Fill missing features with the training median; return vector and imputed names."""
        imputed = [f for f in self.features if row.get(f) is None]
        values = [
            self.stats[f].median if row.get(f) is None else float(row[f])  # type: ignore[arg-type]
            for f in self.features
        ]
        return np.array(values, dtype=float).reshape(1, -1), imputed

    def predict_proba(self, x: np.ndarray) -> float:
        """Probability of the positive (risk) class."""
        return float(self.pipeline.predict_proba(x)[0, 1])

    def contributions(self, x: np.ndarray) -> dict[str, float]:
        """Exact SHAP values for a linear model on standardised inputs (log-odds units)."""
        scaler: StandardScaler = self.pipeline.named_steps["scale"]
        model: LogisticRegression = self.pipeline.named_steps["model"]
        z = scaler.transform(x)[0]
        return {f: float(c * v) for f, c, v in zip(self.features, model.coef_[0], z, strict=True)}

    def ood_score(self, x: np.ndarray) -> tuple[float, list[str]]:
        """Distance-from-training score in [0, 1] plus the features driving it.

        Each feature is compared with the 0.5th-99.5th percentile envelope of the training
        data; the score is twice the overshoot as a fraction of the envelope width, capped
        at 1. Percentiles (not z-scores) are used because risk features are heavily skewed.
        """
        flagged: list[str] = []
        worst = 0.0
        for f, v in zip(self.features, x[0], strict=True):
            s = self.stats[f]
            span = (s.hi - s.lo) or 1.0
            score = min(2.0 * max(s.lo - v, v - s.hi, 0.0) / span, 1.0)
            if score > 0.25:
                flagged.append(f)
            worst = max(worst, score)
        return float(worst), flagged


def make_pipeline(seed: int) -> Pipeline:
    """Simple, auditable baseline: standardise then logistic regression."""
    return Pipeline(
        [
            ("scale", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000, random_state=seed)),
        ]
    )


def compute_stats(data: np.ndarray, features: list[str]) -> dict[str, FeatureStats]:
    """Summarise the training distribution of each feature."""
    return {
        f: FeatureStats(
            mean=float(col.mean()),
            std=float(col.std()),
            median=float(np.median(col)),
            lo=float(np.percentile(col, 0.5)),
            hi=float(np.percentile(col, 99.5)),
        )
        for f, col in zip(features, data.T, strict=True)
    }
