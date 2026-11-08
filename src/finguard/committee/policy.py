"""Versioned, validated committee policy loaded from YAML."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Aggregation(_Strict):
    max_signal_floor: float = Field(ge=0, le=1)


class RiskBands(_Strict):
    medium: float
    medium_high: float
    high: float

    @model_validator(mode="after")
    def _ordered(self) -> RiskBands:
        if not self.medium < self.medium_high < self.high:
            raise ValueError("risk bands must be strictly increasing")
        return self


class Disagreement(_Strict):
    medium: float
    high: float


class Ood(_Strict):
    abstain_above: float = Field(ge=0, le=1)


class Penalties(_Strict):
    critical_missing: float
    optional_missing: float
    stale_income: float
    short_history: float
    future_transactions: float
    duplicate_transactions: float
    income_conflict: float


class DataReliabilityPolicy(_Strict):
    abstain_below: float = Field(ge=0, le=1)
    review_below: float = Field(ge=0, le=1)
    critical_fields: list[str]
    income_max_age_days: int
    min_history_days: int
    income_conflict_ratio: float = Field(gt=1)
    penalties: Penalties


class ConfidencePolicy(_Strict):
    disagreement_penalty: float = Field(ge=0, le=1)
    ood_penalty: float = Field(ge=0, le=1)


class AgentActions(_Strict):
    request_more_information: float
    human_review: float


class Policy(_Strict):
    """Root policy document."""

    policy_id: str
    version: str
    effective_from: str
    description: str = ""
    weights: dict[str, float]
    aggregation: Aggregation
    risk_bands: RiskBands
    disagreement: Disagreement
    ood: Ood
    data_reliability: DataReliabilityPolicy
    confidence: ConfidencePolicy
    agent_actions: AgentActions

    @model_validator(mode="after")
    def _weights_positive(self) -> Policy:
        if not self.weights or any(w <= 0 for w in self.weights.values()):
            raise ValueError("weights must be non-empty and positive")
        return self


def load_policy(path: Path) -> Policy:
    """Parse and validate a policy file."""
    with path.open(encoding="utf-8") as fh:
        return Policy.model_validate(yaml.safe_load(fh))


@lru_cache(maxsize=8)
def load_policy_cached(path: str) -> Policy:
    """Cached loader keyed by path string."""
    return load_policy(Path(path))
