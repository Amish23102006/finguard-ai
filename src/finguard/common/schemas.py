"""Shared contracts: Case input, AgentResponse output and CommitteeDecision.

Every specialist agent MUST return an ``AgentResponse``. This keeps the system auditable
and prevents free-form "agent" output from leaking into decisions.
"""

from __future__ import annotations

from datetime import date, datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

CASE_ID_PATTERN = r"^FG-[A-Z0-9]+-\d{4}$"


class RecommendedAction(StrEnum):
    """Actions the system may *recommend*. None of them is an autonomous decision."""

    NORMAL_PROCESSING = "normal_processing"
    REQUEST_MORE_INFORMATION = "request_more_information"
    HUMAN_REVIEW = "human_review"
    ENHANCED_HUMAN_VERIFICATION = "enhanced_human_verification"
    ABSTAIN = "abstain"


class Scenario(StrEnum):
    """Named, seeded demo scenarios."""

    NORMAL = "normal"
    CONFLICTING_EVIDENCE = "conflicting_evidence"
    UNKNOWN_PATTERN = "unknown_pattern"
    MISSING_DATA = "missing_data"


# --------------------------------------------------------------------------- inputs


class Transaction(BaseModel):
    """One synthetic account transaction."""

    model_config = ConfigDict(extra="forbid")

    txn_id: str
    timestamp: datetime
    amount: float = Field(ge=0)
    direction: Literal["in", "out"]
    counterparty_id: str
    is_new_beneficiary: bool = False
    channel: Literal["card", "transfer", "atm", "online"] = "card"
    device_id: str = "DEV-000"


class BorrowerProfile(BaseModel):
    """Synthetic borrower attributes. ``None`` means the value is missing."""

    model_config = ConfigDict(extra="forbid")

    annual_income: float | None = Field(default=None, ge=0)
    income_verified_at: date | None = None
    debt_to_income: float | None = Field(default=None, ge=0)
    credit_history_months: int | None = Field(default=None, ge=0)
    missed_payments_12m: int | None = Field(default=None, ge=0)
    credit_utilization: float | None = Field(default=None, ge=0)
    employment_years: float | None = Field(default=None, ge=0)
    loan_amount: float | None = Field(default=None, ge=0)


class Case(BaseModel):
    """A synthetic case submitted for assessment."""

    model_config = ConfigDict(extra="forbid")

    case_id: str = Field(pattern=CASE_ID_PATTERN)
    synthetic: Literal[True] = True
    entity_type: Literal["individual_borrower", "small_business_owner"] = "individual_borrower"
    entity_label: str = Field(default="Synthetic Borrower", max_length=80)
    scenario: Scenario | None = None
    as_of: datetime
    profile: BorrowerProfile
    transactions: list[Transaction] = Field(default_factory=list, max_length=5000)
    ground_truth: dict[str, Any] = Field(
        default_factory=dict, description="Seeded labels for evaluation only; never used by agents."
    )


# --------------------------------------------------------------------------- outputs


class FeatureContribution(BaseModel):
    """Additive contribution of one feature to a model's log-odds (linear SHAP)."""

    feature: str
    value: float | None
    contribution: float


class AgentResponse(BaseModel):
    """Common response contract for every specialist agent."""

    model_config = ConfigDict(extra="forbid")

    case_id: str
    agent_name: str
    risk_score: float = Field(ge=0, le=100)
    confidence: float = Field(ge=0, le=1)
    data_quality: float = Field(ge=0, le=1)
    ood_score: float = Field(ge=0, le=1)
    evidence: list[str]
    limitations: list[str]
    recommended_action: RecommendedAction
    model_version: str
    contributions: list[FeatureContribution] = Field(default_factory=list)
    details: dict[str, Any] = Field(default_factory=dict)


class CommitteeDecision(BaseModel):
    """Aggregated, explainable recommendation for a human reviewer (a Decision Passport)."""

    model_config = ConfigDict(extra="forbid")

    case_id: str
    engine_version: str
    policy_id: str
    policy_version: str
    decided_at: datetime
    overall_risk: float = Field(ge=0, le=100)
    risk_band: Literal["low", "medium", "medium_high", "high"]
    confidence: float = Field(ge=0, le=1)
    data_reliability: float = Field(ge=0, le=1)
    disagreement: float = Field(ge=0, le=100)
    disagreement_level: Literal["low", "medium", "high"]
    unknown_pattern_score: float = Field(ge=0, le=1)
    recommended_action: RecommendedAction
    abstained: bool
    reasons: list[str]
    limitations: list[str]
    human_checks: list[str]
    narrative: dict[str, str]
    agent_responses: list[AgentResponse]
    disclaimer: str = (
        "Educational demo on synthetic data. Not financial advice. FinGuard never approves, "
        "denies or accuses; it recommends what a human should review."
    )
