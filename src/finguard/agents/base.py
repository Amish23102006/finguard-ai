"""Agent interface. Agents calculate; they never make the final decision."""

from __future__ import annotations

from abc import ABC, abstractmethod

from finguard.committee.policy import Policy
from finguard.common.schemas import AgentResponse, Case, RecommendedAction


class Agent(ABC):
    """A specialist that returns a structured ``AgentResponse``."""

    name: str

    def __init__(self, policy: Policy) -> None:
        self.policy = policy

    @abstractmethod
    def assess(self, case: Case) -> AgentResponse:
        """Assess one case."""

    def action_for(self, risk: float, ood: float) -> RecommendedAction:
        """Agent-level recommendation from its own score (the committee has the final say)."""
        if ood >= self.policy.ood.abstain_above:
            return RecommendedAction.ABSTAIN
        if risk >= self.policy.agent_actions.human_review:
            return RecommendedAction.HUMAN_REVIEW
        if risk >= self.policy.agent_actions.request_more_information:
            return RecommendedAction.REQUEST_MORE_INFORMATION
        return RecommendedAction.NORMAL_PROCESSING


def clip01(x: float) -> float:
    """Clamp to [0, 1]."""
    return float(min(max(x, 0.0), 1.0))


def model_confidence(ood: float, data_quality: float) -> float:
    """Confidence shrinks with distance from training data and with missing inputs."""
    return round(clip01(0.95 * (1 - 0.7 * ood) * (0.4 + 0.6 * data_quality)), 3)
