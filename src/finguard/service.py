"""Assessment service: runs agents, isolates failures, calls the committee."""

from __future__ import annotations

import logging
import time
from pathlib import Path

from finguard.agents.base import Agent
from finguard.agents.credit import CreditRiskAgent
from finguard.agents.data_reliability import DataReliabilityAgent
from finguard.agents.fraud import FraudDetectionAgent
from finguard.committee.engine import decide
from finguard.committee.policy import Policy, load_policy
from finguard.common.schemas import AgentResponse, Case, CommitteeDecision
from finguard.config import Settings, get_settings
from finguard.models.registry import get_bundles
from finguard.models.train import CREDIT_MODEL, FRAUD_MODEL

logger = logging.getLogger(__name__)


class AssessmentService:
    """Facade used by the API, the dashboard and tests."""

    def __init__(
        self,
        settings: Settings | None = None,
        policy: Policy | None = None,
        model_dir: Path | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.policy = policy or load_policy(self.settings.resolve(self.settings.policy_path))
        bundles = get_bundles(
            model_dir or self.settings.resolve(self.settings.model_dir), self.settings.random_seed
        )
        self.agents: list[Agent] = [
            CreditRiskAgent(self.policy, bundles[CREDIT_MODEL]),
            FraudDetectionAgent(self.policy, bundles[FRAUD_MODEL]),
            DataReliabilityAgent(self.policy),
        ]

    def assess(self, case: Case) -> CommitteeDecision:
        """Run every agent; a failing agent degrades the result instead of crashing it."""
        start = time.perf_counter()
        responses: list[AgentResponse] = []
        failed: list[str] = []
        for agent in self.agents:
            try:
                responses.append(agent.assess(case))
            except Exception:  # noqa: BLE001 - isolate any agent failure
                logger.exception(
                    "agent failed", extra={"case_id": case.case_id, "agent": agent.name}
                )
                failed.append(agent.name)
        decision = decide(case.case_id, responses, self.policy, failed)
        logger.info(
            "case assessed",
            extra={
                "case_id": case.case_id,
                "action": decision.recommended_action.value,
                "latency_ms": round((time.perf_counter() - start) * 1000, 1),
            },
        )
        return decision
