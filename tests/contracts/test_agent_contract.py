"""Every agent must return a valid AgentResponse for every demo scenario."""

import pytest
from pydantic import ValidationError

from finguard.common.schemas import AgentResponse


def test_all_agents_return_valid_contract(service, demo_cases) -> None:
    for case in demo_cases.values():
        for agent in service.agents:
            r = agent.assess(case)
            assert isinstance(r, AgentResponse)
            AgentResponse.model_validate(r.model_dump())  # round-trips cleanly
            assert r.case_id == case.case_id
            assert r.agent_name == agent.name
            assert 0 <= r.risk_score <= 100 and 0 <= r.confidence <= 1
            assert r.evidence and r.limitations and r.model_version


def test_contract_rejects_extra_fields_and_bad_ranges() -> None:
    base = dict(
        case_id="FG-DEMO-0001",
        agent_name="x",
        risk_score=10,
        confidence=0.5,
        data_quality=1,
        ood_score=0,
        evidence=[],
        limitations=[],
        recommended_action="human_review",
        model_version="v",
    )
    AgentResponse(**base)
    with pytest.raises(ValidationError):
        AgentResponse(**{**base, "free_text_opinion": "trust me"})
    with pytest.raises(ValidationError):
        AgentResponse(**{**base, "risk_score": 140})


def test_blueprint_example_payload_is_valid() -> None:
    AgentResponse.model_validate(
        {
            "case_id": "FG-DEMO-0001",
            "agent_name": "fraud_detection",
            "risk_score": 84,
            "confidence": 0.89,
            "data_quality": 0.76,
            "ood_score": 0.22,
            "evidence": ["high transaction velocity", "new beneficiary"],
            "limitations": ["only 60 days of history"],
            "recommended_action": "human_review",
            "model_version": "fraud-xgb-1.0.0",
        }
    )
