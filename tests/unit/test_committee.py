import pytest

from finguard.committee.engine import decide
from finguard.committee.policy import Policy
from finguard.common.schemas import AgentResponse, RecommendedAction


def _resp(
    name: str, risk: float, *, quality: float = 1.0, ood: float = 0.0, details: dict | None = None
) -> AgentResponse:
    return AgentResponse(
        case_id="FG-TEST-0001",
        agent_name=name,
        risk_score=risk,
        confidence=0.9,
        data_quality=quality,
        ood_score=ood,
        evidence=[f"{name} evidence"],
        limitations=[f"{name} limitation"],
        recommended_action=RecommendedAction.NORMAL_PROCESSING,
        model_version="test-1.0.0",
        details=details or {},
    )


def _dr(quality: float = 1.0, critical_missing: list[str] | None = None) -> AgentResponse:
    return _resp(
        "data_reliability",
        100 * (1 - quality),
        quality=quality,
        details={
            "critical_missing": critical_missing or [],
            "stale_fields": [],
            "history_days": 365,
        },
    )


def test_all_agents_agree_low_risk_still_shows_evidence(policy: Policy) -> None:
    d = decide(
        "FG-TEST-0001", [_resp("credit_risk", 5), _resp("fraud_detection", 8), _dr()], policy
    )
    assert d.recommended_action == RecommendedAction.NORMAL_PROCESSING
    assert d.risk_band == "low"
    assert d.limitations and all(r.evidence for r in d.agent_responses)


def test_fraud_high_credit_low_is_high_disagreement(policy: Policy) -> None:
    d = decide(
        "FG-TEST-0001", [_resp("credit_risk", 5), _resp("fraud_detection", 90), _dr()], policy
    )
    assert d.disagreement_level == "high"
    assert d.recommended_action == RecommendedAction.ENHANCED_HUMAN_VERIFICATION
    # No silent averaging: the strong signal is not diluted to the weighted mean.
    assert d.overall_risk >= policy.aggregation.max_signal_floor * 90


def test_disagreement_reduces_confidence(policy: Policy) -> None:
    agree = decide(
        "FG-TEST-0001", [_resp("credit_risk", 50), _resp("fraud_detection", 50), _dr()], policy
    )
    clash = decide(
        "FG-TEST-0001", [_resp("credit_risk", 5), _resp("fraud_detection", 95), _dr()], policy
    )
    assert clash.confidence < agree.confidence


def test_missing_critical_data_abstains(policy: Policy) -> None:
    d = decide(
        "FG-TEST-0001",
        [_resp("credit_risk", 10), _resp("fraud_detection", 10), _dr(0.1, ["annual_income"])],
        policy,
    )
    assert d.abstained and d.recommended_action == RecommendedAction.ABSTAIN
    assert any("Collect missing information" in c for c in d.human_checks)


def test_ood_case_abstains_with_reduced_confidence(policy: Policy) -> None:
    normal = decide(
        "FG-TEST-0001", [_resp("credit_risk", 60), _resp("fraud_detection", 60), _dr()], policy
    )
    ood = decide(
        "FG-TEST-0001",
        [_resp("credit_risk", 60, ood=0.9), _resp("fraud_detection", 60), _dr()],
        policy,
    )
    assert ood.recommended_action == RecommendedAction.ABSTAIN
    assert ood.confidence < normal.confidence


def test_failed_agent_degrades_to_human_review(policy: Policy) -> None:
    d = decide(
        "FG-TEST-0001", [_resp("credit_risk", 5), _dr()], policy, failed_agents=["fraud_detection"]
    )
    assert d.recommended_action == RecommendedAction.HUMAN_REVIEW
    assert any("unavailable" in lim for lim in d.limitations)


def test_policy_version_change_changes_outcome(policy: Policy) -> None:
    responses = [_resp("credit_risk", 30), _resp("fraud_detection", 60), _dr()]
    before = decide("FG-TEST-0001", responses, policy)
    stricter = policy.model_copy(
        update={
            "version": "1.1.0",
            "disagreement": policy.disagreement.model_copy(update={"high": 25}),
        }
    )
    after = decide("FG-TEST-0001", responses, stricter)
    assert before.policy_version == "1.0.0" and after.policy_version == "1.1.0"
    assert before.recommended_action != after.recommended_action
    assert after.recommended_action == RecommendedAction.ENHANCED_HUMAN_VERIFICATION


def test_invalid_policy_is_rejected(policy: Policy) -> None:
    data = policy.model_dump()
    data["risk_bands"] = {"medium": 60, "medium_high": 50, "high": 75}
    with pytest.raises(ValueError):
        Policy.model_validate(data)


def test_narrative_uses_safe_wording(policy: Policy) -> None:
    d = decide(
        "FG-TEST-0001", [_resp("credit_risk", 5), _resp("fraud_detection", 95), _dr()], policy
    )
    text = " ".join(d.narrative.values()).lower()
    assert set(d.narrative) == {"what_happened", "why", "how_certain", "what_next"}
    for banned in ("committed fraud", "fraudster", "denied", "approved", "guaranteed"):
        assert banned not in text
