"""Acceptance criteria from the blueprint, end to end through the service."""

from finguard.common.schemas import RecommendedAction


def test_conflicting_case_escalates(service, demo_cases) -> None:
    d = service.assess(demo_cases["FG-DEMO-0001"])
    agents = {r.agent_name: r for r in d.agent_responses}
    assert agents["credit_risk"].risk_score < agents["fraud_detection"].risk_score
    assert agents["fraud_detection"].evidence and agents["fraud_detection"].limitations
    assert d.disagreement_level == "high"
    assert d.recommended_action == RecommendedAction.ENHANCED_HUMAN_VERIFICATION
    assert any("income" in c.lower() for c in d.human_checks)


def test_normal_case_is_low_risk(service, demo_cases) -> None:
    d = service.assess(demo_cases["FG-DEMO-0002"])
    assert d.recommended_action == RecommendedAction.NORMAL_PROCESSING
    assert d.confidence > 0.8


def test_unknown_pattern_triggers_abstention(service, demo_cases) -> None:
    d = service.assess(demo_cases["FG-DEMO-0003"])
    assert d.abstained and d.unknown_pattern_score >= 0.6
    assert d.confidence < 0.3


def test_missing_data_abstains(service, demo_cases) -> None:
    d = service.assess(demo_cases["FG-DEMO-0004"])
    assert d.abstained and d.data_reliability < 0.5


def test_ground_truth_expectations_match(service, demo_cases) -> None:
    for case in demo_cases.values():
        assert service.assess(case).recommended_action.value == case.ground_truth["expected_action"]


def test_replay_is_reproducible(service, demo_cases) -> None:
    a = service.assess(demo_cases["FG-DEMO-0001"]).model_dump(exclude={"decided_at"})
    b = service.assess(demo_cases["FG-DEMO-0001"]).model_dump(exclude={"decided_at"})
    assert a == b


def test_failing_agent_degrades_gracefully(service, demo_cases, monkeypatch) -> None:
    fraud = next(a for a in service.agents if a.name == "fraud_detection")

    def boom(case):
        raise RuntimeError("model artefact missing")

    monkeypatch.setattr(fraud, "assess", boom)
    d = service.assess(demo_cases["FG-DEMO-0002"])
    assert d.recommended_action == RecommendedAction.HUMAN_REVIEW
