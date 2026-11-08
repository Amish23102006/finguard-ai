from datetime import timedelta

from finguard.agents.data_reliability import DataReliabilityAgent
from finguard.common.schemas import RecommendedAction


def test_clean_case_passes(policy, demo_cases) -> None:
    r = DataReliabilityAgent(policy).assess(demo_cases["FG-DEMO-0002"])
    assert r.data_quality == 1.0
    assert r.details["automation_permitted"] is True


def test_missing_income_and_short_history_abstains(policy, demo_cases) -> None:
    r = DataReliabilityAgent(policy).assess(demo_cases["FG-DEMO-0004"])
    assert r.recommended_action == RecommendedAction.ABSTAIN
    assert "annual_income" in r.details["critical_missing"]
    assert r.details["history_days"] < policy.data_reliability.min_history_days


def test_stale_income_lowers_quality(policy, demo_cases) -> None:
    r = DataReliabilityAgent(policy).assess(demo_cases["FG-DEMO-0001"])
    assert "annual_income" in r.details["stale_fields"]
    assert r.data_quality < policy.data_reliability.review_below
    assert any("months old" in e for e in r.evidence)


def test_future_and_duplicate_transactions_detected(policy, demo_cases) -> None:
    case = demo_cases["FG-DEMO-0002"].model_copy(deep=True)
    future = case.transactions[0].model_copy(update={"timestamp": case.as_of + timedelta(days=3)})
    case.transactions.append(future)  # duplicated id AND future-dated
    r = DataReliabilityAgent(policy).assess(case)
    assert any("after the assessment time" in e for e in r.evidence)
    assert any("duplicated" in e for e in r.evidence)
