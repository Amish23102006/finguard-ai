import pytest

from finguard.common.schemas import BorrowerProfile, RecommendedAction
from finguard.synthetic.builder import BEHAVIOURS, build_custom_case

HEALTHY = BorrowerProfile(
    annual_income=70_000,
    income_verified_at=None,
    debt_to_income=0.2,
    credit_history_months=120,
    missed_payments_12m=0,
    credit_utilization=0.2,
    employment_years=5,
    loan_amount=15_000,
)


def test_builder_is_deterministic() -> None:
    a = build_custom_case(HEALTHY, behaviour="fraud_strong", seed=3)
    b = build_custom_case(HEALTHY, behaviour="fraud_strong", seed=3)
    assert a.model_dump() == b.model_dump()
    assert a.synthetic is True and a.case_id == "FG-USER-0001"


def test_unknown_behaviour_rejected() -> None:
    with pytest.raises(ValueError):
        build_custom_case(HEALTHY, behaviour="steal")


def test_every_behaviour_can_be_assessed(service) -> None:
    for behaviour in BEHAVIOURS:
        d = service.assess(build_custom_case(HEALTHY, behaviour=behaviour))
        assert d.recommended_action in RecommendedAction


def test_strong_anomaly_escalates_vs_normal(service) -> None:
    profile = HEALTHY.model_copy(update={"income_verified_at": None})
    normal = service.assess(build_custom_case(profile, behaviour="normal"))
    strong = service.assess(build_custom_case(profile, behaviour="fraud_strong"))
    assert strong.overall_risk > normal.overall_risk
