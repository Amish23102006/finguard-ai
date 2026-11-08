from finguard.features import case_to_arrays, fraud_features
from finguard.synthetic.generator import (
    generate_credit_training,
    generate_demo_cases,
    generate_fraud_training,
)


def test_demo_cases_are_deterministic() -> None:
    a = [c.model_dump() for c in generate_demo_cases(seed=7)]
    b = [c.model_dump() for c in generate_demo_cases(seed=7)]
    assert a == b


def test_training_sets_are_deterministic() -> None:
    assert generate_credit_training(200, seed=1).equals(generate_credit_training(200, seed=1))
    assert generate_fraud_training(100, seed=1).equals(generate_fraud_training(100, seed=1))


def test_required_scenarios_exist() -> None:
    scenarios = {c.scenario.value for c in generate_demo_cases() if c.scenario}
    assert {"normal", "conflicting_evidence", "unknown_pattern", "missing_data"} <= scenarios


def test_no_personal_identifiers() -> None:
    for case in generate_demo_cases():
        assert case.synthetic is True
        assert "Synthetic" in case.entity_label
        for t in case.transactions:
            assert t.txn_id.startswith("TXN-")
            assert t.counterparty_id.startswith("CP-")
            assert t.device_id.startswith("DEV-")


def test_conflicting_case_has_injected_fraud_burst() -> None:
    case = next(c for c in generate_demo_cases() if c.case_id == "FG-DEMO-0001")
    feats = fraud_features(case_to_arrays(case))
    assert feats["velocity_ratio"] > 2
    assert feats["new_beneficiary_share_7d"] > 0.3
