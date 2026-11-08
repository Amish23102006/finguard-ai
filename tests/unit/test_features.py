from datetime import UTC, datetime, timedelta

from finguard.common.schemas import BorrowerProfile, Case, Transaction
from finguard.features import case_to_arrays, credit_features, fraud_features, history_days

AS_OF = datetime(2026, 1, 31, 12, tzinfo=UTC)


def _txn(
    i: int,
    days_ago: float,
    amount: float,
    new: bool = False,
    device: str = "DEV-000",
    direction: str = "out",
) -> Transaction:
    return Transaction(
        txn_id=f"TXN-T-{i:05d}",
        timestamp=AS_OF - timedelta(days=days_ago),
        amount=amount,
        direction=direction,
        counterparty_id="CP-0001",  # type: ignore[arg-type]
        is_new_beneficiary=new,
        device_id=device,
    )


def _case(txns: list[Transaction]) -> Case:
    return Case(case_id="FG-TEST-0001", as_of=AS_OF, profile=BorrowerProfile(), transactions=txns)


def test_loan_to_income_and_missing_values() -> None:
    feats = credit_features(BorrowerProfile(annual_income=50_000, loan_amount=25_000))
    assert feats["loan_to_income"] == 0.5
    assert feats["debt_to_income"] is None


def test_hand_checkable_fraud_features() -> None:
    # Baseline: 1 txn/day of 100 for days 10..69 (60 txns over 63 baseline days).
    txns = [_txn(i, 10 + i, 100) for i in range(60)]
    # Recent: 14 txns in 7 days, max 1000, half to new beneficiaries, one new device.
    txns += [
        _txn(
            100 + i,
            0.5 + i * 0.4,
            1000 if i == 0 else 100,
            new=i % 2 == 0,
            device="DEV-009" if i == 0 else "DEV-000",
        )
        for i in range(14)
    ]
    arr = case_to_arrays(_case(txns))
    feats = fraud_features(arr)
    assert history_days(arr) == 69
    assert round(feats["velocity_ratio"], 2) == round((14 / 7) / (60 / 62), 2)
    assert feats["amount_spike_ratio"] == 10.0
    assert feats["new_beneficiary_share_7d"] == 0.5
    assert feats["new_devices_7d"] == 1.0


def test_future_transactions_are_excluded() -> None:
    txns = [_txn(i, 10 + i, 100) for i in range(30)] + [_txn(99, -2, 99_999)]
    feats = fraud_features(case_to_arrays(_case(txns)))
    assert feats["amount_spike_ratio"] == 1.0  # the future txn is ignored
