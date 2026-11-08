"""Deterministic synthetic data generator.

All identities are fictional (``FG-``, ``CP-``, ``DEV-`` codes) and every case is labelled
``synthetic=True``. No realistic personal data is ever produced.

Run ``python -m finguard.synthetic.generator --out data/sample`` to write the demo cases.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from finguard.common.schemas import BorrowerProfile, Case, Scenario, Transaction
from finguard.features import CREDIT_FEATURES, FRAUD_FEATURES, TxnArrays, fraud_features

DEFAULT_SEED = 42
REF_HOUR = 12
DEMO_AS_OF = datetime(2026, 10, 31, REF_HOUR, 0, tzinfo=UTC)


# --------------------------------------------------------------------------- timelines


@dataclass(frozen=True)
class TimelineSpec:
    """Parameters for one simulated account timeline."""

    history_days: int
    daily_rate: float
    mean_amount: float
    pattern: str = "normal"  # normal | burst_legit | fraud_subtle | fraud_strong | extreme
    n_devices: int = 1


def simulate_timeline(rng: np.random.Generator, spec: TimelineSpec) -> TxnArrays:
    """Simulate transactions as arrays relative to an ``as_of`` reference."""
    n_base = rng.poisson(spec.daily_rate * spec.history_days)
    age = rng.uniform(0, spec.history_days, n_base)
    amount = rng.lognormal(np.log(spec.mean_amount), 0.6, n_base)
    is_out = rng.random(n_base) < 0.75
    is_new = rng.random(n_base) < 0.04
    device = rng.integers(0, spec.n_devices, n_base)
    hour = rng.choice(np.arange(7, 23), n_base)

    # Monthly salary-like inflows so declared income can be cross-checked.
    salary_age = np.arange(15, spec.history_days, 30, dtype=float)
    parts = [(age, amount, is_out, is_new, device, hour)]
    parts.append(
        (
            salary_age,
            np.full(salary_age.size, spec.mean_amount * 40),
            np.zeros(salary_age.size, bool),
            np.zeros(salary_age.size, bool),
            np.zeros(salary_age.size, int),
            np.full(salary_age.size, 9),
        )
    )

    burst = {
        "normal": None,
        "burst_legit": (
            rng.uniform(1.5, 4.0),
            rng.uniform(1.5, 6.0),
            0.3,
            int(rng.random() < 0.3),
            0.1,
        ),
        "fraud_subtle": (
            rng.uniform(1.0, 2.5),
            rng.uniform(1.2, 3.5),
            0.3,
            int(rng.random() < 0.5),
            0.12,
        ),
        "fraud_strong": (rng.uniform(4.0, 9.0), rng.uniform(4.0, 12.0), 0.75, 1, 0.55),
        "extreme": (40.0, 45.0, 1.0, 12, 1.0),
    }[spec.pattern]
    if burst is not None:
        vel, amt_mult, p_new, new_devs, p_night = burst
        n_burst = max(int(rng.poisson(spec.daily_rate * 7 * vel)), 3)
        b_dev = (
            rng.integers(spec.n_devices, spec.n_devices + new_devs, n_burst)
            if new_devs
            else rng.integers(0, spec.n_devices, n_burst)
        )
        night = rng.random(n_burst) < p_night
        parts.append(
            (
                rng.uniform(0, 6.9, n_burst),
                rng.lognormal(np.log(spec.mean_amount * amt_mult), 0.4, n_burst),
                np.ones(n_burst, bool),
                rng.random(n_burst) < p_new,
                b_dev,
                np.where(night, rng.integers(0, 5, n_burst), rng.integers(7, 23, n_burst)),
            )
        )

    cols = [np.concatenate(c) for c in zip(*parts, strict=True)]
    cols[0] = _align_age_with_hour(cols[0], cols[5])
    order = np.argsort(-cols[0])  # oldest first
    return TxnArrays(*(c[order] for c in cols))


def _align_age_with_hour(age: np.ndarray, hour: np.ndarray) -> np.ndarray:
    """Make fractional ages agree with the hour of day, given an as_of at ``REF_HOUR``."""
    aligned = np.floor(age) + (REF_HOUR - hour) / 24.0
    return np.where(aligned < 0, aligned + 1.0, aligned)


def arrays_to_transactions(arr: TxnArrays, as_of: datetime, prefix: str) -> list[Transaction]:
    """Materialise timeline arrays as validated ``Transaction`` objects."""
    txns: list[Transaction] = []
    for i in range(arr.age_days.size):
        ts = as_of - timedelta(days=float(arr.age_days[i]))
        txns.append(
            Transaction(
                txn_id=f"TXN-{prefix}-{i:05d}",
                timestamp=ts,
                amount=round(float(arr.amount[i]), 2),
                direction="out" if arr.is_out[i] else "in",
                counterparty_id=f"CP-{(i * 7919) % 997:04d}",
                is_new_beneficiary=bool(arr.is_new_beneficiary[i]),
                channel="transfer" if arr.is_new_beneficiary[i] else "card",
                device_id=f"DEV-{int(arr.device[i]):03d}",
            )
        )
    return txns


# --------------------------------------------------------------------------- training sets


def generate_credit_training(n: int = 5000, seed: int = DEFAULT_SEED) -> pd.DataFrame:
    """Synthetic borrower table with a seeded default label (documented generative rule)."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "debt_to_income": rng.beta(2, 5, n) * 1.2,
            "credit_utilization": rng.beta(2, 3, n),
            "missed_payments_12m": rng.poisson(0.4, n).astype(float),
            "credit_history_months": rng.integers(6, 360, n).astype(float),
            "employment_years": rng.gamma(2.0, 3.0, n).clip(0, 40),
            "loan_to_income": rng.uniform(0.05, 1.5, n),
        }
    )
    logit = (
        -3.4
        + 3.2 * df.debt_to_income
        + 2.2 * df.credit_utilization
        + 0.7 * df.missed_payments_12m
        - 0.004 * df.credit_history_months
        - 0.05 * df.employment_years
        + 0.9 * df.loan_to_income
        + rng.normal(0, 0.5, n)
    )
    df["default"] = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    df["application_order"] = np.arange(n)
    return df[[*CREDIT_FEATURES, "application_order", "default"]]


def generate_fraud_training(n: int = 2500, seed: int = DEFAULT_SEED) -> pd.DataFrame:
    """Synthetic behavioural features with seeded fraud labels and noisy legit bursts."""
    rng = np.random.default_rng(seed + 1)
    rows = []
    for i in range(n):
        is_fraud = rng.random() < 0.08
        if is_fraud:
            pattern = "fraud_subtle" if rng.random() < 0.45 else "fraud_strong"
        else:
            pattern = "burst_legit" if rng.random() < 0.15 else "normal"
        spec = TimelineSpec(
            history_days=int(rng.integers(45, 365)),
            daily_rate=float(rng.uniform(0.5, 3.0)),
            mean_amount=float(rng.lognormal(np.log(60), 0.5)),
            pattern=pattern,
            n_devices=int(rng.integers(1, 3)),
        )
        feats = fraud_features(simulate_timeline(rng, spec))
        label = int(is_fraud) if rng.random() > 0.02 else int(not is_fraud)  # 2% label noise
        rows.append({**feats, "event_order": i, "fraud": label})
    return pd.DataFrame(rows)[[*FRAUD_FEATURES, "event_order", "fraud"]]


# --------------------------------------------------------------------------- demo cases


def _case(
    case_id: str,
    scenario: Scenario,
    profile: BorrowerProfile,
    spec: TimelineSpec,
    seed: int,
    ground_truth: dict[str, object],
    entity_type: str = "individual_borrower",
    entity_label: str = "Synthetic Borrower",
) -> Case:
    rng = np.random.default_rng(seed)
    arr = simulate_timeline(rng, spec)
    return Case(
        case_id=case_id,
        entity_type=entity_type,
        entity_label=entity_label,
        scenario=scenario,
        as_of=DEMO_AS_OF,
        profile=profile,
        transactions=arrays_to_transactions(arr, DEMO_AS_OF, case_id[-4:]),
        ground_truth=ground_truth,
    )


def generate_demo_cases(seed: int = DEFAULT_SEED) -> list[Case]:
    """The named demo scenarios used by tests, the API and the dashboard."""
    stale = (DEMO_AS_OF - timedelta(days=456)).date()  # ~15 months old
    fresh = (DEMO_AS_OF - timedelta(days=40)).date()
    return [
        _case(
            "FG-DEMO-0001",
            Scenario.CONFLICTING_EVIDENCE,
            BorrowerProfile(
                annual_income=86_000,
                income_verified_at=stale,
                debt_to_income=0.21,
                credit_history_months=124,
                missed_payments_12m=0,
                credit_utilization=0.18,
                employment_years=8,
                loan_amount=20_000,
            ),
            TimelineSpec(180, 1.6, 60, "fraud_strong", 1),
            seed + 1,
            {
                "expected_action": "enhanced_human_verification",
                "injected": ["fraud_burst", "stale_income"],
            },
            entity_type="small_business_owner",
            entity_label="Synthetic Small Business Owner",
        ),
        _case(
            "FG-DEMO-0002",
            Scenario.NORMAL,
            BorrowerProfile(
                annual_income=72_000,
                income_verified_at=fresh,
                debt_to_income=0.18,
                credit_history_months=150,
                missed_payments_12m=0,
                credit_utilization=0.22,
                employment_years=6,
                loan_amount=15_000,
            ),
            TimelineSpec(365, 1.4, 70, "normal", 1),
            seed + 2,
            {"expected_action": "normal_processing", "injected": []},
        ),
        _case(
            "FG-DEMO-0003",
            Scenario.UNKNOWN_PATTERN,
            BorrowerProfile(
                annual_income=64_000,
                income_verified_at=fresh,
                debt_to_income=3.4,
                credit_history_months=700,
                missed_payments_12m=11,
                credit_utilization=2.5,
                employment_years=48,
                loan_amount=600_000,
            ),
            TimelineSpec(200, 1.0, 50, "extreme", 1),
            seed + 3,
            {"expected_action": "abstain", "injected": ["out_of_distribution"]},
        ),
        _case(
            "FG-DEMO-0004",
            Scenario.MISSING_DATA,
            BorrowerProfile(
                annual_income=None,
                income_verified_at=None,
                debt_to_income=None,
                credit_history_months=None,
                missed_payments_12m=0,
                credit_utilization=0.3,
                employment_years=2,
                loan_amount=10_000,
            ),
            TimelineSpec(25, 1.2, 55, "normal", 1),
            seed + 4,
            {"expected_action": "abstain", "injected": ["missing_income", "short_history"]},
        ),
    ]


def get_demo_case(case_id: str, seed: int = DEFAULT_SEED) -> Case | None:
    """Return one demo case by ID, or ``None``."""
    return next((c for c in generate_demo_cases(seed) if c.case_id == case_id), None)


def main() -> None:
    """CLI: write demo cases as JSON files."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data/sample"))
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for case in generate_demo_cases(args.seed):
        path = args.out / f"{case.case_id}.json"
        path.write_text(json.dumps(case.model_dump(mode="json"), indent=2))
        print(f"wrote {path} ({len(case.transactions)} transactions)")
    print(f"generated on {date.today().isoformat()} with seed={args.seed}")


if __name__ == "__main__":
    main()
