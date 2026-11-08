"""Build a synthetic case from user-entered values (used by the dashboard's input form)."""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np

from finguard.common.schemas import BorrowerProfile, Case
from finguard.synthetic.generator import (
    DEMO_AS_OF,
    TimelineSpec,
    arrays_to_transactions,
    simulate_timeline,
)

BEHAVIOURS: dict[str, str] = {
    "normal": "Normal everyday spending",
    "burst_legit": "Legitimate spending burst (e.g. holiday, moving house)",
    "fraud_subtle": "Subtle unusual activity in the last 7 days",
    "fraud_strong": "Strong unusual activity in the last 7 days",
    "extreme": "Extreme / never-seen-before pattern",
}


def build_custom_case(
    profile: BorrowerProfile,
    *,
    behaviour: str = "normal",
    history_days: int = 180,
    daily_rate: float = 1.5,
    typical_amount: float = 60.0,
    seed: int = 7,
    case_id: str = "FG-USER-0001",
    entity_label: str = "Synthetic Custom Borrower",
    as_of: datetime = DEMO_AS_OF,
) -> Case:
    """Combine a user-entered profile with a simulated, seeded transaction timeline."""
    if behaviour not in BEHAVIOURS:
        raise ValueError(f"behaviour must be one of {sorted(BEHAVIOURS)}")
    spec = TimelineSpec(
        history_days=max(int(history_days), 1),
        daily_rate=max(float(daily_rate), 0.05),
        mean_amount=max(float(typical_amount), 1.0),
        pattern=behaviour,
    )
    arr = simulate_timeline(np.random.default_rng(seed), spec)
    return Case(
        case_id=case_id,
        entity_label=entity_label,
        as_of=as_of,
        profile=profile,
        transactions=arrays_to_transactions(arr, as_of, case_id[-4:]),
    )


def verified_date(months_ago: int | None, as_of: datetime = DEMO_AS_OF) -> datetime | None:
    """Convert 'income verified N months ago' to a datetime (None if unknown)."""
    return None if months_ago is None else as_of - timedelta(days=30 * months_ago)
