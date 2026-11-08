"""Feature engineering shared by training and scoring (one code path avoids train/serve skew)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np

from finguard.common.schemas import BorrowerProfile, Case

CREDIT_FEATURES: list[str] = [
    "debt_to_income",
    "credit_utilization",
    "missed_payments_12m",
    "credit_history_months",
    "employment_years",
    "loan_to_income",
]

FRAUD_FEATURES: list[str] = [
    "velocity_ratio",
    "amount_spike_ratio",
    "new_beneficiary_share_7d",
    "new_devices_7d",
    "night_share_7d",
]

RECENT_WINDOW_DAYS = 7
_FEATURE_CAP = 50.0


def credit_features(profile: BorrowerProfile) -> dict[str, float | None]:
    """Map a borrower profile to credit features; missing inputs stay ``None``."""
    lti: float | None = None
    if profile.loan_amount is not None and profile.annual_income:
        lti = profile.loan_amount / profile.annual_income
    return {
        "debt_to_income": profile.debt_to_income,
        "credit_utilization": profile.credit_utilization,
        "missed_payments_12m": (
            float(profile.missed_payments_12m) if profile.missed_payments_12m is not None else None
        ),
        "credit_history_months": (
            float(profile.credit_history_months)
            if profile.credit_history_months is not None
            else None
        ),
        "employment_years": profile.employment_years,
        "loan_to_income": lti,
    }


@dataclass(frozen=True)
class TxnArrays:
    """Column-oriented transactions relative to an ``as_of`` reference time."""

    age_days: np.ndarray  # days before as_of (>= 0 for valid history)
    amount: np.ndarray
    is_out: np.ndarray
    is_new_beneficiary: np.ndarray
    device: np.ndarray  # integer-coded device ids
    hour: np.ndarray


def case_to_arrays(case: Case) -> TxnArrays:
    """Convert a case's transactions to arrays relative to ``case.as_of``."""
    txns = case.transactions
    devices = {d: i for i, d in enumerate(sorted({t.device_id for t in txns}))}
    as_of: datetime = case.as_of
    return TxnArrays(
        age_days=np.array([(as_of - t.timestamp).total_seconds() / 86400 for t in txns], float),
        amount=np.array([t.amount for t in txns], float),
        is_out=np.array([t.direction == "out" for t in txns], bool),
        is_new_beneficiary=np.array([t.is_new_beneficiary for t in txns], bool),
        device=np.array([devices[t.device_id] for t in txns], int),
        hour=np.array([t.timestamp.hour for t in txns], int),
    )


def history_days(arr: TxnArrays) -> float:
    """Length of the observed (non-future) transaction history in days."""
    valid = arr.age_days[arr.age_days >= 0]
    return float(valid.max()) if valid.size else 0.0


def fraud_features(arr: TxnArrays) -> dict[str, float]:
    """Behavioural features comparing the last 7 days with the earlier baseline.

    Transactions dated after ``as_of`` (negative age) are excluded to prevent leakage.
    """
    valid = arr.age_days >= 0
    recent = valid & (arr.age_days < RECENT_WINDOW_DAYS)
    base = valid & (arr.age_days >= RECENT_WINDOW_DAYS)
    base_days = max(history_days(arr) - RECENT_WINDOW_DAYS, 1.0)

    recent_rate = float(recent.sum()) / RECENT_WINDOW_DAYS
    base_rate = float(base.sum()) / base_days
    velocity = recent_rate / max(base_rate, 0.1) if base.any() else 1.0

    recent_out = recent & arr.is_out
    base_out = base & arr.is_out
    if recent_out.any() and base_out.any():
        spike = arr.amount[recent_out].max() / max(float(np.median(arr.amount[base_out])), 1.0)
    else:
        spike = 1.0

    new_benef = float(arr.is_new_beneficiary[recent_out].mean()) if recent_out.any() else 0.0
    base_devices = set(arr.device[base].tolist())
    new_devices = len(set(arr.device[recent].tolist()) - base_devices) if base.any() else 0
    night = (
        float(((arr.hour[recent] >= 0) & (arr.hour[recent] < 5)).mean()) if recent.any() else 0.0
    )

    return {
        "velocity_ratio": float(min(velocity, _FEATURE_CAP)),
        "amount_spike_ratio": float(min(spike, _FEATURE_CAP)),
        "new_beneficiary_share_7d": new_benef,
        "new_devices_7d": float(new_devices),
        "night_share_7d": night,
    }


def observed_annual_inflow(arr: TxnArrays) -> float | None:
    """Annualised inflows seen in the transaction history (``None`` if too little history)."""
    days = history_days(arr)
    if days < 30:
        return None
    inflow = arr.amount[(arr.age_days >= 0) & ~arr.is_out].sum()
    return float(inflow / days * 365)
