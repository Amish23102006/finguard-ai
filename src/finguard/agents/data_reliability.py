"""Data Reliability Agent: completeness, freshness, conflicts and leakage checks (rule-based)."""

from __future__ import annotations

from collections import Counter

from finguard.agents.base import Agent, clip01
from finguard.common.schemas import AgentResponse, Case, RecommendedAction
from finguard.features import case_to_arrays, history_days, observed_annual_inflow

RULES_VERSION = "data-reliability-rules-1.0.0"


class DataReliabilityAgent(Agent):
    """Decides whether the evidence is good enough for any automated recommendation."""

    name = "data_reliability"

    def assess(self, case: Case) -> AgentResponse:
        """Run deterministic data-quality checks."""
        pol = self.policy.data_reliability
        pen = pol.penalties
        penalty = 0.0
        evidence: list[str] = []
        limitations: list[str] = []
        profile = case.profile.model_dump()

        critical_missing = [f for f in pol.critical_fields if profile.get(f) is None]
        optional_missing = [
            f
            for f, v in profile.items()
            if v is None and f not in pol.critical_fields and f != "income_verified_at"
        ]
        if critical_missing:
            penalty += pen.critical_missing * len(critical_missing)
            evidence.append(f"Critical fields missing: {', '.join(critical_missing)}")
        if optional_missing:
            penalty += pen.optional_missing * len(optional_missing)
            evidence.append(f"Optional fields missing: {', '.join(optional_missing)}")

        stale_fields: list[str] = []
        income_age_days: int | None = None
        if case.profile.income_verified_at is not None:
            income_age_days = (case.as_of.date() - case.profile.income_verified_at).days
            if income_age_days > pol.income_max_age_days:
                penalty += pen.stale_income
                stale_fields.append("annual_income")
                evidence.append(
                    f"Income evidence is {income_age_days // 30} months old "
                    f"(policy maximum {pol.income_max_age_days // 30} months)"
                )
        elif case.profile.annual_income is not None:
            penalty += pen.stale_income
            stale_fields.append("annual_income")
            evidence.append("Income has no verification date")

        arr = case_to_arrays(case)
        hist = history_days(arr)
        if hist < pol.min_history_days:
            penalty += pen.short_history
            evidence.append(
                f"Only {hist:.0f} days of transaction history (min {pol.min_history_days})"
            )

        n_future = int((arr.age_days < 0).sum())
        if n_future:
            penalty += pen.future_transactions
            evidence.append(
                f"{n_future} transaction(s) dated after the assessment time (possible leakage)"
            )

        dupes = [k for k, n in Counter(t.txn_id for t in case.transactions).items() if n > 1]
        if dupes:
            penalty += pen.duplicate_transactions
            evidence.append(f"{len(dupes)} duplicated transaction id(s)")

        observed = observed_annual_inflow(arr)
        if observed and case.profile.annual_income:
            ratio = case.profile.annual_income / observed
            r = pol.income_conflict_ratio
            if not (1 / r <= ratio <= r):
                penalty += pen.income_conflict
                evidence.append(f"Declared income differs from observed inflows by {ratio:.1f}x")
        elif observed is None:
            limitations.append("Too little history to cross-check declared income")

        quality = round(clip01(1 - penalty), 3)
        automation_permitted = not critical_missing and quality >= pol.review_below
        if critical_missing or quality < pol.abstain_below:
            action = RecommendedAction.ABSTAIN
        elif quality < pol.review_below:
            action = RecommendedAction.REQUEST_MORE_INFORMATION
        else:
            action = RecommendedAction.NORMAL_PROCESSING
        if not evidence:
            evidence.append("All completeness, freshness, conflict and leakage checks passed")
        limitations.append("Rule-based checks; cannot detect fabricated but consistent data")

        return AgentResponse(
            case_id=case.case_id,
            agent_name=self.name,
            risk_score=round(100 * (1 - quality), 1),
            confidence=0.95,
            data_quality=quality,
            ood_score=0.0,
            evidence=evidence,
            limitations=limitations,
            recommended_action=action,
            model_version=RULES_VERSION,
            details={
                "critical_missing": critical_missing,
                "optional_missing": optional_missing,
                "stale_fields": stale_fields,
                "income_age_days": income_age_days,
                "history_days": round(hist, 1),
                "automation_permitted": automation_permitted,
            },
        )
