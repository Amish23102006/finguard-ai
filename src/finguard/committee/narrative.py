"""Template-based explanations. Built only from structured evidence, never from a free LLM.

Every narrative answers: What happened? Why? How certain? What should a human check next?
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from finguard.common.schemas import RecommendedAction

if TYPE_CHECKING:
    from finguard.common.schemas import CommitteeDecision

ACTION_TEXT = {
    RecommendedAction.NORMAL_PROCESSING: "standard processing by a human reviewer",
    RecommendedAction.REQUEST_MORE_INFORMATION: "request more information before review",
    RecommendedAction.HUMAN_REVIEW: "human review",
    RecommendedAction.ENHANCED_HUMAN_VERIFICATION: "enhanced human verification",
    RecommendedAction.ABSTAIN: "no automated recommendation (insufficient or unfamiliar evidence)",
}


def build_human_checks(d: CommitteeDecision) -> list[str]:
    """Concrete next steps for the reviewer, derived from agent details."""
    checks: list[str] = []
    agents = {r.agent_name: r for r in d.agent_responses}
    dr = agents.get("data_reliability")
    if dr:
        missing = dr.details.get("critical_missing") or []
        if missing:
            checks.append(f"Collect missing information: {', '.join(missing)}")
        if "annual_income" in (dr.details.get("stale_fields") or []):
            checks.append("Re-verify income with current documentation")
        if (dr.details.get("history_days") or 0) < 60:
            checks.append("Obtain a longer transaction history before relying on behaviour")
    fraud = agents.get("fraud_detection")
    if fraud and fraud.risk_score >= 50:
        checks.append(
            "Confirm recent payments to new beneficiaries with the customer via a trusted channel"
        )
    if d.unknown_pattern_score >= 0.6:
        checks.append("Check whether this case type is within the models' intended scope")
    if d.disagreement_level == "high":
        checks.append("Compare the conflicting agent evidence side by side in the Agent Room")
    if not checks:
        checks.append("No additional checks suggested beyond standard procedure")
    return checks


def build_narrative(d: CommitteeDecision) -> dict[str, str]:
    """Plain-language summary with safe wording (no accusations, no promises)."""
    top_evidence = [
        e
        for r in d.agent_responses
        if r.risk_score >= 40 or r.agent_name == "data_reliability"
        for e in r.evidence[:2]
    ][:4]
    why = "; ".join(reason.split("] ", 1)[-1] for reason in d.reasons)
    if top_evidence:
        why += ". Key evidence: " + "; ".join(top_evidence)
    return {
        "what_happened": (
            f"{len(d.agent_responses)} specialist agents assessed case {d.case_id}. "
            f"Overall risk {d.overall_risk:.0f}/100 ({d.risk_band.replace('_', '-')}). "
            f"Suggested route: {ACTION_TEXT[d.recommended_action]}."
        ),
        "why": why + ".",
        "how_certain": (
            f"Confidence {d.confidence:.0%}. Data reliability {d.data_reliability:.0%}, "
            f"agent disagreement {d.disagreement:.0f}/100 ({d.disagreement_level}), "
            f"unknown-pattern score {d.unknown_pattern_score:.0%}."
        ),
        "what_next": " ".join(f"{i + 1}) {c}." for i, c in enumerate(d.human_checks)),
    }
