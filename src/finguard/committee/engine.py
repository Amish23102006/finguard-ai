"""Risk Committee: transparent, policy-driven aggregation of agent opinions.

Principles: never silently average away disagreement; abstain when evidence is weak or the
case is unfamiliar; every rule that fires is recorded as a reason.
"""

from __future__ import annotations

from datetime import UTC, datetime

from finguard import __version__
from finguard.committee.narrative import build_human_checks, build_narrative
from finguard.committee.policy import Policy
from finguard.common.schemas import AgentResponse, CommitteeDecision, RecommendedAction

ROADMAP_LIMITATION = (
    "Network, market, insurance and document agents are not yet part of this assessment"
)


def _band(risk: float, policy: Policy) -> str:
    b = policy.risk_bands
    if risk >= b.high:
        return "high"
    if risk >= b.medium_high:
        return "medium_high"
    if risk >= b.medium:
        return "medium"
    return "low"


def _level(spread: float, policy: Policy) -> str:
    d = policy.disagreement
    return "high" if spread >= d.high else "medium" if spread >= d.medium else "low"


def decide(
    case_id: str,
    responses: list[AgentResponse],
    policy: Policy,
    failed_agents: list[str] | None = None,
    now: datetime | None = None,
) -> CommitteeDecision:
    """Combine agent responses into a recommendation for a human reviewer."""
    failed_agents = failed_agents or []
    by_name = {r.agent_name: r for r in responses}
    scoring = [r for r in responses if r.agent_name in policy.weights]
    dr = by_name.get("data_reliability")
    reasons: list[str] = []

    if scoring:
        total_w = sum(policy.weights[r.agent_name] for r in scoring)
        weighted = sum(policy.weights[r.agent_name] * r.risk_score for r in scoring) / total_w
        strongest = max(scoring, key=lambda r: r.risk_score)
        floor = policy.aggregation.max_signal_floor * strongest.risk_score
        overall = max(weighted, floor)
        if floor - weighted >= 5:
            reasons.append(
                f"[R0-no-dilution] Strongest signal ({strongest.agent_name}, "
                f"{strongest.risk_score:.0f}) sets the floor instead of the weighted mean "
                f"({weighted:.0f})"
            )
        spread = max(r.risk_score for r in scoring) - min(r.risk_score for r in scoring)
        mean_conf = sum(policy.weights[r.agent_name] * r.confidence for r in scoring) / total_w
    else:
        overall, spread, mean_conf = 0.0, 0.0, 0.0

    data_rel = dr.data_quality if dr else 0.0
    max_ood = max((r.ood_score for r in responses), default=0.0)
    level = _level(spread, policy)
    band = _band(overall, policy)
    c = policy.confidence
    confidence = (
        mean_conf
        * (1 - c.disagreement_penalty * spread / 100)
        * (1 - c.ood_penalty * max_ood)
        * (0.5 + 0.5 * data_rel)
    )

    critical_missing = bool(dr and dr.details.get("critical_missing"))
    dpol = policy.data_reliability
    if dr is None or critical_missing or data_rel < dpol.abstain_below:
        action = RecommendedAction.ABSTAIN
        reasons.append(
            f"[R1-data-abstain] Insufficient evidence: data reliability {data_rel:.0%}"
            + (" with critical fields missing" if critical_missing else "")
        )
    elif max_ood >= policy.ood.abstain_above:
        action = RecommendedAction.ABSTAIN
        reasons.append(
            f"[R2-ood-abstain] Case differs from training data (unknown-pattern score "
            f"{max_ood:.0%} >= {policy.ood.abstain_above:.0%})"
        )
    elif failed_agents:
        action = RecommendedAction.HUMAN_REVIEW
        reasons.append(f"[R3-degraded] Agent(s) unavailable: {', '.join(failed_agents)}")
    elif spread >= policy.disagreement.high:
        action = RecommendedAction.ENHANCED_HUMAN_VERIFICATION
        hi = max(scoring, key=lambda r: r.risk_score)
        lo = min(scoring, key=lambda r: r.risk_score)
        reasons.append(
            f"[R4-disagreement] High disagreement ({spread:.0f}/100): {hi.agent_name} "
            f"{hi.risk_score:.0f} vs {lo.agent_name} {lo.risk_score:.0f}"
        )
    elif band in ("high", "medium_high"):
        action = RecommendedAction.HUMAN_REVIEW
        reasons.append(f"[R5-risk] Overall risk {overall:.0f}/100 is {band.replace('_', '-')}")
    elif band == "medium" or data_rel < dpol.review_below or level == "medium":
        action = RecommendedAction.REQUEST_MORE_INFORMATION
        reasons.append(
            f"[R6-more-info] Risk band {band}, data reliability {data_rel:.0%}, "
            f"disagreement {level}"
        )
    else:
        action = RecommendedAction.NORMAL_PROCESSING
        reasons.append("[R7-normal] All agents report low risk with reliable data")

    if data_rel < dpol.review_below and action != RecommendedAction.ABSTAIN:
        reasons.append(f"[R8-stale] Data reliability {data_rel:.0%} below review level")

    limitations: list[str] = []
    for r in responses:
        for lim in r.limitations:
            entry = f"{r.agent_name}: {lim}"
            if entry not in limitations:
                limitations.append(entry)
    limitations.append(ROADMAP_LIMITATION)
    limitations += [f"{a}: unavailable during this assessment" for a in failed_agents]

    decision = CommitteeDecision(
        case_id=case_id,
        engine_version=__version__,
        policy_id=policy.policy_id,
        policy_version=policy.version,
        decided_at=now or datetime.now(UTC),
        overall_risk=round(overall, 1),
        risk_band=band,
        confidence=round(min(max(confidence, 0.0), 1.0), 3),
        data_reliability=round(data_rel, 3),
        disagreement=round(spread, 1),
        disagreement_level=level,
        unknown_pattern_score=round(max_ood, 3),
        recommended_action=action,
        abstained=action == RecommendedAction.ABSTAIN,
        reasons=reasons,
        limitations=limitations,
        human_checks=[],
        narrative={},
        agent_responses=responses,
    )
    decision.human_checks = build_human_checks(decision)
    decision.narrative = build_narrative(decision)
    return decision
