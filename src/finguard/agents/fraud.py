"""Fraud Detection Agent: anomalous transaction behaviour. Never accuses; flags for review."""

from __future__ import annotations

from finguard.agents.base import Agent, clip01, model_confidence
from finguard.committee.policy import Policy
from finguard.common.schemas import AgentResponse, Case, FeatureContribution
from finguard.features import case_to_arrays, fraud_features, history_days
from finguard.models.bundle import ModelBundle


def _describe(f: str, v: float) -> str | None:
    """Plain-language reason code for a notable feature value."""
    if f == "velocity_ratio" and v >= 2:
        return f"Transaction velocity is {v:.1f}x the account's baseline (last 7 days)"
    if f == "amount_spike_ratio" and v >= 4:
        return f"Largest recent payment is {v:.1f}x the typical outgoing amount"
    if f == "new_beneficiary_share_7d" and v >= 0.3:
        return f"{v:.0%} of recent outgoing payments go to new beneficiaries"
    if f == "new_devices_7d" and v >= 1:
        return f"{int(v)} previously unseen device(s) used in the last 7 days"
    if f == "night_share_7d" and v >= 0.2:
        return f"{v:.0%} of recent activity occurred between 00:00 and 05:00"
    return None


class FraudDetectionAgent(Agent):
    """Scores behavioural anomaly; output language is 'pattern requires review'."""

    name = "fraud_detection"

    def __init__(self, policy: Policy, bundle: ModelBundle) -> None:
        super().__init__(policy)
        self.bundle = bundle

    def assess(self, case: Case) -> AgentResponse:
        """Score the transaction timeline."""
        arr = case_to_arrays(case)
        hist = history_days(arr)
        limitations = ["Baseline logistic model trained on synthetic transaction timelines"]
        if not case.transactions:
            return AgentResponse(
                case_id=case.case_id,
                agent_name=self.name,
                risk_score=0.0,
                confidence=0.05,
                data_quality=0.0,
                ood_score=0.0,
                evidence=["No transactions supplied"],
                limitations=[*limitations, "Cannot assess behaviour without transactions"],
                recommended_action=self.action_for(0.0, 0.0),
                model_version=f"{self.bundle.name}-{self.bundle.version}",
            )

        feats = fraud_features(arr)
        x, _ = self.bundle.impute(dict(feats))
        p = self.bundle.predict_proba(x)
        contrib = self.bundle.contributions(x)
        ood, ood_features = self.bundle.ood_score(x)
        data_quality = clip01(hist / max(self.policy.data_reliability.min_history_days, 1))
        if hist < self.policy.data_reliability.min_history_days:
            limitations.append(f"Only {hist:.0f} days of history; baseline behaviour is uncertain")
        if ood_features:
            limitations.append(f"Outside training range: {', '.join(ood_features)}")

        ranked = sorted(contrib.items(), key=lambda kv: kv[1], reverse=True)
        evidence = [d for f, c in ranked if c > 0 and (d := _describe(f, feats[f]))]
        if p >= 0.5:
            evidence.insert(0, f"Behavioural pattern requires review (anomaly probability {p:.0%})")
        elif not evidence:
            evidence = [f"No unusual behaviour detected (anomaly probability {p:.0%})"]

        risk = round(p * 100, 1)
        return AgentResponse(
            case_id=case.case_id,
            agent_name=self.name,
            risk_score=risk,
            confidence=model_confidence(ood, data_quality),
            data_quality=round(data_quality, 3),
            ood_score=round(ood, 3),
            evidence=evidence,
            limitations=limitations,
            recommended_action=self.action_for(risk, ood),
            model_version=f"{self.bundle.name}-{self.bundle.version}",
            contributions=[
                FeatureContribution(feature=f, value=feats[f], contribution=round(c, 4))
                for f, c in ranked
            ],
            details={
                "anomaly_probability": round(p, 4),
                "features": feats,
                "history_days": round(hist, 1),
                "ood_features": ood_features,
            },
        )
