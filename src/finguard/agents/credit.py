"""Credit Risk Agent: probability of default from a calibrated baseline model."""

from __future__ import annotations

from finguard.agents.base import Agent, clip01, model_confidence
from finguard.committee.policy import Policy
from finguard.common.schemas import AgentResponse, Case, FeatureContribution
from finguard.features import credit_features
from finguard.models.bundle import ModelBundle

LABELS = {
    "debt_to_income": "Debt-to-income ratio",
    "credit_utilization": "Credit utilisation",
    "missed_payments_12m": "Missed payments (12m)",
    "credit_history_months": "Credit history (months)",
    "employment_years": "Years employed",
    "loan_to_income": "Loan-to-income ratio",
}


class CreditRiskAgent(Agent):
    """Estimates probability of default (PD) and explains it with linear SHAP values."""

    name = "credit_risk"

    def __init__(self, policy: Policy, bundle: ModelBundle) -> None:
        super().__init__(policy)
        self.bundle = bundle

    def assess(self, case: Case) -> AgentResponse:
        """Score the borrower profile."""
        row = credit_features(case.profile)
        x, imputed = self.bundle.impute(row)
        pd_ = self.bundle.predict_proba(x)
        contrib = self.bundle.contributions(x)
        ood, ood_features = self.bundle.ood_score(x)
        data_quality = clip01(1 - len(imputed) / len(self.bundle.features))

        ranked = sorted(contrib.items(), key=lambda kv: abs(kv[1]), reverse=True)
        evidence = [
            f"{LABELS[f]} = {row[f]:.3g} {'raises' if c > 0 else 'lowers'} estimated default risk"
            for f, c in ranked[:3]
            if row[f] is not None
        ]
        evidence.insert(0, f"Estimated probability of default {pd_:.1%}")
        limitations = ["Baseline logistic model trained on synthetic borrower data"]
        if imputed:
            limitations.append(f"Imputed with training median: {', '.join(imputed)}")
        if ood_features:
            limitations.append(f"Outside training range: {', '.join(ood_features)}")

        risk = round(pd_ * 100, 1)
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
                FeatureContribution(feature=f, value=row[f], contribution=round(c, 4))
                for f, c in ranked
            ],
            details={
                "probability_of_default": round(pd_, 4),
                "imputed_features": imputed,
                "ood_features": ood_features,
            },
        )
