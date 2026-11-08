# Model card: credit-logreg 1.0.0

- **Purpose:** estimate probability of default (PD) for a *synthetic* borrower to inform a human reviewer.
- **Not for:** real lending decisions, approvals or denials.
- **Model:** StandardScaler → LogisticRegression (seed 42). Explanations: exact linear SHAP (coef × standardised value).
- **Features:** debt_to_income, credit_utilization, missed_payments_12m, credit_history_months, employment_years, loan_to_income.
- **Data:** 5,000 synthetic borrowers from `generate_credit_training` (documented logistic generative rule + noise). No protected attributes.
- **Split:** temporal 80/20 on `application_order`.
- **Measured (synthetic test set):** ROC-AUC 0.733 · PR-AUC 0.507 · Brier 0.161 · ECE 0.025. Regenerate with `make train`.
- **Safeguards:** missing inputs are imputed with the training median and lower data quality; OOD score from the 0.5–99.5 percentile envelope; committee abstains at OOD ≥ 0.6.
- **Limitations:** synthetic relationships only; linear model; no fairness audit yet (Phase 8).
