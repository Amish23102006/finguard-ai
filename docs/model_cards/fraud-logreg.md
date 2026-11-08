# Model card: fraud-logreg 1.0.0

- **Purpose:** flag *behavioural patterns that require review* in synthetic transaction timelines.
- **Not for:** accusing anyone of fraud, blocking real payments.
- **Model:** StandardScaler → LogisticRegression (seed 42); linear-SHAP reason codes.
- **Features (last 7 days vs baseline):** velocity_ratio, amount_spike_ratio, new_beneficiary_share_7d, new_devices_7d, night_share_7d. Transactions after `as_of` are excluded (leakage control).
- **Data:** 2,500 synthetic timelines, ~8% injected anomalies (subtle and strong), 15% legitimate bursts, 2% label noise.
- **Split:** temporal 80/20 on `event_order`.
- **Measured (synthetic test set):** ROC-AUC 0.929 · PR-AUC 0.747 · Brier 0.036 · ECE 0.037. Regenerate with `make train`.
- **Limitations:** class imbalance handled only via threshold policy; no cost-sensitive threshold yet (Phase 14); synthetic patterns are simpler than real fraud.
