# Data

All data in this repository is **synthetic** and generated deterministically by
`src/finguard/synthetic/generator.py` (seed 42). No real personal or customer data is used or accepted.

| Set | Generator | Notes |
|---|---|---|
| Credit training | `generate_credit_training` | 5,000 rows, logistic generative rule + noise |
| Fraud training | `generate_fraud_training` | 2,500 timelines, injected anomaly patterns, 2% label noise |
| Demo cases | `generate_demo_cases` → `data/sample/*.json` | FG-DEMO-0001 conflicting · 0002 normal · 0003 unknown pattern · 0004 missing data |

IDs are fictional codes (`FG-`, `TXN-`, `CP-`, `DEV-`); every case carries `synthetic: true`.
Ground-truth labels in `ground_truth` are for evaluation only and are never read by agents.
