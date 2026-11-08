# Architecture

Target architecture (blueprint §4) with what exists today marked ✅.

```
User / Demo Dashboard ✅ (Streamlit)
  │
  ▼
FastAPI Gateway ✅ ── Authentication / Rate limiting (Phase 9)
  ├── Case Service ✅ (seeded demo cases) ── PostgreSQL (Phase 15)
  ├── Digital Twin Service (Phase 4) ── Feature Store (features.py ✅)
  ├── Agent Orchestrator ✅ (AssessmentService)
  │     ├── Credit Risk Agent ✅
  │     ├── Fraud Agent ✅
  │     ├── Data Reliability Agent ✅
  │     ├── Insurance Agent · Market Regime Agent · Portfolio Risk Agent (Phase 6)
  │     ├── Financial Network Agent (Phase 5)
  │     ├── Document Intelligence Agent / RAG (Phase 7)
  │     └── Model Challenge Agent (Phase 8)
  ├── Risk Committee / Policy Engine ✅ (configs/policies/committee_v1.yaml)
  ├── Scenario and Counterfactual Engine (Phase 4)
  ├── Decision Passport ✅ (JSON) / Audit Service (Phase 8)
  └── Monitoring: MLflow + drift + logs (Phase 8)
```

## Committee rules (v1.0.0)
| Rule | Condition | Action |
|---|---|---|
| R0 | strongest signal × 0.85 > weighted mean | overall risk uses the floor (no dilution) |
| R1 | critical field missing or data reliability < 0.5 | abstain |
| R2 | any agent OOD ≥ 0.6 | abstain |
| R3 | an agent failed | human review |
| R4 | score spread ≥ 40 | enhanced human verification |
| R5 | overall ≥ 55 | human review |
| R6 | overall ≥ 35, reliability < 0.85, or spread ≥ 20 | request more information |
| R7 | otherwise | normal processing (still by a human) |
