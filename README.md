# 🛡️ FinGuard AI

**Explainable multi-agent financial risk intelligence with digital twins, disagreement scoring and human oversight.**

![python](https://img.shields.io/badge/python-3.11%2B-blue) [![ci](https://github.com/Amish23102006/finguard-ai/actions/workflows/ci.yml/badge.svg)](https://github.com/Amish23102006/finguard-ai/actions/workflows/ci.yml) [![security](https://github.com/Amish23102006/finguard-ai/actions/workflows/security.yml/badge.svg)](https://github.com/Amish23102006/finguard-ai/actions/workflows/security.yml) ![data](https://img.shields.io/badge/data-synthetic%20only-orange) ![license](https://img.shields.io/badge/license-MIT-lightgrey)

> FinGuard is built on one idea: financial AI should do more than predict. It should
> expose conflicting evidence, recognise unfamiliar situations, estimate the cost of being
> wrong and keep an auditable path for human decisions.

⚠️ **Educational demo on synthetic data. Not financial advice.** FinGuard never approves loans, denies claims, accuses anyone of fraud or executes trades. It recommends what a human should review.

---

## Why it's different

| Differentiator | What it shows |
|---|---|
| **Agent Disagreement Intelligence** | Specialist models can disagree. The disagreement is measured and explained, never silently averaged away |
| **Unknown-Risk Detection** | Abstains when a case is unlike the training data |
| **Data Reliability Agent** | Checks completeness, freshness, conflicts and leakage before any recommendation |
| **Policy-as-Code Committee** | Thresholds live in versioned YAML; every rule that fires is recorded |
| **Evidence-grounded narrative** | Template explanations answer *what happened, why, how certain, what to check next* |
| **Linear-SHAP explanations** | Exact per-feature contributions for every model score |
| **Decision Passport** | Downloadable, reproducible audit record (inputs, versions, evidence, outcome) |
| **Graceful degradation** | A failed agent downgrades the result to human review instead of crashing |

## Three-minute demo

1. `make setup && make dashboard`, then open http://localhost:8501 (Recruiter demo mode is on by default).
2. **FG-DEMO-0002 · normal**: every agent agrees the risk is low, and evidence and limitations are still shown.
3. **FG-DEMO-0001 · conflicting evidence**: credit risk ≈ 4, fraud pattern ≈ 98, income evidence 15 months old. → **Enhanced human verification** with high disagreement and reduced confidence.
4. **FG-DEMO-0003 · unknown pattern**: inputs far outside training data → **Abstain**.
5. **FG-DEMO-0004 · missing data**: critical fields missing → **Abstain: insufficient evidence**.
6. Open the *Decision Passport* tab and download the audit JSON.

## Architecture

```
 Streamlit console ─┐                      ┌─ CreditRiskAgent ──────── calibrated logistic PD + SHAP + OOD
                    ├─► AssessmentService ─┼─ FraudDetectionAgent ──── behavioural features + SHAP + OOD
 FastAPI gateway ───┘         │            └─ DataReliabilityAgent ─── completeness / freshness / leakage rules
                              ▼
                    Risk Committee (policy YAML) ─► CommitteeDecision / Decision Passport
```

| Component | Path |
|---|---|
| Shared contracts (`Case`, `AgentResponse`, `CommitteeDecision`) | `src/finguard/common/schemas.py` |
| Synthetic generator (seeded scenarios) | `src/finguard/synthetic/generator.py` |
| Feature engineering (one path for train & serve) | `src/finguard/features.py` |
| Models, training, registry | `src/finguard/models/` |
| Agents | `src/finguard/agents/` |
| Committee policy engine + narratives | `src/finguard/committee/`, `configs/policies/` |
| API | `src/finguard/api/main.py` |
| Dashboard | `src/finguard/dashboard/app.py` |

## Quick start

**VS Code (recommended):** open the folder, accept the recommended extensions, then run
*Terminal → Run Task → Setup: create .venv and install*. Press **F5** and pick
**FinGuard: API + Dashboard**. Full guide: [docs/VSCODE.md](docs/VSCODE.md).

**Terminal:**
```bash
make setup        # venv + dev deps + .env
make test         # 36 tests
make train        # writes artifacts/models + metrics.json
make api          # http://localhost:8000/docs
make dashboard    # http://localhost:8501
```

**Docker:** `docker compose up --build` (API on :8000, dashboard on :8501).

## Evaluation (synthetic data, temporal 80/20 split, seed 42)

Measured by `make train`. These numbers describe **synthetic** data and say nothing about real-world performance.

| Model | ROC-AUC | PR-AUC | Brier | ECE |
|---|---|---|---|---|
| credit-logreg-1.0.0 | 0.733 | 0.507 | 0.161 | 0.025 |
| fraud-logreg-1.0.0 | 0.929 | 0.747 | 0.036 | 0.037 |

## Responsible use

- Synthetic data only; the `Case` schema rejects `synthetic: false`.
- No autonomous high-impact decisions. Every outcome is a *recommendation for a human*.
- Safe wording ("pattern requires review"), enforced by tests.
- Abstention when data is weak or unfamiliar.
- See [SECURITY.md](SECURITY.md) and [docs/model_cards](docs/model_cards).

## Reproduce

```bash
python -m finguard.synthetic.generator --out data/sample --seed 42
python -m finguard.models.train --seed 42
pytest
```

## Roadmap

- ✅ Phase 1: repository skeleton, schemas, synthetic generator, CI
- ✅ Phase 2: credit + fraud MVP with explanations and model cards
- ✅ Phase 3: data reliability + policy-driven risk committee + dashboard
- ⏭ Phases 4–9: digital twin & scenarios, graph risk, portfolio/market, grounded RAG, governance & replay, recruiter release. Prompts are in [docs/PHASE_PROMPTS.md](docs/PHASE_PROMPTS.md)

## Author

**Amish Chaturvedi**, B.Tech CSE (AI & ML), VIT Bhopal

[LinkedIn](https://www.linkedin.com/in/amish-chaturvedi-756606333/) · [GitHub](https://github.com/Amish23102006) · [Email](mailto:chaturvediamish06@gmail.com)
