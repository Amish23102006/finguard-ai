# Phase prompts (from the FinGuard Build Blueprint)

Give the **master prompt** (in `CLAUDE.md` / `.github/copilot-instructions.md`) first, then
**one phase at a time**. Require file-by-file changes, tests and commands. Never invent metrics.

Phases 1–3 are implemented in this repository. The rest follow.

## Phase 4: Digital Twin and Scenarios
Implement a FinancialDigitalTwin for a synthetic borrower and transaction profile. Store
time-stamped state, risk history and scenario inputs. Add what-if operations for income, debt
ratio, missed payments, transaction anomalies and interest-rate shock. Add counterfactual
suggestions subject to realistic constraints. Show before/after risk and confidence.
Acceptance criteria: scenarios are deterministic, reversible and audit logged; explanations
never promise an outcome.

## Phase 5: Graph Risk
Implement a synthetic financial network using NetworkX with nodes for accounts, customers,
devices, beneficiaries and claims. Generate labelled normal and suspicious clusters. Compute
interpretable graph features and a NetworkRiskAgent. Add a graph visualisation and tests for
circular transfers, shared devices and rapid fund movement. Do not include instructions for
evading real controls.

## Phase 6: Portfolio and Market Risk
Implement MarketRegimeAgent and PortfolioRiskAgent using bundled versioned sample price data.
Calculate returns, volatility, drawdown, concentration, historical VaR, Expected Shortfall and
scenario losses. Add unit tests with small hand-checkable fixtures. Do not predict guaranteed
returns or issue buy/sell recommendations. Add a portfolio stress lab and risk-contribution chart.

## Phase 7: Grounded Document Intelligence
Implement a small RAG pipeline over approved sample or synthetic financial documents. Every
extracted risk event must include source document, passage, date, risk category, impact and
confidence. Add retrieval and faithfulness tests, prompt-injection defences, file-type/size
limits and a no-evidence abstention response. LLM output must validate against a Pydantic schema.

## Phase 8: Governance and Challenge
Implement OOD detection, ModelChallengeAgent, confidence decomposition, fairness audit on
synthetic groups, model registry metadata, drift report, human override and Decision Passport.
Add Decision Replay capable of reconstructing a case from a versioned case pack. Acceptance
criteria: unfamiliar data triggers abstention; missing model artefacts fail safely; replay
reproduces the stored result within defined tolerance.

## Phase 9: Recruiter Demo and Release
Build Recruiter Demo Mode with three seeded scenarios: Normal, Conflicting Evidence and Unknown
Pattern. Add a guided 3-minute path, tooltips, reset button, downloadable Decision Passport and
architecture/about pages. Add end-to-end tests, accessibility labels, loading/error states and
telemetry that contains no personal data. Produce a polished README, demo script, screenshots
checklist, release notes and a v1.0.0 tag checklist.

## Later (P2 / research)
Cost-of-wrong-decision simulator and analyst workload, fairness stress lab, canary model
comparison, outcome feedback loop, privacy sandbox / federated learning, GNNs. Only after the
items above are polished and tested.
