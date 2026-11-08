# ADR 0001: Policy-driven committee instead of a learned ensemble

**Status:** accepted · **Date:** 2026-11-08

## Context
A stacked ensemble would average credit and fraud signals into one opaque number, hiding
disagreement and making threshold changes hard to audit.

## Decision
Aggregate agent outputs with transparent rules from versioned YAML (`committee_v1.yaml`).
Overall risk = max(weighted mean, 0.85 × strongest signal). Disagreement is the score spread.
Abstention and escalation rules are ordered and each fired rule is recorded in `reasons`.

## Consequences
+ Reviewers can see when and why policy changed; tests pin behaviour per policy version.
− Rules need manual tuning; a cost-sensitive threshold optimiser is planned for Phase 14.
