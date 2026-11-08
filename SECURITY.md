# Security policy

FinGuard AI is an educational demo on synthetic data. Please report vulnerabilities privately
via GitHub Security Advisories rather than public issues.

## Controls in place
| Area | Control |
|---|---|
| Secrets | `.env.example` only; gitleaks in pre-commit and CI |
| Input safety | Strict Pydantic schemas (`extra="forbid"`), 2 MB body limit, max 5,000 transactions, `synthetic: true` required |
| Dependencies | Version floors in `pyproject.toml`; `pip-audit` weekly in CI |
| Data | Synthetic only; no raw personal data committed |
| Logging | Case IDs only, no payloads |
| Human oversight | No automatic high-impact decision; abstention + escalation rules |
| Audit | Decision Passport with model + policy versions |

## Planned
Authentication & roles, rate limiting, prompt-injection defences for RAG (Phase 7), immutable audit log (Phase 8).
