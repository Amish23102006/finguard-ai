# Working on FinGuard AI in VS Code

## 1. Open the project
`File → Open Folder… → finguard-ai`. When prompted, click **Install** for the recommended
extensions (Python, Pylance, Ruff, Mypy, YAML, Docker, REST Client…).

## 2. Create the environment (once)
`Terminal → Run Task… → Setup: create .venv and install`
(macOS/Linux/Windows all supported). Then `Ctrl/Cmd+Shift+P → Python: Select Interpreter → .venv`.

> Prefer containers? `Ctrl/Cmd+Shift+P → Dev Containers: Reopen in Container`. Everything,
> including model training, is set up automatically.

## 3. Run and debug (F5 / Run and Debug panel)

| Launch config | What it does |
|---|---|
| **FinGuard: API + Dashboard** | Starts both; breakpoints work in agents/committee code |
| FinGuard: API (FastAPI + reload) | http://localhost:8000/docs |
| FinGuard: Dashboard (Streamlit) | http://localhost:8501 |
| FinGuard: Train models | Writes `artifacts/models/*.joblib` + `metrics.json` |
| FinGuard: Generate demo cases | Writes `data/sample/*.json` |
| FinGuard: Assess a case | Runs `examples/assess_case.py FG-DEMO-0001` |
| Pytest: current file | Debug the open test file |

Models train automatically on first use if `artifacts/models` is empty.

## 4. Tests
Open the **Testing** panel (beaker icon). All tests are discovered from `tests/`.
Or `Ctrl/Cmd+Shift+B` runs **Check all (lint + types + tests)**.

## 5. Try the API
Open `examples/requests.http` and click **Send Request** above any request (REST Client).

## 6. Continue building with an AI assistant in VS Code
- **GitHub Copilot Chat** reads `.github/copilot-instructions.md` automatically.
- **Claude Code** reads `CLAUDE.md` automatically.
- Paste one phase from `docs/PHASE_PROMPTS.md` at a time; review, run tests, commit.

## Weekly rhythm (from the blueprint)
Mon: one issue with acceptance criteria · Tue–Thu: feature branch + tests · Fri: integrate,
docs, 60-second demo · Sat: review metrics/limitations/security · Sun: tag a release only if reproducible.
