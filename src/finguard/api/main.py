"""FastAPI gateway. Run: ``uvicorn finguard.api.main:app --reload``."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from finguard import __version__
from finguard.common.schemas import Case, CommitteeDecision
from finguard.config import get_settings
from finguard.logging_config import configure_logging
from finguard.service import AssessmentService
from finguard.synthetic.generator import generate_demo_cases

logger = logging.getLogger(__name__)
MAX_BODY_BYTES = 2_000_000


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load models and policy once at startup."""
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_json)
    app.state.service = AssessmentService(settings)
    app.state.demo_cases = {c.case_id: c for c in generate_demo_cases(settings.random_seed)}
    logger.info("FinGuard API ready", extra={"version": __version__, "env": settings.app_env})
    yield


app = FastAPI(
    title="FinGuard AI",
    version=__version__,
    description=(
        "Explainable multi-agent financial risk intelligence on **synthetic data**. "
        "Recommends what a human should review; never approves, denies or accuses."
    ),
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.middleware("http")
async def limit_body_size(request: Request, call_next: Any) -> Any:
    """Reject oversized payloads early (input safety)."""
    length = request.headers.get("content-length")
    if length and int(length) > MAX_BODY_BYTES:
        return JSONResponse(status_code=413, content={"detail": "Payload too large"})
    return await call_next(request)


def _service(request: Request) -> AssessmentService:
    return request.app.state.service


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness probe."""
    return {"status": "ok", "version": __version__, "environment": get_settings().app_env}


@app.get("/policy")
def policy(request: Request) -> dict[str, Any]:
    """The active committee policy (policy-as-code)."""
    return _service(request).policy.model_dump()


@app.get("/cases")
def list_cases(request: Request) -> list[dict[str, Any]]:
    """Summaries of the seeded demo cases."""
    return [
        {
            "case_id": c.case_id,
            "scenario": c.scenario,
            "entity_label": c.entity_label,
            "n_transactions": len(c.transactions),
        }
        for c in request.app.state.demo_cases.values()
    ]


@app.get("/cases/{case_id}", response_model=Case)
def get_case(case_id: str, request: Request) -> Case:
    """Full synthetic demo case."""
    case = request.app.state.demo_cases.get(case_id)
    if case is None:
        raise HTTPException(status_code=404, detail="Unknown case_id")
    return case


@app.post("/cases/{case_id}/assess", response_model=CommitteeDecision)
def assess_demo_case(case_id: str, request: Request) -> CommitteeDecision:
    """Assess a seeded demo case."""
    return _service(request).assess(get_case(case_id, request))


@app.post("/assess", response_model=CommitteeDecision)
def assess(case: Case, request: Request) -> CommitteeDecision:
    """Assess a caller-supplied synthetic case (schema-validated)."""
    return _service(request).assess(case)
