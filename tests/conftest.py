"""Shared fixtures. Models train once per session into a temp dir (deterministic)."""

from __future__ import annotations

from pathlib import Path

import pytest

from finguard.committee.policy import Policy, load_policy
from finguard.common.schemas import Case
from finguard.config import PROJECT_ROOT, Settings
from finguard.models.bundle import ModelBundle
from finguard.models.registry import get_bundles
from finguard.service import AssessmentService
from finguard.synthetic.generator import generate_demo_cases

POLICY_PATH = PROJECT_ROOT / "configs" / "policies" / "committee_v1.yaml"


@pytest.fixture(scope="session")
def model_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("models")


@pytest.fixture(scope="session")
def bundles(model_dir: Path) -> dict[str, ModelBundle]:
    return get_bundles(model_dir, seed=42)


@pytest.fixture(scope="session")
def policy() -> Policy:
    return load_policy(POLICY_PATH)


@pytest.fixture(scope="session")
def service(model_dir: Path, bundles: dict[str, ModelBundle], policy: Policy) -> AssessmentService:
    return AssessmentService(Settings(model_dir=model_dir), policy=policy, model_dir=model_dir)


@pytest.fixture(scope="session")
def demo_cases() -> dict[str, Case]:
    return {c.case_id: c for c in generate_demo_cases()}
