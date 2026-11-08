from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from finguard import __version__
from finguard.config import get_settings


@pytest.fixture(scope="module")
def client(model_dir, bundles, monkeypatch_module) -> Iterator[TestClient]:
    monkeypatch_module.setenv("FINGUARD_MODEL_DIR", str(model_dir))
    get_settings.cache_clear()
    from finguard.api.main import app

    with TestClient(app) as c:
        yield c
    get_settings.cache_clear()


@pytest.fixture(scope="module")
def monkeypatch_module() -> Iterator[pytest.MonkeyPatch]:
    mp = pytest.MonkeyPatch()
    yield mp
    mp.undo()


def test_health(client: TestClient) -> None:
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["version"] == __version__ and "environment" in body


def test_list_and_get_cases(client: TestClient) -> None:
    ids = {c["case_id"] for c in client.get("/cases").json()}
    assert "FG-DEMO-0001" in ids
    assert client.get("/cases/FG-DEMO-0001").json()["synthetic"] is True
    assert client.get("/cases/FG-DEMO-9999").status_code == 404


def test_assess_demo_case(client: TestClient) -> None:
    body = client.post("/cases/FG-DEMO-0001/assess").json()
    assert body["recommended_action"] == "enhanced_human_verification"
    assert body["policy_version"] == "1.0.0"
    assert len(body["agent_responses"]) == 3


def test_assess_custom_case_and_validation(client: TestClient) -> None:
    case = client.get("/cases/FG-DEMO-0002").json()
    case["case_id"] = "FG-USER-0001"
    assert client.post("/assess", json=case).json()["case_id"] == "FG-USER-0001"
    case["synthetic"] = False  # real data is not accepted
    assert client.post("/assess", json=case).status_code == 422


def test_policy_endpoint(client: TestClient) -> None:
    assert client.get("/policy").json()["policy_id"] == "finguard-committee"
