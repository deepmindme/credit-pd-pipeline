import pytest
from fastapi.testclient import TestClient

from src.api.app import app
from src.api.schemas import EXAMPLE_CLIENT


@pytest.fixture
def client(model_path, monkeypatch):
    monkeypatch.setenv("MODEL_PATH", str(model_path))
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def client_without_model(tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_PATH", str(tmp_path / "missing.joblib"))
    with TestClient(app) as test_client:
        yield test_client


def test_root(client):
    assert client.get("/").status_code == 200


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["model_version"] == "test_model-v1"
    assert 0 < body["threshold"] < 1


def test_predict_returns_class_and_probability(client):
    response = client.post("/predict", json=EXAMPLE_CLIENT)
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"default_prediction", "default_probability", "threshold"}
    assert 0 <= body["default_probability"] <= 1
    assert body["default_prediction"] == int(body["default_probability"] >= body["threshold"])


@pytest.mark.parametrize(
    "field, value",
    [
        ("AGE", 12),
        ("SEX", 5),
        ("PAY_0", 11),
        ("PAY_AMT1", -1),
        ("LIMIT_BAL", 0),
        ("BILL_AMT3", 9_000_000),
        ("AGE", "old"),
    ],
)
def test_invalid_input_is_rejected(client, field, value):
    response = client.post("/predict", json={**EXAMPLE_CLIENT, field: value})
    assert response.status_code == 422


def test_missing_field_is_rejected(client):
    payload = {k: v for k, v in EXAMPLE_CLIENT.items() if k != "PAY_AMT6"}
    assert client.post("/predict", json=payload).status_code == 422


def test_batch_prediction(client):
    late_payer = {**EXAMPLE_CLIENT, "PAY_0": 3, "PAY_2": 3}
    response = client.post("/predict/batch", json={"clients": [EXAMPLE_CLIENT, late_payer]})
    assert response.status_code == 200
    body = response.json()
    assert body["model_version"] == "test_model-v1"
    assert len(body["predictions"]) == 2


def test_empty_batch_is_rejected(client):
    assert client.post("/predict/batch", json={"clients": []}).status_code == 422


def test_service_without_model(client_without_model):
    health = client_without_model.get("/health")
    assert health.status_code == 503
    assert health.json()["model_loaded"] is False
    assert client_without_model.post("/predict", json=EXAMPLE_CLIENT).status_code == 503
