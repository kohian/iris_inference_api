import os

os.environ["MODEL_PATH"] = "model_artifacts/logreg.joblib"

from fastapi.testclient import TestClient

from iris_inference_api.main import app


def test_health():
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"
        assert response.json()["model_loaded"] is True


def test_predict():
    with TestClient(app) as client:
        response = client.post(
            "/predict",
            json={"features": [5.1, 3.5, 1.4, 0.2]},
        )

        assert response.status_code == 200

        data = response.json()
        assert "class_id" in data
        assert "class_name" in data
        assert "confidence" in data
        assert "probabilities" in data