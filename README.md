# Iris Inference API (Production-Style)

A production-oriented FastAPI service for serving machine learning models, built with a focus on clean architecture, containerization, CI/CD, and cloud deployment.

This project demonstrates how to package a trained model into a reproducible, deployable inference API using modern MLOps practices.

---

##  Features

* FastAPI-based inference service
* Supports both **local** and **GCS-based model loading**
* Structured request/response validation using Pydantic
* Async batch client for concurrent inference testing
* Dockerized with multi-stage builds (test + production)
* CI pipeline with:

  * Ruff (linting)
  * Pytest (API testing)
* Deployable to **Google Cloud Run**
* Image stored in **Artifact Registry**

---

##  Project Structure

```
iris_inference_api/
│
├── src/iris_inference_api/
│   ├── main.py              # FastAPI app entrypoint
│   ├── model_loader.py      # Handles local + GCS model loading
│   ├── schemas.py           # Request/response models
│   ├── batch_async_client.py# Async client for load testing
│
├── model_artifacts/
│   ├── logreg.joblib
│   ├── xgb.joblib
│
├── tests/
│   └── test_api.py          # API tests (TestClient)
│
├── Dockerfile               # Multi-stage (test + prod)
├── pyproject.toml           # Packaging + tooling config
├── requirements.txt         # Runtime dependencies
├── requirements_dev.txt     # Dev/test dependencies
├── .github/workflows/
│   └── build_docker.yml     # CI pipeline
```

---

##  Setup

### 1. Create virtual environment

```bash
python -m venv .venv
source .venv/bin/activate   # or .venv\Scripts\activate (Windows)
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

---

##  Run Locally

```bash
uvicorn iris_inference_api.main:app --reload
```

API will be available at:

```
http://127.0.0.1:8000
```

Swagger docs:

```
http://127.0.0.1:8000/docs
```

---

##  Example Request

```json
POST /predict
{
  "features": [5.1, 3.5, 1.4, 0.2]
}
```

Response:

```json
{
  "class_id": 0,
  "class_name": "setosa",
  "confidence": 0.98,
  "probabilities": [0.98, 0.01, 0.01]
}
```

---

##  Testing

Run tests locally:

```bash
pytest
```

Tests include:

* Health endpoint validation
* Prediction endpoint correctness
* Model loading via FastAPI lifespan

---

##  Linting

```bash
ruff check src tests
```

---

##  Docker

### Build (test stage)

```bash
docker build -t iris-test --target test .
docker run iris-test
```

### Build (production)

```bash
docker build -t iris-api --target prod .
docker run -p 8080:8080 iris-api
```

---

##  Cloud Deployment

### Build & Push via GitHub Actions

* Uses Workload Identity Federation (no service account keys)
* Pushes image to Google Artifact Registry

### Deploy to Cloud Run

```bash
gcloud run deploy iris-api \
  --image us-central1-docker.pkg.dev/<PROJECT>/<REPO>/iris-api:latest \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars MODEL_PATH=gs://<BUCKET>/model_artifacts/logreg.joblib
```

---

##  Environment Variables

| Variable   | Description                         |
| ---------- | ----------------------------------- |
| MODEL_PATH | Path to model (local or `gs://...`) |

---

##  Design Decisions

### Model Loading Abstraction

Supports both local and GCS paths, enabling flexible deployment across environments.

### FastAPI Lifespan

Model is loaded once at startup to avoid repeated loading during inference.

### Multi-stage Docker

Separates test and production images to keep production lightweight.

### CI/CD Pipeline

Docker-based testing ensures consistency between local and deployment environments.

---

##  Author

Ian Koh

---

##  Notes

This project is intentionally over-engineered relative to the Iris dataset to demonstrate production-ready ML system design rather than model complexity.
